"""Cross-field validation rules evaluated at emission time.

Rules are defined in the contract schema under ``cross_field_rules``.
This module parses them into typed Pydantic models and evaluates them
against emitted field values.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, model_validator

from . import schema as registry
from .logging import get_logger

logger = get_logger(__name__)


class CrossFieldRule(BaseModel):
    """One cross-field validation rule from the contract schema."""

    type: Literal["lte", "date_gte", "date_max_gap_days", "eq"]
    field: str | None = None
    """Field being validated/observed. Required for action='validate'; must be None for action='set'."""
    reference: str
    severity: Literal["warning", "error"] = "error"
    message_template: str = ""
    tolerance: float = 0.0
    max_days: int = 0
    # Derivation-only fields (action="set")
    action: Literal["validate", "set"] = "validate"
    target: str | None = None
    """Field that receives the derived value. Required for action='set'; must be None for action='validate'."""
    set_value: str = ""
    condition_value: str = ""

    @model_validator(mode="after")
    def _check_action_coherence(self) -> CrossFieldRule:
        if self.action == "set":
            if self.target is None:
                raise ValueError(
                    f"action='set' rule (reference={self.reference!r}) requires 'target'"
                )
            missing = [
                f for f in ("set_value", "condition_value") if not getattr(self, f)
            ]
            if missing:
                raise ValueError(
                    f"action='set' rule (target={self.target!r}) requires: {', '.join(missing)}"
                )
            if self.type != "eq":
                raise ValueError("action='set' rules must use type='eq'")
            if self.field is not None:
                raise ValueError(
                    "action='set' rules must not specify 'field' (use 'target' for the write destination)"
                )
        else:
            # validate action — field required, target/set_value must not be present
            if self.field is None:
                raise ValueError("action='validate' rules require 'field'")
            if self.target is not None:
                raise ValueError("action='validate' rules must not specify 'target'")
            if self.set_value:
                raise ValueError("action='validate' rules must not specify 'set_value'")
        return self


class Violation(BaseModel):
    """Result of a failed cross-field validation rule."""

    rule_type: str
    field: str
    reference: str
    severity: Literal["warning", "error"]
    message: str


class Derivation(BaseModel):
    """Result of a fired cross-field derivation rule (action='set')."""

    rule_type: str
    reference: str
    condition_value: str
    target: str
    set_value: str


# Module-level cache: loaded once on first call.
_rules_cache: list[CrossFieldRule] | None = None


def load_rules() -> list[CrossFieldRule]:
    """Load and cache cross-field rules from the contract schema."""
    global _rules_cache
    if _rules_cache is not None:
        return _rules_cache

    raw_rules = registry.cross_field_rules()
    parsed: list[CrossFieldRule] = []
    for raw in raw_rules:
        try:
            parsed.append(CrossFieldRule(**raw))
        except Exception:  # noqa: PERF203 — per-rule isolation: bad rule must not abort the whole list
            logger.warning("invalid_cross_field_rule", rule=raw)
    _rules_cache = parsed
    return _rules_cache


def clear_cache() -> None:
    """Reset the rule cache (for testing)."""
    global _rules_cache
    _rules_cache = None


def _parse_amount(value: Any) -> float | None:
    """Extract a numeric amount from a field value."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Strip currency symbols and commas
        cleaned = (
            value.replace(",", "")
            .replace("$", "")
            .replace("€", "")
            .replace("£", "")
            .strip()
        )
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_date(value: Any) -> date | None:
    """Parse an ISO-format date string."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def evaluate_rules(
    fields: dict[str, dict[str, Any]],
    rules: list[CrossFieldRule] | None = None,
) -> list[Violation | Derivation]:
    """Evaluate cross-field rules against emitted contract fields.

    Validate rules (action='validate'):
      Only evaluates when both fields are PREDICTED (non-null). Skips
      rules when either field is missing/abstained — those are already
      flagged by the per-field review logic.

    Derivation rules (action='set'):
      Only requires the reference field to be PREDICTED. The target
      field may be ABSTAIN or DEFAULT — that's the point of filling it in.

    Args:
        fields: The ``contract["fields"]`` dict from emission.
        rules: Parsed rules (defaults to lazy-loaded schema rules).

    Returns:
        List of Violation and/or Derivation results.
    """
    if rules is None:
        rules = load_rules()

    results: list[Violation | Derivation] = []

    for rule in rules:
        if rule.action == "set":
            # Derivation: only reference field needs to be PREDICTED
            ref_data = fields.get(rule.reference, {})
            if ref_data.get("status") != "PREDICTED":
                continue
            ref_val = ref_data.get("value")
            if ref_val is None:
                continue
            # eq predicate: condition fires when reference == condition_value
            if rule.type == "eq" and str(ref_val) == rule.condition_value:
                results.append(
                    Derivation(
                        rule_type=rule.type,
                        reference=rule.reference,
                        condition_value=rule.condition_value,
                        target=rule.target,  # type: ignore[arg-type]  # validator ensures non-None
                        set_value=rule.set_value,
                    )
                )
            continue

        # action == "validate" path — both fields must be PREDICTED
        # model_validator guarantees rule.field is non-None here
        assert rule.field is not None
        field_data = fields.get(rule.field, {})
        ref_data = fields.get(rule.reference, {})

        if (
            field_data.get("status") != "PREDICTED"
            or ref_data.get("status") != "PREDICTED"
        ):
            continue

        field_val = field_data.get("value")
        ref_val = ref_data.get("value")

        if field_val is None or ref_val is None:
            continue

        if rule.type == "lte":
            f_amt = _parse_amount(field_val)
            r_amt = _parse_amount(ref_val)
            if f_amt is None or r_amt is None:
                continue
            if f_amt > r_amt + rule.tolerance:
                msg = rule.message_template.format(
                    field=rule.field,
                    reference=rule.reference,
                    field_val=f_amt,
                    ref_val=r_amt,
                )
                results.append(
                    Violation(
                        rule_type=rule.type,
                        field=rule.field,
                        reference=rule.reference,
                        severity=rule.severity,
                        message=msg,
                    )
                )

        elif rule.type == "date_gte":
            f_date = _parse_date(field_val)
            r_date = _parse_date(ref_val)
            if f_date is None or r_date is None:
                continue
            if f_date < r_date:
                msg = rule.message_template.format(
                    field=rule.field,
                    reference=rule.reference,
                    field_val=field_val,
                    ref_val=ref_val,
                )
                results.append(
                    Violation(
                        rule_type=rule.type,
                        field=rule.field,
                        reference=rule.reference,
                        severity=rule.severity,
                        message=msg,
                    )
                )

        elif rule.type == "date_max_gap_days":
            f_date = _parse_date(field_val)
            r_date = _parse_date(ref_val)
            if f_date is None or r_date is None:
                continue
            gap = (f_date - r_date).days
            if gap > rule.max_days:
                msg = rule.message_template.format(
                    field=rule.field,
                    reference=rule.reference,
                    field_val=field_val,
                    ref_val=ref_val,
                    days=gap,
                )
                results.append(
                    Violation(
                        rule_type=rule.type,
                        field=rule.field,
                        reference=rule.reference,
                        severity=rule.severity,
                        message=msg,
                    )
                )

    return results
