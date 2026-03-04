"""Schema registry: single source of truth for field metadata.

Loads contract schema once and provides typed accessors for all field properties.
Replaces scattered dicts in decoder.py, config.py, emit.py with unified schema-driven API.
"""

from __future__ import annotations

from typing import Any

from . import utils

# Module-level schema cache
_SCHEMA: dict[str, Any] | None = None

# Type -> anchor type mapping (default if anchor_override not set)
TYPE_TO_ANCHOR: dict[str, str | None] = {
    "amount": "total",
    "date": "date",
    "id": "id",
    "text": "name",
    "address": "name",
    "currency": None,
    "email": None,
    "phone": None,
    "number": "total",
}

# Types that default to keyword_proximal=True
_KEYWORD_PROXIMAL_TYPES: frozenset[str] = frozenset({"id", "amount", "date"})


def _load() -> dict[str, Any]:
    """Load schema once, cache in module-level _SCHEMA."""
    global _SCHEMA
    if _SCHEMA is None:
        _SCHEMA = utils.load_contract_schema()
    return _SCHEMA


def field_def(name: str) -> dict[str, Any]:
    """Return field definition dict from schema (or empty dict if unknown)."""
    schema = _load()
    field_defs: dict[str, Any] = schema.get("field_definitions", {})
    result: dict[str, Any] = field_defs.get(name, {})
    return result


def field_type(name: str) -> str:
    """Return field type (default: 'text')."""
    fdef = field_def(name)
    result: str = fdef.get("type", "text")
    return result


def fields_by_type(type_name: str) -> list[str]:
    """Return list of field names matching the given type."""
    schema = _load()
    field_defs = schema.get("field_definitions", {})
    return [
        fname for fname, fdef in field_defs.items() if fdef.get("type") == type_name
    ]


def name_fields() -> list[str]:
    """Return list of field names that are text fields with fuzzy name matching.

    These are text fields with keyword_proximal=false, which indicates they should
    use case-insensitive fuzzy matching instead of keyword proximity.
    """
    schema = _load()
    field_defs = schema.get("field_definitions", {})
    return [
        fname
        for fname, fdef in field_defs.items()
        if fdef.get("type") == "text" and fdef.get("keyword_proximal") is False
    ]


def anchor_type(name: str) -> str | None:
    """Return anchor type for field.

    Returns anchor_override if explicitly set (even if null).
    Otherwise returns TYPE_TO_ANCHOR default for field type.
    """
    fdef = field_def(name)

    # Check if anchor_override is explicitly set (even if null)
    if "anchor_override" in fdef:
        result: str | None = fdef["anchor_override"]
        return result

    # Fall back to type-based default
    ftype: str = fdef.get("type", "text")
    return TYPE_TO_ANCHOR.get(ftype)


def is_footer(name: str) -> bool:
    """Return True if field has spatial_region == 'footer'."""
    fdef = field_def(name)
    return fdef.get("spatial_region") == "footer"


def is_header(name: str) -> bool:
    """Return True if field has spatial_region == 'header'."""
    fdef = field_def(name)
    return fdef.get("spatial_region") == "header"


def priority_bonus(name: str) -> float:
    """Return priority_bonus from schema (default: 0.0)."""
    fdef = field_def(name)
    return float(fdef.get("priority_bonus", 0.0))


def importance(name: str) -> float:
    """Return importance from schema (default: 0.5)."""
    fdef = field_def(name)
    return float(fdef.get("importance", 0.5))


def confidence_threshold(name: str) -> float:
    """Return confidence_threshold from schema (default: 0.75)."""
    fdef = field_def(name)
    return float(fdef.get("confidence_threshold", 0.75))


def is_keyword_proximal(name: str) -> bool:
    """Return keyword_proximal flag.

    If explicitly set in schema, use that value.
    Otherwise default to True if type in {id, amount, date}.
    """
    fdef = field_def(name)

    # Check for explicit value
    if "keyword_proximal" in fdef:
        return bool(fdef["keyword_proximal"])

    # Fall back to type-based default
    ftype = fdef.get("type", "text")
    return ftype in _KEYWORD_PROXIMAL_TYPES


def computed_fields() -> list[str]:
    """Return list of fields where computed=True."""
    schema = _load()
    field_defs = schema.get("field_definitions", {})
    return [fname for fname, fdef in field_defs.items() if fdef.get("computed", False)]


def dataverse_column(name: str) -> str | None:
    """Return dataverse_column from schema (default: None)."""
    fdef = field_def(name)
    return fdef.get("dataverse_column")


def spatial_bias(name: str) -> dict[str, Any] | None:
    """Return spatial_bias metadata from schema (or None if not set).

    spatial_bias controls position-based penalties for fields:
    - position: "top" or "bottom" (expected region on page)
    - penalty: cost addition when candidate is in wrong region
    - threshold: center_y threshold for triggering penalty
    """
    fdef = field_def(name)
    return fdef.get("spatial_bias")


def anchor_bonus_override(name: str) -> dict[str, Any] | None:
    """Return anchor_bonus_override metadata from schema (or None if not set).

    anchor_bonus_override enables extra directional bonuses for specific
    anchor types beyond the default anchor logic:
    - anchor: which anchor type to check (e.g., "tax")
    - below_bonus: bonus when candidate is below this anchor
    - below_dist_threshold: max distance for below bonus
    - reading_order_bonus: bonus for reading-order match
    - reading_order_dist_threshold: max distance for reading-order bonus
    """
    fdef = field_def(name)
    return fdef.get("anchor_bonus_override")


def cross_field_rules() -> list[dict[str, Any]]:
    """Return cross-field validation rules from schema."""
    schema = _load()
    result: list[dict[str, Any]] = schema.get("cross_field_rules", [])
    return result


def clear_cache() -> None:
    """Clear cached schema (for testing)."""
    global _SCHEMA
    _SCHEMA = None
