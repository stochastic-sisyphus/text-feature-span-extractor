"""Top-level emit_document_with_data orchestration."""

from __future__ import annotations

from typing import Any

from .. import confidence as conf
from .. import utils
from ..constants import CURRENCY_CODES
from ..decoder.cost import (
    DISAGREEMENT_CONFIDENCE_DEMOTION,
    DISAGREEMENT_HIGH_THRESHOLD,
)
from ..logging import get_logger
from ..schema import load_field_defs
from ._computed import _compute_field
from ._constants import EVALUATOR_VERSION
from ._field_output import create_field_output
from ._priority import create_evaluation_entry

logger = get_logger(__name__)


def emit_document_with_data(
    sha256: str,
    doc_id: str,
    assignments: dict[str, Any],
    schema_fields: list[str],
    page_count: int,
    confidence_auto_approve: float,
    *,
    heuristic_base: float,
    decoder_base_cost: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    """Pure emit: build contract JSON from assignments.

    All pure compute — no file I/O, no disk writes, no metrics updates.

    Args:
        sha256: Document SHA256 hash
        doc_id: Document identifier (e.g. "fs:abc123")
        assignments: Normalized assignments from decoder
        schema_fields: List of field names from the contract schema
        page_count: Number of pages in the document
        confidence_auto_approve: Threshold for auto-approval (0-1)
        heuristic_base: Base value for heuristic confidence mapping
        decoder_base_cost: Decoder base cost used to compute heuristic scale

    Returns:
        (contract_dict, evaluation_entries, needs_review)
    """
    from ..types import Assignment as _Assignment

    # Partition schema_fields into regular and line-item base names.
    # Line-item base names have line_item_field=True in the schema; they are
    # not emitted as flat fields — their row variants are grouped into line_items.
    _defs = load_field_defs()
    _line_item_bases: set[str] = {
        f
        for f in schema_fields
        if (f_def := _defs.get(f)) is not None and f_def.line_item_field
    }
    _regular_fields: list[str] = [f for f in schema_fields if f not in _line_item_bases]

    # Create contract JSON with version info
    contract: dict[str, Any] = {
        "document_id": doc_id,
        "pages": page_count,
        **utils.get_version_info(),
        "fields": {},
        "line_items": [],
        "experimental": {},  # Top-level experimental object
    }

    evaluation_entries: list[dict[str, Any]] = []
    field_confidences: dict[str, float] = {}  # Track for routing decision

    # Process regular (non-line-item) schema fields
    for field in _regular_fields:
        if field in assignments:
            # Field has assignment from decoder
            assignment = assignments[field]
            field_output = create_field_output(
                field,
                assignment,
                heuristic_base=heuristic_base,
                decoder_base_cost=decoder_base_cost,
            )
            contract["fields"][field] = field_output

            # Track confidence for routing
            field_confidences[field] = field_output["confidence"]

            # Signal disagreement: demote confidence when signals conflict.
            # The candidate dict carries _signal_disagreement:{field} (0-1) from cost.py.
            candidate_obj = assignment.candidate
            disagreement = (
                float(candidate_obj.get(f"_signal_disagreement:{field}", 0.0))
                if candidate_obj is not None
                else 0.0
            )

            if (
                disagreement > DISAGREEMENT_HIGH_THRESHOLD
                and field_output["status"] == "PREDICTED"
            ):
                demotion = 1.0 - DISAGREEMENT_CONFIDENCE_DEMOTION * disagreement
                field_output["confidence"] = round(
                    field_output["confidence"] * demotion, 4
                )
                field_confidences[field] = field_output["confidence"]

            # Add to evaluation entries if ABSTAIN
            if field_output["status"] == "ABSTAIN":
                reason = "ABSTAIN"
                if disagreement > DISAGREEMENT_HIGH_THRESHOLD:
                    reason = "CONFLICTING_SIGNALS | " + reason
                eval_entry = create_evaluation_entry(
                    doc_id,
                    field,
                    assignment,
                    reason,
                    confidence=field_output["confidence"],
                )
                eval_entry["signal_disagreement"] = (
                    disagreement > DISAGREEMENT_HIGH_THRESHOLD
                )
                evaluation_entries.append(eval_entry)
            # Also add if low confidence PREDICTED or heuristic-only
            # (no trained model): when used_ml_model is False, confidence is a
            # heuristic proxy that can exceed auto-approve even on a cold-start.
            # Gate auto-approve on model presence to keep the queue populated
            # until enough labels exist to train.
            elif field_output["status"] == "PREDICTED":
                if (
                    field_output["confidence"] < confidence_auto_approve
                    or not assignment.used_ml_model
                ):
                    reason = "LOW_CONFIDENCE"
                    if disagreement > DISAGREEMENT_HIGH_THRESHOLD:
                        reason = "CONFLICTING_SIGNALS | " + reason
                    eval_entry = create_evaluation_entry(
                        doc_id,
                        field,
                        assignment,
                        reason,
                        confidence=field_output["confidence"],
                    )
                    eval_entry["signal_disagreement"] = (
                        disagreement > DISAGREEMENT_HIGH_THRESHOLD
                    )
                    evaluation_entries.append(eval_entry)
        else:
            # Field missing from assignments - status MISSING
            contract["fields"][field] = {
                "value": None,
                "confidence": conf.CONFIDENCE_ABSTAIN,
                "status": "MISSING",
                "provenance": None,
                "raw_text": None,
            }

            # Track as zero confidence for routing
            field_confidences[field] = conf.CONFIDENCE_ABSTAIN

            # Add to evaluation entries
            eval_entry = create_evaluation_entry(
                doc_id,
                field,
                _Assignment(
                    assignment_type="NONE",
                    candidate_index=None,
                    cost=0.0,
                    field=field,
                    used_ml_model=False,
                    ml_probability=None,
                ),
                "MISSING",
                confidence=conf.CONFIDENCE_ABSTAIN,
            )
            evaluation_entries.append(eval_entry)

    # Collect line-item row assignments into contract["line_items"].
    # Row variant keys follow the pattern "{base}__row_{i}" where base is a
    # line_item_field name.  We group by row index across all base fields so
    # each row becomes one dict entry under line_items, ordered by row index.
    if _line_item_bases:
        # Collect: row_index -> {base_name -> field_output}
        row_map: dict[int, dict[str, Any]] = {}
        for li_key, li_asgn in assignments.items():
            parts = li_key.rsplit("__row_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            base, idx_str = parts[0], parts[1]
            if base not in _line_item_bases:
                continue
            row_idx = int(idx_str)
            row_map.setdefault(row_idx, {})
            row_output = create_field_output(
                base,
                li_asgn,
                heuristic_base=heuristic_base,
                decoder_base_cost=decoder_base_cost,
            )
            row_map[row_idx][base] = row_output

        contract["line_items"] = [row_map[i] for i in sorted(row_map)]

    # Post-processing: compute fields driven by schema
    for cf_name, cf_fd in load_field_defs().items():
        if not cf_fd.computed:
            continue
        computed_fn = cf_fd.computed_fn
        source_fields = list(cf_fd.computed_from)
        if not computed_fn:
            continue

        # Special handling for Currency: check if inference is needed
        if cf_name == "Currency":
            currency_field = contract["fields"].get("Currency")
            currency_needs_inference = False
            if currency_field:
                if currency_field["status"] in ("ABSTAIN", "MISSING"):
                    currency_needs_inference = True
                elif (
                    currency_field["status"] == "PREDICTED"
                    and currency_field.get("value")
                    and currency_field["value"].upper()
                    not in {c.upper() for c in CURRENCY_CODES}
                ):
                    currency_needs_inference = True
            if not currency_needs_inference:
                continue

        result = _compute_field(
            cf_name, computed_fn, source_fields, assignments, contract["fields"]
        )
        if result:
            contract["fields"][cf_name] = result
            field_confidences[cf_name] = result["confidence"]
            # Remove from review queue if it was there
            evaluation_entries = [
                e for e in evaluation_entries if e["field"] != cf_name
            ]
            logger.info(
                "custom_field_computed",
                field=cf_name,
                doc_id=doc_id,
                value=result["value"],
            )
        elif cf_name in schema_fields and cf_name not in contract["fields"]:
            # Source fields missing — mark as MISSING
            contract["fields"][cf_name] = {
                "value": None,
                "confidence": 0.0,
                "status": "MISSING",
                "provenance": None,
                "raw_text": None,
            }

    # Fallback chain: for every regular field whose current output has no value
    # (status ABSTAIN or MISSING), apply:
    #   1. computed_fn as fallback (when computed=False and computed_fn is set)
    #   2. default_value (when set)
    # This is distinct from the primary-derive pass above (computed=True).
    for fb_name, fb_fd in load_field_defs().items():
        if fb_name not in contract["fields"]:
            continue
        current = contract["fields"][fb_name]
        if current.get("value") is not None:
            # Extraction already produced a value — chain does not fire
            continue

        # Step 1: fallback computed_fn (only when computed=False)
        if not fb_fd.computed and fb_fd.computed_fn:
            fb_source_fields = list(fb_fd.computed_from)
            fb_result = _compute_field(
                fb_name,
                fb_fd.computed_fn,
                fb_source_fields,
                assignments,
                contract["fields"],
            )
            if fb_result is not None:
                contract["fields"][fb_name] = fb_result
                field_confidences[fb_name] = fb_result["confidence"]
                evaluation_entries = [
                    e for e in evaluation_entries if e["field"] != fb_name
                ]
                logger.info(
                    "fallback_computed_fn_applied",
                    field=fb_name,
                    doc_id=doc_id,
                    fn=fb_fd.computed_fn,
                    value=fb_result["value"],
                )
                continue

        # Step 2: literal default_value
        if fb_fd.default_value is not None:
            contract["fields"][fb_name] = {
                "value": fb_fd.default_value,
                "confidence": conf.CONFIDENCE_INFERRED,
                "status": "DEFAULT",
                "provenance": {"page": 0, "bbox": [0, 0, 0, 0], "token_span": []},
                "raw_text": None,
            }
            field_confidences[fb_name] = conf.CONFIDENCE_INFERRED
            evaluation_entries = [
                e for e in evaluation_entries if e["field"] != fb_name
            ]
            logger.info(
                "default_value_applied",
                field=fb_name,
                doc_id=doc_id,
                value=fb_fd.default_value,
            )

    # Cross-field validation and derivation: check/apply inter-field rules
    from ..cross_field_rules import (
        Derivation as _Derivation,
    )
    from ..cross_field_rules import (
        evaluate_rules as evaluate_cross_field_rules,
    )
    from ..settings import settings as _settings

    cross_results = evaluate_cross_field_rules(contract["fields"])
    penalty = _settings.cross_field_penalty

    for v in cross_results:
        # Derivation rules run after default_value — intentionally overwrite DEFAULT.
        if isinstance(v, _Derivation):
            contract["fields"][v.target] = {
                "value": v.set_value,
                "confidence": conf.CONFIDENCE_INFERRED,
                "status": "PREDICTED",
                "provenance": {
                    "page": 0,
                    "bbox": [0, 0, 0, 0],
                    "token_span": [],
                    "derived_from": v.reference,
                },
                "raw_text": None,
            }
            field_confidences[v.target] = conf.CONFIDENCE_INFERRED
            evaluation_entries = [
                e for e in evaluation_entries if e["field"] != v.target
            ]
            logger.info(
                "derivation_applied",
                doc_id=doc_id,
                field=v.target,
                value=v.set_value,
                reference=v.reference,
                condition_value=v.condition_value,
            )
            continue

        involved = [v.field, v.reference]
        if v.severity == "error":
            # Penalise confidence of involved fields and block auto-approval
            for fname in involved:
                if fname in field_confidences:
                    old_conf = field_confidences[fname]
                    new_conf = max(conf.CONFIDENCE_FLOOR, old_conf - penalty)
                    field_confidences[fname] = new_conf
                    contract["fields"][fname]["confidence"] = new_conf
            evaluation_entries.append(
                {
                    "doc_id": doc_id,
                    "field": v.field,
                    "evaluator_version": EVALUATOR_VERSION,
                    "priority_score": 0.9,
                    "reason": f"CROSS_FIELD_ERROR: {v.message}",
                    "signal_disagreement": False,
                    "used_ml_model": False,
                }
            )
            logger.warning("cross_field_error", doc_id=doc_id, violation=v.message)
        else:
            # Warning: log + add review reason but don't penalise confidence
            evaluation_entries.append(
                {
                    "doc_id": doc_id,
                    "field": v.field,
                    "evaluator_version": EVALUATOR_VERSION,
                    "priority_score": 0.5,
                    "reason": f"CROSS_FIELD_WARNING: {v.message}",
                    "signal_disagreement": False,
                    "used_ml_model": False,
                }
            )
            logger.info("cross_field_warning", doc_id=doc_id, violation=v.message)

    # Deduplicate evaluation entries by (doc_id, field): merge reasons, keep higher priority.
    seen: dict[tuple[str, str], int] = {}  # (doc_id, field) -> index in deduped list
    deduped: list[dict[str, Any]] = []
    for entry in evaluation_entries:
        key = (entry["doc_id"], entry["field"])
        if key in seen:
            existing = deduped[seen[key]]
            existing["reason"] = existing["reason"] + "; " + entry["reason"]
            if (entry.get("priority_score") or 0.0) > (
                existing.get("priority_score") or 0.0
            ):
                existing["priority_score"] = entry["priority_score"]
            if entry.get("signal_disagreement"):
                existing["signal_disagreement"] = True
        else:
            seen[key] = len(deduped)
            deduped.append(dict(entry))
    evaluation_entries = deduped

    # Determine if document needs human review
    needs_review = conf.needs_review(field_confidences, confidence_auto_approve)

    return contract, evaluation_entries, needs_review
