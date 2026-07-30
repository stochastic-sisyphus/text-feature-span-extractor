"""Computed-field dispatcher for schema-driven derived fields."""

from __future__ import annotations

from typing import Any

from .. import confidence as conf


def _compute_field(
    field_name: str,
    computed_fn: str,
    source_fields: list[str],
    assignments: dict[str, Any],
    contract_fields: dict[str, Any],
) -> dict[str, Any] | None:
    """Generic computed field dispatcher.

    Args:
        field_name: Name of computed field
        computed_fn: Function to dispatch on (infer_currency, concat_strip)
        source_fields: List of source field names from schema
        assignments: Raw decoder output (for currency inference)
        contract_fields: Processed contract fields dict (for billing reference)

    Returns:
        Field output dict (value, confidence, status, provenance, raw_text) or None
    """
    if computed_fn == "infer_currency":
        # Source fields from schema: ["TotalAmount", "Subtotal", "TaxAmount"]
        inferred_code = None
        source_field = None
        for fname in source_fields:
            assignment = assignments.get(fname)
            if assignment is None:
                continue
            if assignment.assignment_type == "NONE":
                continue
            if assignment.currency_code:
                inferred_code = str(assignment.currency_code)
                source_field = fname
                break

        if not inferred_code:
            return None

        # Build provenance from the source amount candidate
        inferred_provenance: dict[str, Any] = {
            "page": 0,
            "bbox": [0, 0, 0, 0],
            "token_span": [],
        }
        if source_field and assignments[source_field].candidate is not None:
            src_cand = assignments[source_field].candidate
            inferred_provenance = {
                "page": int(src_cand.get("page_idx", 0)),
                "bbox": [
                    float(src_cand.get("bbox_norm_x0", 0)),
                    float(src_cand.get("bbox_norm_y0", 0)),
                    float(src_cand.get("bbox_norm_x1", 0)),
                    float(src_cand.get("bbox_norm_y1", 0)),
                ],
                "token_span": [
                    str(idx)
                    for idx in src_cand.get(
                        "token_indices", [src_cand.get("token_idx", 0)]
                    )
                ],
            }

        src_assignment = assignments.get(source_field or "")
        return {
            "value": inferred_code,
            "confidence": conf.CONFIDENCE_INFERRED,
            "status": "PREDICTED",
            "provenance": inferred_provenance,
            "raw_text": src_assignment.raw_text if src_assignment else None,
        }

    if computed_fn == "concat_strip":
        # Source fields from schema: ["CustomerAccount", "InvoiceDate"]
        # Use contract_fields (already processed)
        if len(source_fields) < 2:
            return None

        field1_name = source_fields[0]
        field2_name = source_fields[1]
        field1 = contract_fields.get(field1_name, {})
        field2 = contract_fields.get(field2_name, {})

        if (
            field1.get("status") == "PREDICTED"
            and field1.get("value")
            and field2.get("status") == "PREDICTED"
            and field2.get("value")
        ):
            # field2 is InvoiceDate (already ISO format), strip dashes
            date_compact = str(field2["value"]).replace("-", "")
            computed_value = str(field1["value"]) + date_compact

            return {
                "value": computed_value,
                "confidence": conf.CONFIDENCE_INFERRED,
                "status": "PREDICTED",
                "provenance": {
                    "page": 0,
                    "bbox": [0, 0, 0, 0],
                    "token_span": [],
                    "computed_from": source_fields,
                },
                "raw_text": None,
            }

        return None

    if computed_fn == "coalesce":
        # Return the first non-null PREDICTED value from source_fields.
        # Looks up each name in contract_fields (already processed output dicts).
        for fname in source_fields:
            field_out = contract_fields.get(fname, {})
            if (
                field_out.get("status") == "PREDICTED"
                and field_out.get("value") is not None
            ):
                return {
                    "value": field_out["value"],
                    "confidence": field_out["confidence"],
                    "status": "PREDICTED",
                    "provenance": {
                        "page": 0,
                        "bbox": [0, 0, 0, 0],
                        "token_span": [],
                        "computed_from": [fname],
                    },
                    "raw_text": None,
                }
        return None

    # Unknown computed_fn
    return None
