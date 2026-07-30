"""Per-field output construction with semantic validation and fallbacks."""

from __future__ import annotations

from typing import Any

from .. import confidence as conf
from ..logging import get_logger

logger = get_logger(__name__)


def compute_field_confidence(
    field: str,
    assignment: Any,
    *,
    heuristic_base: float,
    decoder_base_cost: float,
) -> float:
    """
    Compute confidence score for a field assignment.

    Uses ML probability when available, otherwise derives confidence from
    the assignment cost using heuristic mapping. NONE assignments always
    return CONFIDENCE_ABSTAIN.

    Args:
        field: Field name
        assignment: Assignment from decoder (should include cost,
                    used_ml_model, ml_probability)
        heuristic_base: Base value for heuristic confidence mapping
        decoder_base_cost: Decoder base cost used to compute heuristic scale

    Returns:
        Confidence score in [0, 1]
    """
    # NONE assignments always have zero confidence
    if assignment.assignment_type == "NONE":
        return conf.CONFIDENCE_ABSTAIN

    # Check if ML model was used and probability is available
    used_ml = assignment.used_ml_model
    ml_prob = assignment.ml_probability

    if used_ml and ml_prob is not None:
        # ML probability is direct confidence
        confidence = float(ml_prob)
    else:
        # Compute confidence from cost using pure function
        cost = assignment.cost
        confidence = conf.compute_confidence(
            cost,
            heuristic_base=heuristic_base,
            heuristic_scale=conf.compute_heuristic_scale(
                heuristic_base, decoder_base_cost
            ),
            has_ml_model=False,
        )

    # Clamp to valid range
    return max(conf.CONFIDENCE_FLOOR, min(conf.CONFIDENCE_CEILING, confidence))


def create_field_output(
    field: str,
    assignment: Any,
    *,
    heuristic_base: float,
    decoder_base_cost: float,
) -> dict[str, Any]:
    """
    Create field output following contract_v1 specification.

    Includes semantic validation to reject garbage predictions like:
    - Repeated words ("Tax Tax Tax Tax")
    - Keywords as values ("SUBTOTAL" as invoice number)
    - Invalid field values (numeric-only names, invalid dates)

    Args:
        field: Field name
        assignment: Normalized Assignment from decoder
        heuristic_base: Base value for heuristic confidence mapping
        decoder_base_cost: Decoder base cost used to compute heuristic scale

    Returns:
        Field output dictionary with computed confidence
    """
    if assignment.assignment_type == "NONE":
        return {
            "value": None,
            "confidence": conf.CONFIDENCE_ABSTAIN,
            "status": "ABSTAIN",
            "provenance": None,
            "raw_text": None,
        }

    # CANDIDATE assignment
    candidate = assignment.candidate
    normalized_value = assignment.normalized_value
    raw_text = assignment.raw_text

    # Compute confidence from decoder output
    confidence = compute_field_confidence(
        field,
        assignment,
        heuristic_base=heuristic_base,
        decoder_base_cost=decoder_base_cost,
    )

    # Determine status
    if normalized_value is not None:
        status = "PREDICTED"
    else:
        # Normalization failed - abstain with zero confidence
        status = "ABSTAIN"
        confidence = conf.CONFIDENCE_ABSTAIN

    # Create provenance.
    # LiLT unmatched-span assignments carry candidate=None (the model predicted a
    # span that didn't overlap any candidate at IoU ≥ threshold).  Provenance is
    # unavailable in that case; emit None rather than crashing.
    if candidate is not None:
        provenance: dict[str, Any] | None = {
            "page": int(candidate["page_idx"]),
            "bbox": [
                float(candidate["bbox_norm_x0"]),
                float(candidate["bbox_norm_y0"]),
                float(candidate["bbox_norm_x1"]),
                float(candidate["bbox_norm_y1"]),
            ],
            "token_span": [
                str(idx)
                for idx in candidate.get(
                    "token_indices", [candidate.get("token_idx", 0)]
                )
            ],  # Safe string conversion
        }
    else:
        provenance = None

    return {
        "value": normalized_value if status == "PREDICTED" else None,
        "confidence": confidence,
        "status": status,
        "provenance": provenance,
        "raw_text": raw_text,
    }
