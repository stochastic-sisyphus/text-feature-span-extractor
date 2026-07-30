"""Review-queue priority scoring and evaluation-entry construction."""

from __future__ import annotations

from typing import Any

from ..schema import load_field_defs
from ._constants import _REASON_WEIGHT, EVALUATOR_VERSION


def compute_priority_score(field: str, reason: str, confidence: float) -> float:
    """Deterministic priority from field importance + reason + confidence.

    Returns score in [0, 1].

    The confidence term uses uncertainty sampling: items near confidence=0.5
    are most valuable to label (maximum information gain). Very low or very
    high confidence items contribute less — the model already knows what to
    do with them. Formula: learning_value = 1 - |2*confidence - 1|, which
    peaks at 1.0 when confidence=0.5 and falls to 0.0 at the extremes.
    """
    _fd = load_field_defs().get(field)
    field_w = _fd.importance if _fd is not None else 0.5
    reason_w = _REASON_WEIGHT.get(reason, 0.5)
    conf_clamped = max(0.0, min(1.0, confidence))
    learning_value = 1.0 - abs(2.0 * conf_clamped - 1.0)

    score = 0.35 * field_w + 0.40 * reason_w + 0.25 * learning_value
    return max(0.0, min(1.0, score))


def create_evaluation_entry(
    doc_id: str,
    field: str,
    assignment: Any,
    reason: str,
    confidence: float = 0.0,
) -> dict[str, Any]:
    """Build a doc_evaluations entry for a field that needs review.

    Args:
        doc_id: Document identifier
        field: Field name
        assignment: Assignment object from decoder
        reason: Reason for review (e.g., "ABSTAIN", "LOW_CONFIDENCE")
        confidence: ML confidence score for the assignment

    Returns:
        Evaluation entry dict shaped for upsert_doc_evaluations.
    """
    priority_score = compute_priority_score(field, reason, confidence)
    used_ml = bool(
        assignment is not None and getattr(assignment, "used_ml_model", False)
    )

    return {
        "doc_id": doc_id,
        "field": field,
        "evaluator_version": EVALUATOR_VERSION,
        "priority_score": round(priority_score, 4),
        "reason": reason,
        "signal_disagreement": False,  # caller overwrites when disagreement detected
        "used_ml_model": used_ml,
    }
