"""Pure confidence-scoring functions with explicit arguments.

All functions are pure: no globals, no Config dependency, no side effects.
Config methods delegate here, passing their own fields as arguments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .doc import Doc

# ── Sentinel constants (canonical source) ──────────────────────────────
CONFIDENCE_FLOOR: float = 0.0
CONFIDENCE_CEILING: float = 1.0
CONFIDENCE_ABSTAIN: float = 0.0
CONFIDENCE_INFERRED: float = 0.90


# ── Pure functions ─────────────────────────────────────────────────────


def compute_heuristic_scale(heuristic_base: float, decoder_base_cost: float) -> float:
    """Derive the heuristic confidence scaling factor.

    Controls the steepness of the linear cost-to-confidence mapping:
        confidence = base - cost * scale

    Derived as ``heuristic_base / decoder_base_cost`` so the cost range
    [-decoder_base_cost, +decoder_base_cost] maps across [0, 1] confidence.
    """
    return heuristic_base / decoder_base_cost


def compute_confidence(
    cost: float,
    *,
    heuristic_base: float,
    heuristic_scale: float,
    floor: float = CONFIDENCE_FLOOR,
    ceiling: float = CONFIDENCE_CEILING,
    has_ml_model: bool = False,
) -> float:
    """Convert assignment cost to confidence score in [floor, ceiling].

    Heuristic path: ``confidence = heuristic_base - cost * heuristic_scale``.
    Different costs produce different confidences (linear, interpretable).

    ML path: maps cost to [0, 1] via ``1 - cost`` (cost already in [0, 1]
    from sigmoid in ``compute_ranker_cost``).
    """
    if has_ml_model:
        confidence = 1.0 - cost
    else:
        confidence = heuristic_base - cost * heuristic_scale
    return max(floor, min(ceiling, confidence))


def needs_review(
    field_confidences: dict[str, float], auto_approve_threshold: float
) -> bool:
    """True if any field confidence is below the auto-approve threshold."""
    return any(c < auto_approve_threshold for c in field_confidences.values())


def text_layer_degenerate(doc: Doc, threshold: float = 0.30) -> bool:
    """True when the PDF text layer is degenerate and cannot produce valid tokens.

    Degeneracy signal: fraction of chars across all pages where both ``width``
    and ``adv`` are effectively zero (< 1e-6 absolute).  CharEntry uses
    ``extra="allow"`` (Pydantic v2), so pdfplumber-native ``width``/``adv``
    extras are attribute-accessible via ``getattr``.

    Empirical calibration (2026-06-10 dump):
      - degenerate doc (53d57132): 67.5% zero-width/adv chars
      - all 9 good docs:             0.0%
    Threshold 0.30 gives clear separation with ~2x safety margin.

    Zero-char docs (n == 0) are treated as degenerate: no usable text layer
    means no valid candidates — same future-OCR hook applies.

    This predicate is the future-OCR routing hook.  When an OCR path is added,
    replace the ``needs_review=True`` short-circuit in ``run_document_pipeline``
    with the OCR branch.
    """
    n = 0
    zero_wadv = 0
    for page in doc.pages:
        for char in page.chars:
            n += 1
            w = abs(float(getattr(char, "width", 0.0) or 0.0))
            adv = abs(float(getattr(char, "adv", 0.0) or 0.0))
            if w < 1e-6 and adv < 1e-6:
                zero_wadv += 1
    if n == 0:
        return True  # No text layer at all — treat as degenerate.
    return (zero_wadv / n) > threshold
