"""Leaf math helpers: signal disagreement, directional bonus, region bonus, amount parsing."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import weights as _w_mod
from ..weights import DecoderWeights

MAX_SIGNAL_VARIANCE: float = 4.0


def compute_signal_disagreement(components: list[float]) -> float:
    """Compute normalized signal disagreement from weighted signal components.

    Variance of the components, normalized to [0, 1] using an empirical
    max-variance ceiling.  The returned value represents how much the
    individual signals pull in conflicting directions — 0 means perfect
    agreement, 1 means extreme conflict.

    Args:
        components: List of weighted signal component values.

    Returns:
        Disagreement score in [0, 1].
    """
    if len(components) < 2:
        return 0.0
    arr = np.array(components, dtype=np.float64)
    variance = float(np.var(arr))
    # Empirical ceiling: with 12 components ranging roughly ±2, max
    # variance is ~4.  We use MAX_SIGNAL_VARIANCE as normaliser so disagreement
    # saturates at 1.0 for extreme cases.
    return min(1.0, variance / MAX_SIGNAL_VARIANCE)


def _header_region_bonus(
    candidate: dict[str, Any],
    w: DecoderWeights | None = None,
) -> float:
    """
    Compute header region bonus for fields with spatial_region='header'.

    Fields like VendorName/CustomerName appear in header region (top of invoice,
    near logo) and benefit from strong spatial signals rather than anchor-based matching.

    Args:
        candidate: Candidate dictionary with spatial features

    Returns:
        Header region bonus (positive = good spatial match)
    """
    if w is None:
        w = _w_mod.DWeights
    bonus = 0.0
    in_header = candidate.get("in_top_quarter", 0.0)
    distance_to_top = candidate.get("distance_to_top", 1.0)

    # Strong bonus for header region (logo/name area)
    if in_header > 0:
        bonus += w.HEADER_REGION_BONUS

    # Bonus for being near top of page (inverse of distance)
    if distance_to_top < w.HEADER_TOP_CLOSE_THRESHOLD:
        bonus += w.HEADER_TOP_CLOSE_BONUS
    elif distance_to_top < w.HEADER_TOP_MID_THRESHOLD:
        bonus += w.HEADER_TOP_MID_BONUS

    return bonus


def _compute_directional_bonus_for_anchor(
    candidate: dict[str, Any],
    anchor_type: str,
    w: DecoderWeights | None = None,
) -> float:
    """
    Compute directional bonus for a candidate based on its proximity to an anchor type.

    This centralizes the directional feature logic that was previously duplicated
    for each field type (amount, date, id, name).

    Args:
        candidate: Candidate dictionary with directional features
        anchor_type: The anchor type to check ("total", "tax", "date", "id", "name")

    Returns:
        Directional bonus (positive = good match, negative = poor match)
    """
    if w is None:
        w = _w_mod.DWeights
    if anchor_type is None:
        return 0.0

    dist = candidate.get(f"dist_to_{anchor_type}", 1.0)
    below = candidate.get(f"below_{anchor_type}", 0.0)
    reading_order = candidate.get(f"reading_order_{anchor_type}", 0.0)
    aligned_y = candidate.get(f"aligned_y_{anchor_type}", 0.0)

    bonus = 0.0

    # Strong bonus for being in same column and below anchor header
    if below > 0 and dist < w.DIRECTIONAL_BELOW_CLOSE_THRESHOLD:
        bonus += w.DIRECTIONAL_BELOW_CLOSE_BONUS

    # Bonus for reading order (label: value pattern)
    if reading_order > 0 and dist < w.DIRECTIONAL_READING_ORDER_THRESHOLD:
        bonus += w.DIRECTIONAL_READING_ORDER_BONUS

    # Bonus for same row, close distance ("Label: Value" horizontal pattern)
    if aligned_y > 0 and dist < w.DIRECTIONAL_SAME_ROW_THRESHOLD:
        bonus += w.DIRECTIONAL_SAME_ROW_BONUS

    # Penalty for being far from anchor
    if dist > w.DIRECTIONAL_FAR_THRESHOLD:
        bonus += w.DIRECTIONAL_FAR_PENALTY

    return bonus


def _parse_amount_value(text: str) -> float | None:
    """Extract numeric value from amount text for cross-field comparison.

    Handles currency symbols, commas, and common formats. Returns None
    if the text cannot be parsed as a numeric amount.
    """
    if not text or not text.strip():
        return None
    clean = text.strip()
    # Strip currency symbols
    for sym in "$\u20ac\u00a3\u00a5\u20b9\u20bd":
        clean = clean.replace(sym, "")
    # Strip currency codes
    for code in ("USD", "EUR", "GBP", "CAD", "AUD", "JPY", "CHF", "CNY", "INR"):
        clean = clean.replace(code, "").replace(code.lower(), "")
    clean = clean.strip().replace(",", "")
    if not clean:
        return None
    try:
        return float(clean)
    except ValueError:
        return None
