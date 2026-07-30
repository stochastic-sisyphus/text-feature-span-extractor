"""Span bucket classification (amount/date/id/name/keyword)."""

from __future__ import annotations

from ..constants import (
    BUCKET_AMOUNT_LIKE,
    BUCKET_DATE_LIKE,
    BUCKET_ID_LIKE,
    BUCKET_KEYWORD_PROXIMAL,
    BUCKET_NAME_LIKE,
)
from ..patterns import classify_bucket


def _classify_span_bucket(
    text: str,
    proximity_score: float,
    coverage_stats: dict[str, int],
) -> str | None:
    """Classify a span into a candidate bucket type, or None to skip.

    Routes through ``classify_bucket`` — the single source of truth.
    """
    bucket = classify_bucket(text)

    if bucket == BUCKET_ID_LIKE:
        coverage_stats["id_like_spans"] += 1
    elif bucket == BUCKET_DATE_LIKE:
        coverage_stats["date_like_spans"] += 1
    elif bucket == BUCKET_AMOUNT_LIKE:
        coverage_stats["amount_like_spans"] += 1
    elif bucket == BUCKET_NAME_LIKE:
        coverage_stats["name_like_spans"] = coverage_stats.get("name_like_spans", 0) + 1
    elif bucket is None and proximity_score > 0.3:
        bucket = BUCKET_KEYWORD_PROXIMAL
        coverage_stats["cue_proximal_spans"] += 1

    return bucket
