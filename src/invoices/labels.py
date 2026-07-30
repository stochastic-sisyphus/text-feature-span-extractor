"""Label alignment for training: corrections, approvals, and candidate matching."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import polars as pl

from . import normalize
from . import schema as schema_registry
from .geometry import compute_iou

__all__ = [
    "AMOUNT_FIELDS",
    "DATE_FIELDS",
    "ID_FIELDS",
    "NAME_FIELDS",
    "match_value_to_candidates",
    "parse_date_variants",
]

# PEP 562 lazy attributes — resolved at access time via __getattr__ below.
# Declared here so static analysers recognise these as module-level names.
AMOUNT_FIELDS: frozenset[str]
DATE_FIELDS: frozenset[str]
ID_FIELDS: frozenset[str]
NAME_FIELDS: frozenset[str]


# ---------------------------------------------------------------------------
# Bbox helpers — multi-span ↔ flat 4-tuple compat
# ---------------------------------------------------------------------------


def _normalize_bbox(raw: Any) -> list[list[float]] | None:
    """Return per-span bbox list regardless of legacy vs. new DB shape.

    Legacy shape (old DB rows): [x0, y0, x1, y1]  → [[x0, y0, x1, y1]]
    New shape (multi-span):     [[x0,y0,x1,y1], …] → as-is
    None:                        None               → None
    """
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    # Flat list of numbers → single-span legacy row
    if isinstance(raw[0], (int, float)):
        return [list(raw)]
    # Already list-of-lists
    return [list(span) for span in raw]


def _union_bbox(spans: list[list[float]]) -> list[float]:
    """Compute the enclosing bbox over a list of [x0,y0,x1,y1] spans."""
    return [
        min(s[0] for s in spans),
        min(s[1] for s in spans),
        max(s[2] for s in spans),
        max(s[3] for s in spans),
    ]


# ---------------------------------------------------------------------------
# Field type classification for matching strategies
# ---------------------------------------------------------------------------


def _date_fields() -> frozenset[str]:
    return frozenset(
        n
        for n, fd in schema_registry.load_field_defs().items()
        if fd.normalizer == "date"
    )


def _amount_fields() -> frozenset[str]:
    return frozenset(
        n
        for n, fd in schema_registry.load_field_defs().items()
        if fd.normalizer == "amount"
    )


def _id_fields() -> frozenset[str]:
    return frozenset(
        n
        for n, fd in schema_registry.load_field_defs().items()
        if fd.normalizer == "id"
    )


def _name_fields() -> frozenset[str]:
    return frozenset(
        n
        for n, fd in schema_registry.load_field_defs().items()
        if fd.normalizer == "name"
    )


def __getattr__(name: str) -> frozenset[str]:
    if name == "DATE_FIELDS":
        return _date_fields()
    if name == "AMOUNT_FIELDS":
        return _amount_fields()
    if name == "ID_FIELDS":
        return _id_fields()
    if name == "NAME_FIELDS":
        return _name_fields()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Matching helpers (adapted from scripts/bridge_ground_truth.py)
# ---------------------------------------------------------------------------


def parse_date_variants(iso_date: str) -> list[str]:
    """Generate common date format variants from an ISO date string."""
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError:
        return [iso_date]

    return [
        iso_date,  # 2024-07-31
        dt.strftime("%m/%d/%Y"),  # 07/31/2024
        dt.strftime("%-m/%-d/%Y"),  # 7/31/2024
        dt.strftime("%m/%d/%y"),  # 07/31/24
        dt.strftime("%-m/%-d/%y"),  # 7/31/24
        dt.strftime("%-m/%d/%y"),  # 7/31/24 (hybrid)
        dt.strftime("%-m/%d/%Y"),  # 7/31/2024 (hybrid)
        dt.strftime("%b %d"),  # Jul 31
        dt.strftime("%b %-d"),  # Jul 31 (no leading zero)
        dt.strftime("%B %d"),  # July 31
        dt.strftime("%d %b"),  # 31 Jul
        dt.strftime("%d %b %Y"),  # 31 Jul 2024
        dt.strftime("%d-%b-%y"),  # 31-Jul-24
        dt.strftime("%d-%b-%Y"),  # 31-Jul-2024
        dt.strftime("%d/%m/%Y"),  # 31/07/2024
        dt.strftime("%d-%m-%Y"),  # 31-07-2024
        dt.strftime("%m-%d-%Y"),  # 07-31-2024
        dt.strftime("%b %d, %Y"),  # Jul 31, 2024
        dt.strftime("%B %d, %Y"),  # July 31, 2024
    ]


def match_value_to_candidates(
    candidates_df: pl.DataFrame,
    field: str,
    value: str,
    correct_bbox: list[float] | None = None,
) -> tuple[int | None, float]:
    """Match a value to the best candidate using field-specific strategies.

    When correct_bbox is provided and multiple candidates match the same text,
    uses IoU (intersection over union) of bounding boxes to pick the intended
    occurrence.

    Args:
        candidates_df: DataFrame of candidates with raw_text and optional bbox columns
        field: Field name (used to select matching strategy)
        value: The value to match against candidate text
        correct_bbox: Optional [x0, y0, x1, y1] bbox to disambiguate duplicates

    Returns:
        (candidate_idx, confidence) or (None, 0.0) if no match
    """
    value_str = str(value)

    if field in _date_fields():
        best_idx, best_conf = _match_date(candidates_df, value_str)
    elif field in _amount_fields():
        best_idx, best_conf = _match_amount(candidates_df, value_str)
    elif field in _id_fields():
        best_idx, best_conf = _match_id(candidates_df, value_str)
    elif field in _name_fields():
        best_idx, best_conf = _match_name(candidates_df, value_str)
    else:
        best_idx, best_conf = _match_exact(candidates_df, value_str)

    # If no bbox hint or no initial match, return as-is
    if correct_bbox is None or best_idx is None:
        return (best_idx, best_conf)

    # Bbox disambiguation: find all candidates with the same text match
    # and pick the one with highest IoU to correct_bbox
    best_row = candidates_df.row(best_idx, named=True)
    best_text = str(best_row.get("raw_text", "")).lower().strip()
    rival_indices: list[int] = []

    for i, row in enumerate(candidates_df.iter_rows(named=True)):
        raw = str(row.get("raw_text", "")).lower().strip()
        if raw == best_text:
            rival_indices.append(i)

    # Only disambiguate if there are actual duplicates
    if len(rival_indices) <= 1:
        return (best_idx, best_conf)

    # Pick the candidate whose bbox has highest IoU with correct_bbox
    best_iou = -1.0
    best_iou_idx = best_idx

    for idx in rival_indices:
        candidate_bbox = candidates_df.row(idx, named=True).get("bbox")
        if candidate_bbox is None:
            continue
        # Handle both list and tuple bbox formats
        if hasattr(candidate_bbox, "__len__") and len(candidate_bbox) == 4:
            iou = compute_iou(tuple(correct_bbox), tuple(candidate_bbox))  # type: ignore[arg-type]
            if iou > best_iou:
                best_iou = iou
                best_iou_idx = idx

    return (best_iou_idx, best_conf)


def _match_date(df: pl.DataFrame, iso_date: str) -> tuple[int | None, float]:
    """Match date values using multiple format variants with fuzzy matching."""
    variants = parse_date_variants(iso_date)
    best_idx = None
    best_conf = 0.0
    best_overlap_len = 0

    for i, row in enumerate(df.iter_rows(named=True)):
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue

        # Normalize candidate text: strip trailing punctuation, lowercase
        raw_normalized = raw.lower().strip().rstrip(",.;:")

        for vi, variant in enumerate(variants):
            variant_normalized = variant.lower().strip().rstrip(",.;:")

            # Calculate overlap length for scoring
            if variant_normalized == raw_normalized:
                # Exact match - highest priority
                conf = 1.0 if vi == 0 else (0.9 if vi <= 4 else 0.8)
                overlap_len = len(raw_normalized)
            elif variant_normalized in raw_normalized:
                # Variant contained in candidate
                conf = 0.9 if vi == 0 else (0.8 if vi <= 4 else 0.6)
                overlap_len = len(variant_normalized)
            elif raw_normalized in variant_normalized:
                # Candidate contained in variant
                conf = 0.85 if vi == 0 else (0.75 if vi <= 4 else 0.55)
                overlap_len = len(raw_normalized)
            else:
                continue

            # Prefer matches with more overlap (e.g., "Nov 10" > "2023")
            # Update best match if: higher confidence OR same confidence but longer overlap
            if conf > best_conf or (
                conf == best_conf and overlap_len > best_overlap_len
            ):
                best_conf = conf
                best_idx = i
                best_overlap_len = overlap_len
                break

    return (best_idx, best_conf)


def _match_amount(df: pl.DataFrame, value: str) -> tuple[int | None, float]:
    """Match amount values with normalization (strip $, commas)."""
    norm_value = normalize.normalize_amount(value)[0] or value
    abs_value = norm_value.lstrip("-")
    best_idx = None
    best_conf = 0.0

    for i, row in enumerate(df.iter_rows(named=True)):
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        norm_raw = normalize.normalize_amount(raw)[0] or raw

        if norm_raw == norm_value:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = i
        elif norm_raw == abs_value:
            if 0.8 > best_conf:
                best_conf = 0.8
                best_idx = i
        elif len(norm_value) >= 3 and (norm_value in norm_raw or abs_value in norm_raw):
            if 0.6 > best_conf:
                best_conf = 0.6
                best_idx = i

    return (best_idx, best_conf)


# Translation table for stripping spaces, tabs, newlines, and hyphens from IDs
_ID_STRIP_TABLE = str.maketrans("", "", " \t\n\r-")


def _match_id(df: pl.DataFrame, value: str) -> tuple[int | None, float]:
    """Match ID values with normalization (strip hyphens, spaces)."""
    norm_value = value.translate(_ID_STRIP_TABLE).lower()
    best_idx = None
    best_conf = 0.0

    for i, row in enumerate(df.iter_rows(named=True)):
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        norm_raw = raw.translate(_ID_STRIP_TABLE).lower()

        if norm_raw == norm_value:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = i
        elif norm_value in norm_raw:
            if 0.8 > best_conf:
                best_conf = 0.8
                best_idx = i
        elif norm_raw in norm_value and len(norm_raw) >= 6:
            if 0.6 > best_conf:
                best_conf = 0.6
                best_idx = i

    return (best_idx, best_conf)


def _match_name(df: pl.DataFrame, value: str) -> tuple[int | None, float]:
    """Match name values with case-insensitive fuzzy matching."""
    value_lower = value.lower().strip()
    best_idx = None
    best_conf = 0.0

    for i, row in enumerate(df.iter_rows(named=True)):
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        raw_lower = raw.lower().strip()

        if raw_lower == value_lower:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = i
        elif value_lower in raw_lower:
            if 0.9 > best_conf:
                best_conf = 0.9
                best_idx = i
        elif raw_lower in value_lower and len(raw_lower) >= 3:
            if 0.7 > best_conf:
                best_conf = 0.7
                best_idx = i

    return (best_idx, best_conf)


def _match_exact(df: pl.DataFrame, value: str) -> tuple[int | None, float]:
    """Exact case-insensitive match."""
    value_lower = value.lower().strip()
    best_idx = None
    best_conf = 0.0

    for i, row in enumerate(df.iter_rows(named=True)):
        raw = str(row.get("raw_text", "")).lower().strip()
        if raw == value_lower:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = i
        elif value_lower in raw:
            if 0.7 > best_conf:
                best_conf = 0.7
                best_idx = i

    return (best_idx, best_conf)
