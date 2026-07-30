"""Shared feature preparation for training and inference.

This module is the SINGLE SOURCE OF TRUTH for converting raw candidate data
(dict or DataFrame) into the canonical feature vector defined by
features.feature_columns(). Both train.py and decoder.py MUST use these
functions to prevent train-serve skew.

Two entry points:
- prepare_candidate_features(dict) -> dict: single candidate, dict-based
- prepare_features_dataframe(DataFrame) -> DataFrame: batch, column-rename path
"""

from __future__ import annotations

from typing import Any

import polars as pl

from invoices.logging import get_logger

from .features import ANCHOR_TYPES, CandidateFeatures

logger = get_logger("invoices.feature_prep")

# Bucket name constants (relocated from PipelineConfig)
BUCKET_DATE_LIKE: str = "date_like"
BUCKET_AMOUNT_LIKE: str = "amount_like"
BUCKET_ID_LIKE: str = "id_like"
BUCKET_NAME_LIKE: str = "name_like"
BUCKET_KEYWORD_PROXIMAL: str = "keyword_proximal"
BUCKET_RANDOM_NEGATIVE: str = "random_negative"


def prepare_candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract the canonical feature vector from a single candidate dict.

    This is the authoritative feature extraction used by both training
    (iterating candidate rows) and inference (single-candidate scoring).
    Feature names and order match features.feature_columns() exactly.

    Args:
        candidate: Dict with bbox_norm_*, raw_text/text, bucket, page_idx,
                   and directional/relative position features.

    Returns:
        Dict mapping each feature name to its float value.
    """
    # Basic geometric features (5)
    if "bbox_norm_x0" in candidate:
        # Raw candidate: derive geometry from the normalized bbox.
        x0 = candidate["bbox_norm_x0"]
        y0 = candidate["bbox_norm_y0"]
        x1 = candidate["bbox_norm_x1"]
        y1 = candidate["bbox_norm_y1"]
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        width = x1 - x0
        height = y1 - y0
        area = width * height
    elif "center_x" in candidate:
        # Already feature-extracted (ranker/cost path): geometry was computed
        # upstream from the same bbox. Reuse it rather than re-deriving from an
        # absent bbox (which would zero it and degrade ranking).
        center_x = candidate.get("center_x", 0.0)
        center_y = candidate.get("center_y", 0.0)
        width = candidate.get("width", 0.0)
        height = candidate.get("height", 0.0)
        area = candidate.get("area", 0.0)
    else:
        # Neither raw bbox nor precomputed geometry — genuinely unrecoverable.
        logger.warning(
            "feature_geometry_missing",
            bucket=str(candidate.get("bucket", "?")),
            candidate_id=str(candidate.get("candidate_id", "?")),
            n_keys=len(candidate),
            keys=sorted(str(k) for k in candidate)[:40],
        )
        center_x = center_y = width = height = area = 0.0

    # Text features (4)
    text = str(candidate.get("raw_text") or candidate.get("text", ""))
    char_count = len(text)
    word_count = len(text.split())
    digit_count = sum(c.isdigit() for c in text)
    alpha_count = sum(c.isalpha() for c in text)

    # Page features (1)
    page_idx = candidate.get("page_idx", 0)

    # Bucket features (7) - one-hot encoding including name_like
    bucket = candidate.get("bucket", "other")
    all_known_buckets = [
        BUCKET_AMOUNT_LIKE,
        BUCKET_DATE_LIKE,
        BUCKET_ID_LIKE,
        BUCKET_NAME_LIKE,
        BUCKET_KEYWORD_PROXIMAL,
        BUCKET_RANDOM_NEGATIVE,
    ]
    bucket_features = {
        "bucket_amount_like": int(bucket == BUCKET_AMOUNT_LIKE),
        "bucket_date_like": int(bucket == BUCKET_DATE_LIKE),
        "bucket_id_like": int(bucket == BUCKET_ID_LIKE),
        "bucket_name_like": int(bucket == BUCKET_NAME_LIKE),
        "bucket_keyword_proximal": int(bucket == BUCKET_KEYWORD_PROXIMAL),
        "bucket_random_negative": int(bucket == BUCKET_RANDOM_NEGATIVE),
        "bucket_other": int(bucket not in all_known_buckets),
    }

    # Directional features (35) - 7 per anchor type x 5 anchor types
    directional_features: dict[str, float] = {}
    for anchor_type in ANCHOR_TYPES:
        directional_features[f"dx_to_{anchor_type}"] = candidate.get(
            f"dx_to_{anchor_type}", 1.0
        )
        directional_features[f"dy_to_{anchor_type}"] = candidate.get(
            f"dy_to_{anchor_type}", 1.0
        )
        directional_features[f"dist_to_{anchor_type}"] = candidate.get(
            f"dist_to_{anchor_type}", 1.414
        )
        directional_features[f"aligned_x_{anchor_type}"] = candidate.get(
            f"aligned_x_{anchor_type}", 0.0
        )
        directional_features[f"aligned_y_{anchor_type}"] = candidate.get(
            f"aligned_y_{anchor_type}", 0.0
        )
        directional_features[f"reading_order_{anchor_type}"] = candidate.get(
            f"reading_order_{anchor_type}", 0.0
        )
        directional_features[f"below_{anchor_type}"] = candidate.get(
            f"below_{anchor_type}", 0.0
        )

    # Anchor presence (5) - binary: was this anchor type found in doc?
    anchor_presence_features: dict[str, float] = {}
    for anchor_type in ANCHOR_TYPES:
        anchor_presence_features[f"has_{anchor_type}_anchor"] = float(
            candidate.get(f"has_{anchor_type}_anchor", 0.0)
        )

    # Relative position features (6)
    relative_position_features = {
        "y_from_bottom": candidate.get("y_from_bottom", 0.5),
        "in_top_half": candidate.get("in_top_half", 0.0),
        "in_bottom_quarter": candidate.get("in_bottom_quarter", 0.0),
        "in_top_quarter": candidate.get("in_top_quarter", 0.0),
        "in_right_third": candidate.get("in_right_third", 0.0),
        "in_amount_region": candidate.get("in_amount_region", 0.0),
    }

    # Page region features (3) - derived from center_y
    # header=top 20%, footer=bottom 15%, body=middle 65%
    in_header_region = float(center_y < 0.20)
    in_footer_region = float(center_y > 0.85)
    in_body_region = float(0.20 <= center_y <= 0.85)

    # Semantic features (3)
    occurrence_rank = float(candidate.get("occurrence_rank", 1.0))
    is_largest_amount_in_doc = float(candidate.get("is_largest_amount_in_doc", 0.0))
    # aligned_x: max of aligned_x_{anchor_type} across all anchors
    aligned_x_values = [
        float(candidate.get(f"aligned_x_{at}", 0.0)) for at in ANCHOR_TYPES
    ]
    aligned_x = max(aligned_x_values) if aligned_x_values else 0.0

    # Combine all features
    features: dict[str, float] = {
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "area": area,
        "char_count": char_count,
        "word_count": word_count,
        "digit_count": digit_count,
        "alpha_count": alpha_count,
        "page_idx": page_idx,
        **bucket_features,
        **directional_features,
        **anchor_presence_features,
        **relative_position_features,
        "in_header_region": in_header_region,
        "in_footer_region": in_footer_region,
        "in_body_region": in_body_region,
        "occurrence_rank": occurrence_rank,
        "is_largest_amount_in_doc": is_largest_amount_in_doc,
        "aligned_x": aligned_x,
    }

    CandidateFeatures(**features)  # validates dict matches dataclass contract
    return features


def prepare_features_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Convert a raw candidates DataFrame to the canonical feature matrix.

    Delegates to prepare_candidate_features() row-by-row so that all derived
    features (center_x/y, width, height, area, in_header_region, aligned_x,
    etc.) are computed identically to the dict path.  This eliminates the
    train-serve skew where the old implementation only zero-filled missing
    columns instead of computing them from bbox/text data.

    Args:
        df: Raw candidates DataFrame (from candidates.py or any source).

    Returns:
        DataFrame with columns matching features.feature_columns().
    """
    from invoices.features import feature_columns as _get_feature_columns

    feature_columns = _get_feature_columns()

    rows: list[dict[str, float]] = [
        prepare_candidate_features(row) for row in df.iter_rows(named=True)
    ]

    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(feature_columns, pl.Float64))

    return pl.DataFrame(rows, schema=dict.fromkeys(feature_columns, pl.Float64))
