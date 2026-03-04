"""Shared feature preparation for training and inference.

This module is the SINGLE SOURCE OF TRUTH for converting raw candidate data
(dict or DataFrame) into the canonical feature vector defined by
Config.get_feature_columns(). Both train.py and decoder.py MUST use these
functions to prevent train-serve skew.

Two entry points:
- prepare_candidate_features(dict) -> dict: single candidate, dict-based
- prepare_features_dataframe(DataFrame) -> DataFrame: batch, column-rename path
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .config import Config


def prepare_candidate_features(candidate: dict[str, Any]) -> dict[str, float]:
    """Extract the canonical feature vector from a single candidate dict.

    This is the authoritative feature extraction used by both training
    (iterating candidate rows) and inference (single-candidate scoring).
    Feature names and order match Config.get_feature_columns() exactly.

    Args:
        candidate: Dict with bbox_norm_*, raw_text/text, bucket, page_idx,
                   and directional/relative position features.

    Returns:
        Dict mapping each feature name to its float value.
    """
    # Basic geometric features (5)
    center_x = (candidate["bbox_norm_x0"] + candidate["bbox_norm_x1"]) / 2
    center_y = (candidate["bbox_norm_y0"] + candidate["bbox_norm_y1"]) / 2
    width = candidate["bbox_norm_x1"] - candidate["bbox_norm_x0"]
    height = candidate["bbox_norm_y1"] - candidate["bbox_norm_y0"]
    area = width * height

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
        Config.BUCKET_AMOUNT_LIKE,
        Config.BUCKET_DATE_LIKE,
        Config.BUCKET_ID_LIKE,
        Config.BUCKET_NAME_LIKE,
        Config.BUCKET_KEYWORD_PROXIMAL,
        Config.BUCKET_RANDOM_NEGATIVE,
    ]
    bucket_features = {
        "bucket_amount_like": int(bucket == Config.BUCKET_AMOUNT_LIKE),
        "bucket_date_like": int(bucket == Config.BUCKET_DATE_LIKE),
        "bucket_id_like": int(bucket == Config.BUCKET_ID_LIKE),
        "bucket_name_like": int(bucket == Config.BUCKET_NAME_LIKE),
        "bucket_keyword_proximal": int(bucket == Config.BUCKET_KEYWORD_PROXIMAL),
        "bucket_random_negative": int(bucket == Config.BUCKET_RANDOM_NEGATIVE),
        "bucket_other": int(bucket not in all_known_buckets),
    }

    # Directional features (35) - 7 per anchor type x 5 anchor types
    directional_features: dict[str, float] = {}
    for anchor_type in Config.ANCHOR_TYPES:
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
        float(candidate.get(f"aligned_x_{at}", 0.0)) for at in Config.ANCHOR_TYPES
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
        **relative_position_features,
        "in_header_region": in_header_region,
        "in_footer_region": in_footer_region,
        "in_body_region": in_body_region,
        "occurrence_rank": occurrence_rank,
        "is_largest_amount_in_doc": is_largest_amount_in_doc,
        "aligned_x": aligned_x,
    }

    return features


def prepare_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a raw candidates DataFrame to the canonical feature matrix.

    Handles column name mismatches from candidates.py:
    - Ensures geometry columns (width, height, area) are present
    - Computes text features from raw_text if missing
    - One-hot encodes bucket string -> 7 binary columns
    - Fills missing columns with 0.0
    - Returns only the canonical columns in Config order

    Args:
        df: Raw candidates DataFrame (from candidates.py or any source).

    Returns:
        DataFrame with columns matching Config.get_feature_columns().
    """
    feature_columns = Config.get_feature_columns()
    result = df.copy()

    # 1. Compute text features from raw_text if missing
    text_col = result.get("raw_text", result.get("text", pd.Series([""] * len(result))))
    text_col = text_col.astype(str)

    if "char_count" not in result.columns:
        result["char_count"] = text_col.str.len()
    if "word_count" not in result.columns:
        result["word_count"] = text_col.str.split().str.len()
    if "digit_count" not in result.columns:
        result["digit_count"] = text_col.apply(lambda t: sum(c.isdigit() for c in t))
    if "alpha_count" not in result.columns:
        result["alpha_count"] = text_col.apply(lambda t: sum(c.isalpha() for c in t))

    # 2. One-hot encode bucket if it's a string column
    if "bucket" in result.columns and "bucket_amount_like" not in result.columns:
        bucket_col = result["bucket"]
        all_known_buckets = [
            Config.BUCKET_AMOUNT_LIKE,
            Config.BUCKET_DATE_LIKE,
            Config.BUCKET_ID_LIKE,
            Config.BUCKET_NAME_LIKE,
            Config.BUCKET_KEYWORD_PROXIMAL,
            Config.BUCKET_RANDOM_NEGATIVE,
        ]
        result["bucket_amount_like"] = (bucket_col == Config.BUCKET_AMOUNT_LIKE).astype(
            int
        )
        result["bucket_date_like"] = (bucket_col == Config.BUCKET_DATE_LIKE).astype(int)
        result["bucket_id_like"] = (bucket_col == Config.BUCKET_ID_LIKE).astype(int)
        result["bucket_name_like"] = (bucket_col == Config.BUCKET_NAME_LIKE).astype(int)
        result["bucket_keyword_proximal"] = (
            bucket_col == Config.BUCKET_KEYWORD_PROXIMAL
        ).astype(int)
        result["bucket_random_negative"] = (
            bucket_col == Config.BUCKET_RANDOM_NEGATIVE
        ).astype(int)
        result["bucket_other"] = (~bucket_col.isin(all_known_buckets)).astype(int)

    # 3. Fill missing columns with 0.0
    for col in feature_columns:
        if col not in result.columns:
            result[col] = 0.0

    # 4. Return only the canonical columns in order
    out: pd.DataFrame = result[feature_columns].astype(np.float64)  # type: ignore[assignment]
    return out
