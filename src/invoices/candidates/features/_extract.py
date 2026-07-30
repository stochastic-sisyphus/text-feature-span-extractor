"""Vectorized feature matrix extraction from candidates DataFrame."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from ...features import ANCHOR_TYPES
from ..constants import (
    BASE_FEATURE_NAMES,
    BUCKET_AMOUNT_LIKE,
    BUCKET_DATE_LIKE,
    BUCKET_ID_LIKE,
    BUCKET_KEYWORD_PROXIMAL,
    BUCKET_NAME_LIKE,
    BUCKET_RANDOM_NEGATIVE,
    DIRECTIONAL_DEFAULTS,
    POSITION_FEATURE_SPECS,
)

if TYPE_CHECKING:
    pass


def _build_directional_feature_names() -> list[str]:
    """Build directional feature names for all anchor types."""
    features = []
    for anchor_type in ANCHOR_TYPES:
        features.extend(
            [
                f"dx_to_{anchor_type}",
                f"dy_to_{anchor_type}",
                f"dist_to_{anchor_type}",
                f"aligned_x_{anchor_type}",
                f"aligned_y_{anchor_type}",
                f"reading_order_{anchor_type}",
                f"below_{anchor_type}",
            ]
        )
    return features


def _get_directional_defaults_for_anchor(anchor_type: str) -> dict[str, float]:
    """Get directional feature defaults for a specific anchor type."""
    return {
        f"dx_to_{anchor_type}": DIRECTIONAL_DEFAULTS["dx"],
        f"dy_to_{anchor_type}": DIRECTIONAL_DEFAULTS["dy"],
        f"dist_to_{anchor_type}": DIRECTIONAL_DEFAULTS["dist"],
        f"aligned_x_{anchor_type}": DIRECTIONAL_DEFAULTS["aligned_x"],
        f"aligned_y_{anchor_type}": DIRECTIONAL_DEFAULTS["aligned_y"],
        f"reading_order_{anchor_type}": DIRECTIONAL_DEFAULTS["reading_order"],
        f"below_{anchor_type}": DIRECTIONAL_DEFAULTS["below"],
    }


def _fill_scalar_feature(
    df: pl.DataFrame,
    features: np.ndarray,
    feature_idx: dict[str, int],
    feature_name: str,
    default_val: float,
) -> None:
    """
    Fill a scalar feature column in the feature matrix.

    Args:
        df: Source DataFrame
        features: Target feature matrix (modified in place)
        feature_idx: Feature name to column index mapping
        feature_name: Name of the feature
        default_val: Default value if column missing
    """
    if feature_name not in feature_idx:
        return

    if feature_name in df.columns:
        features[:, feature_idx[feature_name]] = (
            df[feature_name].fill_null(default_val).to_numpy()
        )
    else:
        features[:, feature_idx[feature_name]] = default_val


def extract_features_vectorized(
    candidates_df: pl.DataFrame,
    feature_names: list[str],
) -> np.ndarray:
    """
    Extract feature matrix from candidates DataFrame using vectorized operations.

    This function creates a feature matrix compatible with XGBoost models,
    using only NumPy/Pandas vectorized operations (NO Python loops over rows).

    Args:
        candidates_df: DataFrame of candidates
        feature_names: List of feature names in model order

    Returns:
        NumPy array of shape (n_candidates, n_features)
    """
    n_candidates = len(candidates_df)
    n_features = len(feature_names)

    # Pre-allocate feature matrix
    features = np.zeros((n_candidates, n_features), dtype=np.float64)

    # Build a mapping of feature_name -> column_index for efficient lookup
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    # === Geometric features (vectorized) ===
    if "center_x" in feature_idx:
        center_x = (
            candidates_df["bbox_norm_x0"].to_numpy()
            + candidates_df["bbox_norm_x1"].to_numpy()
        ) / 2
        features[:, feature_idx["center_x"]] = center_x

    if "center_y" in feature_idx:
        center_y = (
            candidates_df["bbox_norm_y0"].to_numpy()
            + candidates_df["bbox_norm_y1"].to_numpy()
        ) / 2
        features[:, feature_idx["center_y"]] = center_y

    if "width" in feature_idx:
        width = (
            candidates_df["bbox_norm_x1"].to_numpy()
            - candidates_df["bbox_norm_x0"].to_numpy()
        )
        features[:, feature_idx["width"]] = width

    if "height" in feature_idx:
        height = (
            candidates_df["bbox_norm_y1"].to_numpy()
            - candidates_df["bbox_norm_y0"].to_numpy()
        )
        features[:, feature_idx["height"]] = height

    if "area" in feature_idx:
        width = (
            candidates_df["bbox_norm_x1"].to_numpy()
            - candidates_df["bbox_norm_x0"].to_numpy()
        )
        height = (
            candidates_df["bbox_norm_y1"].to_numpy()
            - candidates_df["bbox_norm_y0"].to_numpy()
        )
        features[:, feature_idx["area"]] = width * height

    # === Text features (vectorized using polars string operations) ===
    if "char_count" in feature_idx:
        features[:, feature_idx["char_count"]] = (
            candidates_df["raw_text"].fill_null("").str.len_chars().to_numpy()
        )

    if "word_count" in feature_idx:
        features[:, feature_idx["word_count"]] = (
            candidates_df["raw_text"]
            .fill_null("")
            .str.count_matches(r"\S+")
            .fill_null(0)
            .to_numpy()
        )

    if "digit_count" in feature_idx:
        features[:, feature_idx["digit_count"]] = (
            candidates_df["raw_text"].fill_null("").str.count_matches(r"\d").to_numpy()
        )

    if "alpha_count" in feature_idx:
        features[:, feature_idx["alpha_count"]] = (
            candidates_df["raw_text"]
            .fill_null("")
            .str.count_matches(r"[a-zA-Z]")
            .to_numpy()
        )

    # === Page features ===
    if "page_idx" in feature_idx:
        if "page_idx" in candidates_df.columns:
            features[:, feature_idx["page_idx"]] = candidates_df["page_idx"].to_numpy()
        # else: stays 0.0 from pre-allocation

    # === Bucket features (one-hot, vectorized) ===
    if "bucket" in candidates_df.columns:
        bucket_col = candidates_df["bucket"]
    else:
        bucket_col = pl.Series(["other"] * n_candidates)

    bucket_mapping = {
        "bucket_amount_like": BUCKET_AMOUNT_LIKE,
        "bucket_date_like": BUCKET_DATE_LIKE,
        "bucket_id_like": BUCKET_ID_LIKE,
        "bucket_name_like": BUCKET_NAME_LIKE,
        "bucket_keyword_proximal": BUCKET_KEYWORD_PROXIMAL,
        "bucket_random_negative": BUCKET_RANDOM_NEGATIVE,
    }

    for feature_name, bucket_value in bucket_mapping.items():
        if feature_name in feature_idx:
            features[:, feature_idx[feature_name]] = (
                (bucket_col == bucket_value).to_numpy().astype(np.float64)
            )

    if "bucket_other" in feature_idx:
        known_buckets = list(bucket_mapping.values())
        features[:, feature_idx["bucket_other"]] = (
            (~bucket_col.is_in(known_buckets)).to_numpy().astype(np.float64)
        )

    # === Directional vector features (from typed anchors, using centralized specs) ===
    for anchor_type in ANCHOR_TYPES:
        # Get defaults from centralized function
        defaults = _get_directional_defaults_for_anchor(anchor_type)

        for feat_name, default_val in defaults.items():
            _fill_scalar_feature(
                candidates_df, features, feature_idx, feat_name, default_val
            )

    # === Relative position features (using centralized specs) ===
    for feat_name, default_val in POSITION_FEATURE_SPECS.items():
        _fill_scalar_feature(
            candidates_df, features, feature_idx, feat_name, default_val
        )

    # === Page region one-hot (header / footer / body) ===
    # Derived from center_y: header=top 20%, footer=bottom 15%, body=middle 65%
    if any(
        f in feature_idx
        for f in ("in_header_region", "in_footer_region", "in_body_region")
    ):
        cy = (
            candidates_df["bbox_norm_y0"].to_numpy()
            + candidates_df["bbox_norm_y1"].to_numpy()
        ) / 2
        if "in_header_region" in feature_idx:
            features[:, feature_idx["in_header_region"]] = (cy < 0.20).astype(
                np.float64
            )
        if "in_footer_region" in feature_idx:
            features[:, feature_idx["in_footer_region"]] = (cy > 0.85).astype(
                np.float64
            )
        if "in_body_region" in feature_idx:
            features[:, feature_idx["in_body_region"]] = (
                (cy >= 0.20) & (cy <= 0.85)
            ).astype(np.float64)

    # === Occurrence rank (reading-order rank among same-text candidates) ===
    _fill_scalar_feature(candidates_df, features, feature_idx, "occurrence_rank", 1.0)

    # === Is largest amount in document ===
    _fill_scalar_feature(
        candidates_df, features, feature_idx, "is_largest_amount_in_doc", 0.0
    )

    # === Column alignment with any typed anchor ===
    # Binary: is this candidate in the same x-column as any keyword anchor?
    # Derived as max(aligned_x_{anchor_type}) across all anchor types.
    if "aligned_x" in feature_idx:
        aligned_x_cols = [f"aligned_x_{anchor_type}" for anchor_type in ANCHOR_TYPES]
        present_cols = [c for c in aligned_x_cols if c in candidates_df.columns]
        if present_cols:
            features[:, feature_idx["aligned_x"]] = (
                candidates_df.select(
                    pl.max_horizontal(*[pl.col(c) for c in present_cols]).fill_null(0.0)
                )
                .to_series()
                .to_numpy()
            )
        # else: stays 0.0 from pre-allocation

    return features


def _get_default_feature_names() -> list[str]:
    """
    Get default feature names matching the training feature set.

    This ensures compatibility when feature names are not available
    from the model directly.

    Uses centralized constants (BASE_FEATURE_NAMES, POSITION_FEATURE_SPECS)
    to avoid duplication with extract_features_vectorized().
    """
    # Use centralized base features
    base_features = list(BASE_FEATURE_NAMES)

    # Add directional features using centralized builder
    directional_features = _build_directional_feature_names()

    # Add relative position features from centralized specs
    position_features = list(POSITION_FEATURE_SPECS.keys())

    return base_features + directional_features + position_features
