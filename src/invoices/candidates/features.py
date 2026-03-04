"""Feature computation functions for candidates."""

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config
from ..geometry import compute_iou
from ..logging import get_logger
from .constants import (
    BASE_FEATURE_NAMES,
    CURRENCY_SYMBOLS,
    DIRECTIONAL_DEFAULTS,
    POSITION_FEATURE_SPECS,
)
from .spans import PageGrid

logger = get_logger(__name__)


def _build_directional_feature_names() -> list[str]:
    """Build directional feature names for all anchor types."""
    features = []
    for anchor_type in Config.ANCHOR_TYPES:
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
    df: pd.DataFrame,
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
            df[feature_name].fillna(default_val).values
        )
    else:
        features[:, feature_idx[feature_name]] = default_val


def compute_text_features_enhanced(text: str) -> dict[str, Any]:
    """Enhanced text features without regex."""
    text_clean = text.strip()

    if not text_clean:
        return {
            "text_length": 0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "currency_flag": False,
            "unigram_hash": "",
            "bigram_hash": "",
        }

    features: dict[str, Any] = {
        "text_length": len(text_clean),
        "digit_ratio": sum(1 for c in text_clean if c.isdigit()) / len(text_clean),
        "uppercase_ratio": sum(1 for c in text_clean if c.isupper()) / len(text_clean),
        "currency_flag": any(symbol in text_clean for symbol in CURRENCY_SYMBOLS),
    }

    # Hashed n-grams (safe string handling)
    words = text_clean.lower().split()

    # Unigram hashes (limit to prevent overflow)
    unigram_hashes = []
    for word in words[:3]:  # Limit to first 3
        hash_obj = hashlib.md5(
            word.encode("utf-8", errors="ignore"), usedforsecurity=False
        )
        unigram_hashes.append(hash_obj.hexdigest()[:8])
    features["unigram_hash"] = ",".join(unigram_hashes)

    # Bigram hashes
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(min(2, len(words) - 1))]
        bigram_hashes = []
        for bigram in bigrams:
            hash_obj = hashlib.md5(
                bigram.encode("utf-8", errors="ignore"), usedforsecurity=False
            )
            bigram_hashes.append(hash_obj.hexdigest()[:8])
        features["bigram_hash"] = ",".join(bigram_hashes)
    else:
        features["bigram_hash"] = ""

    return features


def compute_geometry_features_enhanced(
    bbox_norm: tuple[float, float, float, float], page_width: float, page_height: float
) -> dict[str, Any]:
    """Enhanced geometry features with relative positioning.

    Addresses the fundamental reality that footers float based on content.
    Uses both top-down (y) and bottom-up (y_from_bottom) coordinates.
    """
    x0, y0, x1, y1 = bbox_norm

    # Center and dimensions
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0

    # Bottom-up coordinate: critical for footer elements (Total, Tax, etc.)
    # On a 1-page invoice, Total might be at y=0.8
    # On a 2-page invoice, Total might still be at y_from_bottom=0.1
    y_from_bottom = 1.0 - y1  # Distance from bottom edge to element bottom

    # Quadrant indicators for coarse spatial reasoning
    in_top_half = cy < 0.5
    in_left_half = cx < 0.5
    in_bottom_quarter = cy > 0.75
    in_top_quarter = cy < 0.25
    in_right_third = cx > 0.67

    return {
        # Absolute position
        "center_x": cx,
        "center_y": cy,
        "width": w,
        "height": h,
        # Top-down positioning
        "distance_to_top": cy,
        "distance_to_bottom": 1.0 - cy,
        "distance_to_left": cx,
        "distance_to_right": 1.0 - cx,
        "distance_to_center": math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2),
        # Bottom-up positioning (critical for footers)
        "y_from_bottom": y_from_bottom,
        "y0_from_bottom": 1.0 - y0,  # Top edge distance from page bottom
        # Shape features
        "aspect_ratio": h / max(w, 0.001),
        "area": w * h,
        # Quadrant indicators
        "in_top_half": float(in_top_half),
        "in_left_half": float(in_left_half),
        "in_bottom_quarter": float(in_bottom_quarter),
        "in_top_quarter": float(in_top_quarter),
        "in_right_third": float(in_right_third),
        # Combined position indicators (common invoice patterns)
        "in_amount_region": float(cx > 0.5 and cy > 0.5),  # Bottom-right quadrant
    }


def compute_style_features_enhanced(
    font_size: float,
    is_bold: bool,
    is_italic: bool,
    font_hash: str,
    page_font_sizes: list[float],
) -> dict[str, Any]:
    """Enhanced style features."""
    # Font size z-score relative to page
    if page_font_sizes and len(page_font_sizes) > 1:
        page_mean = np.mean(page_font_sizes)
        page_std = np.std(page_font_sizes)
        font_size_z = (font_size - page_mean) / max(page_std, 1e-6)
    else:
        font_size_z = 0.0

    return {
        "font_size": font_size,
        "font_size_z": font_size_z,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "font_hash": str(font_hash),  # Ensure string for safety
        "font_size_large": font_size_z > 1.0,
        "font_size_small": font_size_z < -1.0,
    }


def compute_local_density_grid(
    bbox_norm: tuple[float, float, float, float],
    page_grid: PageGrid,
    window_size: float = 0.1,
) -> float:
    """Compute local density using grid neighbors."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # Get neighbors from grid
    neighbors = page_grid.get_neighbors(cx, cy, radius=2)

    # Count neighbors within window
    count = 0
    for neighbor in neighbors:
        if neighbor is None:
            continue

        if (
            isinstance(neighbor, dict)
            and "center_x" in neighbor
            and "center_y" in neighbor
        ):
            nx, ny = neighbor["center_x"], neighbor["center_y"]
            if abs(nx - cx) <= window_size / 2 and abs(ny - cy) <= window_size / 2:
                count += 1

    return count / max(window_size * window_size, 0.01)


def compute_section_prior(
    bbox_norm: tuple[float, float, float, float], page_idx: int
) -> float:
    """Compute section-based prior weights."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    prior = 0.0

    # Early-page boost: invoice metadata clusters on the first few pages
    if page_idx <= Config.EARLY_PAGE_MAX_IDX:
        decay = (Config.EARLY_PAGE_MAX_IDX + 1 - page_idx) / (
            Config.EARLY_PAGE_MAX_IDX + 1
        )
        prior += Config.EARLY_PAGE_BOOST * decay

    # Top-right corner of first page (common for amounts)
    if page_idx == 0 and cx > 0.6 and cy < 0.3:
        prior += 0.1

    # Bottom totals band (any page)
    if cy > 0.8:
        prior += 0.05

    # Header band (top 20%)
    if cy < 0.2:
        prior += 0.03

    return prior


# Re-export compute_iou from geometry.py as compute_overlap_iou for backward compat
compute_overlap_iou = compute_iou


def apply_soft_nms_grid(
    candidates: list[dict[str, Any]], page_grid: PageGrid, lambda_param: float = 0.5
) -> list[dict[str, Any]]:
    """Apply soft non-maximum suppression using grid neighbors."""
    if not candidates:
        return []

    # Sort by score (descending)
    candidates_sorted = sorted(
        candidates, key=lambda x: x.get("total_score", 0.0), reverse=True
    )

    # Apply soft decay to overlapping candidates
    for _i, candidate in enumerate(candidates_sorted):
        bbox = candidate["bbox_norm"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Get higher-scored neighbors
        neighbors = page_grid.get_neighbors(cx, cy, radius=1)

        for neighbor in neighbors:
            if neighbor is candidate:
                continue

            # Only consider higher-scored neighbors
            neighbor_score = neighbor.get("total_score", 0.0)
            if neighbor_score <= candidate.get("total_score", 0.0):
                continue

            # Compute IoU
            iou = compute_iou(candidate["bbox_norm"], neighbor["bbox_norm"])

            # Apply soft decay
            if iou > 0.1:  # Only apply if there's meaningful overlap
                decay = math.exp(-lambda_param * iou)
                candidate["total_score"] = candidate.get("total_score", 0.0) * decay

    return candidates_sorted


def diversity_sampling(
    candidates: list[dict[str, Any]], max_candidates: int = 200
) -> list[dict[str, Any]]:
    """Apply diversity sampling with type-based stratification."""
    from collections import defaultdict

    from .patterns import (
        is_amount_like_soft,
        is_date_like_soft,
        is_id_like_soft,
        is_name_like_soft,
    )

    if len(candidates) <= max_candidates:
        return candidates

    # Phase 0.5: Reserve slots per type for early-page candidates (pages 0-2).
    # Multi-page invoices (e.g., 149 pages) have summary info on the first
    # few pages. Without type-aware reservation, critical amounts/IDs from
    # summary pages get drowned out by hundreds of candidates from later pages.
    #
    # Strategy: collect early-page candidates by type, sort by score DESC,
    # then pick the best per type. This ensures high-value candidates like
    # "AT&T" or "-$6,035.39" survive even when competing with many peers.
    max_reserved = max(1, max_candidates // 5)  # Reserve up to 20% of slots
    per_type_limit = max(5, max_reserved // 3)

    # Collect early-page candidates by type
    early_by_type: dict[str, list[dict[str, Any]]] = {
        "amount": [],
        "id": [],
        "date": [],
        "name": [],
    }
    non_early: list[dict[str, Any]] = []

    for candidate in candidates:
        text = candidate.get("raw_text", "").strip()
        page_idx = candidate.get("page_idx", 99)

        if page_idx <= 2:
            ctype = None
            if is_amount_like_soft(text):
                ctype = "amount"
            elif is_id_like_soft(text):
                ctype = "id"
            elif is_date_like_soft(text):
                ctype = "date"
            elif is_name_like_soft(text):
                ctype = "name"

            if ctype:
                early_by_type[ctype].append(candidate)
                continue
        non_early.append(candidate)

    # Sort each type by score (descending) and pick top UNIQUE candidates.
    # Text-level dedup within each type ensures diverse candidates survive
    # (e.g., "AT&T" doesn't lose to 10 copies of "STATEMENT").
    reserved: list[dict[str, Any]] = []
    remaining_candidates: list[dict[str, Any]] = list(non_early)

    for ctype, early_cands in early_by_type.items():
        # For names, use page_frequency as a sorting boost since names
        # that repeat across pages (like vendor names) are more important
        # than one-off words that happen to score high.
        if ctype == "name":
            early_cands.sort(
                key=lambda x: (
                    -(x.get("total_score", 0.0) + x.get("page_frequency", 0.0) * 3.0)
                )
            )
        else:
            early_cands.sort(key=lambda x: -x.get("total_score", 0.0))
        seen_texts: set[str] = set()
        reserved_for_type: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for c in early_cands:
            text_key = c.get("normalized_text", c.get("raw_text", "")).strip().lower()
            if text_key not in seen_texts and len(reserved_for_type) < per_type_limit:
                seen_texts.add(text_key)
                reserved_for_type.append(c)
            else:
                overflow.append(c)
        reserved.extend(reserved_for_type)
        remaining_candidates.extend(overflow)

    # Adjust quota for the main sampling to account for reserved slots
    adjusted_max = max_candidates - len(reserved)
    candidates = remaining_candidates

    # Group by type shape characteristics
    type_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        # Determine type shape based on content and features
        text = candidate.get("raw_text", "").strip()

        if is_date_like_soft(text):
            # Group by approximate date format
            if "/" in text or "-" in text:
                type_key = "date_numeric"
            else:
                type_key = "date_text"
        elif is_amount_like_soft(text):
            # Group by magnitude (rough bins)
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count <= 2:
                type_key = "amount_small"
            elif digit_count <= 4:
                type_key = "amount_medium"
            else:
                type_key = "amount_large"
        elif is_id_like_soft(text):
            # Group by alphanumeric shape
            has_letters = any(c.isalpha() for c in text)
            has_separators = any(c in "-_#" for c in text)
            if has_letters and has_separators:
                type_key = "id_complex"
            elif has_letters:
                type_key = "id_alphanum"
            else:
                type_key = "id_numeric"
        elif is_name_like_soft(text):
            # Group name-like candidates (company/person names)
            # Subgroup by length to preserve both short (AT&T) and long names
            if len(text) <= 10:
                type_key = "name_short"
            else:
                type_key = "name_long"
        else:
            type_key = "other"

        type_groups[type_key].append(candidate)

    # Phase 1: Guarantee small groups (<=5 members) get ALL their candidates.
    # Rare type groups are likely high-value (e.g., 2 currency candidates
    # shouldn't compete with 80 amount candidates for proportional quota).
    result: list[dict[str, Any]] = []
    large_groups: dict[str, list[dict[str, Any]]] = {}

    if len(type_groups) == 0:
        return reserved + candidates[:adjusted_max]

    for group_key, group_candidates in type_groups.items():
        group_candidates.sort(
            key=lambda x: (-x.get("total_score", 0.0), x.get("page_idx", 0))
        )
        if len(group_candidates) <= 5:
            result.extend(group_candidates)
        else:
            large_groups[group_key] = group_candidates

    # Phase 2: Fill remaining quota proportionally from large groups
    remaining = adjusted_max - len(result)
    total_large = len(large_groups)

    if remaining > 0 and total_large > 0:
        for _group_key, group_candidates in large_groups.items():
            group_quota = max(1, remaining // total_large)
            slots_left = adjusted_max - len(result)
            actual_quota = min(group_quota, len(group_candidates), slots_left)
            result.extend(group_candidates[:actual_quota])

            if len(result) >= adjusted_max:
                break

    return reserved + result[:adjusted_max]


def extract_features_vectorized(
    candidates_df: pd.DataFrame,
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
            candidates_df["bbox_norm_x0"].values + candidates_df["bbox_norm_x1"].values
        ) / 2
        features[:, feature_idx["center_x"]] = center_x

    if "center_y" in feature_idx:
        center_y = (
            candidates_df["bbox_norm_y0"].values + candidates_df["bbox_norm_y1"].values
        ) / 2
        features[:, feature_idx["center_y"]] = center_y

    if "width" in feature_idx:
        width = (
            candidates_df["bbox_norm_x1"].values - candidates_df["bbox_norm_x0"].values
        )
        features[:, feature_idx["width"]] = width

    if "height" in feature_idx:
        height = (
            candidates_df["bbox_norm_y1"].values - candidates_df["bbox_norm_y0"].values
        )
        features[:, feature_idx["height"]] = height

    if "area" in feature_idx:
        width = (
            candidates_df["bbox_norm_x1"].values - candidates_df["bbox_norm_x0"].values
        )
        height = (
            candidates_df["bbox_norm_y1"].values - candidates_df["bbox_norm_y0"].values
        )
        features[:, feature_idx["area"]] = width * height

    # === Text features (vectorized using apply with numpy conversion) ===
    if "char_count" in feature_idx:
        features[:, feature_idx["char_count"]] = (
            candidates_df["raw_text"].fillna("").str.len().values
        )

    if "word_count" in feature_idx:
        features[:, feature_idx["word_count"]] = (
            candidates_df["raw_text"].fillna("").str.split().str.len().fillna(0).values
        )

    if "digit_count" in feature_idx:
        features[:, feature_idx["digit_count"]] = (
            candidates_df["raw_text"].fillna("").str.count(r"\d").values
        )

    if "alpha_count" in feature_idx:
        features[:, feature_idx["alpha_count"]] = (
            candidates_df["raw_text"].fillna("").str.count(r"[a-zA-Z]").values
        )

    # === Page features ===
    if "page_idx" in feature_idx:
        features[:, feature_idx["page_idx"]] = candidates_df.get(
            "page_idx", pd.Series([0] * n_candidates)
        ).values

    # === Bucket features (one-hot, vectorized) ===
    bucket_col = candidates_df.get("bucket", pd.Series(["other"] * n_candidates))

    bucket_mapping = {
        "bucket_amount_like": Config.BUCKET_AMOUNT_LIKE,
        "bucket_date_like": Config.BUCKET_DATE_LIKE,
        "bucket_id_like": Config.BUCKET_ID_LIKE,
        "bucket_name_like": Config.BUCKET_NAME_LIKE,
        "bucket_keyword_proximal": Config.BUCKET_KEYWORD_PROXIMAL,
        "bucket_random_negative": Config.BUCKET_RANDOM_NEGATIVE,
    }

    for feature_name, bucket_value in bucket_mapping.items():
        if feature_name in feature_idx:
            features[:, feature_idx[feature_name]] = (
                bucket_col.values == bucket_value
            ).astype(np.float64)

    if "bucket_other" in feature_idx:
        known_buckets = set(bucket_mapping.values())
        # Use vectorized .isin() instead of list comprehension for better performance
        features[:, feature_idx["bucket_other"]] = (
            ~bucket_col.isin(known_buckets)
        ).values.astype(np.float64)

    # === Directional vector features (from typed anchors, using centralized specs) ===
    for anchor_type in Config.ANCHOR_TYPES:
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
            candidates_df["bbox_norm_y0"].values + candidates_df["bbox_norm_y1"].values
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
        aligned_x_cols = [
            f"aligned_x_{anchor_type}" for anchor_type in Config.ANCHOR_TYPES
        ]
        present_cols = [c for c in aligned_x_cols if c in candidates_df.columns]
        if present_cols:
            features[:, feature_idx["aligned_x"]] = (
                candidates_df[present_cols].max(axis=1).fillna(0.0).values
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


def prune_candidates(
    candidates_df: pd.DataFrame,
    model: Any,
    threshold: float | None = None,
    feature_names: list[str] | None = None,
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """
    Prune low-probability candidates using global XGBoost inference.

    This function addresses the O(n^3) Hungarian decoder scaling issue by
    removing unlikely candidates BEFORE the decoder runs. Uses vectorized
    Pandas/NumPy operations for efficiency.

    Args:
        candidates_df: DataFrame of all candidates
        model: Trained XGBoost model with predict_proba()
        threshold: Discard candidates with max probability < threshold.
                   If None, uses Config.PRUNING_THRESHOLD.
        feature_names: Feature names expected by model. If None, uses
                       model's get_booster().feature_names if available.
        max_candidates: Maximum candidates to keep (hard cap).
                        If None, uses Config.PRUNING_MAX_CANDIDATES.

    Returns:
        Pruned DataFrame with only candidates above threshold,
        limited to max_candidates (sorted by probability descending).

    Notes:
        - Uses vectorized operations (NO Python loops over rows)
        - Maintains determinism (model should already have n_jobs=1)
        - Handles missing features gracefully (default to 0.0)
        - Logs pruning statistics for observability
    """
    if threshold is None:
        threshold = Config.PRUNING_THRESHOLD

    if max_candidates is None:
        max_candidates = Config.PRUNING_MAX_CANDIDATES

    original_count = len(candidates_df)

    # Early exit if no candidates or already within limits
    if candidates_df.empty:
        logger.debug("pruning_no_candidates")
        return candidates_df

    # Guard against negative PRUNING_MIN_TRIGGER (treat as 0)
    pruning_min_trigger = max(0, Config.PRUNING_MIN_TRIGGER)
    if Config.PRUNING_MIN_TRIGGER < 0:
        logger.warning(
            "pruning_min_trigger_negative",
            value=Config.PRUNING_MIN_TRIGGER,
        )

    if original_count <= pruning_min_trigger:
        logger.debug(
            "pruning_skipped",
            candidate_count=original_count,
            trigger_threshold=Config.PRUNING_MIN_TRIGGER,
        )
        return candidates_df

    # Get feature names from model if not provided
    if feature_names is None:
        try:
            booster = model.get_booster()
            feature_names = booster.feature_names
            if feature_names is None:
                # Fallback to default feature names matching training
                feature_names = _get_default_feature_names()
        except (AttributeError, TypeError):
            feature_names = _get_default_feature_names()

    # Extract features using vectorized operations
    try:
        feature_matrix = extract_features_vectorized(candidates_df, feature_names)
    except Exception as e:
        logger.warning("pruning_feature_extraction_failed", error=str(e))
        return candidates_df

    # Get model predictions (probabilities)
    try:
        # predict_proba returns shape (n_samples, 2) for binary classification
        # [:, 1] gives probability of positive class
        probabilities = model.predict_proba(feature_matrix)[:, 1]
    except Exception as e:
        logger.warning("pruning_model_prediction_failed", error=str(e))
        return candidates_df

    # Add probability column to DataFrame (use unique name to avoid collisions)
    _PRUNE_PROB_COL = "__invoicex_prune_prob__"
    pruned_df = candidates_df.copy()
    pruned_df[_PRUNE_PROB_COL] = probabilities

    # Filter by threshold
    pruned_df = pruned_df[pruned_df[_PRUNE_PROB_COL] >= threshold]

    # Sort by probability descending with deterministic secondary sort key
    # Using candidate_id as tiebreaker ensures stable ordering for reproducibility
    pruned_df = pruned_df.sort_values(
        [_PRUNE_PROB_COL, "candidate_id"],
        ascending=[False, True],
    )
    if len(pruned_df) > max_candidates:
        pruned_df = pruned_df.head(max_candidates)

    # Remove temporary probability column
    pruned_df = pruned_df.drop(columns=[_PRUNE_PROB_COL])

    # Reset index for clean DataFrame
    pruned_df = pruned_df.reset_index(drop=True)

    final_count = len(pruned_df)

    # Log pruning statistics
    logger.info(
        "candidates_pruned",
        original=original_count,
        final=final_count,
        threshold=round(threshold, 3),
        max_candidates=max_candidates,
    )

    return pruned_df
