"""Decoder module for Hungarian assignment with NONE option per field."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

from . import candidates, ingest, paths, train, utils
from . import schema_registry as registry
from .config import Config, DWeights
from .exceptions import DecodingError, ModelLoadError, ModelNotFoundError
from .feature_prep import prepare_candidate_features
from .logging import get_logger
from .metrics import pipeline_duration, pipeline_errors
from .patterns import (
    validate_amount,
    validate_date,
    validate_invoice_number,
    validate_name,
)


def compute_text_pattern_bonus(
    field: str,
    candidate: dict[str, Any],
    profile: FieldProfile | None = None,
    document_labels: set[str] | None = None,
    cross_page_headers: set[str] | None = None,
    address_city_tokens: set[str] | None = None,
) -> float:
    """
    Compute text pattern validation bonus/penalty for field-candidate pair.

    This function validates that the candidate's text actually matches the
    expected pattern for the field type. This addresses issues where:
    - "2023" (bare year) was selected for InvoiceNumber
    - Random word salad was selected for VendorName/TotalAmount

    Args:
        field: Field name (e.g., "InvoiceNumber", "TotalAmount")
        candidate: Candidate dictionary with raw_text or text
        profile: Optional pre-built FieldProfile to avoid registry lookups
        document_labels: Optional set of structural label tokens for this document
        cross_page_headers: Optional set of cross-page header tokens (3+ pages)
        address_city_tokens: Optional set of address city tokens (near ZIP codes)

    Returns:
        Bonus (positive = good match) or penalty (negative = bad match)
    """
    # Get candidate text
    text = str(candidate.get("raw_text") or candidate.get("text", ""))
    if not text.strip():
        return DWeights.EMPTY_TEXT_PENALTY

    # Get field type from profile (cached) or schema registry (fallback)
    field_type = (
        profile.field_type if profile is not None else registry.field_type(field)
    )

    # Apply type-specific validation
    if field_type == "id":
        # Field-aware validation: INV* prefix is ONLY valid for fields whose
        # format_hint matches it. Penalize other id fields to prevent stealing.
        bonus = validate_invoice_number(text)
        fmt_hint = (
            profile.field_def.get("format_hint")
            if profile is not None
            else registry.field_def(field).get("format_hint")
        )
        text_upper = text.strip().upper()
        if text_upper.startswith(("INV", "INVOICE")):
            # If this field has no format_hint matching the INV prefix, penalize
            if not fmt_hint or not re.search(r"(?i)\binv", fmt_hint):
                bonus = DWeights.INV_PREFIX_WRONG_FIELD_PENALTY
        return bonus
    elif field_type == "amount":
        return validate_amount(text)
    elif field_type == "date":
        return validate_date(text)
    elif field_type == "text":
        return validate_name(
            text,
            document_labels=document_labels,
            cross_page_headers=cross_page_headers,
            address_city_tokens=address_city_tokens,
        )
    elif field_type == "address":
        return validate_name(
            text,
            document_labels=document_labels,
            cross_page_headers=cross_page_headers,
            address_city_tokens=address_city_tokens,
        )

    return 0.0


logger = get_logger(__name__)

# Header region bonus constants moved to config.DecoderWeights (DWeights)

# Global cache for loaded models
_LOADED_MODELS = None
_MODEL_LOAD_TIME: float = 0.0
_LAST_DECODE_MODEL_STATE: dict[str, Any] = {}


def clear_model_cache() -> None:
    """Clear cached models so next decode loads fresh from disk."""
    global _LOADED_MODELS, _MODEL_LOAD_TIME
    _LOADED_MODELS = None
    _MODEL_LOAD_TIME = 0.0


def get_last_decode_model_state() -> dict[str, Any]:
    """Return metadata about the last model load attempt."""
    return dict(_LAST_DECODE_MODEL_STATE)


def _models_are_stale() -> bool:
    """Check if on-disk models are newer than cache. Cheap stat() call."""
    if _MODEL_LOAD_TIME == 0.0:
        return False
    manifest = paths.get_models_dir() / "manifest.json"
    try:
        return manifest.exists() and manifest.stat().st_mtime > _MODEL_LOAD_TIME
    except OSError:
        return False


# =============================================================================
# SCHEMA-DRIVEN FIELD DEFINITIONS
# =============================================================================
# These caches and mappings enable dynamic field configuration from schema,
# reducing hardcoded field sets and enabling new fields to be added via schema
# alone without code changes.
# =============================================================================

# Global cache for field definitions from schema
_FIELD_DEFINITIONS_CACHE: dict[str, dict] | None = None


def get_field_definitions() -> dict[str, dict]:
    """
    Load field definitions from schema, with caching for performance.

    Reads the `field_definitions` section from schema/contract.invoice.json
    which contains:
    - type: "amount", "date", "id", "text", etc.
    - bucket_preference: ["amount_like"], ["date_like"], etc.
    - confidence_threshold, validation rules, etc.

    Returns:
        Dict mapping field name -> definition dict with 'type',
        'bucket_preference', etc.

    Example:
        >>> defs = get_field_definitions()
        >>> defs["TotalAmount"]["type"]
        'amount'
        >>> defs["TotalAmount"]["bucket_preference"]
        ['amount_like']
    """
    global _FIELD_DEFINITIONS_CACHE
    if _FIELD_DEFINITIONS_CACHE is not None:
        return _FIELD_DEFINITIONS_CACHE

    schema = utils.load_contract_schema()
    _FIELD_DEFINITIONS_CACHE = schema.get("field_definitions", {})
    return _FIELD_DEFINITIONS_CACHE


# NOTE: TYPE_TO_ANCHOR moved to schema_registry.py
# FIELD_ANCHOR_OVERRIDES now handled via anchor_override in schema JSON
# FOOTER_AMOUNT_FIELDS_LOWER now handled via spatial_region='footer' in schema JSON


# =============================================================================
# FIELD PROFILE - Centralized field metadata for decoder logic
# =============================================================================


@dataclass(frozen=True, slots=True)
class FieldProfile:
    """
    Centralized field metadata for decoder scoring logic.

    This dataclass captures all field-specific attributes needed for cost
    computation, consolidating schema lookups and legacy fallback logic
    into a single cached object per field.

    Attributes:
        name: The field name (original case)
        field_type: The semantic type (amount, date, id, text, etc.)
        bucket_prefs: List of preferred bucket types from schema
        anchor_type: The anchor type for directional features
                     (total, date, id, name, etc.)
        is_footer: Whether the field typically appears in footer region
        is_header: Whether the field typically appears in header region
        is_keyword_proximal: Whether the field benefits from keyword proximity
        priority_bonus: Tiny tie-breaker bonus for field importance
        field_def: Full field definition dict from schema registry
    """

    name: str
    field_type: str
    bucket_prefs: tuple[str, ...]
    anchor_type: str | None
    is_footer: bool
    is_header: bool
    is_keyword_proximal: bool
    priority_bonus: float
    field_def: dict[str, Any]


# Cache for FieldProfile objects
_FIELD_PROFILE_CACHE: dict[str, FieldProfile] = {}


def build_field_profile(field: str) -> FieldProfile:
    """
    Build a FieldProfile for a field using schema_registry.

    Uses caching to avoid repeated lookups.

    Args:
        field: Field name (any case)

    Returns:
        FieldProfile with all relevant metadata for the field
    """
    # Check cache first
    if field in _FIELD_PROFILE_CACHE:
        return _FIELD_PROFILE_CACHE[field]

    # Get all field metadata from schema_registry
    field_type = registry.field_type(field)
    fdef = registry.field_def(field)
    bucket_prefs = tuple(fdef.get("bucket_preference", []))
    anchor_type = registry.anchor_type(field)
    is_footer = registry.is_footer(field)
    is_header = registry.is_header(field)
    is_kw_proximal = registry.is_keyword_proximal(field)
    pri_bonus = registry.priority_bonus(field)

    profile = FieldProfile(
        name=field,
        field_type=field_type,
        bucket_prefs=bucket_prefs,
        anchor_type=anchor_type,
        is_footer=is_footer,
        is_header=is_header,
        is_keyword_proximal=is_kw_proximal,
        priority_bonus=pri_bonus,
        field_def=fdef,
    )

    # Cache it
    _FIELD_PROFILE_CACHE[field] = profile
    return profile


def build_field_profiles(fields: list[str]) -> dict[str, FieldProfile]:
    """
    Pre-build FieldProfiles for all fields at once.

    Call this once before the cost matrix loop to avoid repeated
    schema registry lookups per cell. Returns a dict keyed by field name.

    Args:
        fields: List of field names to build profiles for

    Returns:
        Dict mapping field name -> FieldProfile
    """
    return {field: build_field_profile(field) for field in fields}


def clear_field_profile_cache() -> None:
    """Clear the field profile cache (call when field definitions change)."""
    _FIELD_PROFILE_CACHE.clear()


def _header_region_bonus(candidate: dict[str, Any]) -> float:
    """
    Compute header region bonus for fields with spatial_region='header'.

    Fields like VendorName/CustomerName appear in header region (top of invoice,
    near logo) and benefit from strong spatial signals rather than anchor-based matching.

    Args:
        candidate: Candidate dictionary with spatial features

    Returns:
        Header region bonus (positive = good spatial match)
    """
    bonus = 0.0
    in_header = candidate.get("in_top_quarter", 0.0)
    distance_to_top = candidate.get("distance_to_top", 1.0)

    # Strong bonus for header region (logo/name area)
    if in_header > 0:
        bonus += DWeights.HEADER_REGION_BONUS

    # Bonus for being near top of page (inverse of distance)
    if distance_to_top < DWeights.HEADER_TOP_CLOSE_THRESHOLD:
        bonus += DWeights.HEADER_TOP_CLOSE_BONUS
    elif distance_to_top < DWeights.HEADER_TOP_MID_THRESHOLD:
        bonus += DWeights.HEADER_TOP_MID_BONUS

    return bonus


def maybe_load_model_v1(expect_models: bool = False) -> dict[str, Any] | None:
    """Load models (tier 1: MLflow, tier 2: storage, tier 3: None/heuristic)."""
    global _LOADED_MODELS, _MODEL_LOAD_TIME, _LAST_DECODE_MODEL_STATE

    if not Config.USE_RANKER_MODEL:
        _LOADED_MODELS = None
        return None

    if _LOADED_MODELS is not None and not _models_are_stale():
        return _LOADED_MODELS

    # --- Tier 1: MLflow registry ---
    if Config.USE_MLFLOW:
        loaded = _try_load_from_mlflow()
        if loaded is not None:
            _LOADED_MODELS = loaded
            _MODEL_LOAD_TIME = time.time()
            _LAST_DECODE_MODEL_STATE = {
                "models_loaded": True,
                "model_fields": list(loaded.get("models", {}).keys()),
                "model_type": loaded.get("model_type", "unknown"),
            }
            return loaded

    # --- Tier 2: Storage backend ---
    manifest_path = paths.get_models_dir() / "manifest.json"
    manifest_exists = manifest_path.exists()

    try:
        loaded_models = train.load_trained_models()
        if loaded_models is not None:
            _LOADED_MODELS = loaded_models
            _MODEL_LOAD_TIME = time.time()
            _LAST_DECODE_MODEL_STATE = {
                "models_loaded": True,
                "model_fields": list(loaded_models.get("models", {}).keys()),
                "model_type": loaded_models.get("model_type", "unknown"),
            }
            return loaded_models
    except FileNotFoundError as e:
        logger.info(
            "model_not_found",
            reason=str(e),
            fallback="baseline_weak_prior",
        )
    except (ModelLoadError, ModelNotFoundError) as e:
        if expect_models and manifest_exists:
            raise ModelLoadError(
                model_id="post_train_models",
                reason=f"Models expected but failed to load: {e}",
            ) from e
        logger.warning(
            "model_load_failed",
            error_type=type(e).__name__,
            reason=str(e),
            fallback="baseline_weak_prior",
        )
    except (OSError, ValueError) as e:
        if expect_models and manifest_exists:
            raise ModelLoadError(
                model_id="post_train_models",
                reason=f"OS/Value error during load: {e}",
            ) from e
        logger.warning(
            "model_load_failed",
            error_type=type(e).__name__,
            reason=str(e),
            fallback="baseline_weak_prior",
        )

    # --- Tier 3: None (heuristic fallback) ---
    _LOADED_MODELS = None
    _LAST_DECODE_MODEL_STATE = {
        "models_loaded": False,
        "model_fields": [],
        "model_type": "none",
    }
    return None


def _try_load_from_mlflow() -> dict[str, Any] | None:
    """Load rankers from MLflow registry, or None on failure."""
    try:
        from .ranker import InvoiceFieldRanker
    except ImportError:
        return None

    models: dict[str, InvoiceFieldRanker] = {}
    for field_name in get_field_definitions():
        try:
            ranker = InvoiceFieldRanker.from_mlflow(field_name)
            models[field_name] = ranker
        except (ModelNotFoundError, ImportError, Exception) as e:
            logger.info(
                "mlflow_field_model_miss",
                field=field_name,
                error=str(e),
            )
            continue

    if not models:
        logger.info("mlflow_no_models_found", fallback="storage_backend")
        return None

    logger.info(
        "mlflow_models_loaded",
        n_fields=len(models),
        fields=list(models.keys()),
    )

    return {
        "models": models,
        "manifest": {"model_type": "ranker", "source": "mlflow"},
        "model_type": "ranker",
    }


def _extract_candidate_feature_vector(
    candidate: dict[str, Any], feature_names: list[str]
) -> list[float]:
    """Extract ordered feature vector from candidate for legacy classifier path.

    Uses the shared prepare_candidate_features() as the single source of truth,
    then converts the dict to an ordered list matching feature_names.

    Args:
        candidate: Candidate dictionary
        feature_names: List of feature names in model order

    Returns:
        Feature vector as list of floats in feature_names order
    """
    features = prepare_candidate_features(candidate)
    return [features.get(name, 0.0) for name in feature_names]


def compute_ranker_cost(
    field: str, candidate: dict[str, Any], loaded_models: dict[str, Any]
) -> tuple[float, float | None]:
    """
    Compute ML-based cost using trained XGBRanker model.

    The ranker returns relevance scores (higher = more relevant).
    We convert to cost by: cost = 1 - sigmoid(score / scale)

    Args:
        field: Field name
        candidate: Candidate dictionary
        loaded_models: Loaded models from maybe_load_model_v1()

    Returns:
        Tuple of (cost, score) where:
        - cost: ML cost (lower = better match), range [0, 1]
        - score: Raw ranker score, or None if fallback used
    """
    import pandas as pd

    models_dict = loaded_models.get("models", {})

    if field not in models_dict:
        # No model for this field, fall back to weak prior
        return compute_weak_prior_cost(field, candidate), None

    model_info = models_dict[field]
    ranker = model_info.get("ranker")

    if ranker is None:
        # This shouldn't happen for ranker models, but fall back gracefully
        return compute_weak_prior_cost(field, candidate), None

    try:
        # Extract canonical 59 features, then wrap as DataFrame for ranker
        features = prepare_candidate_features(candidate)
        candidate_df = pd.DataFrame([features])

        # Get ranking score (higher = more relevant)
        scores = ranker.predict(candidate_df)
        score = float(scores[0]) if len(scores) > 0 else 0.0

        # Convert score to cost using sigmoid normalization
        # Score can be any real number; sigmoid maps to (0, 1)
        # Then cost = 1 - sigmoid(score) so higher score = lower cost
        # Use reduced weight for bootstrap models (less influence until
        # more data accumulates)
        manifest = loaded_models.get("manifest", {})
        is_bootstrap = manifest.get("bootstrap_mode", False)
        ml_score_weight = (
            Config.BOOTSTRAP_ML_SCORE_WEIGHT if is_bootstrap else Config.ML_SCORE_WEIGHT
        )

        # Sigmoid with scaling: large positive scores → cost near 0
        # Negative scores → cost near 1
        sigmoid_score = 1.0 / (1.0 + np.exp(-score * ml_score_weight))
        ml_cost = 1.0 - sigmoid_score

        return max(0.0, min(1.0, ml_cost)), score

    except Exception as e:
        logger.warning(
            "ranker_cost_computation_failed",
            field=field,
            error_type=type(e).__name__,
            reason=str(e),
            fallback="weak_prior_cost",
        )
        return compute_weak_prior_cost(field, candidate), None


def compute_ml_cost_with_prob(
    field: str, candidate: dict[str, Any], loaded_models: dict[str, Any]
) -> tuple[float, float | None]:
    """
    Compute ML-based cost and probability using trained model.

    Automatically handles both XGBRanker and XGBClassifier models based on
    the model_type in loaded_models.

    Args:
        field: Field name
        candidate: Candidate dictionary
        loaded_models: Loaded models from maybe_load_model_v1()

    Returns:
        Tuple of (cost, probability/score) where:
        - cost: ML cost (lower = better match), range [0, 1]
        - probability: Raw model probability/score, or None if fallback used
    """
    # Check model type to dispatch to appropriate handler
    model_type = loaded_models.get("model_type", "classifier")

    if model_type == "ranker":
        return compute_ranker_cost(field, candidate, loaded_models)

    # Legacy classifier path
    models_dict = loaded_models.get("models", {})

    if field not in models_dict:
        # No model for this field, fall back to weak prior
        return compute_weak_prior_cost(field, candidate), None

    model_info = models_dict[field]
    model = model_info.get("model")

    if model is None:
        return compute_weak_prior_cost(field, candidate), None

    feature_names = model_info.get("feature_names", Config.get_feature_columns())

    try:
        feature_vector = _extract_candidate_feature_vector(candidate, feature_names)

        # Get prediction probability
        prob_positive = float(model.predict_proba([feature_vector])[0][1])

        # Convert to cost (lower probability = higher cost)
        ml_cost = 1.0 - prob_positive

        return max(0.0, ml_cost), prob_positive

    except (KeyError, IndexError, TypeError) as e:
        # Feature extraction or prediction shape errors
        logger.warning(
            "ml_cost_computation_failed",
            field=field,
            error_type=type(e).__name__,
            reason=str(e),
            fallback="weak_prior_cost",
        )
        return compute_weak_prior_cost(field, candidate), None
    except (ValueError, AttributeError) as e:
        # Model prediction errors (invalid input, missing method)
        logger.warning(
            "ml_cost_computation_failed",
            field=field,
            error_type=type(e).__name__,
            reason=str(e),
            fallback="weak_prior_cost",
        )
        return compute_weak_prior_cost(field, candidate), None


def try_import_scipy_hungarian() -> Any:
    """Try to import scipy's Hungarian algorithm implementation."""
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

        return linear_sum_assignment
    except ImportError:
        return None


def simple_hungarian_fallback(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple fallback Hungarian algorithm implementation.
    Not optimal but works for small matrices.
    """
    n_rows, n_cols = cost_matrix.shape

    # Greedy assignment - assign each row to its minimum cost column
    row_indices = []
    col_indices = []
    used_cols = set()

    # Sort rows by their minimum cost to prioritize easier assignments
    row_min_costs = [(i, np.min(cost_matrix[i])) for i in range(n_rows)]
    row_min_costs.sort(key=lambda x: x[1])

    for row_idx, _ in row_min_costs:
        # Find best available column for this row
        best_col = None
        best_cost = float("inf")

        for col_idx in range(n_cols):
            if col_idx not in used_cols and cost_matrix[row_idx, col_idx] < best_cost:
                best_col = col_idx
                best_cost = cost_matrix[row_idx, col_idx]

        if best_col is not None:
            row_indices.append(row_idx)
            col_indices.append(best_col)
            used_cols.add(best_col)

    return np.array(row_indices), np.array(col_indices)


def _compute_directional_bonus_for_anchor(
    candidate: dict[str, Any],
    anchor_type: str,
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
    if anchor_type is None:
        return 0.0

    dist = candidate.get(f"dist_to_{anchor_type}", 1.0)
    below = candidate.get(f"below_{anchor_type}", 0.0)
    reading_order = candidate.get(f"reading_order_{anchor_type}", 0.0)
    aligned_y = candidate.get(f"aligned_y_{anchor_type}", 0.0)

    bonus = 0.0

    # Strong bonus for being in same column and below anchor header
    if below > 0 and dist < DWeights.DIRECTIONAL_BELOW_CLOSE_THRESHOLD:
        bonus += DWeights.DIRECTIONAL_BELOW_CLOSE_BONUS

    # Bonus for reading order (label: value pattern)
    if reading_order > 0 and dist < DWeights.DIRECTIONAL_READING_ORDER_THRESHOLD:
        bonus += DWeights.DIRECTIONAL_READING_ORDER_BONUS

    # Bonus for same row, close distance ("Label: Value" horizontal pattern)
    if aligned_y > 0 and dist < DWeights.DIRECTIONAL_SAME_ROW_THRESHOLD:
        bonus += DWeights.DIRECTIONAL_SAME_ROW_BONUS

    # Penalty for being far from anchor
    if dist > DWeights.DIRECTIONAL_FAR_THRESHOLD:
        bonus += DWeights.DIRECTIONAL_FAR_PENALTY

    return bonus


def _compute_schema_driven_costs(
    field: str,
    candidate: dict[str, Any],
    field_def: dict[str, Any],
    profile: FieldProfile | None = None,
) -> tuple[float, float]:
    """Compute bucket affinity and directional bonus using schema field definition."""
    field_type = field_def.get("type", "text")
    bucket_prefs = field_def.get("bucket_preference", [])
    bucket = candidate.get("bucket", "")

    # ---------------------------------------------------------
    # Bucket affinity based on schema's bucket_preference
    # Hard bucket match is the primary signal. Soft probabilities
    # only soften the mismatch penalty (never inflate the bonus).
    # ---------------------------------------------------------
    bucket_bonus = 0.0

    if bucket in bucket_prefs:
        # Hard match: full bucket bonus regardless of soft probabilities
        bucket_bonus = DWeights.BUCKET_MATCH_BONUS
    elif bucket_prefs:
        # Field has preferences but candidate doesn't match.
        # Build mismatch penalty from type-bucket compatibility table.
        # Each type has "strong mismatch" buckets (completely wrong type),
        # "moderate mismatch" buckets (plausible but unlikely), and default.
        _STRONG_MISMATCH: dict[str, set[str]] = {
            "amount": {"date_like"},
            "date": {"amount_like"},
            "id": {"date_like"},
            "text": {"amount_like", "date_like"},
        }
        _MODERATE_MISMATCH: dict[str, set[str]] = {
            "amount": {"id_like"},
            "date": {"id_like"},
            "id": {"amount_like"},
            "text": {"id_like"},
        }
        # Neutral buckets (no penalty) — keyword_proximal for amounts
        _NEUTRAL: dict[str, set[str]] = {
            "amount": {"keyword_proximal"},
        }

        # Check soft bucket probabilities: if the candidate has a decent
        # probability of being a preferred bucket type, soften the penalty.
        soft_prob_sum = sum(
            candidate.get(f"bucket_prob_{bp}", 0.0) for bp in bucket_prefs
        )

        if bucket in _NEUTRAL.get(field_type, set()):
            bucket_bonus = 0.0
        elif bucket in _STRONG_MISMATCH.get(field_type, set()):
            base_penalty = DWeights.BUCKET_MISMATCH_STRONG
            # Soften if soft probability suggests some affinity
            if soft_prob_sum > 0.3:
                bucket_bonus = base_penalty * 0.5  # halve the penalty
            else:
                bucket_bonus = base_penalty
        elif bucket in _MODERATE_MISMATCH.get(field_type, set()):
            base_penalty = DWeights.BUCKET_MISMATCH_MODERATE
            if soft_prob_sum > 0.3:
                bucket_bonus = base_penalty * 0.5
            else:
                bucket_bonus = base_penalty
        else:
            base_penalty = DWeights.BUCKET_MISMATCH_MILD
            if soft_prob_sum > 0.3:
                bucket_bonus = base_penalty * 0.5
            else:
                bucket_bonus = base_penalty

    # ---------------------------------------------------------
    # Directional bonus based on field type -> anchor mapping
    # ---------------------------------------------------------
    directional_bonus = 0.0

    # Special handling for header fields: use header region, not anchors
    _is_header = profile.is_header if profile is not None else registry.is_header(field)
    if _is_header:
        directional_bonus += _header_region_bonus(candidate)
    else:
        # Get anchor type from profile (cached) or registry (fallback)
        anchor_type = (
            profile.anchor_type if profile is not None else registry.anchor_type(field)
        )

        if anchor_type:
            directional_bonus = _compute_directional_bonus_for_anchor(
                candidate, anchor_type
            )

            # Schema-driven anchor bonus override: extra directional bonus
            # for fields that need stronger anchor matching (e.g., TaxAmount→tax)
            abo = (
                profile.field_def.get("anchor_bonus_override")
                if profile
                else registry.anchor_bonus_override(field)
            )
            if abo:
                abo_anchor = abo.get("anchor", "")
                abo_dist = candidate.get(f"dist_to_{abo_anchor}", 1.0)
                abo_below = candidate.get(f"below_{abo_anchor}", 0.0)
                abo_ro = candidate.get(f"reading_order_{abo_anchor}", 0.0)

                if abo_below > 0 and abo_dist < abo.get("below_dist_threshold", 0.3):
                    directional_bonus += abo.get("below_bonus", 0.5)
                if abo_ro > 0 and abo_dist < abo.get(
                    "reading_order_dist_threshold", 0.2
                ):
                    directional_bonus += abo.get("reading_order_bonus", 0.4)

    return bucket_bonus, directional_bonus


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


def apply_cross_field_adjustments(
    cost_matrix: np.ndarray,
    candidates_list: list[dict[str, Any]],
    schema_fields: list[str],
) -> None:
    """Adjust cost matrix to reward cross-field consistency BEFORE Hungarian.

    Checks if candidate combinations satisfy semantic relationships:
    - Subtotal + TaxAmount ~ TotalAmount (within tolerance)
    - Subtotal <= TotalAmount

    When consistent candidate sets are found, their costs are reduced so
    the Hungarian algorithm favors internally-consistent assignments.

    Mutates cost_matrix in place.

    Args:
        cost_matrix: Shape (n_fields, n_candidates + n_fields). Modified in place.
        candidates_list: List of candidate dicts with raw_text.
        schema_fields: Ordered list of field names matching cost_matrix rows.
    """
    if not candidates_list:
        return

    # Find amount-type fields from schema (no hardcoded names)
    amount_fields = registry.fields_by_type("amount")
    if not amount_fields:
        return

    # Build field index lookup for the fields present in schema_fields
    field_to_idx: dict[str, int] = {}
    for idx, fname in enumerate(schema_fields):
        if fname in amount_fields:
            field_to_idx[fname] = idx

    # We need at least 2 amount fields to do cross-field checks
    if len(field_to_idx) < 2:
        return

    # Get field definitions to identify relationships
    # Look for the "total" field (highest importance among amounts)
    # and fields that should sum to it
    total_field: str | None = None
    component_fields: list[str] = []

    for fname in field_to_idx:
        fdef = registry.field_def(fname)
        # The field with highest importance or priority_bonus is likely the total
        if fdef.get("priority_bonus", 0) > 0 or fdef.get("importance", 0) >= 0.9:
            total_field = fname
        else:
            component_fields.append(fname)

    if total_field is None or not component_fields:
        return

    n_candidates = len(candidates_list)
    total_idx = field_to_idx[total_field]

    # Pre-parse all candidate amounts once
    parsed_amounts: list[float | None] = []
    for cand in candidates_list:
        text = str(cand.get("raw_text") or cand.get("text", ""))
        parsed_amounts.append(_parse_amount_value(text))

    # Cross-field consistency bonus: -0.3 cost reduction, applied at most ONCE
    # per (field_idx, candidate_idx) cell to prevent runaway accumulation.
    consistency_bonus = 0.3
    tolerance = 0.05  # 5% relative tolerance

    # Track which cells have already received a consistency bonus
    adjusted_cells: set[tuple[int, int]] = set()

    # For each candidate pair (total_candidate, component_candidate),
    # check if any subset of components sums to the total
    for total_cand_idx in range(n_candidates):
        total_val = parsed_amounts[total_cand_idx]
        if total_val is None or total_val <= 0:
            continue

        for comp_field in component_fields:
            comp_idx = field_to_idx[comp_field]
            for comp_cand_idx in range(n_candidates):
                if comp_cand_idx == total_cand_idx:
                    continue  # Same candidate can't be both total and component
                comp_val = parsed_amounts[comp_cand_idx]
                if comp_val is None or comp_val <= 0:
                    continue

                # Check: component <= total (basic sanity)
                if comp_val <= total_val * (1.0 + tolerance):
                    # Check: can we find a complementary component?
                    remainder = total_val - comp_val
                    if remainder < 0:
                        continue

                    # Look for another component that fills the gap
                    for other_field in component_fields:
                        if other_field == comp_field:
                            continue
                        other_idx = field_to_idx[other_field]
                        for other_cand_idx in range(n_candidates):
                            if other_cand_idx in (total_cand_idx, comp_cand_idx):
                                continue
                            other_val = parsed_amounts[other_cand_idx]
                            if other_val is None:
                                continue

                            # Check: comp + other ~ total
                            combined = comp_val + other_val
                            if (
                                total_val > 0
                                and abs(combined - total_val) / total_val <= tolerance
                            ):
                                # Consistent triple found! Reduce costs
                                # (at most once per cell).
                                for cell in [
                                    (total_idx, total_cand_idx),
                                    (comp_idx, comp_cand_idx),
                                    (other_idx, other_cand_idx),
                                ]:
                                    if cell not in adjusted_cells:
                                        adjusted_cells.add(cell)
                                        cost_matrix[cell[0], cell[1]] -= (
                                            consistency_bonus
                                        )


def compute_weak_prior_cost(
    field: str,
    candidate: dict[str, Any],
    profile: FieldProfile | None = None,
    document_labels: set[str] | None = None,
    colon_name_values: dict[str, int] | None = None,
    cross_page_headers: set[str] | None = None,
    address_city_tokens: set[str] | None = None,
) -> float:
    """
    Compute weak prior cost for field-candidate assignment based on heuristics.
    This serves as a fallback when no ML model is available.
    Lower cost = better match.

    Schema-Driven Design:
    Dynamically reads field configuration from schema/contract.invoice.json:
    - Field type determines anchor preferences automatically
    - Bucket preferences come from schema's bucket_preference array
    - Adding new fields requires only schema updates, no code changes

    Uses directional vector features to match fields with typed anchors:
    - A value BELOW a "Total" header is likely the total amount
    - A value TO THE RIGHT of "Invoice Date:" is likely the date
    - Direction matters, not just distance

    Args:
        field: Field name (e.g., "TotalAmount", "InvoiceDate")
        candidate: Candidate dictionary with features
        profile: Optional pre-built FieldProfile to avoid registry lookups.
                 When provided, all schema registry lookups use cached values.
        document_labels: Optional set of structural label tokens for this document.
        cross_page_headers: Optional set of cross-page header tokens (3+ pages).
        address_city_tokens: Optional set of address city tokens (near ZIP codes).

    Returns:
        Cost value (lower = better match for this field)
    """
    bucket = candidate.get("bucket", "")
    proximity_score = candidate.get("proximity_score", 0.0)
    section_prior = candidate.get("section_prior", 0.0)
    cohesion_score = candidate.get("cohesion_score", 0.0)

    # Feature-based cost using ML-extracted features
    base_cost = 1.0

    # ================================================================
    # FIELD-SPECIFIC BUCKET AFFINITY + DIRECTIONAL BONUS
    # All fields now schema-driven via registry
    # ================================================================
    field_def = profile.field_def if profile is not None else registry.field_def(field)
    bucket_bonus, directional_bonus = _compute_schema_driven_costs(
        field, candidate, field_def, profile=profile
    )

    # ================================================================
    # COMMON BONUSES/PENALTIES (apply to all fields)
    # ================================================================

    # Keyword proximity bonus applies to structured fields (id, amount, date)
    # For text fields (names), keyword_proximal is often noise (generic words near keywords)
    if bucket == "keyword_proximal":
        # Only fields with keyword_proximal=True benefit from keyword proximity
        _is_kw_proximal = (
            profile.is_keyword_proximal
            if profile is not None
            else registry.is_keyword_proximal(field)
        )
        if _is_kw_proximal:
            bucket_bonus += DWeights.BUCKET_KEYWORD_PROXIMAL_BONUS

    # Random negative penalty applies to all fields
    elif bucket == "random_negative":
        bucket_bonus = Config.BUCKET_PENALTY_STRONG  # -0.4

    # ================================================================
    # TEXT PATTERN VALIDATION (Critical for avoiding garbage assignments)
    # ================================================================
    # Validates that the candidate text actually looks like the expected
    # field type. High weight because bucket matching alone is insufficient.
    # ================================================================
    text_pattern_bonus = compute_text_pattern_bonus(
        field,
        candidate,
        profile=profile,
        document_labels=document_labels,
        cross_page_headers=cross_page_headers,
        address_city_tokens=address_city_tokens,
    )

    # ================================================================
    # FOOTER REGION BONUS
    # Footer elements (Total, Subtotal, Tax) benefit from being near page bottom
    # ================================================================
    y_from_bottom = candidate.get("y_from_bottom", 0.5)
    in_footer_region = candidate.get("in_bottom_quarter", 0.0)

    _is_footer = profile.is_footer if profile is not None else registry.is_footer(field)
    if _is_footer:
        if in_footer_region > 0 or y_from_bottom < DWeights.FOOTER_Y_THRESHOLD:
            directional_bonus += DWeights.FOOTER_REGION_BONUS

    # ================================================================
    # FIELD PRIORITY TIE-BREAKER
    # ================================================================
    # When multiple fields have identical costs for the same candidate,
    # add a tiny bias favoring more important fields.
    # This prevents Hungarian assignment from arbitrarily preferring Subtotal
    # over TotalAmount when both match equally well.
    # ================================================================
    field_priority_bonus = (
        profile.priority_bonus
        if profile is not None
        else registry.priority_bonus(field)
    )

    # ================================================================
    # FIELD-SPECIFIC COLON-VALUE BONUS (Fix #3)
    # "To: GTT Americas LLC" pattern — VendorName gets +3.0
    # (rank-weighted: first word gets most). Other fields get +0.5.
    # Computed before page-frequency bonus so it can gate that bonus.
    # ================================================================
    colon_bonus = 0.0
    if colon_name_values:
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        words = raw_text.lower().split()
        matched_ranks = [colon_name_values[w] for w in words if w in colon_name_values]
        if matched_ranks:
            if field == "VendorName":
                # Rank-weighted: rank 0 gets 3.0, rank 1 gets 2.0, etc.
                colon_bonus = sum(
                    max(0.5, 3.0 - rank * 1.0) for rank in matched_ranks
                ) / len(matched_ranks)
            else:
                colon_bonus = 0.5

    # ================================================================
    # VENDORNAME PAGE-FREQUENCY BONUS (Fix #2)
    # Vendor logos/names repeat across pages; other fields don't.
    # Only VendorName gets this bonus (scale 5.0).
    # ================================================================
    page_freq_bonus = 0.0
    if field == "VendorName":
        # Only apply page-frequency bonus if the candidate passed text
        # validation (non-negative score) OR has colon-value support.
        # Blacklisted words like "CONSOLIDATED", "BILLING" repeat across
        # pages but are labels, not vendor names — they must NOT benefit
        # from page-frequency. Cross-page headers with colon support
        # (like "AT&T" after "To:") are legitimate vendor names that
        # should still get the bonus.
        has_colon_support = colon_bonus > 0
        if text_pattern_bonus >= 0 or has_colon_support:
            page_freq = candidate.get("page_frequency", 0.0)
            page_freq_bonus = page_freq * 5.0

    # ================================================================
    # TOTALAMOUNT MAGNITUDE BONUS
    # The total is usually the largest amount on the invoice.
    # Give a small bonus proportional to the log of the parsed amount.
    # This helps disambiguate when multiple amounts have similar costs.
    # ================================================================
    magnitude_bonus = 0.0
    if field == "TotalAmount":
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        parsed_val = _parse_amount_value(raw_text)
        if parsed_val is not None and abs(parsed_val) > 0:
            import math

            # log10 scaling: $100 = 0.8, $1000 = 1.2, $10000 = 1.6, $100000 = 2.0
            # Scale by 2.0 so magnitude can overcome directional/proximity bonuses
            magnitude_bonus = min(2.0, math.log10(max(1.0, abs(parsed_val))) / 2.5)

    # ================================================================
    # DENSE LABEL DETECTION (Fix #10)
    # Count both x and y alignments for anchor type detection.
    # Candidates aligned with many other candidates on both axes
    # are likely in a table/label region — penalize for name fields.
    # ================================================================
    dense_label_penalty = 0.0
    _field_type_dense = (
        profile.field_type if profile is not None else registry.field_type(field)
    )
    if _field_type_dense in ("text", "address"):
        x_aligned = candidate.get("aligned_x_name", 0.0)
        y_aligned = candidate.get("aligned_y_name", 0.0)
        if x_aligned > 0 and y_aligned > 0:
            dense_label_penalty = -0.3

    # ================================================================
    # COMBINE ALL FEATURES INTO FINAL COST
    # ================================================================
    # Weights: text pattern (highest) > bucket > directional > secondary signals
    # ================================================================

    # For text/name fields, apply directional dampening to anchor bonuses separately
    # The 0.5 dampening should ONLY affect anchor-based directional bonuses,
    # NOT the header bonus (which is a strong spatial signal)
    directional_weight = 1.0
    header_bonus_component = 0.0
    anchor_directional_component = directional_bonus

    _is_header_final = (
        profile.is_header if profile is not None else registry.is_header(field)
    )
    _field_type_final = (
        profile.field_type if profile is not None else registry.field_type(field)
    )
    if _is_header_final or _field_type_final == "text":
        # For header fields, separate header bonus from anchor-based directional bonus
        if _is_header_final:
            header_bonus_component = _header_region_bonus(candidate)
            anchor_directional_component = directional_bonus - header_bonus_component
        # Reduce weight for anchor-based spatial features (not header bonus)
        directional_weight = DWeights.DIRECTIONAL_DAMPENING

    # Amplify negative text_pattern_bonus (wrong type penalty)
    amplified_text_pattern = (
        text_pattern_bonus * DWeights.TEXT_PATTERN_NEGATIVE_AMPLIFIER
        if text_pattern_bonus < 0
        else text_pattern_bonus
    )

    feature_cost = base_cost - (
        bucket_bonus
        + header_bonus_component  # Full weight for header bonus
        + anchor_directional_component * directional_weight  # Dampened for name fields
        + amplified_text_pattern  # Critical: validates text matches expected field type
        + proximity_score * DWeights.PROXIMITY_WEIGHT
        + section_prior * DWeights.SECTION_PRIOR_WEIGHT
        + (cohesion_score / DWeights.COHESION_NORMALIZER) * DWeights.COHESION_WEIGHT
        + field_priority_bonus  # Tiny tie-breaker for important fields
        + page_freq_bonus  # VendorName page-frequency bonus
        + colon_bonus  # Colon-value pattern bonus
        + dense_label_penalty  # Dense label region penalty
        + magnitude_bonus  # TotalAmount prefers larger amounts
    )

    # Apply format_hint penalty from schema (if defined for this field)
    # Schema-driven: format_hint is a regex; format_penalty is the cost addition
    profile = build_field_profile(field)
    fmt_hint = profile.field_def.get("format_hint")
    if fmt_hint:
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        if not re.search(fmt_hint, raw_text):
            fmt_penalty = float(profile.field_def.get("format_penalty", 0.3))
            feature_cost += fmt_penalty

    # Schema-driven spatial bias: penalize candidates in wrong page region.
    # e.g., InvoiceDate has spatial_bias.position="top" → penalize if in bottom half
    # e.g., DueDate has spatial_bias.position="bottom" → penalize if in top third
    s_bias = profile.field_def.get("spatial_bias")
    if s_bias:
        center_y = candidate.get("center_y", 0.5)
        bias_pos = s_bias.get("position")
        bias_penalty = float(s_bias.get("penalty", 0.3))
        bias_threshold = float(s_bias.get("threshold", 0.5))
        if bias_pos == "top" and center_y > bias_threshold:
            feature_cost += bias_penalty
        elif bias_pos == "bottom" and center_y < bias_threshold:
            feature_cost += bias_penalty

    # Allow negative costs - Hungarian algorithm works with any cost values
    # Lower cost = better match. Clamping at 0 loses all discrimination.
    return feature_cost  # type: ignore[no-any-return]


def _prune_with_ranker(
    candidates_df: pd.DataFrame,
    ranker_models: dict[str, Any],
    max_candidates: int | None = None,
) -> pd.DataFrame:
    """Prune candidates using ranker predict() scores with percentile threshold.

    Ranker models output unbounded relevance scores (not probabilities).
    We aggregate each candidate's max score across all field rankers, then
    discard candidates below the 10th percentile — these are almost certainly
    irrelevant across all fields.

    Args:
        candidates_df: DataFrame of candidates to prune.
        ranker_models: Dict mapping field name -> model info with 'ranker' key.
        max_candidates: Hard cap on candidates to keep.

    Returns:
        Pruned DataFrame, respecting max_candidates.
    """
    import pandas as pd

    if max_candidates is None:
        max_candidates = Config.PRUNING_MAX_CANDIDATES

    original_count = len(candidates_df)

    # Skip if already within limits
    pruning_min_trigger = max(0, Config.PRUNING_MIN_TRIGGER)
    if original_count <= pruning_min_trigger:
        return candidates_df

    # Score each candidate across all field rankers, keeping the max score.
    # A candidate only needs to be relevant to ONE field to survive pruning.
    max_scores = np.full(original_count, -np.inf)

    for field_name, model_info in ranker_models.items():
        ranker = model_info.get("ranker") if isinstance(model_info, dict) else None
        if ranker is None:
            continue
        try:
            cand_features = pd.DataFrame(
                [
                    prepare_candidate_features(row)
                    for row in candidates_df.to_dict("records")
                ]
            )
            scores = ranker.predict(cand_features)
            max_scores = np.maximum(max_scores, scores)
        except Exception as e:
            logger.warning(
                "ranker_pruning_score_failed",
                field=field_name,
                error=str(e),
            )
            continue

    # If no ranker succeeded, skip pruning
    if np.all(np.isinf(max_scores)):
        logger.info("ranker_pruning_skipped_no_scores")
        return candidates_df

    # Percentile threshold: prune bottom 10% of scores
    threshold = float(np.percentile(max_scores, 10))
    keep_mask = max_scores >= threshold

    pruned_df = candidates_df[keep_mask].copy()

    # Hard cap: keep top-scoring candidates
    if len(pruned_df) > max_candidates:
        # Sort by max_scores descending, keep top max_candidates
        pruned_scores = max_scores[keep_mask]
        top_indices = np.argsort(-pruned_scores)[:max_candidates]
        pruned_df = pruned_df.iloc[top_indices]

    pruned_df = pruned_df.reset_index(drop=True)

    logger.info(
        "candidates_pruned_ranker",
        original=original_count,
        final=len(pruned_df),
        score_threshold=round(threshold, 4),
        max_candidates=max_candidates,
    )

    return pruned_df


def decode_document(
    sha256: str,
    none_bias: float | None = None,
    candidates_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Decode a single document using Hungarian assignment with ML models when available.

    Args:
        sha256: Document SHA256 hash
        none_bias: Cost for NONE assignment (higher = more likely to abstain).
                   If None, uses Config.DECODER_NONE_BIAS.
        candidates_df: Optional pre-filtered DataFrame of candidates.
                       If provided, uses this instead of loading from disk.
                       This enables XGBoost-based pruning before decoding to
                       address O(n^3) Hungarian algorithm scaling.

    Returns:
        Assignment results for each field, including:
        - assignment_type: "CANDIDATE" or "NONE"
        - candidate_index: Index of assigned candidate (or None)
        - cost: Assignment cost (lower = better match)
        - field: Field name
        - candidate: Candidate data (if CANDIDATE assignment)
        - used_ml_model: Whether ML model was used for scoring
        - ml_probability: ML model probability (if ML was used)
    """
    start_time = time.perf_counter()
    import pandas as pd  # Import here to avoid circular import issues

    if none_bias is None:
        none_bias = Config.DECODER_NONE_BIAS

    # Try to load trained models
    loaded_models = maybe_load_model_v1()

    # Get schema fields dynamically
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    if not schema_fields:
        raise ValueError("Schema contains no fields")

    # Get candidates - use provided DataFrame or load from disk
    if candidates_df is None:
        candidates_df = candidates.get_document_candidates(sha256)
    elif not isinstance(candidates_df, pd.DataFrame):
        raise TypeError("candidates_df must be a pandas DataFrame")

    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")

    # If no candidates, return all NONE assignments
    if candidates_df.empty:
        assignments = {}
        for field in schema_fields:
            assignments[field] = {
                "assignment_type": "NONE",
                "candidate_index": None,
                "cost": none_bias,
                "field": field,
                "used_ml_model": False,
                "ml_probability": None,
            }
        return assignments

    candidates_list = candidates_df.to_dict("records")
    n_candidates = len(candidates_list)
    n_fields = len(schema_fields)

    # Detect structural labels once per document for text pattern scoring
    from .adaptive import (
        detect_address_city_tokens,
        detect_colon_name_values,
        detect_cross_page_headers,
        detect_document_labels,
    )

    doc_labels = detect_document_labels(sha256)
    colon_name_values = detect_colon_name_values(sha256)
    cross_page_headers = detect_cross_page_headers(sha256)
    address_city_tokens = detect_address_city_tokens(sha256)

    # Build cost matrix: fields × (candidates + per-field NONE columns)
    # Each field can be assigned to any candidate or to its own NONE column,
    # so multiple fields can abstain independently.
    cost_matrix = np.full((n_fields, n_candidates + n_fields), Config.DECODER_BASE_COST)

    # Track ML probabilities for confidence scoring
    # ml_probs[field_idx][cand_idx] = probability (or None if heuristic)
    ml_probs: dict[int, dict[int, float | None]] = {i: {} for i in range(n_fields)}

    # Track which fields have ML models
    fields_with_ml: set[str] = set()
    if loaded_models:
        fields_with_ml = set(loaded_models.get("models", {}).keys())

    # Pre-build field profiles ONCE to avoid repeated schema registry lookups
    # in the O(fields x candidates) cost matrix loop
    field_profiles = build_field_profiles(schema_fields)

    # Fill candidate costs (NONE column set in second pass below)
    for field_idx, field in enumerate(schema_fields):
        profile = field_profiles[field]
        for cand_idx, candidate in enumerate(candidates_list):
            if loaded_models and field in fields_with_ml:
                # Use ML cost when model available for this field
                cost, probability = compute_ml_cost_with_prob(
                    field, candidate, loaded_models
                )
                ml_probs[field_idx][cand_idx] = probability
            else:
                # Fall back to weak prior cost (with cached profile)
                cost = compute_weak_prior_cost(
                    field,
                    candidate,
                    profile=profile,
                    document_labels=doc_labels,
                    colon_name_values=colon_name_values,
                    cross_page_headers=cross_page_headers,
                    address_city_tokens=address_city_tokens,
                )
                ml_probs[field_idx][cand_idx] = None
            cost_matrix[field_idx, cand_idx] = cost

    # Cross-field consistency adjustment: reward candidate sets where
    # amount fields are internally consistent (e.g., Subtotal + Tax ~ Total).
    # This runs BEFORE Hungarian so it can influence the assignment.
    apply_cross_field_adjustments(cost_matrix, candidates_list, schema_fields)

    # Auto-derive per-field NONE bias from data when sentinel (<0) is set.
    # Each field gets its own NONE cost based on its best candidate, so fields
    # with strong candidates have low NONE thresholds (less abstention) and
    # fields with weak candidates have high NONE thresholds (more abstention).
    use_per_field_none = none_bias < 0
    per_field_none_costs: np.ndarray | None = None
    if use_per_field_none:
        if n_candidates == 0:
            none_bias = 0.0
        else:
            candidate_costs = cost_matrix[:, :n_candidates]
            # Use MEDIAN of per-field costs (not min). The min is dominated
            # by the single best candidate which might be garbage. The median
            # is more robust — NONE should be 80% of median so only candidates
            # significantly better than average can beat NONE.
            per_field_medians = np.median(candidate_costs, axis=1)
            per_field_none_costs = per_field_medians * 0.8

            # Required fields get prohibitive NONE cost — they should
            # always pick a candidate rather than abstain.
            schema_obj_for_defs = utils.load_contract_schema()
            field_defs = schema_obj_for_defs.get("field_definitions", {})
            for field_idx, field in enumerate(schema_fields):
                fdef = field_defs.get(field, {})
                if fdef.get("required", False):
                    per_field_none_costs[field_idx] = 1e9

            # Log shared summary for debugging
            none_bias = float(np.percentile(per_field_medians, 75))
        logger.info(
            "none_bias_auto_derived",
            mode="per_field",
            summary_p75=none_bias,
            n_fields=n_fields,
            n_candidates=n_candidates,
            per_field_none_sample={
                schema_fields[i]: round(float(per_field_none_costs[i]), 4)
                for i in range(min(5, n_fields))
            }
            if per_field_none_costs is not None
            else {},
        )

    # Second pass: set per-field NONE columns (diagonal in the NONE block).
    # Off-diagonal NONE entries must be prohibitive so the Hungarian algorithm
    # never assigns a field to another field's NONE column.
    cost_matrix[:, n_candidates:] = 1e9  # block all cross-field NONE assignments
    for field_idx in range(n_fields):
        none_col = n_candidates + field_idx
        if per_field_none_costs is not None:
            cost_matrix[field_idx, none_col] = per_field_none_costs[field_idx]
        else:
            cost_matrix[field_idx, none_col] = none_bias  # explicit bias fallback
        ml_probs[field_idx][none_col] = None  # NONE has no probability

    # Apply Hungarian algorithm
    hungarian_fn = try_import_scipy_hungarian()

    if hungarian_fn is not None:
        try:
            row_indices, col_indices = hungarian_fn(cost_matrix)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Numerical issues with cost matrix (inf, nan, singular)
            pipeline_errors.labels(stage="decoder").inc()
            logger.warning(
                "hungarian_algorithm_failed",
                error_type=type(e).__name__,
                reason=str(e),
                fallback="simple_hungarian",
            )
            row_indices, col_indices = simple_hungarian_fallback(cost_matrix)
        except MemoryError as e:
            # Cost matrix too large
            pipeline_errors.labels(stage="decoder").inc()
            logger.warning(
                "hungarian_algorithm_failed",
                error_type="MemoryError",
                reason=str(e),
                fallback="simple_hungarian",
            )
            row_indices, col_indices = simple_hungarian_fallback(cost_matrix)
    else:
        logger.info("scipy_not_available", fallback="simple_hungarian")
        row_indices, col_indices = simple_hungarian_fallback(cost_matrix)

    # Build assignments
    assignments = {}

    for field_idx, field in enumerate(schema_fields):
        # Find assignment for this field
        field_assignment = None
        for _i, (row_idx, col_idx) in enumerate(
            zip(row_indices, col_indices, strict=False)
        ):
            if row_idx == field_idx:
                field_assignment = (col_idx, cost_matrix[row_idx, col_idx])
                break

        # Determine if ML was used for this field
        field_used_ml = field in fields_with_ml

        if field_assignment is None:
            # No assignment found, default to NONE
            assignments[field] = {
                "assignment_type": "NONE",
                "candidate_index": None,
                "cost": none_bias,
                "field": field,
                "used_ml_model": False,
                "ml_probability": None,
            }
        else:
            col_idx, cost = field_assignment

            if col_idx >= n_candidates:  # NONE assignment (per-field NONE columns)
                assignments[field] = {
                    "assignment_type": "NONE",
                    "candidate_index": None,
                    "cost": cost,
                    "field": field,
                    "used_ml_model": False,
                    "ml_probability": None,
                }
            else:  # Candidate assignment
                raw_score = ml_probs[field_idx].get(col_idx)
                # XGBRanker returns unbounded relevance scores, not
                # probabilities.  Apply sigmoid to map to [0, 1] so
                # downstream confidence is calibrated, not binary.
                if raw_score is not None:
                    ml_prob: float | None = float(1.0 / (1.0 + np.exp(-raw_score)))
                else:
                    ml_prob = None

                # Build fallback candidates: top 3 alternatives sorted
                # by cost (excluding assigned candidate and NONE cols)
                field_costs = cost_matrix[field_idx, :n_candidates]
                # Get indices of all candidates except the assigned one
                alt_indices = [i for i in range(n_candidates) if i != col_idx]
                # Sort by cost ascending, take top 3
                alt_indices.sort(key=lambda i: field_costs[i])
                fallbacks = []
                for alt_idx in alt_indices[:3]:
                    alt_raw = ml_probs[field_idx].get(alt_idx)
                    if alt_raw is not None:
                        alt_ml_prob: float | None = float(
                            1.0 / (1.0 + np.exp(-alt_raw))
                        )
                    else:
                        alt_ml_prob = None
                    fallbacks.append(
                        {
                            "candidate_index": alt_idx,
                            "cost": float(field_costs[alt_idx]),
                            "candidate": candidates_list[alt_idx],
                            "used_ml_model": field_used_ml and alt_ml_prob is not None,
                            "ml_probability": alt_ml_prob,
                        }
                    )

                assignments[field] = {
                    "assignment_type": "CANDIDATE",
                    "candidate_index": col_idx,
                    "cost": cost,
                    "field": field,
                    "candidate": candidates_list[col_idx],
                    "used_ml_model": field_used_ml and ml_prob is not None,
                    "ml_probability": ml_prob,
                    "fallback_candidates": fallbacks,
                }

    duration = time.perf_counter() - start_time
    pipeline_duration.labels(stage="decode").observe(duration)
    return assignments


def decode_all_documents(
    none_bias: float | None = None,
    enable_pruning: bool = True,
    expect_models: bool = False,
) -> dict[str, dict[str, Any]]:
    """Decode all documents in the index.

    Args:
        none_bias: Cost for NONE assignment. If None, uses Config.DECODER_NONE_BIAS.
        enable_pruning: Whether to apply XGBoost-based candidate pruning before
                        decoding. This addresses O(n^3) Hungarian algorithm scaling
                        for large documents. Requires trained models.
        expect_models: Whether to raise an error if models fail to load when
                       manifest exists. Used in retrain endpoint to catch
                       silent failures.

    Returns:
        Dict mapping sha256 to field assignments
    """
    global _LAST_DECODE_MODEL_STATE

    if none_bias is None:
        none_bias = Config.DECODER_NONE_BIAS

    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        logger.info("decode_all_documents_no_docs")
        return {}

    results = {}

    # Try to load models for pruning (if enabled)
    pruning_model = None
    pruning_feature_names = None
    pruning_ranker_models: dict[str, Any] | None = None

    if enable_pruning:
        loaded_models = maybe_load_model_v1(expect_models=expect_models)
        model_fields = (
            list(loaded_models.get("models", {}).keys()) if loaded_models else []
        )
        model_type = (
            loaded_models.get("model_type", "none") if loaded_models else "none"
        )
        _LAST_DECODE_MODEL_STATE = {
            "models_loaded": loaded_models is not None,
            "model_fields": model_fields,
            "model_type": model_type,
        }
        if loaded_models:
            model_type = loaded_models.get("model_type", "classifier")
            if model_type == "ranker":
                # Ranker models use predict() scores (unbounded) instead of
                # predict_proba(). We prune via percentile thresholding:
                # candidates below the 10th percentile of scores are pruned.
                pruning_ranker_models = loaded_models.get("models", {})
                if pruning_ranker_models:
                    logger.info(
                        "pruning_enabled_ranker_models",
                        n_fields=len(pruning_ranker_models),
                    )
                else:
                    logger.info("pruning_disabled_no_models")
            else:
                # Legacy classifier path
                models_dict = loaded_models.get("models", {})
                if models_dict:
                    first_field = next(iter(models_dict))
                    model_info = models_dict[first_field]
                    pruning_model = model_info["model"]
                    pruning_feature_names = model_info.get("feature_names")
                    logger.info("pruning_enabled", field=first_field)
                else:
                    logger.info("pruning_disabled_no_models")
        else:
            logger.info("pruning_disabled_load_failed")

    logger.info(
        "decode_all_documents_start",
        doc_count=len(indexed_docs),
        none_bias=none_bias,
    )

    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info["sha256"]

        try:
            # Load candidates
            candidates_df = candidates.get_document_candidates(sha256)
            original_count = len(candidates_df)

            # Apply pruning if enabled and model is available
            if pruning_model is not None and not candidates_df.empty:
                candidates_df = candidates.prune_candidates(
                    candidates_df,
                    pruning_model,
                    threshold=Config.PRUNING_THRESHOLD,
                    feature_names=pruning_feature_names,
                    max_candidates=Config.PRUNING_MAX_CANDIDATES,
                )
                logger.debug(
                    "candidates_pruned",
                    sha256=sha256[:16],
                    original_count=original_count,
                    pruned_count=len(candidates_df),
                )
            elif pruning_ranker_models and not candidates_df.empty:
                candidates_df = _prune_with_ranker(
                    candidates_df,
                    pruning_ranker_models,
                    max_candidates=Config.PRUNING_MAX_CANDIDATES,
                )
                logger.debug(
                    "candidates_pruned_ranker",
                    sha256=sha256[:16],
                    original_count=original_count,
                    pruned_count=len(candidates_df),
                )

            # Decode with (potentially pruned) candidates
            assignments = decode_document(
                sha256, none_bias, candidates_df=candidates_df
            )
            results[sha256] = assignments

            # Count assignments
            candidate_count = sum(
                1 for a in assignments.values() if a["assignment_type"] == "CANDIDATE"
            )
            none_count = sum(
                1 for a in assignments.values() if a["assignment_type"] == "NONE"
            )

            logger.debug(
                "document_decoded",
                sha256=sha256[:16],
                candidate_count=candidate_count,
                none_count=none_count,
            )

        except (ValueError, TypeError, KeyError) as e:
            # Data validation or missing key errors during decoding
            pipeline_errors.labels(stage="decoder").inc()
            logger.error(
                "decode_document_failed",
                sha256=sha256[:16],
                error_type=type(e).__name__,
                reason=str(e),
            )
            # Create default NONE assignments using schema fields
            try:
                schema_obj = utils.load_contract_schema()
                schema_fields = schema_obj.get("fields", [])
                assignments = {}
                for field in schema_fields:
                    assignments[field] = {
                        "assignment_type": "NONE",
                        "candidate_index": None,
                        "cost": none_bias,
                        "field": field,
                        "used_ml_model": False,
                        "ml_probability": None,
                    }
                results[sha256] = assignments
            except (FileNotFoundError, KeyError, ValueError) as schema_error:
                logger.error(
                    "fallback_assignments_failed",
                    sha256=sha256[:16],
                    error_type=type(schema_error).__name__,
                    reason=str(schema_error),
                )
                results[sha256] = {}
        except DecodingError as e:
            # Typed decoding errors from within the pipeline
            pipeline_errors.labels(stage="decoder").inc()
            logger.error(
                "decode_document_failed",
                sha256=sha256[:16],
                error_type="DecodingError",
                reason=e.reason if hasattr(e, "reason") else str(e),
            )
            results[sha256] = {}
        except OSError as e:
            # File system errors (candidate file not readable)
            pipeline_errors.labels(stage="decoder").inc()
            logger.error(
                "decode_document_failed",
                sha256=sha256[:16],
                error_type=type(e).__name__,
                reason=str(e),
            )
            results[sha256] = {}

    return results
