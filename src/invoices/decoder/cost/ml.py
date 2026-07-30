"""ML cost dispatch: XGBRanker and legacy classifier paths."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ...exceptions import ContractMismatchError, RecoverableError
from ...feature_prep import prepare_candidate_features
from ...logging import get_logger
from ...schemas import CandidatesDF
from .heuristic import compute_weak_prior_cost

logger = get_logger(__name__)


# Required structural columns derived from CandidatesDF at import time.
# Checked cheaply (set lookup) at compute_ml_cost_with_prob entry — no
# Pandera full-validation on the hot path.
_REQUIRED_CANDIDATE_COLS: frozenset[str] = frozenset(
    CandidatesDF.to_schema().columns.keys()
)


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
    field: str,
    candidate: dict[str, Any],
    loaded_models: dict[str, Any],
    ml_score_weight: float = 0.7,
    bootstrap_ml_score_weight: float = 0.3,
    heuristic_cost: float | None = None,
) -> tuple[float, float | None]:
    """
    Compute ML-based cost using trained XGBRanker model.

    The ranker returns relevance scores (higher = more relevant).
    We convert to cost by: cost = 1 - sigmoid(score / scale)

    Args:
        field: Field name
        candidate: Candidate dictionary
        loaded_models: Loaded models from load_models()

    Returns:
        Tuple of (cost, score) where:
        - cost: ML cost (lower = better match), range [0, 1]
        - score: Raw ranker score, or None if fallback used
    """

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
        # Extract the canonical FEATURE_COLUMNS set, then wrap as DataFrame for ranker
        features = prepare_candidate_features(candidate)
        candidate_df = pl.DataFrame([features])

        # Get ranking score (higher = more relevant)
        scores = ranker.predict(candidate_df)
        score = float(scores[0]) if len(scores) > 0 else 0.0

        # Convert score to cost using sigmoid normalization
        # Score can be any real number; sigmoid maps to (0, 1)
        # Then cost = 1 - sigmoid(score) so higher score = lower cost
        # Use reduced weight for bootstrap models (less influence until
        # more data accumulates)
        manifest = loaded_models.get("manifest")
        is_bootstrap = manifest.bootstrap_mode if manifest is not None else False
        weight = bootstrap_ml_score_weight if is_bootstrap else ml_score_weight

        # Sigmoid with scaling: large positive scores → cost near 0
        # Negative scores → cost near 1
        sigmoid_score = 1.0 / (1.0 + np.exp(-score * weight))
        ml_cost = 1.0 - sigmoid_score

        # Cap bootstrap cost floor so confidence can't exceed ~0.80
        if is_bootstrap:
            ml_cost = max(ml_cost, 0.20)

        # ML-heuristic divergence: how much the model disagrees with
        # structural signals.  build_cost_matrix passes the pre-computed
        # heuristic cost explicitly so we don't re-call
        # compute_weak_prior_cost (which is expensive and has side effects).
        if heuristic_cost is not None:
            ml_heuristic_divergence = min(1.0, abs(ml_cost - heuristic_cost))
            candidate[f"_signal_disagreement:{field}"] = ml_heuristic_divergence

        if math.isnan(ml_cost):
            logger.warning(
                "ranker_cost_nan",
                field=field,
                score=score,
                fallback="weak_prior_cost",
            )
            return compute_weak_prior_cost(field, candidate), None

        return max(0.0, min(1.0, ml_cost)), score

    except RecoverableError as e:
        logger.warning(
            "ranker_cost_computation_failed",
            field=field,
            error_type=type(e).__name__,
            reason=str(e),
            fallback="weak_prior_cost",
        )
        return compute_weak_prior_cost(field, candidate), None


def compute_ml_cost_with_prob(
    field: str,
    candidate: dict[str, Any],
    loaded_models: dict[str, Any],
    ml_score_weight: float = 0.7,
    bootstrap_ml_score_weight: float = 0.3,
    heuristic_cost: float | None = None,
) -> tuple[float, float | None]:
    """
    Compute ML-based cost and probability using trained model.

    Automatically handles both XGBRanker and XGBClassifier models based on
    the model_type in loaded_models.

    Args:
        field: Field name
        candidate: Candidate dictionary
        loaded_models: Loaded models from load_models()
        ml_score_weight: Blend weight for ML ranker scores (non-bootstrap).
        bootstrap_ml_score_weight: Blend weight for ML ranker scores (bootstrap mode).

    Returns:
        Tuple of (cost, probability/score) where:
        - cost: ML cost (lower = better match), range [0, 1]
        - probability: Raw model probability/score, or None if fallback used
    """
    # ── Entry validation (hot-path: column-presence only, no value checks) ──
    # candidate is a dict; CandidatesDF.to_schema() gives the required column
    # names without running any value-range or dtype coercion.
    candidate_keys = set(candidate.keys())
    missing = _REQUIRED_CANDIDATE_COLS - candidate_keys
    if missing:
        first_missing = next(iter(sorted(missing)))
        logger.error(
            "candidate_contract_mismatch",
            missing_columns=sorted(missing),
            source="cost.compute_ml_cost_with_prob candidate",
        )
        raise ContractMismatchError(
            field=first_missing,
            expected="present",
            actual="missing",
            source="cost.compute_ml_cost_with_prob candidate",
        )

    # Check model type to dispatch to appropriate handler
    model_type = loaded_models.get("model_type", "classifier")

    if model_type == "ranker":
        return compute_ranker_cost(
            field,
            candidate,
            loaded_models,
            ml_score_weight=ml_score_weight,
            bootstrap_ml_score_weight=bootstrap_ml_score_weight,
            heuristic_cost=heuristic_cost,
        )

    # Legacy classifier path
    models_dict = loaded_models.get("models", {})

    if field not in models_dict:
        # No model for this field, fall back to weak prior
        return compute_weak_prior_cost(field, candidate), None

    model_info = models_dict[field]
    model = model_info.get("model")

    if model is None:
        return compute_weak_prior_cost(field, candidate), None

    from invoices.features import feature_columns as _get_feature_columns

    feature_names = model_info.get("feature_names", _get_feature_columns())

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
