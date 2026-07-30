"""XGBoost-based ranker for invoice field extraction.

This module implements a learning-to-rank approach using XGBRanker with
the pairwise objective to replace heuristic scoring with learned field
assignment. The ranker learns to score candidates based on how likely
they are to be the correct value for a given field.

Key Design Principles:
- Deterministic: Fixed seeds, no random sampling, single-threaded
- Graceful fallback: Uses heuristic scoring when no model is available
- Feature alignment: Uses same features as train.py and decoder.py
- Group-aware: Ranks candidates within (doc_id, field) groups

Usage:
    from invoices.ranker import InvoiceFieldRanker

    # Train a ranker
    ranker = InvoiceFieldRanker()
    metrics = ranker.train(training_df)
    ranker.save(Path("data/models/ranker"))

    # Score candidates
    ranker = InvoiceFieldRanker(model_path=Path("data/models/ranker"))
    scores = ranker.predict(candidates_df)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .exceptions import ConfigurationError
from .feature_prep import prepare_features_dataframe
from .logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# RANKER CONFIGURATION
# =============================================================================
# XGBoost parameters for learning-to-rank with pairwise objective.
# Inlined from _XGB_BASE + rank objective (relocated from PipelineConfig).

RANKER_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "seed": 42,
    "n_jobs": 1,
    "verbosity": 0,
    "objective": "rank:pairwise",
}

# Model version for tracking
RANKER_VERSION = "1.0.0"


class InvoiceFieldRanker:
    """XGBoost-based ranker for invoice field extraction.

    This class implements a learning-to-rank approach using XGBRanker to
    score candidates for field assignment. It can be trained on labeled
    data and used to predict relevance scores for new candidates.

    Attributes:
        random_state: Seed for reproducibility.
        model: The trained XGBRanker model, or None if untrained.
        feature_columns: List of feature column names used by the model.
        model_path: Path to the loaded model, if any.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        random_state: int = 42,
    ) -> None:
        """Initialize the ranker.

        Args:
            model_path: Path to saved model directory (None for untrained).
            random_state: Seed for reproducibility.
        """
        self.random_state = random_state
        self.model: Any = None  # xgb.XGBRanker when loaded
        self.feature_columns: list[str] = []
        self.model_path: Path | None = None
        self._params = RANKER_PARAMS.copy()
        self._params["random_state"] = random_state
        self._params["seed"] = random_state

        # Load model if path provided
        if model_path is not None:
            self.load(model_path)

    def train(
        self,
        train_df: pl.DataFrame,
        label_column: str = "label",
        group_column: str = "doc_id",
        field_column: str = "target_field",
        validation_df: pl.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train the ranker on labeled data.

        Uses XGBRanker with pairwise objective to learn candidate scoring.
        Groups are defined by (doc_id, target_field) so that candidates
        for the same field in the same document are ranked together.

        Args:
            train_df: DataFrame with features, labels, and grouping.
                Must contain:
                - Feature columns (numeric)
                - label_column: Relevance labels (0/1)
                - group_column: Document identifier
                - field_column: Target field type
            label_column: Column with relevance labels (0/1).
            group_column: Column for document grouping.
            field_column: Column indicating target field type.
            validation_df: Optional validation DataFrame for metrics.

        Returns:
            Training metrics dictionary with:
            - n_groups: Number of ranking groups
            - n_samples: Total number of samples
            - n_positive: Number of positive labels
            - n_features: Number of features used

        Raises:
            ImportError: If XGBoost is not installed.
            ValueError: If training data is invalid.
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Run: pip install 'xgboost>=2.1'"
            ) from e

        # Validate input
        if train_df.is_empty():
            raise ValueError("Training DataFrame is empty")

        required_cols = {label_column, group_column}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get feature columns - prefer Config's canonical features if available
        from invoices.features import feature_columns as _get_feature_columns

        config_features = _get_feature_columns()
        available_features = [c for c in config_features if c in train_df.columns]

        if not available_features:
            missing_features = sorted(
                c for c in config_features if c not in train_df.columns
            )
            extra_features = sorted(
                c for c in train_df.columns if c not in config_features
            )
            extras_display = extra_features[:20]
            extras_suffix = (
                f" … and {len(extra_features) - 20} more"
                if len(extra_features) > 20
                else ""
            )
            raise ConfigurationError(
                f"Feature column mismatch: config defines {len(config_features)} features "
                f"but none overlap with training data columns. "
                f"Expected by config, absent from train_df: {missing_features}. "
                f"Unexpected in train_df: {extras_display}{extras_suffix}."
            )

        self.feature_columns = available_features
        logger.info(
            "training_ranker",
            n_features=len(self.feature_columns),
            features_sample=self.feature_columns[:5],
        )

        # Prepare features and labels
        X = train_df[self.feature_columns].to_numpy().astype(np.float32)
        y = train_df[label_column].to_numpy().astype(np.float32)

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute group sizes for XGBRanker
        # Groups are (doc_id, target_field) if field_column exists
        if field_column in train_df.columns:
            group_cols = [group_column, field_column]
        else:
            group_cols = [group_column]

        # Sort by group columns to ensure consistent grouping
        sort_indices = np.lexsort(
            [train_df[c].to_numpy() for c in reversed(group_cols)]
        )
        X = X[sort_indices]
        y = y[sort_indices]
        sorted_df = train_df[sort_indices]

        # Compute group sizes
        group_sizes = self._compute_group_sizes(sorted_df, group_cols)

        # Validate group sizes
        if len(group_sizes) == 0:
            raise ValueError("No valid groups found in training data")

        if group_sizes.sum() != len(X):
            raise ValueError(
                f"Group sizes sum ({group_sizes.sum()}) != sample count ({len(X)})"
            )

        # Compute class imbalance for scale_pos_weight
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        # Initialize and train model
        params_with_scale = self._params.copy()
        params_with_scale["scale_pos_weight"] = scale_pos_weight
        self.model = xgb.XGBRanker(**params_with_scale)

        logger.info(
            "training_xgb_ranker",
            n_samples=len(X),
            n_groups=len(group_sizes),
            n_positive=int(y.sum()),
            scale_pos_weight=scale_pos_weight,
        )

        # Fit on the (training) split. xgboost 3.x removed early_stopping_rounds
        # from fit() (it is a constructor arg), and early stopping on the tiny
        # validation set in this small-label regime underfits — so we do a plain
        # fit and compute HONEST held-out metrics separately below on validation_df
        # via _compute_validation_metrics. (Fixes the TypeError that aborted retrain
        # whenever a validation split was active.)
        self.model.fit(X, y, group=group_sizes)

        # Compute metrics
        metrics: dict[str, float] = {
            "n_groups": float(len(group_sizes)),
            "n_samples": float(len(X)),
            "n_positive": float(y.sum()),
            "n_features": float(len(self.feature_columns)),
            "scale_pos_weight": scale_pos_weight,
        }

        # Compute NDCG on validation set if provided
        if validation_df is not None and not validation_df.is_empty():
            val_metrics = self._compute_validation_metrics(
                validation_df, label_column, group_cols
            )
            metrics.update(val_metrics)

        logger.info("training_complete", metrics=metrics)
        return metrics

    def predict(
        self,
        candidates_df: pl.DataFrame,
        group_column: str = "doc_id",
    ) -> np.ndarray:
        """Score candidates for ranking.

        If no model is trained, falls back to heuristic scoring using
        the total_score column if available.

        Args:
            candidates_df: DataFrame with features.
            group_column: Column for document grouping (unused but kept
                for API consistency).

        Returns:
            Array of scores (higher = more relevant).
        """
        import time

        start_time = time.time()
        n_candidates = len(candidates_df)

        if candidates_df.is_empty():
            return np.array([])

        if self.model is None:
            # Fallback to heuristic: use total_score from candidates
            return self._fallback_predict(candidates_df)

        # Normalize columns (handles candidates.py name mismatches)
        prepared_df = prepare_features_dataframe(candidates_df)

        # Extract features as numpy array
        X = prepared_df.to_numpy().astype(np.float32)

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Get predictions
        try:
            scores = self.model.predict(X)
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                "ranker_prediction_complete",
                n_candidates=n_candidates,
                n_features=len(self.feature_columns),
                duration_ms=round(duration_ms, 2),
                model_loaded=True,
            )
            return scores.astype(np.float64)  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(
                "prediction_failed",
                error=str(e),
                fallback="heuristic",
            )
            return self._fallback_predict(candidates_df)

    def rank_candidates_for_field(
        self,
        candidates_df: pl.DataFrame,
        field_name: str,
    ) -> pl.DataFrame:
        """Rank candidates for a specific field type.

        Scores all candidates and returns them sorted by predicted
        relevance (highest first).

        Args:
            candidates_df: DataFrame with candidate features.
            field_name: The field type to rank for (currently unused
                but reserved for field-specific models).

        Returns:
            DataFrame sorted by ranker_score (highest first).
        """
        if candidates_df.is_empty():
            return candidates_df

        # Score candidates
        scores = self.predict(candidates_df)
        df = candidates_df.with_columns(pl.Series("ranker_score", scores))

        # Sort by score descending
        return df.sort("ranker_score", descending=True)

    def _build_metadata(self, **extra: object) -> dict:
        """Build metadata dict for model serialization."""
        meta = {
            "feature_columns": self.feature_columns,
            "random_state": self.random_state,
            "version": RANKER_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "params": {k: v for k, v in self._params.items() if k != "verbosity"},
        }
        meta.update(extra)
        return meta

    def save(self, path: Path) -> None:
        """Save model and feature columns to disk.

        Creates a directory containing:
        - model.json: The XGBoost model in JSON format
        - metadata.json: Feature columns and training metadata

        Args:
            path: Directory path to save model to.

        Raises:
            ValueError: If no model has been trained.
        """
        if self.model is None:
            raise ValueError("No model to save - train first")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost model
        model_file = path / "model.json"
        self.model.save_model(str(model_file))

        # Save metadata
        metadata = self._build_metadata()

        metadata_file = path / "metadata.json"
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

        logger.info(
            "ranker_saved",
            path=str(path),
            n_features=len(self.feature_columns),
        )

    def load(self, path: Path) -> None:
        """Load model and feature columns from disk.

        Args:
            path: Directory path to load model from.

        Raises:
            FileNotFoundError: If model files don't exist.
            ValueError: If metadata is invalid.
        """
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Run: pip install 'xgboost>=2.1'"
            ) from e

        path = Path(path)
        model_file = path / "model.json"
        metadata_file = path / "metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load metadata
        with metadata_file.open(encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_columns = metadata.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("No feature columns in metadata")

        # Load model
        self.model = xgb.XGBRanker(**self._params)
        self.model.load_model(str(model_file))
        self.model_path = path

        logger.info(
            "ranker_loaded",
            path=str(path),
            version=metadata.get("version", "unknown"),
            n_features=len(self.feature_columns),
        )

    def _compute_group_sizes(
        self, df: pl.DataFrame, group_cols: list[str]
    ) -> np.ndarray:
        """Compute group sizes for XGBRanker.

        Args:
            df: DataFrame with group columns.
            group_cols: List of columns defining groups.

        Returns:
            Array of group sizes.
        """
        # Group by specified columns and count, maintaining sort order aligned with lexsort
        group_sizes = (
            df.group_by(group_cols, maintain_order=True)
            .len()
            .sort(by=group_cols)["len"]
            .to_numpy()
        )
        return group_sizes.astype(np.int32)  # type: ignore[no-any-return]

    def _prepare_validation_data(
        self,
        val_df: pl.DataFrame,
        label_column: str,
        group_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare validation data for early stopping.

        Args:
            val_df: Validation DataFrame.
            label_column: Column with relevance labels.
            group_cols: Columns defining groups.

        Returns:
            Tuple of (X_val, y_val, val_group_sizes).
        """
        X_val = val_df[self.feature_columns].to_numpy().astype(np.float32)
        y_val = val_df[label_column].to_numpy().astype(np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Sort by group columns
        sort_indices = np.lexsort([val_df[c].to_numpy() for c in reversed(group_cols)])
        X_val = X_val[sort_indices]
        y_val = y_val[sort_indices]
        sorted_val_df = val_df[sort_indices]

        val_group_sizes = self._compute_group_sizes(sorted_val_df, group_cols)

        return X_val, y_val, val_group_sizes

    def _fallback_predict(self, candidates_df: pl.DataFrame) -> np.ndarray:
        """Fallback prediction using heuristic scoring.

        Uses the total_score column if available, otherwise returns zeros.

        Args:
            candidates_df: DataFrame with candidates.

        Returns:
            Array of fallback scores.
        """
        if "total_score" in candidates_df.columns:
            scores = candidates_df["total_score"].to_numpy().astype(np.float64)
            # Handle NaN values
            return np.nan_to_num(scores, nan=0.0)  # type: ignore[no-any-return]

        # No heuristic score available
        logger.debug(
            "fallback_predict_no_total_score",
            n_candidates=len(candidates_df),
        )
        return np.zeros(len(candidates_df), dtype=np.float64)

    def _compute_validation_metrics(
        self,
        val_df: pl.DataFrame,
        label_column: str,
        group_cols: list[str],
    ) -> dict[str, float]:
        """Compute validation metrics on held-out data.

        Computes NDCG@1 (whether the top prediction is correct) and
        mean reciprocal rank.

        Args:
            val_df: Validation DataFrame.
            label_column: Column with relevance labels.
            group_cols: Columns defining groups.

        Returns:
            Dictionary with validation metrics.
        """
        if val_df.is_empty() or self.model is None:
            return {}

        metrics: dict[str, float] = {}

        try:
            # Get predictions for validation data
            # Predict on the extracted feature matrix directly (val_df is training-row
            # format with features already as self.feature_columns) — NOT via self.predict(),
            # which re-runs prepare_features_dataframe() and expects raw candidate geometry
            # (bbox_norm_x0 etc.) absent here.
            _X_val = val_df[self.feature_columns].to_numpy().astype(np.float32)
            _X_val = np.nan_to_num(_X_val, nan=0.0, posinf=0.0, neginf=0.0)
            scores = self.model.predict(_X_val)
            labels = val_df[label_column].to_numpy()

            # Add scores and labels to DataFrame for group iteration
            df_for_grouping = val_df.with_columns(
                pl.Series("__scores", scores),
                pl.Series("__labels", labels),
            )

            ndcg_at_1_scores: list[float] = []
            mrr_scores: list[float] = []

            # Iterate over groups
            for _, group_df in df_for_grouping.group_by(
                group_cols, maintain_order=True
            ):
                group_scores = group_df["__scores"].to_numpy()
                group_labels = group_df["__labels"].to_numpy()

                if len(group_scores) == 0:
                    continue

                # NDCG@1 and MRR are computed ONLY over groups that contain a
                # correct candidate. A field with no positive (absent in the doc)
                # is not a selection the ranker can get right; counting it as a 0
                # measures field sparsity, not ranker skill — which craters the
                # metric at the ~0.5% positive rate.
                if np.any(group_labels == 1):
                    best_idx = np.argmax(group_scores)
                    ndcg_at_1_scores.append(1.0 if group_labels[best_idx] == 1 else 0.0)
                    sorted_indices = np.argsort(-group_scores)
                    for rank, idx in enumerate(sorted_indices, 1):
                        if group_labels[idx] == 1:
                            mrr_scores.append(1.0 / rank)
                            break

            if ndcg_at_1_scores:
                metrics["val_ndcg_at_1"] = float(np.mean(ndcg_at_1_scores))

            if mrr_scores:
                metrics["val_mrr"] = float(np.mean(mrr_scores))

        except Exception as e:
            logger.warning(
                "validation_metrics_failed",
                error=str(e),
            )

        return metrics

    @property
    def is_trained(self) -> bool:
        """Check if the ranker has a trained model."""
        return self.model is not None
