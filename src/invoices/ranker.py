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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .config import Config
from .feature_prep import prepare_features_dataframe
from .logging import get_logger

if TYPE_CHECKING:
    from .storage.base import StorageBackend

logger = get_logger(__name__)

# =============================================================================
# RANKER CONFIGURATION
# =============================================================================
# XGBoost parameters for learning-to-rank with pairwise objective.
# Centralized in Config.XGBOOST_RANKER_PARAMS — this alias preserves
# backward compatibility for imports (e.g., tests).

RANKER_PARAMS: dict[str, Any] = Config.XGBOOST_RANKER_PARAMS

# Model version for tracking
RANKER_VERSION = "1.0.0"


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns suitable for XGBoost.

    Filters out ID columns, grouping columns, and label columns.
    Only keeps numeric columns that can be used as ML features.

    Args:
        df: DataFrame to extract feature columns from.

    Returns:
        List of numeric feature column names.
    """
    # Get all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Explicitly exclude ID/grouping/label columns
    exclude: set[str] = {
        "candidate_id",
        "doc_id",
        "sha256",
        "page_idx",
        "token_idx",
        "label",
        "target_field_encoded",
    }

    return [c for c in numeric_cols if c not in exclude]


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
        train_df: pd.DataFrame,
        label_column: str = "label",
        group_column: str = "doc_id",
        field_column: str = "target_field",
        validation_df: pd.DataFrame | None = None,
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
                "XGBoost not installed. Run: pip install 'xgboost>=2.0.0,<2.1'"
            ) from e

        # Validate input
        if train_df.empty:
            raise ValueError("Training DataFrame is empty")

        required_cols = {label_column, group_column}
        missing = required_cols - set(train_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Get feature columns - prefer Config's canonical features if available
        config_features = Config.get_feature_columns()
        available_features = [c for c in config_features if c in train_df.columns]

        if not available_features:
            # Fall back to detecting numeric columns
            available_features = get_numeric_feature_columns(train_df)

        if not available_features:
            raise ValueError("No numeric feature columns found in training data")

        self.feature_columns = available_features
        logger.info(
            "training_ranker",
            n_features=len(self.feature_columns),
            features_sample=self.feature_columns[:5],
        )

        # Prepare features and labels
        X = train_df[self.feature_columns].values.astype(np.float32)
        y = train_df[label_column].values.astype(np.float32)

        # Handle NaN/inf values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute group sizes for XGBRanker
        # Groups are (doc_id, target_field) if field_column exists
        if field_column in train_df.columns:
            group_cols = [group_column, field_column]
        else:
            group_cols = [group_column]

        # Sort by group columns to ensure consistent grouping
        sort_indices = np.lexsort([train_df[c].values for c in reversed(group_cols)])
        X = X[sort_indices]
        y = y[sort_indices]
        sorted_df = train_df.iloc[sort_indices]

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

        # Early stopping on validation NDCG if validation set provided
        if validation_df is not None and not validation_df.empty:
            X_val, y_val, val_group_sizes = self._prepare_validation_data(
                validation_df, label_column, group_cols
            )
            eval_set = [(X_val, y_val)]
            self.model.fit(
                X,
                y,
                group=group_sizes,
                eval_set=eval_set,
                eval_group=[val_group_sizes],
                early_stopping_rounds=10,
                verbose=False,
            )
        else:
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
        if validation_df is not None and not validation_df.empty:
            val_metrics = self._compute_validation_metrics(
                validation_df, label_column, group_cols
            )
            metrics.update(val_metrics)

        logger.info("training_complete", metrics=metrics)
        return metrics

    def predict(
        self,
        candidates_df: pd.DataFrame,
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

        if candidates_df.empty:
            return np.array([])

        if self.model is None:
            # Fallback to heuristic: use total_score from candidates
            return self._fallback_predict(candidates_df)

        # Normalize columns (handles candidates.py name mismatches)
        prepared_df = prepare_features_dataframe(candidates_df)

        # Extract features as numpy array
        X = prepared_df.values.astype(np.float32)

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
        candidates_df: pd.DataFrame,
        field_name: str,
    ) -> pd.DataFrame:
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
        if candidates_df.empty:
            return candidates_df.copy()

        df = candidates_df.copy()

        # Score candidates
        scores = self.predict(df)
        df["ranker_score"] = scores

        # Sort by score descending
        return df.sort_values("ranker_score", ascending=False)

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
        with open(metadata_file, "w", encoding="utf-8") as f:
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
                "XGBoost not installed. Run: pip install 'xgboost>=2.0.0,<2.1'"
            ) from e

        path = Path(path)
        model_file = path / "model.json"
        metadata_file = path / "metadata.json"

        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        # Load metadata
        with open(metadata_file, encoding="utf-8") as f:
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

    def save_to_storage(
        self,
        storage: StorageBackend,
        prefix: str,
    ) -> None:
        """Save model and metadata to a storage backend.

        Creates files at:
        - {prefix}/model.json: The XGBoost model
        - {prefix}/metadata.json: Feature columns and training metadata

        Args:
            storage: StorageBackend instance (local or blob).
            prefix: Path prefix (e.g., "models/TotalAmount").

        Raises:
            ValueError: If no model has been trained.
            StorageError: If write fails.
        """
        from .storage import StorageBackend  # noqa: F401 - for type checking

        if self.model is None:
            raise ValueError("No model to save - train first")

        import tempfile

        # Normalize prefix
        prefix = prefix.rstrip("/")

        # XGBoost requires filesystem paths, so use temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = Path(tmpdir) / "model.json"
            self.model.save_model(str(tmp_model))

            # Upload model.json
            model_bytes = tmp_model.read_bytes()
            storage.write_bytes(f"{prefix}/model.json", model_bytes)

        # Create and upload metadata
        metadata = self._build_metadata()

        metadata_json = json.dumps(metadata, indent=2, sort_keys=True)
        storage.write_text(f"{prefix}/metadata.json", metadata_json)

        logger.info(
            "ranker_saved_to_storage",
            prefix=prefix,
            n_features=len(self.feature_columns),
        )

    def load_from_storage(
        self,
        storage: StorageBackend,
        prefix: str,
    ) -> None:
        """Load model and metadata from a storage backend.

        Args:
            storage: StorageBackend instance (local or blob).
            prefix: Path prefix (e.g., "models/TotalAmount").

        Raises:
            StorageError: If files don't exist or read fails.
            ValueError: If metadata is invalid.
        """
        from .storage import StorageBackend  # noqa: F401 - for type checking

        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Run: pip install 'xgboost>=2.0.0,<2.1'"
            ) from e

        import tempfile

        # Normalize prefix
        prefix = prefix.rstrip("/")

        # Load metadata first (lightweight, validates existence)
        metadata_json = storage.read_text(f"{prefix}/metadata.json")
        metadata = json.loads(metadata_json)

        self.feature_columns = metadata.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("No feature columns in metadata")

        # XGBoost requires filesystem paths, so download to temp file
        model_bytes = storage.read_bytes(f"{prefix}/model.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = Path(tmpdir) / "model.json"
            tmp_model.write_bytes(model_bytes)

            self.model = xgb.XGBRanker(**self._params)
            self.model.load_model(str(tmp_model))

        logger.info(
            "ranker_loaded_from_storage",
            prefix=prefix,
            version=metadata.get("version", "unknown"),
            n_features=len(self.feature_columns),
        )

    def save_to_mlflow(self, run_id: str, field_name: str) -> None:
        """Save ranker to MLflow artifacts (avoids mlflow.xgboost to preserve XGBRanker type)."""
        if self.model is None:
            raise ValueError("No model to save - train first")

        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "MLflow not installed. Run: pip install 'mlflow>=2.10.0,<3.0'"
            ) from e

        import tempfile

        artifact_path = f"models/{field_name}"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_model = Path(tmpdir) / "model.json"
            self.model.save_model(str(tmp_model))

            metadata = self._build_metadata(field_name=field_name)
            tmp_meta = Path(tmpdir) / "metadata.json"
            tmp_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True))

            active_run = mlflow.active_run()
            if active_run and active_run.info.run_id == run_id:
                mlflow.log_artifacts(str(tmpdir), artifact_path=artifact_path)
            else:
                with mlflow.start_run(run_id=run_id):
                    mlflow.log_artifacts(str(tmpdir), artifact_path=artifact_path)

        logger.info(
            "ranker_saved_to_mlflow",
            run_id=run_id,
            field_name=field_name,
            n_features=len(self.feature_columns),
        )

    @classmethod
    def from_mlflow(
        cls,
        field_name: str,
        *,
        alias: str = "production",
        tracking_uri: str | None = None,
    ) -> InvoiceFieldRanker:
        """Load ranker from MLflow registry by alias (default: 'production')."""
        try:
            import mlflow  # type: ignore[import-not-found]
            from mlflow.exceptions import (  # type: ignore[import-not-found]
                MlflowException,
            )
        except ImportError as e:
            raise ImportError(
                "MLflow not installed. Run: pip install 'mlflow>=2.10.0,<3.0'"
            ) from e

        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "XGBoost not installed. Run: pip install 'xgboost>=2.0.0,<2.1'"
            ) from e

        from .exceptions import ModelNotFoundError

        uri = tracking_uri or Config.MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(uri)

        model_name = f"{Config.MLFLOW_MODEL_PREFIX}-{field_name}"

        try:
            mv = mlflow.MlflowClient().get_model_version_by_alias(model_name, alias)
        except MlflowException as e:
            raise ModelNotFoundError(
                model_id=model_name,
                path=f"mlflow://{model_name}@{alias}",
            ) from e

        import tempfile

        artifact_uri = f"runs:/{mv.run_id}/models/{field_name}"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=artifact_uri, dst_path=tmpdir
            )
            local_path = Path(local_dir)

            meta_file = local_path / "metadata.json"
            if not meta_file.exists():
                raise ModelNotFoundError(
                    model_id=model_name,
                    path=str(meta_file),
                )
            metadata = json.loads(meta_file.read_text())

            model_file = local_path / "model.json"
            if not model_file.exists():
                raise ModelNotFoundError(
                    model_id=model_name,
                    path=str(model_file),
                )

            ranker = cls()
            ranker.feature_columns = metadata.get("feature_columns", [])
            if not ranker.feature_columns:
                raise ValueError(
                    f"No feature columns in MLflow metadata for {model_name}"
                )
            ranker.model = xgb.XGBRanker(**ranker._params)
            ranker.model.load_model(str(model_file))

        logger.info(
            "ranker_loaded_from_mlflow",
            model_name=model_name,
            alias=alias,
            run_id=mv.run_id,
            version=metadata.get("version", "unknown"),
            n_features=len(ranker.feature_columns),
        )
        return ranker

    def _compute_group_sizes(
        self, df: pd.DataFrame, group_cols: list[str]
    ) -> np.ndarray:
        """Compute group sizes for XGBRanker.

        Args:
            df: DataFrame with group columns.
            group_cols: List of columns defining groups.

        Returns:
            Array of group sizes.
        """
        # Group by specified columns and count
        group_sizes = df.groupby(group_cols, sort=False).size().values
        return group_sizes.astype(np.int32)  # type: ignore[no-any-return]

    def _prepare_validation_data(
        self,
        val_df: pd.DataFrame,
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
        X_val = val_df[self.feature_columns].values.astype(np.float32)
        y_val = val_df[label_column].values.astype(np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        # Sort by group columns
        sort_indices = np.lexsort([val_df[c].values for c in reversed(group_cols)])
        X_val = X_val[sort_indices]
        y_val = y_val[sort_indices]
        sorted_val_df = val_df.iloc[sort_indices]

        val_group_sizes = self._compute_group_sizes(sorted_val_df, group_cols)

        return X_val, y_val, val_group_sizes

    def _fallback_predict(self, candidates_df: pd.DataFrame) -> np.ndarray:
        """Fallback prediction using heuristic scoring.

        Uses the total_score column if available, otherwise returns zeros.

        Args:
            candidates_df: DataFrame with candidates.

        Returns:
            Array of fallback scores.
        """
        if "total_score" in candidates_df.columns:
            scores = candidates_df["total_score"].values.astype(np.float64)
            # Handle NaN values
            scores = np.nan_to_num(scores, nan=0.0)
            return scores  # type: ignore[no-any-return]

        # No heuristic score available
        logger.debug(
            "fallback_predict_no_total_score",
            n_candidates=len(candidates_df),
        )
        return np.zeros(len(candidates_df), dtype=np.float64)

    def _compute_validation_metrics(
        self,
        val_df: pd.DataFrame,
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
        if val_df.empty or self.model is None:
            return {}

        metrics: dict[str, float] = {}

        try:
            # Get predictions for validation data
            scores = self.predict(val_df)
            labels = val_df[label_column].values

            # Add scores and labels to DataFrame for groupby
            df_for_grouping = val_df.copy()
            df_for_grouping["__scores"] = scores
            df_for_grouping["__labels"] = labels

            ndcg_at_1_scores: list[float] = []
            mrr_scores: list[float] = []

            # Use pandas groupby for cleaner iteration over groups
            for _, group_df in df_for_grouping.groupby(group_cols, sort=False):
                group_scores = group_df["__scores"].values
                group_labels = group_df["__labels"].values

                if len(group_scores) == 0:
                    continue

                # Find best prediction
                best_idx = np.argmax(group_scores)

                # NDCG@1: 1 if top prediction is correct, 0 otherwise
                if group_labels[best_idx] == 1:
                    ndcg_at_1_scores.append(1.0)
                else:
                    ndcg_at_1_scores.append(0.0)

                # MRR: 1/rank of first correct answer
                if np.any(group_labels == 1):
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
