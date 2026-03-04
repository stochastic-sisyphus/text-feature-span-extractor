"""XGBoost training module with deterministic hyperparameters and byte-stable persistence.

Uses XGBRanker (learning-to-rank) to directly optimize for ranking candidates
rather than treating it as independent binary classifications.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from . import candidates, labels, utils
from .config import Config
from .feature_prep import prepare_candidate_features
from .logging import get_logger
from .ranker import InvoiceFieldRanker

logger = get_logger(__name__)


def create_training_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix from candidates for training.

    Uses the shared prepare_candidate_features() from feature_prep.py to
    ensure training/inference alignment. Feature columns are defined in
    Config.get_feature_columns().

    Args:
        candidates_df: Candidates DataFrame

    Returns:
        Feature DataFrame with 59 columns in exact order matching decoder.py
    """
    # Get the canonical feature column order from config
    feature_columns = Config.get_feature_columns()

    features = []

    for candidate in candidates_df.itertuples(index=False):
        # Convert namedtuple to dict for consistent access
        candidate_dict = candidate._asdict()

        # Extract all 59 features
        feature_row = prepare_candidate_features(candidate_dict)
        features.append(feature_row)

    # Create DataFrame with columns in the exact order expected by the model
    result_df = pd.DataFrame(features)

    # Ensure columns are in the correct order (XGBoost is order-sensitive)
    # This also validates that we extracted all expected features
    missing_cols = set(feature_columns) - set(result_df.columns)
    if missing_cols:
        raise ValueError(
            f"Training feature extraction missing columns: {missing_cols}. "
            "This indicates a mismatch between train.py and Config.get_feature_columns()."
        )

    extra_cols = set(result_df.columns) - set(feature_columns)
    if extra_cols:
        logger.warning(
            "extra_feature_columns",
            extra_cols=list(extra_cols),
            msg="Extra columns in training features (will be ignored)",
        )

    return result_df[feature_columns]


def prepare_training_data() -> tuple[dict[str, pd.DataFrame], int, int]:
    """
    Prepare training data from aligned labels.

    Returns:
        Tuple of (field_datasets, total_docs, total_rows)
    """
    # Load aligned labels
    aligned_df = labels.load_aligned_labels()

    if aligned_df.empty:
        return {}, 0, 0

    total_docs = aligned_df["doc_id"].nunique()
    total_rows = len(aligned_df)

    # Group by field for per-field binary classification
    field_datasets = {}

    # Only use aligned rows (those with a valid candidate_idx)
    aligned_only = aligned_df[aligned_df["is_aligned"] == True].copy()  # noqa: E712

    for field_name in aligned_only["field"].unique():
        field_labels = aligned_only[aligned_only["field"] == field_name]

        # Collect all candidate features for documents with this field
        field_data = []

        for _, label_row in field_labels.iterrows():
            sha256 = label_row["sha256"]

            # Load candidates for this document
            try:
                candidates_df = candidates.get_document_candidates(sha256)
                if candidates_df.empty:
                    continue

                # Create features
                features_df = create_training_features(candidates_df)

                # Create binary labels (1 for labeled candidate, 0 for others).
                # For not_applicable/reject actions, candidate_idx is None/NaN
                # meaning NO candidate is correct — all-zeros is the right label.
                labels_array = np.zeros(len(candidates_df))
                raw_idx = label_row["candidate_idx"]
                if raw_idx is not None and not (
                    isinstance(raw_idx, float) and np.isnan(raw_idx)
                ):
                    idx = int(raw_idx)
                    if 0 <= idx < len(candidates_df):
                        labels_array[idx] = 1

                # Combine features and labels
                data_df = features_df.copy()
                data_df["label"] = labels_array
                data_df["doc_id"] = label_row["doc_id"]
                data_df["sha256"] = sha256

                field_data.append(data_df)

            except Exception as e:
                logger.warning(
                    "candidate_processing_failed",
                    sha256=sha256,
                    field=field_name,
                    error=str(e),
                )
                continue

        if field_data:
            field_datasets[field_name] = pd.concat(field_data, ignore_index=True)

    return field_datasets, total_docs, total_rows


def _augment_with_jitter(
    training_data: pd.DataFrame, n_copies: int = 2, jitter_pct: float = 0.02
) -> pd.DataFrame:
    """Create augmented copies of training data with bbox coordinate jitter.

    For bootstrap mode: perturbs bbox-related features by a small random amount
    to give the ranker more training signal from few examples.

    Args:
        training_data: Original training DataFrame
        n_copies: Number of augmented copies per sample
        jitter_pct: Maximum perturbation as fraction of value (default 2%)

    Returns:
        Concatenation of original + augmented copies
    """
    bbox_cols = [
        c
        for c in training_data.columns
        if any(
            kw in c
            for kw in ("center_x", "center_y", "width", "height", "area", "bbox")
        )
    ]
    if not bbox_cols:
        return training_data

    rng = np.random.RandomState(42)
    augmented_parts = [training_data]

    for _ in range(n_copies):
        copy_df = training_data.copy()
        for col in bbox_cols:
            if col in copy_df.columns:
                noise = rng.uniform(-jitter_pct, jitter_pct, size=len(copy_df))
                copy_df[col] = copy_df[col] * (1.0 + noise)
        augmented_parts.append(copy_df)

    return pd.concat(augmented_parts, ignore_index=True)


def train_field_ranker(
    field_name: str, training_data: pd.DataFrame
) -> dict[str, Any] | None:
    """
    Train XGBRanker model for a single field (recommended approach).

    Uses learning-to-rank with pairwise objective to directly optimize
    for ranking candidates rather than binary classification.

    When docs < BOOTSTRAP_DOC_THRESHOLD, uses bootstrap mode:
    - 2-fold stratified CV instead of LOOCV
    - Feature jitter augmentation for more training signal

    Otherwise uses leave-one-document-out cross-validation.

    Args:
        field_name: Name of field to train
        training_data: Training DataFrame with features, labels, doc_id

    Returns:
        Trained ranker info or None if insufficient data
    """
    # Check for sufficient positive examples
    pos_count = int(training_data["label"].sum())
    neg_count = int((training_data["label"] == 0).sum())

    if pos_count < Config.MIN_POSITIVE_EXAMPLES:
        logger.info(
            "insufficient_positive_examples",
            field=field_name,
            pos_count=pos_count,
            min_required=Config.MIN_POSITIVE_EXAMPLES,
        )
        return None

    # Get unique documents
    unique_docs = training_data["doc_id"].unique()
    n_docs = len(unique_docs)
    bootstrap_mode = n_docs < Config.BOOTSTRAP_DOC_THRESHOLD

    logger.info(
        "training_ranker",
        field=field_name,
        pos_count=pos_count,
        neg_count=neg_count,
        n_docs=n_docs,
        bootstrap_mode=bootstrap_mode,
    )

    # Cross-validation
    loocv_ndcg_scores: list[float] = []
    loocv_train_ndcg_scores: list[float] = []

    if n_docs > 1:
        if bootstrap_mode and n_docs >= 2:
            # Bootstrap mode: 2-fold stratified CV (split docs into 2 groups)
            rng = np.random.RandomState(42)
            shuffled_docs = rng.permutation(unique_docs)
            mid = len(shuffled_docs) // 2
            folds = [shuffled_docs[:mid], shuffled_docs[mid:]]

            for fold_idx in range(2):
                val_docs = set(folds[fold_idx])
                train_docs = set(folds[1 - fold_idx])

                fold_val_df = training_data[training_data["doc_id"].isin(val_docs)]
                fold_train_df = training_data[training_data["doc_id"].isin(train_docs)]

                if fold_train_df["label"].sum() < 1 or fold_val_df["label"].sum() < 1:
                    continue

                # Augment training data with jitter in bootstrap mode
                fold_train_df = _augment_with_jitter(fold_train_df)

                fold_ranker = InvoiceFieldRanker()
                try:
                    field_col: str | None = (
                        "target_field"
                        if "target_field" in fold_train_df.columns
                        else None
                    )
                    fold_metrics = fold_ranker.train(
                        train_df=fold_train_df,
                        validation_df=fold_val_df,
                        label_column="label",
                        group_column="doc_id",
                        field_column=field_col or "",  # type: ignore[arg-type]
                    )
                    fold_ndcg = fold_metrics.get("val_ndcg_at_1")
                    if fold_ndcg is not None:
                        loocv_ndcg_scores.append(fold_ndcg)
                except Exception as e:
                    logger.warning(
                        "bootstrap_fold_failed",
                        field=field_name,
                        fold=fold_idx,
                        error=str(e),
                    )
                    continue

            logger.info(
                "bootstrap_cv_complete",
                field=field_name,
                n_folds=len(loocv_ndcg_scores),
                n_docs=n_docs,
                mean_ndcg=float(np.mean(loocv_ndcg_scores))
                if loocv_ndcg_scores
                else None,
            )
        else:
            # Standard LOOCV
            for val_doc in unique_docs:
                fold_val_df = training_data[training_data["doc_id"] == val_doc]
                fold_train_df = training_data[training_data["doc_id"] != val_doc]

                # Skip folds where train or val has no positive examples
                if fold_train_df["label"].sum() < 1 or fold_val_df["label"].sum() < 1:
                    continue

                fold_ranker = InvoiceFieldRanker()
                try:
                    field_col_loocv: str | None = (
                        "target_field"
                        if "target_field" in fold_train_df.columns
                        else None
                    )
                    fold_metrics = fold_ranker.train(
                        train_df=fold_train_df,
                        validation_df=fold_val_df,
                        label_column="label",
                        group_column="doc_id",
                        field_column=field_col_loocv or "",  # type: ignore[arg-type]
                    )
                    fold_ndcg = fold_metrics.get("val_ndcg_at_1")
                    if fold_ndcg is not None:
                        loocv_ndcg_scores.append(fold_ndcg)

                    # Compute train accuracy for overfitting gap
                    group_cols = (
                        ["doc_id", "target_field"]
                        if "target_field" in fold_train_df.columns
                        else ["doc_id"]
                    )
                    train_metrics = fold_ranker._compute_validation_metrics(
                        fold_train_df, "label", group_cols
                    )
                    train_ndcg = train_metrics.get("val_ndcg_at_1")
                    if train_ndcg is not None:
                        loocv_train_ndcg_scores.append(train_ndcg)
                except Exception as e:
                    logger.warning(
                        "loocv_fold_failed",
                        field=field_name,
                        val_doc=val_doc,
                        error=str(e),
                    )
                    continue

            logger.info(
                "loocv_complete",
                field=field_name,
                n_folds=len(loocv_ndcg_scores),
                n_docs=n_docs,
                mean_ndcg=float(np.mean(loocv_ndcg_scores))
                if loocv_ndcg_scores
                else None,
                mean_train_ndcg=float(np.mean(loocv_train_ndcg_scores))
                if loocv_train_ndcg_scores
                else None,
            )

    # Train final model on ALL data (no holdout) for deployment
    # In bootstrap mode, augment training data with jitter
    final_training_data = (
        _augment_with_jitter(training_data) if bootstrap_mode else training_data
    )
    final_ranker = InvoiceFieldRanker()

    try:
        field_col_final: str | None = (
            "target_field" if "target_field" in final_training_data.columns else None
        )
        metrics = final_ranker.train(
            train_df=final_training_data,
            validation_df=None,
            label_column="label",
            group_column="doc_id",
            field_column=field_col_final or "",  # type: ignore[arg-type]
        )
    except Exception as e:
        logger.warning(
            "ranker_training_failed",
            field=field_name,
            error=str(e),
        )
        return None

    # Store CV scores in metrics for quality gate
    if loocv_ndcg_scores:
        metrics["loocv_ndcg_at_1"] = float(np.mean(loocv_ndcg_scores))
        metrics["loocv_std"] = float(np.std(loocv_ndcg_scores))
        metrics["loocv_n_folds"] = float(len(loocv_ndcg_scores))
        # Also set val_ndcg_at_1 so quality gate reads it
        metrics["val_ndcg_at_1"] = metrics["loocv_ndcg_at_1"]

    # Store train accuracy for overfitting gap detection
    if loocv_train_ndcg_scores:
        metrics["loocv_train_ndcg_at_1"] = float(np.mean(loocv_train_ndcg_scores))

    return {
        "ranker": final_ranker,
        "field_name": field_name,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "total_samples": len(training_data),
        "metrics": metrics,
        "bootstrap_mode": bootstrap_mode,
    }


def save_ranker_models(
    trained_rankers: dict[str, dict[str, Any]],
    n_docs: int = 0,
) -> str:
    """
    Save trained ranker models with per-field directory structure.

    Uses the configured storage backend (local or blob) based on
    Config.STORAGE_BACKEND. Models are stored under the "models/" prefix.

    Args:
        trained_rankers: Dictionary of field -> ranker_info
        n_docs: Total number of labeled documents (for bootstrap quality gate)

    Returns:
        Path or prefix to models directory
    """
    from .storage import get_storage

    storage = get_storage()

    # Determine if any model was trained in bootstrap mode
    any_bootstrap = any(
        r.get("bootstrap_mode", False) for r in trained_rankers.values()
    )

    # Create manifest for ranker models
    manifest: dict[str, Any] = {
        "model_type": "ranker",  # Distinguishes from legacy classifier
        "model_version": "v2",
        "training_timestamp": utils.get_current_utc_iso(),
        "fields": {},
        "quality_gate_passed": False,  # Will be updated after evaluation
        "bootstrap_mode": any_bootstrap,
    }

    # Save each ranker using storage backend
    for field_name, ranker_info in trained_rankers.items():
        ranker: InvoiceFieldRanker = ranker_info["ranker"]
        prefix = f"models/{field_name}"

        # Use the ranker's storage-aware save method
        ranker.save_to_storage(storage, prefix)

        metrics = ranker_info.get("metrics", {})
        manifest["fields"][field_name] = {  # type: ignore[index]
            "pos_count": ranker_info["pos_count"],
            "neg_count": ranker_info["neg_count"],
            "total_samples": ranker_info["total_samples"],
            "metrics": metrics,
            "val_ndcg_at_1": metrics.get("val_ndcg_at_1"),
            "val_mrr": metrics.get("val_mrr"),
            "loocv_ndcg_at_1": metrics.get("loocv_ndcg_at_1"),
            "loocv_std": metrics.get("loocv_std"),
            "bootstrap_mode": ranker_info.get("bootstrap_mode", False),
        }

        logger.info(
            "saved_ranker",
            field=field_name,
            prefix=prefix,
        )

    # Evaluate quality gate before saving manifest (pass n_docs for bootstrap)
    quality_passed = _evaluate_ranker_quality(manifest, n_docs=n_docs)
    manifest["quality_gate_passed"] = quality_passed

    # Save manifest
    manifest_json = json.dumps(manifest, indent=2, sort_keys=True)
    storage.write_text("models/manifest.json", manifest_json)

    logger.info(
        "saved_ranker_manifest",
        prefix="models/manifest.json",
        fields=list(trained_rankers.keys()),
        quality_gate_passed=quality_passed,
        bootstrap_mode=any_bootstrap,
    )

    # Log to MLflow if enabled
    if Config.USE_MLFLOW:
        _log_rankers_to_mlflow(trained_rankers, manifest)

    return "models/"


def _log_rankers_to_mlflow(
    trained_rankers: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Log rankers to MLflow and register with 'production' alias."""
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow_not_installed", msg="Skipping MLflow logging")
        return

    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="ranker-train") as run:
        mlflow.log_params(
            {
                "model_type": "ranker",
                "model_version": manifest.get("model_version", "v2"),
                "n_fields": len(trained_rankers),
                "fields": ",".join(sorted(trained_rankers.keys())),
            }
        )

        for field_name, ranker_info in trained_rankers.items():
            ranker: InvoiceFieldRanker = ranker_info["ranker"]

            metrics = ranker_info.get("metrics", {})
            for metric_key, metric_val in metrics.items():
                if isinstance(metric_val, (int, float)):
                    mlflow.log_metric(f"{field_name}/{metric_key}", metric_val)

            mlflow.log_metric(f"{field_name}/pos_count", ranker_info["pos_count"])
            mlflow.log_metric(f"{field_name}/neg_count", ranker_info["neg_count"])

            ranker.save_to_mlflow(run.info.run_id, field_name)

        mlflow.log_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            "models/manifest.json",
        )

        logger.info(
            "mlflow_run_logged",
            run_id=run.info.run_id,
            fields=list(trained_rankers.keys()),
        )

    client = mlflow.MlflowClient()
    for field_name in trained_rankers:
        model_name = f"{Config.MLFLOW_MODEL_PREFIX}-{field_name}"
        artifact_uri = f"runs:/{run.info.run_id}/models/{field_name}"

        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass  # Already exists

        mv = client.create_model_version(
            name=model_name,
            source=artifact_uri,
            run_id=run.info.run_id,
        )
        client.set_registered_model_alias(model_name, "production", mv.version)

        logger.info(
            "mlflow_model_registered",
            model_name=model_name,
            version=mv.version,
            alias="production",
        )


def _evaluate_ranker_quality(
    field_metrics: dict[str, Any],
    n_docs: int = 0,
) -> bool:
    """
    Evaluate if trained rankers are good enough to use in production.

    Accepts either manifest format (field_info dicts with top-level keys)
    or training_stats format (field dicts with nested "metrics" sub-dict).
    Prefers loocv_ndcg_at_1 over val_ndcg_at_1.

    When n_docs < BOOTSTRAP_DOC_THRESHOLD, uses the relaxed bootstrap
    threshold (default 0.3 vs 0.5) to allow early ranker enablement.

    Args:
        field_metrics: Dict of field_name -> metrics info. Supports two layouts:
            - Manifest: {"fields": {"F": {"loocv_ndcg_at_1": ..., "val_ndcg_at_1": ...}}}
            - Training stats: {"F": {"metrics": {"val_ndcg_at_1": ...}}}
        n_docs: Total number of labeled documents (used for bootstrap threshold)

    Returns:
        True if ranker quality passes threshold, False otherwise
    """
    # Normalize: manifest wraps fields in a "fields" key, training_stats does not
    fields = field_metrics.get("fields", field_metrics)

    val_ndcg_scores: list[float] = []
    for field_info in fields.values():
        # Handle both layouts: manifest has top-level keys, training_stats nests under "metrics"
        if isinstance(field_info, dict):
            score = (
                field_info.get("loocv_ndcg_at_1")
                or field_info.get("val_ndcg_at_1")
                or field_info.get("metrics", {}).get("val_ndcg_at_1")
            )
            if score is not None:
                val_ndcg_scores.append(score)

    if not val_ndcg_scores:
        logger.warning(
            "ranker_quality_gate_failed",
            reason="no_validation_scores",
            recommendation="Need more training data with validation split",
        )
        return False

    mean_ndcg = sum(val_ndcg_scores) / len(val_ndcg_scores)

    # Use bootstrap threshold when docs are below threshold
    bootstrap_mode = 0 < n_docs < Config.BOOTSTRAP_DOC_THRESHOLD
    quality_threshold = (
        Config.BOOTSTRAP_QUALITY_GATE_THRESHOLD
        if bootstrap_mode
        else Config.QUALITY_GATE_NDCG_THRESHOLD
    )

    passed: bool = mean_ndcg >= quality_threshold

    if passed:
        logger.info(
            "ranker_quality_gate_passed",
            mean_ndcg=mean_ndcg,
            threshold=quality_threshold,
            n_fields=len(val_ndcg_scores),
            bootstrap_mode=bootstrap_mode,
            recommendation="Ranker is ready for production use",
        )
    else:
        logger.warning(
            "ranker_quality_gate_failed",
            mean_ndcg=mean_ndcg,
            threshold=quality_threshold,
            n_fields=len(val_ndcg_scores),
            bootstrap_mode=bootstrap_mode,
            recommendation="Ranker not ready - will fall back to heuristics. Add more training data.",
        )

    return passed


def train_models() -> dict[str, Any]:
    """
    Train XGBRanker models for all fields with available labels.

    Returns:
        Training summary
    """
    logger.info("training_started", model_type="ranker")

    # Load training data
    field_datasets, total_docs, total_rows = prepare_training_data()

    if not field_datasets:
        logger.info("training_skipped", reason="no_aligned_labels")
        return {
            "status": "skipped",
            "reason": "no_aligned_labels",
            "total_docs": 0,
            "total_rows": 0,
            "models_trained": 0,
        }

    # Guard: require minimum 2 labeled docs for bootstrap 2-fold CV
    min_docs_for_training = 2
    if total_docs < min_docs_for_training:
        logger.warning(
            "training_skipped_insufficient_docs",
            total_docs=total_docs,
            min_required=min_docs_for_training,
            reason="Need at least 2 labeled documents for 2-fold CV",
        )
        return {
            "status": "skipped",
            "reason": f"insufficient_docs ({total_docs} < {min_docs_for_training})",
            "total_docs": total_docs,
            "total_rows": total_rows,
            "models_trained": 0,
        }

    logger.info(
        "training_data_loaded",
        total_docs=total_docs,
        total_rows=total_rows,
        fields=list(field_datasets.keys()),
    )

    # Train ranker per field
    trained_models = {}
    training_stats = {}

    for field_name, field_data in field_datasets.items():
        model_info = train_field_ranker(field_name, field_data)

        if model_info:
            trained_models[field_name] = model_info
            training_stats[field_name] = {
                "pos_count": model_info["pos_count"],
                "neg_count": model_info["neg_count"],
                "total_samples": model_info["total_samples"],
                "metrics": model_info.get("metrics", {}),
            }
        else:
            training_stats[field_name] = {"status": "insufficient_data"}

    # Save models if any were trained
    model_path = None
    if trained_models:
        model_path = save_ranker_models(trained_models, n_docs=total_docs)

    models_trained = len(trained_models)
    total_pos = sum(stats.get("pos_count", 0) for stats in training_stats.values())
    total_neg = sum(stats.get("neg_count", 0) for stats in training_stats.values())

    logger.info(
        "training_complete",
        model_type="ranker",
        models_trained=models_trained,
        total_pos=total_pos,
        total_neg=total_neg,
    )

    # Quality gate: decide if model is good enough to use in production
    ranker_quality_passed = False
    if trained_models:
        ranker_quality_passed = _evaluate_ranker_quality(
            training_stats, n_docs=total_docs
        )

    # Update Prometheus training metrics
    try:
        from .metrics import update_training_metrics

        # Aggregate validation NDCG across all fields
        val_ndcg_scores = [
            stats.get("metrics", {}).get("val_ndcg_at_1")
            for stats in training_stats.values()
            if stats.get("metrics", {}).get("val_ndcg_at_1") is not None
        ]
        # Aggregate train NDCG across all fields (for overfitting gap)
        train_ndcg_scores = [
            stats.get("metrics", {}).get("loocv_train_ndcg_at_1")
            for stats in training_stats.values()
            if stats.get("metrics", {}).get("loocv_train_ndcg_at_1") is not None
        ]

        avg_val_ndcg = (
            sum(val_ndcg_scores) / len(val_ndcg_scores) if val_ndcg_scores else None
        )
        avg_train_ndcg = (
            sum(train_ndcg_scores) / len(train_ndcg_scores)
            if train_ndcg_scores
            else None
        )

        # Build per-field NDCG dict for labeled gauges
        per_field_ndcg = {
            field_name: stats["metrics"]["val_ndcg_at_1"]
            for field_name, stats in training_stats.items()
            if isinstance(stats.get("metrics"), dict)
            and stats["metrics"].get("val_ndcg_at_1") is not None
        }

        update_training_metrics(
            val_ndcg=avg_val_ndcg,
            train_accuracy=avg_train_ndcg,
            val_accuracy=avg_val_ndcg,
            per_field_ndcg=per_field_ndcg or None,
        )
    except ImportError:
        pass  # prometheus-client not installed

    return {
        "status": "success",
        "model_type": "ranker",
        "total_docs": total_docs,
        "total_rows": total_rows,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "models_trained": models_trained,
        "model_path": model_path,
        "training_stats": training_stats,
        "ranker_quality_passed": ranker_quality_passed,
    }


def load_trained_models() -> dict[str, Any] | None:
    """
    Load trained models from storage.

    Uses the configured storage backend (local or blob) based on
    Config.STORAGE_BACKEND. Automatically detects model type (ranker vs
    classifier) from manifest and loads the appropriate format.

    Returns:
        Dict with 'models', 'manifest', and 'model_type' or None if not available
    """
    from .storage import get_storage

    storage = get_storage()

    # Check if manifest exists
    if not storage.exists("models/manifest.json"):
        return None

    try:
        # Load manifest to detect model type
        manifest_json = storage.read_text("models/manifest.json")
        manifest = json.loads(manifest_json)
    except Exception as e:
        logger.warning("manifest_load_failed", error=str(e))
        return None

    # Enforce quality gate — never load models that failed validation
    if manifest.get("quality_gate_passed", False) is not True:
        logger.warning(
            "quality_gate_blocked_model_load",
            msg="Model failed quality gate, falling back to heuristics",
        )
        return None

    model_type = manifest.get("model_type", "classifier")

    if model_type != "ranker":
        logger.warning(
            "legacy_classifier_unsupported",
            model_type=model_type,
            msg="Only ranker models are supported. Retrain with USE_RANKER_MODEL=true.",
        )
        return None

    return _load_ranker_models_from_storage(storage, manifest)


def _load_ranker_models_from_storage(
    storage: Any, manifest: dict[str, Any]
) -> dict[str, Any] | None:
    """
    Load XGBRanker models from storage backend.

    Args:
        storage: StorageBackend instance (local or blob)
        manifest: Loaded manifest dict

    Returns:
        Dict with 'models', 'manifest', 'model_type' or None on failure
    """
    loaded_models = {}

    for field_name in manifest.get("fields", {}).keys():
        prefix = f"models/{field_name}"

        # Check if model exists in storage
        if not storage.exists(f"{prefix}/metadata.json"):
            logger.warning("ranker_missing", field=field_name, prefix=prefix)
            continue

        try:
            ranker = InvoiceFieldRanker()
            ranker.load_from_storage(storage, prefix)

            loaded_models[field_name] = {
                "ranker": ranker,
                "model_type": "ranker",
            }

            logger.info("loaded_ranker", field=field_name)
        except Exception as e:
            logger.warning("ranker_load_failed", field=field_name, error=str(e))
            continue

    if not loaded_models:
        return None

    logger.info("loaded_ranker_models", count=len(loaded_models))

    return {
        "models": loaded_models,
        "manifest": manifest,
        "model_type": "ranker",
    }
