"""XGBoost training module with deterministic hyperparameters and byte-stable persistence."""

import json
from typing import Any

import numpy as np
import pandas as pd

from . import candidates, labels, paths, utils

# Fixed deterministic hyperparameters
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,  # No subsampling for determinism
    "colsample_bytree": 1.0,  # No column subsampling for determinism
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "seed": 42,
    "n_jobs": 1,  # Single threaded for determinism
    "verbosity": 0,
}


def create_training_features(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create feature matrix from candidates for training.

    Args:
        candidates_df: Candidates DataFrame

    Returns:
        Feature DataFrame
    """
    features = []

    for _, candidate in candidates_df.iterrows():
        # Basic geometric features
        center_x = (candidate["bbox_norm_x0"] + candidate["bbox_norm_x1"]) / 2
        center_y = (candidate["bbox_norm_y0"] + candidate["bbox_norm_y1"]) / 2
        width = candidate["bbox_norm_x1"] - candidate["bbox_norm_x0"]
        height = candidate["bbox_norm_y1"] - candidate["bbox_norm_y0"]
        area = width * height

        # Text features
        text = str(candidate.get("raw_text") or candidate.get("text", ""))
        char_count = len(text)
        word_count = len(text.split())
        digit_count = sum(bool(c.isdigit()) for c in text)
        alpha_count = sum(bool(c.isalpha()) for c in text)

        # Bucket features (one-hot)
        bucket = candidate.get("bucket", "other")
        bucket_features = {
            "bucket_amount_like": int(bucket == "amount_like"),
            "bucket_date_like": int(bucket == "date_like"),
            "bucket_id_like": int(bucket == "id_like"),
            "bucket_keyword_proximal": int(bucket == "keyword_proximal"),
            "bucket_random_negative": int(bucket == "random_negative"),
            "bucket_other": int(
                bucket
                not in [
                    "amount_like",
                    "date_like",
                    "id_like",
                    "keyword_proximal",
                    "random_negative",
                ]
            ),
        }

        # Page features
        page_idx = candidate.get("page_idx", 0)

        feature_row = {
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
        }

        features.append(feature_row)

    return pd.DataFrame(features)


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

    for field_name in aligned_df["field"].unique():
        field_labels = aligned_df[aligned_df["field"] == field_name]

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

                # Create binary labels (1 for labeled candidate, 0 for others)
                labels_array = np.zeros(len(candidates_df))
                labels_array[label_row["candidate_idx"]] = 1

                # Combine features and labels
                data_df = features_df.copy()
                data_df["label"] = labels_array
                data_df["doc_id"] = label_row["doc_id"]
                data_df["sha256"] = sha256

                field_data.append(data_df)

            except Exception as e:
                print(
                    f"Warning: Could not process {sha256} for field {field_name}: {e}"
                )
                continue

        if field_data:
            field_datasets[field_name] = pd.concat(field_data, ignore_index=True)

    return field_datasets, total_docs, total_rows


def train_field_model(
    field_name: str, training_data: pd.DataFrame
) -> dict[str, Any] | None:
    """
    Train XGBoost model for a single field.

    Args:
        field_name: Name of field to train
        training_data: Training DataFrame with features and labels

    Returns:
        Trained model info or None if insufficient data
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost>=2.0.0")

    # Separate features and labels
    feature_cols = [
        col for col in training_data.columns if col not in ["label", "doc_id", "sha256"]
    ]

    X = training_data[feature_cols]
    y = training_data["label"]

    # Check for sufficient positive examples
    pos_count = int(y.sum())
    neg_count = int((y == 0).sum())

    if pos_count < 2:
        print(f"Insufficient positive examples for {field_name}: {pos_count}")
        return None

    print(f"Training {field_name}: {pos_count} positive, {neg_count} negative examples")

    # Train model with fixed hyperparameters
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)

    return {
        "model": model,
        "feature_names": feature_cols,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "total_samples": len(training_data),
    }


def save_models(trained_models: dict[str, dict[str, Any]]) -> str:
    """
    Save trained models with byte-stable persistence.

    Args:
        trained_models: Dictionary of field -> model_info

    Returns:
        Path to saved model file
    """
    models_dir = paths.get_repo_root() / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Create model data for persistence
    model_data = {}
    manifest = {
        "model_version": "v1",
        "xgboost_version": None,
        "training_timestamp": utils.get_current_utc_iso(),
        "hyperparameters": XGBOOST_PARAMS,
        "fields": {},
    }

    try:
        import xgboost as xgb

        manifest["xgboost_version"] = xgb.__version__
    except ImportError:
        pass

    # Save each model
    for field_name, model_info in trained_models.items():
        model = model_info["model"]

        # Convert model to bytes for deterministic storage
        try:
            model_bytes = model.save_raw(raw_format="json")
        except (TypeError, ValueError, AttributeError) as e:
            print(
                f"Warning: 'json' raw_format not supported for field '{field_name}', using default format. Error: {e}"
            )
            model_bytes = model.save_raw()

        model_data[field_name] = {
            "model_bytes": model_bytes.decode("utf-8"),
            "feature_names": model_info["feature_names"],
        }

        manifest["fields"][field_name] = {
            "pos_count": model_info["pos_count"],
            "neg_count": model_info["neg_count"],
            "total_samples": model_info["total_samples"],
            "feature_count": len(model_info["feature_names"]),
        }

    # Save model data to npz (byte-stable)
    model_file = models_dir / "model_v1.npz"

    # Create deterministic byte representation
    serialized_data = json.dumps(model_data, sort_keys=True, separators=(",", ":"))
    data_bytes = serialized_data.encode("utf-8")

    # Use numpy for byte-stable storage
    np.savez_compressed(model_file, data=data_bytes)

    # Save manifest
    manifest_file = models_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"Saved models to: {model_file}")
    print(f"Saved manifest to: {manifest_file}")

    return str(model_file)


def train_models() -> dict[str, Any]:
    """
    Train XGBoost models for all fields with available labels.

    Returns:
        Training summary
    """
    print("Preparing training data from aligned labels...")

    # Load training data
    field_datasets, total_docs, total_rows = prepare_training_data()

    if not field_datasets:
        print("0 docs, 0 rows, 0 pos, 0 neg; training skipped (no aligned labels)")
        return {
            "status": "skipped",
            "reason": "no_aligned_labels",
            "total_docs": 0,
            "total_rows": 0,
            "models_trained": 0,
        }

    print(f"Found training data: {total_docs} docs, {total_rows} aligned labels")
    print(f"Training models for {len(field_datasets)} fields...")

    # Train models per field
    trained_models = {}
    training_stats = {}

    for field_name, field_data in field_datasets.items():
        model_info = train_field_model(field_name, field_data)

        if model_info:
            trained_models[field_name] = model_info
            training_stats[field_name] = {
                "pos_count": model_info["pos_count"],
                "neg_count": model_info["neg_count"],
                "total_samples": model_info["total_samples"],
            }
        else:
            training_stats[field_name] = {"status": "insufficient_data"}

    # Save models if any were trained
    model_file = None
    if trained_models:
        model_file = save_models(trained_models)

    models_trained = len(trained_models)
    total_pos = sum(stats.get("pos_count", 0) for stats in training_stats.values())
    total_neg = sum(stats.get("neg_count", 0) for stats in training_stats.values())

    print(f"Training complete: {models_trained} models trained")
    print(f"Total examples: {total_pos} positive, {total_neg} negative")

    return {
        "status": "success",
        "total_docs": total_docs,
        "total_rows": total_rows,
        "total_pos": total_pos,
        "total_neg": total_neg,
        "models_trained": models_trained,
        "model_file": model_file,
        "training_stats": training_stats,
    }


def load_trained_models() -> dict[str, Any] | None:
    """
    Load trained models from disk.

    Returns:
        Loaded models or None if not available
    """
    models_dir = paths.get_repo_root() / "data" / "models"
    model_file = models_dir / "model_v1.npz"
    manifest_file = models_dir / "manifest.json"

    if not model_file.exists() or not manifest_file.exists():
        return None

    try:
        import xgboost as xgb
    except ImportError:
        print("Warning: XGBoost not available, cannot load models")
        return None

    try:
        # Load model data
        npz_data = np.load(model_file)
        data_bytes = npz_data["data"]
        serialized_data = data_bytes.tobytes().decode("utf-8")
        model_data = json.loads(serialized_data)

        # Load manifest
        with open(manifest_file, encoding="utf-8") as f:
            manifest = json.load(f)

        # Reconstruct models
        loaded_models = {}

        for field_name, field_data in model_data.items():
            model = xgb.XGBClassifier(**XGBOOST_PARAMS)
            model.load_model(bytearray(field_data["model_bytes"], "utf-8"))

            loaded_models[field_name] = {
                "model": model,
                "feature_names": field_data["feature_names"],
            }

        print(f"Loaded {len(loaded_models)} trained models")

        return {"models": loaded_models, "manifest": manifest}

    except Exception as e:
        print(f"Failed to load models: {e}")
        return None
