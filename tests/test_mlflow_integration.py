"""MLflow integration tests (local SQLite tracking, no external server)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invoices.config import Config
from invoices.ranker import RANKER_VERSION, InvoiceFieldRanker


def _make_trained_ranker() -> InvoiceFieldRanker:
    """Create a minimal trained ranker for testing."""
    ranker = InvoiceFieldRanker()

    # Build minimal training data: 2 docs × 3 candidates each
    rows = []
    for doc_idx in range(2):
        for cand_idx in range(3):
            rows.append(
                {
                    "doc_id": f"doc_{doc_idx}",
                    "field": "total_amount",
                    "candidate_id": f"doc{doc_idx}_cand{cand_idx}",
                    "label": 1 if cand_idx == 0 else 0,
                    "center_x": cand_idx * 0.2,
                    "center_y": 0.5,
                    "bbox_norm_x0": cand_idx * 0.1,
                    "bbox_norm_y0": 0.3,
                }
            )

    df = pd.DataFrame(rows)
    feature_cols = ["center_x", "center_y", "bbox_norm_x0", "bbox_norm_y0"]
    ranker.feature_columns = feature_cols

    # Use raw xgboost to train quickly
    import xgboost as xgb

    group_sizes = np.array([3, 3])
    X = df[feature_cols].values
    y = df["label"].values

    ranker.model = xgb.XGBRanker(**ranker._params)
    ranker.model.fit(X, y, group=group_sizes)
    return ranker


@pytest.fixture
def mlflow_tmpdir():
    """Provide a temp directory for MLflow tracking with SQLite backend."""
    try:
        import mlflow
    except ImportError:
        pytest.skip("mlflow not installed")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracking_uri = f"sqlite:///{tmpdir}/mlflow.db"
        artifact_root = str(Path(tmpdir) / "artifacts")
        mlflow.set_tracking_uri(tracking_uri)
        # Create experiment with explicit artifact location so tests
        # don't depend on CWD being writable (fails in Docker as non-root)
        client = mlflow.MlflowClient()
        client.create_experiment("test-invoicex", artifact_location=artifact_root)
        mlflow.set_experiment("test-invoicex")
        yield tmpdir, tracking_uri


class TestRankerMLflowRoundtrip:
    """Test save_to_mlflow / from_mlflow roundtrip."""

    def test_save_and_load_produces_same_predictions(self, mlflow_tmpdir):
        """Model loaded from MLflow produces identical predictions."""
        import mlflow

        tmpdir, tracking_uri = mlflow_tmpdir
        ranker = _make_trained_ranker()

        # Create test input
        test_input = pd.DataFrame(
            {
                "center_x": [0.1, 0.5, 0.9],
                "center_y": [0.5, 0.5, 0.5],
                "bbox_norm_x0": [0.1, 0.3, 0.7],
                "bbox_norm_y0": [0.3, 0.3, 0.3],
            }
        )
        original_preds = ranker.model.predict(test_input.values)

        # Save to MLflow
        field_name = "TotalAmount"
        with mlflow.start_run() as run:
            ranker.save_to_mlflow(run.info.run_id, field_name)

        # Register model so from_mlflow can find it
        client = mlflow.MlflowClient()
        model_name = f"{Config.MLFLOW_MODEL_PREFIX}-{field_name}"
        artifact_uri = f"runs:/{run.info.run_id}/models/{field_name}"

        try:
            client.create_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            pass
        mv = client.create_model_version(
            name=model_name, source=artifact_uri, run_id=run.info.run_id
        )
        client.set_registered_model_alias(model_name, "production", mv.version)

        # Load from MLflow
        loaded = InvoiceFieldRanker.from_mlflow(field_name, tracking_uri=tracking_uri)

        # Predictions must match exactly (determinism)
        loaded_preds = loaded.model.predict(test_input.values)
        np.testing.assert_array_equal(original_preds, loaded_preds)
        assert loaded.feature_columns == ranker.feature_columns

    def test_save_preserves_metadata(self, mlflow_tmpdir):
        """Saved metadata includes version, feature columns, and params."""
        import mlflow

        tmpdir, tracking_uri = mlflow_tmpdir
        ranker = _make_trained_ranker()

        with mlflow.start_run() as run:
            ranker.save_to_mlflow(run.info.run_id, "InvoiceNumber")

        # Download and inspect metadata
        artifact_uri = f"runs:/{run.info.run_id}/models/InvoiceNumber"
        local_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri, dst_path=tmpdir
        )
        meta = json.loads((Path(local_dir) / "metadata.json").read_text())

        assert meta["version"] == RANKER_VERSION
        assert meta["feature_columns"] == ranker.feature_columns
        assert meta["field_name"] == "InvoiceNumber"
        assert "created_at" in meta
        assert "params" in meta


class TestRankerMLflowErrors:
    """Test error handling in MLflow methods."""

    def test_save_without_model_raises(self, mlflow_tmpdir):
        """save_to_mlflow raises ValueError when no model is trained."""
        import mlflow

        _, tracking_uri = mlflow_tmpdir
        ranker = InvoiceFieldRanker()

        with mlflow.start_run() as run:
            with pytest.raises(ValueError, match="No model to save"):
                ranker.save_to_mlflow(run.info.run_id, "TotalAmount")

    def test_from_mlflow_missing_model_raises(self, mlflow_tmpdir):
        """from_mlflow raises ModelNotFoundError for unregistered models."""
        from invoices.exceptions import ModelNotFoundError

        _, tracking_uri = mlflow_tmpdir

        with pytest.raises(ModelNotFoundError):
            InvoiceFieldRanker.from_mlflow(
                "NonExistentField", tracking_uri=tracking_uri
            )
