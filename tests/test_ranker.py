"""Tests for the InvoiceFieldRanker module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from invoices.ranker import (
    RANKER_PARAMS,
    InvoiceFieldRanker,
    get_numeric_feature_columns,
)


class TestInvoiceFieldRanker:
    """Test the InvoiceFieldRanker class."""

    @pytest.fixture
    def synthetic_candidates_df(self) -> pd.DataFrame:
        """Create synthetic candidates for testing."""
        candidates = []
        for doc_idx in range(2):
            for cand_idx in range(5):
                candidates.append(
                    {
                        "candidate_id": f"doc{doc_idx}_cand{cand_idx}",
                        "doc_id": f"doc_{doc_idx}",
                        "page_idx": 0,
                        "raw_text": f"Text{cand_idx}",
                        "total_score": 1.0 - cand_idx * 0.1,
                        "feature_1": np.random.rand(),
                        "feature_2": np.random.rand(),
                        "center_x": cand_idx * 0.1,
                        "center_y": 0.5,
                        "bbox_norm_x0": cand_idx * 0.1,
                        "bbox_norm_y0": 0.0,
                    }
                )
        return pd.DataFrame(candidates)

    @pytest.fixture
    def synthetic_labels_df(self) -> pd.DataFrame:
        """Create synthetic labels for training."""
        # Label the first candidate of each doc as correct for "invoice_number"
        labels = [
            {
                "doc_id": "doc_0",
                "field": "invoice_number",
                "candidate_idx": 0,
            },
            {
                "doc_id": "doc_1",
                "field": "invoice_number",
                "candidate_idx": 0,
            },
        ]
        return pd.DataFrame(labels)

    def test_predict_fallback_uses_total_score(
        self, synthetic_candidates_df: pd.DataFrame
    ) -> None:
        """Test that predict falls back to total_score when no model."""
        ranker = InvoiceFieldRanker()
        scores = ranker.predict(synthetic_candidates_df)

        # Should use total_score as fallback
        np.testing.assert_array_almost_equal(
            scores, synthetic_candidates_df["total_score"].values
        )

    def test_predict_fallback_without_total_score(self) -> None:
        """Test fallback returns zeros when no total_score column."""
        ranker = InvoiceFieldRanker()
        df = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2"],
                "doc_id": ["d1", "d1"],
                "feature_1": [0.5, 0.6],
            }
        )
        scores = ranker.predict(df)
        np.testing.assert_array_equal(scores, [0.0, 0.0])

    def test_predict_empty_dataframe(self) -> None:
        """Test predict with empty DataFrame returns empty array."""
        ranker = InvoiceFieldRanker()
        empty_df = pd.DataFrame()
        scores = ranker.predict(empty_df)
        assert len(scores) == 0

    def test_rank_candidates_for_field(
        self, synthetic_candidates_df: pd.DataFrame
    ) -> None:
        """Test ranking candidates for a specific field."""
        ranker = InvoiceFieldRanker()
        ranked = ranker.rank_candidates_for_field(
            synthetic_candidates_df, "invoice_number"
        )

        # Should be sorted by score descending
        assert "ranker_score" in ranked.columns
        scores = ranked["ranker_score"].values
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_save_without_model_raises_error(self) -> None:
        """Test that save raises ValueError when no model is trained."""
        ranker = InvoiceFieldRanker()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model"

            with pytest.raises(ValueError, match="No model to save"):
                ranker.save(model_path)

    def test_load_missing_model_raises_error(self) -> None:
        """Test that load raises FileNotFoundError for missing model."""
        ranker = InvoiceFieldRanker()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "nonexistent_model"

            with pytest.raises(FileNotFoundError):
                ranker.load(model_path)


class TestRankerParams:
    """Test ranker configuration parameters."""

    def test_ranker_params_are_deterministic(self) -> None:
        """Verify RANKER_PARAMS enforce determinism."""
        assert RANKER_PARAMS["n_jobs"] == 1
        assert RANKER_PARAMS["subsample"] == 0.8
        assert RANKER_PARAMS["colsample_bytree"] == 0.8
        assert RANKER_PARAMS["random_state"] == 42
        assert RANKER_PARAMS["seed"] == 42


class TestGetNumericFeatureColumns:
    """Test feature column extraction."""

    def test_excludes_id_columns(self) -> None:
        """Verify ID columns are excluded from features."""
        df = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2"],
                "doc_id": ["d1", "d2"],
                "page_idx": [0, 1],
                "feature_1": [0.5, 0.6],
                "feature_2": [1.0, 2.0],
            }
        )

        cols = get_numeric_feature_columns(df)
        assert "candidate_id" not in cols
        assert "doc_id" not in cols
        assert "feature_1" in cols
        assert "feature_2" in cols

    def test_excludes_label_columns(self) -> None:
        """Verify label columns are excluded from features."""
        df = pd.DataFrame(
            {
                "feature_1": [0.5, 0.6],
                "label": [0, 1],
                "target_field_encoded": [1, 2],
            }
        )

        cols = get_numeric_feature_columns(df)
        assert "label" not in cols
        assert "target_field_encoded" not in cols
        assert "feature_1" in cols


class TestRankerTrainingValidation:
    """Test ranker training input validation."""

    def test_train_empty_dataframe_raises_error(self) -> None:
        """Training with empty DataFrame should raise ValueError."""
        ranker = InvoiceFieldRanker()
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="empty"):
            ranker.train(empty_df)

    def test_train_missing_columns_raises_error(self) -> None:
        """Training without required columns should raise ValueError."""
        ranker = InvoiceFieldRanker()
        df = pd.DataFrame(
            {
                "feature_1": [0.5, 0.6],
                # Missing 'label' and 'doc_id' columns
            }
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            ranker.train(df)


class TestQualityGateEnforcement:
    """Test that load_trained_models() blocks models that failed quality gate."""

    def test_quality_gate_blocks_failed_models(self, tmp_path: Path) -> None:
        """Models with quality_gate_passed=False must NOT be loaded."""
        import json

        from invoices.train import load_trained_models

        # Create a manifest where quality gate explicitly failed
        manifest = {
            "model_type": "ranker",
            "model_version": "v2",
            "quality_gate_passed": False,
            "fields": {
                "TotalAmount": {
                    "pos_count": 5,
                    "neg_count": 50,
                    "total_samples": 55,
                    "val_ndcg_at_1": 0.2,
                }
            },
        }

        # Write manifest to a temp storage location
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        # Patch get_storage to use our temp directory
        from unittest.mock import patch

        from invoices.storage.local import LocalStorageBackend

        mock_storage = LocalStorageBackend(base_path=tmp_path)

        with patch("invoices.storage.get_storage", return_value=mock_storage):
            result = load_trained_models()

        # Must return None — quality gate blocks loading
        assert result is None, (
            "load_trained_models() should return None when quality_gate_passed=False, "
            f"but got: {result}"
        )

    def test_quality_gate_allows_passed_models(self, tmp_path: Path) -> None:
        """Models with quality_gate_passed=True should proceed to loading."""
        import json

        from invoices.train import load_trained_models

        # Create a manifest where quality gate passed but no model files
        # (will return None because model files are missing, not because of gate)
        manifest = {
            "model_type": "ranker",
            "model_version": "v2",
            "quality_gate_passed": True,
            "fields": {
                "TotalAmount": {
                    "pos_count": 20,
                    "neg_count": 200,
                    "total_samples": 220,
                    "val_ndcg_at_1": 0.8,
                }
            },
        }

        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        from unittest.mock import patch

        from invoices.storage.local import LocalStorageBackend

        mock_storage = LocalStorageBackend(base_path=tmp_path)

        with patch("invoices.storage.get_storage", return_value=mock_storage):
            with patch("invoices.train._load_ranker_models_from_storage") as mock_load:
                mock_load.return_value = None
                result = load_trained_models()

        # Quality gate passed, so _load_ranker_models_from_storage was called
        # (unlike the previous test where the gate blocked before reaching it)
        mock_load.assert_called_once()


class TestRankerDeterminism:
    """Test that ranker produces deterministic results."""

    def test_predict_is_deterministic(self) -> None:
        """Predict should return same scores for same input."""
        ranker = InvoiceFieldRanker(random_state=42)

        df = pd.DataFrame(
            {
                "candidate_id": ["c1", "c2", "c3"],
                "doc_id": ["d1", "d1", "d1"],
                "total_score": [0.9, 0.7, 0.5],
            }
        )

        scores1 = ranker.predict(df)
        scores2 = ranker.predict(df)

        np.testing.assert_array_equal(scores1, scores2)
