"""Pipeline integration tests: End-to-end pipeline validation.

These tests verify that the full pipeline runs without errors and produces
valid outputs. They MUST fail when:
- Pipeline crashes or raises unhandled exceptions
- Outputs are missing required fields
- Data flow between stages is broken
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from invoices import (
    candidates,
    decoder,
    emit,
    ingest,
    io_utils,
    normalize,
    paths,
    tokenize,
)


def ingest_single_pdf(pdf_path: Path) -> str:
    """Helper to ingest a single PDF and return its SHA256.

    Args:
        pdf_path: Path to PDF file to ingest

    Returns:
        SHA256 hash of the ingested PDF
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temp seed folder with just this PDF
        seed_folder = Path(tmpdir) / "seed"
        seed_folder.mkdir()
        shutil.copy(pdf_path, seed_folder / pdf_path.name)

        # Ingest the folder
        ingest.ingest_seed_folder(str(seed_folder))

        # Get the SHA256 from the index
        docs = ingest.get_indexed_documents()
        if docs.empty:
            raise ValueError(f"Failed to ingest {pdf_path}")

        # Return the SHA256 of the most recently ingested document
        return str(docs.iloc[-1]["sha256"])


@pytest.fixture(scope="module")
def seed_pdfs() -> list[Path]:
    """Get list of seed PDFs."""
    pdfs = list(Path("seed_pdfs").glob("*.pdf"))
    if not pdfs:
        pytest.skip("No seed PDFs found")
    return pdfs


class TestPipelineExecution:
    """Test that pipeline executes end-to-end without errors."""

    @pytest.mark.slow
    def test_pipeline_runs_without_exceptions(self, seed_pdfs: list[Path]) -> None:
        """Pipeline must run end-to-end without raising exceptions."""
        test_pdf = seed_pdfs[0]

        # Ingest
        sha256 = ingest_single_pdf(test_pdf)
        assert sha256 is not None, "Ingest returned None"
        assert len(sha256) == 64, f"SHA256 has invalid length: {len(sha256)}"

        # Tokenize
        token_count = tokenize.tokenize_document(sha256)
        assert token_count > 0, f"Tokenization produced {token_count} tokens"

        # Generate candidates
        candidate_count = candidates.generate_candidates(sha256)
        assert candidate_count > 0, (
            f"Candidate generation produced {candidate_count} candidates"
        )

        # Decode
        result = decoder.decode_document(sha256)
        assert result is not None, "Decoder returned None"
        assert isinstance(result, dict), f"Decoder result is not a dict: {type(result)}"

        # Normalize
        normalized = normalize.normalize_assignments(result, sha256=sha256)
        assert normalized is not None, "Normalize returned None"

        # Emit
        predictions_path, review_entries, needs_review = emit.emit_document(
            sha256, normalized
        )
        assert predictions_path is not None, "Emit returned None predictions_path"
        assert isinstance(predictions_path, str), (
            f"Predictions path is not a string: {type(predictions_path)}"
        )
        assert isinstance(review_entries, list), (
            f"Review entries is not a list: {type(review_entries)}"
        )
        assert isinstance(needs_review, bool), (
            f"Needs review is not a bool: {type(needs_review)}"
        )

    @pytest.mark.slow
    def test_all_seed_pdfs_process_successfully(self, seed_pdfs: list[Path]) -> None:
        """All seed PDFs must process without errors."""
        failures = []

        for pdf_path in seed_pdfs:
            try:
                # Just test ingestion and tokenization (lightweight)
                sha256 = ingest_single_pdf(pdf_path)
                token_count = tokenize.tokenize_document(sha256)
                assert token_count > 0
            except Exception as e:
                failures.append(f"{pdf_path.name}: {str(e)}")

        if failures:
            pytest.fail(
                f"Pipeline failed for {len(failures)} PDFs:\n" + "\n".join(failures)
            )


class TestOutputStructure:
    """Test that pipeline outputs have the required structure."""

    def test_extensions_routing_present(self) -> None:
        """Output must include extensions.routing metadata."""
        predictions_dir = paths.get_predictions_dir()
        prediction_files = list(predictions_dir.glob("*.json"))

        if not prediction_files:
            pytest.skip("No predictions found")

        pred_file = prediction_files[0]
        with open(pred_file, encoding="utf-8") as f:
            prediction = json.load(f)

        assert "extensions" in prediction, "Output missing 'extensions'"
        assert "routing" in prediction["extensions"], "extensions missing 'routing'"

        routing = prediction["extensions"]["routing"]
        assert "needs_review" in routing, "routing missing 'needs_review'"
        assert "predicted_count" in routing, "routing missing 'predicted_count'"
        assert "abstain_count" in routing, "routing missing 'abstain_count'"


class TestEntryPointParity:
    """Test that different code paths through the pipeline produce identical output."""

    @pytest.mark.slow
    def test_decode_explicit_vs_implicit_candidates(
        self, seed_pdfs: list[Path]
    ) -> None:
        """decode_document with explicit candidates_df must match implicit load.

        This catches bugs where the candidate loading path diverges from
        the explicit-pass path (e.g., stale cache, column mismatch).
        """
        test_pdf = seed_pdfs[0]

        # Set up: ingest + tokenize + generate candidates
        sha256 = ingest_single_pdf(test_pdf)
        tokenize.tokenize_document(sha256)
        candidates.generate_candidates(sha256)

        # Path 1: Let decode_document load candidates from disk (implicit)
        result_implicit = decoder.decode_document(sha256)

        # Path 2: Load candidates ourselves and pass explicitly
        candidates_path = paths.get_candidates_path(sha256)
        candidates_df = io_utils.read_parquet_safe(candidates_path)
        result_explicit = decoder.decode_document(sha256, candidates_df=candidates_df)

        # Both paths must produce identical assignments
        # Compare field-by-field since dicts may contain numpy arrays
        assert sorted(result_implicit.keys()) == sorted(result_explicit.keys()), (
            "decode_document produces different field sets: "
            f"implicit={sorted(result_implicit.keys())}, "
            f"explicit={sorted(result_explicit.keys())}"
        )
        for field in result_implicit:
            impl_val = result_implicit[field]
            expl_val = result_explicit[field]
            # Compare dicts with potential numpy values
            assert str(impl_val) == str(expl_val), (
                f"Field '{field}' differs between implicit and explicit paths: "
                f"implicit={impl_val}, explicit={expl_val}"
            )


class TestPipelineOutputQuality:
    """Test that pipeline produces meaningful outputs."""

    def test_confidence_diversity(self) -> None:
        """Confidence scores should show diversity (not all the same value)."""
        predictions_dir = paths.get_predictions_dir()
        prediction_files = list(predictions_dir.glob("*.json"))

        if not prediction_files:
            pytest.skip("No predictions found")

        all_confidences = []

        for pred_file in prediction_files:
            with open(pred_file, encoding="utf-8") as f:
                prediction = json.load(f)

            for field_data in prediction["fields"].values():
                all_confidences.append(field_data["confidence"])

        # Should have more than 1 unique confidence value
        unique_confidences = set(all_confidences)
        assert len(unique_confidences) > 1, (
            f"All confidence scores are identical ({unique_confidences}). "
            "Confidence calibration may be broken."
        )
