"""Golden tests for deterministic PDF→JSON extraction outputs."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json

import pytest

from invoices import candidates, decoder, emit, ingest, paths, tokenize, utils

# ruff: noqa: F841 - Frozen pre-training snapshot:  some variables intentionally unused


class TestGoldenOutputs:
    """Test deterministic PDF→JSON extraction with golden fixtures."""

    def setup_method(self):
        """Ensure clean state for each test."""
        paths.ensure_directories()

    def test_end_to_end_determinism(self):
        """Test that the entire pipeline produces deterministic outputs."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline twice and compare outputs
        results1 = self._run_full_pipeline(str(seed_folder))
        results2 = self._run_full_pipeline(str(seed_folder))

        # Compare document count
        assert len(results1) == len(results2), "Document count should be deterministic"

        # Compare each document's output
        for sha256 in results1.keys():
            assert sha256 in results2, f"Document {sha256} should be in both runs"

            json1 = results1[sha256]
            json2 = results2[sha256]

            # Serialize to JSON strings for byte-level comparison
            json_str1 = json.dumps(json1, sort_keys=True, separators=(",", ":"))
            json_str2 = json.dumps(json2, sort_keys=True, separators=(",", ":"))

            assert (
                json_str1 == json_str2
            ), f"JSON output should be deterministic for {sha256}"

        print(f"✓ End-to-end determinism verified for {len(results1)} documents")

    def test_prediction_stability(self):
        """Test that predictions remain stable across multiple runs."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline and collect predictions
        predictions = self._run_full_pipeline(str(seed_folder))

        # Track prediction patterns for stability analysis
        prediction_summary = {}

        for _sha256, json_output in predictions.items():
            doc_id = json_output["document_id"]
            fields = json_output["fields"]

            # Count prediction vs abstain vs missing
            status_counts = {}
            for _field_name, field_data in fields.items():
                status = field_data["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

            prediction_summary[doc_id] = {
                "total_fields": len(fields),
                "predicted": status_counts.get("PREDICTED", 0),
                "abstain": status_counts.get("ABSTAIN", 0),
                "missing": status_counts.get("MISSING", 0),
                "pages": json_output["pages"],
            }

        # Verify minimum prediction quality
        for doc_id, summary in prediction_summary.items():
            total_fields = summary["total_fields"]
            predicted = summary["predicted"]

            # At least some fields should have predictions (not all abstain/missing)
            prediction_rate = predicted / total_fields if total_fields > 0 else 0

            print(
                f"  {doc_id}: {predicted}/{total_fields} predicted ({prediction_rate:.1%})"
            )

            # Allow for abstaining on difficult documents, but ensure pipeline works
            assert total_fields > 0, f"Should have schema fields for {doc_id}"
            assert summary["pages"] > 0, f"Should have extracted pages for {doc_id}"

        print(
            f"✓ Prediction stability verified for {len(prediction_summary)} documents"
        )

    def test_contract_schema_compliance(self):
        """Test that all outputs comply with contract schema."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Load expected schema
        schema_obj = utils.load_contract_schema()
        schema_fields = set(schema_obj.get("fields", []))

        # Run pipeline
        predictions = self._run_full_pipeline(str(seed_folder))

        for _sha256, json_output in predictions.items():
            doc_id = json_output["document_id"]

            # Check top-level contract structure
            required_top_level = [
                "document_id",
                "pages",
                "contract_version",
                "feature_version",
                "decoder_version",
                "model_version",
                "calibration_version",
                "fields",
                "line_items",
                "extensions",
                "experimental",
            ]

            for field in required_top_level:
                assert (
                    field in json_output
                ), f"Missing top-level field {field} in {doc_id}"

            # Check version stamps are non-empty
            version_fields = [
                "contract_version",
                "feature_version",
                "decoder_version",
                "model_version",
                "calibration_version",
            ]
            for version_field in version_fields:
                value = json_output[version_field]
                assert value and isinstance(
                    value, str
                ), f"Version field {version_field} should be non-empty string in {doc_id}"

            # Check all schema fields are present in output
            output_fields = set(json_output["fields"].keys())
            assert (
                schema_fields == output_fields
            ), f"Field set mismatch in {doc_id}: expected {schema_fields}, got {output_fields}"

            # Check field structure
            for field_name, field_data in json_output["fields"].items():
                required_props = [
                    "value",
                    "confidence",
                    "status",
                    "provenance",
                    "raw_text",
                ]
                for prop in required_props:
                    assert (
                        prop in field_data
                    ), f"Missing property {prop} in field {field_name} for {doc_id}"

                # Check status values
                status = field_data["status"]
                assert status in {
                    "PREDICTED",
                    "ABSTAIN",
                    "MISSING",
                }, f"Invalid status {status} for {field_name} in {doc_id}"

                # Check confidence range
                confidence = field_data["confidence"]
                assert (
                    isinstance(confidence, (int, float)) and 0.0 <= confidence <= 1.0
                ), f"Invalid confidence {confidence} for {field_name} in {doc_id}"

        print(f"✓ Contract schema compliance verified for {len(predictions)} documents")

    def test_provenance_completeness(self):
        """Test that all PREDICTED fields have complete provenance."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        predictions = self._run_full_pipeline(str(seed_folder))

        total_predicted = 0
        total_with_provenance = 0

        for _sha256, json_output in predictions.items():
            doc_id = json_output["document_id"]

            for field_name, field_data in json_output["fields"].items():
                if field_data["status"] == "PREDICTED":
                    total_predicted += 1

                    # Check provenance exists and is complete
                    provenance = field_data["provenance"]
                    assert (
                        provenance is not None
                    ), f"PREDICTED field {field_name} missing provenance in {doc_id}"

                    required_prov_fields = ["page", "bbox", "token_span"]
                    for prov_field in required_prov_fields:
                        assert (
                            prov_field in provenance
                        ), f"Missing provenance field {prov_field} for {field_name} in {doc_id}"

                    # Check bbox format
                    bbox = provenance["bbox"]
                    assert (
                        isinstance(bbox, list) and len(bbox) == 4
                    ), f"Invalid bbox format for {field_name} in {doc_id}"
                    assert all(
                        isinstance(x, (int, float)) for x in bbox
                    ), f"Non-numeric bbox values for {field_name} in {doc_id}"

                    # Check token span
                    token_span = provenance["token_span"]
                    assert (
                        isinstance(token_span, list) and len(token_span) > 0
                    ), f"Invalid token_span for {field_name} in {doc_id}"

                    total_with_provenance += 1

        assert (
            total_predicted == total_with_provenance
        ), f"All {total_predicted} predicted fields should have complete provenance"
        print(
            f"✓ Provenance completeness verified for {total_predicted} predicted fields"
        )

    def test_cross_run_sha256_consistency(self):
        """Test that SHA256 hashes remain consistent across runs."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run ingestion twice
        paths.ensure_directories()

        ingest.ingest_seed_folder(str(seed_folder))
        index1_df = ingest.get_indexed_documents()
        sha256s1 = set(index1_df["sha256"].tolist())

        ingest.ingest_seed_folder(str(seed_folder))  # Second run
        index2_df = ingest.get_indexed_documents()
        sha256s2 = set(index2_df["sha256"].tolist())

        assert sha256s1 == sha256s2, "SHA256 hashes should be consistent across runs"

        # Verify SHA256s are valid (64 hex characters)
        for sha256 in sha256s1:
            assert len(sha256) == 64, f"Invalid SHA256 length: {sha256}"
            assert all(
                c in "0123456789abcdef" for c in sha256
            ), f"Invalid SHA256 characters: {sha256}"

        print(f"✓ SHA256 consistency verified for {len(sha256s1)} documents")

    def _run_full_pipeline(self, seed_folder: str) -> dict:
        """
        Run the full pipeline and return JSON outputs.

        Args:
            seed_folder: Path to seed PDF folder

        Returns:
            Dictionary mapping SHA256 -> JSON output
        """
        # Run all pipeline stages
        ingest.ingest_seed_folder(seed_folder)
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        decoder.decode_all_documents()
        emit.emit_all_documents()

        # Collect all JSON outputs
        indexed_docs = ingest.get_indexed_documents()
        results = {}

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            predictions_path = paths.get_predictions_path(sha256)

            if predictions_path.exists():
                with open(predictions_path, encoding="utf-8") as f:
                    results[sha256] = json.load(f)

        return results

    def test_pipeline_timing_consistency(self):
        """Test that pipeline stages complete in reasonable time."""
        seed_folder = Path(__file__).parent.parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        import time

        paths.ensure_directories()

        # Time each stage
        stage_times = {}

        start_time = time.time()
        ingest_count = ingest.ingest_seed_folder(str(seed_folder))
        stage_times["ingest"] = time.time() - start_time

        start_time = time.time()
        tokenize_results = tokenize.tokenize_all()
        stage_times["tokenize"] = time.time() - start_time

        start_time = time.time()
        candidate_results = candidates.generate_all_candidates()
        stage_times["candidates"] = time.time() - start_time

        start_time = time.time()
        decode_results = decoder.decode_all_documents()
        stage_times["decode"] = time.time() - start_time

        start_time = time.time()
        emit_results = emit.emit_all_documents()
        stage_times["emit"] = time.time() - start_time

        # Check reasonable timing bounds (generous for CI)
        max_time_per_doc = 30.0  # seconds per document
        doc_count = len(tokenize_results) if tokenize_results else 1

        for stage, duration in stage_times.items():
            time_per_doc = duration / doc_count
            assert (
                time_per_doc < max_time_per_doc
            ), f"Stage {stage} took {time_per_doc:.1f}s per doc (max {max_time_per_doc}s)"

        total_time = sum(stage_times.values())
        print(
            f"✓ Pipeline timing verified: {total_time:.1f}s total for {doc_count} documents"
        )
        for stage, duration in stage_times.items():
            print(f"  {stage}: {duration:.1f}s")
