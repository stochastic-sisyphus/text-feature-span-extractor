"""Test review queue functionality and completeness."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json

import pandas as pd
import pytest

from invoices import candidates, emit, ingest, paths, tokenize


class TestReviewQueue:
    """Test that review queue correctly captures ABSTAIN decisions."""

    def test_review_queue_exists(self):
        """Test that review queue file is created after emit."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run full pipeline
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()

        # Check that review queue exists
        review_queue_path = paths.get_review_queue_path()
        assert review_queue_path.exists(), "Review queue file should exist after emit"

        # Load and check basic structure
        review_df = pd.read_parquet(review_queue_path)

        # Check required columns
        required_columns = [
            "doc_id",
            "field",
            "reason",
            "page",
            "bbox_x0",
            "bbox_y0",
            "bbox_x1",
            "bbox_y1",
            "token_span",
            "raw_text",
        ]

        for col in required_columns:
            assert (
                col in review_df.columns
            ), f"Required column {col} missing from review queue"

        print(f"✓ Review queue exists with {len(review_df)} entries")

    def test_abstain_entries_in_queue(self):
        """Test that every ABSTAIN decision appears in review queue."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()

        # Get all predictions and count ABSTAIN decisions
        indexed_docs = ingest.get_indexed_documents()

        total_abstains = 0
        abstain_by_doc = {}

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            doc_id = doc_info["doc_id"]

            predictions_path = paths.get_predictions_path(sha256)
            if not predictions_path.exists():
                continue

            with open(predictions_path, encoding="utf-8") as f:
                predictions = json.load(f)

            doc_abstains = 0
            for _field_name, field_data in predictions["fields"].items():
                if field_data["status"] == "ABSTAIN":
                    total_abstains += 1
                    doc_abstains += 1

            abstain_by_doc[doc_id] = doc_abstains

        # Get review queue entries
        review_df = emit.get_review_queue()

        # Count ABSTAIN entries in review queue
        abstain_review_entries = review_df[review_df["reason"] == "ABSTAIN"]
        queue_abstain_count = len(abstain_review_entries)

        # Every ABSTAIN should have a review queue entry
        assert (
            queue_abstain_count >= total_abstains
        ), f"Review queue should have at least {total_abstains} ABSTAIN entries, found {queue_abstain_count}"

        print(
            f"✓ ABSTAIN entries test passed: {total_abstains} abstains, {queue_abstain_count} queue entries"
        )

    def test_review_queue_data_quality(self):
        """Test review queue data quality and completeness."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()

        # Get review queue
        review_df = emit.get_review_queue()

        if review_df.empty:
            pytest.skip("No review queue entries to test")

        # Check that all entries have valid doc_ids
        indexed_docs = ingest.get_indexed_documents()
        valid_doc_ids = set(indexed_docs["doc_id"].tolist())

        for _, entry in review_df.iterrows():
            doc_id = entry["doc_id"]
            assert doc_id in valid_doc_ids, f"Invalid doc_id in review queue: {doc_id}"

            field = entry["field"]
            # Get schema fields for validation
            from invoices import utils

            schema_obj = utils.load_contract_schema()
            schema_fields = schema_obj.get("fields", [])
            assert field in schema_fields, f"Invalid field in review queue: {field}"

            reason = entry["reason"]
            assert reason in [
                "ABSTAIN",
                "MISSING",
            ], f"Invalid reason in review queue: {reason}"

            # If page info is present, validate it
            if pd.notna(entry["page"]):
                page = entry["page"]
                assert isinstance(page, (int, float)), f"Page should be numeric: {page}"
                assert page >= 0, f"Page should be non-negative: {page}"

                # Check bbox coordinates if present
                if all(
                    pd.notna(entry[f"bbox_{coord}"])
                    for coord in ["x0", "y0", "x1", "y1"]
                ):
                    bbox = [
                        entry["bbox_x0"],
                        entry["bbox_y0"],
                        entry["bbox_x1"],
                        entry["bbox_y1"],
                    ]
                    assert all(
                        isinstance(x, (int, float)) for x in bbox
                    ), "Bbox coordinates should be numeric"
                    assert 0 <= bbox[0] <= 1, "Bbox x0 should be in [0,1]"
                    assert 0 <= bbox[1] <= 1, "Bbox y0 should be in [0,1]"
                    assert 0 <= bbox[2] <= 1, "Bbox x1 should be in [0,1]"
                    assert 0 <= bbox[3] <= 1, "Bbox y1 should be in [0,1]"
                    assert bbox[2] >= bbox[0], "x1 should be >= x0"
                    assert bbox[3] >= bbox[1], "y1 should be >= y0"

        print(
            f"✓ Review queue data quality test passed: {len(review_df)} entries validated"
        )

    def test_review_queue_field_coverage(self):
        """Test that review queue covers all expected fields."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()

        # Get review queue
        review_df = emit.get_review_queue()

        if review_df.empty:
            print("✓ Review queue is empty (no abstains/missing)")
            return

        # Check field coverage
        queue_fields = set(review_df["field"].unique())

        # All fields in queue should be valid schema fields
        from invoices import utils

        schema_obj = utils.load_contract_schema()
        schema_fields = schema_obj.get("fields", [])
        for field in queue_fields:
            assert field in schema_fields, f"Invalid field in review queue: {field}"

        # Count entries by field
        field_counts = review_df["field"].value_counts().to_dict()

        print("✓ Review queue field coverage test passed")
        print(f"  Fields in queue: {len(queue_fields)}")
        for field, count in sorted(field_counts.items()):
            print(f"    {field}: {count} entries")

    def test_review_queue_idempotency(self):
        """Test that review queue updates are idempotent."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Run pipeline first time
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()
        candidates.generate_all_candidates()
        emit.emit_all_documents()

        # Get review queue after first run
        review_df1 = emit.get_review_queue()

        # Run emit again (should be idempotent)
        emit.emit_all_documents()

        # Get review queue after second run
        review_df2 = emit.get_review_queue()

        # Should have same number of entries (duplicates removed)
        assert len(review_df2) == len(
            review_df1
        ), "Review queue should not grow on re-emit"

        # Check that entries are essentially the same
        if not review_df1.empty and not review_df2.empty:
            # Sort both for comparison
            df1_sorted = review_df1.sort_values(["doc_id", "field"]).reset_index(
                drop=True
            )
            df2_sorted = review_df2.sort_values(["doc_id", "field"]).reset_index(
                drop=True
            )

            # Compare key columns
            key_columns = ["doc_id", "field", "reason"]
            for col in key_columns:
                if col in df1_sorted.columns and col in df2_sorted.columns:
                    assert (
                        df1_sorted[col].tolist() == df2_sorted[col].tolist()
                    ), f"Column {col} should be identical after re-emit"

        print(
            f"✓ Review queue idempotency test passed: {len(review_df2)} entries stable"
        )
