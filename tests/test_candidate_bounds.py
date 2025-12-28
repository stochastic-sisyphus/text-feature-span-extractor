"""Test candidate generation bounds and bucket distribution."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from invoices import candidates, ingest, paths, tokenize


class TestCandidateBounds:
    """Test that candidate generation respects bounds and bucket requirements."""

    def test_candidate_count_bounds(self):
        """Test that candidate count is ≤ 200 per document."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure pipeline is ready
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()

        # Test all documents
        indexed_docs = ingest.get_indexed_documents()
        assert not indexed_docs.empty, "Should have ingested documents"

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            doc_id = doc_info["doc_id"]

            # Generate candidates
            candidate_count = candidates.generate_candidates(sha256)

            # Check bound
            assert (
                candidate_count <= 200
            ), f"Document {doc_id} has {candidate_count} candidates (max 200)"

            # Load candidates to verify
            candidates_df = candidates.get_document_candidates(sha256)
            assert (
                len(candidates_df) == candidate_count
            ), "Candidate count should match DataFrame length"
            assert (
                len(candidates_df) <= 200
            ), "Candidates DataFrame should not exceed 200 rows"

        print("✓ Candidate bounds test passed: all documents ≤ 200 candidates")

    def test_bucket_distribution(self):
        """Test that all five buckets contribute when applicable."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure pipeline is ready
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()

        # Expected bucket types
        expected_buckets = {
            "date_like",
            "amount_like",
            "id_like",
            "keyword_proximal",
            "random_negative",
        }

        # Test all documents
        indexed_docs = ingest.get_indexed_documents()

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            doc_id = doc_info["doc_id"]

            # Check if document has tokens first
            tokens_df = tokenize.get_document_tokens(sha256)
            if tokens_df.empty:
                print(f"Skipping {doc_id}: no tokens found")
                continue

            # Generate candidates
            candidates.generate_candidates(sha256)
            candidates_df = candidates.get_document_candidates(sha256)

            if candidates_df.empty:
                print(f"Skipping {doc_id}: no candidates generated")
                continue

            # Check bucket distribution
            bucket_counts = candidates_df["bucket"].value_counts().to_dict()
            found_buckets = set(bucket_counts.keys())

            print(f"Document {doc_id}: buckets = {bucket_counts}")

            # For documents with sufficient tokens, we should have most bucket types
            if (
                len(tokens_df) >= 10
            ):  # Only check distribution for docs with enough tokens
                # Should have at least 3 different bucket types
                assert (
                    len(found_buckets) >= 3
                ), f"Document {doc_id} should have at least 3 bucket types"

                # Check that buckets make sense
                for bucket in found_buckets:
                    assert bucket in expected_buckets, f"Unknown bucket type: {bucket}"
                    assert (
                        bucket_counts[bucket] > 0
                    ), f"Bucket {bucket} should have positive count"

        print("✓ Bucket distribution test passed")

    def test_candidate_features(self):
        """Test that candidates have all required features."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure pipeline is ready
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()

        # Required feature columns
        required_features = [
            # Basic info
            "candidate_id",
            "doc_id",
            "sha256",
            "page_idx",
            "token_idx",
            "token_id",
            "text",
            "bucket",
            # Bounding box
            "bbox_norm_x0",
            "bbox_norm_y0",
            "bbox_norm_x1",
            "bbox_norm_y1",
            # Text features
            "text_length",
            "digit_ratio",
            "uppercase_ratio",
            "currency_flag",
            "unigram_hash",
            "bigram_hash",
            # Geometry features
            "center_x",
            "center_y",
            "width_norm",
            "height_norm",
            "distance_to_top",
            "distance_to_bottom",
            "distance_to_left",
            "distance_to_right",
            "distance_to_center",
            # Style features
            "font_size",
            "font_size_z",
            "is_bold",
            "is_italic",
            "font_hash",
            # Context features
            "context_bow_hash",
            "same_row_count",
            "block_index",
            "local_density",
            "is_remittance_band",
        ]

        # Test all documents
        indexed_docs = ingest.get_indexed_documents()

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            doc_id = doc_info["doc_id"]

            # Generate candidates
            candidates.generate_candidates(sha256)
            candidates_df = candidates.get_document_candidates(sha256)

            if candidates_df.empty:
                continue

            # Check all required columns exist
            for feature in required_features:
                assert (
                    feature in candidates_df.columns
                ), f"Required feature {feature} missing from candidates for {doc_id}"

            # Check data types and ranges
            assert (
                candidates_df["page_idx"].min() >= 0
            ), "Page indices should be non-negative"
            assert (
                candidates_df["token_idx"].min() >= 0
            ), "Token indices should be non-negative"
            assert (
                candidates_df["text_length"].min() > 0
            ), "Text length should be positive"
            assert (
                candidates_df["digit_ratio"].min() >= 0
            ), "Digit ratio should be non-negative"
            assert candidates_df["digit_ratio"].max() <= 1, "Digit ratio should be ≤ 1"
            assert (
                candidates_df["uppercase_ratio"].min() >= 0
            ), "Uppercase ratio should be non-negative"
            assert (
                candidates_df["uppercase_ratio"].max() <= 1
            ), "Uppercase ratio should be ≤ 1"

            # Check normalized coordinates are in valid range
            assert candidates_df["center_x"].min() >= 0, "Center X should be >= 0"
            assert candidates_df["center_x"].max() <= 1, "Center X should be <= 1"
            assert candidates_df["center_y"].min() >= 0, "Center Y should be >= 0"
            assert candidates_df["center_y"].max() <= 1, "Center Y should be <= 1"

            # Check bounding box consistency
            assert (
                candidates_df["bbox_norm_x1"] >= candidates_df["bbox_norm_x0"]
            ).all(), "x1 should be >= x0"
            assert (
                candidates_df["bbox_norm_y1"] >= candidates_df["bbox_norm_y0"]
            ).all(), "y1 should be >= y0"

        print("✓ Candidate features test passed")

    def test_nms_functionality(self):
        """Test that Non-Maximum Suppression removes overlapping candidates."""
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure pipeline is ready
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))
        tokenize.tokenize_all()

        # Generate candidates for all documents
        candidates.generate_all_candidates()

        # Test NMS worked by checking for overlaps
        indexed_docs = ingest.get_indexed_documents()

        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            candidates_df = candidates.get_document_candidates(sha256)

            if len(candidates_df) < 2:
                continue  # Skip if too few candidates to check overlaps

            # Check for high overlaps that should have been removed
            high_overlap_count = 0

            for i in range(len(candidates_df)):
                for j in range(i + 1, len(candidates_df)):
                    bbox1 = (
                        candidates_df.iloc[i]["bbox_norm_x0"],
                        candidates_df.iloc[i]["bbox_norm_y0"],
                        candidates_df.iloc[i]["bbox_norm_x1"],
                        candidates_df.iloc[i]["bbox_norm_y1"],
                    )
                    bbox2 = (
                        candidates_df.iloc[j]["bbox_norm_x0"],
                        candidates_df.iloc[j]["bbox_norm_y0"],
                        candidates_df.iloc[j]["bbox_norm_x1"],
                        candidates_df.iloc[j]["bbox_norm_y1"],
                    )

                    # Compute IoU
                    iou = candidates.compute_overlap_iou(bbox1, bbox2)

                    if iou > 0.5:  # High overlap threshold used in NMS
                        high_overlap_count += 1

            # Should have very few high overlaps after NMS
            overlap_ratio = high_overlap_count / max(len(candidates_df), 1)
            assert (
                overlap_ratio < 0.1
            ), f"Too many overlapping candidates remain after NMS: {high_overlap_count}/{len(candidates_df)}"

        print("✓ NMS functionality test passed")
