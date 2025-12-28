"""Test Label Studio alignment with synthetic data."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from invoices import labels, normalize


class TestLabelAlignment:
    """Test IoU-based label alignment with synthetic data."""

    def test_char_iou_computation(self):
        """Test character-level IoU computation."""
        # Perfect overlap
        iou = labels.char_iou(10, 20, 10, 20)
        assert iou == 1.0, "Perfect overlap should have IoU = 1.0"

        # No overlap
        iou = labels.char_iou(10, 20, 25, 30)
        assert iou == 0.0, "No overlap should have IoU = 0.0"

        # Partial overlap
        iou = labels.char_iou(10, 20, 15, 25)
        # Intersection: [15, 20] = 5 chars
        # Union: (20-10) + (25-15) - 5 = 10 + 10 - 5 = 15 chars
        # IoU = 5/15 = 1/3
        expected = 5.0 / 15.0
        assert abs(iou - expected) < 0.001, f"Expected IoU {expected}, got {iou}"

        print("✓ Character IoU computation verified")

    def test_bbox_iou_computation(self):
        """Test bounding box IoU computation."""
        # Perfect overlap
        bbox1 = [0.1, 0.1, 0.3, 0.2]
        bbox2 = [0.1, 0.1, 0.3, 0.2]
        iou = labels.compute_iou(bbox1, bbox2)
        assert iou == 1.0, "Perfect overlap should have IoU = 1.0"

        # No overlap
        bbox1 = [0.1, 0.1, 0.2, 0.2]
        bbox2 = [0.3, 0.3, 0.4, 0.4]
        iou = labels.compute_iou(bbox1, bbox2)
        assert iou == 0.0, "No overlap should have IoU = 0.0"

        # Partial overlap
        bbox1 = [0.1, 0.1, 0.3, 0.2]  # width=0.2, height=0.1, area=0.02
        bbox2 = [0.2, 0.1, 0.4, 0.2]  # width=0.2, height=0.1, area=0.02
        iou = labels.compute_iou(bbox1, bbox2)
        # Intersection: [0.2, 0.1, 0.3, 0.2] = width=0.1, height=0.1, area=0.01
        # Union: 0.02 + 0.02 - 0.01 = 0.03
        # IoU = 0.01/0.03 = 1/3
        expected = 1.0 / 3.0
        assert abs(iou - expected) < 0.001, f"Expected IoU {expected}, got {iou}"

        print("✓ Bounding box IoU computation verified")

    def test_synthetic_alignment_workflow(self):
        """Test synthetic label alignment with mock data."""
        # This test verifies the alignment logic without requiring real PDFs

        # Create mock candidates DataFrame
        candidates_data = [
            {
                "candidate_id": "test_doc_0_0",
                "doc_id": "test_doc",
                "sha256": "test_sha",
                "page_idx": 0,
                "bbox_norm_x0": 0.1,
                "bbox_norm_y0": 0.1,
                "bbox_norm_x1": 0.3,
                "bbox_norm_y1": 0.2,
                "raw_text": "INV-12345",
                "bucket": "id_like"
            },
            {
                "candidate_id": "test_doc_0_1",
                "doc_id": "test_doc",
                "sha256": "test_sha",
                "page_idx": 0,
                "bbox_norm_x0": 0.7,
                "bbox_norm_y0": 0.1,
                "bbox_norm_x1": 0.9,
                "bbox_norm_y1": 0.15,
                "raw_text": "$1,234.56",
                "bucket": "amount_like"
            }
        ]

        candidates_df = pd.DataFrame(candidates_data)

        # Create mock label (overlaps with first candidate)
        label_bbox = [0.1, 0.1, 0.3, 0.2]  # Perfect match with first candidate

        # Find best match
        best_iou = 0.0
        best_idx = None

        for idx, candidate in candidates_df.iterrows():
            candidate_bbox = [
                candidate["bbox_norm_x0"],
                candidate["bbox_norm_y0"],
                candidate["bbox_norm_x1"],
                candidate["bbox_norm_y1"]
            ]

            iou = labels.compute_iou(label_bbox, candidate_bbox)

            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        # Should match first candidate perfectly
        assert best_idx == 0, f"Should match first candidate, got index {best_idx}"
        assert best_iou == 1.0, f"Should have perfect IoU=1.0, got {best_iou}"

        print("✓ Synthetic alignment workflow verified")

    def test_normalization_guards(self):
        """Test that normalization version and checksums work."""
        # Test the normalization guard functions
        test_text = "Sample invoice text for testing"

        # Get checksum
        checksum1 = normalize.text_len_checksum(test_text)
        checksum2 = normalize.text_len_checksum(test_text)

        # Should be deterministic
        assert checksum1 == checksum2, "Checksums should be deterministic"

        # Different text should give different checksum
        different_text = "Different text"
        checksum3 = normalize.text_len_checksum(different_text)
        assert checksum1 != checksum3, "Different text should have different checksum"

        # Check normalize version exists
        assert hasattr(normalize, 'NORMALIZE_VERSION'), "Should have NORMALIZE_VERSION constant"
        assert normalize.NORMALIZE_VERSION, "NORMALIZE_VERSION should not be empty"

        print(f"✓ Normalization guards verified (version: {normalize.NORMALIZE_VERSION})")

    def test_empty_label_handling(self):
        """Test that empty label inputs are handled gracefully."""
        # Test align_labels with no files
        result = labels.align_labels()

        # Should handle empty state gracefully
        assert result["status"] in ["no_labels", "no_files"], "Should handle empty state"
        assert result.get("total_aligned", 0) == 0, "Should report 0 alignments"

        # Test load_aligned_labels with no data
        aligned_df = labels.load_aligned_labels()
        assert aligned_df.empty, "Should return empty DataFrame when no aligned labels"

        print("✓ Empty label handling verified")

    def test_label_import_structure(self):
        """Test label import structure validation."""
        # Test that import_labels validates file existence
        with pytest.raises(FileNotFoundError):
            labels.import_labels("nonexistent_file.json")

        print("✓ Label import structure validation verified")
