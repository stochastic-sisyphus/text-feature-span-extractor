"""Test idempotent ingestion behavior."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import pytest

from invoices import ingest, paths


class TestIdempotency:
    """Test that ingestion is idempotent."""

    def test_ingest_idempotency(self):
        """Test that running ingest twice doesn't create duplicates."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure clean state
        paths.ensure_directories()

        # First ingestion
        count1 = ingest.ingest_seed_folder(str(seed_folder))

        # Get index after first run
        index_path = paths.get_ingest_index_path()
        assert index_path.exists(), "Index file should exist after first ingestion"

        df1 = pd.read_parquet(index_path)
        row_count1 = len(df1)
        sha256s1 = set(df1["sha256"].tolist())

        # Check that files exist in raw directory
        raw_dir = paths.get_ingest_raw_dir()
        raw_files1 = list(raw_dir.glob("*.pdf"))

        # The count may be 0 if files were already ingested, but files should exist
        if count1 > 0:
            assert (
                len(raw_files1) == count1
            ), "Number of raw files should match newly ingested count"
        else:
            # Files already existed, should have same number as index rows
            assert (
                len(raw_files1) == row_count1
            ), "Raw files should match index row count for previously ingested files"

        # Second ingestion (should be idempotent)
        count2 = ingest.ingest_seed_folder(str(seed_folder))

        # Should ingest 0 new documents
        assert count2 == 0, "Second ingestion should add 0 new documents"

        # Get index after second run
        df2 = pd.read_parquet(index_path)
        row_count2 = len(df2)
        sha256s2 = set(df2["sha256"].tolist())

        # Check that row count hasn't increased
        assert (
            row_count2 == row_count1
        ), "Index row count should not increase on second run"

        # Check that SHA256s are identical
        assert sha256s2 == sha256s1, "SHA256 sets should be identical"

        # Check that no duplicate files were created
        raw_files2 = list(raw_dir.glob("*.pdf"))
        assert len(raw_files2) == len(
            raw_files1
        ), "Number of raw files should not increase"

        # Verify no duplicate SHA256s in index
        assert len(df2["sha256"].unique()) == len(
            df2
        ), "No duplicate SHA256s should exist in index"

        print(
            f"✓ Idempotency test passed: {count1} documents ingested, 0 duplicates on re-run"
        )
