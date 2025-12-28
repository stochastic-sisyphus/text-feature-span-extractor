"""Test deterministic tokenization behavior."""

import sys
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from invoices import ingest, paths, tokenize, utils


class TestTokenDeterminism:
    """Test that tokenization is deterministic."""

    def test_token_determinism(self):
        """Test that tokenizing the same PDF twice produces identical results."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure clean state and ingest PDFs
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))

        # Get first document from index
        indexed_docs = ingest.get_indexed_documents()
        assert not indexed_docs.empty, "Should have ingested documents"

        first_doc = indexed_docs.iloc[0]
        sha256 = first_doc["sha256"]
        doc_id = first_doc["doc_id"]

        print(f"Testing determinism for document: {doc_id}")

        # Delete existing tokens if present (to force re-tokenization)
        tokens_path = paths.get_tokens_path(sha256)
        if tokens_path.exists():
            tokens_path.unlink()

        # First tokenization
        token_count1 = tokenize.tokenize_document(sha256)
        tokens_df1 = tokenize.get_document_tokens(sha256)
        summary1 = tokenize.get_token_summary(sha256)

        # Verify tokens were created
        assert token_count1 > 0, "Should have extracted tokens"
        assert not tokens_df1.empty, "Tokens DataFrame should not be empty"
        assert summary1["token_count"] == token_count1, "Token counts should match"

        # Get first and last token IDs
        first_token_id1 = summary1["first_token_id"]
        last_token_id1 = summary1["last_token_id"]

        assert first_token_id1 is not None, "Should have first token ID"
        assert last_token_id1 is not None, "Should have last token ID"

        # Delete tokens file to force re-tokenization
        tokens_path.unlink()

        # Second tokenization
        token_count2 = tokenize.tokenize_document(sha256)
        tokens_df2 = tokenize.get_document_tokens(sha256)
        summary2 = tokenize.get_token_summary(sha256)

        # Get first and last token IDs from second run
        first_token_id2 = summary2["first_token_id"]
        last_token_id2 = summary2["last_token_id"]

        # Assert identical results
        assert (
            token_count2 == token_count1
        ), "Token counts should be identical across runs"
        assert (
            summary2["token_count"] == summary1["token_count"]
        ), "Summary token counts should match"
        assert (
            summary2["page_count"] == summary1["page_count"]
        ), "Page counts should match"
        assert first_token_id2 == first_token_id1, "First token ID should be identical"
        assert last_token_id2 == last_token_id1, "Last token ID should be identical"

        # Check that DataFrames are identical
        assert len(tokens_df2) == len(
            tokens_df1
        ), "DataFrame lengths should be identical"

        # Compare key columns that should be deterministic
        deterministic_columns = [
            "token_id",
            "page_idx",
            "token_idx",
            "text",
            "bbox_norm_x0",
            "bbox_norm_y0",
            "bbox_norm_x1",
            "bbox_norm_y1",
            "font_hash",
            "font_size",
            "is_bold",
            "is_italic",
        ]

        for col in deterministic_columns:
            if col in tokens_df1.columns and col in tokens_df2.columns:
                assert (
                    tokens_df1[col].tolist() == tokens_df2[col].tolist()
                ), f"Column {col} should be identical"

        print(
            f"✓ Token determinism test passed: {token_count1} tokens, consistent token IDs"
        )

    def test_token_id_computation(self):
        """Test that token ID computation is correct and stable."""
        # Get the seed PDFs directory
        seed_folder = Path(__file__).parent.parent / "seed_pdfs"

        if not seed_folder.exists():
            pytest.skip(f"Seed folder not found: {seed_folder}")

        # Ensure PDFs are ingested
        paths.ensure_directories()
        ingest.ingest_seed_folder(str(seed_folder))

        # Get a document
        indexed_docs = ingest.get_indexed_documents()
        assert not indexed_docs.empty, "Should have ingested documents"

        first_doc = indexed_docs.iloc[0]
        sha256 = first_doc["sha256"]

        # Tokenize
        tokenize.tokenize_document(sha256)
        tokens_df = tokenize.get_document_tokens(sha256)

        if tokens_df.empty:
            pytest.skip("No tokens found in document")

        # Check that token IDs are unique
        token_ids = tokens_df["token_id"].tolist()
        assert len(token_ids) == len(set(token_ids)), "All token IDs should be unique"

        # Verify token ID computation for first few tokens
        for i in range(min(3, len(tokens_df))):
            token = tokens_df.iloc[i]
            doc_id = token["doc_id"]
            page_idx = token["page_idx"]
            token_idx = token["token_idx"]
            text = token["text"]
            bbox_norm = (
                token["bbox_norm_x0"],
                token["bbox_norm_y0"],
                token["bbox_norm_x1"],
                token["bbox_norm_y1"],
            )

            # Manually compute token ID
            expected_token_id = utils.compute_stable_token_id(
                doc_id, page_idx, token_idx, text, bbox_norm
            )
            actual_token_id = token["token_id"]

            assert (
                actual_token_id == expected_token_id
            ), f"Token ID mismatch for token {i}: {actual_token_id} != {expected_token_id}"

        print(f"✓ Token ID computation test passed: {len(token_ids)} unique token IDs")
