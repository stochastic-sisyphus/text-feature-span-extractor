"""Tokenization module for deterministic text extraction from PDFs using pdfplumber."""

import hashlib
from typing import Any

import pandas as pd
import pdfplumber
from tqdm import tqdm

from . import ingest, paths, utils


def extract_page_tokens(page, page_idx: int, doc_id: str) -> list[dict[str, Any]]:
    """
    Extract tokens from a single PDF page.

    Args:
        page: pdfplumber page object
        page_idx: 0-based page index
        doc_id: Document identifier

    Returns:
        List of token dictionaries
    """
    tokens = []

    # Get page dimensions
    page_width = float(page.width)
    page_height = float(page.height)

    # Extract words with their properties
    words = page.extract_words()

    for token_idx, word in enumerate(words):
        # Extract text
        text = str(word.get("text", ""))

        # Skip empty text
        if not text.strip():
            continue

        # Extract bounding box in PDF units
        x0 = float(word.get("x0", 0))
        y0 = float(word.get("y0", 0))
        x1 = float(word.get("x1", 0))
        y1 = float(word.get("y1", 0))

        # Normalize bounding box to [0,1]
        bbox_norm = (
            x0 / page_width if page_width > 0 else 0,
            y0 / page_height if page_height > 0 else 0,
            x1 / page_width if page_width > 0 else 0,
            y1 / page_height if page_height > 0 else 0,
        )

        # Extract font properties
        font_name = word.get("fontname", "")
        font_size = float(word.get("size", 0))

        # Hash font name for consistency (as allowed by spec)
        font_hash = hashlib.md5(str(font_name).encode()).hexdigest()[:16]

        # Infer bold/italic from font name (basic heuristics)
        font_name_lower = str(font_name).lower()
        is_bold = any(
            indicator in font_name_lower for indicator in ["bold", "heavy", "black"]
        )
        is_italic = any(
            indicator in font_name_lower for indicator in ["italic", "oblique"]
        )

        # Simple color bucket (pdfplumber may not always have color info)
        # Default to 'black' if no color information
        color_bucket = "black"  # Simplified for now

        # Simple line_id and block_id based on y-coordinate clustering
        # For deterministic results, use discretized y-coordinates
        line_id = int(y0 // 2)  # Cluster by 2-point intervals
        block_id = int(y0 // 10)  # Larger clusters for blocks

        # Reading order: simple left-to-right, top-to-bottom
        reading_order = page_idx * 10000 + int(y0) * 100 + int(x0)

        # Compute stable token ID
        token_id = utils.compute_stable_token_id(
            doc_id, page_idx, token_idx, text, bbox_norm
        )

        token = {
            "token_id": token_id,
            "doc_id": doc_id,
            "page_idx": page_idx,
            "token_idx": token_idx,
            "text": text,
            "bbox_pdf_units_x0": x0,
            "bbox_pdf_units_y0": y0,
            "bbox_pdf_units_x1": x1,
            "bbox_pdf_units_y1": y1,
            "bbox_norm_x0": bbox_norm[0],
            "bbox_norm_y0": bbox_norm[1],
            "bbox_norm_x1": bbox_norm[2],
            "bbox_norm_y1": bbox_norm[3],
            "page_width": page_width,
            "page_height": page_height,
            "font_name": font_name,
            "font_hash": font_hash,
            "font_size": font_size,
            "is_bold": is_bold,
            "is_italic": is_italic,
            "color_bucket": color_bucket,
            "line_id": line_id,
            "block_id": block_id,
            "reading_order": reading_order,
        }

        tokens.append(token)

    return tokens


def tokenize_document(sha256: str) -> int:
    """
    Tokenize a single PDF document.

    Args:
        sha256: Document SHA256 hash

    Returns:
        Number of tokens extracted
    """
    # Get document info
    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found in index: {sha256}")

    doc_id = doc_info["doc_id"]
    raw_pdf_path = paths.get_raw_pdf_path(sha256)

    if not raw_pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {raw_pdf_path}")

    tokens_path = paths.get_tokens_path(sha256)

    # Skip if already tokenized (idempotency)
    if tokens_path.exists():
        existing_df = pd.read_parquet(tokens_path)
        print(f"Tokens already exist for {sha256[:16]}: {len(existing_df)} tokens")
        return len(existing_df)

    all_tokens = []

    try:
        with pdfplumber.open(raw_pdf_path) as pdf:
            print(f"Tokenizing {doc_id} ({len(pdf.pages)} pages)")

            for page_idx, page in enumerate(pdf.pages):
                page_tokens = extract_page_tokens(page, page_idx, doc_id)
                all_tokens.extend(page_tokens)

        # Convert to DataFrame and save
        if all_tokens:
            df = pd.DataFrame(all_tokens)
            df.to_parquet(tokens_path, index=False)

            print(f"Extracted {len(all_tokens)} tokens for {doc_id}")
            return len(all_tokens)
        else:
            print(f"No tokens found in {doc_id}")
            # Create empty file to mark as processed
            pd.DataFrame(
                columns=[
                    "token_id",
                    "doc_id",
                    "page_idx",
                    "token_idx",
                    "text",
                    "bbox_pdf_units_x0",
                    "bbox_pdf_units_y0",
                    "bbox_pdf_units_x1",
                    "bbox_pdf_units_y1",
                    "bbox_norm_x0",
                    "bbox_norm_y0",
                    "bbox_norm_x1",
                    "bbox_norm_y1",
                    "page_width",
                    "page_height",
                    "font_name",
                    "font_hash",
                    "font_size",
                    "is_bold",
                    "is_italic",
                    "color_bucket",
                    "line_id",
                    "block_id",
                    "reading_order",
                ]
            ).to_parquet(tokens_path, index=False)
            return 0

    except Exception as e:
        print(f"Error tokenizing {doc_id}: {e}")
        raise


def tokenize_all() -> dict[str, int]:
    """
    Tokenize all documents in the index.

    Returns:
        Dictionary mapping sha256 to token count
    """
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        print("No documents found in index")
        return {}

    results = {}

    print(f"Tokenizing {len(indexed_docs)} documents")

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]

        try:
            token_count = tokenize_document(sha256)
            results[sha256] = token_count
        except Exception as e:
            print(f"Failed to tokenize {sha256[:16]}: {e}")
            results[sha256] = 0

    return results


def get_document_tokens(sha256: str) -> pd.DataFrame:
    """Get tokens for a specific document."""
    tokens_path = paths.get_tokens_path(sha256)

    if not tokens_path.exists():
        return pd.DataFrame()

    return pd.read_parquet(tokens_path)


def get_token_summary(sha256: str) -> dict[str, Any]:
    """Get summary statistics for document tokens."""
    tokens_df = get_document_tokens(sha256)

    if tokens_df.empty:
        return {
            "sha256": sha256,
            "token_count": 0,
            "page_count": 0,
            "first_token_id": None,
            "last_token_id": None,
        }

    return {
        "sha256": sha256,
        "token_count": len(tokens_df),
        "page_count": tokens_df["page_idx"].nunique()
        if "page_idx" in tokens_df.columns
        else 0,
        "first_token_id": tokens_df.iloc[0]["token_id"] if len(tokens_df) > 0 else None,
        "last_token_id": tokens_df.iloc[-1]["token_id"] if len(tokens_df) > 0 else None,
    }
