"""Tokenization module for deterministic text extraction from PDFs using pdfplumber."""

import hashlib
import time
from typing import Any

import pandas as pd
import pdfplumber
from pdfminer.pdfparser import PDFSyntaxError
from tqdm import tqdm

from . import ingest, io_utils, paths, utils
from .exceptions import TokenizationError
from .logging import get_logger
from .metrics import pipeline_duration, pipeline_errors

logger = get_logger(__name__)


# =============================================================================
# ERROR HANDLING HELPERS
# =============================================================================


def _raise_tokenization_error(
    doc_id: str, sha256: str, error: Exception, error_context: str
) -> None:
    """Log and raise a TokenizationError.

    Args:
        doc_id: Document identifier
        sha256: Document SHA256 hash
        error: The original exception
        error_context: Context description for the error message
    """
    logger.error(
        "tokenization_failed",
        doc_id=doc_id,
        sha256=sha256[:16],
        error_type=type(error).__name__,
        reason=str(error),
    )
    raise TokenizationError(sha256, f"{error_context}: {error}") from error


def _mark_tokenization_skipped(sha256: str, error: Exception) -> None:
    """Log a tokenization skip warning.

    Args:
        sha256: Document SHA256 hash
        error: The exception that caused the skip
    """
    reason = error.reason if hasattr(error, "reason") else str(error)
    logger.warning(
        "tokenize_all_skipped_document",
        sha256=sha256[:16],
        error_type=type(error).__name__,
        reason=reason,
    )


def extract_page_tokens(page: Any, page_idx: int, doc_id: str) -> list[dict[str, Any]]:
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
        # Note: pdfplumber uses 'x0', 'x1' for horizontal and 'top', 'bottom' for vertical
        x0 = float(word.get("x0", 0))
        y0 = float(word.get("top", 0))  # pdfplumber uses 'top', not 'y0'
        x1 = float(word.get("x1", 0))
        y1 = float(word.get("bottom", 0))  # pdfplumber uses 'bottom', not 'y1'

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
        font_hash = hashlib.md5(
            str(font_name).encode(), usedforsecurity=False
        ).hexdigest()[:16]

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


def tokenize_document(sha256: str, chunk_pages: int = 20) -> int:
    """
    Tokenize a single PDF document with streaming to avoid memory bombs.

    Args:
        sha256: Document SHA256 hash
        chunk_pages: Number of pages to process before writing (default: 20)

    Returns:
        Number of tokens extracted
    """
    start_time = time.perf_counter()
    try:
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
        existing_df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
        if existing_df is not None:
            logger.debug(
                "tokens_already_exist", sha256=sha256[:16], token_count=len(existing_df)
            )
            return len(existing_df)

        chunk_tokens = []
        total_tokens = 0
        with pdfplumber.open(raw_pdf_path) as pdf:
            logger.info("tokenizing_document", doc_id=doc_id, pages=len(pdf.pages))

            for page_idx, page in enumerate(pdf.pages):
                page_tokens = extract_page_tokens(page, page_idx, doc_id)
                chunk_tokens.extend(page_tokens)

                # Write chunk when buffer is full
                if (page_idx + 1) % chunk_pages == 0:
                    if chunk_tokens:
                        df_chunk = pd.DataFrame(chunk_tokens)
                        if total_tokens == 0:
                            # First chunk: write fresh
                            io_utils.write_parquet_safe(df_chunk, tokens_path)
                        else:
                            # Subsequent chunks: append
                            io_utils.append_to_parquet(df_chunk, tokens_path)
                        total_tokens += len(chunk_tokens)
                        chunk_tokens = []

            # Write remaining tokens
            if chunk_tokens:
                df_chunk = pd.DataFrame(chunk_tokens)
                if total_tokens == 0:
                    # Only chunk: write fresh
                    io_utils.write_parquet_safe(df_chunk, tokens_path)
                else:
                    # Final chunk: append
                    io_utils.append_to_parquet(df_chunk, tokens_path)
                total_tokens += len(chunk_tokens)

            if total_tokens > 0:
                logger.info("tokens_extracted", doc_id=doc_id, token_count=total_tokens)

                return total_tokens
            else:
                logger.warning("no_tokens_found", doc_id=doc_id)
                # Create empty file to mark as processed
                empty_df = pd.DataFrame(
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
                )
                io_utils.write_parquet_safe(empty_df, tokens_path)

                return 0

    except PDFSyntaxError as e:
        # PDF parsing error (corrupted or malformed PDF)
        pipeline_errors.labels(stage="tokenize").inc()

        _raise_tokenization_error(doc_id, sha256, e, "PDF syntax error")
    except OSError as e:
        # File system errors
        pipeline_errors.labels(stage="tokenize").inc()

        _raise_tokenization_error(doc_id, sha256, e, "File error")
    except (ValueError, TypeError, KeyError) as e:
        # Data extraction or processing errors
        pipeline_errors.labels(stage="tokenize").inc()

        _raise_tokenization_error(doc_id, sha256, e, "Processing error")
    finally:
        duration = time.perf_counter() - start_time
        pipeline_duration.labels(stage="tokenize").observe(duration)

    return 0


def tokenize_all() -> dict[str, int]:
    """
    Tokenize all documents in the index.

    Returns:
        Dictionary mapping sha256 to token count
    """
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        logger.info("no_documents_in_index")
        return {}

    results = {}

    logger.info("tokenizing_all", doc_count=len(indexed_docs))

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]

        try:
            token_count = tokenize_document(sha256)
            results[sha256] = token_count
        except TokenizationError as e:
            # Typed tokenization error (already logged in tokenize_document)
            _mark_tokenization_skipped(sha256, e)
            results[sha256] = 0
        except (FileNotFoundError, ValueError) as e:
            # Document not found or invalid state
            _mark_tokenization_skipped(sha256, e)
            results[sha256] = 0

    return results


def get_document_tokens(sha256: str) -> pd.DataFrame:
    """Get tokens for a specific document."""
    tokens_path = paths.get_tokens_path(sha256)
    df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
    return df if df is not None else pd.DataFrame()


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
        "page_count": (
            tokens_df["page_idx"].nunique() if "page_idx" in tokens_df.columns else 0
        ),
        "first_token_id": tokens_df.iloc[0]["token_id"] if len(tokens_df) > 0 else None,
        "last_token_id": tokens_df.iloc[-1]["token_id"] if len(tokens_df) > 0 else None,
    }
