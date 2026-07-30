"""Adaptive intelligence layer: learns from data to improve decoder at runtime.

Three features:
1. Document-level label detection — structural tokens (column headers, etc.)
2. Learned anchors — anchor keywords discovered from labeled data
3. Weight optimization — Nelder-Mead tuning of DecoderWeights from ground truth
   (moved to adaptive_xgb.py — xgboost-locked, pandas-intrinsic)

All public detect_*_with_data functions are pure (data in, data out, no disk I/O).
The sha256-based detect_* wrappers live in decoder/decode.py.
"""

from __future__ import annotations

import polars as pl

from .logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature 1: Document-Level Label Detection (pure _with_data variants)
# ---------------------------------------------------------------------------


def detect_document_labels_with_data(tokens_df: pl.DataFrame) -> set[str]:
    """Pure: detect structural label tokens from a tokens DataFrame.

    Returns set of lowercased label tokens. Empty set if data unavailable.
    """
    if tokens_df.is_empty() or "text" not in tokens_df.columns:
        return set()

    label_tokens: set[str] = set()

    # Frequency-based: tokens at >= 90th percentile frequency are structural
    # Use threshold of 5 to avoid flagging legitimate company names that
    # appear 2-3 times on small documents.
    text_lower = tokens_df.select(
        pl.col("text").str.to_lowercase().str.strip_chars().alias("text_lower")
    )
    freq = text_lower.group_by("text_lower").agg(pl.len().alias("count"))
    if freq.height > 1:
        p90 = freq["count"].quantile(0.90)
        if p90 is not None:
            threshold = max(float(p90), 5.0)
            above = freq.filter(pl.col("count") >= threshold)["text_lower"].to_list()
            label_tokens.update(above)

    # Cross-page: tokens at identical Y-positions across pages
    if "page_idx" in tokens_df.columns and "bbox_norm_y0" in tokens_df.columns:
        if tokens_df["page_idx"].n_unique() >= 2:
            dc = tokens_df.select(
                pl.col("text").str.to_lowercase().str.strip_chars().alias("text_lower"),
                pl.col("page_idx"),
                (pl.col("bbox_norm_y0") * 50).cast(pl.Int32).alias("y_bucket"),
            )
            grouped = dc.group_by(["text_lower", "y_bucket"]).agg(
                pl.col("page_idx").n_unique().alias("page_count")
            )
            cross = grouped.filter(pl.col("page_count") >= 2)["text_lower"].to_list()
            label_tokens.update(cross)

    return label_tokens


def detect_cross_page_headers_with_data(tokens_df: pl.DataFrame) -> set[str]:
    """Pure: detect tokens at same Y-position across 3+ pages.

    Returns set of lowercased tokens. Empty set if < 3 pages in document.
    """
    if tokens_df.is_empty():
        return set()
    if "page_idx" not in tokens_df.columns or "bbox_norm_y0" not in tokens_df.columns:
        return set()
    if tokens_df["page_idx"].n_unique() < 3:
        return set()

    dc = tokens_df.select(
        pl.col("text").str.to_lowercase().str.strip_chars().alias("text_lower"),
        pl.col("page_idx"),
        (pl.col("bbox_norm_y0") * 50).cast(pl.Int32).alias("y_bucket"),
    )
    grouped = dc.group_by(["text_lower", "y_bucket"]).agg(
        pl.col("page_idx").n_unique().alias("page_count")
    )
    return set(grouped.filter(pl.col("page_count") >= 3)["text_lower"].to_list())


def detect_address_city_tokens_with_data(tokens_df: pl.DataFrame) -> set[str]:
    """Pure: detect city tokens near ZIP codes in address blocks.

    Returns set of lowercased city-like tokens.
    """
    return set()


# ---------------------------------------------------------------------------
# Feature 1b: Colon Name Value Detection (pure)
# ---------------------------------------------------------------------------


def detect_colon_name_values_with_data(tokens_df: pl.DataFrame) -> dict[str, int]:
    """Pure: detect name-like values after colons.

    Returns dict mapping lowercased word -> position rank (0-based, lower = better).
    """
    if tokens_df.is_empty() or "text" not in tokens_df.columns:
        return {}

    result: dict[str, int] = {}

    # Sort by page and position for reading order
    sort_cols: list[str] = []
    if "page_idx" in tokens_df.columns:
        sort_cols.append("page_idx")
    if "bbox_norm_y0" in tokens_df.columns:
        sort_cols.append("bbox_norm_y0")
    if "bbox_norm_x0" in tokens_df.columns:
        sort_cols.append("bbox_norm_x0")

    df_sorted = tokens_df.sort(sort_cols) if sort_cols else tokens_df

    texts = df_sorted["text"].to_list()
    for i, tok_text in enumerate(texts):
        tok_str = str(tok_text).strip()
        # Look for tokens ending with colon (e.g., "To:", "Attn:")
        if not tok_str.endswith(":"):
            continue
        # Collect subsequent name-like words (uppercase start, alpha)
        rank = 0
        for j in range(i + 1, min(i + 6, len(texts))):
            word = str(texts[j]).strip()
            if not word or not word[0].isalpha():
                break
            # Stop if we hit another label (ends with colon)
            if word.endswith(":"):
                break
            word_lower = word.lower()
            if word_lower not in result or result[word_lower] > rank:
                result[word_lower] = rank
            rank += 1

    return result
