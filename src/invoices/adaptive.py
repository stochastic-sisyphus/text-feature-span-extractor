"""Adaptive intelligence layer: learns from data to improve decoder at runtime.

Three features:
1. Document-level label detection — structural tokens (column headers, etc.)
2. Learned anchors — anchor keywords discovered from labeled data
3. Weight optimization — Nelder-Mead tuning of DecoderWeights from ground truth
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import io_utils, labels, paths, schema_registry
from .config import DecoderWeights
from .logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature 1: Document-Level Label Detection
# ---------------------------------------------------------------------------


def detect_document_labels(sha256: str) -> set[str]:
    """Detect structural label tokens (column headers, repeated text).

    Returns set of lowercased label tokens. Empty set if data unavailable.
    """
    tokens_path = paths.get_tokens_path(sha256)
    df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
    if df is None or df.empty or "text" not in df.columns:
        return set()

    label_tokens: set[str] = set()

    # Frequency-based: tokens at >= 90th percentile frequency are structural
    # Use threshold of 5 to avoid flagging legitimate company names that
    # appear 2-3 times on small documents.
    text_lower = df["text"].str.lower().str.strip()
    freq = text_lower.value_counts()
    if len(freq) > 1:
        p90 = float(np.percentile(freq.values, 90))
        threshold = max(p90, 5)  # must appear 5+ times to be structural
        label_tokens.update(freq[freq >= threshold].index)

    # Cross-page: tokens at identical Y-positions across pages
    if "page_idx" in df.columns and "bbox_norm_y0" in df.columns:
        if df["page_idx"].nunique() >= 2:
            dc = df[["text", "page_idx", "bbox_norm_y0"]].copy()
            dc["y_bucket"] = (dc["bbox_norm_y0"] * 50).astype(int)
            dc["text_lower"] = dc["text"].str.lower().str.strip()
            grouped = (
                dc.groupby(["text_lower", "y_bucket"])["page_idx"]
                .nunique()
                .reset_index(name="page_count")
            )
            label_tokens.update(grouped[grouped["page_count"] >= 2]["text_lower"])

    return label_tokens


def detect_cross_page_headers(sha256: str) -> set[str]:
    """Detect tokens appearing at the same Y-position across 3+ pages.

    These are almost certainly structural elements (column headers, address
    headers, page footers) — NOT field values. A vendor name that appears
    on many pages (like "AT&T" in headers) will be caught, but the penalty
    is mild enough for colon-value and other signals to overcome.

    Returns set of lowercased tokens. Empty set if < 3 pages in document.
    """
    tokens_path = paths.get_tokens_path(sha256)
    df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
    if df is None or df.empty:
        return set()
    if "page_idx" not in df.columns or "bbox_norm_y0" not in df.columns:
        return set()
    if df["page_idx"].nunique() < 3:
        return set()

    dc = df[["text", "page_idx", "bbox_norm_y0"]].copy()
    dc["y_bucket"] = (dc["bbox_norm_y0"] * 50).astype(int)
    dc["text_lower"] = dc["text"].str.lower().str.strip()
    grouped = (
        dc.groupby(["text_lower", "y_bucket"])["page_idx"]
        .nunique()
        .reset_index(name="page_count")
    )
    return set(grouped[grouped["page_count"] >= 3]["text_lower"])


def detect_address_city_tokens(sha256: str) -> set[str]:
    """Detect tokens likely to be city names in address blocks.

    Looks for alphabetic words spatially adjacent to 5-digit ZIP codes
    (within 0.15 normalized X distance). These are city names
    (e.g., "Dallas" next to "75202") and should not be vendor names.

    Returns set of lowercased city-like tokens.
    """
    import re

    tokens_path = paths.get_tokens_path(sha256)
    df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
    if df is None or df.empty or "text" not in df.columns:
        return set()
    if "bbox_norm_x0" not in df.columns or "bbox_norm_y0" not in df.columns:
        return set()

    zip_re = re.compile(r"^\d{5}(-\d{4})?$")
    city_tokens: set[str] = set()

    # Find ZIP code token positions (page 0 only, header region y < 0.35)
    # Address blocks are in the header region of the first page.
    # ZIP codes deeper in the document are typically in tables/details.
    zip_positions: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if (
            zip_re.match(text)
            and int(row.get("page_idx", 0)) == 0
            and float(row["bbox_norm_y0"]) < 0.35
        ):
            zip_positions.append(
                (float(row["bbox_norm_x0"]), float(row["bbox_norm_y0"]))
            )

    if not zip_positions:
        return set()

    # Collect alphabetic words near ZIP codes on page 0 (same Y, close X)
    page0 = df[df["page_idx"] == 0]
    for _, row in page0.iterrows():
        text = str(row["text"]).strip()
        if zip_re.match(text):
            continue
        clean = text.rstrip(",.")
        if not clean.isalpha() or len(clean) < 2:
            continue
        tok_x = float(row["bbox_norm_x0"])
        tok_y = float(row["bbox_norm_y0"])
        for zx, zy in zip_positions:
            # Same Y line (within bucket tolerance), to the LEFT of ZIP,
            # and within 0.10 X distance (city names precede ZIP codes)
            if abs(tok_y - zy) < 0.01 and tok_x < zx and (zx - tok_x) < 0.10:
                city_tokens.add(clean.lower())
                break

    return city_tokens


# ---------------------------------------------------------------------------
# Feature 1b: Colon Name Value Detection
# ---------------------------------------------------------------------------


def detect_colon_name_values(sha256: str) -> dict[str, int]:
    """Detect name-like values after colons in the document.

    Finds patterns like "To: GTT Americas LLC" and returns a dict mapping
    each word to its position rank (0 = first word after colon, highest bonus).

    Returns:
        Dict mapping lowercased word -> position rank (0-based, lower = better).
    """
    tokens_path = paths.get_tokens_path(sha256)
    df = io_utils.read_parquet_safe(tokens_path, on_error="empty")
    if df is None or df.empty or "text" not in df.columns:
        return {}

    result: dict[str, int] = {}

    # Sort by page and position for reading order
    sort_cols = ["page_idx"]
    if "bbox_norm_y0" in df.columns:
        sort_cols.append("bbox_norm_y0")
    if "bbox_norm_x0" in df.columns:
        sort_cols.append("bbox_norm_x0")
    df_sorted = df.sort_values(sort_cols).reset_index(drop=True)

    texts = df_sorted["text"].tolist()
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


# ---------------------------------------------------------------------------
# Feature 2: Learned Anchors
# ---------------------------------------------------------------------------


def learn_anchors(data_dir: Path | None = None) -> dict[str, set[str]]:
    """Learn anchor keywords from labeled data by finding tokens near correct values.

    Tokens within 0.15 normalized distance of correct candidates that appear
    across >= 2 documents become learned anchors for that anchor type.

    Returns dict mapping anchor_type -> set of learned anchor words.
    """
    aligned = labels.load_aligned_labels()
    if aligned.empty or "sha256" not in aligned.columns:
        return {}

    anchor_freq: dict[str, dict[str, int]] = {}

    for sha256, group in aligned.groupby("sha256"):
        cands_df = io_utils.read_parquet_safe(
            paths.get_candidates_path(str(sha256)), on_error="empty"
        )
        toks_df = io_utils.read_parquet_safe(
            paths.get_tokens_path(str(sha256)), on_error="empty"
        )
        if cands_df is None or cands_df.empty or toks_df is None or toks_df.empty:
            continue
        if "bbox_norm_x0" not in toks_df.columns:
            continue

        for _, row in group.iterrows():
            cand_idx = row.get("candidate_idx")
            if cand_idx is None or pd.isna(cand_idx):
                continue
            cand_idx = int(cand_idx)
            if cand_idx >= len(cands_df):
                continue

            cand = cands_df.iloc[cand_idx]
            cx = float((cand.get("bbox_norm_x0", 0) + cand.get("bbox_norm_x1", 0)) / 2)
            cy = float((cand.get("bbox_norm_y0", 0) + cand.get("bbox_norm_y1", 0)) / 2)
            cand_text = str(cand.get("raw_text", "")).lower().strip()

            anchor_type = schema_registry.anchor_type(row["field"])
            if anchor_type is None:
                continue
            if anchor_type not in anchor_freq:
                anchor_freq[anchor_type] = {}

            for tok in toks_df.itertuples(index=False):
                tok_cx = (tok.bbox_norm_x0 + tok.bbox_norm_x1) / 2
                tok_cy = (tok.bbox_norm_y0 + tok.bbox_norm_y1) / 2
                if math.sqrt((cx - tok_cx) ** 2 + (cy - tok_cy) ** 2) > 0.15:
                    continue
                tok_text = tok.text.lower().strip()
                if tok_text == cand_text or len(tok_text) < 2:
                    continue
                # Track unique docs via composite key
                anchor_freq[anchor_type][f"{tok_text}|{sha256}"] = 1

    # Consolidate: count distinct docs per anchor word, require >= 2
    result: dict[str, set[str]] = {}
    for atype, freq_map in anchor_freq.items():
        word_docs: dict[str, int] = {}
        for key in freq_map:
            word = key.split("|")[0]
            word_docs[word] = word_docs.get(word, 0) + 1
        learned = {w for w, c in word_docs.items() if c >= 2}
        if learned:
            result[atype] = learned

    total_anchors = sum(len(v) for v in result.values())
    if result:
        logger.info(
            "learned_anchors",
            total=total_anchors,
            types={k: len(v) for k, v in result.items()},
        )
    else:
        logger.debug("learned_anchors_none")

    return result


# ---------------------------------------------------------------------------
# Feature 3: Weight Optimization
# ---------------------------------------------------------------------------

_TUNABLE_WEIGHTS = (
    "HEADER_REGION_BONUS",
    "BUCKET_MATCH_BONUS",
    "DIRECTIONAL_BELOW_CLOSE_BONUS",
    "PROXIMITY_WEIGHT",
    "TEXT_PATTERN_NEGATIVE_AMPLIFIER",
    "FOOTER_REGION_BONUS",
)


def tune_weights(data_dir: Path | None = None) -> dict[str, float] | None:
    """Optimize decoder weights using Nelder-Mead on labeled data.

    Minimizes negative accuracy across labeled documents by tuning 6 key
    DecoderWeights parameters. Returns None if insufficient data (< 2 docs).
    Saves results to data/tuned_weights.json.
    """
    from scipy.optimize import minimize  # type: ignore[import-untyped]

    from . import candidates as cand_mod
    from .decoder import compute_weak_prior_cost

    aligned = labels.load_aligned_labels()
    if aligned.empty or len(aligned.groupby("sha256")) < 2:
        logger.info("tune_weights_insufficient_data")
        return None

    # Collect labeled documents
    labeled_docs: list[tuple[pd.DataFrame, dict[str, int]]] = []
    for sha256, group in aligned.groupby("sha256"):
        cands_df = cand_mod.get_document_candidates(str(sha256))
        if cands_df.empty:
            continue
        field_map: dict[str, int] = {}
        for _, row in group.iterrows():
            ci = row.get("candidate_idx")
            if ci is not None and not pd.isna(ci):
                field_map[row["field"]] = int(ci)
        if field_map:
            labeled_docs.append((cands_df, field_map))

    if len(labeled_docs) < 2:
        logger.info("tune_weights_insufficient_labeled_docs", count=len(labeled_docs))
        return None

    from .config import DWeights as current_weights

    x0 = np.array([getattr(current_weights, w) for w in _TUNABLE_WEIGHTS])

    def objective(x: np.ndarray) -> float:
        overrides = dict(zip(_TUNABLE_WEIGHTS, x, strict=True))
        temp_weights = DecoderWeights(**overrides)
        import invoices.config as config_mod

        original = config_mod.DWeights
        object.__setattr__(config_mod, "DWeights", temp_weights)
        try:
            correct = total = 0
            for cands_df, field_map in labeled_docs:
                cands_list = cands_df.to_dict("records")
                for field, expected_idx in field_map.items():
                    if expected_idx >= len(cands_list):
                        continue
                    best_idx = min(
                        range(len(cands_list)),
                        key=lambda i: compute_weak_prior_cost(field, cands_list[i]),
                    )
                    correct += best_idx == expected_idx
                    total += 1
        finally:
            object.__setattr__(config_mod, "DWeights", original)
        return -(correct / total) if total > 0 else 0.0

    result = minimize(objective, x0, method="Nelder-Mead", options={"maxiter": 200})
    tuned = dict(zip(_TUNABLE_WEIGHTS, result.x, strict=True))

    out_path = paths.get_data_dir() / "tuned_weights.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tuned, f, indent=2)

    logger.info("weights_tuned", accuracy=-result.fun, iterations=result.nit)
    return tuned
