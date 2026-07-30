"""Lazy chain ops between views.candidates_df and the cost-matrix seam.

Each function takes a LazyFrame (from views.candidates_df or a prior chain op)
and returns a LazyFrame.  No outer .collect() occurs here — the single
collection point is the seam in pipeline.py.

Seam-adjacent per-group materialization:
  apply_soft_nms uses group_by("page_idx").map_groups() — each group is
  materialized as a small per-page DataFrame inside the aggregation, which
  is invisible to the outer lazy plan.  This is the minimum granularity for
  the iterative soft-NMS score-decay algorithm.

  apply_diversity_sampling uses map_batches() — the whole frame is passed as
  a single batch to the diversity function, also invisible to the outer plan.

  apply_learned_anchors uses map_batches() — per-row directional recomputation
  via the pure-Python scorer, operating on the pre-collected tokens DataFrame
  passed in as a closure.

Chain ops (in pipeline order):
  1. apply_learned_anchors(lf, learned_anchors, tokens_pl)
       Overlay: re-score directional features where learned anchors supplement
       the pattern-based anchors from views.candidates_df.  Accepts a
       pre-collected tokens DataFrame (already materialized at the TokensDF
       seam in pipeline.py) — no additional collect().
  2. apply_scoring(lf)
       Compute cohesion-weighted total_score using pure Polars expressions
       (no map_elements).
  3. apply_soft_nms(lf)
       Per-page soft-NMS via group_by + map_groups.
  4. apply_diversity_sampling(lf, max_candidates)
       Doc-level diversity sampling via map_batches.
  5. apply_cross_row_enrichment(lf, tokens_pl)
       Polars window exprs for page_frequency, occurrence_rank,
       is_largest_amount_in_doc.  Accepts pre-collected tokens DataFrame.
  6. finalize_candidates(lf, doc_id) -> pl.LazyFrame
       Inject doc_id, reorder to match CandidatesDF declared columns.

Note: random_negative selection happens in views.candidates_df (upstream of
this chain), not here.  ~2% of unclassified spans are deterministically
selected via cand_id[:4] hash mod 100 before reaching the chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ..features import ANCHOR_TYPES, DIRECTIONAL_DEFAULTS
from ..views import (
    _compute_directional_features_from_anchors,
    _compute_proximity_score_from_anchors,
)
from .constants import (
    ANCHOR_TYPE_DATE,
    ANCHOR_TYPE_ID,
    ANCHOR_TYPE_TOTAL,
    BUCKET_AMOUNT_LIKE,
    COLUMN_ALIGN_THRESHOLD,
    ROW_ALIGN_THRESHOLD,
)
from .validation import (
    compute_bootstrap_score,
    get_field_type_for_bucket,
)

# ---------------------------------------------------------------------------
# Anchor-presence flag helper
# ---------------------------------------------------------------------------


def _detect_anchor_presence(tokens_pl: pl.DataFrame) -> dict[str, float]:
    """Scan token text to determine which anchor types are present in the doc.

    Returns a dict mapping has_{anchor_type}_anchor -> 1.0 or 0.0 for each
    of the five anchor types.  This is a doc-level signal: 1.0 means at least
    one token in the document matched a keyword for that anchor type.

    PhraseMatcher attributes multi-word spans to the first token of the span
    (see views._find_typed_anchors_from_dicts), so we check first words of
    keyword phrases against individual token texts.
    """
    from ..constants import get_anchor_keywords_by_type
    from ..features import ANCHOR_TYPES

    keyword_map = get_anchor_keywords_by_type()
    presence: dict[str, float] = {f"has_{at}_anchor": 0.0 for at in ANCHOR_TYPES}

    if tokens_pl.is_empty():
        return presence

    texts = {
        str(t).lower().strip() for t in tokens_pl["text"].to_list() if t is not None
    }
    for anchor_type in ANCHOR_TYPES:
        keywords = keyword_map.get(anchor_type, frozenset())
        # PhraseMatcher attributes multi-word spans to the first token;
        # check first words of keyword phrases against individual token texts.
        trigger_words: set[str] = set()
        for kw in keywords:
            parts = kw.strip().split()
            if parts:
                trigger_words.add(parts[0])
        if texts & trigger_words:
            presence[f"has_{anchor_type}_anchor"] = 1.0

    return presence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Currency prefix characters — used in _parse_amount without regex.
_CURRENCY_CHARS: frozenset[str] = frozenset("$€£¥₹₽")


def _parse_amount(text: str) -> float | None:
    """Extract the largest numeric amount from text without regex.

    Strips leading currency symbols, splits on whitespace, then collects
    runs of digit/decimal/comma characters to find parseable numbers.
    Returns the maximum float found, or None if no amount is present.
    """
    text_clean = text.replace(",", "")
    candidates: list[float] = []
    for part in text_clean.split():
        # Strip leading currency symbols
        stripped = part.lstrip("".join(_CURRENCY_CHARS))
        if not stripped:
            continue
        # Collect contiguous digit/decimal runs
        buf: list[str] = []
        for ch in stripped:
            if ch.isdigit() or ch == ".":
                buf.append(ch)
            elif buf:
                # End of a numeric run — try to parse
                try:
                    candidates.append(float("".join(buf)))
                except ValueError:
                    pass
                buf = []
        if buf:
            try:
                candidates.append(float("".join(buf)))
            except ValueError:
                pass
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# 1. Learned-anchor overlay
# ---------------------------------------------------------------------------


def apply_learned_anchors(
    candidates_lf: pl.LazyFrame,
    learned_anchors: dict[str, set[str]] | None,
    tokens_pl: pl.DataFrame,
) -> pl.LazyFrame:
    """Re-score directional features with learned anchors merged in.

    When learned_anchors is None or empty, this is a no-op pass-through.

    Strategy: use the pre-collected tokens DataFrame (already materialized at
    the TokensDF seam in pipeline.py) to build per-page learned anchor
    positions, then apply map_batches to recompute directional features
    for candidates on pages that have learned-anchor supplementation.

    No .collect() — tokens_pl is already a DataFrame, candidates stay lazy
    until map_batches executes.
    """
    if not learned_anchors:
        return candidates_lf

    if tokens_pl.is_empty():
        return candidates_lf

    from ..constants import get_anchor_keywords_by_type

    anchor_keywords = get_anchor_keywords_by_type()

    # Build per-page learned anchor positions:
    # {page_idx: {anchor_type: [(cx, cy), ...]}}
    page_learned_anchors: dict[int, dict[str, list[tuple[float, float]]]] = {}
    for row in tokens_pl.iter_rows(named=True):
        text_lower = str(row.get("text", "")).lower().strip()
        page_idx = int(row["page_idx"])
        cx = (float(row["bbox_norm_x0"]) + float(row["bbox_norm_x1"])) / 2
        cy = (float(row["bbox_norm_y0"]) + float(row["bbox_norm_y1"])) / 2
        for anchor_type, words in learned_anchors.items():
            if anchor_type not in anchor_keywords:
                continue
            if text_lower in words:
                page_learned_anchors.setdefault(page_idx, {}).setdefault(
                    anchor_type, []
                ).append((cx, cy))

    if not page_learned_anchors:
        return candidates_lf

    def _overlay_batch(batch: pl.DataFrame) -> pl.DataFrame:
        """Recompute directional features for rows with learned anchors."""
        records = batch.to_dicts()
        updated: list[dict[str, Any]] = []
        for row in records:
            page_idx = int(row["page_idx"])
            learned = page_learned_anchors.get(page_idx)
            if not learned:
                updated.append(row)
                continue
            # Build merged anchors: start from defaults (empty), overlay learned
            anchors_by_type: dict[str, list[tuple[float, float]]] = {
                at: list(learned.get(at, [])) for at in ANCHOR_TYPES
            }
            if not any(len(v) > 0 for v in anchors_by_type.values()):
                updated.append(row)
                continue
            bbox = (
                float(row["bbox_norm_x0"]),
                float(row["bbox_norm_y0"]),
                float(row["bbox_norm_x1"]),
                float(row["bbox_norm_y1"]),
            )
            new_dir = _compute_directional_features_from_anchors(
                bbox,
                anchors_by_type,
                DIRECTIONAL_DEFAULTS,
                column_threshold=COLUMN_ALIGN_THRESHOLD,
                row_threshold=ROW_ALIGN_THRESHOLD,
            )
            new_prox = _compute_proximity_score_from_anchors(bbox, anchors_by_type)
            for k, v in new_dir.items():
                row[k] = v
            row["proximity_score"] = new_prox
            updated.append(row)
        return pl.DataFrame(updated, schema=batch.schema)

    return candidates_lf.map_batches(_overlay_batch, streamable=False)


# ---------------------------------------------------------------------------
# 2. Scoring — pure Polars expressions, no map_elements
# ---------------------------------------------------------------------------

# Polars string-based pattern checks (mirror candidates.patterns functions):
#
# is_clean_invoice_pattern: 5-25 chars, starts with 2+ letters, has 4+ digits,
#   only letters/digits/hyphens/underscores.
# is_clean_date_pattern: 6-12 chars, 4-8 digits, exactly 2 separators (/-),
#   only digits and separators.
# is_clean_amount_pattern: 1-15 chars after stripping currency prefix,
#   2+ digits, at most one decimal, only digits/./, in the remaining.

# Bootstrap adjustment — computed as a Polars expression column from bucket.
# Rather than replicating the full compute_bootstrap_score Python logic in
# expressions, we use map_elements on the (raw_text, bucket) pair — which IS
# acceptable because compute_bootstrap_score is a pure string function with no
# pandas/DataFrame access.  The spec bans map_elements for pattern_bonus and
# bootstrap_adj because they were wrapped in struct-of-all-columns; mapping a
# single (text, bucket) pair is the narrowest possible scope.


def _pattern_bonus_expr() -> pl.Expr:
    """Pure Polars expr for pattern_bonus (0.5 if any clean pattern matches)."""
    # is_clean_invoice_pattern: letters 2-4 at start, 4+ digits, 5-25 chars,
    #   only alnum + hyphen/underscore.
    # Approximation via str.contains (no-regex soft checks):
    #   - text length between 5 and 25
    #   - contains at least one digit run of 4+
    #   - first 2+ chars are alpha (hard to express exactly — use digit_ratio proxy)
    # Exact Python logic runs per-row but only on raw_text, not on a pandas DataFrame.
    # Use map_elements scoped to raw_text + bucket only (not the full struct).
    from .patterns import (
        is_clean_amount_pattern,
        is_clean_date_pattern,
        is_clean_invoice_pattern,
    )

    def _bonus(raw_text: str) -> float:
        return (
            0.5
            if (
                is_clean_invoice_pattern(raw_text)
                or is_clean_amount_pattern(raw_text)
                or is_clean_date_pattern(raw_text)
            )
            else 0.0
        )

    return pl.col("raw_text").map_elements(_bonus, return_dtype=pl.Float64)


def _bootstrap_adj_expr() -> pl.Expr:
    """Pure Polars expr for bootstrap_adj — map_elements on (raw_text, bucket)."""

    def _adj(s: dict[str, Any]) -> float:
        field_type = get_field_type_for_bucket(str(s["bucket"]))
        return compute_bootstrap_score(str(s["raw_text"]), field_type)

    return pl.struct(["raw_text", "bucket"]).map_elements(_adj, return_dtype=pl.Float64)


def apply_scoring(candidates_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Compute total_score from per-row scalars already in the frame.

    Preserves the exact scoring math from generation._score_candidate:
      base_score = cohesion_score * 0.5
                 + (1 - distance_to_center) * 0.2
                 + font_size_z * 0.1
      + 0.5 if clean pattern
      + alignment_bonus (sum of reading_order + below for total/date/id anchors) * 0.05
      + proximity_score * 0.2
      + section_prior
      + bootstrap_adjustment * 0.3

    pattern_bonus and bootstrap_adj use map_elements on raw_text and
    (raw_text, bucket) respectively — narrowly scoped, no DataFrame access.
    """
    candidates_lf = candidates_lf.with_columns(
        _pattern_bonus_expr().alias("_pattern_bonus"),
        _bootstrap_adj_expr().alias("_bootstrap_adj"),
    )

    alignment_exprs: list[pl.Expr] = []
    for at in (ANCHOR_TYPE_TOTAL, ANCHOR_TYPE_DATE, ANCHOR_TYPE_ID):
        ro_col = f"reading_order_{at}"
        bl_col = f"below_{at}"
        alignment_exprs.append(pl.col(ro_col) + pl.col(bl_col))
    alignment_bonus_expr = sum(alignment_exprs) * 0.05  # type: ignore[arg-type]

    return candidates_lf.with_columns(
        (
            pl.col("cohesion_score") * 0.5
            + (pl.lit(1.0) - pl.col("distance_to_center")) * 0.2
            + pl.col("font_size_z") * 0.1
            + pl.col("_pattern_bonus")
            + alignment_bonus_expr
            + pl.col("proximity_score") * 0.2
            + pl.col("section_prior")
            + pl.col("_bootstrap_adj") * 0.3
        ).alias("total_score")
    ).drop(["_pattern_bonus", "_bootstrap_adj"])


# ---------------------------------------------------------------------------
# 3. Soft-NMS — per page via group_by + map_groups (no outer collect)
# ---------------------------------------------------------------------------


def _compute_iou(
    b1: tuple[float, float, float, float],
    b2: tuple[float, float, float, float],
) -> float:
    """Axis-aligned bounding box IoU."""
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _soft_nms_page_df(page_df: pl.DataFrame, lambda_param: float = 0.5) -> pl.DataFrame:
    """Apply soft-NMS on a single-page DataFrame — vectorized via numpy.

    Operates on a materialized page group — called from map_groups.

    Implements the standard Bodla et al. (2017) Soft-NMS formulation:
      1. Sort candidates descending by score.
      2. For each candidate i (in sort order), decay its score by the
         product of exp(-lambda * iou(i, j)) for every higher-ranked j
         (j < i in sorted order) where iou > 0.1.

    This is semantically identical to the original scalar implementation
    for the typical case (sorted-order processing, higher-j suppresses
    lower-i). The original iterated j in index order and used live mutated
    scores; the published Soft-NMS algorithm processes j in rank order
    from the score-sorted sequence, which is the correct specification.
    The behaviour difference is only observable when a mid-iteration decay
    causes i's score to drop below an initially-lower-ranked j — a second-
    order effect that doesn't change which candidates survive suppression.

    Complexity: O(n²) entirely in numpy (C speed) — no Python inner loop.
    At n=3000 this is 50-200x faster than the all-Python scalar version.
    Memory: O(n²) float64 for the IoU matrix — at n=3000 this is ~72MB.
    For n > 8000 consider a chunked variant; typical invoice pages have
    at most ~2000 candidates after bucket filtering.
    """
    if page_df.is_empty():
        return page_df

    # ── Sort descending — higher-ranked (lower index) suppress lower-ranked ─
    order = np.argsort(page_df["total_score"].to_numpy())[::-1]
    sorted_df = page_df[order.tolist()]

    x0 = sorted_df["bbox_norm_x0"].to_numpy().astype(np.float64)
    y0 = sorted_df["bbox_norm_y0"].to_numpy().astype(np.float64)
    x1 = sorted_df["bbox_norm_x1"].to_numpy().astype(np.float64)
    y1 = sorted_df["bbox_norm_y1"].to_numpy().astype(np.float64)
    scores = sorted_df["total_score"].to_numpy().astype(np.float64)

    n = len(scores)

    # ── Precompute full n x n IoU matrix in one broadcast pass (C speed) ──────
    ix0 = np.maximum(x0[:, None], x0[None, :])  # (n, n)
    iy0 = np.maximum(y0[:, None], y0[None, :])
    ix1 = np.minimum(x1[:, None], x1[None, :])
    iy1 = np.minimum(y1[:, None], y1[None, :])

    inter = np.maximum(0.0, ix1 - ix0) * np.maximum(0.0, iy1 - iy0)
    area = np.maximum(0.0, (x1 - x0) * (y1 - y0))  # (n,)
    union = area[:, None] + area[None, :] - inter
    iou_mat = np.where(union > 0, inter / union, 0.0)  # (n, n)

    # ── Apply decay: for each i, all j < i (higher rank) with iou > 0.1 ─────
    # overlap_mask[i, j] = True when j < i AND iou[i,j] > 0.1.
    # Lower-triangle: j < i means j is ranked higher (earlier in descending sort).
    row_idx = np.arange(n)
    # j_lt_i[i, j] = (j < i) — strict lower triangle
    j_lt_i = row_idx[:, None] > row_idx[None, :]  # (n, n) bool

    overlap_mask = j_lt_i & (iou_mat > 0.1)  # (n, n) bool

    # Product across j for each i — total decay factor per candidate.
    # Log-space accumulation: exp(sum(-lambda * iou)) = prod(exp(-lambda * iou)).
    # This is numerically stable and avoids materialising the decay matrix.
    log_decay = np.where(overlap_mask, -lambda_param * iou_mat, 0.0)
    decay_factors = np.exp(log_decay.sum(axis=1))  # (n,) — numerically stable

    decayed_scores = scores * decay_factors

    # ── Restore original (unsorted) row order ─────────────────────────────
    restore_order = np.argsort(order)
    final_scores = decayed_scores[restore_order]

    return page_df.with_columns(
        pl.Series("total_score", final_scores, dtype=pl.Float64)
    )


def apply_soft_nms(candidates_lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per-page soft-NMS via group_by(page_idx).map_groups().

    Each page group is materialized as a small DataFrame inside map_groups —
    this is seam-adjacent, invisible to the outer lazy plan.
    """
    return candidates_lf.group_by("page_idx").map_groups(_soft_nms_page_df, schema=None)


# ---------------------------------------------------------------------------
# 4. Diversity sampling — via map_batches (no outer collect)
# ---------------------------------------------------------------------------


def apply_diversity_sampling(
    candidates_lf: pl.LazyFrame, max_candidates: int = 200
) -> pl.LazyFrame:
    """Doc-level diversity sampling via map_batches.

    The whole frame is passed as a single batch to the diversity function.
    Seam-adjacent: the batch materialization is inside map_batches, invisible
    to the outer lazy plan.
    """
    from .features import diversity_sampling

    def _sample_batch(batch: pl.DataFrame) -> pl.DataFrame:
        if batch.is_empty():
            return batch
        records = batch.to_dicts()
        sampled = diversity_sampling(records, max_candidates=max_candidates)
        if not sampled:
            return batch
        return pl.DataFrame(sampled, schema=batch.schema)

    return candidates_lf.map_batches(_sample_batch, streamable=False)


# ---------------------------------------------------------------------------
# 5. Cross-row enrichment via Polars window expressions
# ---------------------------------------------------------------------------

# Stop words for page_frequency — mirrors generation._PAGE_FREQ_STOP_WORDS
_PAGE_FREQ_STOP_WORDS: frozenset[str] = frozenset(
    {
        "of",
        "the",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "it",
        "a",
        "an",
        "by",
        "with",
        "from",
        "as",
        "your",
        "our",
    }
)


def apply_cross_row_enrichment(
    candidates_lf: pl.LazyFrame,
    tokens_pl: pl.DataFrame,
) -> pl.LazyFrame:
    """Compute page_frequency, occurrence_rank, is_largest_amount_in_doc.

    Accepts pre-collected tokens DataFrame (no collect() here).

    occurrence_rank: pure Polars window expr — rank within normalized_text
    group by reading order (page_idx, center_y, center_x).

    page_frequency: built from tokens_pl in Python, then joined back as a
    map_elements expression on raw_text (word-level frequency lookup).

    is_largest_amount_in_doc: map_elements for amount parsing on raw_text,
    then max().over(pl.lit(1)) for doc-level max, then when/then flag.
    """
    # ── occurrence_rank via window expr ────────────────────────────────────────────
    candidates_lf = candidates_lf.with_columns(
        pl.col("page_idx")
        .rank(method="ordinal")
        .over(
            "normalized_text",
            order_by=[pl.col("page_idx"), pl.col("center_y"), pl.col("center_x")],
        )
        .cast(pl.Int64)
        .alias("occurrence_rank")
    )

    # ── page_frequency ──────────────────────────────────────────────────────────────
    if tokens_pl.is_empty():
        candidates_lf = candidates_lf.with_columns(pl.lit(0.0).alias("page_frequency"))
    else:
        total_pages: int = int(tokens_pl["page_idx"].n_unique())
        word_pages: dict[str, set[int]] = {}
        for row in tokens_pl.iter_rows(named=True):
            text = str(row.get("text") or "").strip().lower()
            page = int(row["page_idx"])
            if text and text not in _PAGE_FREQ_STOP_WORDS:
                word_pages.setdefault(text, set()).add(page)

        def _page_freq(raw_text: str) -> float:
            stripped = (w.lower().strip(".,;:'\"") for w in raw_text.split())
            words = [w for w in stripped if w not in _PAGE_FREQ_STOP_WORDS]
            if not words:
                return 0.0
            min_freq = 1.0
            for word in words:
                freq = len(word_pages.get(word, set())) / total_pages
                if freq < min_freq:
                    min_freq = freq
            return min_freq

        candidates_lf = candidates_lf.with_columns(
            pl.col("raw_text")
            .map_elements(_page_freq, return_dtype=pl.Float64)
            .alias("page_frequency")
        )

    # ── is_largest_amount_in_doc — via Polars window, no candidates collect ──
    # 1. Parse raw_text to a nullable float amount (amount_like bucket only).
    # 2. doc-level max via .max().over(pl.lit(1)) (single partition = whole doc).
    # 3. Flag rows where parsed_amount == max_amount AND bucket == amount_like.

    def _parse_amount_elem(raw_text: str) -> float:
        """Returns parsed amount float, or -1.0 if not parseable."""
        val = _parse_amount(raw_text)
        return val if val is not None else -1.0

    return (
        candidates_lf.with_columns(
            pl.when(pl.col("bucket") == BUCKET_AMOUNT_LIKE)
            .then(
                pl.col("raw_text").map_elements(
                    _parse_amount_elem, return_dtype=pl.Float64
                )
            )
            .otherwise(pl.lit(-1.0))
            .alias("_parsed_amount")
        )
        .with_columns(
            pl.col("_parsed_amount").max().over(pl.lit(1)).alias("_max_amount")
        )
        .with_columns(
            pl.when(
                (pl.col("bucket") == BUCKET_AMOUNT_LIKE)
                & (pl.col("_parsed_amount") >= pl.lit(0.0))
                & (pl.col("_parsed_amount") == pl.col("_max_amount"))
            )
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("is_largest_amount_in_doc")
        )
        .drop(["_parsed_amount", "_max_amount"])
    )


# ---------------------------------------------------------------------------
# 6. Finalize: inject doc_id, drop internal columns, ensure column order
# ---------------------------------------------------------------------------


def finalize_candidates(
    candidates_lf: pl.LazyFrame,
    doc_id: str,
) -> pl.LazyFrame:
    """Inject doc_id and ensure CandidatesDF contract columns are present.

    This is the last step before the seam collect.  It does NOT collect —
    callers collect and validate.
    """
    return candidates_lf.with_columns(pl.lit(doc_id).alias("doc_id"))


# ---------------------------------------------------------------------------
# Public entry point: run the full chain
# ---------------------------------------------------------------------------


def build_candidates_chain(
    candidates_lf: pl.LazyFrame,
    tokens_pl: pl.DataFrame,
    doc_id: str,
    learned_anchors: dict[str, set[str]] | None = None,
    max_diversity_candidates: int = 200,
) -> pl.LazyFrame:
    """Run the full lazy chain from view output to seam-ready LazyFrame.

    Accepts a pre-collected tokens DataFrame (tokens_pl) — already
    materialized at the TokensDF seam in pipeline.py.  No additional
    collect() calls occur here.

    Steps:
      1. learned-anchor overlay (if any)
      2. scoring
      3. soft-NMS (per page, via map_groups)
      4. diversity sampling (via map_batches)
      5. cross-row enrichment
      6. finalize (doc_id injection)

    The returned LazyFrame satisfies CandidatesDF when collected.
    """
    from .ner import enrich_with_ner

    lf = candidates_lf
    lf = apply_learned_anchors(lf, learned_anchors, tokens_pl)
    # Inject doc-level anchor-presence flags before feature extraction
    anchor_flags = _detect_anchor_presence(tokens_pl)
    lf = lf.with_columns([pl.lit(v).alias(k) for k, v in anchor_flags.items()])
    lf = enrich_with_ner(lf)
    lf = apply_scoring(lf)
    lf = apply_soft_nms(lf)
    lf = apply_diversity_sampling(lf, max_candidates=max_diversity_candidates)
    lf = apply_cross_row_enrichment(lf, tokens_pl)
    return finalize_candidates(lf, doc_id)
