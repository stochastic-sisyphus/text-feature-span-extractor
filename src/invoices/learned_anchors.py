"""Compute learned anchor keywords from labeled data.

Pure computation — no I/O.  Analyzes which tokens tend to appear near
correctly-extracted values, grouped by anchor type, and returns novel
anchor words that aren't already in the static keyword sets.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

import polars as pl

from invoices import schema as schema_mod
from invoices.constants import get_anchor_keywords_by_type

if TYPE_CHECKING:
    from invoices.accumulators import AnchorCounters

# Character set for numeric token detection — replaces regex for zero-import approach.
# Matches: digits, decimal/thousand separators, sign chars, %, currency symbols, slash.
_NUMERIC_CHARS: frozenset[str] = frozenset("0123456789.,\\-+%$€£¥/")

# Tokens to always exclude — pure noise in anchor context.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "of",
        "to",
        "for",
        "and",
        "or",
        "in",
        "on",
        "at",
        "by",
    }
)


def _static_anchor_words() -> set[str]:
    """Collect all words from every static anchor keyword set."""
    words: set[str] = set()
    for kw_set in get_anchor_keywords_by_type().values():
        for phrase in kw_set:
            for token in phrase.lower().split():
                words.add(token)
    return words


def compute_learned_anchors(
    aligned: pl.DataFrame,
    tokens_by_doc: dict[str, pl.DataFrame],
    candidates_by_doc: dict[str, pl.DataFrame],
    min_field_labels: int = 3,
    spatial_window: float = 0.15,
    min_support: int = 2,
    min_lift: float = 2.0,
) -> dict[str, set[str]]:
    """Discover anchor words from labeled data via spatial co-occurrence.

    For each anchor type that has enough correctly-aligned labels, counts
    which tokens appear nearby (within *spatial_window* Euclidean distance
    in normalised page coordinates) and keeps those whose lift over the
    background frequency exceeds *min_lift* and whose count meets
    *min_support*.

    Args:
        aligned: Merged corrections + approvals DataFrame with columns
            ``sha256``, ``field``, ``candidate_idx``, ``is_aligned``.
        tokens_by_doc: ``sha256`` → tokens DataFrame with bbox columns
            and ``text``.
        candidates_by_doc: ``sha256`` → candidates DataFrame with
            ``bbox_norm_x0/y0/x1/y1``.
        min_field_labels: Minimum correct labels per field to consider.
        spatial_window: Euclidean distance threshold (normalised coords).
        min_support: Minimum foreground count for a token to be kept.
        min_lift: Minimum lift (foreground rate / background rate).

    Returns:
        ``anchor_type → set[str]`` of novel learned anchor words.
    """
    if aligned.is_empty():
        return {}

    static_words = _static_anchor_words()

    # ── 1. Group aligned rows by anchor_type ────────────────────────
    # field → anchor_type, skipping fields with no anchor_type
    rows_by_anchor: dict[str, list[dict]] = {}  # type: ignore[type-arg]

    correct = aligned.filter(pl.col("is_aligned"))
    if correct.is_empty():
        return {}

    for field_name, group in correct.group_by("field", maintain_order=True):
        fdef = schema_mod.load_field_defs().get(str(field_name))
        atype: str | None = fdef.anchor_family if fdef else None
        if atype is None:
            continue
        if len(group) < min_field_labels:
            continue
        rows_by_anchor.setdefault(atype, []).extend(group.to_dicts())

    if not rows_by_anchor:
        return {}

    # ── 2. Background token frequencies ─────────────────────────────
    bg_counter: Counter[str] = Counter()
    bg_total = 0
    for tokens_df in tokens_by_doc.values():
        if tokens_df.is_empty():
            continue
        for text in tokens_df["text"]:
            word = str(text).lower().strip()
            if _is_noise(word, static_words):
                continue
            bg_counter[word] += 1
            bg_total += 1

    if bg_total == 0:
        return {}

    # ── 3. Foreground: collect nearby tokens per anchor type ────────
    result: dict[str, set[str]] = {}

    for atype, rows in rows_by_anchor.items():
        fg_counter: Counter[str] = Counter()
        fg_total = 0

        for row in rows:
            sha256 = row["sha256"]
            cand_idx = row["candidate_idx"]
            if cand_idx is None or (
                isinstance(cand_idx, float) and math.isnan(cand_idx)
            ):
                continue

            cands_df_opt = candidates_by_doc.get(sha256)
            tokens_df_opt = tokens_by_doc.get(sha256)
            if cands_df_opt is None or tokens_df_opt is None:
                continue
            if cands_df_opt.is_empty() or tokens_df_opt.is_empty():
                continue

            cand_idx_int = int(cand_idx)
            if cand_idx_int < 0 or cand_idx_int >= len(cands_df_opt):
                continue

            cand = cands_df_opt.row(cand_idx_int, named=True)
            cx = (cand["bbox_norm_x0"] + cand["bbox_norm_x1"]) / 2
            cy = (cand["bbox_norm_y0"] + cand["bbox_norm_y1"]) / 2

            for tok in tokens_df_opt.iter_rows(named=True):
                tx = (tok["bbox_norm_x0"] + tok["bbox_norm_x1"]) / 2
                ty = (tok["bbox_norm_y0"] + tok["bbox_norm_y1"]) / 2
                dist = math.hypot(cx - tx, cy - ty)
                if dist > spatial_window:
                    continue
                word = str(tok["text"]).lower().strip()
                if _is_noise(word, static_words):
                    continue
                fg_counter[word] += 1
                fg_total += 1

        if fg_total == 0:
            continue

        # ── 4. Compute lift and filter ──────────────────────────────
        anchors: set[str] = set()
        for word, count in fg_counter.items():
            if count < min_support:
                continue
            bg_count = bg_counter.get(word, 0)
            if bg_count == 0:
                # Token only appears near correct labels — infinite lift.
                anchors.add(word)
                continue
            fg_rate = count / fg_total
            bg_rate = bg_count / bg_total
            lift = fg_rate / bg_rate
            if lift >= min_lift:
                anchors.add(word)

        if anchors:
            result[atype] = anchors

    return result


def accumulate_anchor_counters(
    aligned_chunk: pl.DataFrame,
    tokens_by_doc: dict[str, pl.DataFrame],
    candidates_by_doc: dict[str, pl.DataFrame],
    spatial_window: float = 0.15,
) -> AnchorCounters:
    """Accumulate anchor foreground/background counters for a chunk.

    Returns an :class:`AnchorCounters` instance with raw counts that can
    be merged across chunks then passed through the lift filter in
    :meth:`ChunkAccumulator.finalize_anchors`.

    Foreground counters are keyed **per field** (not per anchor_type).
    ``min_field_labels`` is NOT applied here — the threshold is applied
    after merging all chunks in ``finalize_anchors``, which maps
    field → anchor_type and filters by total correct-label count.
    """
    from invoices.accumulators import AnchorCounters

    static_words = _static_anchor_words()
    result = AnchorCounters()

    # Background: count all tokens in this chunk's documents
    for tokens_df in tokens_by_doc.values():
        if tokens_df.is_empty():
            continue
        for text in tokens_df["text"]:
            word = str(text).lower().strip()
            if _is_noise(word, static_words):
                continue
            result.background[word] += 1
            result.bg_total += 1

    if aligned_chunk.is_empty():
        return result

    correct = aligned_chunk.filter(pl.col("is_aligned"))
    if correct.is_empty():
        return result

    # Group by field — record label counts and foreground per field.
    # Fields without an anchor_type are skipped (no anchors possible).
    for field_name, group in correct.group_by("field", maintain_order=True):
        fname = str(field_name)
        fdef = schema_mod.load_field_defs().get(fname)
        atype: str | None = fdef.anchor_family if fdef else None
        if atype is None:
            continue

        # Track correct-label count per field (for min_field_labels filter)
        result.field_label_counts[fname] = result.field_label_counts.get(
            fname, 0
        ) + len(group)

        rows = group.to_dicts()
        fg_counter: Counter[str] = Counter()
        fg_total = 0

        for row in rows:
            sha256 = row["sha256"]
            cand_idx = row["candidate_idx"]
            if cand_idx is None or (
                isinstance(cand_idx, float) and math.isnan(cand_idx)
            ):
                continue

            cands_df_opt = candidates_by_doc.get(sha256)
            tokens_df_opt = tokens_by_doc.get(sha256)
            if cands_df_opt is None or tokens_df_opt is None:
                continue
            if cands_df_opt.is_empty() or tokens_df_opt.is_empty():
                continue

            cand_idx_int = int(cand_idx)
            if cand_idx_int < 0 or cand_idx_int >= len(cands_df_opt):
                continue

            cand = cands_df_opt.row(cand_idx_int, named=True)
            cx = (cand["bbox_norm_x0"] + cand["bbox_norm_x1"]) / 2
            cy = (cand["bbox_norm_y0"] + cand["bbox_norm_y1"]) / 2

            for tok in tokens_df_opt.iter_rows(named=True):
                tx = (tok["bbox_norm_x0"] + tok["bbox_norm_x1"]) / 2
                ty = (tok["bbox_norm_y0"] + tok["bbox_norm_y1"]) / 2
                dist = math.hypot(cx - tx, cy - ty)
                if dist > spatial_window:
                    continue
                word = str(tok["text"]).lower().strip()
                if _is_noise(word, static_words):
                    continue
                fg_counter[word] += 1
                fg_total += 1

        if fg_counter:
            if fname in result.foreground:
                result.foreground[fname] += fg_counter
                result.fg_totals[fname] = result.fg_totals.get(fname, 0) + fg_total
            else:
                result.foreground[fname] = fg_counter
                result.fg_totals[fname] = fg_total

    return result


def _is_noise(word: str, static_words: set[str]) -> bool:
    """Return True if word should be excluded from anchor analysis."""
    if len(word) <= 1:
        return True
    # Numeric token: every character is a digit, separator, sign, currency symbol, etc.
    if word and all(c in _NUMERIC_CHARS for c in word):
        return True
    if word in _STOPWORDS:
        return True
    if word in static_words:
        return True
    return False
