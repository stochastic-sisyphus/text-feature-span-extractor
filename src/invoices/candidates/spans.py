"""Span building and spatial grid structures for candidate generation."""

import itertools
from collections import defaultdict
from typing import Any

import polars as pl

from .patterns import (
    compute_pattern_score_bonus,
    compute_token_count_penalty,
    normalize_text_for_dedup,
)

# Empirically (n=1541 real spaces): shattered intra-glyph gaps measure ~0.0x the
# token height (glyphs touching), real word spaces ~0.23-0.31x. 0.15 cleanly
# separates them. Height-relative, so font-size invariant (the token font_size
# field is unreliably 0.0; bbox height is the robust size proxy).
_SPACE_GAP_HEIGHT_RATIO = 0.15


class PageGrid:
    """Coarse grid for neighbor-only lookups."""

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.cells: defaultdict[tuple[int, int], list[Any]] = defaultdict(
            list
        )  # cell_coord -> list of items

    def add_item(self, x_norm: float, y_norm: float, item: Any) -> None:
        """Add an item to the grid at normalized coordinates."""
        cell_x = min(int(x_norm * self.grid_size), self.grid_size - 1)
        cell_y = min(int(y_norm * self.grid_size), self.grid_size - 1)
        self.cells[(cell_x, cell_y)].append(item)

    def get_neighbors(self, x_norm: float, y_norm: float, radius: int = 1) -> list[Any]:
        """Get all items in neighboring cells."""
        cell_x = min(int(x_norm * self.grid_size), self.grid_size - 1)
        cell_y = min(int(y_norm * self.grid_size), self.grid_size - 1)

        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cell_x + dx, cell_y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.extend(self.cells[(nx, ny)])

        return neighbors


class SpanBuilder:
    """Build multi-token spans from line-local adjacency."""

    def __init__(self, max_gap: float = 0.05, max_span_tokens: int = 8):
        self.max_gap = max_gap  # Maximum normalized x-gap between adjacent tokens
        self.max_span_tokens = max_span_tokens  # Maximum tokens per span

    def build_spans(self, page_tokens: pl.DataFrame) -> list[dict[str, Any]]:
        """Build spans from tokens on a single page.

        Accepts a materialized polars DataFrame. LazyFrame is intentionally
        rejected: span construction is a sequential per-line loop over rows,
        so lazy evaluation buys nothing and forces an extra .collect() at each
        group boundary. Callers must materialize before calling.
        """
        if page_tokens.is_empty():
            return []

        # Sort once up front; partition_by preserves order within groups.
        sorted_tokens = page_tokens.sort("bbox_norm_x0")

        spans: list[dict[str, Any]] = []

        # partition_by returns a list of DataFrames, one per unique line_id.
        # maintain_order=True ensures groups appear in the order their first
        # row appears after the sort (left-to-right reading order).
        for line_df in sorted_tokens.partition_by("line_id", maintain_order=True):
            line_rows = line_df.iter_rows(named=True)
            line_spans = self._build_line_spans(list(line_rows))
            spans.extend(line_spans)

        return spans

    def _build_line_spans(
        self, tokens_list: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build spans within a single line using adjacency and cohesion scoring.

        CRITICAL: Creates spans of ALL lengths from 1 to max_span_tokens,
        not just the maximally-extended span. This ensures single-token
        candidates (like invoice numbers) are preserved.
        """
        if not tokens_list:
            return []

        spans = []

        # Start with each token as a potential span start
        for i, start_token in enumerate(tokens_list):
            span_tokens = [start_token]

            # CRITICAL FIX: Always create single-token span FIRST
            # This ensures ID-like tokens like "US002650-41" are candidates
            span = self._create_span_from_tokens(span_tokens)
            if span:
                spans.append(span)

            # Try to extend the span with adjacent tokens up to max length
            for j in range(i + 1, min(i + self.max_span_tokens, len(tokens_list))):
                candidate_token = tokens_list[j]

                # Check if adjacent (small gap)
                gap = candidate_token["bbox_norm_x0"] - span_tokens[-1]["bbox_norm_x1"]
                if gap <= self.max_gap:
                    span_tokens.append(candidate_token)
                    # Create span at each extension point (2, 3, 4 tokens)
                    span = self._create_span_from_tokens(span_tokens)
                    if span:
                        spans.append(span)
                else:
                    break  # Gap too large, stop extending

        return spans

    def _join_tokens_text(self, tokens: list[dict[str, Any]]) -> str:
        """Reconstruct span text from tokens using inter-token gaps.

        Insert a space between consecutive tokens only when the horizontal gap
        is a real word space (>= _SPACE_GAP_HEIGHT_RATIO of the token height);
        concatenate sub-threshold gaps so glyph-shattered runs (e.g. a kerned
        "$38.94" that extract_words split into "$","3","8",".","9","4") are
        reassembled rather than rendered "$ 3 8 . 9 4". Height-relative and
        derived from the tokens' own bbox — no magic absolute, no regex. Falls
        back to a space when geometry is missing (height <= 0), preserving the
        previous behaviour for those tokens.
        """
        if not tokens:
            return ""
        parts = [str(tokens[0]["text"])]
        for prev, cur in itertools.pairwise(tokens):
            gap = cur.get("bbox_pdf_units_x0", 0.0) - prev.get("bbox_pdf_units_x1", 0.0)
            height = prev.get("bbox_pdf_units_y1", 0.0) - prev.get(
                "bbox_pdf_units_y0", 0.0
            )
            sep = (
                " " if (height <= 0 or gap >= _SPACE_GAP_HEIGHT_RATIO * height) else ""
            )
            parts.append(sep + str(cur["text"]))
        return "".join(parts)

    def _create_span_from_tokens(
        self, tokens: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Create a span from a list of token row dicts."""
        if not tokens:
            return None

        # Combine text
        raw_text = self._join_tokens_text(tokens)
        normalized_text = normalize_text_for_dedup(raw_text)

        # Skip very short spans
        if len(raw_text.strip()) < 2:
            return None

        # Compute bounding box
        min_x = min(token["bbox_norm_x0"] for token in tokens)
        min_y = min(token["bbox_norm_y0"] for token in tokens)
        max_x = max(token["bbox_norm_x1"] for token in tokens)
        max_y = max(token["bbox_norm_y1"] for token in tokens)

        # Cohesion score - INVERTED to prefer SHORT spans over long ones
        # Old (bad): token_count / span_width - rewards MORE tokens
        # New (good): compactness per token, with bonus for fewer tokens
        span_width = max_x - min_x
        token_count = len(tokens)

        # Base cohesion: how compact is each token (smaller width per token = more compact)
        # Inverse of span_width gives higher score for narrower spans
        compactness = 1.0 / max(span_width, 0.01)

        # Apply token count adjustment: prefer 1-3 tokens, penalize 4+
        token_count_adjustment = compute_token_count_penalty(token_count)

        # Pattern quality bonus/penalty for the text
        pattern_bonus = compute_pattern_score_bonus(raw_text, token_count)

        # Combined cohesion score
        cohesion_score = compactness * 0.01 + token_count_adjustment + pattern_bonus

        # Use first token's metadata as representative
        first_token = tokens[0]

        return {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "token_count": token_count,
            "cohesion_score": cohesion_score,
            "bbox_norm": (min_x, min_y, max_x, max_y),
            "token_ids": [str(token["token_id"]) for token in tokens],
            "token_indices": [int(token["token_idx"]) for token in tokens],
            "page_idx": int(first_token["page_idx"]),
            "line_id": int(first_token["line_id"]),
            "font_size": float(first_token["font_size"]),
            "is_bold": bool(first_token["is_bold"]),
            "is_italic": bool(first_token["is_italic"]),
            "font_hash": str(first_token["font_hash"]),
            "page_width": float(first_token["page_width"]),
            "page_height": float(first_token["page_height"]),
        }
