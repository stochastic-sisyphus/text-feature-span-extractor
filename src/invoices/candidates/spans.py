"""Span building and spatial grid structures for candidate generation."""

from collections import defaultdict
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from .patterns import (
    compute_pattern_score_bonus,
    compute_token_count_penalty,
    normalize_text_for_dedup,
)


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

    def build_spans(self, page_tokens: pd.DataFrame) -> list[dict[str, Any]]:
        """Build spans from tokens on a single page."""
        if page_tokens.empty:
            return []

        spans = []

        # Group tokens by line_id
        for line_id in page_tokens["line_id"].unique():
            line_tokens = page_tokens[page_tokens["line_id"] == line_id].copy()
            line_tokens = line_tokens.sort_values("bbox_norm_x0")

            # Build spans within this line
            line_spans = self._build_line_spans(line_tokens)
            spans.extend(line_spans)

        return spans

    def _build_line_spans(self, line_tokens: pd.DataFrame) -> list[dict[str, Any]]:
        """Build spans within a single line using adjacency and cohesion scoring.

        CRITICAL: Creates spans of ALL lengths from 1 to max_span_tokens,
        not just the maximally-extended span. This ensures single-token
        candidates (like invoice numbers) are preserved.
        """
        if line_tokens.empty:
            return []

        spans = []
        tokens_list = list(line_tokens.iterrows())

        # Start with each token as a potential span start
        for i, (_, start_token) in enumerate(tokens_list):
            span_tokens = [start_token]

            # CRITICAL FIX: Always create single-token span FIRST
            # This ensures ID-like tokens like "US002650-41" are candidates
            span = self._create_span_from_tokens(span_tokens)
            if span:
                spans.append(span)

            # Try to extend the span with adjacent tokens up to max length
            for j in range(i + 1, min(i + self.max_span_tokens, len(tokens_list))):
                _, candidate_token = tokens_list[j]

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

    def _create_span_from_tokens(
        self, tokens: list[pd.Series]
    ) -> dict[str, Any] | None:
        """Create a span from a list of tokens."""
        if not tokens:
            return None

        # Combine text
        raw_text = " ".join(token["text"] for token in tokens)
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
