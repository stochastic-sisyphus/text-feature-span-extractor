"""Tests for SpanBuilder._join_tokens_text — gap-aware token reassembly."""

from __future__ import annotations

import pytest

from invoices.candidates.spans import SpanBuilder


def _tok(text: str, x0: float, x1: float, y0: float = 0.0, y1: float = 10.0) -> dict:
    return {
        "text": text,
        "bbox_pdf_units_x0": x0,
        "bbox_pdf_units_x1": x1,
        "bbox_pdf_units_y0": y0,
        "bbox_pdf_units_y1": y1,
    }


@pytest.fixture
def builder() -> SpanBuilder:
    return SpanBuilder()


def test_shattered_run_reassembled(builder: SpanBuilder) -> None:
    """Glyphs touching (gap=0) must concatenate without spaces."""
    # "$","3","8",".","9","4" each width=2, touching end-to-end
    tokens = [
        _tok("$", 0.0, 2.0),
        _tok("3", 2.0, 4.0),
        _tok("8", 4.0, 6.0),
        _tok(".", 6.0, 8.0),
        _tok("9", 8.0, 10.0),
        _tok("4", 10.0, 12.0),
    ]
    assert builder._join_tokens_text(tokens) == "$38.94"


def test_real_space_preserved(builder: SpanBuilder) -> None:
    """Gap >= 0.30x height (well above 0.15 threshold) must produce a space."""
    # height=10, gap=3.0 => ratio 0.30 >= 0.15
    tokens = [
        _tok("OAK", 0.0, 10.0),
        _tok("PARK", 13.0, 23.0),  # gap = 3.0
    ]
    assert builder._join_tokens_text(tokens) == "OAK PARK"


def test_boundary_at_threshold_inserts_space(builder: SpanBuilder) -> None:
    """Gap exactly == 0.15 * height must insert a space (>= is inclusive)."""
    # height=10, gap=1.5 => ratio exactly 0.15
    tokens = [
        _tok("A", 0.0, 5.0),
        _tok("B", 6.5, 11.5),  # gap = 1.5
    ]
    assert builder._join_tokens_text(tokens) == "A B"


def test_missing_geometry_falls_back_to_space(builder: SpanBuilder) -> None:
    """Tokens without bbox keys must be joined with spaces (previous behaviour)."""
    tokens = [{"text": "A"}, {"text": "B"}]
    assert builder._join_tokens_text(tokens) == "A B"


def test_single_token_unchanged(builder: SpanBuilder) -> None:
    """Single token returns its text verbatim."""
    tokens = [_tok("HELLO", 0.0, 20.0)]
    assert builder._join_tokens_text(tokens) == "HELLO"
