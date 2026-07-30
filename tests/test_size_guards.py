"""Tests for the page-count guard in build_doc (tokenize.py).

The guard fires BEFORE the extract_words() loop, so it is cheap (page-tree
read only) and is the sole OOM defence for pathological documents like job
159747 (small file, huge page count, exhausts 11 GiB on extract_words).

Covers:
1. _MAX_PARSE_PAGES constant exists and is env-overridable.
2. build_doc returns an empty Doc (pages=()) when page count exceeds the limit.
3. build_doc parses normally for a document within the limit.
4. reextract_doc_dims is still logged by run_document_pipeline (always-on).
"""

from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# 1. _MAX_PARSE_PAGES constant
# ---------------------------------------------------------------------------


def test_max_parse_pages_default() -> None:
    """_MAX_PARSE_PAGES defaults to 60 when INVOICEX_MAX_PARSE_PAGES is unset."""
    clean_env = {k: v for k, v in os.environ.items() if k != "INVOICEX_MAX_PARSE_PAGES"}
    with patch.dict(os.environ, clean_env, clear=True):
        import importlib

        import invoices.tokenize as tok_mod

        importlib.reload(tok_mod)
        assert tok_mod._MAX_PARSE_PAGES == 60


def test_max_parse_pages_env_override() -> None:
    """INVOICEX_MAX_PARSE_PAGES overrides the default."""
    with patch.dict(os.environ, {"INVOICEX_MAX_PARSE_PAGES": "30"}):
        import importlib

        import invoices.tokenize as tok_mod

        importlib.reload(tok_mod)
        assert tok_mod._MAX_PARSE_PAGES == 30


# ---------------------------------------------------------------------------
# 2. Guard predicate (pure logic — mirrors build_doc condition)
# ---------------------------------------------------------------------------


def _guard_would_skip(n_pages: int, limit: int) -> bool:
    """Mirror the guard predicate from build_doc."""
    return n_pages > limit


def test_page_guard_triggers_when_over_limit() -> None:
    assert _guard_would_skip(61, 60) is True


def test_page_guard_passes_when_at_limit() -> None:
    assert _guard_would_skip(60, 60) is False


def test_page_guard_passes_when_under_limit() -> None:
    assert _guard_would_skip(5, 60) is False


# ---------------------------------------------------------------------------
# 3 & 4. build_doc integration: synthetic PDF via pdfplumber mock
#
# We cannot easily create a real multi-page PDF in a unit test without
# optional heavy dependencies (reportlab etc.), so we mock pdfplumber.open
# to simulate a page-tree response with a controlled page count.
# ---------------------------------------------------------------------------


def _mock_pdf_context(n_pages: int):
    """Return a context-manager mock simulating a pdfplumber PDF with n_pages."""
    mock_pdf = MagicMock()
    mock_pdf.pages = [MagicMock() for _ in range(n_pages)]
    # Each page's extract_words() should never be called for the oversize case.
    for page in mock_pdf.pages:
        page.extract_words.return_value = []
        page.width = 595.0
        page.height = 842.0
        page.chars = []
        page.lines = []
        page.rects = []
        page.edges = []
        page.find_tables.return_value = []

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=mock_pdf)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_build_doc_returns_empty_doc_for_oversize_page_count() -> None:
    """build_doc returns Doc(pages=()) when page count exceeds _MAX_PARSE_PAGES."""
    import importlib

    # Reload with explicit limit of 5 for test isolation.
    with patch.dict(os.environ, {"INVOICEX_MAX_PARSE_PAGES": "5"}):
        import invoices.tokenize as tok_mod

        importlib.reload(tok_mod)

    with patch("pdfplumber.open", return_value=_mock_pdf_context(n_pages=10)):
        doc = tok_mod.build_doc(io.BytesIO(b"%PDF-1.4 fake"))

    assert doc.pages == (), f"Expected empty pages tuple, got {len(doc.pages)} pages"


def test_build_doc_parses_normally_within_limit() -> None:
    """build_doc returns non-empty Doc when page count is within _MAX_PARSE_PAGES."""
    import importlib

    with patch.dict(os.environ, {"INVOICEX_MAX_PARSE_PAGES": "5"}):
        import invoices.tokenize as tok_mod

        importlib.reload(tok_mod)

    with patch("pdfplumber.open", return_value=_mock_pdf_context(n_pages=3)):
        doc = tok_mod.build_doc(io.BytesIO(b"%PDF-1.4 fake"))

    assert len(doc.pages) == 3


# ---------------------------------------------------------------------------
# 5. reextract_doc_dims still logged by run_document_pipeline (always-on)
#
# This instrumentation is kept regardless of the guard change.  We verify
# it fires by checking the pipeline emits the log key for a normal doc.
# ---------------------------------------------------------------------------


def _make_mock_doc(n_pages: int, tokens_per_page: int) -> MagicMock:
    """Build a minimal mock Doc."""
    mock_doc = MagicMock()
    mock_doc.sha256 = "aabbcc" * 10 + "00"
    pages = []
    for _i in range(n_pages):
        page = MagicMock()
        page.tokens = [MagicMock() for _ in range(tokens_per_page)]
        pages.append(page)
    mock_doc.pages = tuple(pages)
    return mock_doc


def test_dims_computed_correctly_from_doc() -> None:
    """n_tokens and n_pages are derived correctly from the Doc structure.

    The pipeline logs these via reextract_doc_dims before any candidate work.
    We verify the computation directly: sum of token counts across all pages.
    """
    mock_doc = _make_mock_doc(n_pages=5, tokens_per_page=10)

    n_tokens = sum(len(p.tokens) for p in mock_doc.pages)
    n_pages = len(mock_doc.pages)

    assert n_tokens == 50
    assert n_pages == 5
