"""Tokenization module for deterministic text extraction from PDFs using pdfplumber.

Phase 0 wave B: the single ``pdfplumber.open()`` context in this module is
the *only* parse window in the system.  Everything the rest of the system
will ever need from the raw PDF is captured here, frozen into a
:class:`invoices.doc.Doc`, and handed off.  No other module opens pdfplumber.
"""

import hashlib
import io
import os
from pathlib import Path
from typing import Any

import pdfplumber

from . import views
from .doc import (
    CharEntry,
    Doc,
    DocPage,
    EdgeSegment,
    RectRegion,
    TableRegion,
    VectorSegment,
)
from .logging import get_logger

logger = get_logger(__name__)

# Maximum number of pages parsed per document.  pdfplumber.open() + len(pdf.pages)
# are lazy (page-tree only — no content extraction), so this check is cheap.
# A document exceeding this limit returns an empty Doc (pages=()) which flows
# through the pipeline as needs_review.  Override via INVOICEX_MAX_PARSE_PAGES.
_MAX_PARSE_PAGES: int = int(os.getenv("INVOICEX_MAX_PARSE_PAGES", "60"))


def _float(val: Any, default: float = 0.0) -> float:
    """Coerce a pdfplumber attr to float, tolerating None/missing."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _json_color(val: Any) -> Any:
    """Normalize a pdfplumber color value to a JSON-native shape.

    pdfplumber emits colors as tuples (``(0,)``, ``(0.5, 0.5, 0.5)``),
    floats, or None depending on colorspace.  Tuples JSON-serialize as
    lists, so to keep the Doc JSONB-round-trippable we canonicalise to
    lists at construction time.  Other scalar types pass through.
    """
    if val is None:
        return None
    if isinstance(val, tuple):
        return [_json_color(v) for v in val]
    if isinstance(val, list):
        return [_json_color(v) for v in val]
    return val


def _build_doc_page(page: Any, page_idx: int, tokens: tuple[Any, ...] = ()) -> DocPage:
    """Freeze a pdfplumber ``Page`` into a :class:`DocPage`.

    Pulls chars, lines (vector segments), rects, edges, and tables —
    everything we keep from the raw PDF — inside the caller's pdfplumber
    context manager.  Do NOT call this outside an open pdfplumber context.
    """
    # Native pdfplumber pass-through: CharEntry/VectorSegment/RectRegion/
    # EdgeSegment all use _LenientBase (extra="allow"), so every key
    # pdfplumber emits per char/line/rect/edge (width, height, doctop, adv,
    # stroking_color, ncs/scs, object_type, page_number, mcid, matrix, ...)
    # lands in the model as an extra and surfaces downstream.  Stop pre-
    # selecting nine fields; capture everything pdfplumber natively gives.
    chars = tuple(CharEntry(**c) for c in (page.chars or []))
    vector_segments = tuple(VectorSegment(**ln) for ln in (page.lines or []))
    rects = tuple(RectRegion(**r) for r in (page.rects or []))
    edges = tuple(EdgeSegment(**e) for e in (page.edges or []))

    table_objs = page.find_tables() or []
    table_regions: list[TableRegion] = []
    for t in table_objs:
        try:
            extracted = t.extract() or []
        except Exception:
            extracted = []
        rows = tuple(
            tuple(cell if cell is None else str(cell) for cell in row)
            for row in extracted
        )
        table_regions.append(
            TableRegion(
                bbox=(
                    float(t.bbox[0]),
                    float(t.bbox[1]),
                    float(t.bbox[2]),
                    float(t.bbox[3]),
                ),
                cells=tuple(
                    (float(cell[0]), float(cell[1]), float(cell[2]), float(cell[3]))
                    for cell in (getattr(t, "cells", None) or [])
                ),
                rows=rows,
            )
        )
    tables: tuple[TableRegion, ...] = tuple(table_regions)

    return DocPage(
        page_idx=page_idx,
        width=float(page.width),
        height=float(page.height),
        chars=chars,
        vector_segments=vector_segments,
        rects=rects,
        edges=edges,
        tables=tables,
        tokens=tokens,
    )


def build_doc(pdf_source: "Path | io.BytesIO") -> Doc:
    """Open the PDF once and return a frozen :class:`Doc`.

    Sole live ingest entry point. The parse window — the single
    ``pdfplumber.open()`` in this module — is the only window in the
    system; no other module may open pdfplumber.
    """
    pages: list[DocPage] = []
    sha256: str | None = None

    with pdfplumber.open(pdf_source) as pdf:  # type: ignore[arg-type]
        # Compute content sha256 for the Doc.  When pdf_source is a path
        # we hash its bytes; when it is a BytesIO we hash the buffer.
        try:
            if isinstance(pdf_source, (str, Path)):
                sha256 = hashlib.sha256(Path(pdf_source).read_bytes()).hexdigest()
            else:
                # BytesIO: snapshot position, read, restore.
                pos = pdf_source.tell()
                pdf_source.seek(0)
                sha256 = hashlib.sha256(pdf_source.read()).hexdigest()
                pdf_source.seek(pos)
        except Exception:
            sha256 = None

        effective_doc_id = sha256 or ""

        # ── Page-count guard (pre-loop, cheap) ───────────────────────────────
        # len(pdf.pages) reads the page-tree only — no content extraction —
        # so this check costs nothing.  A pathological page count (job 159747)
        # caused extract_words() to exhaust 11 GiB; returning an empty Doc
        # lets the job complete as needs_review without OOM-killing the worker.
        if len(pdf.pages) > _MAX_PARSE_PAGES:
            logger.warning(
                "build_doc_skipped_oversize_pages",
                pages=len(pdf.pages),
                limit=_MAX_PARSE_PAGES,
                sha256=effective_doc_id[:16] if effective_doc_id else "",
            )
            return Doc(sha256=effective_doc_id, pages=())

        for page_idx, raw_page in enumerate(pdf.pages):
            # Use pdfplumber-native defaults: extract_words() with no
            # extra_attrs filter and default x/y tolerances.  Prior attempt
            # to constrain with extra_attrs=[fontname,size,non_stroking_color]
            # + x_tolerance=1 silently dropped any char missing one of those
            # attrs and shattered words on sub-point gaps — yielding empty
            # token sets across native-text PDFs.  Dedupe is opt-in: only
            # apply when chars actually overlap (handled inside pdfplumber's
            # own char extraction when needed).
            page = raw_page
            word_dicts = page.extract_words()
            tokens = tuple(
                tok
                for token_idx, word in enumerate(word_dicts)
                for tok in [
                    views._build_token_dict(
                        effective_doc_id,
                        page_idx,
                        float(page.width),
                        float(page.height),
                        token_idx,
                        word,
                        reading_order=token_idx,
                    )
                ]
                if tok is not None
            )
            pages.append(_build_doc_page(page, page_idx, tokens=tokens))

    return Doc(sha256=effective_doc_id, pages=tuple(pages))
