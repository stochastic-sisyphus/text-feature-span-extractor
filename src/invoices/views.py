"""Polars-backed views over a :class:`invoices.doc.Doc`.

Phase 0 wave B invariants:

* **Polars owns the view chain.**  Every view function returns a
  :class:`polars.LazyFrame`.  No eager ``.collect()`` inside this module —
  the cost-matrix seam (later phase) is the single collection point.
* **Pure functions.**  Views take a frozen ``Doc`` and return a lazy frame.
  No I/O, no mutation, no pandas.
* **Views own identity.**  ``compute_stable_token_id`` lives here: the
  view that produces token rows is the owner of token identity.
  :mod:`invoices.utils` re-exports it as a deprecation shim so existing
  call sites keep working during Phase 0 overlap.

``token_id`` is ephemeral: deterministic within a run, recomputed on every
invocation.  The recipe lives here and is the sole tuning surface.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Any

import polars as pl
import spacy
from rapidfuzz import fuzz
from spacy.matcher import PhraseMatcher

from .doc import Doc
from .features import ANCHOR_TYPES
from .logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# spaCy pipeline — module-level singleton, initialised once.
#
# We use spacy.blank("en") to avoid downloading the full en_core_web_sm model
# on every import during tests.  The _nlp_pipeline() factory switches to
# en_core_web_sm when available (needed for the full production run so that
# token.like_num / token.is_currency attributes are populated).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _nlp_pipeline() -> spacy.language.Language:
    """Return the shared spaCy Language instance (loaded once per process)."""
    try:
        # Keep ner enabled so downstream text-pattern scoring can inspect doc.ents.
        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except OSError:
        # Fallback for unit-test / CI environments where the model isn't installed.
        logger.warning("en_core_web_sm_not_found_fallback")
        return spacy.blank("en")


@lru_cache(maxsize=1)
def _anchor_phrase_matcher() -> PhraseMatcher:
    """Return the PhraseMatcher seeded from static anchor keyword sets.

    Uses LOWER attribute for case-insensitive matching.  Built once and
    cached — the anchor keyword sets are module-level constants so this is
    safe to cache across requests.
    """
    from .constants import get_anchor_keywords_by_type

    nlp = _nlp_pipeline()
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for anchor_type, keyword_set in get_anchor_keywords_by_type().items():
        patterns = [nlp.make_doc(kw) for kw in keyword_set]
        matcher.add(anchor_type, patterns)
    return matcher


# ---------------------------------------------------------------------------
# Rapidfuzz vendor matching — fuzzy anchor detection for low-confidence names.
# ---------------------------------------------------------------------------

# Minimum fuzzy ratio (0-100) to accept a token as a vendor name anchor.
_FUZZY_VENDOR_THRESHOLD: int = 75


def _fuzzy_vendor_match(text: str, vendor_anchors: set[str]) -> bool:
    """Return True if *text* fuzzy-matches any vendor anchor phrase.

    Uses token_set_ratio to handle word-order variation in company names.
    Only applied when static PhraseMatcher produces no name-type anchor.
    """
    text_lower = text.lower().strip()
    if not text_lower or not vendor_anchors:
        return False
    return any(
        fuzz.token_set_ratio(text_lower, anchor) >= _FUZZY_VENDOR_THRESHOLD
        for anchor in vendor_anchors
    )


@lru_cache(maxsize=1)
def _entity_ruler_nlp() -> spacy.language.Language:
    """Return a spaCy pipeline with an EntityRuler for bucket classification.

    Patterns map token-attribute sequences to BUCKET_* entity labels so that
    `candidates_df` can use ruler entity spans as a supplementary signal
    alongside the existing soft-classifiers in `candidates/patterns.py`.

    Field-type-to-pattern mapping:
      BUCKET_DATE_LIKE   — IS_DIGIT + date-separator or month-shape tokens
      BUCKET_AMOUNT_LIKE — LIKE_NUM token OR IS_CURRENCY + LIKE_NUM sequence
      BUCKET_ID_LIKE     — alphanumeric token with embedded digits (IS_ASCII)
      BUCKET_NAME_LIKE   — sequence of title-cased or ALL-CAPS words

    The ruler runs on per-span text only (small strings), not page text,
    so the overhead is negligible.
    """
    from spacy.pipeline import EntityRuler as _EntityRuler

    nlp = spacy.blank("en")
    ruler: _EntityRuler = nlp.add_pipe(  # type: ignore[assignment]
        "entity_ruler", config={"overwrite_ents": True}
    )

    patterns: list[dict[str, Any]] = [
        # Amount: currency symbol followed by a numeric-like token
        {
            "label": "BUCKET_AMOUNT_LIKE",
            "pattern": [{"IS_CURRENCY": True}, {"LIKE_NUM": True}],
        },
        # Amount: bare numeric-like token (catches "1,234.56")
        {"label": "BUCKET_AMOUNT_LIKE", "pattern": [{"LIKE_NUM": True}]},
        # Date: digit-separator-digit patterns (e.g. "12/31/2024", "2024-01-01")
        {
            "label": "BUCKET_DATE_LIKE",
            "pattern": [
                {"IS_DIGIT": True},
                {"ORTH": {"IN": ["/", "-"]}},
                {"IS_DIGIT": True},
            ],
        },
        # ID: single token, alphanumeric, contains digits
        {
            "label": "BUCKET_ID_LIKE",
            "pattern": [{"IS_ALPHA": False, "IS_ASCII": True, "IS_PUNCT": False}],
        },
        # Name: two+ title-case or ALL-CAPS words
        {
            "label": "BUCKET_NAME_LIKE",
            "pattern": [{"IS_TITLE": True}, {"IS_TITLE": True}],
        },
        {
            "label": "BUCKET_NAME_LIKE",
            "pattern": [{"IS_UPPER": True}, {"IS_UPPER": True}],
        },
    ]
    ruler.add_patterns(patterns)
    return nlp


def _ruler_bucket_hint(text: str) -> str | None:
    """Return an EntityRuler-derived bucket hint for *text*, or None.

    Runs the EntityRuler pipeline on the span text and returns the label of
    the first entity found.  Used as a supplementary signal in candidates_df
    — it does not override the existing soft-classifiers but raises
    confidence when both agree.
    """
    ruler_nlp = _entity_ruler_nlp()
    doc = ruler_nlp(text)
    if doc.ents:
        return str(ent.label_) if (ent := doc.ents[0]) else None
    return None


# ---------------------------------------------------------------------------
# Tolerance constants — declared once here; views.py is the sole word-
# extraction path.  No second path exists to drift against.
#
# _TOKEN_Y_TOLERANCE: pdfplumber extract_words y_tolerance (default = 3).
# Keeping it explicit prevents silent divergence if the pdfplumber default
# ever changes.
# ---------------------------------------------------------------------------

_TOKEN_Y_TOLERANCE: int = 3

# ---------------------------------------------------------------------------
# Stable token-id — canonical recipe.
#
#   SHA1, usedforsecurity=False,
#   input = f"{doc_id}|{page_idx}|{token_idx}|{text}|{x0:.6f},{y0:.6f},{x1:.6f},{y1:.6f}"
#
# Ephemeral: recomputed each run, not persisted.  Tune freely.
# ---------------------------------------------------------------------------


def compute_stable_token_id(
    doc_id: str,
    page_idx: int,
    token_idx: int,
    text: str,
    bbox_norm: tuple[float, float, float, float] | tuple,
) -> str:
    """Compute the deterministic token id for this run.

    Ephemeral: recomputed on every invocation, not persisted to Postgres.
    """
    bbox_str = (
        f"{bbox_norm[0]:.6f},{bbox_norm[1]:.6f},{bbox_norm[2]:.6f},{bbox_norm[3]:.6f}"
    )
    hash_input = f"{doc_id}|{page_idx}|{token_idx}|{text}|{bbox_str}"
    # Deterministic content-addressed ID; usedforsecurity=False per stdlib API.
    # Rebuild owns ID strategy in step 1+ (SQLModel layer); deferring rewrite.
    return hashlib.sha1(  # nosemgrep: python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1
        hash_input.encode("utf-8"), usedforsecurity=False
    ).hexdigest()


# ---------------------------------------------------------------------------
# Views — pure functions of a Doc, returning LazyFrames.
#
# Downstream consumers chain these with further lazy ops and collect once
# at the cost-matrix seam (owned by a later phase).
# ---------------------------------------------------------------------------


def chars_df(doc: Doc) -> pl.LazyFrame:
    """Flat per-character view over a Doc (``CharsDF``).

    One row per :class:`invoices.doc.CharEntry`, with page coordinates and
    normalized bbox.  No grouping / tokenization is performed here —
    that is :func:`tokens_df`'s job.
    """
    rows: list[dict[str, object]] = []
    for page in doc.pages:
        page_w = page.width
        page_h = page.height
        for ch in page.chars:
            rows.append(  # noqa: PERF401 — page_w/page_h are per-outer-iteration; comprehension loses per-page scope
                {
                    "doc_sha256": doc.sha256,
                    "page_idx": page.page_idx,
                    "text": ch.text,
                    "x0": ch.x0,
                    "y0": ch.top,
                    "x1": ch.x1,
                    "y1": ch.bottom,
                    "bbox_norm_x0": (ch.x0 / page_w) if page_w > 0 else 0.0,
                    "bbox_norm_y0": (ch.top / page_h) if page_h > 0 else 0.0,
                    "bbox_norm_x1": (ch.x1 / page_w) if page_w > 0 else 0.0,
                    "bbox_norm_y1": (ch.bottom / page_h) if page_h > 0 else 0.0,
                    "fontname": ch.fontname,
                    "size": ch.size,
                    "page_width": page_w,
                    "page_height": page_h,
                }
            )
    return pl.LazyFrame(rows)


# ---------------------------------------------------------------------------
# Internal helpers for candidates_df — pure Python, no pandas, no collect.
# ---------------------------------------------------------------------------


def _classify_color_bucket(color: Any) -> str:
    """Classify a pdfplumber non_stroking_color into a coarse bucket.

    pdfplumber emits colors as None / scalar / tuple / list across colorspaces
    (DeviceGray scalar in [0, 1]; DeviceRGB / DeviceCMYK tuples).  We collapse
    to two deterministic buckets based on perceived darkness:

    * ``"black"`` — near-black or unknown (default for body copy)
    * ``"colored"`` — anything visibly non-black

    Heuristic: any channel >= 0.3 lands the token in ``"colored"``.  Good
    enough as a ranking signal; downstream code only treats this as an
    opaque categorical.
    """
    if color is None:
        return "black"
    if isinstance(color, (int, float)):
        return "black" if float(color) < 0.3 else "colored"
    if isinstance(color, (list, tuple)):
        try:
            return (
                "black"
                if all(float(c) < 0.3 for c in color if isinstance(c, (int, float)))
                else "colored"
            )
        except (TypeError, ValueError):
            return "black"
    return "black"


def _build_token_dict(
    effective_doc_id: str,
    page_idx: int,
    page_w: float,
    page_h: float,
    token_idx: int,
    word: dict[str, Any],
    *,
    reading_order: int,
) -> dict[str, Any] | None:
    """Build a single token dict from a pdfplumber word entry.

    Used at ingest (tokenize.build_doc) to produce the token dicts persisted
    in DocPage.tokens.  Read-path views project those dicts directly without
    re-invoking this function.

    ``reading_order`` is supplied by the caller as a per-page enumerate index
    over ``page.extract_words()`` output — collision-free and monotonic in
    pdfplumber's native word ordering.

    Returns ``None`` for whitespace-only words (callers must skip those).
    """
    text: str = word["text"]
    if not text.strip():
        return None

    x0 = float(word["x0"])
    y0 = float(word["top"])
    x1 = float(word["x1"])
    y1 = float(word["bottom"])
    doctop = float(word.get("doctop", y0))

    bbox_norm = (
        x0 / page_w if page_w > 0 else 0.0,
        y0 / page_h if page_h > 0 else 0.0,
        x1 / page_w if page_w > 0 else 0.0,
        y1 / page_h if page_h > 0 else 0.0,
    )

    token_id = compute_stable_token_id(
        effective_doc_id, page_idx, token_idx, text, bbox_norm
    )

    font_name: str = str(word.get("fontname", ""))
    font_size: float = float(word.get("size", 0))
    # font_hash is the raw fontname — stable, opaque, no hashing needed.
    # Callers (style features, span aggregation) treat it as an opaque str.
    font_hash: str = font_name
    font_name_lower = font_name.lower()
    is_bold: bool = any(ind in font_name_lower for ind in ("bold", "heavy", "black"))
    is_italic: bool = any(ind in font_name_lower for ind in ("italic", "oblique"))

    # line_id buckets by doctop (page-aware vertical position) at the same
    # y-tolerance pdfplumber used to segment words into lines.  Stable across
    # runs; safe for partition_by in span assembly.
    line_id: int = int(doctop // _TOKEN_Y_TOLERANCE)
    color_bucket: str = _classify_color_bucket(word.get("non_stroking_color"))

    return {
        "token_id": token_id,
        "doc_id": effective_doc_id,
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
        "page_width": page_w,
        "page_height": page_h,
        "font_name": font_name,
        "font_hash": font_hash,
        "font_size": font_size,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "color_bucket": color_bucket,
        "line_id": line_id,
        "reading_order": reading_order,
    }


def _find_typed_anchors_from_dicts(
    token_dicts: list[dict[str, Any]],
    anchor_keywords: dict[str, Any],
    *,
    learned_vendor_anchors: set[str] | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Anchor detection over a list of token dicts via spaCy PhraseMatcher.

    Builds a page-level text string from token texts, runs the cached
    PhraseMatcher (LOWER attribute → case-insensitive) over the spaCy Doc,
    then maps each match span back to the originating token dict by index to
    recover normalised bbox coordinates.

    Rapidfuzz fuzzy matching supplements the PhraseMatcher for the 'name'
    anchor type when ``learned_vendor_anchors`` is supplied: any token whose
    text fuzzy-matches a learned vendor anchor (token_set_ratio ≥ 75) is
    added as a 'name'-type anchor, covering spelling variants and partial
    company-name tokens that exact PhraseMatcher patterns would miss.

    Returns {anchor_type: [(cx, cy), ...]} — same contract as before.
    """
    anchors: dict[str, list[tuple[float, float]]] = {at: [] for at in anchor_keywords}
    if not token_dicts:
        return anchors

    nlp = _nlp_pipeline()
    matcher = _anchor_phrase_matcher()

    # Build a plain-text string from the page tokens, preserving 1-to-1
    # alignment between spaCy token indices and token_dicts indices.
    # We use a single-space join so spaCy's whitespace tokeniser splits on
    # the same boundaries as pdfplumber's word extractor.
    texts = [str(tok.get("text", "")) for tok in token_dicts]
    page_text = " ".join(texts)

    doc = nlp.make_doc(page_text)
    matches = matcher(doc)

    # matched_indices: set of token_dict indices covered by any match
    matched_token_types: dict[int, str] = {}
    for match_id, start, _end in matches:
        anchor_type: str = nlp.vocab.strings[match_id]
        # Each spaCy token in the span maps 1-to-1 to a token_dict entry
        # because we built the doc from a space-joined string.  However, a
        # multi-word phrase like "total due" spans two spaCy tokens (indices
        # start, start+1 = "total", "due"); we assign the anchor to the
        # *first* token of the span, which is the head keyword.
        if start < len(token_dicts):
            # Prefer higher-specificity anchor types (don't overwrite 'total'
            # with 'name' if both match the same token).
            if start not in matched_token_types:
                matched_token_types[start] = anchor_type

    for tok_idx, anchor_type in matched_token_types.items():
        tok = token_dicts[tok_idx]
        cx = (
            float(tok.get("bbox_norm_x0", 0.0)) + float(tok.get("bbox_norm_x1", 0.0))
        ) / 2
        cy = (
            float(tok.get("bbox_norm_y0", 0.0)) + float(tok.get("bbox_norm_y1", 0.0))
        ) / 2
        anchors[anchor_type].append((cx, cy))

    # Rapidfuzz supplement for 'name' anchor type — fuzzy vendor matching.
    # Only runs when learned_vendor_anchors is provided and non-empty, and
    # only for tokens not already matched by the PhraseMatcher.
    if learned_vendor_anchors and "name" in anchors:
        matched_indices = set(matched_token_types)
        for idx, tok in enumerate(token_dicts):
            if idx in matched_indices:
                continue
            text = str(tok.get("text", ""))
            if _fuzzy_vendor_match(text, learned_vendor_anchors):
                cx = (
                    float(tok.get("bbox_norm_x0", 0.0))
                    + float(tok.get("bbox_norm_x1", 0.0))
                ) / 2
                cy = (
                    float(tok.get("bbox_norm_y0", 0.0))
                    + float(tok.get("bbox_norm_y1", 0.0))
                ) / 2
                anchors["name"].append((cx, cy))

    return anchors


def _compute_directional_features_from_anchors(
    bbox_norm: tuple[float, float, float, float],
    anchors_by_type: dict[str, list[tuple[float, float]]],
    directional_defaults: dict[str, float],
    column_threshold: float = 0.08,
    row_threshold: float = 0.03,
) -> dict[str, float]:
    """Pure-Python directional feature computation.

    Mirrors TypedProximityScorer.compute_directional_features exactly.
    No pandas, no DataFrame.
    """
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    features: dict[str, float] = {}
    for anchor_type, anchor_positions in anchors_by_type.items():
        if not anchor_positions:
            features[f"dx_to_{anchor_type}"] = directional_defaults["dx"]
            features[f"dy_to_{anchor_type}"] = directional_defaults["dy"]
            features[f"dist_to_{anchor_type}"] = directional_defaults["dist"]
            features[f"aligned_x_{anchor_type}"] = directional_defaults["aligned_x"]
            features[f"aligned_y_{anchor_type}"] = directional_defaults["aligned_y"]
            features[f"reading_order_{anchor_type}"] = directional_defaults[
                "reading_order"
            ]
            features[f"below_{anchor_type}"] = directional_defaults["below"]
            continue

        min_dist = float("inf")
        nearest_cx = 0.0
        nearest_cy = 0.0
        for acx, acy in anchor_positions:
            dx = cx - acx
            dy = cy - acy
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                nearest_cx = acx
                nearest_cy = acy

        dx = cx - nearest_cx
        dy = cy - nearest_cy

        features[f"dx_to_{anchor_type}"] = dx
        features[f"dy_to_{anchor_type}"] = dy
        features[f"dist_to_{anchor_type}"] = min_dist
        features[f"aligned_x_{anchor_type}"] = (
            1.0 if abs(dx) < column_threshold else 0.0
        )
        features[f"aligned_y_{anchor_type}"] = 1.0 if abs(dy) < row_threshold else 0.0
        features[f"reading_order_{anchor_type}"] = (
            1.0 if dx > 0 and abs(dy) < row_threshold else 0.0
        )
        features[f"below_{anchor_type}"] = (
            1.0 if dy > 0 and abs(dx) < column_threshold else 0.0
        )
    return features


def _compute_proximity_score_from_anchors(
    bbox_norm: tuple[float, float, float, float],
    anchors_by_type: dict[str, list[tuple[float, float]]],
) -> float:
    """Pure-Python proximity score computation.

    Mirrors TypedProximityScorer.compute_proximity_score exactly.
    """
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    min_distance = float("inf")
    for anchor_positions in anchors_by_type.values():
        for acx, acy in anchor_positions:
            dx = abs(cx - acx)
            dy = abs(cy - acy)
            same_line_bonus = 0.1 if dy < 0.02 else 0.0
            reading_order_bonus = 0.05 if cx > acx and dy < 0.02 else 0.0
            distance = math.hypot(dx, dy) - same_line_bonus - reading_order_bonus
            min_distance = min(min_distance, distance)

    if min_distance == float("inf"):
        return 0.0
    return max(0.0, 1.0 - min_distance)


def _has_any_anchor_relationship(
    bbox_norm: tuple[float, float, float, float],
    anchors_by_type: dict[str, list[tuple[float, float]]],
    max_distance: float = 0.5,
) -> bool:
    """Check if candidate is near any anchor."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    for anchor_positions in anchors_by_type.values():
        for acx, acy in anchor_positions:
            dx = cx - acx
            dy = cy - acy
            if math.hypot(dx, dy) <= max_distance:
                return True
    return False


def candidates_df(doc: Doc) -> pl.LazyFrame:
    """Full span-level candidate view over a Doc.

    Phase 0a: pure function of Doc — no doc_id, no learned anchors, no
    persistence.  Emits all columns derivable from the Doc alone.

    This view replaces the thin 1-token projection.  It runs:
      - SpanBuilder: line-local adjacency grouping → multi-token spans
      - typed anchor detection: keyword scan per page → typed anchor positions
      - per-span: geometry, text, style, bucket, bucket_probs, section_prior,
        base directional features, cohesion_score, proximity_score
      - local_density: spatial neighbour count per span

    The downstream chain (candidates.chain.build_candidates_chain) adds:
      - learned-anchor overlay, total_score, soft-NMS, diversity sampling,
        cross-row enrichment (page_frequency, occurrence_rank,
        is_largest_amount_in_doc), doc_id injection.

    ``doc_id`` is intentionally absent — injected at the seam in pipeline.py
    via chain.finalize_candidates.

    ``candidate_id`` is a content-addressed SHA-1 of the sorted list of
    token_ids belonging to the span.  Stable across pipeline runs.

    ``sha256`` is the PDF content hash (from doc.sha256).

    No .collect(), no pandas — pure Python iteration over doc.pages.
    """
    from .candidates.constants import (
        DIRECTIONAL_DEFAULTS,
    )
    from .candidates.features import (
        ANCHOR_PROXIMITY_MAX_DISTANCE,
        _classify_span_bucket,
        compute_geometry_features_enhanced,
        compute_local_density_grid,
        compute_section_prior,
        compute_style_features_enhanced,
        compute_text_features_enhanced,
    )
    from .candidates.patterns import (
        classify_bucket,
        compute_bucket_probabilities,
    )
    from .candidates.spans import PageGrid, SpanBuilder
    from .candidates.validation import filter_candidate_by_bucket
    from .constants import get_anchor_keywords_by_type

    sha256: str = doc.sha256

    anchor_keywords = get_anchor_keywords_by_type()
    span_builder = SpanBuilder()

    # Coverage counters (discarded — pure view, no side effects)
    _coverage: dict[str, int] = {
        "total_spans": 0,
        "date_like_spans": 0,
        "amount_like_spans": 0,
        "id_like_spans": 0,
        "name_like_spans": 0,
        "cue_proximal_spans": 0,
        "region_prior_spans": 0,
        "filtered_no_anchor": 0,
        "filtered_garbage": 0,
    }

    all_records: list[dict[str, Any]] = []

    # Build the sharing map once — list of (source_field, target_field) pairs.
    # Candidates emitted for source_field also get a synthetic clone for target_field.
    _share_pairs: list[tuple[str, str]] = []
    try:
        from . import schema as _schema

        for _fname, _fdef in _schema.load_field_defs().items():
            for _target in _fdef.share_candidate_with:
                _share_pairs.append((_fname, _target))  # noqa: PERF401 — inside try/except; two-level loop is clearer as loop
    except Exception:
        # Schema not seeded (e.g. unit-test contexts) — skip sharing.
        pass

    for page in doc.pages:
        page_idx: int = page.page_idx

        # ── Read token dicts from persisted Doc.pages — no extraction ────────
        token_dicts = [
            dict(tok) if not isinstance(tok, dict) else tok for tok in page.tokens
        ]
        if not token_dicts:
            continue

        # ── SpanBuilder requires a materialized pl.DataFrame ────────────────
        # We construct it here from the already-built token dicts.
        # This is a page-local materialization, NOT a collect() of any LazyFrame.
        page_tokens_df = pl.DataFrame(token_dicts)

        # ── Typed anchor detection — pure Python, no pandas ──────────────────
        anchors_by_type = _find_typed_anchors_from_dicts(token_dicts, anchor_keywords)
        page_has_anchors = any(len(v) > 0 for v in anchors_by_type.values())

        # ── Font size list for style features ───────────────────────────────
        page_font_sizes = [float(t["font_size"]) for t in token_dicts]

        # ── Span assembly ────────────────────────────────────────────────────
        spans: list[dict[str, Any]] = span_builder.build_spans(page_tokens_df)
        if not spans:
            continue
        _coverage["total_spans"] += len(spans)

        # ── Page grid for local density ──────────────────────────────────────
        page_grid = PageGrid()
        page_records: list[dict[str, Any]] = []

        for span in spans:
            # Signal-to-noise filter
            if page_has_anchors:
                text_s = span["raw_text"]
                is_interesting = classify_bucket(text_s) is not None
                if not is_interesting:
                    if not _has_any_anchor_relationship(
                        span["bbox_norm"],
                        anchors_by_type,
                        max_distance=ANCHOR_PROXIMITY_MAX_DISTANCE,
                    ):
                        _coverage["filtered_no_anchor"] += 1
                        continue

            raw_text: str = span["raw_text"]
            _sub_doc = _nlp_pipeline()(raw_text)
            _ent_label: str | None = _sub_doc.ents[0].label_ if _sub_doc.ents else None

            # Text features
            text_features = compute_text_features_enhanced(raw_text)

            # Geometry features
            geometry_features = compute_geometry_features_enhanced(
                span["bbox_norm"], span["page_width"], span["page_height"]
            )

            # Style features
            style_features = compute_style_features_enhanced(
                span["font_size"],
                span["is_bold"],
                span["is_italic"],
                span["font_hash"],
                page_font_sizes,
            )

            # Base directional features — pure Python, no pandas
            directional_features = _compute_directional_features_from_anchors(
                span["bbox_norm"],
                anchors_by_type,
                DIRECTIONAL_DEFAULTS,
            )

            # Proximity score
            proximity_score: float = _compute_proximity_score_from_anchors(
                span["bbox_norm"], anchors_by_type
            )

            # Section prior
            section_prior: float = compute_section_prior(span["bbox_norm"], page_idx)

            # Content-addressed candidate_id: SHA-1 of sorted token_ids
            sorted_tids = sorted(span["token_ids"])
            # Deterministic content-addressed ID; usedforsecurity=False per stdlib API.
            # Rebuild owns ID strategy in step 1+ (SQLModel layer); deferring rewrite.
            cand_id = hashlib.sha1(  # nosemgrep: python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1
                "|".join(sorted_tids).encode("utf-8"), usedforsecurity=False
            ).hexdigest()

            # Bucket classification — primary soft-classifiers, supplemented by
            # EntityRuler hint.  When the soft path returns None (no strong
            # signal), the EntityRuler-derived label upgrades proximity_score
            # enough to pass the keyword_proximal threshold, keeping the span.
            ruler_hint = _ruler_bucket_hint(raw_text)
            effective_proximity = (
                max(proximity_score, 0.35) if ruler_hint else proximity_score
            )
            bucket = _classify_span_bucket(raw_text, effective_proximity, _coverage)
            if bucket is None:
                # Deterministic ~2% random-negative selection via candidate content hash.
                # Legacy used random.random() seeded from sha256 — replaced with
                # per-span hash for order-independence and no global RNG state.
                if int(cand_id[:4], 16) % 100 < 2:
                    bucket = "random_negative"
                else:
                    continue

            # Garbage filter
            if not filter_candidate_by_bucket(raw_text, bucket):
                _coverage["filtered_garbage"] += 1
                continue

            # Bucket probabilities
            bucket_probs = compute_bucket_probabilities(raw_text)

            x0, y0, x1, y1 = span["bbox_norm"]

            record: dict[str, Any] = {
                "candidate_id": cand_id,
                "sha256": sha256,
                "page_idx": page_idx,
                "token_ids": [str(tid) for tid in span["token_ids"]],
                "token_indices": [int(idx) for idx in span["token_indices"]],
                "raw_text": raw_text,
                "normalized_text": span["normalized_text"],
                "ent_label": _ent_label,
                "bucket": bucket,
                "token_count": int(span["token_count"]),
                "cohesion_score": float(span["cohesion_score"]),
                "proximity_score": float(proximity_score),
                "section_prior": float(section_prior),
                # total_score placeholder — set by chain.apply_scoring
                "total_score": 0.0,
                # Bounding box
                "bbox_norm_x0": float(x0),
                "bbox_norm_y0": float(y0),
                "bbox_norm_x1": float(x1),
                "bbox_norm_y1": float(y1),
                # Geometry (schema core + extras used by cost.py)
                "center_x": float(geometry_features["center_x"]),
                "center_y": float(geometry_features["center_y"]),
                "width": float(geometry_features["width"]),
                "height": float(geometry_features["height"]),
                "area": float(geometry_features["area"]),
                "local_density": 0.0,  # filled below after grid is built
                "is_remittance_band": bool(geometry_features["center_y"] > 0.85),
                # Geometry extras (non-schema, strict=False allows them)
                "distance_to_center": float(geometry_features["distance_to_center"]),
                "distance_to_top": float(geometry_features["distance_to_top"]),
                "distance_to_bottom": float(geometry_features["distance_to_bottom"]),
                "distance_to_left": float(geometry_features["distance_to_left"]),
                "distance_to_right": float(geometry_features["distance_to_right"]),
                "y_from_bottom": float(geometry_features["y_from_bottom"]),
                "y0_from_bottom": float(geometry_features["y0_from_bottom"]),
                "aspect_ratio": float(geometry_features["aspect_ratio"]),
                "in_top_half": float(geometry_features["in_top_half"]),
                "in_left_half": float(geometry_features["in_left_half"]),
                "in_bottom_quarter": float(geometry_features["in_bottom_quarter"]),
                "in_top_quarter": float(geometry_features["in_top_quarter"]),
                "in_right_third": float(geometry_features["in_right_third"]),
                "in_amount_region": float(geometry_features["in_amount_region"]),
                # Text (schema core + extras)
                "text_length": int(text_features["text_length"]),
                "digit_ratio": float(text_features["digit_ratio"]),
                "uppercase_ratio": float(text_features["uppercase_ratio"]),
                "currency_flag": bool(text_features["currency_flag"]),
                "unigram_hash": str(text_features.get("unigram_hash", "")),
                "bigram_hash": str(text_features.get("bigram_hash", "")),
                # Style (non-schema extras, strict=False allows them)
                "font_size": float(style_features["font_size"]),
                "font_size_z": float(style_features["font_size_z"]),
                "is_bold": bool(style_features["is_bold"]),
                "is_italic": bool(style_features["is_italic"]),
                "font_hash": str(style_features["font_hash"]),
                "font_size_large": bool(style_features["font_size_large"]),
                "font_size_small": bool(style_features["font_size_small"]),
                # Bucket probabilities (schema core)
                "bucket_prob_date_like": float(bucket_probs.get("date_like", 0.0)),
                "bucket_prob_amount_like": float(bucket_probs.get("amount_like", 0.0)),
                "bucket_prob_id_like": float(bucket_probs.get("id_like", 0.0)),
                "bucket_prob_name_like": float(bucket_probs.get("name_like", 0.0)),
                # Enrichment placeholders — filled by chain.apply_cross_row_enrichment
                "page_frequency": 0.0,
                "occurrence_rank": 1,
                "is_largest_amount_in_doc": 0.0,
                # Anchor presence (5) — binary: was this anchor type found on this page?
                **{
                    f"has_{anchor_type}_anchor": float(
                        len(anchors_by_type.get(anchor_type, [])) > 0
                    )
                    for anchor_type in ANCHOR_TYPES
                },
                # Directional features (35 cols = 5 anchor types x 7 metrics)
                **directional_features,
                # Internal reference — needed for soft-NMS in chain
                "_bbox_norm": span["bbox_norm"],
            }

            page_records.append(record)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            page_grid.add_item(cx, cy, record)

        # Fill local_density now that grid is complete
        for rec in page_records:
            rec["local_density"] = float(
                compute_local_density_grid(rec["_bbox_norm"], page_grid)
            )
            del rec["_bbox_norm"]

        all_records.extend(page_records)

    # Stamp shared_for_field on originals (None = not a synthetic clone).
    for r in all_records:
        r["shared_for_field"] = None

    # Synthetic duplicate emission for field-pairs that share candidate spans.
    # For every (source, target) pair, clone each original record and assign a
    # synthetic candidate_id = sha1(original_id + "|" + target_field_name).
    # shared_for_field marks the clone so downstream routing can identify it.
    # Clones are appended after all originals — pre-existing row indices are stable.
    if _share_pairs:
        synthetic_records: list[dict[str, Any]] = []
        for _original in all_records:
            _original_id = _original["candidate_id"]
            for _source, _target in _share_pairs:
                # Deterministic content-addressed ID; usedforsecurity=False per stdlib API.
                # Rebuild owns ID strategy in step 1+ (SQLModel layer); deferring rewrite.
                synthetic_id = hashlib.sha1(  # nosemgrep: python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1
                    f"{_original_id}|{_target}".encode(), usedforsecurity=False
                ).hexdigest()
                clone = dict(_original)
                clone["candidate_id"] = synthetic_id
                clone["shared_for_field"] = _target
                synthetic_records.append(clone)
        all_records.extend(synthetic_records)

    if not all_records:
        return pl.LazyFrame()

    return pl.LazyFrame(all_records)
