"""Pandera DataFrameModel schemas for pipeline stage boundaries.

Polars schemas (LenientDataFrameModel base):
  TokensDF: validates the DataFrame produced by tokenize.py
  CandidatesDF: validates the core structural columns produced by candidates/generation.py

These schemas make implicit DataFrame contracts mechanical -- a column rename
or dtype change produces a SchemaError at the stage boundary instead of a
silent downstream bug.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import pandera.polars as ppa
import polars as pl
from pandera.typing.polars import Series as PSeries

from .schema.pandera_base import LenientDataFrameModel

if TYPE_CHECKING:
    pass

# ─── Token DataFrame ────────────────────────────────────────────────────────


class TokensDF(ppa.DataFrameModel):
    """Schema for the DataFrame produced by tokenize.py.

    23 columns, fixed set. A column rename in tokenize will produce a
    SchemaError here rather than a KeyError in candidates or decoder.
    """

    # Identity
    token_id: PSeries[str] = ppa.Field()
    doc_id: PSeries[str] = ppa.Field()

    # Position
    page_idx: PSeries[int] = ppa.Field(ge=0)
    token_idx: PSeries[int] = ppa.Field(ge=0)

    # Content
    text: PSeries[str] = ppa.Field()

    # Bounding box — PDF coordinate space
    bbox_pdf_units_x0: PSeries[float] = ppa.Field()
    bbox_pdf_units_y0: PSeries[float] = ppa.Field()
    bbox_pdf_units_x1: PSeries[float] = ppa.Field()
    bbox_pdf_units_y1: PSeries[float] = ppa.Field()

    # Bounding box — normalized [0, 1]
    bbox_norm_x0: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_y0: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_x1: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_y1: PSeries[float] = ppa.Field(ge=0.0, le=1.0)

    # Page dimensions
    page_width: PSeries[float] = ppa.Field(gt=0.0)
    page_height: PSeries[float] = ppa.Field(gt=0.0)

    # Typography
    font_name: PSeries[str] = ppa.Field()
    font_hash: PSeries[str] = ppa.Field()
    font_size: PSeries[float] = ppa.Field(ge=0.0)
    is_bold: PSeries[bool] = ppa.Field()
    is_italic: PSeries[bool] = ppa.Field()
    color_bucket: PSeries[str] = ppa.Field()

    # Structural
    line_id: PSeries[int] = ppa.Field(ge=0)
    reading_order: PSeries[int] = ppa.Field(ge=0)

    class Config:
        strict = True
        coerce = True


# ─── Candidate DataFrame ────────────────────────────────────────────────────


class CandidatesDF(LenientDataFrameModel):
    """Schema for the DataFrame produced by candidates/generation.py.

    Defines the CORE structural columns explicitly. The candidates DataFrame
    also contains dynamic feature columns (directional, position, style, text
    hash, geometry detail) which are allowed by strict=False.

    strict=False because:
    - Directional columns (35) are dynamically named from ANCHOR_TYPES
    - Geometry detail columns (distance_to_*, y0_from_bottom, aspect_ratio, etc.)
    - Style detail columns (font_size_z, font_size_large, font_size_small, etc.)
    - Text hash columns (unigram_hash, bigram_hash)
    - Position indicator columns (in_left_half, in_top_half, etc.)
    - The CandidateFeatures dataclass governs the ML feature contract separately
    """

    # ── Identity ──
    candidate_id: PSeries[str] = ppa.Field()
    doc_id: PSeries[str] = ppa.Field()
    sha256: PSeries[str] = ppa.Field()
    page_idx: PSeries[int] = ppa.Field(ge=0)

    # ── Token references (polars List columns) ──
    token_ids: Annotated[pl.List, pl.Utf8()] = ppa.Field()
    token_indices: Annotated[pl.List, pl.Int64()] = ppa.Field()

    # ── Text ──
    raw_text: PSeries[str] = ppa.Field()
    normalized_text: PSeries[str] = ppa.Field()
    bucket: PSeries[str] = ppa.Field()

    # ── Scoring ──
    token_count: PSeries[int] = ppa.Field(ge=1)
    cohesion_score: PSeries[float] = ppa.Field()
    proximity_score: PSeries[float] = ppa.Field()
    section_prior: PSeries[float] = ppa.Field()
    total_score: PSeries[float] = ppa.Field()

    # ── Bounding box — normalized [0, 1] ──
    bbox_norm_x0: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_y0: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_x1: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bbox_norm_y1: PSeries[float] = ppa.Field(ge=0.0, le=1.0)

    # ── Core spatial features (from compute_geometry_features_enhanced) ──
    center_x: PSeries[float] = ppa.Field()
    center_y: PSeries[float] = ppa.Field()
    width: PSeries[float] = ppa.Field()
    height: PSeries[float] = ppa.Field()
    area: PSeries[float] = ppa.Field()
    local_density: PSeries[float] = ppa.Field()
    is_remittance_band: PSeries[bool] = ppa.Field(coerce=True)

    # ── Core text features (from compute_text_features_enhanced) ──
    text_length: PSeries[int] = ppa.Field(ge=0, coerce=True)
    digit_ratio: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    uppercase_ratio: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    currency_flag: PSeries[bool] = ppa.Field(coerce=True)

    # ── Enrichment features ──
    page_frequency: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    occurrence_rank: PSeries[int] = ppa.Field(ge=1, coerce=True)
    is_largest_amount_in_doc: PSeries[float] = ppa.Field(ge=0.0, le=1.0, coerce=True)

    # ── Bucket probabilities ──
    bucket_prob_date_like: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bucket_prob_amount_like: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bucket_prob_id_like: PSeries[float] = ppa.Field(ge=0.0, le=1.0)
    bucket_prob_name_like: PSeries[float] = ppa.Field(ge=0.0, le=1.0)


# ─── Projection-boundary validation helper ───────────────────────────────────


def lenient_validate(
    df: pl.DataFrame,
    model: type[LenientDataFrameModel],
    **ctx: str,
) -> None:
    """Validate *df* against *model*; log on violation, never raise.

    Uses ``lazy=True`` to collect all errors rather than stopping on the
    first — consistent with projection-boundary semantics (observe, don't gate).

    Optional ``**ctx`` kwargs are merged into the structured log payload so
    callers can attach diagnostic fields (e.g. ``sha256=sha256[:16]``,
    ``field=field_name``).

    Example::

        lenient_validate(df, CandidatesDF)
        lenient_validate(df, LabeledCandidatesDF, sha256=sha256[:16])
        lenient_validate(df, FeatureDatasetDF, field=field_name)
    """
    if df.is_empty():
        return
    try:
        model.validate(df, lazy=True)
    except Exception as exc:
        try:
            from invoices.logging import get_logger

            logger = get_logger("invoices.schemas")
        except Exception:
            return
        payload: dict = {
            "validator": model.__name__,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        payload.update(ctx)
        logger.warning("schema_validation_drift", **payload)
