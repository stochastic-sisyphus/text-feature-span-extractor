"""Top-level assignment normalization across a document."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._date import extract_page_year
from ._dispatch import normalize_field_value

if TYPE_CHECKING:
    import polars as pl


def _narrow_span_to_field_entity(field: str, raw_text: str) -> str:
    """Narrow an over-greedy candidate span to its atomic typed value via NER.

    When the decoder selects a multi-token span (e.g. "Please pay $184.97 by
    Dec 16,") for a typed field, extract the spaCy NER entity matching the
    field's declared normalizer (amount/currency -> MONEY, date -> DATE) so the
    downstream normalizer sees the atomic value rather than the whole phrase.

    Type-aware and NLP-native: reuses ``_nlp`` and ``NORMALIZER_TO_ENTITY_LABEL``.
    Only narrows amount/currency/date fields, where the NER span *is* the value;
    name/id/text fields are returned unchanged (NER narrowing would truncate
    multi-word names / mismatch alphanumeric ids). Returns ``raw_text`` unchanged
    whenever narrowing does not apply or is ambiguous, so clean single-value
    spans are never altered (regression-safe).
    """
    text = raw_text.strip()
    if not text or " " not in text:
        # Already atomic — nothing to narrow.
        return raw_text
    from .. import schema as schema_mod
    from ..constants import NORMALIZER_TO_ENTITY_LABEL

    fd = schema_mod.load_field_defs().get(field)
    if fd is None:
        return raw_text
    labels = NORMALIZER_TO_ENTITY_LABEL.get(
        getattr(fd, "normalizer", None) or "", frozenset()
    )
    # Only narrow types whose NER entity is the atomic value itself.
    if labels not in (frozenset({"MONEY"}), frozenset({"DATE"})):
        return raw_text
    try:
        from ..candidates.patterns import _nlp

        ents = [str(e.text) for e in _nlp()(text).ents if e.label_ in labels]  # type: ignore[operator]
    except Exception:
        return raw_text
    # Only narrow when exactly one matching entity exists (unambiguous) and it is
    # a strict, non-empty substring of the span.
    if len(ents) == 1 and ents[0].strip() and ents[0] != text:
        return ents[0]
    return raw_text


def normalize_assignments(
    assignments: dict[str, Any],
    sha256: str | None = None,
    tokens_df: pl.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Normalize all field assignments for a document.

    Preserves ML metadata (used_ml_model, ml_probability) from decoder
    for confidence scoring downstream.

    When tokens_df is provided directly, uses it for page-year extraction.
    Otherwise, when sha256 is provided, loads document tokens from disk.

    Args:
        assignments: Raw assignments from decoder
        sha256: Optional document hash for loading tokens (enables page-year fallback)
        tokens_df: Optional pre-loaded tokens DataFrame (avoids disk read)

    Returns:
        Normalized assignments with cleaned values and preserved ML metadata
    """
    # Use provided tokens_df, or load from disk if sha256 given
    if tokens_df is not None and tokens_df.is_empty():
        tokens_df = None
    normalized = {}

    from ..types import Assignment

    for field, assignment in assignments.items():
        if assignment.assignment_type == "NONE":
            # No normalization needed for NONE assignments
            normalized[field] = Assignment(
                assignment_type="NONE",
                candidate_index=None,
                cost=assignment.cost,
                field=field,
                used_ml_model=assignment.used_ml_model,
                ml_probability=assignment.ml_probability,
                normalized_value=None,
                raw_text=None,
                currency_code=None,
            )
        else:
            # Normalize the candidate value.
            # LiLT unmatched-span assignments carry candidate=None and raw_text on
            # the assignment itself; IoU-matched assignments carry a candidate dict.
            candidate = assignment.candidate
            # The decoder's extracted span text (assignment.raw_text) is the
            # authoritative value when present — the LiLT token-classifier span,
            # tighter/fuller than the IoU-matched candidate (which is provenance
            # only: bbox / candidate_index for the UI). The Hungarian decoder
            # leaves raw_text=None → candidate text is used (unchanged behavior).
            # ``... or ...`` (not dict.get default) guards the present-but-None
            # raw_text column footgun.
            raw_text = assignment.raw_text or ""
            if not raw_text and candidate is not None:
                raw_text = candidate.get("raw_text") or candidate.get("text") or ""
            raw_text = _narrow_span_to_field_entity(field, raw_text)

            # Extract page year for date fallback
            page_year = None
            if tokens_df is not None and candidate is not None:
                page_idx = candidate.get("page_idx", 0)
                page_year = extract_page_year(tokens_df, page_idx)

            normalization_result = normalize_field_value(
                field, raw_text, page_year=page_year
            )

            normalized[field] = Assignment(
                assignment_type="CANDIDATE",
                candidate_index=assignment.candidate_index,
                cost=assignment.cost,
                field=field,
                candidate=candidate,
                used_ml_model=assignment.used_ml_model,
                ml_probability=assignment.ml_probability,
                fallback_candidates=assignment.fallback_candidates,
                normalized_value=normalization_result["value"],
                raw_text=normalization_result["raw_text"],
                currency_code=normalization_result["currency_code"],
            )

    return normalized
