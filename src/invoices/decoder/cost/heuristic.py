"""Heuristic (weak-prior) cost computation and vendor/disagreement constants."""

from __future__ import annotations

import math
from typing import Any

from ... import fuzzy as _fuzzy
from ...constants import BASE_TYPES_WITH_MAGNITUDE_BONUS, FIELD_TYPES_WITH_TEXT_BONUS
from ...logging import get_logger
from ...normalize import normalize_vendor_name as _nvn
from ...schema import FieldSpec, build_field_spec
from .. import weights as _w_mod
from ..weights import DecoderWeights
from ._helpers import (
    _header_region_bonus,
    _parse_amount_value,
    compute_signal_disagreement,
)
from .kernel import _compute_schema_driven_costs, compute_text_pattern_bonus

logger = get_logger(__name__)


BUCKET_PENALTY_STRONG: float = -0.4

# Signal disagreement thresholds
DISAGREEMENT_HIGH_THRESHOLD: float = 0.7
DISAGREEMENT_CONFIDENCE_DEMOTION: float = 0.25


def compute_weak_prior_cost(
    field: str,
    candidate: dict[str, Any],
    profile: FieldSpec | None = None,
    document_labels: set[str] | None = None,
    colon_name_values: dict[str, int] | None = None,
    cross_page_headers: set[str] | None = None,
    address_city_tokens: set[str] | None = None,
    *,
    weights: DecoderWeights | None = None,
    vendor_corpus: set[str] | None = None,
) -> float:
    """
    Compute weak prior cost for field-candidate assignment based on heuristics.
    This serves as a fallback when no ML model is available.
    Lower cost = better match.

    Schema-Driven Design:
    Dynamically reads field configuration from the active contract schema
    (postgres-backed, seeded from schema/contract.invoice.seed.json on first boot):
    - Field type determines anchor preferences automatically
    - Bucket preferences come from schema's bucket_preference array
    - Adding new fields requires only schema updates, no code changes

    Uses directional vector features to match fields with typed anchors:
    - A value BELOW a "Total" header is likely the total amount
    - A value TO THE RIGHT of "Invoice Date:" is likely the date
    - Direction matters, not just distance

    Args:
        field: Field name (e.g., "TotalAmount", "InvoiceDate")
        candidate: Candidate dictionary with features
        profile: Optional pre-built FieldSpec to avoid registry lookups.
                 When provided, all schema registry lookups use cached values.
        document_labels: Optional set of structural label tokens for this document.
        cross_page_headers: Optional set of cross-page header tokens (3+ pages).
        address_city_tokens: Optional set of address city tokens (near ZIP codes).

    Returns:
        Cost value (lower = better match for this field)
    """
    w = weights or _w_mod.DWeights
    bucket = candidate.get("bucket", "")
    proximity_score = candidate.get("proximity_score", 0.0)
    section_prior = candidate.get("section_prior", 0.0)
    cohesion_score = candidate.get("cohesion_score", 0.0)

    # Build spec unconditionally — avoids registry-fallback taxonomy drift.
    if profile is None:
        profile = build_field_spec(field)

    # Feature-based cost using ML-extracted features
    base_cost = 1.0

    # ================================================================
    # FIELD-SPECIFIC BUCKET AFFINITY + DIRECTIONAL BONUS
    # All fields now schema-driven via registry
    # ================================================================
    bucket_bonus, directional_bonus = _compute_schema_driven_costs(
        field, candidate, profile, w=w
    )

    # ================================================================
    # COMMON BONUSES/PENALTIES (apply to all fields)
    # ================================================================

    # Keyword proximity bonus applies to structured fields (id, amount, date)
    # For text fields (names), keyword_proximal is often noise (generic words near keywords)
    if bucket == "keyword_proximal":
        # Only fields with keyword_proximal=True benefit from keyword proximity
        if profile.is_keyword_proximal:
            bucket_bonus += w.BUCKET_KEYWORD_PROXIMAL_BONUS

    # Random negative penalty applies to all fields
    elif bucket == "random_negative":
        bucket_bonus = BUCKET_PENALTY_STRONG  # -0.4

    # ================================================================
    # TEXT PATTERN VALIDATION (Critical for avoiding garbage assignments)
    # ================================================================
    # Uses native spaCy entity labels to confirm the candidate text matches the
    # expected field type. High weight because bucket matching alone is insufficient.
    # ================================================================
    text_pattern_bonus = compute_text_pattern_bonus(profile, candidate, weights=w)

    # ================================================================
    # FOOTER REGION BONUS
    # Footer elements (Total, Subtotal, Tax) benefit from being near page bottom
    # ================================================================
    y_from_bottom = candidate.get("y_from_bottom", 0.5)
    in_footer_region = candidate.get("in_bottom_quarter", 0.0)

    if profile.is_footer:
        if in_footer_region > 0 or y_from_bottom < w.FOOTER_Y_THRESHOLD:
            directional_bonus += w.FOOTER_REGION_BONUS

    # ================================================================
    # FIELD PRIORITY TIE-BREAKER
    # ================================================================
    # When multiple fields have identical costs for the same candidate,
    # add a tiny bias favoring more important fields.
    # This prevents Hungarian assignment from arbitrarily preferring Subtotal
    # over TotalAmount when both match equally well.
    # ================================================================
    field_priority_bonus = profile.priority_bonus

    # ================================================================
    # FIELD-SPECIFIC COLON-VALUE BONUS
    # "To: GTT Americas LLC" pattern — header name fields get rank-weighted
    # bonus (first word gets most). Other fields get +0.5.
    # Computed before page-frequency bonus so it can gate that bonus.
    # ================================================================
    colon_bonus = 0.0
    if colon_name_values:
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        words = raw_text.lower().split()
        matched_ranks = [
            colon_name_values[wd] for wd in words if wd in colon_name_values
        ]
        if matched_ranks:
            _is_header_text_field = (
                profile.normalizer in FIELD_TYPES_WITH_TEXT_BONUS and profile.is_header
            )
            if _is_header_text_field:
                # Rank-weighted: rank 0 gets 3.0, rank 1 gets 2.0, etc.
                colon_bonus = sum(
                    max(0.5, 3.0 - rank * 1.0) for rank in matched_ranks
                ) / len(matched_ranks)
            else:
                colon_bonus = 0.5

    # ================================================================
    # PAGE-FREQUENCY BONUS
    # Vendor logos/names repeat across pages; other fields don't.
    # Only header name fields get this bonus (scale 5.0).
    # ================================================================
    page_freq_bonus = 0.0
    _is_header_text_field_pf = (
        profile.normalizer in FIELD_TYPES_WITH_TEXT_BONUS and profile.is_header
    )
    if _is_header_text_field_pf:
        # Only apply page-frequency bonus if the candidate passed text
        # validation (non-negative score) OR has colon-value support.
        # Blacklisted words like "CONSOLIDATED", "BILLING" repeat across
        # pages but are labels, not vendor names — they must NOT benefit
        # from page-frequency. Cross-page headers with colon support
        # (like "AT&T" after "To:") are legitimate vendor names that
        # should still get the bonus.
        has_colon_support = colon_bonus > 0
        if text_pattern_bonus >= 0 or has_colon_support:
            page_freq = candidate.get("page_frequency", 0.0)
            page_freq_bonus = page_freq * 5.0

    # ================================================================
    # PRIMARY AMOUNT MAGNITUDE BONUS
    # The total is usually the largest amount on the invoice.
    # Give a small bonus proportional to the log of the parsed amount.
    # This helps disambiguate when multiple amounts have similar costs.
    # ================================================================
    magnitude_bonus = 0.0
    _is_primary_magnitude = (
        profile.base_type in BASE_TYPES_WITH_MAGNITUDE_BONUS
        and profile.importance >= 0.9
    )
    if _is_primary_magnitude:
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        parsed_val = _parse_amount_value(raw_text)
        if parsed_val is not None and abs(parsed_val) > 0:
            # log10 scaling: $100 = 0.8, $1000 = 1.2, $10000 = 1.6, $100000 = 2.0
            # Scale by 2.0 so magnitude can overcome directional/proximity bonuses
            magnitude_bonus = min(2.0, math.log10(max(1.0, abs(parsed_val))) / 2.5)

    # ================================================================
    # DENSE LABEL DETECTION
    # Count both x and y alignments for anchor type detection.
    # Candidates aligned with many other candidates on both axes
    # are likely in a table/label region — penalize for name fields.
    # ================================================================
    dense_label_penalty = 0.0
    if profile.normalizer in FIELD_TYPES_WITH_TEXT_BONUS:
        x_aligned = candidate.get("aligned_x_name", 0.0)
        y_aligned = candidate.get("aligned_y_name", 0.0)
        if x_aligned > 0 and y_aligned > 0:
            dense_label_penalty = -0.3

    # ================================================================
    # VENDOR FUZZY CORPUS PRIOR (T11)
    # Gate: VendorName field only (is_header=True, normalizer="name").
    # Compares normalized candidate text against the corrections corpus
    # using rapidfuzz token_set_ratio.  Raw continuous score (0-100)
    # written to candidate["vendor_fuzzy_score"] for XGBRanker.
    # ================================================================
    _is_vendor_gate = (
        profile.normalizer in FIELD_TYPES_WITH_TEXT_BONUS and profile.is_header
    )
    if _is_vendor_gate and vendor_corpus:
        raw_text = str(candidate.get("raw_text") or candidate.get("text", ""))
        candidate_norm = _nvn(raw_text)
        if candidate_norm:
            hit = _fuzzy.fuzzy_best_match(candidate_norm, vendor_corpus)
            if hit is not None:
                candidate["vendor_fuzzy_score"] = hit[1]
                logger.debug(
                    "vendor_fuzzy_hit",
                    field=field,
                    score=hit[1],
                    matched=hit[0][:40] if hit[0] else None,
                )

    # ================================================================
    # COMBINE ALL FEATURES INTO FINAL COST
    # ================================================================
    # Weights: text pattern (highest) > bucket > directional > secondary signals
    # ================================================================

    # For text/name fields, apply directional dampening to anchor bonuses separately
    # The 0.5 dampening should ONLY affect anchor-based directional bonuses,
    # NOT the header bonus (which is a strong spatial signal)
    directional_weight = 1.0
    header_bonus_component = 0.0
    anchor_directional_component = directional_bonus

    if profile.is_header or profile.normalizer in FIELD_TYPES_WITH_TEXT_BONUS:
        # For header fields, separate header bonus from anchor-based directional bonus
        if profile.is_header:
            header_bonus_component = _header_region_bonus(candidate, w=w)
            anchor_directional_component = directional_bonus - header_bonus_component
        # Reduce weight for anchor-based spatial features (not header bonus)
        directional_weight = w.DIRECTIONAL_DAMPENING

    # Amplify negative text_pattern_bonus (wrong type penalty)
    amplified_text_pattern = (
        text_pattern_bonus * w.TEXT_PATTERN_NEGATIVE_AMPLIFIER
        if text_pattern_bonus < 0
        else text_pattern_bonus
    )

    # Collect weighted signal components for disagreement computation.
    # Each value represents the contribution of one signal dimension to
    # the final cost.  Variance across these captures conflict.
    signal_components = [
        bucket_bonus,
        header_bonus_component,
        anchor_directional_component * directional_weight,
        amplified_text_pattern,
        proximity_score * w.PROXIMITY_WEIGHT,
        section_prior * w.SECTION_PRIOR_WEIGHT,
        (cohesion_score / w.COHESION_NORMALIZER) * w.COHESION_WEIGHT,
        field_priority_bonus,
        page_freq_bonus,
        colon_bonus,
        dense_label_penalty,
        magnitude_bonus,
    ]

    # Attach per-field signal disagreement to candidate dict (metadata, not cost).
    candidate[f"_signal_disagreement:{field}"] = compute_signal_disagreement(
        signal_components
    )

    feature_cost = base_cost - sum(signal_components)

    # Schema-driven spatial bias: penalize candidates in wrong page region.
    # e.g., InvoiceDate has spatial_bias.position="top" → penalize if in bottom half
    # e.g., DueDate has spatial_bias.position="bottom" → penalize if in top third
    s_bias = profile.spatial_bias
    if s_bias:
        center_y = candidate.get("center_y", 0.5)
        if s_bias.position == "top" and center_y > s_bias.threshold:
            feature_cost += s_bias.penalty
        elif s_bias.position == "bottom" and center_y < s_bias.threshold:
            feature_cost += s_bias.penalty

    # Allow negative costs - Hungarian algorithm works with any cost values
    # Lower cost = better match. Clamping at 0 loses all discrimination.

    return feature_cost  # type: ignore[no-any-return]
