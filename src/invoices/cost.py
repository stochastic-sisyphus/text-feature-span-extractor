"""Top-level cost matrix builder for Hungarian assignment.

Encapsulates the cost matrix construction logic: candidate scoring,
cross-field adjustments, and per-field NONE derivation.  All inputs
are explicit — no Config reads, no schema loading, no adaptive
detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .decoder.cost import (
    apply_cross_field_adjustments,
    compute_ml_cost_with_prob,
    compute_weak_prior_cost,
)
from .logging import get_logger

if TYPE_CHECKING:
    from .schema import FieldSpec

logger = get_logger(__name__)

# NONE-penalty factors for auto-derived per-field NONE costs.
# Optional fields get a cost *below* the candidate median — a soft pull toward
# NONE so the solver doesn't force a bad match when no good candidate exists.
# Required fields get a cost *above* the median — a soft nudge that makes NONE
# slightly less attractive without hard-gating it.  The labeler's "required" markings
# are intent signals, not guarantees, so we never use a prohibitive penalty.
NONE_COST_OPTIONAL_FACTOR = 0.8  # soft pull toward NONE for non-required fields
NONE_COST_REQUIRED_FACTOR = 1.5  # soft nudge away from NONE for required fields


def build_cost_matrix(
    candidates_list: list[dict[str, Any]],
    schema_fields: list[str],
    field_profiles: dict[str, FieldSpec],
    loaded_models: dict[str, Any] | None,
    doc_labels: set[str],
    colon_name_values: dict[str, int],
    cross_page_headers: set[str],
    address_city_tokens: set[str],
    field_defs: dict[str, dict[str, Any]],
    base_cost: float,
    none_bias: float,
    ml_score_weight: float = 0.7,
    bootstrap_ml_score_weight: float = 0.3,
    field_blend_weights: dict[str, Any] | None = None,
    vendor_corpus: set[str] | None = None,
) -> tuple[np.ndarray, dict[int, dict[int, float | None]], set[str]]:
    """Build the cost matrix for Hungarian assignment.

    Takes ALL inputs as explicit args — no Config imports, no schema
    loading, no adaptive detection.

    Args:
        candidates_list: List of candidate dicts (from DataFrame.to_dict("records")).
        schema_fields: Ordered list of field names.
        field_profiles: Pre-built FieldSpec per field name.
        loaded_models: Loaded ML models dict, or None.
        doc_labels: Structural label tokens detected for this document.
        colon_name_values: Colon-separated name-value tokens.
        cross_page_headers: Cross-page header tokens (3+ pages).
        address_city_tokens: Address city tokens (near ZIP codes).
        field_defs: Field definitions from schema (field_name -> def dict).
        base_cost: Default cost for every cell (Config.decoder_base_cost).
        none_bias: Cost for NONE assignment.  Negative = auto-derive per-field.
        ml_score_weight: Blend weight for ML ranker scores (non-bootstrap).
        bootstrap_ml_score_weight: Blend weight for ML ranker scores (bootstrap mode).
        field_blend_weights: Per-field ML blend weights (field_name -> weight).
            When provided, overrides ``ml_score_weight`` for fields present in the dict.

    Returns:
        (cost_matrix, ml_probs, fields_with_ml)
        - cost_matrix: shape (n_fields, n_candidates + n_fields)
        - ml_probs: field_idx -> cand_idx -> ML probability (or None)
        - fields_with_ml: set of field names with ML models
    """
    n_candidates = len(candidates_list)
    n_fields = len(schema_fields)

    # Init cost matrix: fields x (candidates + per-field NONE columns)
    cost_matrix = np.full((n_fields, n_candidates + n_fields), base_cost)

    # Track ML probabilities for confidence scoring
    ml_probs: dict[int, dict[int, float | None]] = {i: {} for i in range(n_fields)}

    # Track which fields have ML models
    fields_with_ml: set[str] = set()
    if loaded_models:
        fields_with_ml = set(loaded_models.get("models", {}).keys())

    # Fill candidate costs (NONE columns set in second pass below)
    for field_idx, field in enumerate(schema_fields):
        profile = field_profiles[field]
        # Per-field blend weight: use field-specific weight if available,
        # otherwise fall back to the global ml_score_weight.
        field_ml_weight = (
            field_blend_weights.get(field, ml_score_weight)
            if field_blend_weights
            else ml_score_weight
        )
        for cand_idx, candidate in enumerate(candidates_list):
            # Always compute heuristic cost first — populates
            # _signal_disagreement:{field} on the candidate.
            # The heuristic cost is passed explicitly to the ML path
            # so compute_ranker_cost can derive ML-heuristic divergence.
            heuristic_cost = compute_weak_prior_cost(
                field,
                candidate,
                profile=profile,
                document_labels=doc_labels,
                colon_name_values=colon_name_values,
                cross_page_headers=cross_page_headers,
                address_city_tokens=address_city_tokens,
                vendor_corpus=vendor_corpus,
            )
            if loaded_models and field in fields_with_ml:
                cost, _ = compute_ml_cost_with_prob(
                    field,
                    candidate,
                    loaded_models,
                    ml_score_weight=field_ml_weight,
                    bootstrap_ml_score_weight=bootstrap_ml_score_weight,
                    heuristic_cost=heuristic_cost,
                )
                ml_probs[field_idx][cand_idx] = 1.0 - cost
            else:
                cost = heuristic_cost
                ml_probs[field_idx][cand_idx] = None
            cost_matrix[field_idx, cand_idx] = cost

    # Cross-field consistency adjustment: reward candidate sets where
    # amount fields are internally consistent (e.g., Subtotal + Tax ~ Total).
    # Runs BEFORE Hungarian so it can influence the assignment.
    apply_cross_field_adjustments(cost_matrix, candidates_list, schema_fields)

    # Auto-derive per-field NONE bias from data when sentinel (<0) is set.
    use_per_field_none = none_bias < 0
    per_field_none_costs: np.ndarray | None = None
    _p75_summary: float = 0.0
    if use_per_field_none:
        if n_candidates == 0:
            none_bias = 0.0
        else:
            candidate_costs = cost_matrix[:, :n_candidates]
            per_field_medians = np.median(candidate_costs, axis=1)
            per_field_none_costs = per_field_medians * NONE_COST_OPTIONAL_FACTOR

            # Required fields get a soft NONE nudge: NONE_COST_REQUIRED_FACTOR x
            # the field's own candidate median.  The labeler's required markings are
            # intent-only (noisy), so we never hard-gate with a prohibitive
            # penalty — NONE assignment remains possible when no good candidate
            # exists.
            for field_idx, field in enumerate(schema_fields):
                req_profile = field_profiles.get(field)
                if req_profile is not None and req_profile.required:
                    per_field_none_costs[field_idx] = (
                        per_field_medians[field_idx] * NONE_COST_REQUIRED_FACTOR
                    )

            _p75_summary = float(np.percentile(per_field_medians, 75))
        logger.info(
            "none_bias_auto_derived",
            mode="per_field",
            summary_p75=_p75_summary,
            n_fields=n_fields,
            n_candidates=n_candidates,
            per_field_none_sample=(
                {
                    schema_fields[i]: round(float(per_field_none_costs[i]), 4)
                    for i in range(min(5, n_fields))
                }
                if per_field_none_costs is not None
                else {}
            ),
        )

    # Second pass: set per-field NONE columns (diagonal in the NONE block).
    # Off-diagonal NONE entries must be prohibitive so the Hungarian algorithm
    # never assigns a field to another field's NONE column.
    cost_matrix[:, n_candidates:] = 1e9
    for field_idx in range(n_fields):
        none_col = n_candidates + field_idx
        if per_field_none_costs is not None:
            cost_matrix[field_idx, none_col] = per_field_none_costs[field_idx]
        else:
            cost_matrix[field_idx, none_col] = none_bias
        ml_probs[field_idx][none_col] = None

    return cost_matrix, ml_probs, fields_with_ml
