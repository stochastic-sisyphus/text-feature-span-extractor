"""Decode: Hungarian assignment for field-candidate matching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..schema import FieldSpec

from ..logging import get_logger
from ..solver import build_assignments, run_hungarian
from ..types import Assignment

logger = get_logger(__name__)


def decode_document_with_data(
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
    calibration_mapping: dict[str, Any] | None = None,
    vendor_corpus: set[str] | None = None,
) -> dict[str, Assignment]:
    """Pure decode: build cost matrix, run Hungarian, build assignments.

    All inputs as arguments, no file I/O, no Config reads. This is the
    computation kernel that ``decode_document`` delegates to after loading
    data from disk.

    Returns a dict mapping field name to Assignment.
    """
    # Empty candidates → all NONE assignments
    if not candidates_list:
        assignments: dict[str, Assignment] = {}
        for field in schema_fields:
            assignments[field] = Assignment(
                assignment_type="NONE",
                candidate_index=None,
                cost=none_bias,
                field=field,
                used_ml_model=False,
                ml_probability=None,
            )
        return assignments

    n_candidates = len(candidates_list)

    # Build cost matrix via top-level cost module (explicit args, no globals)
    from ..cost import build_cost_matrix

    cost_matrix, ml_probs, fields_with_ml = build_cost_matrix(
        candidates_list=candidates_list,
        schema_fields=schema_fields,
        field_profiles=field_profiles,
        loaded_models=loaded_models,
        doc_labels=doc_labels,
        colon_name_values=colon_name_values,
        cross_page_headers=cross_page_headers,
        address_city_tokens=address_city_tokens,
        field_defs=field_defs,
        base_cost=base_cost,
        none_bias=none_bias,
        ml_score_weight=ml_score_weight,
        bootstrap_ml_score_weight=bootstrap_ml_score_weight,
        field_blend_weights=field_blend_weights,
        vendor_corpus=vendor_corpus,
    )

    # Apply Hungarian algorithm
    row_indices, col_indices = run_hungarian(cost_matrix)

    # Extract bootstrap state to cap ML confidence in solver
    _manifest = loaded_models.get("manifest") if loaded_models else None
    is_bootstrap = _manifest.bootstrap_mode if _manifest is not None else False

    # Build assignments
    return build_assignments(
        row_indices,
        col_indices,
        cost_matrix,
        schema_fields,
        candidates_list,
        ml_probs,
        fields_with_ml,
        n_candidates,
        none_bias,
        is_bootstrap=is_bootstrap,
        calibration_mapping=calibration_mapping,
    )
