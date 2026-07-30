"""Pure Hungarian assignment wrapper.

Takes a cost matrix and returns assignments.  Depends on numpy, scipy,
and Config (for ``bootstrap_confidence_cap``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import Config
from .logging import get_logger
from .types import Assignment, FallbackCandidate

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------


def run_hungarian(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the Hungarian algorithm on *cost_matrix*.

    Returns ``(row_indices, col_indices)`` - the optimal assignment.
    """
    if cost_matrix.size == 0 or cost_matrix.ndim < 2:
        logger.info("hungarian_skipped_empty_matrix", shape=cost_matrix.shape)
        return np.array([], dtype=int), np.array([], dtype=int)

    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    return row_indices, col_indices


# ---------------------------------------------------------------------------
# Post-processing: raw indices -> assignments dict
# ---------------------------------------------------------------------------


def build_assignments(
    row_indices: np.ndarray,
    col_indices: np.ndarray,
    cost_matrix: np.ndarray,
    schema_fields: list[str],
    candidates_list: list[dict[str, Any]],
    ml_probs: dict[int, dict[int, float | None]],
    fields_with_ml: set[str],
    n_candidates: int,
    none_bias: float,
    is_bootstrap: bool = False,
    calibration_mapping: dict[str, Any] | None = None,
) -> dict[str, Assignment]:
    """Convert Hungarian solution indices into an assignments dict.

    Parameters
    ----------
    row_indices, col_indices:
        Output of :func:`run_hungarian`.
    cost_matrix:
        The cost matrix that was solved.
    schema_fields:
        Ordered list of field names (one per row in the cost matrix).
    candidates_list:
        List of candidate dicts (one per candidate column).
    ml_probs:
        ``field_idx -> cand_idx -> ML probability (or None)``.
    fields_with_ml:
        Set of field names that have an ML model.
    n_candidates:
        Number of real candidate columns (columns >= n_candidates are NONE).
    none_bias:
        Default cost used when a field has no assignment.
    is_bootstrap:
        Whether in bootstrap mode (reduces ML confidence influence).
    calibration_mapping:
        Persisted calibration mapping (isotonic or Platt). When provided,
        ``apply_calibration`` is applied to each ``ml_prob`` after the
        bootstrap_cap step.  ``None`` means no calibration.
    """
    assignments: dict[str, Assignment] = {}

    for field_idx, field in enumerate(schema_fields):
        # Find assignment for this field
        field_assignment: tuple[int, float] | None = None
        for row_idx, col_idx in zip(row_indices, col_indices, strict=False):
            if row_idx == field_idx:
                field_assignment = (int(col_idx), float(cost_matrix[row_idx, col_idx]))
                break

        # Determine if ML was used for this field
        field_used_ml = field in fields_with_ml

        if field_assignment is None:
            # No assignment found, default to NONE
            assignments[field] = Assignment(
                assignment_type="NONE",
                candidate_index=None,
                cost=none_bias,
                field=field,
                used_ml_model=False,
                ml_probability=None,
            )
        else:
            col_idx_val, cost = field_assignment

            if col_idx_val >= n_candidates:
                # NONE assignment (per-field NONE columns)
                assignments[field] = Assignment(
                    assignment_type="NONE",
                    candidate_index=None,
                    cost=cost,
                    field=field,
                    used_ml_model=False,
                    ml_probability=None,
                )
            else:
                # Candidate assignment
                ml_prob: float | None = ml_probs[field_idx].get(col_idx_val)
                if ml_prob is not None:
                    if is_bootstrap:
                        ml_prob = min(ml_prob, Config.bootstrap_confidence_cap)
                    if calibration_mapping is not None:
                        from .calibration import apply_calibration

                        ml_prob = apply_calibration(
                            ml_prob, calibration_mapping, field=field
                        )

                # Build fallback candidates: top 3 alternatives sorted
                # by cost (excluding assigned candidate and NONE cols)
                field_costs = cost_matrix[field_idx, :n_candidates]
                # Get indices of all candidates except the assigned one
                alt_indices = [i for i in range(n_candidates) if i != col_idx_val]
                # Build fallback candidates: all alternatives sorted by cost
                # (excluding assigned candidate and NONE cols)
                alt_indices.sort(key=lambda i: field_costs[i])
                fallbacks: list[FallbackCandidate] = []
                for alt_idx in alt_indices:
                    alt_ml_prob: float | None = ml_probs[field_idx].get(alt_idx)
                    if alt_ml_prob is not None:
                        if is_bootstrap:
                            alt_ml_prob = min(
                                alt_ml_prob, Config.bootstrap_confidence_cap
                            )
                        if calibration_mapping is not None:
                            from .calibration import apply_calibration

                            alt_ml_prob = apply_calibration(
                                alt_ml_prob, calibration_mapping, field=field
                            )
                    fallbacks.append(
                        FallbackCandidate(
                            candidate_index=alt_idx,
                            cost=float(field_costs[alt_idx]),
                            candidate=candidates_list[alt_idx],
                            used_ml_model=(field_used_ml and alt_ml_prob is not None),
                            ml_probability=alt_ml_prob,
                        )
                    )

                assignments[field] = Assignment(
                    assignment_type="CANDIDATE",
                    candidate_index=col_idx_val,
                    cost=cost,
                    field=field,
                    candidate=candidates_list[col_idx_val],
                    used_ml_model=field_used_ml and ml_prob is not None,
                    ml_probability=ml_prob,
                    fallback_candidates=tuple(fallbacks),
                )

    return assignments
