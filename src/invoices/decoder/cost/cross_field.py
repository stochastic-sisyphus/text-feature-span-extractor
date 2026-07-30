"""Cross-field consistency adjustments applied to the cost matrix before Hungarian."""

from __future__ import annotations

from typing import Any

import numpy as np

from ... import schema as registry
from ...schema import build_field_spec
from ._helpers import _parse_amount_value


def apply_cross_field_adjustments(
    cost_matrix: np.ndarray,
    candidates_list: list[dict[str, Any]],
    schema_fields: list[str],
) -> None:
    """Adjust cost matrix to reward cross-field consistency BEFORE Hungarian.

    Checks if candidate combinations satisfy semantic relationships:
    - Subtotal + TaxAmount ~ TotalAmount (within tolerance)
    - Subtotal <= TotalAmount

    When consistent candidate sets are found, their costs are reduced so
    the Hungarian algorithm favors internally-consistent assignments.

    Mutates cost_matrix in place.

    Args:
        cost_matrix: Shape (n_fields, n_candidates + n_fields). Modified in place.
        candidates_list: List of candidate dicts with raw_text.
        schema_fields: Ordered list of field names matching cost_matrix rows.
    """
    if not candidates_list:
        return

    # Find decimal-typed (amount) fields from schema (no hardcoded names)
    amount_fields = {
        name
        for name in registry.get_field_names()
        if build_field_spec(name).base_type == "decimal"
    }
    if not amount_fields:
        return

    # Build field index lookup for the fields present in schema_fields
    field_to_idx: dict[str, int] = {
        fname: idx for idx, fname in enumerate(schema_fields) if fname in amount_fields
    }

    # We need at least 2 amount fields to do cross-field checks
    if len(field_to_idx) < 2:
        return

    # Get field definitions to identify relationships
    # Look for the "total" field (highest importance among amounts)
    # and fields that should sum to it
    total_field: str | None = None
    component_fields: list[str] = []

    for fname in field_to_idx:
        spec = build_field_spec(fname)
        # The field with highest importance or priority_bonus is likely the total
        if spec.priority_bonus > 0 or spec.importance >= 0.9:
            total_field = fname
        else:
            component_fields.append(fname)

    if total_field is None or not component_fields:
        return

    n_candidates = len(candidates_list)
    total_idx = field_to_idx[total_field]

    # Pre-parse all candidate amounts once
    parsed_amounts: list[float | None] = []
    for cand in candidates_list:
        text = str(cand.get("raw_text") or cand.get("text", ""))
        parsed_amounts.append(_parse_amount_value(text))

    # Cross-field consistency bonus: -0.3 cost reduction, applied at most ONCE
    # per (field_idx, candidate_idx) cell to prevent runaway accumulation.
    consistency_bonus = 0.3
    tolerance = 0.05  # 5% relative tolerance

    # Track which cells have already received a consistency bonus
    adjusted_cells: set[tuple[int, int]] = set()

    # For each candidate pair (total_candidate, component_candidate),
    # check if any subset of components sums to the total
    for total_cand_idx in range(n_candidates):
        total_val = parsed_amounts[total_cand_idx]
        if total_val is None or total_val <= 0:
            continue

        for comp_field in component_fields:
            comp_idx = field_to_idx[comp_field]
            for comp_cand_idx in range(n_candidates):
                if comp_cand_idx == total_cand_idx:
                    continue  # Same candidate can't be both total and component
                comp_val = parsed_amounts[comp_cand_idx]
                if comp_val is None or comp_val <= 0:
                    continue

                # Check: component <= total (basic sanity)
                if comp_val <= total_val * (1.0 + tolerance):
                    # Check: can we find a complementary component?
                    remainder = total_val - comp_val
                    if remainder < 0:
                        continue

                    # Look for another component that fills the gap
                    for other_field in component_fields:
                        if other_field == comp_field:
                            continue
                        other_idx = field_to_idx[other_field]
                        for other_cand_idx in range(n_candidates):
                            if other_cand_idx in (total_cand_idx, comp_cand_idx):
                                continue
                            other_val = parsed_amounts[other_cand_idx]
                            if other_val is None:
                                continue

                            # Check: comp + other ~ total (total_val > 0 guaranteed above)
                            combined = comp_val + other_val
                            if abs(combined - total_val) / total_val <= tolerance:
                                # Consistent triple found! Reduce costs
                                # (at most once per cell).
                                for cell in [
                                    (total_idx, total_cand_idx),
                                    (comp_idx, comp_cand_idx),
                                    (other_idx, other_cand_idx),
                                ]:
                                    if cell not in adjusted_cells:
                                        adjusted_cells.add(cell)
                                        cost_matrix[cell[0], cell[1]] -= (
                                            consistency_bonus
                                        )
