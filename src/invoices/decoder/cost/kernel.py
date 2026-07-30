"""Kernel cost functions: text pattern validation and schema-driven cost computation."""

from __future__ import annotations

from typing import Any

from ...constants import (
    FIELD_TYPE_BUCKET_MATCHES,
    FIELD_TYPE_BUCKET_MISS_MODERATE,
    FIELD_TYPE_BUCKET_MISS_NEUTRAL,
    FIELD_TYPE_BUCKET_MISS_STRONG,
    FIELD_TYPES_WITH_TEXT_BONUS,
    NORMALIZER_TO_ENTITY_LABEL,
)
from ...schema import FieldSpec
from .. import weights as _w_mod
from ..weights import DecoderWeights
from ._helpers import _compute_directional_bonus_for_anchor, _header_region_bonus


def _bucket_probability_total(
    sample: dict[str, Any], bucket_labels: frozenset[str]
) -> float:
    return float(
        sum(
            sample.get(f"bucket_prob_{bucket_label}", 0.0)
            for bucket_label in bucket_labels
        )
    )


def compute_text_pattern_bonus(
    field_spec: FieldSpec,
    sample: dict[str, Any],
    *,
    weights: DecoderWeights | None = None,
) -> float:
    """Return entity-label match bonus for field-candidate pair.

    Reads field_spec.normalizer, looks up NORMALIZER_TO_ENTITY_LABEL, and checks
    whether the candidate's pre-computed ent_label (set by views.candidates_df)
    matches the expected OntoNotes entity set.  If normalizer is absent from
    the mapping, returns 0.0 (no bonus, no penalty).
    """
    w = weights or _w_mod.DWeights
    normalizer = field_spec.normalizer
    labels = NORMALIZER_TO_ENTITY_LABEL.get(normalizer)
    if labels is None:
        return 0.0

    ent_label: str | None = sample.get("ent_label")
    if ent_label in labels:
        return w.ENTITY_LABEL_MATCH_BONUS
    return 0.0


def _compute_schema_driven_costs(
    field: str,
    sample: dict[str, Any],
    spec: FieldSpec,
    w: DecoderWeights | None = None,
) -> tuple[float, float]:
    if w is None:
        w = _w_mod.DWeights
    field_type = spec.normalizer
    bucket = sample.get("bucket", "")
    bucket_labels = FIELD_TYPE_BUCKET_MATCHES.get(field_type, frozenset())
    bucket_bonus = 0.0

    if bucket in bucket_labels:
        bucket_bonus = w.BUCKET_MATCH_BONUS
    elif bucket_labels:
        soft_prob_sum = _bucket_probability_total(sample, bucket_labels)
        if bucket in FIELD_TYPE_BUCKET_MISS_NEUTRAL.get(field_type, frozenset()):
            bucket_bonus = 0.0
        elif bucket in FIELD_TYPE_BUCKET_MISS_STRONG.get(field_type, frozenset()):
            base_penalty = w.BUCKET_MISMATCH_STRONG
            bucket_bonus = base_penalty * 0.5 if soft_prob_sum > 0.3 else base_penalty
        elif bucket in FIELD_TYPE_BUCKET_MISS_MODERATE.get(field_type, frozenset()):
            base_penalty = w.BUCKET_MISMATCH_MODERATE
            bucket_bonus = base_penalty * 0.5 if soft_prob_sum > 0.3 else base_penalty
        else:
            base_penalty = w.BUCKET_MISMATCH_MILD
            bucket_bonus = base_penalty * 0.5 if soft_prob_sum > 0.3 else base_penalty

    directional_bonus = 0.0
    if spec.is_header:
        directional_bonus += _header_region_bonus(sample, w=w)
    else:
        anchor_family = spec.anchor_family
        if anchor_family:
            directional_bonus = _compute_directional_bonus_for_anchor(
                sample, anchor_family, w=w
            )
            if field_type in FIELD_TYPES_WITH_TEXT_BONUS:
                anchor_distance = sample.get(f"dist_to_{anchor_family}", 1.0)
                anchor_below = sample.get(f"below_{anchor_family}", 0.0)
                anchor_order = sample.get(f"reading_order_{anchor_family}", 0.0)
                bonus = spec.anchor_bonus_override
                if bonus:
                    if (
                        anchor_below > 0
                        and anchor_distance < bonus.below_dist_threshold
                    ):
                        directional_bonus += bonus.below_bonus
                    if (
                        anchor_order > 0
                        and anchor_distance < bonus.reading_order_dist_threshold
                    ):
                        directional_bonus += bonus.reading_order_bonus

    return bucket_bonus, directional_bonus
