"""Decoder scoring weights — frozen dataclass with optional tuned overrides."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderWeights:
    """Named constants for decoder scoring — not env-configurable."""

    # Header region
    HEADER_REGION_BONUS: float = 0.8

    HEADER_TOP_CLOSE_BONUS: float = 0.5
    HEADER_TOP_MID_BONUS: float = 0.3
    HEADER_TOP_CLOSE_THRESHOLD: float = 0.3
    HEADER_TOP_MID_THRESHOLD: float = 0.5

    # Bucket affinity
    BUCKET_MATCH_BONUS: float = 0.8
    BUCKET_MISMATCH_STRONG: float = -0.5
    BUCKET_MISMATCH_MODERATE: float = -0.3
    BUCKET_MISMATCH_MILD: float = -0.4
    BUCKET_KEYWORD_PROXIMAL_BONUS: float = 0.2

    # Directional (anchor-based)
    DIRECTIONAL_BELOW_CLOSE_BONUS: float = 0.4
    DIRECTIONAL_BELOW_CLOSE_THRESHOLD: float = 0.3
    DIRECTIONAL_READING_ORDER_BONUS: float = 0.3
    DIRECTIONAL_READING_ORDER_THRESHOLD: float = 0.2
    DIRECTIONAL_SAME_ROW_BONUS: float = 0.3
    DIRECTIONAL_SAME_ROW_THRESHOLD: float = 0.2
    DIRECTIONAL_FAR_PENALTY: float = -0.2
    DIRECTIONAL_FAR_THRESHOLD: float = 0.5

    # Footer region
    FOOTER_REGION_BONUS: float = 0.05
    FOOTER_Y_THRESHOLD: float = 0.25

    # Text pattern validation
    EMPTY_TEXT_PENALTY: float = -0.5
    INV_PREFIX_WRONG_FIELD_PENALTY: float = -1.5
    TEXT_PATTERN_NEGATIVE_AMPLIFIER: float = 1.5
    ENTITY_LABEL_MATCH_BONUS: float = 0.4

    # Cost combination weights
    PROXIMITY_WEIGHT: float = 0.15
    SECTION_PRIOR_WEIGHT: float = 0.1
    COHESION_WEIGHT: float = 0.1
    COHESION_NORMALIZER: float = 100.0
    DIRECTIONAL_DAMPENING: float = 0.5


DWeights = DecoderWeights()
