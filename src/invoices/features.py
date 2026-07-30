"""Canonical ML feature vector definition.

The CandidateFeatures dataclass is the single source of truth for the 69-dim
feature vector used by XGBoost.  Field order is significant (XGBoost is
order-sensitive) and is locked by the dataclass field declaration order.

Everything else — FEATURE_COLUMNS, feature_columns(), default dicts — is
derived from the dataclass so the definition lives in exactly one place.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

# ---------------------------------------------------------------------------
# Building-block constants (canonical source — Config duplicates removed)
# ---------------------------------------------------------------------------

ANCHOR_TYPES: tuple[str, ...] = ("total", "tax", "date", "id", "name")

DIRECTIONAL_SUFFIXES: tuple[str, ...] = (
    "dx_to_",
    "dy_to_",
    "dist_to_",
    "aligned_x_",
    "aligned_y_",
    "reading_order_",
    "below_",
)

RELATIVE_POSITION_FEATURES: tuple[str, ...] = (
    "y_from_bottom",
    "in_top_half",
    "in_bottom_quarter",
    "in_top_quarter",
    "in_right_third",
    "in_amount_region",
)

# Default values for position features (used by candidate generation)
POSITION_FEATURE_SPECS: dict[str, float] = {
    "y_from_bottom": 0.5,
    "in_top_half": 0.0,
    "in_bottom_quarter": 0.0,
    "in_top_quarter": 0.0,
    "in_right_third": 0.0,
    "in_amount_region": 0.0,
}

# Default values for directional features per anchor type template
DIRECTIONAL_DEFAULTS: dict[str, float] = {
    "dx": 1.0,
    "dy": 1.0,
    "dist": 1.414,  # sqrt(2) diagonal
    "aligned_x": 0.0,
    "aligned_y": 0.0,
    "reading_order": 0.0,
    "below": 0.0,
}


# ---------------------------------------------------------------------------
# Frozen dataclass — the contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateFeatures:
    """69-dim feature vector for a single candidate span.

    Construction fails with TypeError if any field is omitted.
    Assignment fails with FrozenInstanceError (immutable).
    Field order defines the canonical column order for XGBoost.

    Groups:
      Geometric (5) + Text (4) + Page (1) + Bucket one-hot (7)
      + Directional (35 = 7 suffixes x 5 anchor types)
      + Anchor presence (5)
      + Relative position (6) + Semantic (6) = 69
    """

    # Geometric (5)
    center_x: float
    center_y: float
    width: float
    height: float
    area: float

    # Text (4)
    char_count: float
    word_count: float
    digit_count: float
    alpha_count: float

    # Page (1)
    page_idx: float

    # Bucket one-hot (7)
    bucket_amount_like: float
    bucket_date_like: float
    bucket_id_like: float
    bucket_name_like: float
    bucket_keyword_proximal: float
    bucket_random_negative: float
    bucket_other: float

    # Directional — total (7)
    dx_to_total: float
    dy_to_total: float
    dist_to_total: float
    aligned_x_total: float
    aligned_y_total: float
    reading_order_total: float
    below_total: float

    # Directional — tax (7)
    dx_to_tax: float
    dy_to_tax: float
    dist_to_tax: float
    aligned_x_tax: float
    aligned_y_tax: float
    reading_order_tax: float
    below_tax: float

    # Directional — date (7)
    dx_to_date: float
    dy_to_date: float
    dist_to_date: float
    aligned_x_date: float
    aligned_y_date: float
    reading_order_date: float
    below_date: float

    # Directional — id (7)
    dx_to_id: float
    dy_to_id: float
    dist_to_id: float
    aligned_x_id: float
    aligned_y_id: float
    reading_order_id: float
    below_id: float

    # Directional — name (7)
    dx_to_name: float
    dy_to_name: float
    dist_to_name: float
    aligned_x_name: float
    aligned_y_name: float
    reading_order_name: float
    below_name: float

    # Anchor presence (5)
    has_total_anchor: float
    has_tax_anchor: float
    has_date_anchor: float
    has_id_anchor: float
    has_name_anchor: float

    # Relative position (6)
    y_from_bottom: float
    in_top_half: float
    in_bottom_quarter: float
    in_top_quarter: float
    in_right_third: float
    in_amount_region: float

    # Semantic (6)
    in_header_region: float
    in_footer_region: float
    in_body_region: float
    occurrence_rank: float
    is_largest_amount_in_doc: float
    aligned_x: float


# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: tuple[str, ...] = tuple(f.name for f in fields(CandidateFeatures))
"""Canonical ordered tuple of all 69 feature names, derived from the dataclass."""


# ---------------------------------------------------------------------------
# Content-addressed contract — the hash IS the contract
# ---------------------------------------------------------------------------
#
# Count (69) is emergent. The contract is the ordered derivation itself,
# collapsed to a stable sha256 hex digest over the pipe-joined column names.
# Any reorder/add/remove/rename produces a different hash — loaded rankers
# trained against an older derivation will refuse to bind at manifest-check
# time (see manifest.ModelManifest.check_feature_compatibility).
#
# Stability guarantees:
#   - dataclass field order is the single source of truth (fields() is ordered)
#   - "|".join over an explicitly ordered tuple (no set/dict iteration)
#   - sha256 hex digest is deterministic across Python runs/platforms
FEATURE_SCHEMA_HASH: str = hashlib.sha256(
    "|".join(FEATURE_COLUMNS).encode("utf-8")
).hexdigest()
"""Stable content-address of the ordered feature derivation."""


def feature_columns() -> list[str]:
    """Return FEATURE_COLUMNS as a list (backward compat with list consumers)."""
    return list(FEATURE_COLUMNS)
