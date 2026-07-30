"""candidates.features sub-package — feature computation and pruning."""

from ._classify import _classify_span_bucket
from ._constants import ANCHOR_PROXIMITY_MAX_DISTANCE
from ._diversity import diversity_sampling
from ._extract import extract_features_vectorized
from ._geometry import (
    apply_soft_nms_grid,
    compute_geometry_features_enhanced,
    compute_local_density_grid,
    compute_overlap_iou,
    compute_section_prior,
)
from ._style import compute_style_features_enhanced
from ._text import compute_text_features_enhanced

__all__ = [
    "ANCHOR_PROXIMITY_MAX_DISTANCE",
    "_classify_span_bucket",
    "apply_soft_nms_grid",
    "compute_geometry_features_enhanced",
    "compute_local_density_grid",
    "compute_overlap_iou",
    "compute_section_prior",
    "compute_style_features_enhanced",
    "compute_text_features_enhanced",
    "diversity_sampling",
    "extract_features_vectorized",
]
