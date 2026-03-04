"""Candidate generation module for balanced span proposals with feature extraction.

This package was decomposed from a single candidates.py module into submodules:
- constants: Bucket types, anchor types, stopwords, feature specs
- validation: Garbage filtering, bucket validation, bootstrap priors
- patterns: Pattern classification (is_*_soft) and quality scoring
- spans: PageGrid and SpanBuilder classes
- proximity: TypedAnchor and TypedProximityScorer
- features: Feature computation, extraction, pruning
- generation: Core pipeline (generate_candidates, etc.)
"""

from .constants import (
    ANCHOR_TYPE_DATE,
    ANCHOR_TYPE_ID,
    ANCHOR_TYPE_NAME,
    ANCHOR_TYPE_TAX,
    ANCHOR_TYPE_TOTAL,
    BASE_FEATURE_NAMES,
    BUCKET_AMOUNT_LIKE,
    BUCKET_DATE_LIKE,
    BUCKET_ID_LIKE,
    BUCKET_KEYWORD_PROXIMAL,
    BUCKET_NAME_LIKE,
    BUCKET_RANDOM_NEGATIVE,
    COMMON_NON_NAMES,
    CURRENCY_CODES,
    CURRENCY_SYMBOLS,
    DATE_ANCHORS,
    DIRECTIONAL_DEFAULTS,
    ID_ANCHORS,
    INVOICE_KEYWORDS,
    MIN_LENGTH_BY_BUCKET,
    MONTH_ABBREVS,
    NAME_ANCHORS,
    POSITION_FEATURE_SPECS,
    STOPWORDS,
    TAX_ANCHORS,
    TOTAL_ANCHORS,
)
from .features import (
    apply_soft_nms_grid,
    compute_geometry_features_enhanced,
    compute_local_density_grid,
    compute_overlap_iou,
    compute_section_prior,
    compute_style_features_enhanced,
    compute_text_features_enhanced,
    diversity_sampling,
    extract_features_vectorized,
    prune_candidates,
)
from .generation import (
    generate_all_candidates,
    generate_candidates,
    generate_candidates_enhanced,
    get_coverage_statistics,
    get_document_candidates,
)
from .patterns import (
    compute_bucket_probabilities,
    compute_pattern_score_bonus,
    compute_token_count_penalty,
    has_repetitive_text,
    is_amount_like_soft,
    is_clean_amount_pattern,
    is_clean_date_pattern,
    is_clean_invoice_pattern,
    is_date_like_soft,
    is_id_like_soft,
    is_name_like_soft,
    is_single_clean_token,
    normalize_text_for_dedup,
)
from .proximity import TypedAnchor, TypedProximityScorer
from .spans import PageGrid, SpanBuilder
from .validation import (
    compute_bootstrap_score,
    filter_candidate_by_bucket,
    get_field_type_for_bucket,
    is_garbage_candidate,
    is_valid_amount_candidate,
    is_valid_date_candidate,
    is_valid_id_candidate,
    is_valid_name_candidate,
    passes_bootstrap_amount_filter,
    passes_bootstrap_invoice_number_filter,
    passes_bootstrap_name_filter,
)

__all__ = [
    # Constants
    "ANCHOR_TYPE_DATE",
    "ANCHOR_TYPE_ID",
    "ANCHOR_TYPE_NAME",
    "ANCHOR_TYPE_TAX",
    "ANCHOR_TYPE_TOTAL",
    "BASE_FEATURE_NAMES",
    "BUCKET_AMOUNT_LIKE",
    "BUCKET_DATE_LIKE",
    "BUCKET_ID_LIKE",
    "BUCKET_KEYWORD_PROXIMAL",
    "BUCKET_NAME_LIKE",
    "BUCKET_RANDOM_NEGATIVE",
    "COMMON_NON_NAMES",
    "CURRENCY_CODES",
    "CURRENCY_SYMBOLS",
    "DATE_ANCHORS",
    "DIRECTIONAL_DEFAULTS",
    "ID_ANCHORS",
    "INVOICE_KEYWORDS",
    "MIN_LENGTH_BY_BUCKET",
    "MONTH_ABBREVS",
    "NAME_ANCHORS",
    "POSITION_FEATURE_SPECS",
    "STOPWORDS",
    "TAX_ANCHORS",
    "TOTAL_ANCHORS",
    # Validation
    "compute_bootstrap_score",
    "filter_candidate_by_bucket",
    "get_field_type_for_bucket",
    "is_garbage_candidate",
    "is_valid_amount_candidate",
    "is_valid_date_candidate",
    "is_valid_id_candidate",
    "is_valid_name_candidate",
    "passes_bootstrap_amount_filter",
    "passes_bootstrap_invoice_number_filter",
    "passes_bootstrap_name_filter",
    # Patterns
    "compute_bucket_probabilities",
    "compute_pattern_score_bonus",
    "compute_token_count_penalty",
    "has_repetitive_text",
    "is_amount_like_soft",
    "is_clean_amount_pattern",
    "is_clean_date_pattern",
    "is_clean_invoice_pattern",
    "is_date_like_soft",
    "is_id_like_soft",
    "is_name_like_soft",
    "is_single_clean_token",
    "normalize_text_for_dedup",
    # Spans
    "PageGrid",
    "SpanBuilder",
    # Proximity
    "TypedAnchor",
    "TypedProximityScorer",
    # Features
    "apply_soft_nms_grid",
    "compute_geometry_features_enhanced",
    "compute_local_density_grid",
    "compute_overlap_iou",
    "compute_section_prior",
    "compute_style_features_enhanced",
    "compute_text_features_enhanced",
    "diversity_sampling",
    "extract_features_vectorized",
    "prune_candidates",
    # Generation
    "generate_all_candidates",
    "generate_candidates",
    "generate_candidates_enhanced",
    "get_coverage_statistics",
    "get_document_candidates",
]
