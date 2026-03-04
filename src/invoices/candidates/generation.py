"""Core candidate generation pipeline."""

import random
import time
from collections import Counter, defaultdict
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from .. import ingest, io_utils, paths, tokenize
from ..adaptive import learn_anchors
from ..config import Config
from ..exceptions import CandidateGenerationError
from ..logging import get_logger
from ..metrics import pipeline_duration, pipeline_errors
from .constants import (
    ANCHOR_TYPE_DATE,
    ANCHOR_TYPE_ID,
    ANCHOR_TYPE_TOTAL,
    BUCKET_AMOUNT_LIKE,
    BUCKET_DATE_LIKE,
    BUCKET_ID_LIKE,
    BUCKET_KEYWORD_PROXIMAL,
    BUCKET_NAME_LIKE,
    BUCKET_RANDOM_NEGATIVE,
    MONTH_ABBREVS,
    POSITION_FEATURE_SPECS,
)
from .features import (
    apply_soft_nms_grid,
    compute_geometry_features_enhanced,
    compute_local_density_grid,
    compute_section_prior,
    compute_style_features_enhanced,
    compute_text_features_enhanced,
    diversity_sampling,
)
from .patterns import (
    compute_bucket_probabilities,
    is_amount_like_soft,
    is_clean_amount_pattern,
    is_clean_date_pattern,
    is_clean_invoice_pattern,
    is_date_like_soft,
    is_id_like_soft,
    is_name_like_soft,
)
from .proximity import TypedProximityScorer
from .spans import PageGrid, SpanBuilder
from .validation import (
    compute_bootstrap_score,
    filter_candidate_by_bucket,
    get_field_type_for_bucket,
)

logger = get_logger(__name__)


def _expected_parquet_columns() -> frozenset[str]:
    """Raw columns that a fresh candidate parquet must contain.

    Derived from the same constants that extract_features_vectorized reads via
    _fill_scalar_feature.  If any of these are missing from a cached parquet,
    the ML feature extraction layer will silently default them to 0.0, producing
    wrong training data and predictions.

    Naming follows compute_directional_features in proximity.py:
        dx_to_{type}, dy_to_{type}, dist_to_{type},
        aligned_x_{type}, aligned_y_{type},
        reading_order_{type}, below_{type}
    """
    cols: set[str] = set()
    # Directional features — mirror the naming in proximity.py / features.py
    _DIR_PREFIXES = (
        "dx_to",
        "dy_to",
        "dist_to",
        "aligned_x",
        "aligned_y",
        "reading_order",
        "below",
    )
    for anchor_type in Config.ANCHOR_TYPES:
        for prefix in _DIR_PREFIXES:
            cols.add(f"{prefix}_{anchor_type}")
    # Relative-position features
    cols.update(POSITION_FEATURE_SPECS.keys())
    # Doc-level enrichment features
    cols.update({"occurrence_rank", "is_largest_amount_in_doc", "page_frequency"})
    return frozenset(cols)


# Computed once at import time — only changes when code changes (restart).
_EXPECTED_COLUMNS = _expected_parquet_columns()


# Stop words excluded from page-frequency computation
_PAGE_FREQ_STOP_WORDS: frozenset[str] = frozenset(
    {
        "of",
        "the",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "it",
        "a",
        "an",
        "by",
        "with",
        "from",
        "as",
        "your",
        "our",
    }
)


def _enrich_page_frequency(
    candidates_list: list[dict[str, Any]],
    tokens_df: pd.DataFrame,
) -> None:
    """Compute page-frequency feature for each candidate.

    For each candidate, computes the fraction of pages that contain
    each of its content words (excluding stop words), then takes the
    MIN across words. This prevents common words from inflating
    multi-word spans.

    Mutates candidates in place, adding 'page_frequency' key.
    """
    if tokens_df.empty or "page_idx" not in tokens_df.columns:
        for c in candidates_list:
            c["page_frequency"] = 0.0
        return

    total_pages = tokens_df["page_idx"].nunique()
    if total_pages == 0:
        for c in candidates_list:
            c["page_frequency"] = 0.0
        return

    # Build word -> set of pages lookup from tokens
    word_pages: dict[str, set[int]] = {}
    for _, row in tokens_df.iterrows():
        text = str(row.get("text", "")).strip().lower()
        page = int(row["page_idx"])
        if text and text not in _PAGE_FREQ_STOP_WORDS:
            if text not in word_pages:
                word_pages[text] = set()
            word_pages[text].add(page)

    for candidate in candidates_list:
        raw_text = str(candidate.get("raw_text", ""))
        words = [
            w.lower().strip(".,;:'\"")
            for w in raw_text.split()
            if w.lower().strip(".,;:'\"") not in _PAGE_FREQ_STOP_WORDS
        ]
        if not words:
            candidate["page_frequency"] = 0.0
            continue

        # MIN of page fractions across content words
        min_freq = 1.0
        for word in words:
            pages = word_pages.get(word, set())
            freq = len(pages) / total_pages
            if freq < min_freq:
                min_freq = freq
        candidate["page_frequency"] = min_freq


def _enrich_occurrence_rank(candidates_list: list[dict[str, Any]]) -> None:
    """Compute reading-order occurrence rank for same-text candidates.

    For each normalized_text value, ranks candidates by (page_idx, center_y, center_x)
    so occurrence_rank=1 is the first appearance in reading order. This lets the
    ranker prefer header/first occurrences over footer repeats.

    Mutates candidates in place, adding 'occurrence_rank' key.
    """
    # Group candidates by normalized text
    text_groups: dict[str, list[dict[str, Any]]] = {}
    for c in candidates_list:
        key = str(c.get("normalized_text", c.get("raw_text", ""))).strip().lower()
        if key not in text_groups:
            text_groups[key] = []
        text_groups[key].append(c)

    for group in text_groups.values():
        # Sort by reading order: page first, then top-to-bottom, left-to-right
        group.sort(
            key=lambda c: (
                c.get("page_idx", 0),
                (c.get("bbox_norm_y0", 0.0) + c.get("bbox_norm_y1", 0.0)) / 2,
                (c.get("bbox_norm_x0", 0.0) + c.get("bbox_norm_x1", 0.0)) / 2,
            )
        )
        for rank, candidate in enumerate(group, start=1):
            candidate["occurrence_rank"] = rank


def _enrich_largest_amount(candidates_list: list[dict[str, Any]]) -> None:
    """Flag the candidate with the largest numeric value among amount-type candidates.

    Sets 'is_largest_amount_in_doc'=1.0 on the single highest-value amount
    candidate, 0.0 on all others. This is the strongest TotalAmount disambiguator.

    Mutates candidates in place, adding 'is_largest_amount_in_doc' key.
    """
    import re

    _AMOUNT_RE = re.compile(r"[\$€£¥₹₽]?\s*([\d,]+(?:\.\d+)?)")

    def _parse_amount(text: str) -> float | None:
        """Extract the largest numeric value from a text string."""
        text_clean = text.replace(",", "")
        matches = _AMOUNT_RE.findall(text_clean)
        if not matches:
            return None
        try:
            return max(float(m.replace(",", "")) for m in matches)
        except (ValueError, TypeError):
            return None

    # Only consider amount-bucket candidates
    amount_candidates = [
        c for c in candidates_list if c.get("bucket") == BUCKET_AMOUNT_LIKE
    ]

    # Default everyone to 0
    for c in candidates_list:
        c["is_largest_amount_in_doc"] = 0.0

    if not amount_candidates:
        return

    # Find the candidate with the maximum parsed numeric value
    best: dict[str, Any] | None = None
    best_val = float("-inf")
    for c in amount_candidates:
        val = _parse_amount(str(c.get("raw_text", "")))
        if val is not None and val > best_val:
            best_val = val
            best = c

    if best is not None:
        best["is_largest_amount_in_doc"] = 1.0


# =============================================================================
# ERROR HANDLING HELPERS
# =============================================================================


def _log_candidate_generation_failure(sha256: str, error: Exception) -> None:
    """Log a candidate generation failure.

    Args:
        sha256: Document SHA256 hash
        error: The exception that occurred
    """
    reason = error.reason if hasattr(error, "reason") else str(error)
    logger.error(
        "candidate_generation_failed",
        sha256=sha256[:16],
        error_type=type(error).__name__,
        reason=reason,
    )


# =============================================================================
# HELPERS FOR generate_candidates_enhanced (decomposed for readability)
# =============================================================================


def _classify_span_bucket(
    text: str,
    proximity_score: float,
    coverage_stats: dict[str, int],
) -> str | None:
    """Classify a span into a candidate bucket type, or None to skip."""
    # PRIORITY ORDER: ID patterns are most specific, check first
    # to prevent invoice numbers like "US002650-41" from being
    # misclassified as date_like due to the hyphen
    bucket = None

    # 1. Check for clean invoice/ID patterns FIRST (most specific)
    #    BUT: if the token contains a month abbreviation (jan-dec),
    #    it's a date like "01-Jul-24", not an ID like "US002650-41".
    text_lower = text.lower()
    has_month_name = any(m in text_lower for m in MONTH_ABBREVS)
    if (is_clean_invoice_pattern(text) or is_id_like_soft(text)) and not has_month_name:
        bucket = BUCKET_ID_LIKE
        coverage_stats["id_like_spans"] += 1
    # 2. Then check for date patterns
    elif is_date_like_soft(text):
        bucket = BUCKET_DATE_LIKE
        coverage_stats["date_like_spans"] += 1
    # 3. Then check for amount patterns
    elif is_amount_like_soft(text):
        bucket = BUCKET_AMOUNT_LIKE
        coverage_stats["amount_like_spans"] += 1
    # 4. Then check for name patterns (company/person names)
    elif is_name_like_soft(text):
        bucket = BUCKET_NAME_LIKE
        coverage_stats["name_like_spans"] = coverage_stats.get("name_like_spans", 0) + 1
    # 5. Keyword proximity fallback
    elif proximity_score > 0.3:  # Proximal to keywords
        bucket = BUCKET_KEYWORD_PROXIMAL
        coverage_stats["cue_proximal_spans"] += 1
    # 6. Random negatives for training
    elif random.random() < 0.02:  # 2% random negatives
        bucket = BUCKET_RANDOM_NEGATIVE

    return bucket


def _score_candidate(
    span: dict[str, Any],
    geometry_features: dict[str, Any],
    style_features: dict[str, Any],
    directional_features: dict[str, float],
    proximity_score: float,
    section_prior: float,
    bucket: str,
) -> float:
    """Compute total score for a candidate span."""
    text = span["raw_text"]

    # cohesion_score now includes:
    #   - compactness (inverse of span width)
    #   - token count penalty/bonus (prefer 1-3 tokens)
    #   - pattern quality bonus (clean invoice/date/amount patterns)
    #   - repetitive text penalty (garbage detection)
    base_score = (
        span["cohesion_score"]
        * 0.5  # Increased weight since it now has pattern bonuses
        + (1.0 - geometry_features.get("distance_to_center", 1.0)) * 0.2
        + style_features.get("font_size_z", 0) * 0.1
    )

    # Additional pattern-specific bonus computed at scoring time
    # (complements the bonus already in cohesion_score from span creation)
    if (
        is_clean_invoice_pattern(text)
        or is_clean_amount_pattern(text)
        or is_clean_date_pattern(text)
    ):
        base_score += 0.5  # Extra boost for high-quality patterns

    # Bonus for directional alignment (reading order or below anchor)
    alignment_bonus = 0.0
    for anchor_type in [ANCHOR_TYPE_TOTAL, ANCHOR_TYPE_DATE, ANCHOR_TYPE_ID]:
        reading_order = directional_features.get(f"reading_order_{anchor_type}", 0.0)
        below = directional_features.get(f"below_{anchor_type}", 0.0)
        alignment_bonus += (reading_order + below) * 0.05

    total_score = base_score + proximity_score * 0.2 + section_prior + alignment_bonus

    # Apply bootstrap scoring adjustment
    bootstrap_field_type = get_field_type_for_bucket(bucket)
    bootstrap_adjustment = compute_bootstrap_score(text, bootstrap_field_type)
    total_score += bootstrap_adjustment * 0.3

    return float(total_score)


def _build_candidate_record(
    span: dict[str, Any],
    doc_id: str,
    sha256: str,
    page_idx: int,
    bucket: str,
    proximity_score: float,
    section_prior: float,
    total_score: float,
    text_features: dict[str, Any],
    geometry_features: dict[str, Any],
    style_features: dict[str, Any],
    directional_features: dict[str, float],
) -> dict[str, Any]:
    """Assemble a candidate record dict from span and computed features."""
    candidate_id = f"{doc_id}_{page_idx}_{min(span['token_indices'])}"

    # Compute soft bucket probabilities for decoder soft matching
    bucket_probs = compute_bucket_probabilities(span["raw_text"])

    record = {
        "candidate_id": str(candidate_id),  # Ensure string for safety
        "doc_id": str(doc_id),
        "sha256": str(sha256),
        "page_idx": int(page_idx),
        "token_ids": [str(tid) for tid in span["token_ids"]],  # Safe strings
        "token_indices": [int(idx) for idx in span["token_indices"]],  # Safe ints
        "raw_text": str(span["raw_text"]),
        "normalized_text": str(span["normalized_text"]),
        "bucket": str(bucket),
        "token_count": int(span["token_count"]),
        "cohesion_score": float(span["cohesion_score"]),
        "proximity_score": float(proximity_score),
        "section_prior": float(section_prior),
        "total_score": float(total_score),
        # Bounding box
        "bbox_norm_x0": float(span["bbox_norm"][0]),
        "bbox_norm_y0": float(span["bbox_norm"][1]),
        "bbox_norm_x1": float(span["bbox_norm"][2]),
        "bbox_norm_y1": float(span["bbox_norm"][3]),
        "bbox_norm": span["bbox_norm"],  # Keep for processing
        # Features
        **text_features,
        **geometry_features,
        **style_features,
        **directional_features,  # Directional vector features
        "local_density": 0.0,  # Will be updated
        "is_remittance_band": bool(geometry_features.get("center_y", 0) > 0.85),
    }

    # Add soft bucket probability columns (prefixed for decoder consumption)
    for bucket_type, prob in bucket_probs.items():
        record[f"bucket_prob_{bucket_type}"] = float(prob)

    return record


def _process_page_tokens(
    page_idx: int,
    tokens_df: pd.DataFrame,
    doc_id: str,
    sha256: str,
    timings: dict[str, float],
    coverage_stats: dict[str, int],
    learned_anchors: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    """Process all tokens on a single page and return scored candidates."""
    page_tokens = tokens_df[tokens_df["page_idx"] == page_idx].copy()
    if page_tokens.empty:
        return []

    # 1. Span assembly (line-local only)
    spans_start = time.time()
    span_builder = SpanBuilder()
    spans = span_builder.build_spans(page_tokens)
    timings["spans"] += time.time() - spans_start

    if not spans:
        return []

    coverage_stats["total_spans"] += len(spans)

    # 2. Find TYPED cue anchors (semantic differentiation)
    anchors_start = time.time()
    proximity_scorer = TypedProximityScorer()
    proximity_scorer.find_typed_anchors(page_tokens)
    # Merge learned anchors from labeled data (supplements static anchors)
    if learned_anchors:
        proximity_scorer.merge_learned_anchors(learned_anchors, page_tokens)
    timings["anchors"] += time.time() - anchors_start

    # 3. Build page grid for neighbor-only operations
    page_grid = PageGrid()

    # 4. Score spans and build candidates with directional features
    scoring_start = time.time()
    page_font_sizes = page_tokens["font_size"].tolist()

    # Check if page has any anchors at all (for signal filter fallback)
    page_has_anchors = any(
        len(anchors) > 0 for anchors in proximity_scorer.anchors_by_type.values()
    )

    page_candidates: list[dict[str, Any]] = []
    for span in spans:
        # ============================================================
        # SIGNAL-TO-NOISE FILTER (Reality #4)
        # If a span has no spatial relationship to ANY anchor type,
        # it's noise. Prune it early before expensive feature computation.
        #
        # IMPORTANT: Only apply this filter if the page has anchors.
        # Pages without any anchors (e.g., scanned images, unusual layouts)
        # should not have all spans filtered out - this would cause recall
        # regression. In such cases, fall back to keeping all candidates.
        #
        # OPTIMIZATION: Check pattern matches first (cheap string ops)
        # before computing spatial relationships (more expensive).
        # ============================================================
        if page_has_anchors:
            # Check patterns first - these are always kept regardless of position
            text = span["raw_text"]
            is_interesting_pattern = (
                is_date_like_soft(text)
                or is_amount_like_soft(text)
                or is_id_like_soft(text)
                or is_name_like_soft(text)  # Include name patterns
            )

            # Only compute expensive spatial check for non-interesting patterns
            if not is_interesting_pattern:
                if not proximity_scorer.has_any_anchor_relationship(
                    span["bbox_norm"],
                    max_distance=Config.ANCHOR_PROXIMITY_MAX_DISTANCE,
                ):
                    coverage_stats["filtered_no_anchor"] += 1
                    continue

        # Compute all features
        text_features = compute_text_features_enhanced(span["raw_text"])
        geometry_features = compute_geometry_features_enhanced(
            span["bbox_norm"], span["page_width"], span["page_height"]
        )
        style_features = compute_style_features_enhanced(
            span["font_size"],
            span["is_bold"],
            span["is_italic"],
            span["font_hash"],
            page_font_sizes,
        )

        # Directional vector features
        directional_features = proximity_scorer.compute_directional_features(
            span["bbox_norm"]
        )

        # Legacy proximity score (for backwards compatibility)
        proximity_score = proximity_scorer.compute_proximity_score(span["bbox_norm"])

        # Section prior
        section_prior = compute_section_prior(span["bbox_norm"], page_idx)

        # Determine bucket type
        text = span["raw_text"]
        bucket = _classify_span_bucket(text, proximity_score, coverage_stats)
        if bucket is None:
            continue  # Skip if doesn't fit any bucket

        # Garbage candidate filtering (before scoring to save compute)
        if not filter_candidate_by_bucket(text, bucket):
            coverage_stats["filtered_garbage"] = (
                coverage_stats.get("filtered_garbage", 0) + 1
            )
            continue  # Skip garbage candidates

        # Track coverage probes
        if proximity_score > 0.2:  # Low-bar proximity
            coverage_stats["cue_proximal_spans"] += 1

        if section_prior > 0.02:  # Within prior zones
            coverage_stats["region_prior_spans"] += 1

        # Score the candidate
        total_score = _score_candidate(
            span,
            geometry_features,
            style_features,
            directional_features,
            proximity_score,
            section_prior,
            bucket,
        )

        # Build the candidate record
        candidate = _build_candidate_record(
            span,
            doc_id,
            sha256,
            page_idx,
            bucket,
            proximity_score,
            section_prior,
            total_score,
            text_features,
            geometry_features,
            style_features,
            directional_features,
        )

        page_candidates.append(candidate)

        # Add to grid for density computation
        cx = (span["bbox_norm"][0] + span["bbox_norm"][2]) / 2
        cy = (span["bbox_norm"][1] + span["bbox_norm"][3]) / 2
        page_grid.add_item(cx, cy, candidate)

    timings["scoring"] += time.time() - scoring_start

    # 5. Update local density using grid
    for candidate in page_candidates:
        candidate["local_density"] = compute_local_density_grid(
            candidate["bbox_norm"], page_grid
        )

    # 6. Proportional soft trim if too many candidates on this page
    if len(page_candidates) > 300:  # Per-page soft limit
        page_candidates.sort(key=lambda x: x["total_score"], reverse=True)
        keep_count = min(300, int(len(page_candidates) * 0.8))  # Keep top 80%
        page_candidates = page_candidates[:keep_count]

    # 7. Soft-NMS
    soft_nms_start = time.time()
    page_candidates = apply_soft_nms_grid(page_candidates, page_grid)
    timings["soft_nms"] += time.time() - soft_nms_start

    return page_candidates


def _deduplicate_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Global deduplication using normalized text + rounded bbox center."""
    seen_keys: set[str] = set()
    deduped: list[dict[str, Any]] = []

    for candidate in candidates:
        # Create dedup key: normalized text + rounded bbox center
        center_x = (candidate["bbox_norm_x0"] + candidate["bbox_norm_x1"]) / 2
        center_y = (candidate["bbox_norm_y0"] + candidate["bbox_norm_y1"]) / 2

        # Round to nearest 0.05 for minor geometry tolerance
        rounded_x = round(center_x / 0.05) * 0.05
        rounded_y = round(center_y / 0.05) * 0.05

        dedup_key = f"{candidate['page_idx']}_{candidate['normalized_text']}_{rounded_x:.2f}_{rounded_y:.2f}"

        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            deduped.append(candidate)

    return deduped


def _finalize_candidates(
    all_candidates: list[dict[str, Any]],
    candidates_path: Any,
    doc_id: str,
    timings: dict[str, float],
    coverage_stats: dict[str, int],
    start_time: float,
) -> int:
    """Apply diversity sampling, clean up, save to parquet, and log results."""
    # Diversity sampling
    diversity_start = time.time()
    all_candidates = diversity_sampling(all_candidates, max_candidates=200)
    timings["diversity"] += time.time() - diversity_start

    # Update bucket counts for reporting
    bucket_counts: dict[str, int] = {
        BUCKET_DATE_LIKE: 0,
        BUCKET_AMOUNT_LIKE: 0,
        BUCKET_ID_LIKE: 0,
        BUCKET_NAME_LIKE: 0,
        BUCKET_KEYWORD_PROXIMAL: 0,
        BUCKET_RANDOM_NEGATIVE: 0,
    }
    for candidate in all_candidates:
        bucket_counts[candidate["bucket"]] += 1

    # Clean up bbox_norm for storage (not needed in final output)
    for candidate in all_candidates:
        if "bbox_norm" in candidate:
            del candidate["bbox_norm"]

    # Save candidates
    if all_candidates:
        df = pd.DataFrame(all_candidates)
        io_utils.write_parquet_safe(df, candidates_path)

        timings["total"] = time.time() - start_time

        logger.info(
            "candidates_generated",
            doc_id=doc_id,
            count=len(all_candidates),
            cue_proximal=coverage_stats["cue_proximal_spans"],
            total_spans=coverage_stats["total_spans"],
            region_prior=coverage_stats["region_prior_spans"],
            filtered_no_anchor=coverage_stats["filtered_no_anchor"],
            filtered_garbage=coverage_stats["filtered_garbage"],
        )
        logger.debug(
            "candidates_bucket_counts",
            doc_id=doc_id,
            **{b: c for b, c in bucket_counts.items() if c > 0},
        )
        logger.debug(
            "candidates_timing",
            doc_id=doc_id,
            spans_s=round(timings["spans"], 3),
            scoring_s=round(timings["scoring"], 3),
            soft_nms_s=round(timings["soft_nms"], 3),
            total_s=round(timings["total"], 3),
        )

    return len(all_candidates)


def generate_candidates_enhanced(sha256: str) -> tuple[int, dict[str, Any]]:
    """Enhanced candidate generation with spans, proximity, soft-NMS, and diversity."""
    start_time_metrics = time.perf_counter()
    try:
        # Timing dictionary for per-page performance
        timings = {
            "spans": 0.0,
            "anchors": 0.0,
            "scoring": 0.0,
            "soft_nms": 0.0,
            "dedupe": 0.0,
            "diversity": 0.0,
            "total": 0.0,
        }

        start_time = time.time()

        # Check if already exists (idempotency)
        candidates_path = paths.get_candidates_path(sha256)
        existing_df = io_utils.read_parquet_safe(candidates_path, on_error="empty")
        if existing_df is not None:
            # Validate cached columns match expected raw feature set
            missing = _EXPECTED_COLUMNS - set(existing_df.columns)
            if missing:
                logger.warning(
                    "stale_candidate_cache_detected",
                    sha256=sha256[:16],
                    missing_columns=sorted(missing),
                    count=len(missing),
                )
                candidates_path.unlink()
            else:
                logger.debug(
                    "candidates_already_exist",
                    sha256=sha256[:16],
                    count=len(existing_df),
                )
                return len(existing_df), timings

        # Get tokens
        tokens_df = tokenize.get_document_tokens(sha256)
        if tokens_df.empty:
            logger.warning("no_tokens_for_candidates", sha256=sha256[:16])
            return 0, timings

        doc_info = ingest.get_document_info(sha256)
        if not doc_info:
            raise ValueError(f"Document not found: {sha256}")

        doc_id = doc_info["doc_id"]

        # Set random seed for deterministic behavior
        seed_value = int(sha256[:8], 16)
        random.seed(seed_value)
        # Note: np.random.default_rng() preferred over np.random.seed() but not used in this function
        # Keeping legacy random.seed() for backward compatibility with existing test outputs

        # Coverage probe counters
        coverage_stats: dict[str, int] = {
            "total_spans": 0,
            "cue_proximal_spans": 0,
            "region_prior_spans": 0,
            "date_like_spans": 0,
            "amount_like_spans": 0,
            "id_like_spans": 0,
            "name_like_spans": 0,
            "filtered_no_anchor": 0,
            "filtered_garbage": 0,
        }

        # Learn anchor keywords from labeled data (if >= 3 labeled docs exist)
        from ..labels import load_aligned_labels

        aligned = load_aligned_labels()
        n_labeled_docs = (
            aligned["sha256"].nunique()
            if not aligned.empty and "sha256" in aligned.columns
            else 0
        )
        learned_anchors = learn_anchors() if n_labeled_docs >= 3 else {}

        # Process each page independently
        all_candidates: list[dict[str, Any]] = []
        for page_idx in sorted(tokens_df["page_idx"].unique()):
            page_candidates = _process_page_tokens(
                page_idx,
                tokens_df,
                doc_id,
                sha256,
                timings,
                coverage_stats,
                learned_anchors=learned_anchors,
            )

            all_candidates.extend(page_candidates)

        if not all_candidates:
            return 0, timings

        # Global deduplication
        dedupe_start = time.time()
        all_candidates = _deduplicate_candidates(all_candidates)
        timings["dedupe"] += time.time() - dedupe_start

        # Enrich with page-frequency feature
        _enrich_page_frequency(all_candidates, tokens_df)

        # Enrich with doc-level features (require all candidates to be present)
        _enrich_occurrence_rank(all_candidates)
        _enrich_largest_amount(all_candidates)

        # Finalize: diversity sampling, cleanup, save, log
        count = _finalize_candidates(
            all_candidates, candidates_path, doc_id, timings, coverage_stats, start_time
        )

        return count, timings
    except Exception:
        pipeline_errors.labels(stage="candidates").inc()
        raise
    finally:
        duration_metrics = time.perf_counter() - start_time_metrics
        pipeline_duration.labels(stage="candidates").observe(duration_metrics)


def generate_candidates(sha256: str) -> int:
    """Legacy interface for compatibility."""
    count, _ = generate_candidates_enhanced(sha256)
    return count


def _invalidate_stale_caches(sha256_list: list[str]) -> int:
    """Delete all candidate parquets missing expected raw feature columns.

    Runs BEFORE per-document generation so cross-document features
    (like learn_anchors) never see a mix of stale and fresh caches.

    Returns the number of stale files removed.
    """
    removed = 0

    for sha256 in sha256_list:
        candidates_path = paths.get_candidates_path(sha256)
        if not candidates_path.exists():
            continue
        df = io_utils.read_parquet_safe(candidates_path, on_error="empty")
        if df is None:
            continue
        missing = _EXPECTED_COLUMNS - set(df.columns)
        if missing:
            logger.warning(
                "stale_candidate_cache_detected",
                sha256=sha256[:16],
                missing_columns=sorted(missing),
                count=len(missing),
            )
            candidates_path.unlink()
            removed += 1

    return removed


def generate_all_candidates() -> dict[str, int]:
    """Generate candidates for all documents."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        logger.info("no_documents_in_index")
        return {}

    # Bulk-invalidate stale caches before any regeneration so cross-document
    # features (learn_anchors) never see a mix of stale and fresh data.
    all_sha256s = indexed_docs["sha256"].tolist()
    stale_count = _invalidate_stale_caches(all_sha256s)
    if stale_count:
        logger.info("stale_caches_invalidated", count=stale_count)

    results = {}
    all_timings: defaultdict[str, list[float]] = defaultdict(list)

    logger.info("generating_all_candidates", doc_count=len(indexed_docs))

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]

        try:
            candidate_count, timings = generate_candidates_enhanced(sha256)
            results[sha256] = candidate_count

            # Collect timing statistics
            for phase, duration in timings.items():
                all_timings[phase].append(duration)

        except CandidateGenerationError as e:
            # Typed candidate generation error
            _log_candidate_generation_failure(sha256, e)
            results[sha256] = 0
        except (ValueError, TypeError, KeyError) as e:
            # Data processing errors
            _log_candidate_generation_failure(sha256, e)
            results[sha256] = 0
        except OSError as e:
            # File system errors (token file not readable)
            _log_candidate_generation_failure(sha256, e)
            results[sha256] = 0

    # Log timing summary
    if all_timings:
        for phase, durations in all_timings.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                logger.info(
                    "candidates_timing_summary",
                    phase=phase,
                    avg_s=round(avg_duration, 3),
                    max_s=round(max_duration, 3),
                )

    return results


def get_document_candidates(sha256: str) -> pd.DataFrame:
    """Get candidates for a specific document."""
    candidates_path = paths.get_candidates_path(sha256)
    df = io_utils.read_parquet_safe(candidates_path, on_error="empty")
    return df if df is not None else pd.DataFrame()


def get_coverage_statistics() -> dict[str, Any]:
    """Get coverage statistics across all candidates."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        return {}

    bucket_dist: Counter[Any] = Counter()
    total_stats: dict[str, Any] = {
        "total_documents": len(indexed_docs),
        "documents_with_candidates": 0,
        "total_candidates": 0,
        "bucket_distribution": bucket_dist,
        "coverage_by_field": {},
    }

    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info["sha256"]
        candidates_df = get_document_candidates(sha256)

        if not candidates_df.empty:
            total_stats["documents_with_candidates"] += 1
            total_stats["total_candidates"] += len(candidates_df)

            # Bucket distribution
            for bucket in candidates_df["bucket"]:
                bucket_dist[bucket] += 1

    return total_stats
