"""Contract emission module for JSON output and review queue management."""

import json
import re
import time
import warnings
from datetime import datetime
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
from tqdm import tqdm  # type: ignore[import-untyped]

from . import decoder, ingest, io_utils, normalize, paths, tokenize, utils
from . import schema_registry as registry
from .config import Config
from .exceptions import EmissionError
from .logging import get_logger
from .metrics import (
    docs_auto_approved,
    docs_needs_review,
    documents_processed,
    pipeline_duration,
    pipeline_errors,
)
from .validation import validate_assignment_semantics

logger = get_logger(__name__)

# Common date formats found on invoices
_DATE_FORMATS = [
    "%Y-%m-%d",  # ISO format (already works, but try first)
    "%d-%b-%y",  # 01-Jul-24
    "%d-%b-%Y",  # 01-Jul-2024
    "%b %d, %Y",  # Oct 20, 2023
    "%b %d %Y",  # Oct 20 2023
    "%d %b %Y",  # 20 Oct 2023
    "%b %d",  # Jul 19 (assume current year)
    "%d/%m/%Y",  # 20/10/2023 (DD/MM/YYYY)
    "%m/%d/%Y",  # 10/20/2023 (MM/DD/YYYY)
    "%Y/%m/%d",  # 2023/10/20
    "%d.%m.%Y",  # 01.07.2024
    "%Y %b",  # 2023 Oct
    "%b %Y",  # Oct 2023
    "%Y-%m",  # 2023-10
]


def normalize_date(value: str) -> str | None:
    """Try to normalize a date string to ISO format (YYYY-MM-DD).

    Tries multiple common invoice date formats before giving up.

    Args:
        value: Raw date string from invoice

    Returns:
        ISO formatted date string (YYYY-MM-DD) or None if unparseable
    """
    if not value or not value.strip():
        return None

    clean_value = value.strip()

    # Handle incomplete dates like "Jul 19 -" (missing year) - common OCR artifact
    # Remove trailing dashes and try to parse
    clean_value = re.sub(r"[\s\-]+$", "", clean_value)

    # If still empty after cleaning, give up
    if not clean_value:
        return None

    # Try each format
    for fmt in _DATE_FORMATS:
        try:
            # Suppress Python 3.15 deprecation warning for day-of-month without year
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                parsed = datetime.strptime(clean_value, fmt)

            # For 2-digit years, assume 2000s if <= 50, else 1900s
            # This is handled by strptime with %y, but let's be explicit
            if "%y" in fmt and parsed.year < 100:
                if parsed.year <= 50:
                    parsed = parsed.replace(year=parsed.year + 2000)
                else:
                    parsed = parsed.replace(year=parsed.year + 1900)

            # For formats without year, default to 1900 (strptime default)
            # Replace with current year for better UX
            if "%Y" not in fmt and "%y" not in fmt and parsed.year == 1900:
                current_year = datetime.now().year
                parsed = parsed.replace(year=current_year)

            # For month-only formats, default to first day of month
            if "%d" not in fmt:
                parsed = parsed.replace(day=1)

            # Validate year is reasonable
            if parsed.year < 1900 or parsed.year > 2100:
                continue

            # Return ISO format
            return parsed.strftime("%Y-%m-%d")

        except ValueError:
            # This format didn't work, try next
            continue

    return None


_REASON_WEIGHT: dict[str, float] = {
    "ABSTAIN": 1.0,
    "MISSING": 0.9,
    "LOW_CONFIDENCE": 0.5,
}


# =============================================================================
# ERROR HANDLING HELPERS
# =============================================================================


def _record_emission_failure(
    doc_id: str, sha256: str, error: Exception, error_type: str | None = None
) -> dict[str, Any]:
    """Record an emission failure and return error result dict.

    Args:
        doc_id: Document identifier
        sha256: Document SHA256 hash
        error: The exception that occurred
        error_type: Optional override for error type name

    Returns:
        Error result dictionary for emission_results
    """
    etype = error_type or type(error).__name__
    reason = error.reason if hasattr(error, "reason") else str(error)

    logger.error(
        "emit_document_failed",
        doc_id=doc_id,
        sha256=sha256[:16],
        error_type=etype,
        reason=reason,
    )

    return {"doc_id": doc_id, "error": str(error)}


def _handle_review_queue_read_failure(path: str, error: Exception) -> None:
    """Log a review queue read failure.

    Args:
        path: Path to the review queue file
        error: The exception that occurred
    """
    logger.warning(
        "review_queue_read_failed",
        path=path,
        error_type=type(error).__name__,
        reason=str(error),
        action="starting_fresh",
    )


def compute_field_confidence(field: str, assignment: dict[str, Any]) -> float:
    """
    Compute confidence score for a field assignment.

    Uses ML probability when available, otherwise derives confidence from
    the assignment cost using heuristic mapping. NONE assignments always
    return CONFIDENCE_ABSTAIN.

    Args:
        field: Field name
        assignment: Assignment from decoder (should include cost,
                    used_ml_model, ml_probability)

    Returns:
        Confidence score in [0, 1]
    """
    # NONE assignments always have zero confidence
    if assignment.get("assignment_type") == "NONE":
        return Config.CONFIDENCE_ABSTAIN

    # Check if ML model was used and probability is available
    used_ml = assignment.get("used_ml_model", False)
    ml_prob = assignment.get("ml_probability")

    if used_ml and ml_prob is not None:
        # ML probability is direct confidence
        confidence = float(ml_prob)
    else:
        # Compute confidence from cost using heuristic mapping
        cost = assignment.get("cost", 1.0)
        confidence = Config.compute_confidence_from_cost(cost, has_ml_model=False)

    # Clamp to valid range
    return max(Config.CONFIDENCE_FLOOR, min(Config.CONFIDENCE_CEILING, confidence))


def create_field_output(field: str, assignment: dict[str, Any]) -> dict[str, Any]:
    """
    Create field output following contract_v1 specification.

    Includes semantic validation to reject garbage predictions like:
    - Repeated words ("Tax Tax Tax Tax")
    - Keywords as values ("SUBTOTAL" as invoice number)
    - Invalid field values (numeric-only names, invalid dates)

    Args:
        field: Field name
        assignment: Normalized assignment from decoder

    Returns:
        Field output dictionary with computed confidence
    """
    if assignment["assignment_type"] == "NONE":
        return {
            "value": None,
            "confidence": Config.CONFIDENCE_ABSTAIN,
            "status": "ABSTAIN",
            "provenance": None,
            "raw_text": None,
        }

    # CANDIDATE assignment
    candidate = assignment["candidate"]
    normalized_value = assignment["normalized_value"]
    raw_text = assignment["raw_text"]

    # Compute confidence from decoder output
    confidence = compute_field_confidence(field, assignment)

    # Determine status
    if normalized_value is not None:
        status = "PREDICTED"
    else:
        # Normalization failed - abstain with zero confidence
        status = "ABSTAIN"
        confidence = Config.CONFIDENCE_ABSTAIN

    # Perform semantic validation on predicted values
    # This catches garbage predictions that passed normalization
    if status == "PREDICTED":
        # For date fields, try to normalize before semantic validation
        field_type = registry.field_type(field)
        if field_type == "date" and normalized_value is not None:
            # Try to normalize the date to ISO format
            iso_date = normalize_date(normalized_value)
            if iso_date:
                # Successfully normalized - use the ISO format
                normalized_value = iso_date
                logger.debug(
                    "date_normalized",
                    field=field,
                    original=raw_text,
                    normalized=normalized_value,
                )
            else:
                # Failed to normalize - log and continue with original value
                # (will likely fail semantic validation)
                logger.debug(
                    "date_normalization_failed",
                    field=field,
                    value=normalized_value,
                    raw_text=raw_text,
                )

        semantic_result = validate_assignment_semantics(
            field_name=field,
            normalized_value=normalized_value,
            raw_text=raw_text,
            field_type=field_type,
        )
        if not semantic_result.is_valid:
            # Primary assignment failed validation — try fallback candidates
            logger.info(
                "semantic_validation_failed",
                field=field,
                value=normalized_value,
                raw_text=raw_text,
                reason=semantic_result.reason,
            )

            fallback_used = False
            for fb in assignment.get("fallback_candidates", []):
                fb_candidate = fb["candidate"]
                fb_raw = str(
                    fb_candidate.get("raw_text") or fb_candidate.get("text", "")
                )
                fb_norm_result = normalize.normalize_field_value(field, fb_raw)
                fb_norm = fb_norm_result.get("value")

                # Date normalization for fallback
                if field_type == "date" and fb_norm is not None:
                    fb_iso = normalize_date(fb_norm)
                    if fb_iso:
                        fb_norm = fb_iso

                fb_semantic = validate_assignment_semantics(
                    field_name=field,
                    normalized_value=fb_norm,
                    raw_text=fb_raw,
                    field_type=field_type,
                )
                if fb_semantic.is_valid and fb_norm is not None:
                    logger.info(
                        "fallback_candidate_accepted",
                        field=field,
                        original_value=normalized_value,
                        fallback_value=fb_norm,
                        fallback_cost=fb["cost"],
                    )
                    # Swap in the fallback
                    candidate = fb_candidate
                    normalized_value = fb_norm
                    raw_text = fb_raw
                    # Recompute confidence from the fallback's cost/ML data
                    fb_assignment = dict(assignment)
                    fb_assignment["cost"] = fb["cost"]
                    fb_assignment["used_ml_model"] = fb["used_ml_model"]
                    fb_assignment["ml_probability"] = fb["ml_probability"]
                    confidence = compute_field_confidence(field, fb_assignment)
                    fallback_used = True
                    break

            if not fallback_used:
                status = "ABSTAIN"
                confidence = Config.CONFIDENCE_ABSTAIN

    # Create provenance
    provenance = {
        "page": int(candidate["page_idx"]),
        "bbox": [
            float(candidate["bbox_norm_x0"]),
            float(candidate["bbox_norm_y0"]),
            float(candidate["bbox_norm_x1"]),
            float(candidate["bbox_norm_y1"]),
        ],
        "token_span": [
            str(idx)
            for idx in candidate.get("token_indices", [candidate.get("token_idx", 0)])
        ],  # Safe string conversion
    }

    return {
        "value": normalized_value if status == "PREDICTED" else None,
        "confidence": confidence,
        "status": status,
        "provenance": provenance,
        "raw_text": raw_text,
    }


def compute_priority_score(
    field: str, reason: str, confidence: float
) -> tuple[float, str]:
    """Deterministic priority from field importance + reason + confidence.

    Returns (score 0-1, level: urgent|medium|low).

    The confidence term uses uncertainty sampling: items near confidence=0.5
    are most valuable to label (maximum information gain). Very low or very
    high confidence items contribute less — the model already knows what to
    do with them. Formula: learning_value = 1 - |2*confidence - 1|, which
    peaks at 1.0 when confidence=0.5 and falls to 0.0 at the extremes.
    """
    field_w = registry.importance(field)
    reason_w = _REASON_WEIGHT.get(reason, 0.5)
    conf_clamped = max(0.0, min(1.0, confidence))
    learning_value = 1.0 - abs(2.0 * conf_clamped - 1.0)

    score = 0.35 * field_w + 0.40 * reason_w + 0.25 * learning_value
    score = max(0.0, min(1.0, score))

    if score >= 0.70:
        level = "urgent"
    elif score >= 0.40:
        level = "medium"
    else:
        level = "low"

    return score, level


def create_review_queue_entry(
    doc_id: str,
    field: str,
    assignment: dict[str, Any],
    reason: str,
    confidence: float = 0.0,
    sha256: str = "",
) -> dict[str, Any]:
    """
    Create a review queue entry for manual review.

    Args:
        doc_id: Document identifier
        field: Field name
        assignment: Assignment from decoder
        reason: Reason for review (e.g., "ABSTAIN")
        confidence: ML confidence score for the assignment
        sha256: Full SHA256 hash of the document

    Returns:
        Review queue entry dictionary
    """
    entry: dict[str, Any] = {
        "doc_id": doc_id,
        "sha256": sha256,
        "field": field,
        "reason": reason,
        "ml_confidence": round(confidence, 4),
        "page": None,
        "bbox_x0": None,
        "bbox_y0": None,
        "bbox_x1": None,
        "bbox_y1": None,
        "token_span": None,
        "raw_text": None,
    }

    # Add details if candidate assignment
    if assignment["assignment_type"] == "CANDIDATE" and "candidate" in assignment:
        candidate = assignment["candidate"]
        entry.update(
            {
                "page": int(candidate["page_idx"]),
                "bbox_x0": float(candidate["bbox_norm_x0"]),
                "bbox_y0": float(candidate["bbox_norm_y0"]),
                "bbox_x1": float(candidate["bbox_norm_x1"]),
                "bbox_y1": float(candidate["bbox_norm_y1"]),
                "token_span": ",".join(
                    str(idx)
                    for idx in candidate.get(
                        "token_indices", [candidate.get("token_idx", 0)]
                    )
                ),  # Safe string conversion
                "raw_text": str(assignment.get("raw_text", "")),  # Ensure string
            }
        )

    priority_score, priority_level = compute_priority_score(field, reason, confidence)
    entry["priority_score"] = round(priority_score, 4)
    entry["priority_level"] = priority_level

    return entry


def _compute_field(
    field_name: str,
    computed_fn: str,
    source_fields: list[str],
    assignments: dict[str, Any],
    contract_fields: dict[str, Any],
) -> dict[str, Any] | None:
    """Generic computed field dispatcher.

    Args:
        field_name: Name of computed field
        computed_fn: Function to dispatch on (infer_currency, concat_strip)
        source_fields: List of source field names from schema
        assignments: Raw decoder output (for currency inference)
        contract_fields: Processed contract fields dict (for billing reference)

    Returns:
        Field output dict (value, confidence, status, provenance, raw_text) or None
    """
    if computed_fn == "infer_currency":
        # Source fields from schema: ["TotalAmount", "Subtotal", "TaxAmount"]
        inferred_code = None
        source_field = None
        for fname in source_fields:
            assignment = assignments.get(fname)
            if assignment is None:
                continue
            if assignment.get("assignment_type") == "NONE":
                continue
            currency_code = assignment.get("currency_code")
            if currency_code:
                inferred_code = str(currency_code)
                source_field = fname
                break

        if not inferred_code:
            return None

        # Build provenance from the source amount candidate
        inferred_provenance = {"page": 0, "bbox": [0, 0, 0, 0], "token_span": []}
        if source_field and "candidate" in assignments[source_field]:
            src_cand = assignments[source_field]["candidate"]
            inferred_provenance = {
                "page": int(src_cand.get("page_idx", 0)),
                "bbox": [
                    float(src_cand.get("bbox_norm_x0", 0)),
                    float(src_cand.get("bbox_norm_y0", 0)),
                    float(src_cand.get("bbox_norm_x1", 0)),
                    float(src_cand.get("bbox_norm_y1", 0)),
                ],
                "token_span": [
                    str(idx)
                    for idx in src_cand.get(
                        "token_indices", [src_cand.get("token_idx", 0)]
                    )
                ],
            }

        return {
            "value": inferred_code,
            "confidence": Config.CONFIDENCE_INFERRED,
            "status": "PREDICTED",
            "provenance": inferred_provenance,
            "raw_text": assignments.get(source_field or "", {}).get("raw_text"),
        }

    elif computed_fn == "concat_strip":
        # Source fields from schema: ["CustomerAccount", "InvoiceDate"]
        # Use contract_fields (already processed)
        if len(source_fields) < 2:
            return None

        field1_name = source_fields[0]
        field2_name = source_fields[1]
        field1 = contract_fields.get(field1_name, {})
        field2 = contract_fields.get(field2_name, {})

        if (
            field1.get("status") == "PREDICTED"
            and field1.get("value")
            and field2.get("status") == "PREDICTED"
            and field2.get("value")
        ):
            # field2 is InvoiceDate (already ISO format), strip dashes
            date_compact = str(field2["value"]).replace("-", "")
            computed_value = str(field1["value"]) + date_compact

            return {
                "value": computed_value,
                "confidence": Config.CONFIDENCE_INFERRED,
                "status": "PREDICTED",
                "provenance": {
                    "page": 0,
                    "bbox": [0, 0, 0, 0],
                    "token_span": [],
                    "computed_from": source_fields,
                },
                "raw_text": None,
            }

        return None

    # Unknown computed_fn
    return None


def _ensure_normalized(assignments: dict[str, Any], sha256: str) -> dict[str, Any]:
    """Ensure assignments are normalized, normalizing in-place if needed.

    Detects raw decoder output (missing ``normalized_value`` on CANDIDATE
    assignments) and runs ``normalize.normalize_assignments`` so that
    every caller of ``emit_document`` gets consistent behaviour regardless
    of entry point (API, CLI, orchestrator).
    """
    # Check any CANDIDATE assignment for the normalized_value key
    for assignment in assignments.values():
        if assignment.get("assignment_type") == "CANDIDATE":
            if "normalized_value" not in assignment:
                return normalize.normalize_assignments(assignments, sha256=sha256)
            break  # Already normalized
    return assignments


def emit_document(
    sha256: str, assignments: dict[str, Any]
) -> tuple[str, list[dict[str, Any]], bool]:
    """
    Emit contract JSON for a single document using schema-driven fields.

    Args:
        sha256: Document SHA256 hash
        assignments: Assignments from decoder (raw or pre-normalized).
            If raw, normalization is applied automatically.

    Returns:
        A tuple of (predictions_json_path, review_queue_entries, needs_review).
        - predictions_json_path: Path to saved JSON file
        - review_queue_entries: List of entries for review queue
        - needs_review: True if document should be routed to human review
    """
    start_time = time.perf_counter()

    # Guarantee normalization regardless of entry point
    assignments = _ensure_normalized(assignments, sha256)

    # Get document info
    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")

    doc_id = doc_info["doc_id"]

    # Load schema for field list
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    if not schema_fields:
        raise ValueError("Schema contains no fields")

    # Get tokens to determine page count
    tokens_df = tokenize.get_document_tokens(sha256)
    page_count = tokens_df["page_idx"].nunique() if not tokens_df.empty else 0

    # Create contract JSON with version info
    contract = {
        "document_id": doc_id,
        "pages": page_count,
        **utils.get_version_info(),
        "fields": {},
        # Line item extraction is not yet implemented. Key is present to
        # satisfy the contract schema; it will be populated once multi-row
        # extraction lands. See schema/contract.invoice.json for spec.
        "line_items": [],
        "extensions": {},  # Top-level extensions object
        "experimental": {},  # Top-level experimental object
    }

    review_queue_entries = []
    field_confidences: dict[str, float] = {}  # Track for routing decision

    # Process each schema field
    for field in schema_fields:
        if field in assignments:
            # Field has assignment from decoder
            assignment = assignments[field]
            field_output = create_field_output(field, assignment)
            contract["fields"][field] = field_output

            # Track confidence for routing
            field_confidences[field] = field_output["confidence"]

            # Record field metrics (best-effort)
            try:
                from .metrics import field_confidence as conf_hist
                from .metrics import field_status as status_counter

                conf_hist.labels(field=field, status=field_output["status"]).observe(
                    field_output["confidence"]
                )
                status_counter.labels(field=field, status=field_output["status"]).inc()
            except ImportError:
                pass

            # Add to review queue if ABSTAIN
            if field_output["status"] == "ABSTAIN":
                review_entry = create_review_queue_entry(
                    doc_id,
                    field,
                    assignment,
                    "ABSTAIN",
                    confidence=field_output["confidence"],
                    sha256=sha256,
                )
                review_queue_entries.append(review_entry)
            # Also add to review queue if low confidence PREDICTED
            elif field_output["status"] == "PREDICTED":
                threshold = Config.get_field_confidence_threshold(field)
                if field_output["confidence"] < threshold:
                    review_entry = create_review_queue_entry(
                        doc_id,
                        field,
                        assignment,
                        "LOW_CONFIDENCE",
                        confidence=field_output["confidence"],
                        sha256=sha256,
                    )
                    review_queue_entries.append(review_entry)
        else:
            # Field missing from assignments - status MISSING
            contract["fields"][field] = {
                "value": None,
                "confidence": Config.CONFIDENCE_ABSTAIN,
                "status": "MISSING",
                "provenance": None,
                "raw_text": None,
            }

            # Track as zero confidence for routing
            field_confidences[field] = Config.CONFIDENCE_ABSTAIN

            # Add to review queue
            review_entry = create_review_queue_entry(
                doc_id,
                field,
                {"assignment_type": "NONE"},
                "MISSING",
                confidence=Config.CONFIDENCE_ABSTAIN,
                sha256=sha256,
            )
            review_queue_entries.append(review_entry)

    # Post-processing: compute fields driven by schema
    for cf_name in registry.computed_fields():
        cf_def = registry.field_def(cf_name)
        computed_fn = cf_def.get("computed_fn")
        source_fields = cf_def.get("computed_from", [])
        if not computed_fn:
            continue

        # Special handling for Currency: check if inference is needed
        if cf_name == "Currency":
            currency_field = contract["fields"].get("Currency")
            currency_needs_inference = False
            if currency_field:
                if currency_field["status"] in ("ABSTAIN", "MISSING"):
                    currency_needs_inference = True
                elif (
                    currency_field["status"] == "PREDICTED"
                    and currency_field.get("value")
                    and currency_field["value"].upper()
                    not in {c.upper() for c in Config.CURRENCY_CODES}
                ):
                    currency_needs_inference = True
            if not currency_needs_inference:
                continue

        result = _compute_field(
            cf_name, computed_fn, source_fields, assignments, contract["fields"]
        )
        if result:
            contract["fields"][cf_name] = result
            field_confidences[cf_name] = result["confidence"]
            # Remove from review queue if it was there
            review_queue_entries = [
                e for e in review_queue_entries if e["field"] != cf_name
            ]
            logger.info(
                f"{cf_name.lower()}_computed", doc_id=doc_id, value=result["value"]
            )
        elif cf_name in schema_fields and cf_name not in contract["fields"]:
            # Source fields missing — mark as MISSING
            contract["fields"][cf_name] = {
                "value": None,
                "confidence": 0.0,
                "status": "MISSING",
                "provenance": None,
                "raw_text": None,
            }

    # Determine if document needs human review
    needs_review = Config.needs_review(field_confidences)

    # Add routing metadata to extensions
    predicted_count = sum(
        1 for f in contract["fields"].values() if f["status"] == "PREDICTED"
    )
    abstain_count = sum(
        1 for f in contract["fields"].values() if f["status"] == "ABSTAIN"
    )
    missing_count = sum(
        1 for f in contract["fields"].values() if f["status"] == "MISSING"
    )

    # Compute aggregate confidence (min of predicted fields, or 0 if none)
    predicted_confidences = [
        f["confidence"]
        for f in contract["fields"].values()
        if f["status"] == "PREDICTED"
    ]
    min_confidence = min(predicted_confidences) if predicted_confidences else 0.0
    avg_confidence = (
        sum(predicted_confidences) / len(predicted_confidences)
        if predicted_confidences
        else 0.0
    )

    contract["extensions"]["routing"] = {
        "needs_review": needs_review,
        "review_reasons": [e["reason"] for e in review_queue_entries],
        "predicted_count": predicted_count,
        "abstain_count": abstain_count,
        "missing_count": missing_count,
        "min_confidence": round(min_confidence, 4),
        "avg_confidence": round(avg_confidence, 4),
    }

    # Count ML-scored vs heuristic fields
    ml_scored = []
    heuristic = []
    for f_name, f_output in contract["fields"].items():
        if f_output["status"] != "PREDICTED":
            continue
        assignment = assignments.get(f_name, {})
        if assignment.get("used_ml_model", False):
            ml_scored.append(f_name)
        else:
            heuristic.append(f_name)

    contract["extensions"]["scoring"] = {
        "ml_scored_fields": ml_scored,
        "heuristic_fields": heuristic,
        "ml_model_used": len(ml_scored) > 0,
    }

    # Write predictions JSON
    predictions_path = paths.get_predictions_path(sha256)
    utils.write_json_with_backup(predictions_path, contract)

    documents_processed.labels(status="success").inc()
    duration = time.perf_counter() - start_time
    pipeline_duration.labels(stage="emit").observe(duration)

    if needs_review:
        docs_needs_review.inc()
    else:
        docs_auto_approved.inc()

    return str(predictions_path), review_queue_entries, needs_review


def emit_all_documents(
    precomputed_assignments: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Emit contract JSON for all documents and manage review queue.

    Args:
        precomputed_assignments: If provided, skip decode and use these
            assignments directly (sha256 → field assignments dict).
            Useful when the caller already ran decode_all_documents().

    Returns:
        Summary statistics
    """
    # Get all documents
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        logger.info("no_documents_in_index")
        return {}

    if precomputed_assignments is not None:
        all_assignments = precomputed_assignments
    else:
        # Decode all documents first
        logger.info("decoding_all_documents")
        # Disable pruning when ranker is disabled — pruning models were trained
        # assuming ranker would boost scores, so they're too aggressive for heuristic-only mode
        enable_pruning = Config.USE_RANKER_MODEL
        all_assignments = decoder.decode_all_documents(enable_pruning=enable_pruning)

    # Normalize assignments
    logger.info("normalizing_assignments")
    normalized_assignments = {}
    for sha256, assignments in all_assignments.items():
        normalized_assignments[sha256] = normalize.normalize_assignments(
            assignments, sha256=sha256
        )

    # Emit predictions and collect review queue entries
    all_review_entries = []
    emission_results = {}
    docs_needing_review = 0
    docs_auto_approved = 0

    logger.info("emitting_predictions", doc_count=len(indexed_docs))

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]
        doc_id = doc_info["doc_id"]

        try:
            assignments = normalized_assignments.get(sha256, {})

            if not assignments:
                logger.warning("no_assignments_found", doc_id=doc_id)
                continue

            predictions_path, review_entries, needs_review = emit_document(
                sha256, assignments
            )
            all_review_entries.extend(review_entries)

            # Track routing statistics
            if needs_review:
                docs_needing_review += 1
            else:
                docs_auto_approved += 1

            # Count statuses
            status_counts: dict[str, int] = {}
            for field_output in assignments.values():
                if (
                    "normalized_value" in field_output
                    and field_output["normalized_value"] is not None
                ):
                    status = "PREDICTED"
                else:
                    status = "ABSTAIN"

                status_counts[status] = status_counts.get(status, 0) + 1

            emission_results[sha256] = {
                "doc_id": doc_id,
                "predictions_path": predictions_path,
                "status_counts": status_counts,
                "review_entries": len(review_entries),
                "needs_review": needs_review,
            }

            logger.info(
                "emitted_document",
                doc_id=doc_id,
                status_counts=status_counts,
                needs_review=needs_review,
            )

        except (ValueError, TypeError, KeyError) as e:
            # Data validation or missing key errors during emission
            pipeline_errors.labels(stage="emit").inc()
            emission_results[sha256] = _record_emission_failure(doc_id, sha256, e)
        except EmissionError as e:
            # Typed emission errors
            pipeline_errors.labels(stage="emit").inc()
            emission_results[sha256] = _record_emission_failure(
                doc_id, sha256, e, error_type="EmissionError"
            )
        except OSError as e:
            # File system errors during emission
            pipeline_errors.labels(stage="emit").inc()
            emission_results[sha256] = _record_emission_failure(doc_id, sha256, e)

    # Update review queue
    update_review_queue(all_review_entries)

    # Summary statistics
    total_predicted = sum(
        r.get("status_counts", {}).get("PREDICTED", 0)
        for r in emission_results.values()
    )
    total_abstain = sum(
        r.get("status_counts", {}).get("ABSTAIN", 0) for r in emission_results.values()
    )
    total_review_entries = len(all_review_entries)

    summary = {
        "documents_processed": len(emission_results),
        "total_predicted": total_predicted,
        "total_abstain": total_abstain,
        "total_review_entries": total_review_entries,
        "docs_needing_review": docs_needing_review,
        "docs_auto_approved": docs_auto_approved,
        "results": emission_results,
    }

    # Update Prometheus metrics
    # NOTE: documents_processed and routing counters (docs_auto_approved,
    # docs_needs_review) are already incremented per-doc in emit_document().
    # Only update aggregate metrics here (queue, predictions).
    try:
        from .metrics import predictions_emitted, update_queue_metrics

        for _sha256, result in emission_results.items():
            if "error" not in result:
                for status, count in result.get("status_counts", {}).items():
                    predictions_emitted.labels(status=status).inc(count)

        # Update queue gauge
        priority_counts: dict[str, int] = {}
        for entry in all_review_entries:
            level = entry.get("priority_level", "low")
            priority_counts[level] = priority_counts.get(level, 0) + 1
        update_queue_metrics(len(all_review_entries), priority_counts)

    except ImportError:
        pass  # prometheus-client not installed

    logger.info(
        "emission_complete",
        predicted=total_predicted,
        abstain=total_abstain,
        review_entries=total_review_entries,
        auto_approved=docs_auto_approved,
        needs_review=docs_needing_review,
    )

    return summary


def update_review_queue(new_entries: list[dict[str, Any]]) -> None:
    """
    Replace review queue with current entries (not append).

    Since emit_all_documents() processes ALL documents, the new entries
    ARE the complete queue. This prevents stale ABSTAIN entries from
    lingering when fields improve to PREDICTED.

    Args:
        new_entries: Complete list of current review queue entries
    """
    review_queue_path = paths.get_review_queue_path()

    if not new_entries:
        # No review needed — clear the queue
        if review_queue_path.exists():
            review_queue_path.unlink()
            logger.info("review_queue_cleared")
        return

    # Replace queue entirely
    new_df = pd.DataFrame(new_entries)
    io_utils.write_parquet_safe(new_df, review_queue_path)

    logger.info("review_queue_replaced", items=len(new_entries))


def get_review_queue() -> pd.DataFrame:
    """Get the current review queue."""
    review_queue_path = paths.get_review_queue_path()
    df = io_utils.read_parquet_safe(review_queue_path, on_error="empty")

    if df is None:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "field",
                "reason",
                "page",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "token_span",
                "raw_text",
                "priority_score",
                "priority_level",
            ]
        )

    return df


def get_document_predictions(sha256: str) -> dict[str, Any] | None:
    """Get predictions for a specific document."""
    predictions_path = paths.get_predictions_path(sha256)

    if not predictions_path.exists():
        return None

    with open(predictions_path, encoding="utf-8") as f:
        return json.load(f)  # type: ignore[no-any-return]
