"""Reporting module for pipeline metrics and statistics."""

import json
from collections import Counter, defaultdict
from typing import Any

from . import candidates, emit, ingest, paths, tokenize, utils
from .logging import get_logger

logger = get_logger(__name__)


def collect_field_statistics() -> dict[str, dict[str, int]]:
    """Collect statistics for all fields across all documents."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        return {}

    # Get schema fields for consistent reporting
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    field_stats: defaultdict[str, Counter[str]] = defaultdict(lambda: Counter())

    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info["sha256"]

        try:
            predictions = emit.get_document_predictions(sha256)

            if predictions and "fields" in predictions:
                for field_name, field_data in predictions["fields"].items():
                    status = field_data.get("status", "MISSING")
                    field_stats[field_name][status] += 1
            else:
                # No predictions found - mark all schema fields as MISSING
                for field in schema_fields:
                    field_stats[field]["MISSING"] += 1

        except Exception as e:
            logger.error("prediction_read_failed", sha256=sha256[:16], error=str(e))
            # Mark all schema fields as MISSING for this document
            for field in schema_fields:
                field_stats[field]["MISSING"] += 1

    return dict(field_stats)


def collect_document_statistics() -> list[dict[str, Any]]:
    """Collect per-document statistics."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        return []

    # Get schema fields for consistent reporting
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    doc_stats = []

    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info["sha256"]
        doc_id = doc_info["doc_id"]

        try:
            # Get tokens info
            token_summary = tokenize.get_token_summary(sha256)

            # Get candidates info
            candidates_df = candidates.get_document_candidates(sha256)
            candidate_count = len(candidates_df)

            # Get predictions info
            predictions = emit.get_document_predictions(sha256)

            status_counts: Counter[str] = Counter()
            if predictions and "fields" in predictions:
                for field_data in predictions["fields"].values():
                    status = field_data.get("status", "MISSING")
                    status_counts[status] += 1

            doc_stat = {
                "sha256": sha256,
                "doc_id": doc_id,
                "pages": token_summary.get("page_count", 0),
                "tokens": token_summary.get("token_count", 0),
                "candidates": candidate_count,
                "predicted": status_counts.get("PREDICTED", 0),
                "abstain": status_counts.get("ABSTAIN", 0),
                "missing": status_counts.get("MISSING", 0),
            }

            doc_stats.append(doc_stat)

        except Exception as e:
            logger.error(
                "doc_stats_collection_failed", sha256=sha256[:16], error=str(e)
            )
            doc_stats.append(
                {
                    "sha256": sha256,
                    "doc_id": doc_id,
                    "pages": 0,
                    "tokens": 0,
                    "candidates": 0,
                    "predicted": 0,
                    "abstain": 0,
                    "missing": len(schema_fields),
                    "error": str(e),
                }
            )

    return doc_stats


def collect_review_queue_statistics() -> dict[str, Any]:
    """Collect review queue statistics."""
    try:
        review_df = emit.get_review_queue()

        if review_df.empty:
            return {
                "total_entries": 0,
                "by_field": {},
                "by_reason": {},
            }

        # Count by field
        by_field = review_df["field"].value_counts().to_dict()

        # Count by reason
        by_reason = review_df["reason"].value_counts().to_dict()

        return {
            "total_entries": len(review_df),
            "by_field": by_field,
            "by_reason": by_reason,
        }

    except Exception as e:
        logger.error("review_queue_read_failed", error=str(e))
        return {
            "total_entries": 0,
            "by_field": {},
            "by_reason": {},
            "error": str(e),
        }


def collect_coverage_statistics() -> dict[str, Any]:
    """Collect coverage probe statistics."""
    try:
        coverage_stats = candidates.get_coverage_statistics()

        if not coverage_stats:
            return {
                "total_documents": 0,
                "coverage_probes": {},
                "bucket_distribution": {},
            }

        # Calculate coverage percentages
        total_candidates = coverage_stats.get("total_candidates", 0)
        bucket_dist = dict(coverage_stats.get("bucket_distribution", {}))

        coverage_probes = {}

        # Field-specific coverage probes
        field_types = {
            "invoice_number": "id_like",
            "invoice_date": "date_like",
            "due_date": "date_like",
            "total_amount_due": "amount_like",
            "account_number": "id_like",
        }

        for field, expected_type in field_types.items():
            type_count = bucket_dist.get(expected_type, 0)
            proximal_count = bucket_dist.get("keyword_proximal", 0)

            # Cue-coverage: percentage with keyword proximity
            cue_coverage = (proximal_count / max(total_candidates, 1)) * 100

            # Region-coverage: percentage of expected type found
            region_coverage = (type_count / max(total_candidates, 1)) * 100

            coverage_probes[field] = {
                "cue_coverage_percent": round(cue_coverage, 1),
                "region_coverage_percent": round(region_coverage, 1),
                "expected_type_count": type_count,
                "total_candidates": total_candidates,
            }

        return {
            "total_documents": coverage_stats.get("total_documents", 0),
            "documents_with_candidates": coverage_stats.get(
                "documents_with_candidates", 0
            ),
            "total_candidates": total_candidates,
            "coverage_probes": coverage_probes,
            "bucket_distribution": bucket_dist,
        }

    except Exception as e:
        logger.warning("coverage_stats_collection_failed", error=str(e))
        return {
            "total_documents": 0,
            "coverage_probes": {},
            "bucket_distribution": {},
            "error": str(e),
        }


def print_field_report(field_stats: dict[str, dict[str, int]]) -> None:
    """Log field statistics report."""
    if not field_stats:
        logger.info("no_field_statistics_available")
        return

    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    for field in schema_fields:
        stats = field_stats.get(field, {})
        predicted = stats.get("PREDICTED", 0)
        abstain = stats.get("ABSTAIN", 0)
        missing = stats.get("MISSING", 0)
        logger.info(
            "field_stats",
            field=field,
            predicted=predicted,
            abstain=abstain,
            missing=missing,
        )

    total_predicted = sum(stats.get("PREDICTED", 0) for stats in field_stats.values())
    total_abstain = sum(stats.get("ABSTAIN", 0) for stats in field_stats.values())
    total_missing = sum(stats.get("MISSING", 0) for stats in field_stats.values())
    logger.info(
        "field_stats_totals",
        predicted=total_predicted,
        abstain=total_abstain,
        missing=total_missing,
    )


def print_document_report(doc_stats: list[dict[str, Any]]) -> None:
    """Log document statistics report."""
    if not doc_stats:
        logger.info("no_document_statistics_available")
        return

    for stat in doc_stats:
        logger.info(
            "document_stats",
            doc_id=stat["doc_id"],
            pages=stat.get("pages", 0),
            tokens=stat.get("tokens", 0),
            candidates=stat.get("candidates", 0),
            predicted=stat.get("predicted", 0),
            abstain=stat.get("abstain", 0),
            missing=stat.get("missing", 0),
        )

    if doc_stats:
        avg_pages = sum(s.get("pages", 0) for s in doc_stats) / len(doc_stats)
        avg_tokens = sum(s.get("tokens", 0) for s in doc_stats) / len(doc_stats)
        avg_candidates = sum(s.get("candidates", 0) for s in doc_stats) / len(doc_stats)
        logger.info(
            "document_stats_averages",
            avg_pages=round(avg_pages, 1),
            avg_tokens=round(avg_tokens, 0),
            avg_candidates=round(avg_candidates, 1),
        )


def print_review_queue_report(review_stats: dict[str, Any]) -> None:
    """Log review queue statistics."""
    total_entries = review_stats.get("total_entries", 0)
    logger.info("review_queue_stats", total_entries=total_entries)

    if total_entries == 0:
        return

    by_field = review_stats.get("by_field", {})
    if by_field:
        logger.info("review_queue_by_field", **by_field)

    by_reason = review_stats.get("by_reason", {})
    if by_reason:
        logger.info("review_queue_by_reason", **by_reason)


def print_coverage_report(coverage_stats: dict[str, Any]) -> None:
    """Log coverage probe report."""
    total_docs = coverage_stats.get("total_documents", 0)
    docs_with_candidates = coverage_stats.get("documents_with_candidates", 0)
    total_candidates = coverage_stats.get("total_candidates", 0)

    logger.info(
        "coverage_stats",
        total_docs=total_docs,
        docs_with_candidates=docs_with_candidates,
        total_candidates=total_candidates,
    )

    coverage_probes = coverage_stats.get("coverage_probes", {})
    for field, stats in coverage_probes.items():
        logger.info(
            "coverage_probe",
            field=field,
            cue_coverage_pct=stats.get("cue_coverage_percent", 0),
            region_coverage_pct=stats.get("region_coverage_percent", 0),
        )

    bucket_dist = coverage_stats.get("bucket_distribution", {})
    if bucket_dist:
        logger.info("bucket_distribution", **bucket_dist)


def generate_report() -> dict[str, Any]:
    """
    Generate comprehensive pipeline report.

    Returns:
        Report data dictionary
    """
    logger.info("generating_pipeline_report")

    with utils.Timer("Report generation"):
        # Collect all statistics
        field_stats = collect_field_statistics()
        doc_stats = collect_document_statistics()
        review_stats = collect_review_queue_statistics()
        coverage_stats = collect_coverage_statistics()

        # Print reports
        print_field_report(field_stats)
        print_document_report(doc_stats)
        print_review_queue_report(review_stats)
        print_coverage_report(coverage_stats)

        # Create summary
        report_data = {
            "field_statistics": field_stats,
            "document_statistics": doc_stats,
            "review_queue_statistics": review_stats,
            "coverage_statistics": coverage_stats,
            "generated_at": utils.get_current_utc_iso(),
            **utils.get_version_stamps(),
        }

        return report_data


def save_report(report_data: dict[str, Any], filename: str | None = None) -> str:
    """Save report data to logs directory."""
    if filename is None:
        timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
        filename = f"report_{timestamp}.json"

    logs_dir = paths.get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)

    report_path = logs_dir / filename

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logger.info("report_saved", path=str(report_path))
    return str(report_path)
