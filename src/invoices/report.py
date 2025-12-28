"""Reporting module for pipeline metrics and statistics."""

import json
from collections import Counter, defaultdict
from typing import Any

from . import candidates, emit, ingest, paths, tokenize, utils


def collect_field_statistics() -> dict[str, dict[str, int]]:
    """Collect statistics for all fields across all documents."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        return {}

    # Get schema fields for consistent reporting
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    field_stats = defaultdict(lambda: Counter())

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
            print(f"Error reading predictions for {sha256[:16]}: {e}")
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

            status_counts = Counter()
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
            print(f"Error collecting stats for {sha256[:16]}: {e}")
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
        print(f"Error reading review queue: {e}")
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
        print(f"Warning: Failed to collect coverage statistics: {e}")
        return {
            "total_documents": 0,
            "coverage_probes": {},
            "bucket_distribution": {},
            "error": str(e),
        }


def print_field_report(field_stats: dict[str, dict[str, int]]) -> None:
    """Print formatted field statistics report."""
    print("\n" + "=" * 80)
    print("FIELD STATISTICS")
    print("=" * 80)

    if not field_stats:
        print("No field statistics available")
        return

    # Print header
    print(
        f"{'Field':<20} {'Predicted':<10} {'Abstain':<10} {'Missing':<10} {'Total':<10}"
    )
    print("-" * 70)

    # Get schema fields for display
    schema_obj = utils.load_contract_schema()
    schema_fields = schema_obj.get("fields", [])

    # Print each field
    for field in schema_fields:
        stats = field_stats.get(field, {})
        predicted = stats.get("PREDICTED", 0)
        abstain = stats.get("ABSTAIN", 0)
        missing = stats.get("MISSING", 0)
        total = predicted + abstain + missing

        print(f"{field:<20} {predicted:<10} {abstain:<10} {missing:<10} {total:<10}")

    # Print totals
    print("-" * 70)
    total_predicted = sum(stats.get("PREDICTED", 0) for stats in field_stats.values())
    total_abstain = sum(stats.get("ABSTAIN", 0) for stats in field_stats.values())
    total_missing = sum(stats.get("MISSING", 0) for stats in field_stats.values())
    grand_total = total_predicted + total_abstain + total_missing

    print(
        f"{'TOTAL':<20} {total_predicted:<10} {total_abstain:<10} {total_missing:<10} {grand_total:<10}"
    )


def print_document_report(doc_stats: list[dict[str, Any]]) -> None:
    """Print formatted document statistics report."""
    print("\n" + "=" * 80)
    print("DOCUMENT STATISTICS")
    print("=" * 80)

    if not doc_stats:
        print("No document statistics available")
        return

    # Print header
    print(
        f"{'Doc ID':<20} {'Pages':<6} {'Tokens':<8} {'Candidates':<11} {'Pred':<5} {'Abst':<5} {'Miss':<5}"
    )
    print("-" * 70)

    # Print each document
    for stat in doc_stats:
        doc_id = stat["doc_id"][:20]  # Truncate for display
        pages = stat.get("pages", 0)
        tokens = stat.get("tokens", 0)
        candidates = stat.get("candidates", 0)
        predicted = stat.get("predicted", 0)
        abstain = stat.get("abstain", 0)
        missing = stat.get("missing", 0)

        print(
            f"{doc_id:<20} {pages:<6} {tokens:<8} {candidates:<11} {predicted:<5} {abstain:<5} {missing:<5}"
        )

    # Print summary statistics
    print("-" * 70)
    if doc_stats:
        avg_pages = sum(s.get("pages", 0) for s in doc_stats) / len(doc_stats)
        avg_tokens = sum(s.get("tokens", 0) for s in doc_stats) / len(doc_stats)
        avg_candidates = sum(s.get("candidates", 0) for s in doc_stats) / len(doc_stats)

        print(
            f"{'AVERAGES':<20} {avg_pages:<6.1f} {avg_tokens:<8.0f} {avg_candidates:<11.1f}"
        )


def print_review_queue_report(review_stats: dict[str, Any]) -> None:
    """Print formatted review queue statistics."""
    print("\n" + "=" * 80)
    print("REVIEW QUEUE STATISTICS")
    print("=" * 80)

    total_entries = review_stats.get("total_entries", 0)
    print(f"Total entries in review queue: {total_entries}")

    if total_entries == 0:
        print("Review queue is empty")
        return

    # By field
    by_field = review_stats.get("by_field", {})
    if by_field:
        print("Entries by field:")
        for field, count in sorted(by_field.items()):
            print(f"  {field}: {count}")

    # By reason
    by_reason = review_stats.get("by_reason", {})
    if by_reason:
        print("Entries by reason:")
        for reason, count in sorted(by_reason.items()):
            print(f"  {reason}: {count}")


def print_coverage_report(coverage_stats: dict[str, Any]) -> None:
    """Print coverage probe report."""
    print("\nCOVERAGE PROBES")
    print("=" * 50)

    total_docs = coverage_stats.get("total_documents", 0)
    docs_with_candidates = coverage_stats.get("documents_with_candidates", 0)
    total_candidates = coverage_stats.get("total_candidates", 0)

    print(f"Documents processed: {total_docs}")
    print(f"Documents with candidates: {docs_with_candidates}")
    print(f"Total candidates: {total_candidates}")

    # Coverage probes by field
    coverage_probes = coverage_stats.get("coverage_probes", {})
    if coverage_probes:
        print("\nField Coverage Probes:")
        print(f"{'Field':<20} {'Cue Coverage':<15} {'Region Coverage':<15}")
        print("-" * 50)

        for field, stats in coverage_probes.items():
            cue_pct = stats.get("cue_coverage_percent", 0)
            region_pct = stats.get("region_coverage_percent", 0)
            print(f"{field:<20} {cue_pct:>13.1f}% {region_pct:>13.1f}%")

    # Bucket distribution
    bucket_dist = coverage_stats.get("bucket_distribution", {})
    if bucket_dist:
        print("\nBucket Distribution:")
        for bucket, count in sorted(bucket_dist.items()):
            percentage = (count / max(total_candidates, 1)) * 100
            print(f"  {bucket}: {count} ({percentage:.1f}%)")


def generate_report() -> dict[str, Any]:
    """
    Generate comprehensive pipeline report.

    Returns:
        Report data dictionary
    """
    print("Generating pipeline report...")

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

    print(f"Report saved to: {report_path}")
    return str(report_path)
