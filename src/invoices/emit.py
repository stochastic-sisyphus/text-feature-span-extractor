"""Contract emission module for JSON output and review queue management."""

import json
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import decoder, ingest, normalize, paths, tokenize, utils


def create_field_output(field: str, assignment: dict[str, Any]) -> dict[str, Any]:
    """
    Create field output following contract_v1 specification.

    Args:
        field: Field name
        assignment: Normalized assignment from decoder

    Returns:
        Field output dictionary
    """
    if assignment["assignment_type"] == "NONE":
        return {
            "value": None,
            "confidence": 0.0,
            "status": "ABSTAIN",
            "provenance": None,
            "raw_text": None,
        }

    # CANDIDATE assignment
    candidate = assignment["candidate"]
    normalized_value = assignment["normalized_value"]
    raw_text = assignment["raw_text"]

    # Determine status
    if normalized_value is not None:
        status = "PREDICTED"
        confidence = 0.0  # Untrained baseline as specified
    else:
        status = "ABSTAIN"  # Normalization failed
        confidence = 0.0

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
        "value": normalized_value,
        "confidence": confidence,
        "status": status,
        "provenance": provenance,
        "raw_text": raw_text,
    }


def create_review_queue_entry(
    doc_id: str, field: str, assignment: dict[str, Any], reason: str
) -> dict[str, Any]:
    """
    Create a review queue entry for manual review.

    Args:
        doc_id: Document identifier
        field: Field name
        assignment: Assignment from decoder
        reason: Reason for review (e.g., "ABSTAIN")

    Returns:
        Review queue entry dictionary
    """
    entry = {
        "doc_id": doc_id,
        "field": field,
        "reason": reason,
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

    return entry


def emit_document(
    sha256: str, assignments: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    """
    Emit contract JSON for a single document using schema-driven fields.

    Args:
        sha256: Document SHA256 hash
        assignments: Normalized assignments from decoder

    Returns:
        Tuple of (predictions_json_path, review_queue_entries)
    """
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
        "line_items": [],  # Empty for now as specified
        "extensions": {},  # Top-level extensions object
        "experimental": {},  # Top-level experimental object
    }

    review_queue_entries = []

    # Process each schema field
    for field in schema_fields:
        if field in assignments:
            # Field has assignment from decoder
            assignment = assignments[field]
            field_output = create_field_output(field, assignment)
            contract["fields"][field] = field_output

            # Add to review queue if ABSTAIN
            if field_output["status"] == "ABSTAIN":
                review_entry = create_review_queue_entry(
                    doc_id, field, assignment, "ABSTAIN"
                )
                review_queue_entries.append(review_entry)
        else:
            # Field missing from assignments - status MISSING
            contract["fields"][field] = {
                "value": None,
                "confidence": 0.0,
                "status": "MISSING",
                "provenance": None,
                "raw_text": None,
            }

            # Add to review queue
            review_entry = create_review_queue_entry(
                doc_id, field, {"assignment_type": "NONE"}, "MISSING"
            )
            review_queue_entries.append(review_entry)

    # Write predictions JSON
    predictions_path = paths.get_predictions_path(sha256)
    utils.write_json_with_backup(predictions_path, contract)

    return str(predictions_path), review_queue_entries


def emit_all_documents() -> dict[str, Any]:
    """
    Emit contract JSON for all documents and manage review queue.

    Returns:
        Summary statistics
    """
    # Get all documents
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        print("No documents found in index")
        return {}

    # Decode all documents first
    print("Decoding all documents...")
    all_assignments = decoder.decode_all_documents()

    # Normalize assignments
    print("Normalizing assignments...")
    normalized_assignments = {}
    for sha256, assignments in all_assignments.items():
        normalized_assignments[sha256] = normalize.normalize_assignments(assignments)

    # Emit predictions and collect review queue entries
    all_review_entries = []
    emission_results = {}

    print(f"Emitting predictions for {len(indexed_docs)} documents...")

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]
        doc_id = doc_info["doc_id"]

        try:
            assignments = normalized_assignments.get(sha256, {})

            if not assignments:
                print(f"No assignments found for {doc_id}")
                continue

            predictions_path, review_entries = emit_document(sha256, assignments)
            all_review_entries.extend(review_entries)

            # Count statuses
            status_counts = {}
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
            }

            print(f"Emitted {doc_id}: {status_counts}")

        except Exception as e:
            print(f"Failed to emit {doc_id}: {e}")
            emission_results[sha256] = {
                "doc_id": doc_id,
                "error": str(e),
            }

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
        "results": emission_results,
    }

    print(
        f"Emission complete: {total_predicted} predicted, {total_abstain} abstain, {total_review_entries} review entries"
    )

    return summary


def update_review_queue(new_entries: list[dict[str, Any]]) -> None:
    """
    Update the review queue with new entries.

    Args:
        new_entries: List of review queue entries to add
    """
    if not new_entries:
        return

    review_queue_path = paths.get_review_queue_path()

    # Load existing queue if it exists
    if review_queue_path.exists():
        try:
            existing_df = pd.read_parquet(review_queue_path)
        except Exception as e:
            print(f"Warning: Could not read existing review queue: {e}")
            existing_df = pd.DataFrame()
    else:
        existing_df = pd.DataFrame()

    # Add new entries
    new_df = pd.DataFrame(new_entries)

    if existing_df.empty:
        combined_df = new_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates based on doc_id + field
    combined_df = combined_df.drop_duplicates(subset=["doc_id", "field"], keep="last")

    # Save updated queue
    combined_df.to_parquet(review_queue_path, index=False)

    print(
        f"Updated review queue: {len(new_entries)} new entries, {len(combined_df)} total"
    )


def get_review_queue() -> pd.DataFrame:
    """Get the current review queue."""
    review_queue_path = paths.get_review_queue_path()

    if not review_queue_path.exists():
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
            ]
        )

    return pd.read_parquet(review_queue_path)


def get_document_predictions(sha256: str) -> dict[str, Any] | None:
    """Get predictions for a specific document."""
    predictions_path = paths.get_predictions_path(sha256)

    if not predictions_path.exists():
        return None

    with open(predictions_path, encoding="utf-8") as f:
        return json.load(f)
