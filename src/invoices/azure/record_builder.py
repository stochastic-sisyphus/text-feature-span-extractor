"""Dynamic Dataverse record builder.

Builds staging and production records from extraction results using
schema-driven field-to-column mappings instead of hardcoded dicts.
"""

from datetime import datetime, timezone
from typing import Any

from invoices.config import Config


def build_staging_record(
    document_id: str,
    extraction_result: dict[str, Any],
    sharepoint_id: str | None = None,
) -> dict[str, Any]:
    """Build a staging record from extraction results.

    Reads field definitions from schema and maps only fields that have
    a dataverse_column defined.

    Args:
        document_id: Internal document ID (fs:...)
        extraction_result: Pipeline extraction result
        sharepoint_id: Source SharePoint item ID

    Returns:
        Staging record dict ready for Dataverse create()
    """
    fields = extraction_result.get("fields", {})

    # Helper to extract field values
    def get_field_value(name: str) -> Any:
        field_data = fields.get(name, {})
        if field_data.get("status") == "PREDICTED":
            return field_data.get("value")
        return None

    # Calculate minimum confidence
    confidences = [
        f.get("confidence", 0.0)
        for f in fields.values()
        if f.get("status") == "PREDICTED"
    ]
    min_confidence = min(confidences) if confidences else 0.0

    # Start with non-field metadata columns
    record: dict[str, Any] = {
        "document_id": document_id,
        "sharepoint_id": sharepoint_id,
        "extraction_confidence": min_confidence,
        "review_needed": min_confidence < Config.CONFIDENCE_AUTO_APPROVE,
        "extraction_version": extraction_result.get("model_version", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    # Dynamically add all fields that have a dataverse_column mapping
    # This reads from schema/contract.invoice.json field_definitions
    from invoices.schema_registry import _load

    schema = _load()
    field_defs = schema.get("field_definitions", {})

    for field_name, field_def in field_defs.items():
        column_name = field_def.get("dataverse_column")
        if column_name:
            # Map field_name → column_name with extracted value
            record[column_name] = get_field_value(field_name)

    return record


def build_production_record(
    staging_record: dict[str, Any],
    staging_id: str,
    approved_by: str,
) -> dict[str, Any]:
    """Build a production record from a staging record.

    Copies only the fields that have dataverse_column mappings,
    plus production-specific metadata.

    Args:
        staging_record: The staging record to promote
        staging_id: Staging record ID
        approved_by: Approver identifier

    Returns:
        Production record dict ready for Dataverse create()
    """
    # Start with production metadata
    record: dict[str, Any] = {
        "staging_id": staging_id,
        "approved_by": approved_by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "source_document": staging_record.get("sharepoint_id"),
    }

    # Dynamically copy all dataverse_column fields from staging
    from invoices.schema_registry import _load

    schema = _load()
    field_defs = schema.get("field_definitions", {})

    for field_def in field_defs.values():
        column_name = field_def.get("dataverse_column")
        if column_name:
            # Copy column from staging record
            # Special handling for invoice_number (required field, default to empty string)
            if column_name == "invoice_number":
                record[column_name] = staging_record.get(column_name, "")
            else:
                record[column_name] = staging_record.get(column_name)

    return record
