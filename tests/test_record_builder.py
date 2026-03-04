"""Tests for build_staging_record and build_production_record.

Pure function tests — no mocks, no Azure SDK.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from invoices.azure.record_builder import build_production_record, build_staging_record
from invoices.schema_registry import _load as load_schema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _schema_columns() -> dict[str, str]:
    """Return {field_name: dataverse_column} for all schema-mapped fields."""
    field_defs: dict[str, Any] = load_schema().get("field_definitions", {})
    return {
        k: v["dataverse_column"]
        for k, v in field_defs.items()
        if v.get("dataverse_column")
    }


def _predicted(value: Any, confidence: float) -> dict[str, Any]:
    return {"status": "PREDICTED", "value": value, "confidence": confidence}


def _abstain() -> dict[str, Any]:
    return {"status": "ABSTAIN"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def realistic_extraction_result() -> dict[str, Any]:
    return {
        "model_version": "test-v1",
        "fields": {
            "InvoiceNumber": _predicted("INV-2024-001", 0.95),
            "TotalAmount": _predicted("1500.00", 0.90),
            "InvoiceDate": _predicted("2024-01-15", 0.88),
            "VendorName": _predicted("ACME Corp", 0.82),
            "CustomerName": _predicted("Customer Inc", 0.91),
            "Subtotal": _abstain(),
        },
    }


@pytest.fixture
def staging(realistic_extraction_result: dict[str, Any]) -> dict[str, Any]:
    return build_staging_record(
        document_id="fs:abc123",
        extraction_result=realistic_extraction_result,
        sharepoint_id="sp-item-42",
    )


# ---------------------------------------------------------------------------
# TestBuildStagingRecord
# ---------------------------------------------------------------------------


class TestBuildStagingRecord:
    def test_metadata_keys_present(self, staging: dict[str, Any]) -> None:
        for key in (
            "document_id",
            "sharepoint_id",
            "extraction_confidence",
            "review_needed",
            "extraction_version",
            "created_at",
            "status",
        ):
            assert key in staging, f"Missing metadata key: {key}"

    def test_field_to_dataverse_column_mapping(self, staging: dict[str, Any]) -> None:
        columns = _schema_columns()
        # InvoiceNumber → invoice_number
        assert columns["InvoiceNumber"] in staging
        assert staging[columns["InvoiceNumber"]] == "INV-2024-001"
        # TotalAmount → total_amount
        assert columns["TotalAmount"] in staging
        assert staging[columns["TotalAmount"]] == "1500.00"

    def test_abstain_field_is_none(self, staging: dict[str, Any]) -> None:
        columns = _schema_columns()
        subtotal_col = columns["Subtotal"]
        assert subtotal_col in staging
        assert staging[subtotal_col] is None

    def test_missing_field_is_none(
        self, realistic_extraction_result: dict[str, Any]
    ) -> None:
        """A schema field not present in extraction at all → None in record."""
        columns = _schema_columns()
        # Remove CustomerAccount from the extraction entirely (it's not in fixture)
        assert "CustomerAccount" not in realistic_extraction_result["fields"]
        record = build_staging_record(
            document_id="fs:xyz",
            extraction_result=realistic_extraction_result,
        )
        customer_account_col = columns.get("CustomerAccount")
        if customer_account_col:
            assert record[customer_account_col] is None

    def test_status_is_pending(self, staging: dict[str, Any]) -> None:
        assert staging["status"] == "pending"

    def test_review_needed_true_when_low_confidence(self) -> None:
        extraction: dict[str, Any] = {
            "model_version": "test-v1",
            "fields": {
                "InvoiceNumber": _predicted("INV-001", 0.70),
                "TotalAmount": _predicted("500.00", 0.95),
            },
        }
        record = build_staging_record(
            document_id="fs:low",
            extraction_result=extraction,
        )
        # min confidence 0.70 < Config.CONFIDENCE_AUTO_APPROVE
        assert record["review_needed"] is True

    def test_review_needed_false_when_all_high_confidence(self) -> None:
        extraction: dict[str, Any] = {
            "model_version": "test-v1",
            "fields": {
                "InvoiceNumber": _predicted("INV-001", 0.95),
                "TotalAmount": _predicted("500.00", 0.97),
                "VendorName": _predicted("Corp", 0.96),
            },
        }
        record = build_staging_record(
            document_id="fs:high",
            extraction_result=extraction,
        )
        # all confidences >= Config.CONFIDENCE_AUTO_APPROVE (0.85)
        assert record["review_needed"] is False

    def test_extraction_confidence_is_minimum_not_average(self) -> None:
        extraction: dict[str, Any] = {
            "model_version": "test-v1",
            "fields": {
                "InvoiceNumber": _predicted("INV-001", 0.95),
                "TotalAmount": _predicted("500.00", 0.60),
                "VendorName": _predicted("Corp", 0.90),
            },
        }
        record = build_staging_record(
            document_id="fs:mintest",
            extraction_result=extraction,
        )
        # min is 0.60, average would be ~0.817
        assert record["extraction_confidence"] == pytest.approx(0.60)

    def test_extraction_confidence_zero_on_empty_fields(self) -> None:
        extraction: dict[str, Any] = {"model_version": "test-v1", "fields": {}}
        record = build_staging_record(
            document_id="fs:empty",
            extraction_result=extraction,
        )
        assert record["extraction_confidence"] == 0.0

    def test_sharepoint_id_none_when_not_passed(
        self, realistic_extraction_result: dict[str, Any]
    ) -> None:
        record = build_staging_record(
            document_id="fs:noshare",
            extraction_result=realistic_extraction_result,
        )
        assert record["sharepoint_id"] is None

    def test_only_schema_mapped_fields_appear(
        self,
        staging: dict[str, Any],
        realistic_extraction_result: dict[str, Any],
    ) -> None:
        """Fields in schema without dataverse_column must not appear in record."""
        schema = load_schema()
        field_defs: dict[str, Any] = schema.get("field_definitions", {})
        unmapped_field_names = {
            k for k, v in field_defs.items() if not v.get("dataverse_column")
        }
        # None of the unmapped field names (not column names) should be keys
        for name in unmapped_field_names:
            assert name not in staging

    def test_created_at_is_iso8601_utc(self, staging: dict[str, Any]) -> None:
        created_at: str = staging["created_at"]
        # Must be parseable
        parsed = datetime.fromisoformat(created_at)
        # Must be UTC (offset == 0)
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestBuildProductionRecord
# ---------------------------------------------------------------------------


class TestBuildProductionRecord:
    def test_required_metadata_keys_present(self, staging: dict[str, Any]) -> None:
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-001",
            approved_by="vanessa@example.com",
        )
        for key in ("staging_id", "approved_by", "approved_at", "source_document"):
            assert key in prod, f"Missing metadata key: {key}"

    def test_copies_dataverse_columns_from_staging(
        self, staging: dict[str, Any]
    ) -> None:
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-001",
            approved_by="vanessa@example.com",
        )
        columns = _schema_columns()
        assert prod[columns["InvoiceNumber"]] == "INV-2024-001"
        assert prod[columns["TotalAmount"]] == "1500.00"

    def test_invoice_number_defaults_to_empty_string_when_missing(self) -> None:
        columns = _schema_columns()
        # Staging record without invoice_number
        sparse_staging: dict[str, Any] = {"sharepoint_id": "sp-999"}
        prod = build_production_record(
            staging_record=sparse_staging,
            staging_id="stg-002",
            approved_by="reviewer",
        )
        assert prod[columns["InvoiceNumber"]] == ""

    def test_other_fields_default_to_none_when_missing(self) -> None:
        columns = _schema_columns()
        sparse_staging: dict[str, Any] = {"sharepoint_id": "sp-999"}
        prod = build_production_record(
            staging_record=sparse_staging,
            staging_id="stg-002",
            approved_by="reviewer",
        )
        # TotalAmount missing from staging → None in prod
        assert prod[columns["TotalAmount"]] is None

    def test_approved_by_preserved(self, staging: dict[str, Any]) -> None:
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-001",
            approved_by="qa-team",
        )
        assert prod["approved_by"] == "qa-team"

    def test_staging_id_preserved(self, staging: dict[str, Any]) -> None:
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-XYZ-999",
            approved_by="reviewer",
        )
        assert prod["staging_id"] == "stg-XYZ-999"

    def test_approved_at_is_iso8601_utc(self, staging: dict[str, Any]) -> None:
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-001",
            approved_by="reviewer",
        )
        approved_at: str = prod["approved_at"]
        parsed = datetime.fromisoformat(approved_at)
        assert parsed.utcoffset() is not None
        assert parsed.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# TestRecordRoundtrip
# ---------------------------------------------------------------------------


class TestRecordRoundtrip:
    def test_key_fields_survive_staging_to_production(
        self, realistic_extraction_result: dict[str, Any]
    ) -> None:
        staging = build_staging_record(
            document_id="fs:roundtrip",
            extraction_result=realistic_extraction_result,
            sharepoint_id="sp-roundtrip",
        )
        prod = build_production_record(
            staging_record=staging,
            staging_id="stg-rt-001",
            approved_by="roundtrip-reviewer",
        )
        columns = _schema_columns()
        assert prod[columns["InvoiceNumber"]] == "INV-2024-001"
        assert prod[columns["TotalAmount"]] == "1500.00"
        assert prod[columns["VendorName"]] == "ACME Corp"
        assert prod[columns["Subtotal"]] is None  # was ABSTAIN
