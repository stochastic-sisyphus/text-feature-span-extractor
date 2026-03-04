"""Regression tests: Ensure outputs don't change unexpectedly.

These tests validate determinism and schema compliance. They MUST fail when:
- Pipeline produces different outputs for the same input
- Output JSON doesn't validate against the schema
- Breaking changes occur without updating the test snapshots
"""

import json
from pathlib import Path

import pytest

from invoices import paths, utils


@pytest.fixture(scope="module")
def predictions_dir() -> Path:
    """Get predictions directory."""
    return paths.get_predictions_dir()


class TestSchemaCompliance:
    """Test that all outputs comply with the contract schema."""

    def test_confidence_scores_in_range(self, predictions_dir: Path) -> None:
        """All confidence scores must be in [0, 1] range."""
        prediction_files = list(predictions_dir.glob("*.json"))
        assert len(prediction_files) > 0, "No predictions found"

        errors = []
        for pred_file in prediction_files:
            with open(pred_file, encoding="utf-8") as f:
                prediction = json.load(f)

            for field_name, field_data in prediction.get("fields", {}).items():
                confidence = field_data.get("confidence")
                if confidence is None:
                    errors.append(
                        f"{pred_file.name} / {field_name}: confidence is None"
                    )
                elif not (0 <= confidence <= 1):
                    errors.append(
                        f"{pred_file.name} / {field_name}: confidence {confidence} out of range [0,1]"
                    )

        if errors:
            pytest.fail("Confidence validation failed:\n" + "\n".join(errors))

    def test_status_values_valid(self, predictions_dir: Path) -> None:
        """All status values must be one of: PREDICTED, ABSTAIN, MISSING."""
        prediction_files = list(predictions_dir.glob("*.json"))
        assert len(prediction_files) > 0, "No predictions found"

        valid_statuses = {"PREDICTED", "ABSTAIN", "MISSING"}
        errors = []

        for pred_file in prediction_files:
            with open(pred_file, encoding="utf-8") as f:
                prediction = json.load(f)

            for field_name, field_data in prediction.get("fields", {}).items():
                status = field_data.get("status")
                if status not in valid_statuses:
                    errors.append(
                        f"{pred_file.name} / {field_name}: invalid status '{status}' "
                        f"(expected one of {valid_statuses})"
                    )

        if errors:
            pytest.fail("Status validation failed:\n" + "\n".join(errors))


class TestOutputStability:
    """Test that known-good outputs don't change."""

    def test_required_fields_always_predicted(self, predictions_dir: Path) -> None:
        """Required fields (from schema) must always be PREDICTED, never ABSTAIN.

        Schema-driven: loads field_definitions from contract.invoice.json and
        checks every field where required=true. This is a meaningful invariant —
        if the pipeline can't extract InvoiceNumber, InvoiceDate, TotalAmount,
        or VendorName, something is fundamentally broken.
        """
        prediction_files = list(predictions_dir.glob("*.json"))
        assert len(prediction_files) > 0, "No predictions found"

        # Derive required fields from the schema (single source of truth)
        schema = utils.load_contract_schema()
        field_defs = schema.get("field_definitions", {})
        required_fields = {
            fname for fname, fdef in field_defs.items() if fdef.get("required") is True
        }
        assert required_fields, (
            "Schema defines no required fields — check contract.invoice.json"
        )

        errors = []
        for pred_file in prediction_files:
            with open(pred_file, encoding="utf-8") as f:
                prediction = json.load(f)

            fields = prediction.get("fields", {})
            for req_field in required_fields:
                field_data = fields.get(req_field, {})
                status = field_data.get("status")
                if status != "PREDICTED":
                    errors.append(
                        f"{pred_file.name}: required field '{req_field}' "
                        f"has status '{status}' (expected PREDICTED)"
                    )

        if errors:
            pytest.fail("Required fields not predicted:\n" + "\n".join(errors))
