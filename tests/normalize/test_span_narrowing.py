"""Tests for _narrow_span_to_field_entity span-narrowing pre-step."""

from __future__ import annotations

import importlib.util

import pytest

import invoices.schema as schema_mod
from invoices.normalize._assignments import _narrow_span_to_field_entity

# The NER-backed narrowing needs the en_core_web_sm model, which the Docker image
# installs (`python -m spacy download en_core_web_sm`) but pyproject does not.
# Without it the code intentionally falls back to a blank pipeline (no entities).
needs_ner_model = pytest.mark.skipif(
    importlib.util.find_spec("en_core_web_sm") is None,
    reason="en_core_web_sm spaCy model not installed",
)

_FIELD_DEFS = {
    "CurrentCharges": {
        "name": "CurrentCharges",
        "base_type": "decimal",
        "normalizer": "amount",
    },
    "DueDate": {"name": "DueDate", "base_type": "date", "normalizer": "date"},
    "BillToName": {"name": "BillToName", "base_type": "str"},
}


@pytest.fixture(scope="module", autouse=True)
def seed_schema() -> None:
    """Seed the in-memory schema with a minimal self-contained contract."""
    schema_mod.set(
        {"version": 1, "fields": list(_FIELD_DEFS), "field_definitions": _FIELD_DEFS}
    )


@needs_ner_model
def test_amount_greedy_span_narrows_to_money() -> None:
    result = _narrow_span_to_field_entity(
        "CurrentCharges", "Please pay $184.97 by Dec 16,"
    )
    assert result == "184.97"


@needs_ner_model
def test_date_span_with_money_narrows_to_date() -> None:
    result = _narrow_span_to_field_entity("DueDate", "pay $184.97 by Dec 16, 2024")
    assert result == "Dec 16, 2024"


def test_name_field_greedy_span_unchanged() -> None:
    raw = "EXAMPLE HOSPITAL SERVICES INC"
    result = _narrow_span_to_field_entity("BillToName", raw)
    assert result == raw


def test_clean_atomic_amount_unchanged() -> None:
    # No internal space — short-circuits before NER.
    result = _narrow_span_to_field_entity("CurrentCharges", "$184.97")
    assert result == "$184.97"


def test_unknown_field_unchanged() -> None:
    raw = "Please pay $184.97 by Dec 16,"
    result = _narrow_span_to_field_entity("NonExistentField", raw)
    assert result == raw
