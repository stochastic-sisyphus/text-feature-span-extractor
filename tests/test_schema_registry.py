"""Tests for schema_registry module."""

from __future__ import annotations

from invoices import schema_registry


def test_field_type():
    """Test field_type returns correct types."""
    assert schema_registry.field_type("TotalAmount") == "amount"
    assert schema_registry.field_type("VendorName") == "text"
    assert schema_registry.field_type("InvoiceDate") == "date"
    assert schema_registry.field_type("InvoiceNumber") == "id"
    assert schema_registry.field_type("Currency") == "currency"
    assert schema_registry.field_type("UnknownField") == "text"  # default


def test_anchor_type():
    """Test anchor_type returns correct anchor mappings."""
    # Explicit override to "tax"
    assert schema_registry.anchor_type("TaxAmount") == "tax"

    # Explicit override to "total"
    assert schema_registry.anchor_type("Subtotal") == "total"

    # Type default (amount -> total)
    assert schema_registry.anchor_type("TotalAmount") == "total"

    # Explicit null override (even though type=text would default to "name")
    assert schema_registry.anchor_type("VendorName") is None

    # Type default (date -> date)
    assert schema_registry.anchor_type("InvoiceDate") == "date"

    # Type default (id -> id)
    assert schema_registry.anchor_type("InvoiceNumber") == "id"


def test_spatial_region():
    """Test is_footer and is_header return correct values."""
    # Footer fields
    assert schema_registry.is_footer("TotalAmount") is True
    assert schema_registry.is_footer("Subtotal") is True
    assert schema_registry.is_footer("TaxAmount") is True
    assert schema_registry.is_footer("Discount") is True

    # Header fields
    assert schema_registry.is_header("VendorName") is True

    # No spatial_region specified
    assert schema_registry.is_footer("InvoiceNumber") is False
    assert schema_registry.is_header("InvoiceNumber") is False


def test_priority_bonus():
    """Test priority_bonus returns correct values."""
    assert schema_registry.priority_bonus("InvoiceNumber") == 0.5
    assert schema_registry.priority_bonus("TotalAmount") == 0.02
    assert schema_registry.priority_bonus("InvoiceDate") == 0.01
    assert schema_registry.priority_bonus("IssueDate") == 0.01
    assert schema_registry.priority_bonus("VendorName") == 0.5


def test_importance():
    """Test importance returns correct values."""
    assert schema_registry.importance("InvoiceNumber") == 1.0
    assert schema_registry.importance("TotalAmount") == 0.95
    assert schema_registry.importance("InvoiceDate") == 0.85
    assert schema_registry.importance("DueDate") == 0.80
    assert schema_registry.importance("VendorName") == 0.75
    assert schema_registry.importance("CustomerName") == 0.55
    assert schema_registry.importance("CustomerAccount") == 0.45
    assert schema_registry.importance("IssueDate") == 0.40
    assert schema_registry.importance("Notes") == 0.5  # default


def test_confidence_threshold():
    """Test confidence_threshold returns correct values."""
    assert schema_registry.confidence_threshold("InvoiceNumber") == 0.90
    assert schema_registry.confidence_threshold("TotalAmount") == 0.95
    assert schema_registry.confidence_threshold("InvoiceDate") == 0.85


def test_is_keyword_proximal():
    """Test is_keyword_proximal returns correct values."""
    # Type defaults (id, amount, date -> True)
    assert schema_registry.is_keyword_proximal("InvoiceNumber") is True
    assert schema_registry.is_keyword_proximal("TotalAmount") is True
    assert schema_registry.is_keyword_proximal("InvoiceDate") is True

    # Explicit False
    assert schema_registry.is_keyword_proximal("VendorName") is False
    assert schema_registry.is_keyword_proximal("CustomerName") is False
    assert schema_registry.is_keyword_proximal("BillingReference") is False
    assert schema_registry.is_keyword_proximal("VendorAddress") is False
    assert schema_registry.is_keyword_proximal("RemittanceAddress") is False
    assert schema_registry.is_keyword_proximal("BillToAddress") is False
    assert schema_registry.is_keyword_proximal("ShipToAddress") is False


def test_computed_fields():
    """Test computed_fields returns correct list."""
    computed = schema_registry.computed_fields()
    assert "BillingReference" in computed
    assert "Currency" in computed
    assert len(computed) == 2


def test_dataverse_column():
    """Test dataverse_column returns correct mappings."""
    assert schema_registry.dataverse_column("InvoiceNumber") == "invoice_number"
    assert schema_registry.dataverse_column("InvoiceDate") == "invoice_date"
    assert schema_registry.dataverse_column("DueDate") == "due_date"
    assert schema_registry.dataverse_column("TotalAmount") == "total_amount"
    assert schema_registry.dataverse_column("Subtotal") == "subtotal"
    assert schema_registry.dataverse_column("TaxAmount") == "tax_amount"
    assert schema_registry.dataverse_column("VendorName") == "vendor_name"
    assert schema_registry.dataverse_column("CustomerName") == "customer_name"
    assert schema_registry.dataverse_column("CustomerAccount") == "customer_account"
    assert schema_registry.dataverse_column("Notes") is None


def test_fields_by_type():
    """Test fields_by_type returns correct field lists."""
    amount_fields = schema_registry.fields_by_type("amount")
    assert "TotalAmount" in amount_fields
    assert "Subtotal" in amount_fields
    assert "TaxAmount" in amount_fields
    assert "Discount" in amount_fields

    date_fields = schema_registry.fields_by_type("date")
    assert "InvoiceDate" in date_fields
    assert "DueDate" in date_fields
    assert "IssueDate" in date_fields


def test_name_fields():
    """Test name_fields returns text fields with fuzzy name matching."""
    name_fields = schema_registry.name_fields()
    # Should return text fields with keyword_proximal=false
    assert "VendorName" in name_fields
    assert "CustomerName" in name_fields
    # Should only be these two
    assert len(name_fields) == 2
    # Should not include other text fields
    assert "ContactName" not in name_fields
    assert "PaymentTerms" not in name_fields
