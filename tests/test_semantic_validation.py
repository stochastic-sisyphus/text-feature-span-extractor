"""Tests for semantic validation of field values.

These tests verify that the semantic validation correctly rejects garbage predictions
like repeated words, keyword-only values, and invalid field values.
"""

from invoices.validation import (
    INVOICE_KEYWORD_BLOCKLIST,
    _has_excessive_word_repetition,
    _is_keyword_only_value,
    _is_numeric_only,
    validate_assignment_semantics,
    validate_field_semantics,
)


class TestExcessiveWordRepetition:
    """Tests for detecting excessive word repetition."""

    def test_single_word_not_repetition(self):
        """Single words should not be flagged as repetition."""
        has_rep, reason = _has_excessive_word_repetition("Invoice")
        assert not has_rep
        assert reason is None

    def test_three_consecutive_repeats_rejected(self):
        """Three consecutive repeats should be rejected."""
        has_rep, reason = _has_excessive_word_repetition("Tax Tax Tax")
        assert has_rep
        assert "Tax" in reason

    def test_case_insensitive_repetition(self):
        """Repetition check should be case-insensitive."""
        has_rep, reason = _has_excessive_word_repetition("Tax TAX tax")
        assert has_rep

    def test_empty_string_ok(self):
        """Empty strings should pass."""
        has_rep, reason = _has_excessive_word_repetition("")
        assert not has_rep
        assert reason is None


class TestKeywordOnlyValue:
    """Tests for detecting keyword-only values."""

    def test_subtotal_is_keyword(self):
        """SUBTOTAL should be detected as keyword."""
        is_kw, reason = _is_keyword_only_value("SUBTOTAL", INVOICE_KEYWORD_BLOCKLIST)
        assert is_kw
        assert "keyword" in reason.lower() or "reserved" in reason.lower()

    def test_invoice_number_is_keyword(self):
        """'Invoice Number' should be detected as keyword."""
        is_kw, reason = _is_keyword_only_value(
            "Invoice Number", INVOICE_KEYWORD_BLOCKLIST
        )
        assert is_kw

    def test_actual_invoice_number_ok(self):
        """Actual invoice numbers should pass."""
        is_kw, reason = _is_keyword_only_value("INV-12345", INVOICE_KEYWORD_BLOCKLIST)
        assert not is_kw
        assert reason is None

    def test_concatenated_keywords_rejected(self):
        """Concatenated keywords like SUBTOTALTAX should be rejected."""
        is_kw, reason = _is_keyword_only_value("SUBTOTALTAX", INVOICE_KEYWORD_BLOCKLIST)
        assert is_kw


class TestNumericOnly:
    """Tests for detecting numeric-only values."""

    def test_pure_numbers_is_numeric(self):
        """Pure numbers should be detected as numeric."""
        assert _is_numeric_only("12345")

    def test_numbers_with_separators_is_numeric(self):
        """Numbers with separators should be numeric."""
        assert _is_numeric_only("12-345-678")
        assert _is_numeric_only("12,345.67")
        assert _is_numeric_only("12 345")

    def test_alphanumeric_not_numeric(self):
        """Alphanumeric should not be numeric."""
        assert not _is_numeric_only("INV-12345")


class TestInvoiceNumberValidation:
    """Tests for invoice number semantic validation."""

    def test_valid_invoice_number(self):
        """Valid invoice numbers should pass."""
        result = validate_field_semantics("InvoiceNumber", "INV-12345")
        assert result.is_valid

    def test_invalid_empty(self):
        """Empty invoice number should fail."""
        result = validate_field_semantics("InvoiceNumber", "")
        assert not result.is_valid

    def test_none_value_passes(self):
        """None values should pass (handled as MISSING)."""
        result = validate_field_semantics("InvoiceNumber", None)
        assert result.is_valid


class TestVendorNameValidation:
    """Tests for vendor/customer name semantic validation."""

    def test_valid_vendor_name(self):
        """Valid vendor names should pass."""
        result = validate_field_semantics("VendorName", "Acme Corporation")
        assert result.is_valid

    def test_invalid_numeric_only(self):
        """Numeric-only vendor names should fail."""
        result = validate_field_semantics("VendorName", "12345")
        assert not result.is_valid
        assert "numeric" in result.reason.lower() or "letter" in result.reason.lower()


class TestAmountValidation:
    """Tests for amount field semantic validation."""

    def test_valid_amount(self):
        """Valid amounts should pass."""
        result = validate_field_semantics("TotalAmount", "1234.56")
        assert result.is_valid

    def test_valid_amount_with_currency(self):
        """Amounts with currency symbols should pass."""
        result = validate_field_semantics("TotalAmount", "$1,234.56")
        assert result.is_valid

    def test_valid_amount_european(self):
        """European-style amounts should pass."""
        result = validate_field_semantics("TotalAmount", "1.234,56")
        assert result.is_valid

    def test_invalid_no_digits(self):
        """Amounts without digits should fail."""
        result = validate_field_semantics("TotalAmount", "Total Due")
        assert not result.is_valid
        assert "numeric" in result.reason.lower() or "digit" in result.reason.lower()


class TestDateValidation:
    """Tests for date field semantic validation."""

    def test_valid_date_iso(self):
        """Valid ISO dates should pass."""
        result = validate_field_semantics("InvoiceDate", "2024-01-15")
        assert result.is_valid

    def test_invalid_date_no_digits(self):
        """Dates without digits should fail."""
        result = validate_field_semantics("InvoiceDate", "January")
        assert not result.is_valid

    def test_invalid_date_feb_30(self):
        """Invalid calendar dates like Feb 30 should fail."""
        result = validate_field_semantics("InvoiceDate", "2024-02-30")
        assert not result.is_valid

    def test_invalid_year_range(self):
        """Years outside reasonable range should fail."""
        result = validate_field_semantics("InvoiceDate", "1800-01-01")
        assert not result.is_valid


class TestAssignmentSemantics:
    """Tests for the combined assignment validation."""

    def test_invalid_raw_text_repetition(self):
        """Raw text with repetition should fail even if normalized is OK."""
        result = validate_assignment_semantics(
            field_name="InvoiceNumber",
            normalized_value="12345",
            raw_text="Tax Tax Tax Tax 12345",
        )
        assert not result.is_valid
        assert "Raw text" in result.reason


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_two_word_repetition_allowed(self):
        """Two consecutive same words should be allowed."""
        has_rep, reason = _has_excessive_word_repetition("Tax Tax")
        assert not has_rep

    def test_invoice_number_with_prefix(self):
        """Invoice numbers with prefixes should pass."""
        result = validate_field_semantics("InvoiceNumber", "TAX-2024-001")
        assert result.is_valid  # TAX as part of a code is OK

    def test_special_characters_only_fails(self):
        """Values with only special characters should fail."""
        result = validate_field_semantics("InvoiceNumber", "---")
        assert not result.is_valid

    def test_na_value_rejected(self):
        """N/A should be rejected as invoice number."""
        result = validate_field_semantics("InvoiceNumber", "N/A")
        assert not result.is_valid

    def test_none_string_rejected(self):
        """'None' as string should be rejected."""
        result = validate_field_semantics("InvoiceNumber", "None")
        assert not result.is_valid
