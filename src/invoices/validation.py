"""Contract output validation module.

Validates extraction output against the contract schema to ensure
data quality and schema compliance before downstream processing.

Semantic Validation:
    from invoices.validation import validate_field_semantics, SemanticValidationResult

    result = validate_field_semantics("InvoiceNumber", "Tax Tax Tax Tax")
    if not result.is_valid:
        # Value should be rejected - change status to ABSTAIN
        print(f"Invalid: {result.reason}")

Schema Structure Validation:
    from invoices.validation import validate_schema_structure

    result = validate_schema_structure()
    if not result.is_valid:
        for error in result.errors:
            print(f"Schema error: {error}")
"""

import calendar
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from . import schema_registry, utils
from .config import Config

# Semver pattern: major.minor.patch with optional prerelease/build metadata
SEMVER_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

# Valid field types defined in the schema
VALID_FIELD_TYPES = {
    "id",
    "date",
    "amount",
    "text",
    "email",
    "phone",
    "currency",
    "address",
    "number",
}

# =============================================================================
# SEMANTIC VALIDATION CONSTANTS
# =============================================================================

# Keywords that should NOT be accepted as invoice numbers or IDs
# These are typically column headers or labels, not actual values
INVOICE_KEYWORD_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Common invoice section headers
        "subtotal",
        "sub-total",
        "sub total",
        "total",
        "total due",
        "amount due",
        "balance due",
        "grand total",
        "tax",
        "taxes",
        "vat",
        "gst",
        "hst",
        "pst",
        "sales tax",
        "discount",
        "discounts",
        "shipping",
        "freight",
        "handling",
        "payment",
        "payments",
        "paid",
        "balance",
        "quantity",
        "qty",
        "price",
        "unit price",
        "amount",
        "description",
        "item",
        "items",
        "product",
        "service",
        # Invoice labels
        "invoice",
        "invoice number",
        "invoice no",
        "invoice#",
        "inv#",
        "inv",
        "bill",
        "bill to",
        "billed to",
        "billing",
        "date",
        "due date",
        "invoice date",
        "issue date",
        "account",
        "account number",
        "account#",
        "acct",
        "customer",
        "vendor",
        "supplier",
        "client",
        "po",
        "po#",
        "purchase order",
        "order",
        "order#",
        "reference",
        "ref",
        "ref#",
        # Common garbage patterns
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "--",
        "---",
    }
)

# Keywords that should NOT be accepted as vendor/customer names
NAME_KEYWORD_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Section headers and labels
        "subtotal",
        "sub-total",
        "sub total",
        "total",
        "total due",
        "amount due",
        "balance due",
        "grand total",
        "tax",
        "taxes",
        "vat",
        "gst",
        "sales tax",
        "discount",
        "shipping",
        "freight",
        "quantity",
        "qty",
        "price",
        "unit price",
        "amount",
        "description",
        "item",
        "items",
        "product",
        "service",
        # Invoice labels
        "invoice",
        "invoice number",
        "bill",
        "billing",
        "date",
        "due date",
        "invoice date",
        "account",
        "account number",
        "po",
        "purchase order",
        "order",
        "reference",
        "ref",
        # Common garbage
        "n/a",
        "na",
        "none",
        "null",
        "-",
        "--",
    }
)

# Maximum allowed ratio of repeated words (prevents "Tax Tax Tax Tax")
MAX_WORD_REPETITION_RATIO: float = 0.5  # If >50% of words are the same, reject

# Minimum unique words required for multi-word values
MIN_UNIQUE_WORDS_RATIO: float = 0.4  # At least 40% of words must be unique

# Maximum consecutive repeated words allowed
MAX_CONSECUTIVE_REPEATS: int = 2


@dataclass
class SemanticValidationResult:
    """Result of semantic validation for a field value."""

    is_valid: bool
    reason: str | None = None
    field_name: str | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "reason": self.reason,
            "field_name": self.field_name,
            "value": self.value,
        }


def _has_excessive_word_repetition(text: str) -> tuple[bool, str | None]:
    """Check if text has excessive word repetition.

    Detects patterns like "Tax Tax Tax Tax" or "TOTAL TOTAL TOTAL".

    Args:
        text: Text to check

    Returns:
        Tuple of (has_repetition, reason_if_invalid)
    """
    if not text or not text.strip():
        return False, None

    # Split into words
    words = text.split()

    if len(words) <= 1:
        return False, None

    # Check for consecutive repeats
    consecutive_count = 1
    max_consecutive = 1
    repeated_word = None

    for i in range(1, len(words)):
        if words[i].lower() == words[i - 1].lower():
            consecutive_count += 1
            if consecutive_count > max_consecutive:
                max_consecutive = consecutive_count
                repeated_word = words[i]
        else:
            consecutive_count = 1

    if max_consecutive > MAX_CONSECUTIVE_REPEATS:
        return (
            True,
            f"Excessive consecutive repetition of '{repeated_word}' ({max_consecutive} times)",
        )

    # Check overall word frequency
    word_counts: dict[str, int] = {}
    for word in words:
        word_lower = word.lower()
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

    # Find most frequent word
    max_count = max(word_counts.values())
    max_word = [w for w, c in word_counts.items() if c == max_count][0]
    repetition_ratio = max_count / len(words)

    if repetition_ratio > MAX_WORD_REPETITION_RATIO and len(words) > 2:
        return (
            True,
            f"Word '{max_word}' repeated too often ({max_count}/{len(words)} = {repetition_ratio:.0%})",
        )

    # Check unique word ratio
    unique_ratio = len(word_counts) / len(words)
    if unique_ratio < MIN_UNIQUE_WORDS_RATIO and len(words) > 3:
        return (
            True,
            f"Too few unique words ({len(word_counts)}/{len(words)} = {unique_ratio:.0%})",
        )

    return False, None


def _is_keyword_only_value(
    text: str, blocklist: frozenset[str]
) -> tuple[bool, str | None]:
    """Check if text is just a keyword from the blocklist.

    Args:
        text: Text to check
        blocklist: Set of blocked keywords

    Returns:
        Tuple of (is_keyword, reason_if_invalid)
    """
    if not text or not text.strip():
        return False, None

    # Normalize text for comparison
    normalized = text.strip().lower()

    # Direct match
    if normalized in blocklist:
        return True, f"Value '{text}' is a reserved keyword/label"

    # Check if it's just keywords concatenated (e.g., "SUBTOTALTAX")
    # Only check for values that are all uppercase and might be concatenated keywords
    if text.isupper() and len(text) > 5:
        text_lower = text.lower()
        for keyword in blocklist:
            keyword_nospace = keyword.replace(" ", "").replace("-", "")
            if keyword_nospace in text_lower and len(keyword_nospace) >= 4:
                # Check if the value is mostly this keyword repeated/concatenated
                if len(text_lower) <= len(keyword_nospace) * 2:
                    return True, f"Value '{text}' appears to be concatenated keywords"

    return False, None


def _is_numeric_only(text: str) -> bool:
    """Check if text contains only numeric characters (and separators)."""
    if not text or not text.strip():
        return False

    # Remove common separators
    cleaned = text.replace(" ", "").replace("-", "").replace(".", "").replace(",", "")
    return cleaned.isdigit()


def _validate_invoice_number_semantics(value: str) -> SemanticValidationResult:
    """Validate that invoice number makes semantic sense.

    Rules:
    - Must contain at least one alphanumeric character
    - Cannot be just a keyword (SUBTOTAL, TAX, TOTAL, etc.)
    - Cannot have excessive word repetition
    - Cannot be just whitespace or punctuation

    Args:
        value: The invoice number value to validate

    Returns:
        SemanticValidationResult
    """
    if not value or not value.strip():
        return SemanticValidationResult(
            is_valid=False,
            reason="Invoice number is empty or whitespace",
            field_name="InvoiceNumber",
            value=value,
        )

    clean_value = value.strip()

    # Check for excessive repetition
    has_repetition, reason = _has_excessive_word_repetition(clean_value)
    if has_repetition:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name="InvoiceNumber",
            value=value,
        )

    # Check if it's just a keyword
    is_keyword, reason = _is_keyword_only_value(clean_value, INVOICE_KEYWORD_BLOCKLIST)
    if is_keyword:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name="InvoiceNumber",
            value=value,
        )

    # Invoice numbers should be concise - reject multi-word garbage
    # Real invoice numbers are typically 1-2 tokens (e.g., "INV-123", "12345")
    # Reject things like "pay $353.10 by Aug" (4 words)
    words = clean_value.split()
    if len(words) >= 4:
        return SemanticValidationResult(
            is_valid=False,
            reason="Invoice number cannot be a multi-word phrase",
            field_name="InvoiceNumber",
            value=value,
        )

    # Must contain at least one alphanumeric character
    if not any(c.isalnum() for c in clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason="Invoice number must contain at least one alphanumeric character",
            field_name="InvoiceNumber",
            value=value,
        )

    return SemanticValidationResult(
        is_valid=True, field_name="InvoiceNumber", value=value
    )


def _validate_name_semantics(value: str, field_name: str) -> SemanticValidationResult:
    """Validate that vendor/customer name makes semantic sense.

    Rules:
    - Cannot be numeric-only
    - Cannot be all-caps keywords (SUBTOTAL, TAX, etc.)
    - Cannot have excessive word repetition
    - Must have reasonable length

    Args:
        value: The name value to validate
        field_name: Field name (VendorName, CustomerName, etc.)

    Returns:
        SemanticValidationResult
    """
    if not value or not value.strip():
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is empty or whitespace",
            field_name=field_name,
            value=value,
        )

    clean_value = value.strip()

    # Check for excessive repetition
    has_repetition, reason = _has_excessive_word_repetition(clean_value)
    if has_repetition:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name=field_name,
            value=value,
        )

    # Check if it's just a keyword
    is_keyword, reason = _is_keyword_only_value(clean_value, NAME_KEYWORD_BLOCKLIST)
    if is_keyword:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name=field_name,
            value=value,
        )

    # Cannot be purely numeric (names should have letters)
    if _is_numeric_only(clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} cannot be purely numeric",
            field_name=field_name,
            value=value,
        )

    # Must have at least one letter
    if not any(c.isalpha() for c in clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} must contain at least one letter",
            field_name=field_name,
            value=value,
        )

    return SemanticValidationResult(is_valid=True, field_name=field_name, value=value)


def _validate_amount_semantics(value: str, field_name: str) -> SemanticValidationResult:
    """Validate that amount makes semantic sense.

    Rules:
    - Must be parseable as a valid decimal number
    - Cannot be a keyword masquerading as an amount

    Args:
        value: The amount value to validate
        field_name: Field name (TotalAmount, Subtotal, etc.)

    Returns:
        SemanticValidationResult
    """
    if not value or not value.strip():
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is empty or whitespace",
            field_name=field_name,
            value=value,
        )

    clean_value = value.strip()

    # Check if it looks like a keyword (no digits at all)
    if not any(c.isdigit() for c in clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} must contain numeric digits",
            field_name=field_name,
            value=value,
        )

    # Try to parse as decimal
    # Remove currency symbols and codes first
    numeric_text = clean_value
    for symbol in Config.CURRENCY_SYMBOLS:
        numeric_text = numeric_text.replace(symbol, "")
    for code in Config.CURRENCY_CODES:
        numeric_text = numeric_text.replace(code, "")

    # Clean up and try to parse
    numeric_text = numeric_text.strip()

    # Handle comma as thousands separator or decimal separator
    if "," in numeric_text and "." in numeric_text:
        # Assume comma is thousands separator if it comes before dot
        if numeric_text.rindex(",") < numeric_text.rindex("."):
            numeric_text = numeric_text.replace(",", "")
    elif "," in numeric_text:
        parts = numeric_text.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            # Likely decimal separator (European style)
            numeric_text = numeric_text.replace(",", ".")
        else:
            # Likely thousands separator
            numeric_text = numeric_text.replace(",", "")

    # Remove any remaining non-numeric chars except decimal point and minus
    numeric_chars = []
    for char in numeric_text:
        if char.isdigit() or char in ".-":
            numeric_chars.append(char)
    numeric_text = "".join(numeric_chars)

    if not numeric_text:
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} has no parseable numeric value",
            field_name=field_name,
            value=value,
        )

    try:
        Decimal(numeric_text)
    except InvalidOperation:
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is not a valid decimal number",
            field_name=field_name,
            value=value,
        )

    return SemanticValidationResult(is_valid=True, field_name=field_name, value=value)


def _validate_date_semantics(value: str, field_name: str) -> SemanticValidationResult:
    """Validate that date makes semantic sense.

    Rules:
    - Must be a valid calendar date (not Feb 30, etc.)
    - Cannot be a keyword or garbage text

    Args:
        value: The date value to validate (expected in ISO8601 format YYYY-MM-DD)
        field_name: Field name (InvoiceDate, DueDate, etc.)

    Returns:
        SemanticValidationResult
    """
    if not value or not value.strip():
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is empty or whitespace",
            field_name=field_name,
            value=value,
        )

    clean_value = value.strip()

    # Check if it looks like a date (has digits)
    if not any(c.isdigit() for c in clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} must contain numeric date components",
            field_name=field_name,
            value=value,
        )

    # Try to parse ISO8601 format
    try:
        parsed = datetime.fromisoformat(clean_value)

        # Validate it's a real calendar date
        # datetime.fromisoformat already validates this, but let's be explicit
        year = parsed.year
        month = parsed.month
        day = parsed.day

        # Check reasonable year range (not year 1 or year 9999)
        if year < 1900 or year > 2100:
            return SemanticValidationResult(
                is_valid=False,
                reason=f"{field_name} has an unreasonable year: {year}",
                field_name=field_name,
                value=value,
            )

        # Verify day is valid for the month
        max_day = calendar.monthrange(year, month)[1]
        if day > max_day:
            return SemanticValidationResult(
                is_valid=False,
                reason=f"{field_name} has invalid day {day} for month {month}",
                field_name=field_name,
                value=value,
            )

    except ValueError as e:
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is not a valid date: {e}",
            field_name=field_name,
            value=value,
        )

    return SemanticValidationResult(is_valid=True, field_name=field_name, value=value)


def _validate_id_semantics(value: str, field_name: str) -> SemanticValidationResult:
    """Validate that ID field makes semantic sense.

    Rules:
    - Must contain at least one alphanumeric character
    - Cannot be just a keyword
    - Cannot have excessive word repetition

    Args:
        value: The ID value to validate
        field_name: Field name (CustomerAccount, PurchaseOrder, etc.)

    Returns:
        SemanticValidationResult
    """
    if not value or not value.strip():
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} is empty or whitespace",
            field_name=field_name,
            value=value,
        )

    clean_value = value.strip()

    # Check for excessive repetition
    has_repetition, reason = _has_excessive_word_repetition(clean_value)
    if has_repetition:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name=field_name,
            value=value,
        )

    # Check if it's just a keyword
    is_keyword, reason = _is_keyword_only_value(clean_value, INVOICE_KEYWORD_BLOCKLIST)
    if is_keyword:
        return SemanticValidationResult(
            is_valid=False,
            reason=reason,
            field_name=field_name,
            value=value,
        )

    # Must contain at least one alphanumeric character
    if not any(c.isalnum() for c in clean_value):
        return SemanticValidationResult(
            is_valid=False,
            reason=f"{field_name} must contain at least one alphanumeric character",
            field_name=field_name,
            value=value,
        )

    return SemanticValidationResult(is_valid=True, field_name=field_name, value=value)


def validate_field_semantics(
    field_name: str,
    value: str | None,
    field_type: str | None = None,
) -> SemanticValidationResult:
    """Validate that a field value makes semantic sense for its field type.

    This is the main entry point for semantic validation. It routes to
    type-specific validation based on the field name or explicit field type.

    Args:
        field_name: Name of the field (e.g., "InvoiceNumber", "VendorName")
        value: The value to validate
        field_type: Optional explicit field type override ("id", "date", "amount", "text")

    Returns:
        SemanticValidationResult indicating whether the value is semantically valid
    """
    if value is None:
        return SemanticValidationResult(
            is_valid=True, field_name=field_name, value=None
        )

    if not isinstance(value, str):
        value = str(value)

    # Determine field type from field name if not provided
    if field_type is None:
        field_type = schema_registry.field_type(field_name)

    # Route to type-specific validation
    if field_type == "id":
        if (
            "invoicenumber" in field_name.lower()
            or "invoice_number" in field_name.lower()
        ):
            return _validate_invoice_number_semantics(value)
        else:
            return _validate_id_semantics(value, field_name)
    elif field_type == "date":
        return _validate_date_semantics(value, field_name)
    elif field_type == "amount":
        return _validate_amount_semantics(value, field_name)
    elif field_type == "text":
        # For text fields that are names, apply name validation
        if any(x in field_name.lower() for x in ["name", "vendor", "customer"]):
            return _validate_name_semantics(value, field_name)
        # For other text fields, just check for repetition
        has_repetition, reason = _has_excessive_word_repetition(value)
        if has_repetition:
            return SemanticValidationResult(
                is_valid=False,
                reason=reason,
                field_name=field_name,
                value=value,
            )
        return SemanticValidationResult(
            is_valid=True, field_name=field_name, value=value
        )
    else:
        # Unknown type - just check for obvious garbage
        has_repetition, reason = _has_excessive_word_repetition(value)
        if has_repetition:
            return SemanticValidationResult(
                is_valid=False,
                reason=reason,
                field_name=field_name,
                value=value,
            )
        return SemanticValidationResult(
            is_valid=True, field_name=field_name, value=value
        )


def validate_assignment_semantics(
    field_name: str,
    normalized_value: str | None,
    raw_text: str | None,
    field_type: str | None = None,
) -> SemanticValidationResult:
    """Validate both normalized value and raw text for semantic validity.

    This checks both the normalized value (if available) and the raw text
    to catch garbage that might slip through normalization.

    Args:
        field_name: Name of the field
        normalized_value: The normalized value (may be None if normalization failed)
        raw_text: The original raw text
        field_type: Optional explicit field type

    Returns:
        SemanticValidationResult
    """
    # First check the normalized value if present
    if normalized_value is not None:
        result = validate_field_semantics(field_name, normalized_value, field_type)
        if not result.is_valid:
            return result

    # Then check the raw text for garbage patterns
    # This catches cases where normalization succeeded but the underlying value is garbage
    if raw_text is not None and raw_text.strip():
        # Check for excessive repetition in raw text
        has_repetition, reason = _has_excessive_word_repetition(raw_text)
        if has_repetition:
            return SemanticValidationResult(
                is_valid=False,
                reason=f"Raw text has issues: {reason}",
                field_name=field_name,
                value=raw_text,
            )

        # For ID and name fields, also check if raw text is just keywords
        field_name_lower = field_name.lower()
        if any(x in field_name_lower for x in ["number", "account", "id", "order"]):
            is_keyword, reason = _is_keyword_only_value(
                raw_text, INVOICE_KEYWORD_BLOCKLIST
            )
            if is_keyword:
                return SemanticValidationResult(
                    is_valid=False,
                    reason=f"Raw text is invalid: {reason}",
                    field_name=field_name,
                    value=raw_text,
                )
        elif any(x in field_name_lower for x in ["name", "vendor", "customer"]):
            is_keyword, reason = _is_keyword_only_value(
                raw_text, NAME_KEYWORD_BLOCKLIST
            )
            if is_keyword:
                return SemanticValidationResult(
                    is_valid=False,
                    reason=f"Raw text is invalid: {reason}",
                    field_name=field_name,
                    value=raw_text,
                )

    return SemanticValidationResult(
        is_valid=True, field_name=field_name, value=normalized_value
    )


# =============================================================================
# CROSS-FIELD CONSISTENCY VALIDATION
# =============================================================================


@dataclass
class CrossFieldValidationResult:
    """Result of cross-field consistency validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error (fails validation)."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def _parse_amount_for_crossfield(value: Any) -> float | None:
    """Parse an amount value to float for cross-field validation."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            cleaned = value
            for sym in Config.CURRENCY_SYMBOLS:
                cleaned = cleaned.replace(sym, "")
            cleaned = cleaned.replace(",", "").strip()
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _parse_date_for_crossfield(value: Any) -> datetime | None:
    """Parse a date value to datetime for cross-field validation."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None
    return None


def _apply_cross_field_rules(
    fields: dict[str, Any],
) -> CrossFieldValidationResult:
    """Apply cross-field rules from schema (no hardcoded field names)."""
    from . import schema_registry

    result = CrossFieldValidationResult(is_valid=True)

    for rule in schema_registry.cross_field_rules():
        rule_type = rule["type"]
        field_name = rule["field"]
        ref_name = rule["reference"]
        template = rule.get("message_template", "")

        field_data = fields.get(field_name, {})
        ref_data = fields.get(ref_name, {})

        # Both fields must be PREDICTED to compare
        if field_data.get("status") != "PREDICTED":
            continue
        if ref_data.get("status") != "PREDICTED":
            continue

        if rule_type == "lte":
            field_val = _parse_amount_for_crossfield(field_data.get("value"))
            ref_val = _parse_amount_for_crossfield(ref_data.get("value"))
            if field_val is None or ref_val is None:
                continue
            tolerance = rule.get("tolerance", 0.0)
            if field_val > ref_val * (1.0 + tolerance):
                result.add_warning(
                    template.format(
                        field=field_name,
                        field_val=field_val,
                        reference=ref_name,
                        ref_val=ref_val,
                    )
                )

        elif rule_type == "date_gte":
            field_dt = _parse_date_for_crossfield(field_data.get("value"))
            ref_dt = _parse_date_for_crossfield(ref_data.get("value"))
            if field_dt is None or ref_dt is None:
                continue
            if field_dt < ref_dt:
                result.add_warning(
                    template.format(
                        field=field_name,
                        field_val=field_dt.date(),
                        reference=ref_name,
                        ref_val=ref_dt.date(),
                    )
                )

        elif rule_type == "date_max_gap_days":
            field_dt = _parse_date_for_crossfield(field_data.get("value"))
            ref_dt = _parse_date_for_crossfield(ref_data.get("value"))
            if field_dt is None or ref_dt is None:
                continue
            days = (field_dt - ref_dt).days
            max_days = rule.get("max_days", 90)
            if days > max_days:
                result.add_warning(
                    template.format(
                        field=field_name,
                        reference=ref_name,
                        days=days,
                    )
                )

    return result


def validate_cross_field_consistency(
    fields: dict[str, Any],
) -> CrossFieldValidationResult:
    """Validate all cross-field relationships using schema-defined rules."""
    return _apply_cross_field_rules(fields)


@dataclass
class SchemaValidationResult:
    """Result of schema structure validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error (fails validation)."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning (doesn't fail validation)."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an informational message."""
        self.info.append(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
        }


# =============================================================================
# SCHEMA VALIDATION HELPERS
# =============================================================================


def _validate_semver_field(
    result: SchemaValidationResult,
    value: Any,
    field_path: str,
    required: bool = True,
) -> str | None:
    """Validate a semver field and add errors to result if invalid.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        value: The value to validate
        field_path: Path to the field for error messages (e.g., "version_history[0].version")
        required: Whether the field is required

    Returns:
        The validated semver string, or None if invalid/missing
    """
    if value is None:
        if required:
            result.add_error(f"Missing required '{field_path}'")
        return None
    if not isinstance(value, str):
        result.add_error(f"'{field_path}' must be string, got {type(value).__name__}")
        return None
    if not SEMVER_PATTERN.match(value):
        result.add_error(f"'{field_path}' is not valid semver: {value}")
        return None
    return value


def _validate_single_field_definition(
    result: SchemaValidationResult,
    field_name: str,
    field_def: Any,
) -> None:
    """Validate a single field definition and add errors/warnings to result.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        field_name: Name of the field being validated
        field_def: The field definition dictionary
    """
    if not isinstance(field_def, dict):
        result.add_error(f"field_definitions.{field_name} must be a dict")
        return

    # Validate type
    field_type = field_def.get("type")
    if field_type is None:
        result.add_error(f"Field '{field_name}' missing 'type'")
    elif field_type not in VALID_FIELD_TYPES:
        result.add_error(f"Field '{field_name}' has invalid type '{field_type}'")

    # Validate introduced_in
    _validate_semver_field(
        result,
        value=field_def.get("introduced_in"),
        field_path=f"field_definitions.{field_name}.introduced_in",
        required=False,
    )
    if field_def.get("introduced_in") is None:
        result.add_warning(f"Field '{field_name}' missing 'introduced_in'")

    # Validate deprecated_in and migration_notes
    deprecated_in = _validate_semver_field(
        result,
        value=field_def.get("deprecated_in"),
        field_path=f"field_definitions.{field_name}.deprecated_in",
        required=False,
    )
    if deprecated_in is not None:
        migration_notes = field_def.get("migration_notes")
        if migration_notes is None:
            result.add_error(
                f"Deprecated field '{field_name}' missing required 'migration_notes'"
            )
        elif not isinstance(migration_notes, str) or not migration_notes.strip():
            result.add_error(
                f"Field '{field_name}' migration_notes must be a non-empty string"
            )

    # Validate confidence_threshold
    threshold = field_def.get("confidence_threshold")
    if threshold is not None:
        if not isinstance(threshold, (int, float)):
            result.add_error(
                f"Field '{field_name}' confidence_threshold must be numeric"
            )
        elif not (0.0 <= threshold <= 1.0):
            result.add_error(
                f"Field '{field_name}' confidence_threshold {threshold} not in [0, 1]"
            )


def _validate_version_history(
    result: SchemaValidationResult,
    schema: dict[str, Any],
    current_version: str | None,
) -> None:
    """Validate version_history section of the schema.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        schema: The full schema dictionary
        current_version: The current schema version (for cross-validation)
    """
    version_history = schema.get("version_history")
    if version_history is None:
        result.add_warning("Schema missing 'version_history' for evolution tracking")
        return

    if not isinstance(version_history, list):
        result.add_error("'version_history' must be a list")
        return

    if len(version_history) == 0:
        result.add_warning("'version_history' is empty")
        return

    # Validate each entry
    for i, entry in enumerate(version_history):
        if not isinstance(entry, dict):
            result.add_error(f"version_history[{i}] must be a dict")
            continue

        _validate_semver_field(
            result,
            value=entry.get("version"),
            field_path=f"version_history[{i}].version",
            required=True,
        )

        if "date" not in entry:
            result.add_warning(f"version_history[{i}] missing 'date'")

        if "changes" not in entry:
            result.add_warning(f"version_history[{i}] missing 'changes'")
        elif not isinstance(entry.get("changes"), list):
            result.add_error(f"version_history[{i}].changes must be a list")

    # Check current version is in history
    if current_version:
        history_versions = [
            e.get("version") for e in version_history if isinstance(e, dict)
        ]
        if current_version not in history_versions:
            result.add_warning(
                f"Current version {current_version} not found in version_history"
            )


def _validate_compatibility(
    result: SchemaValidationResult,
    schema: dict[str, Any],
) -> None:
    """Validate compatibility section of the schema.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        schema: The full schema dictionary
    """
    compatibility = schema.get("compatibility")
    if compatibility is None:
        result.add_warning("Schema missing 'compatibility' object")
        return

    if not isinstance(compatibility, dict):
        result.add_error("'compatibility' must be a dict")
        return

    min_version = compatibility.get("minimum_consumer_version")
    if min_version is None:
        result.add_warning("compatibility missing 'minimum_consumer_version'")
    else:
        _validate_semver_field(
            result,
            value=min_version,
            field_path="compatibility.minimum_consumer_version",
            required=False,
        )


def _validate_field_definitions(
    result: SchemaValidationResult,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Validate field_definitions section of the schema.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        schema: The full schema dictionary

    Returns:
        The field_definitions dict (may be empty)
    """
    field_definitions: dict[str, Any] = schema.get("field_definitions", {})
    if not field_definitions:
        result.add_error("Schema missing or empty 'field_definitions'")
        return {}

    for field_name, field_def in field_definitions.items():
        _validate_single_field_definition(result, field_name, field_def)

    return field_definitions


def _validate_fields_list(
    result: SchemaValidationResult,
    schema: dict[str, Any],
    field_definitions: dict[str, Any],
) -> None:
    """Validate that fields list matches field_definitions.

    Args:
        result: SchemaValidationResult to add errors/warnings to
        schema: The full schema dictionary
        field_definitions: The validated field_definitions dict
    """
    fields_list = schema.get("fields", [])
    if not fields_list:
        return

    fields_set = set(fields_list)
    definitions_set = set(field_definitions.keys())

    missing_in_list = definitions_set - fields_set
    missing_in_defs = fields_set - definitions_set

    if missing_in_list:
        result.add_warning(
            f"Fields in definitions but not in fields list: {sorted(missing_in_list)}"
        )
    if missing_in_defs:
        result.add_error(
            f"Fields in list but missing definitions: {sorted(missing_in_defs)}"
        )


def validate_schema_structure(
    schema: dict[str, Any] | None = None,
) -> SchemaValidationResult:
    """Validate the structure of the contract schema itself.

    This function validates that the schema follows the expected structure
    and lifecycle conventions, including:
    - Version is valid semver
    - All fields have introduced_in metadata
    - Deprecated fields have migration_notes
    - version_history exists and is properly structured
    - Field types are valid

    Args:
        schema: Schema dictionary to validate. If None, loads from default location.

    Returns:
        SchemaValidationResult with errors, warnings, and info messages
    """
    result = SchemaValidationResult(is_valid=True)

    # Load schema if not provided
    if schema is None:
        try:
            schema = utils.load_contract_schema()
        except (FileNotFoundError, OSError, json.JSONDecodeError) as e:
            result.add_error(f"Failed to load schema: {e}")
            return result

    # Validate version
    version = _validate_semver_field(result, schema.get("version"), "version")
    if version:
        result.add_info(f"Schema version: {version}")

    # Use helper functions for cleaner, more maintainable validation
    _validate_version_history(result, schema, version)
    _validate_compatibility(result, schema)
    field_definitions = _validate_field_definitions(result, schema)
    _validate_fields_list(result, schema, field_definitions)

    return result
