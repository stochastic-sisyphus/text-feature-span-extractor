"""Normalization module for cleaning predicted field values."""

from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Any

from dateutil import parser as date_parser

# Normalization version for guard against drift
NORMALIZE_VERSION = "1.0.0+text_layer"


def normalize_date(raw_text: str) -> tuple[str | None, str]:
    """
    Normalize date text to ISO8601 format.

    Args:
        raw_text: Original date text from PDF

    Returns:
        Tuple of (normalized_value, original_raw_text)
        normalized_value is None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text

    clean_text = raw_text.strip()

    try:
        # Use dateutil parser for flexible date parsing
        parsed_date = date_parser.parse(clean_text, fuzzy=True)

        # Convert to ISO8601 date format (YYYY-MM-DD)
        iso_date = parsed_date.strftime("%Y-%m-%d")

        return iso_date, clean_text

    except (ValueError, TypeError, date_parser.ParserError):
        # If parsing fails, return None but keep original text
        return None, clean_text


def normalize_amount(raw_text: str) -> tuple[str | None, str | None, str]:
    """
    Normalize amount text to decimal with currency code.

    Args:
        raw_text: Original amount text from PDF

    Returns:
        Tuple of (normalized_value, currency_code, original_raw_text)
        normalized_value is None if parsing fails
    """
    if not raw_text or not raw_text.strip():
        return None, None, raw_text

    clean_text = raw_text.strip()

    # Currency symbol mapping
    currency_map = {
        "$": "USD",
        "€": "EUR",
        "£": "GBP",
        "¥": "JPY",
        "₹": "INR",
        "₽": "RUB",
    }

    # Extract currency code
    currency_code = None

    # Check for currency symbols
    for symbol, code in currency_map.items():
        if symbol in clean_text:
            currency_code = code
            break

    # Check for currency codes in text
    currency_codes = ["USD", "EUR", "GBP", "JPY", "INR", "RUB", "CAD", "AUD"]
    for code in currency_codes:
        if code.upper() in clean_text.upper():
            currency_code = code
            break

    # Extract numeric value
    # Remove currency symbols and letters, keep digits, dots, commas
    numeric_chars = []
    for char in clean_text:
        if char.isdigit() or char in ".,-":
            numeric_chars.append(char)
    numeric_text = "".join(numeric_chars)

    if not numeric_text:
        return None, currency_code, clean_text

    # Handle comma as thousands separator
    if "," in numeric_text and "." in numeric_text:
        # Assume comma is thousands separator if it comes before dot
        if numeric_text.rindex(",") < numeric_text.rindex("."):
            numeric_text = numeric_text.replace(",", "")
    elif "," in numeric_text:
        # Check if comma might be decimal separator (European style)
        parts = numeric_text.split(",")
        if len(parts) == 2 and len(parts[1]) == 2:
            # Likely decimal separator
            numeric_text = numeric_text.replace(",", ".")
        else:
            # Likely thousands separator
            numeric_text = numeric_text.replace(",", "")

    # Handle negative amounts
    is_negative = "-" in numeric_text or "(" in clean_text
    numeric_text = numeric_text.replace("-", "")

    try:
        # Parse as decimal
        amount = Decimal(numeric_text)

        if is_negative:
            amount = -amount

        # Format to two decimal places
        normalized_value = f"{amount:.2f}"

        return normalized_value, currency_code, clean_text

    except (InvalidOperation, ValueError):
        return None, currency_code, clean_text


def normalize_id(raw_text: str) -> tuple[str | None, str]:
    """
    Normalize ID text by stripping zero-width and control characters.

    Args:
        raw_text: Original ID text from PDF

    Returns:
        Tuple of (normalized_value, original_raw_text)
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text

    clean_text = raw_text.strip()

    # Remove zero-width and control characters but keep hyphens
    normalized = ""
    for char in clean_text:
        # Keep alphanumeric, hyphens, underscores, and basic punctuation
        if char.isalnum() or char in "-_#.":
            normalized += char
        elif char == " ":
            normalized += char

    # Clean up multiple spaces
    normalized = " ".join(normalized.split())

    if not normalized:
        return None, clean_text

    return normalized, clean_text


def normalize_text(raw_text: str) -> tuple[str | None, str]:
    """
    Normalize general text fields (carrier name, document type, etc.).

    Args:
        raw_text: Original text from PDF

    Returns:
        Tuple of (normalized_value, original_raw_text)
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text

    clean_text = raw_text.strip()

    # Basic normalization - remove extra whitespace
    normalized = " ".join(clean_text.split())

    return normalized, clean_text


def text_len_checksum(text: str) -> str:
    """Compute deterministic checksum of text length for normalization guard."""
    return sha256(f"{len(text)}:{text[:100]}".encode()).hexdigest()[:16]


def normalize_field_value(field: str, raw_text: str) -> dict[str, Any]:
    """
    Normalize a field value using pattern-based inference, not field name rules.

    Args:
        field: Field name from schema
        raw_text: Raw text value to normalize

    Returns:
        Dictionary with normalized value, currency_code (if applicable), and raw_text
    """
    if not raw_text or not raw_text.strip():
        return {
            "value": None,
            "raw_text": raw_text,
            "currency_code": None,
        }

    # Infer normalization type from text patterns, not field names
    clean_text = raw_text.strip()

    # Try date parsing if text looks date-like
    if _looks_like_date(clean_text):
        normalized_value, original_text = normalize_date(raw_text)
        return {
            "value": normalized_value,
            "raw_text": original_text,
            "currency_code": None,
        }

    # Try amount parsing if text looks amount-like
    elif _looks_like_amount(clean_text):
        normalized_value, currency_code, original_text = normalize_amount(raw_text)
        return {
            "value": normalized_value,
            "raw_text": original_text,
            "currency_code": currency_code,
        }

    # Try ID parsing if text looks ID-like
    elif _looks_like_id(clean_text):
        normalized_value, original_text = normalize_id(raw_text)
        return {
            "value": normalized_value,
            "raw_text": original_text,
            "currency_code": None,
        }

    # Default to text normalization
    else:
        normalized_value, original_text = normalize_text(raw_text)
        return {
            "value": normalized_value,
            "raw_text": original_text,
            "currency_code": None,
        }


def _looks_like_date(text: str) -> bool:
    """Pattern-based date detection."""
    text = text.strip()
    if len(text) < 4 or len(text) > 20:
        return False

    # Count digits, separators, and letters
    digits = sum(bool(c.isdigit()) for c in text)
    separators = sum(bool(c in "/-.") for c in text)

    # Date-like if mostly digits with separators
    if digits >= 4 and separators >= 1:
        return True

    # Month name check
    month_words = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    text_lower = text.lower()
    if any(month in text_lower for month in month_words):
        return True

    return False


def _looks_like_amount(text: str) -> bool:
    """Pattern-based amount detection."""
    text = text.strip()
    if len(text) < 1:
        return False

    # Currency symbols
    currency_symbols = {"$", "€", "£", "¥", "₹", "₽", "USD", "EUR", "GBP", "CAD"}
    has_currency = any(symbol in text for symbol in currency_symbols)

    # Numeric patterns
    digits = sum(bool(c.isdigit()) for c in text)
    decimals = text.count(".")
    commas = text.count(",")

    return has_currency or (digits >= 2 and (decimals == 1 or commas >= 1))


def _looks_like_id(text: str) -> bool:
    """Pattern-based ID detection."""
    text = text.strip()
    if len(text) < 3 or len(text) > 30:
        return False

    # Must be alphanumeric with some structure
    if not all(c.isalnum() or c in "-_#." for c in text):
        return False

    # Must have at least one digit
    return any(c.isdigit() for c in text)


def normalize_assignments(assignments: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize all field assignments for a document.

    Args:
        assignments: Raw assignments from decoder

    Returns:
        Normalized assignments with cleaned values
    """
    normalized = {}

    for field, assignment in assignments.items():
        if assignment["assignment_type"] == "NONE":
            # No normalization needed for NONE assignments
            normalized[field] = {
                "assignment_type": "NONE",
                "candidate_index": None,
                "cost": assignment["cost"],
                "field": field,
                "normalized_value": None,
                "raw_text": None,
                "currency_code": None,
            }
        else:
            # Normalize the candidate value
            candidate = assignment["candidate"]
            raw_text = candidate.get("raw_text", candidate.get("text", ""))

            normalization_result = normalize_field_value(field, raw_text)

            normalized[field] = {
                "assignment_type": "CANDIDATE",
                "candidate_index": assignment["candidate_index"],
                "cost": assignment["cost"],
                "field": field,
                "candidate": candidate,
                "normalized_value": normalization_result["value"],
                "raw_text": normalization_result["raw_text"],
                "currency_code": normalization_result["currency_code"],
            }

    return normalized
