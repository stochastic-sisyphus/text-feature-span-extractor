"""Pattern-based normalizer dispatch (looks_like_* + normalize_field_value)."""

from __future__ import annotations

from typing import Any

from ..constants import (
    CURRENCY_CODES,
    CURRENCY_SYMBOLS,
    MONTH_ABBREVS,
)
from ._amount import normalize_amount
from ._date import normalize_date
from ._helpers import _extract_digit_groups, normalize_id, normalize_text


def normalize_field_value(
    field: str, raw_text: str, page_year: str | None = None
) -> dict[str, Any]:
    """
    Normalize a field value using pattern-based inference, not field name rules.

    Args:
        field: Field name from schema
        raw_text: Raw text value to normalize
        page_year: Optional 4-digit year from the candidate's page for date fallback

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
    # Priority: amount → ID → date → text
    # Amounts have the most specific signals (currency symbols, comma-formatted
    # decimals). IDs have alphanum+hyphens. Dates are the most ambiguous pattern
    # and must be checked last among structured types.
    clean_text = raw_text.strip()

    # Try amount parsing first — most specific signals
    if _looks_like_amount(clean_text):
        normalized_value, currency_code, original_text = normalize_amount(raw_text)
        if normalized_value is not None:
            return {
                "value": normalized_value,
                "raw_text": original_text,
                "currency_code": currency_code,
            }
        # Fall through to try other normalizers (e.g., "Oct 20, 2023" looks like amount but isn't)

    # Try ID parsing next
    if _looks_like_id(clean_text):
        normalized_value, original_text = normalize_id(raw_text)
        if normalized_value is not None:
            return {
                "value": normalized_value,
                "raw_text": original_text,
                "currency_code": None,
            }
        # Fall through to try other normalizers

    # Try date parsing last among structured types
    if _looks_like_date(clean_text):
        normalized_value, original_text = normalize_date(raw_text, page_year=page_year)
        if normalized_value is not None:
            return {
                "value": normalized_value,
                "raw_text": original_text,
                "currency_code": None,
            }
        # Fall through to text normalization

    # Default to text normalization
    normalized_value, original_text = normalize_text(raw_text)
    return {
        "value": normalized_value,
        "raw_text": original_text,
        "currency_code": None,
    }


def _looks_like_date(text: str) -> bool:
    """Pattern-based date detection.

    Conservative: rejects obvious amounts and IDs to avoid false positives.
    """
    text = text.strip()
    if len(text) < 4 or len(text) > 20:
        return False

    # --- Negative checks: exclude patterns that belong to other types ---

    # Currency symbol → not a date
    if text[0] in "$€£¥₹₽":
        return False

    # Comma followed by 3 digits (thousands separator like 1,234) → amount
    if any(
        text[i + 1 : i + 4].isdigit()
        for i, c in enumerate(text)
        if c == "," and i + 3 < len(text)
    ):
        return False

    # Month name check (do this early — month names are strong date signals)
    text_lower = text.lower()
    if any(month in text_lower for month in MONTH_ABBREVS):
        return True

    # Count digits and separators
    digits = sum(bool(c.isdigit()) for c in text)
    separators = sum(bool(c in "/-.") for c in text)

    # Digits-and-hyphens only (like 90503-6515) → ID, not date
    # Require date-like structure: 2-4 digit groups separated by date separators
    if separators >= 1 and digits >= 4:
        # Must have date-like digit groups (2-4 digits each)
        groups = _extract_digit_groups(text)
        if all(len(g) <= 4 for g in groups) and len(groups) >= 2:
            # Check at least one group is plausible month/day (1-2 digits)
            # or all groups together form a date pattern
            has_short_group = any(len(g) <= 2 for g in groups)
            has_year_like = any(len(g) == 4 for g in groups)
            if has_short_group or (has_year_like and len(groups) >= 2):
                return True

    return False


def _looks_like_amount(text: str) -> bool:
    """Pattern-based amount detection."""
    text = text.strip()
    if len(text) < 1:
        return False

    # Currency symbols and codes
    has_currency = any(s in text for s in CURRENCY_SYMBOLS) or any(
        c in text.upper() for c in CURRENCY_CODES
    )

    # Numeric patterns
    digits = sum(bool(c.isdigit()) for c in text)
    decimals = text.count(".")
    commas = text.count(",")

    return has_currency or (digits >= 2 and (decimals == 1 or commas >= 1))


def _looks_like_id(text: str) -> bool:
    """Pattern-based ID detection."""
    text = text.strip()
    if len(text) < 3 or len(text) > 50:
        return False

    # Must be alphanumeric with some structure (allow spaces for multi-part IDs)
    if not all(c.isalnum() or c in "-_#. " for c in text):
        return False

    # Must have at least one digit
    return any(c.isdigit() for c in text)
