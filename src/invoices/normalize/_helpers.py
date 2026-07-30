"""Small leaf utilities: digit/date/calendar helpers, vendor/id/text normalizers."""

from __future__ import annotations

import calendar
from hashlib import sha256

from ..constants import MONTH_NAMES_ALL
from ._constants import _VENDOR_SUFFIX_TOKENS


def normalize_vendor_name(s: str) -> str:
    """Lowercase, strip punctuation, remove legal-form suffixes.

    Used as a pre-processor for fuzzy corpus matching — NOT a field-level
    output normalizer.  Returns a cleaned string suitable for
    ``rapidfuzz.utils.default_process`` to finish.

    The transform is idempotent: ``f(f(x)) == f(x)`` for all inputs.
    Whitespace collapse happens first so that control characters (``\\r``,
    ``\\t``, etc.) embedded before trailing punctuation cannot shift the
    result on a second pass.

    Example:
        >>> normalize_vendor_name("Acme Corp.")
        'acme'
        >>> normalize_vendor_name("AT&T Inc")
        'at&t'
    """
    # 1. Lowercase and collapse ALL whitespace (tabs, \r, \n, multiple spaces)
    text = " ".join(s.lower().split())
    # 2. Strip trailing comma or period that might precede a suffix
    text = text.rstrip(",.")
    # 3. Final trim — rstrip(",.")  can expose a trailing space on edge inputs
    text = text.strip()
    # 4. Strip known legal-form suffix tokens (one pass — single trailing suffix)
    parts = text.rsplit(None, 1)
    if len(parts) == 2 and parts[1] in _VENDOR_SUFFIX_TOKENS:
        text = parts[0].rstrip(",.")
        text = text.strip()
    return text


def _extract_digit_groups(text: str) -> list[str]:
    """Extract groups of consecutive digits from text."""
    groups: list[str] = []
    current: list[str] = []
    for c in text:
        if c.isdigit():
            current.append(c)
        else:
            if current:
                groups.append("".join(current))
                current = []
    if current:
        groups.append("".join(current))
    return groups


def _has_month_and_day_only(text: str) -> bool:
    """Check if text has a month name + plausible day but NO 4-digit year.

    Used as a guard to prevent augmenting non-date text with a page year.
    """
    text_lower = text.lower().strip()

    has_month = any(month in text_lower for month in MONTH_NAMES_ALL)
    if not has_month:
        return False

    # Must have at least one 1-2 digit number (plausible day)
    numeric_groups = _extract_digit_groups(text)
    has_day = any(1 <= int(n) <= 31 for n in numeric_groups if len(n) <= 2)
    if not has_day:
        return False

    # Must NOT have a 4-digit year already
    has_year = any(len(n) == 4 and 1900 <= int(n) <= 2100 for n in numeric_groups)
    return not has_year


def _has_complete_date_components(text: str) -> bool:
    """
    Check if text contains all necessary date components (day, month, year).

    Returns True only if the text appears to have explicit day, month, and year.
    """
    text_lower = text.lower().strip()

    has_month_name = any(month in text_lower for month in MONTH_NAMES_ALL)

    # Count numeric groups (potential day, month number, year)
    numeric_groups = _extract_digit_groups(text)

    if has_month_name:
        # With month name, need at least day and year (2 numeric groups)
        # Year should be 2 or 4 digits
        if len(numeric_groups) < 2:
            return False
        # Check for a plausible year (2 or 4 digit)
        has_year = any(
            (len(n) == 4 and 1900 <= int(n) <= 2100) or (len(n) == 2 and int(n) <= 99)
            for n in numeric_groups
        )
        # Check for a plausible day (1-31)
        has_day = any(1 <= int(n) <= 31 for n in numeric_groups if len(n) <= 2)
        return has_year and has_day
    # Without month name, need numeric date format (MM/DD/YYYY, DD-MM-YYYY, etc.)
    # Must have separators and at least 3 numeric components OR 2 with 4-digit year
    separators = [c for c in text if c in "/-."]

    if len(separators) < 1:
        return False

    if len(numeric_groups) >= 3:
        # Three parts like MM/DD/YYYY or DD/MM/YY
        return True
    if len(numeric_groups) == 2 and len(separators) >= 1:
        # Could be MM/YYYY or similar - need 4-digit year
        return any(len(n) == 4 and 1900 <= int(n) <= 2100 for n in numeric_groups)

    return False


def _is_valid_calendar_date(year: int, month: int, day: int) -> bool:
    """Check if the given date components form a valid calendar date."""
    try:
        if month < 1 or month > 12:
            return False
        if day < 1:
            return False
        # Get the number of days in the month
        _, max_day = calendar.monthrange(year, month)
        return day <= max_day
    except (ValueError, OverflowError):
        return False


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
