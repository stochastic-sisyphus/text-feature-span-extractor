"""Normalization module for cleaning predicted field values."""

from __future__ import annotations

import calendar
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from dateutil import parser as date_parser  # type: ignore[import-untyped]

from .config import Config

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

# Normalization version for guard against drift
NORMALIZE_VERSION = "1.3.0+page_year_fallback"


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


def extract_page_year(tokens_df: pd.DataFrame, page_idx: int) -> str | None:
    """Extract the most common 4-digit year from tokens on a given page.

    Scans all token text on the page for years matching 19xx or 20xx,
    returns the most frequent one. Deterministic via Counter on sorted tokens.
    """
    from collections import Counter

    if tokens_df is None or tokens_df.empty:
        return None

    page_tokens = tokens_df[tokens_df["page_idx"] == page_idx]
    if page_tokens.empty:
        return None

    years: list[str] = []
    for text in page_tokens["text"]:
        for group in _extract_digit_groups(str(text)):
            if len(group) == 4 and group[:2] in ("19", "20"):
                years.append(group)

    if not years:
        return None

    # Most common year; deterministic because token order is stable
    counter = Counter(years)
    return counter.most_common(1)[0][0]


def _has_month_and_day_only(text: str) -> bool:
    """Check if text has a month name + plausible day but NO 4-digit year.

    Used as a guard to prevent augmenting non-date text with a page year.
    """
    text_lower = text.lower().strip()

    has_month = any(month in text_lower for month in Config.MONTH_NAMES_ALL)
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

    has_month_name = any(month in text_lower for month in Config.MONTH_NAMES_ALL)

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
    else:
        # Without month name, need numeric date format (MM/DD/YYYY, DD-MM-YYYY, etc.)
        # Must have separators and at least 3 numeric components OR 2 with 4-digit year
        separators = [c for c in text if c in "/-."]

        if len(separators) < 1:
            return False

        if len(numeric_groups) >= 3:
            # Three parts like MM/DD/YYYY or DD/MM/YY
            return True
        elif len(numeric_groups) == 2 and len(separators) >= 1:
            # Could be MM/YYYY or similar - need 4-digit year
            has_four_digit_year = any(
                len(n) == 4 and 1900 <= int(n) <= 2100 for n in numeric_groups
            )
            return has_four_digit_year

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


def normalize_date(
    raw_text: str, page_year: str | None = None
) -> tuple[str | None, str]:
    """
    Normalize date text to ISO8601 format.

    STRICT VALIDATION: Only accepts dates with clear day, month, and year components.
    Returns None (ABSTAIN) for incomplete or invalid dates.

    If page_year is provided and the text has month+day but no year, the page year
    is appended as fallback context before parsing. The original raw_text is always
    preserved in the return tuple.

    Args:
        raw_text: Original date text from PDF
        page_year: Optional 4-digit year from the same page (e.g., "2024")

    Returns:
        Tuple of (normalized_value, original_raw_text)
        normalized_value is None if parsing fails or date is incomplete
    """
    if not raw_text or not raw_text.strip():
        return None, raw_text

    clean_text = raw_text.strip()

    # CRITICAL: First check if the text has all required date components
    # This prevents fabrication of missing components
    if not _has_complete_date_components(clean_text):
        # Fallback: if page_year is available and text looks like month+day only,
        # augment with the page year and re-check
        if page_year and _has_month_and_day_only(clean_text):
            augmented = f"{clean_text} {page_year}"
            if _has_complete_date_components(augmented):
                clean_text = augmented
            else:
                return None, raw_text.strip()
        else:
            return None, clean_text

    try:
        # Parse WITHOUT fuzzy mode to avoid inventing components
        # Use dayfirst=False for US format preference, but dateutil will
        # handle explicit formats
        parsed_date = date_parser.parse(clean_text, fuzzy=False, dayfirst=False)

        # Validate that the parsed date is a real calendar date
        if not _is_valid_calendar_date(
            parsed_date.year, parsed_date.month, parsed_date.day
        ):
            return None, clean_text

        # Sanity check: year should be reasonable (1900-2100)
        if parsed_date.year < 1900 or parsed_date.year > 2100:
            return None, clean_text

        # Convert to ISO8601 date format (YYYY-MM-DD)
        iso_date = parsed_date.strftime("%Y-%m-%d")

        return iso_date, clean_text

    except (ValueError, TypeError, date_parser.ParserError):
        # If parsing fails, return None but keep original text
        return None, clean_text


def _is_coherent_amount(text: str) -> bool:
    """
    Check if text represents a single coherent monetary amount.

    Rejects:
    - Multiple separate numbers (e.g., "200M $9.99")
    - Fragments that don't form a valid amount pattern
    - Text with letters mixed into digits (except currency codes at boundaries)

    Accepts:
    - "$1,234.56", "1234.56", "€100", "1,000", "100.00 USD"
    - European format: "1.000,50" (dot as thousands, comma as decimal)
    """
    text = text.strip()

    # Remove currency symbols for structure analysis
    cleaned = text
    for symbol in Config.CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(symbol, "")

    # Remove currency codes (case-insensitive, word-level)
    words = cleaned.split()
    cleaned = " ".join(w for w in words if w.upper() not in Config.CURRENCY_CODES)
    cleaned = cleaned.strip()

    if not cleaned:
        return False

    # CRITICAL: Check for letters mixed with digits (e.g., "abc123def", "200M")
    # This should be rejected as not a coherent amount
    # Allow only: digits, commas, dots, minus, spaces, and parentheses (for negative)
    allowed_chars = set("0123456789.,- ()")
    if not all(c in allowed_chars for c in cleaned):
        return False

    # Check for multiple separate numeric values (the key bug fix)
    # A coherent amount should have only ONE contiguous numeric region
    # (possibly with commas/dots as separators)

    # Extract groups of consecutive digits, commas, and dots
    numeric_chunks: list[str] = []
    current: list[str] = []
    for c in cleaned:
        if c.isdigit() or c in ",.":
            current.append(c)
        else:
            if current:
                numeric_chunks.append("".join(current))
                current = []
    if current:
        numeric_chunks.append("".join(current))

    # Filter out chunks that are just punctuation
    numeric_chunks = [c for c in numeric_chunks if any(ch.isdigit() for ch in c)]

    if len(numeric_chunks) == 0:
        return False

    if len(numeric_chunks) > 1:
        # Multiple numeric chunks - this indicates separate values like "200 9.99"
        # Reject unless it's clearly a single amount split by space after currency removal
        return False

    # Single numeric chunk - validate its structure
    chunk = numeric_chunks[0]

    # Should not start or end with separator
    if chunk.startswith(".") or chunk.startswith(","):
        return False
    if chunk.endswith(",") or chunk.endswith("."):
        return False

    # Detect format: US (comma=thousands, dot=decimal) vs European (dot=thousands, comma=decimal)
    has_comma = "," in chunk
    has_dot = "." in chunk

    if has_comma and has_dot:
        # Mixed separators - determine which is decimal
        last_comma = chunk.rindex(",")
        last_dot = chunk.rindex(".")

        if last_comma > last_dot:
            # European format: 1.000,50 (comma is decimal)
            # Validate: dots should be thousands separators (groups of 3)
            # and there should be only one comma (decimal)
            if chunk.count(",") > 1:
                return False
            # Parts before comma should be valid thousands-separated
            integer_part = chunk[:last_comma]
            dot_parts = integer_part.split(".")
            for i, part in enumerate(dot_parts):
                if i == 0:
                    # First part can be 1-3 digits
                    if not (1 <= len(part) <= 3) or not part.isdigit():
                        return False
                else:
                    # Subsequent parts must be exactly 3 digits
                    if len(part) != 3 or not part.isdigit():
                        return False
        else:
            # US format: 1,000.50 (dot is decimal)
            if chunk.count(".") > 1:
                return False
            # Parts before dot should be valid thousands-separated
            integer_part = chunk[:last_dot]
            comma_parts = integer_part.split(",")
            for i, part in enumerate(comma_parts):
                if i == 0:
                    # First part can be 1-3 digits
                    if not (1 <= len(part) <= 3) or not part.isdigit():
                        return False
                else:
                    # Subsequent parts must be exactly 3 digits
                    if len(part) != 3 or not part.isdigit():
                        return False

    elif has_dot:
        # Only dots - could be decimal or European thousands
        dot_count = chunk.count(".")
        if dot_count > 1:
            # Multiple dots = European thousands separator (no decimal shown)
            parts = chunk.split(".")
            for i, part in enumerate(parts):
                if i == 0:
                    if not (1 <= len(part) <= 3) or not part.isdigit():
                        return False
                else:
                    if len(part) != 3 or not part.isdigit():
                        return False
        # Single dot is fine (decimal point)

    elif has_comma:
        # Only commas - could be US thousands or European decimal
        comma_count = chunk.count(",")
        if comma_count == 1:
            # Single comma - could be decimal (European) or thousands
            # If second part is 1-2 digits, likely decimal
            # If second part is 3 digits, could be either
            # Accept both interpretations
            pass
        else:
            # Multiple commas = thousands separators
            parts = chunk.split(",")
            for i, part in enumerate(parts):
                if i == 0:
                    if not (1 <= len(part) <= 3) or not part.isdigit():
                        return False
                else:
                    if len(part) != 3 or not part.isdigit():
                        return False

    return True


def normalize_amount(raw_text: str) -> tuple[str | None, str | None, str]:
    """
    Normalize amount text to decimal with currency code.

    STRICT VALIDATION: Only accepts coherent monetary amounts.
    Returns None (ABSTAIN) for fragmented or invalid amounts.

    Args:
        raw_text: Original amount text from PDF

    Returns:
        Tuple of (normalized_value, currency_code, original_raw_text)
        normalized_value is None if parsing fails or amount is incoherent
    """
    if not raw_text or not raw_text.strip():
        return None, None, raw_text

    clean_text = raw_text.strip()

    # CRITICAL: First check if this is a coherent single amount
    # This prevents concatenation of fragments like "200M $9.99"
    if not _is_coherent_amount(clean_text):
        return None, None, clean_text

    # Extract currency code
    currency_code = None

    # Check for currency symbols
    for symbol, code in Config.CURRENCY_SYMBOL_MAP.items():
        if symbol in clean_text:
            currency_code = code
            break

    # Check for currency codes in text
    if currency_code is None:
        for code in Config.CURRENCY_CODES:
            if code.upper() in clean_text.upper():
                currency_code = code
                break

    # Extract the single coherent numeric value
    # Remove currency symbols and codes first
    numeric_text = clean_text
    for symbol in Config.CURRENCY_SYMBOL_MAP:
        numeric_text = numeric_text.replace(symbol, "")
    # Remove currency codes (word-level, case-insensitive)
    words = numeric_text.split()
    numeric_text = " ".join(w for w in words if w.upper() not in Config.CURRENCY_CODES)

    numeric_text = numeric_text.strip()

    # Now extract just digits, dots, commas, and minus
    numeric_chars = []
    for char in numeric_text:
        if char.isdigit() or char in ".,-":
            numeric_chars.append(char)
    numeric_text = "".join(numeric_chars)

    if not numeric_text or not any(c.isdigit() for c in numeric_text):
        return None, currency_code, clean_text

    # Handle different number formats (US vs European)
    if "," in numeric_text and "." in numeric_text:
        last_comma = numeric_text.rindex(",")
        last_dot = numeric_text.rindex(".")

        if last_comma < last_dot:
            # US format: 1,234.56 (comma=thousands, dot=decimal)
            numeric_text = numeric_text.replace(",", "")
        else:
            # European format: 1.234,56 (dot=thousands, comma=decimal)
            numeric_text = numeric_text.replace(".", "")  # Remove thousands separator
            numeric_text = numeric_text.replace(",", ".")  # Convert decimal separator

    elif "," in numeric_text:
        # Only commas - could be US thousands or European decimal
        parts = numeric_text.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely European decimal separator (e.g., "100,50")
            numeric_text = numeric_text.replace(",", ".")
        else:
            # Likely US thousands separator (e.g., "1,000" or "1,000,000")
            numeric_text = numeric_text.replace(",", "")

    elif "." in numeric_text:
        # Only dots - check if it's European thousands separator
        dot_count = numeric_text.count(".")
        if dot_count > 1:
            # Multiple dots = European thousands separator (no decimal)
            numeric_text = numeric_text.replace(".", "")
        # Single dot is treated as decimal point

    # Handle negative amounts
    is_negative = "-" in numeric_text or "(" in clean_text
    numeric_text = numeric_text.replace("-", "")

    # Final validation: should be a valid decimal number now
    if numeric_text.count(".") > 1:
        return None, currency_code, clean_text

    # Should not be empty or just a dot
    if not numeric_text or numeric_text == ".":
        return None, currency_code, clean_text

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
    if any(month in text_lower for month in Config.MONTH_ABBREVS):
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
    has_currency = any(s in text for s in Config.CURRENCY_SYMBOLS) or any(
        c in text.upper() for c in Config.CURRENCY_CODES
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


def normalize_assignments(
    assignments: dict[str, Any], sha256: str | None = None
) -> dict[str, Any]:
    """
    Normalize all field assignments for a document.

    Preserves ML metadata (used_ml_model, ml_probability) from decoder
    for confidence scoring downstream.

    When sha256 is provided, loads document tokens once and extracts
    per-page year context for date fallback augmentation.

    Args:
        assignments: Raw assignments from decoder
        sha256: Optional document hash for loading tokens (enables page-year fallback)

    Returns:
        Normalized assignments with cleaned values and preserved ML metadata
    """
    # Load tokens once if sha256 provided (for page-year extraction)
    tokens_df = None
    if sha256:
        from . import tokenize

        tokens_df = tokenize.get_document_tokens(sha256)
        if tokens_df is not None and tokens_df.empty:
            tokens_df = None

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
                # Preserve ML metadata
                "used_ml_model": assignment.get("used_ml_model", False),
                "ml_probability": assignment.get("ml_probability"),
            }
        else:
            # Normalize the candidate value
            candidate = assignment["candidate"]
            raw_text = candidate.get("raw_text", candidate.get("text", ""))

            # Extract page year for date fallback
            page_year = None
            if tokens_df is not None:
                page_idx = candidate.get("page_idx", 0)
                page_year = extract_page_year(tokens_df, page_idx)

            normalization_result = normalize_field_value(
                field, raw_text, page_year=page_year
            )

            normalized[field] = {
                "assignment_type": "CANDIDATE",
                "candidate_index": assignment["candidate_index"],
                "cost": assignment["cost"],
                "field": field,
                "candidate": candidate,
                "normalized_value": normalization_result["value"],
                "raw_text": normalization_result["raw_text"],
                "currency_code": normalization_result["currency_code"],
                # Preserve ML metadata for confidence scoring
                "used_ml_model": assignment.get("used_ml_model", False),
                "ml_probability": assignment.get("ml_probability"),
            }

    return normalized
