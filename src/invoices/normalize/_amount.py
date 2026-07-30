"""Monetary amount validation and normalization."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from ..constants import (
    CURRENCY_CODES,
    CURRENCY_SYMBOL_MAP,
    CURRENCY_SYMBOLS,
)


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
    for symbol in CURRENCY_SYMBOLS:
        cleaned = cleaned.replace(symbol, "")

    # Remove currency codes (case-insensitive, word-level)
    words = cleaned.split()
    cleaned = " ".join(w for w in words if w.upper() not in CURRENCY_CODES)
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
    for symbol, code in CURRENCY_SYMBOL_MAP.items():
        if symbol in clean_text:
            currency_code = code
            break

    # Check for currency codes in text
    if currency_code is None:
        for code in CURRENCY_CODES:
            if code.upper() in clean_text.upper():
                currency_code = code
                break

    # Extract the single coherent numeric value
    # Remove currency symbols and codes first
    numeric_text = clean_text
    for symbol in CURRENCY_SYMBOL_MAP:
        numeric_text = numeric_text.replace(symbol, "")
    # Remove currency codes (word-level, case-insensitive)
    words = numeric_text.split()
    numeric_text = " ".join(w for w in words if w.upper() not in CURRENCY_CODES)

    numeric_text = numeric_text.strip()

    # Now extract just digits, dots, commas, and minus
    numeric_text = "".join(
        char for char in numeric_text if char.isdigit() or char in ".,-"
    )

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
