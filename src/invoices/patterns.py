"""Consolidated text patterns for invoice field validation.

This module provides heuristic functions for text pattern matching and validation
without using regex. Provides a single source of truth for text classification.
"""

from __future__ import annotations

import re

from .config import Config

# =============================================================================
# CONSTANT SETS
# =============================================================================

# Currency symbols — single source of truth is Config
CURRENCY_SYMBOLS: frozenset[str] = Config.CURRENCY_SYMBOLS

# Month name prefixes (3-letter minimum match)
MONTH_PREFIXES: frozenset[str] = frozenset(
    {
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
    }
)

# US state abbreviations (for address detection)
US_STATE_ABBREVS: frozenset[str] = frozenset(
    {
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
        "DC",
    }
)

# Stop words for garbage detection
STOP_WORDS: frozenset[str] = frozenset(
    {
        "of",
        "your",
        "our",
        "the",
        "and",
        "or",
        "to",
        "for",
        "in",
        "on",
        "at",
        "is",
        "it",
        "a",
        "an",
        "by",
        "with",
        "from",
        "as",
    }
)

# Common garbage verbs
GARBAGE_VERBS: frozenset[str] = frozenset(
    {"check", "make", "do", "not", "please", "see"}
)

# Name validation blacklist (context-specific non-name words)
NAME_BLACKLIST: frozenset[str] = frozenset(
    {
        "payable",
        "remit",
        "payment",
        "check",
        "invoice",
        "bill",
        "account",
        "issue",
        "total",
        "due",
        "please",
        "include",
        "thru",
        "through",
        "charges",
        "monthly",
        "regular",
        "current",
        "statement",
        "description",
        "services",
        "billing",
        "consolidated",
        "summary",
        "hello",
        "page",
        "number",
        "amount",
        "balance",
        "forward",
        "previous",
        "enclosed",
        "received",
        "worldwide",
    }
)

# Common nouns that suggest trailing noise in multi-word names
# Example: "AT&T bills," has "bills" which is noise, prefer clean "AT&T"
NAME_NOISE_WORDS: frozenset[str] = frozenset(
    {
        "bills",
        "services",
        "company",
        "inc",
        "group",
        "solutions",
        "corporation",
        "systems",
        "technologies",
        "network",
        "communications",
        "enterprises",
    }
)


# =============================================================================
# PRIVATE HEURISTIC HELPERS
# =============================================================================


def _has_currency_symbol(text: str) -> bool:
    """Check if text contains any currency symbol."""
    return any(c in CURRENCY_SYMBOLS for c in text)


def _has_any_digit(text: str) -> bool:
    """Check if text contains any digit."""
    return any(c.isdigit() for c in text)


def _count_digits(text: str) -> int:
    """Count digits in text."""
    return sum(1 for c in text if c.isdigit())


def _has_month_name(text: str) -> bool:
    """Check if text contains a month name (3+ letter prefix match)."""
    for word in text.split():
        # Strip trailing punctuation (e.g., "Jan." or "January,")
        clean = word.rstrip(".,;:").lower()
        if len(clean) >= 3 and clean[:3] in MONTH_PREFIXES:
            return True
    return False


def _is_bare_year(text: str) -> bool:
    """Check if text is a bare 4-digit year (1900-2099)."""
    return len(text) == 4 and text.isdigit() and text[:2] in ("19", "20")


def _is_invoice_number_like(text: str) -> bool:
    """Check if text has invoice-number-like structure.

    Matches:
    - Letters then digits (INV-2024-001)
    - Digits then letters (123-ABC)
    - Mixed with separators (A1-B2)
    - 5+ pure digits (12345)
    - Letters-digits-letters (ABC123DEF)
    - Hash prefix (#1234)
    - Short prefix + 4+ digits (INV2024001)
    """
    if not text or len(text) < 1:
        return False

    has_digit = any(c.isdigit() for c in text)
    has_alpha = any(c.isalpha() for c in text)
    has_separator = any(c in "-#_" for c in text)

    # Pure digits: need 5+
    if text.isdigit():
        return len(text) >= 5

    # Hash prefix: #1234
    if text.startswith("#") and len(text) > 1 and text[1:].isdigit():
        return True

    # Must have digits for remaining cases
    if not has_digit:
        return False

    # Alpha + digits (with optional separators) — the common case
    if has_alpha and has_digit:
        return True

    # Digits with separators connecting to more alphanumeric
    if has_separator and has_digit:
        return True

    return False


def _looks_like_amount_text(text: str, strict: bool = False) -> bool:
    """Check if text looks like a monetary amount.

    Args:
        text: Text to check
        strict: If True, require clean format (mostly digits + currency + separators)
    """
    has_currency = _has_currency_symbol(text)
    has_digit = _has_any_digit(text)
    has_decimal = "." in text
    has_comma = "," in text

    if strict:
        # Strict: currency symbol optional, digits required, clean format
        if not has_digit:
            return False
        # Should be mostly digits, currency, separators, whitespace
        # Allow up to 3 alpha chars (currency code like USD)
        non_amount_chars = sum(1 for c in text if c.isalpha() and c not in "eE")
        return non_amount_chars <= 3
    else:
        # General: currency + digits, or digits with decimal
        if has_currency and has_digit:
            return True
        if has_digit and has_decimal:
            # Check decimal part is 1-2 digits (cents)
            parts = text.rsplit(".", 1)
            if len(parts) == 2:
                decimal_part = parts[1].strip().rstrip("$€£¥₹₽").strip()
                if (
                    decimal_part
                    and len(decimal_part) <= 2
                    and decimal_part[:2].replace(" ", "").isdigit()
                ):
                    return True
        if has_digit and has_comma:
            return True
        # Pure numeric (no alpha) with 2+ digits — plausible amount
        if has_digit and not any(c.isalpha() for c in text):
            return True
        return False


def _looks_like_date_text(text: str, clean_only: bool = False) -> bool:
    """Check if text looks like a date.

    Args:
        text: Text to check
        clean_only: If True, only match numeric dates (DD/MM/YYYY style)
    """
    digits = _count_digits(text)
    separators = sum(1 for c in text if c in "/-.")
    has_month = _has_month_name(text)

    if clean_only:
        # Numeric only: DD/MM/YYYY or YYYY-MM-DD style
        return separators >= 2 and digits >= 4 and not any(c.isalpha() for c in text)

    # Has month name + digits = date
    if has_month and digits >= 1:
        return True
    # Numeric with separators: need 2+ separators and 4+ digits
    if separators >= 2 and digits >= 4:
        return True
    return False


def _is_name_like(text: str) -> bool:
    """Check if text looks like a person/company name."""
    if len(text) < 3:
        return False
    # Must start with a letter
    if not text[0].isalpha():
        return False
    # Title Case, ALL CAPS, or mixed text with common punctuation
    alpha_count = sum(1 for c in text if c.isalpha())
    return alpha_count >= 2 and all(c.isalpha() or c in " .,&'-" for c in text)


def _is_garbage_text(text: str) -> bool:
    """Detect garbage/word-salad text using stop-word density."""
    words = text.lower().split()
    if len(words) < 3:
        return False

    stop_count = sum(1 for w in words if w in STOP_WORDS)
    garbage_verb_count = sum(1 for w in words if w in GARBAGE_VERBS)

    # Original regex caught: 2+ prepositions, or repeated common verbs
    if stop_count >= 2 and stop_count / len(words) > 0.4:
        return True
    if garbage_verb_count >= 2:
        return True

    # "8529 Nov Do Make" pattern: digit followed by 3+ short lowercase words
    if words[0].replace(".", "").isdigit() and len(words) >= 4:
        short_words = sum(1 for w in words[1:] if len(w) <= 4 and w.isalpha())
        if short_words >= 3:
            return True

    return False


def _find_currency_code(text: str) -> str | None:
    """Find a currency code in text by word splitting."""
    for word in text.split():
        clean = word.strip(".,;:()").upper()
        if clean in Config.CURRENCY_CODES:
            return clean
    return None


def _strip_currency_codes(text: str) -> str:
    """Remove currency code words from text."""
    words = text.split()
    return " ".join(
        w for w in words if w.strip(".,;:()").upper() not in Config.CURRENCY_CODES
    )


def _is_id_like(text: str) -> bool:
    """Check if text has ID-like structure (alphanumeric with separators)."""
    if not text or len(text) == 0:
        return False
    allowed_internal = set("-_#.")
    # Must start and end with alphanumeric
    if not text[0].isalnum():
        return False
    if len(text) > 1 and not text[-1].isalnum():
        return False
    return all(c.isalnum() or c in allowed_internal for c in text)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def is_valid_for_field(text: str, field_type: str) -> bool:
    """Check if text matches expected pattern for a field type.

    Args:
        text: Text to validate
        field_type: Type of field ("id", "amount", "date", "text")

    Returns:
        True if text matches expected pattern for the field type
    """
    text = text.strip()
    if not text:
        return False

    if field_type == "id":
        # Must look like an invoice number (not just a year)
        if _is_bare_year(text):
            return False
        return _is_invoice_number_like(text)

    elif field_type == "amount":
        # Must look like a monetary amount
        if _is_garbage_text(text):
            return False
        return _looks_like_amount_text(text, strict=False)

    elif field_type == "date":
        # Must look like a date
        return _looks_like_date_text(text, clean_only=False)

    elif field_type == "text":
        # Name/text fields - reject garbage
        if _is_garbage_text(text):
            return False
        # Must have reasonable structure
        return len(text) >= 2 and any(c.isalpha() for c in text)

    return True  # Unknown type - accept


def is_garbage_candidate(text: str) -> bool:
    """Check if text is likely garbage (word salad, nonsense).

    This function identifies candidates that should be rejected because they
    contain incoherent text patterns that are clearly not valid field values.

    Args:
        text: Text to check

    Returns:
        True if text appears to be garbage

    Example:
        >>> is_garbage_candidate("on of your more our 3")
        True
        >>> is_garbage_candidate("Acme Corporation")
        False
    """
    text = text.strip()
    if not text:
        return True

    # Check garbage indicators
    if _is_garbage_text(text):
        return True

    # Check for punctuation-only
    if all(not c.isalnum() for c in text):
        return True

    # Check for repetitive text (same word >50%)
    words = text.split()
    if len(words) > 1:
        word_counts: dict[str, int] = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
        max_count = max(word_counts.values())
        if max_count / len(words) > 0.5:
            return True

    return False


def validate_invoice_number(text: str) -> float:
    """Validate if text looks like a proper invoice number.

    Returns a score indicating confidence (positive = good, negative = bad).

    Args:
        text: Text to validate

    Returns:
        Validation score in range [-1.5, +1.5]
    """
    text = text.strip()

    # Strong penalty for date-like patterns (month names + digits)
    # Example: "Jul 19 -" has month name, should be rejected
    if _has_month_name(text):
        return -1.5  # Dates are NOT invoice numbers

    # Strong penalty for text starting with action verbs/prepositions (fragments, not IDs)
    # Example: "pay $353.10 by Aug" starts with "pay"
    first_word = text.split()[0].lower() if text.split() else ""
    if first_word in {"pay", "by", "please", "make", "the", "for", "and", "or"}:
        return -1.5  # Strong penalty - action verb/preposition, not an ID

    # Strong penalty for multi-word phrases containing field names (labels, not values)
    # Example: "Invoice Date", "Due Date", "Total Amount"
    words = text.split()
    if len(words) >= 2:
        # Check if last word is a field type keyword
        last_word = words[-1].lower()
        if last_word in {"date", "amount", "total", "number", "due", "tax"}:
            # This is a label like "Invoice Date", not an invoice number
            return -1.5

    # Strong penalty for text containing currency symbols (not IDs)
    if _has_currency_symbol(text):
        return -1.5

    # Heavy penalty for bare years
    if _is_bare_year(text):
        return -1.5

    # Heavy penalty for ZIP code patterns (e.g., "90503-6515")
    # ZIP codes are 5 digits, optional hyphen, optional 4 digits
    parts = text.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        if len(parts[0]) == 5 and len(parts[1]) == 4:
            return -1.3  # Strong ZIP code signal

    # Penalty for bare 5-digit numbers (likely addresses/ZIPs, not invoice numbers)
    # Real invoice numbers usually have prefixes, suffixes, or separators
    if text.isdigit() and len(text) == 5:
        return -1.0  # Street addresses like "21515"

    # Bonus for standard invoice number prefixes
    # Very strong bonus - these are definitive invoice number markers
    upper = text.upper()
    if upper.startswith(("INV", "INVOICE", "#")):
        return 1.5  # Very strong signal - overcomes spatial features

    # 7-9 digit pure numbers: WEAK signal (could be invoice OR account number)
    # These are ambiguous - many invoices use pure numeric IDs, but so do
    # account numbers, customer IDs, etc. Without additional context, we
    # give only a weak positive signal. If this is truly an invoice number,
    # it should have strong spatial/proximity signals to overcome NONE_BIAS.
    # Example: "305319049" could be invoice number OR account number
    if text.isdigit() and 7 <= len(text) <= 9:
        return 0.1  # Weak signal - needs strong spatial features to win

    # 10+ digits: likely phone/account/tracking number
    if text.isdigit() and len(text) >= 10:
        return -0.8

    # Good invoice number pattern (has structure: letters+digits or separators)
    if _is_invoice_number_like(text):
        # If pure digits, check length and structure
        if text.isdigit():
            # 4-digit pure numbers: ambiguous (could be year, reference, etc.)
            if len(text) == 4:
                return -0.3  # Slight penalty - too generic
            # 6 digits: weak signal
            elif len(text) == 6:
                return 0.05  # Very weak - could be many things
            # Other pure digit lengths: weak
            return 0.1

        # Penalize product/plan name patterns: common-word + short suffix
        # e.g., "Access-3Yr", "Basic-1Mo", "Plus-2Yr"
        # Real invoice numbers have longer digit sequences
        parts = re.split(r"[-_]", text)
        if len(parts) == 2:
            left, right = parts
            # Left part is a common word (all alpha, >= 3 chars)
            # Right part is short with few digits (< 4 digits)
            left_is_word = left.isalpha() and len(left) >= 3
            right_digit_count = sum(1 for c in right if c.isdigit())
            if left_is_word and right_digit_count < 4:
                return -0.5  # Likely a product/plan name

        # Has structure (letters or separators) = stronger signal
        return 0.4

    # Penalty for very short
    if len(text) < 3:
        return -0.5

    # Penalty for short pure-digit numbers (3-4 digits)
    # These are too generic - could be page numbers, years, etc.
    if text.isdigit() and 3 <= len(text) <= 4:
        return -0.8

    # Penalty for pure alphabetic
    if text.isalpha():
        return -0.3

    return 0.0


def validate_amount(text: str) -> float:
    """Validate if text looks like a monetary amount.

    Returns a score indicating confidence (positive = good, negative = bad).

    Args:
        text: Text to validate

    Returns:
        Validation score in range [-1.5, +0.5]
    """
    text = text.strip()

    # Heavy penalty for garbage
    if _is_garbage_text(text):
        return -1.5

    # Strong penalty for text starting with non-amount words (fragments)
    # Example: "pay $353.10" starts with "pay"
    first_word = text.split()[0].lower() if text.split() else ""
    if first_word in {
        "pay",
        "by",
        "please",
        "make",
        "due",
        "total",
        "the",
        "for",
        "and",
        "or",
    }:
        return -1.5  # Strong penalty - sentence fragment, not clean amount

    # Date fragment pattern: "21, 2024" (day-comma-year) - NOT an amount
    # This must be checked early before other validation
    if re.match(r"^\d{1,2},\s*\d{4}$", text):
        return -1.5

    # Trailing comma is never a valid amount format
    if text.endswith(","):
        return -0.3

    # Check word count
    if len(text.split()) > 4:
        return -1.0

    # Penalty for incomplete decimal amounts (e.g., "$30." missing second cent digit)
    # Valid: "$30.00", "$30.5" (single digit OK for some formats), "$30"
    # Invalid: "$30." (ends with period but no digits after)
    if "." in text and text.rstrip().endswith("."):
        return -0.8  # Incomplete amount - missing decimal digits

    # Penalty for excessive decimal places (e.g., "1.000000000")
    # Real currency amounts have at most 2 decimal places.
    # Values like "1.000000000" are multipliers/rates, not monetary amounts.
    if "." in text:
        parts = text.rsplit(".", 1)
        if len(parts) == 2:
            decimal_part = parts[1].rstrip()
            # Strip trailing non-digit chars
            decimal_digits = ""
            for c in decimal_part:
                if c.isdigit():
                    decimal_digits += c
                else:
                    break
            if len(decimal_digits) > 2:
                return -1.0  # Not a currency amount

    # Bare year penalty: 4-digit numbers that look like years are NOT amounts
    if _is_bare_year(text):
        return -1.0

    # Pure integer penalty: amounts almost always have decimal, comma, or
    # currency symbol. Pure integers (especially 4-6 digits) are ambiguous
    # and could be years, ZIP codes, account numbers, etc.
    if text.isdigit() and not _has_currency_symbol(text):
        if len(text) <= 6:
            return -0.5  # Short integers: weak penalty
        else:
            return -0.3  # Long integers: could be account numbers

    # Strict amount pattern
    if _looks_like_amount_text(text, strict=True):
        return 0.5

    # General amount pattern
    if _looks_like_amount_text(text, strict=False):
        return 0.3

    # Has currency and digits
    if _has_currency_symbol(text) and _has_any_digit(text):
        return 0.2

    # Penalty for no digits
    if not _has_any_digit(text):
        return -0.8

    return 0.0


def validate_date(text: str) -> float:
    """Validate if text looks like a date.

    Returns a score indicating confidence (positive = good, negative = bad).

    Args:
        text: Text to validate

    Returns:
        Validation score in range [-1.5, +0.5]
    """
    text = text.strip()

    # Strong penalty for text starting with non-date words (fragments)
    # Example: "pay by Aug 21," or "by Aug 21," are fragments, not clean dates
    first_word = text.split()[0].lower() if text.split() else ""
    if first_word in {
        "pay",
        "by",
        "please",
        "make",
        "due",
        "total",
        "the",
        "for",
        "and",
        "or",
        "on",
    }:
        return (
            -1.5
        )  # Strong penalty - sentence fragment with date in it, not a clean date

    # Strong penalty for trailing comma without year
    # Example: "Aug 21," is incomplete (missing year), prefer "Aug 21, 2024"
    if text.endswith(",") and not any(
        c.isdigit()
        and len(
            [
                x
                for x in text.split()
                if x.strip(",").isdigit() and len(x.strip(",")) == 4
            ]
        )
        > 0
        for c in text
    ):
        # Check if there's a 4-digit year before the comma
        words = text.rstrip(",").split()
        has_year = any(w.isdigit() and len(w) == 4 for w in words)
        if not has_year:
            return -0.8  # Penalty for incomplete date fragment

    # Good date pattern (month name + digits, or numeric date)
    if _looks_like_date_text(text, clean_only=False):
        # Bonus for having 4-digit year (more complete)
        if any(w.isdigit() and len(w) == 4 for w in text.split()):
            return 0.5
        # Slightly lower for dates without year
        return 0.3

    # Bare year - slight penalty
    if _is_bare_year(text):
        return -0.2

    # Garbage penalty
    if _is_garbage_text(text):
        return -1.0

    # Month name bonus
    if _has_month_name(text):
        return 0.2

    return 0.0


def validate_name(
    text: str,
    document_labels: set[str] | None = None,
    cross_page_headers: set[str] | None = None,
    address_city_tokens: set[str] | None = None,
) -> float:
    """Validate if text looks like a name (person or company).

    Returns a score indicating confidence (positive = good, negative = bad).

    Args:
        text: Text to validate
        document_labels: Optional set of document-level structural label tokens
            (lowercased). Candidates overlapping with these get a penalty since
            they are likely column headers or repeated structural text.
        cross_page_headers: Optional set of tokens appearing at the same Y-position
            across 3+ pages. These are strongly structural (column headers, address
            headers). Single-word names in this set get penalized.
        address_city_tokens: Optional set of tokens detected as city names in
            address blocks (adjacent to ZIP codes).

    Returns:
        Validation score in range [-1.5, +0.4]
    """
    text = text.strip()
    lower = text.lower()

    # Heavy penalty for garbage
    if _is_garbage_text(text):
        return -1.5

    # Month name penalty: "Oct", "Jan", "Aug" etc. are dates, not names.
    # Single-word month names should never be vendor/customer names.
    if len(text.split()) == 1 and _has_month_name(text):
        return -1.5

    # Repeated-character garbage detection: "NNNNNNNN", "XXXXXXXX", etc.
    # 4+ identical consecutive characters indicate garbage, not a company name.
    if re.search(r"(.)\1{3,}", text):
        return -1.5

    # Contraction detection: "It's a", "Don't", "We're" — not company names.
    # Excludes possessive company names like "McDonald's" (uppercase after ').
    if re.search(r"[a-z]'[a-z]", text):
        return -1.5

    # Penalty for common non-name words that appear near name contexts
    # "payable" appears in "Make payable to..." but is NOT a vendor name
    # "issue" appears in "Issue Date:" but is NOT a vendor name
    # "total", "due", "please" are invoice terms, not vendor names
    # Strong penalty to overcome bucket/spatial bonuses
    # Check if entire text OR any word in text is blacklisted
    if lower in NAME_BLACKLIST or any(word in NAME_BLACKLIST for word in lower.split()):
        return -1.5

    # Document-level label penalty: if candidate text overlaps with structural
    # tokens detected in this document, it's likely a header/label, not a value.
    # Conservative: only penalize when ALL words overlap (not 50%), exempt
    # short names (1-2 words like "AT&T", "Comcast"), and use reduced penalty.
    if document_labels:
        words = lower.split()
        if len(words) >= 2:
            overlap = sum(1 for w in words if w in document_labels)
            if overlap == len(words):  # ALL words must overlap
                return -0.5

    # Address city penalty: words on the same line as ZIP codes are city names
    # (e.g., "Dallas" next to "75202"). These should not be vendor names.
    if address_city_tokens:
        if lower in address_city_tokens:
            return -1.0

    # Cross-page header penalty: single-word tokens appearing at the same
    # Y-position across 3+ pages are likely structural (column headers,
    # repeated address components like "End", "Date", "Description").
    # This catches structural words that slip through the document_labels check
    # (which exempts single-word names to protect "AT&T", "Comcast").
    # Note: vendor names that repeat in headers (like "AT&T") also match,
    # so this penalty is mild — colon-value bonus (3.0) easily overcomes it.
    if cross_page_headers and len(lower.split()) == 1:
        if lower in cross_page_headers:
            return -0.3

    # Detect address patterns: "CITY STATE" or "CITY, STATE"
    # Example: "CAROL STREAM, IL" or "CAROL STREAM IL"
    # Address fragments should not be selected as vendor names
    words = text.upper().replace(",", " ").split()
    if len(words) >= 2:
        # Check if last word is state abbreviation
        if words[-1] in US_STATE_ABBREVS:
            # Strong penalty - this is likely an address line
            return -1.5

    # Penalty for short
    if len(text) < 3:
        return -0.5

    # Penalty for pure numbers
    if text.replace(" ", "").replace(",", "").replace(".", "").isdigit():
        return -0.8

    # Good name pattern
    if _is_name_like(text):
        # Trademark suffix check removed - _is_name_like already rejects these
        # via the trailing punctuation and alphanumeric structure rules.
        # Candidates like "U-verseSM" fail _is_name_like because 'S' or 'M'
        # at the end without proper word boundaries violate name structure.
        # "PRISM" and "AT&T" pass _is_name_like correctly.

        # Penalize single-word names ending with period (likely address fragments)
        # Example: "California." from address line "Carol Stream, California."
        # Real vendor names with periods are usually abbreviations (Inc., Corp.) which are multi-word
        upper = text.upper()
        if (
            len(text.split()) == 1
            and text.endswith(".")
            and not upper.endswith(("INC.", "CORP.", "LTD.", "LLC."))
        ):
            return -0.6

        # Penalize multi-word names with trailing noise (e.g., "AT&T bills,")
        # Clean short names should beat longer names with generic suffixes
        words = text.split()
        if len(words) >= 2:
            # Check if any trailing word (after first) is noise or if text ends with comma
            last_word = words[-1].rstrip(".,;:").lower()
            if last_word in NAME_NOISE_WORDS or text.rstrip().endswith(","):
                return -0.3  # Penalty allows cleaner 1-token name to win

        return 0.4

    # Check word quality
    words = text.split()
    if len(words) >= 2:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len >= 3:
            return 0.2
        else:
            return -0.3

    return 0.0


def get_field_type_validator(field_type: str) -> object:
    """Get the appropriate validator function for a field type.

    Args:
        field_type: Type of field ("id", "amount", "date", "text")

    Returns:
        Validator function that takes text and returns score
    """
    validators: dict[str, object] = {
        "id": validate_invoice_number,
        "amount": validate_amount,
        "date": validate_date,
        "text": validate_name,
        "address": validate_name,
    }
    return validators.get(field_type, lambda x: 0.0)


# =============================================================================
# PATTERN DETECTION FUNCTIONS
# =============================================================================


def looks_like_date(text: str) -> bool:
    """Quick check if text looks like a date.

    Args:
        text: Text to check

    Returns:
        True if text has date-like characteristics
    """
    text = text.strip()
    if len(text) < 4 or len(text) > 20:
        return False

    digits = sum(1 for c in text if c.isdigit())
    separators = sum(1 for c in text if c in "/-.")

    # Date-like if mostly digits with separators
    if digits >= 4 and separators >= 1:
        return True

    # Month name check
    if _has_month_name(text):
        return True

    return False


def looks_like_amount(text: str) -> bool:
    """Quick check if text looks like a monetary amount.

    Args:
        text: Text to check

    Returns:
        True if text has amount-like characteristics
    """
    text = text.strip()
    if not text:
        return False

    # Currency symbols
    if _has_currency_symbol(text):
        return True

    # Numeric patterns
    digits = sum(1 for c in text if c.isdigit())
    return digits >= 2 and ("." in text or "," in text)


def looks_like_id(text: str) -> bool:
    """Quick check if text looks like an ID.

    Args:
        text: Text to check

    Returns:
        True if text has ID-like characteristics
    """
    text = text.strip()
    if len(text) < 3 or len(text) > 30:
        return False

    # Must be alphanumeric with optional separators
    if not all(c.isalnum() or c in "-_#." for c in text):
        return False

    # Must have at least one digit
    return any(c.isdigit() for c in text)


def looks_like_name(text: str) -> bool:
    """Quick check if text looks like a company/person name.

    Args:
        text: Text to check

    Returns:
        True if text has name-like characteristics
    """
    text = text.strip()
    if len(text) < 2 or len(text) > 60:
        return False

    # Must have alphabetic content
    if sum(1 for c in text if c.isalpha()) < 2:
        return False

    # Reject if garbage
    if _is_garbage_text(text):
        return False

    return True


def extract_currency_info(text: str) -> tuple[str | None, str]:
    """Extract currency symbol/code and clean numeric value from text.

    Args:
        text: Text possibly containing currency

    Returns:
        Tuple of (currency_code, cleaned_text)
    """
    currency_code = None

    # Check for currency symbols
    for symbol, code in Config.CURRENCY_SYMBOL_MAP.items():
        if symbol in text:
            currency_code = code
            break

    # Check for currency codes
    if currency_code is None:
        currency_code = _find_currency_code(text)

    # Clean the text (remove currency symbols and codes)
    cleaned = text
    for symbol in Config.CURRENCY_SYMBOL_MAP:
        cleaned = cleaned.replace(symbol, "")
    cleaned = _strip_currency_codes(cleaned)
    cleaned = cleaned.strip()

    return currency_code, cleaned
