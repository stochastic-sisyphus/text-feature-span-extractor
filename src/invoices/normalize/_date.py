"""Date normalization and per-page year extraction."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from dateutil import parser as date_parser  # type: ignore[import-untyped]

from ._helpers import (
    _extract_digit_groups,
    _has_complete_date_components,
    _has_month_and_day_only,
    _is_valid_calendar_date,
)

if TYPE_CHECKING:
    import polars as pl


def extract_page_year(tokens_df: pl.DataFrame, page_idx: int) -> str | None:
    """Extract the most common 4-digit year from tokens on a given page.

    Scans all token text on the page for years matching 19xx or 20xx,
    returns the most frequent one. Deterministic via Counter on sorted tokens.
    """
    import polars as pl

    if tokens_df is None or tokens_df.is_empty():
        return None

    page_tokens = tokens_df.filter(pl.col("page_idx") == page_idx)
    if page_tokens.is_empty():
        return None

    years: list[str] = [
        group
        for text in page_tokens["text"].to_list()
        for group in _extract_digit_groups(str(text))
        if len(group) == 4 and group[:2] in ("19", "20")
    ]

    if not years:
        return None

    # Most common year; deterministic because token order is stable
    counter = Counter(years)
    return counter.most_common(1)[0][0]


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
