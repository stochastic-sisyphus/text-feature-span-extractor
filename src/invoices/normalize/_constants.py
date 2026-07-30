"""Module-level constants for normalize package."""

from __future__ import annotations

# Normalization version for guard against drift
NORMALIZE_VERSION = "1.3.0+page_year_fallback"

# Legal-form suffixes stripped before fuzzy vendor comparison.
# Only truly universal tokens — NOT contextual words (session 48 anti-pattern).
_VENDOR_SUFFIX_TOKENS: tuple[str, ...] = (
    "inc",
    "inc.",
    "llc",
    "llc.",
    "ltd",
    "ltd.",
    "corp",
    "corp.",
    "co",
    "co.",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "bv",
    "nv",
)
