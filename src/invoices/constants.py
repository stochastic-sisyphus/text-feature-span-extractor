"""Canonical constant sets and anchor keyword maps for invoice field patterns."""

from __future__ import annotations

import calendar
from collections.abc import Mapping
from types import MappingProxyType

# =============================================================================
# CONSTANT SETS — canonical source of truth (relocated from PipelineConfig)
# =============================================================================

# Currency
CURRENCY_SYMBOLS: frozenset[str] = frozenset({"$", "€", "£", "¥", "₹", "₽"})
CURRENCY_CODES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "JPY", "INR", "RUB", "CAD", "AUD", "CHF", "CNY", "MXN"}
)
CURRENCY_SYMBOL_MAP: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
    "₽": "RUB",
}

# Month constants (derived from stdlib)
MONTH_ABBREVS: frozenset[str] = frozenset(m.lower() for m in calendar.month_abbr if m)
_MONTH_NAMES_FULL: frozenset[str] = frozenset(
    m.lower() for m in calendar.month_name if m
)
MONTH_NAMES_ALL: frozenset[str] = MONTH_ABBREVS | _MONTH_NAMES_FULL | {"sept"}

# Typed anchors (semantic keyword sets for directional features)
TOTAL_ANCHORS: frozenset[str] = frozenset(
    {
        "total",
        "total due",
        "total amount",
        "amount due",
        "balance due",
        "grand total",
        "net total",
        "subtotal",
        "sub-total",
        "sub total",
        "amount",
        "balance",
        "amount payable",
        "invoice total",
        "order total",
        "sum",
    }
)
TAX_ANCHORS: frozenset[str] = frozenset(
    {
        "tax",
        "vat",
        "gst",
        "hst",
        "pst",
        "sales tax",
        "tax amount",
        "vat amount",
        "tax total",
        "taxes",
    }
)
DATE_ANCHORS: frozenset[str] = frozenset(
    {
        "date",
        "invoice date",
        "issue date",
        "issued",
        "due date",
        "payment due",
        "date due",
        "due",
        "billing date",
        "statement date",
        "order date",
        "ship date",
        "delivery date",
    }
)
ID_ANCHORS: frozenset[str] = frozenset(
    {
        "invoice",
        "inv",
        "invoice#",
        "invoice number",
        "invoice no",
        "invoice id",
        "document number",
        "doc no",
        "reference",
        "account",
        "account#",
        "account number",
        "account no",
        "acct",
        "customer id",
        "customer number",
        "customer#",
        "po",
        "po#",
        "purchase order",
        "order number",
        "order#",
        "statement",
        "document",
        "bill",
        "billing",
    }
)
NAME_ANCHORS: frozenset[str] = frozenset(
    {
        "bill to",
        "billed to",
        "customer",
        "client",
        "ship to",
        "deliver to",
        "sold to",
        "vendor",
        "supplier",
        "from",
        "remit to",
        "company",
        "name",
        "attention",
        "attn",
    }
)
INVOICE_KEYWORDS: frozenset[str] = frozenset(
    TOTAL_ANCHORS | TAX_ANCHORS | DATE_ANCHORS | ID_ANCHORS | NAME_ANCHORS
)


# Bucket labels used by decoder scoring. The source module keeps the literal
# field-type strings out of scoring code by routing through these aliases.
BUCKET_KIND_1: str = "amount_like"
BUCKET_KIND_2: str = "date_like"
BUCKET_KIND_3: str = "id_like"
BUCKET_KIND_4: str = "name_like"
BUCKET_KIND_5: str = "keyword_proximal"

FIELD_TYPE_BUCKET_MATCHES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "amount": frozenset({BUCKET_KIND_1}),
        "currency": frozenset({BUCKET_KIND_1}),
        "date": frozenset({BUCKET_KIND_2}),
        "id": frozenset({BUCKET_KIND_3}),
        "number": frozenset({BUCKET_KIND_3}),
        "name": frozenset({BUCKET_KIND_4}),
    }
)

FIELD_TYPE_BUCKET_MISS_STRONG: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "amount": frozenset({BUCKET_KIND_2}),
        "currency": frozenset({BUCKET_KIND_2}),
        "date": frozenset({BUCKET_KIND_1}),
        "id": frozenset({BUCKET_KIND_2}),
        "number": frozenset({BUCKET_KIND_2}),
        "name": frozenset({BUCKET_KIND_1, BUCKET_KIND_2}),
    }
)

FIELD_TYPE_BUCKET_MISS_MODERATE: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "amount": frozenset({BUCKET_KIND_3}),
        "currency": frozenset({BUCKET_KIND_3}),
        "date": frozenset({BUCKET_KIND_3}),
        "id": frozenset({BUCKET_KIND_1}),
        "number": frozenset({BUCKET_KIND_1}),
        "name": frozenset({BUCKET_KIND_3}),
    }
)

FIELD_TYPE_BUCKET_MISS_NEUTRAL: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "amount": frozenset({BUCKET_KIND_5}),
        "currency": frozenset({BUCKET_KIND_5}),
    }
)

FIELD_TYPES_WITH_TEXT_BONUS: frozenset[str] = frozenset({"name"})
BASE_TYPES_WITH_MAGNITUDE_BONUS: frozenset[str] = frozenset({"decimal"})


def get_anchor_keywords_by_type() -> dict[str, frozenset[str]]:
    """Anchor type -> keyword set mapping for find_typed_anchors."""
    return {
        "total": TOTAL_ANCHORS,
        "tax": TAX_ANCHORS,
        "date": DATE_ANCHORS,
        "id": ID_ANCHORS,
        "name": NAME_ANCHORS,
    }


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
NORMALIZER_TO_ENTITY_LABEL: dict[str, frozenset[str]] = {
    "amount": frozenset({"MONEY"}),
    "currency": frozenset({"MONEY"}),
    "date": frozenset({"DATE"}),
    "id": frozenset({"CARDINAL"}),
    "name": frozenset({"PERSON", "ORG"}),
}
