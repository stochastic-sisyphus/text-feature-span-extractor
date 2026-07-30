"""Constants for candidate generation module."""

# Re-export barrel: all names below are intentionally public.
__all__ = [
    "ANCHOR_TYPE_DATE",
    "ANCHOR_TYPE_ID",
    "ANCHOR_TYPE_NAME",
    "ANCHOR_TYPE_TAX",
    "ANCHOR_TYPE_TOTAL",
    "BASE_FEATURE_NAMES",
    "BUCKET_AMOUNT_LIKE",
    "BUCKET_DATE_LIKE",
    "BUCKET_ID_LIKE",
    "BUCKET_KEYWORD_PROXIMAL",
    "BUCKET_NAME_LIKE",
    "BUCKET_RANDOM_NEGATIVE",
    "COLUMN_ALIGN_THRESHOLD",
    "COMMON_NON_NAMES",
    "CURRENCY_CODES",
    "CURRENCY_SYMBOLS",
    "DATE_ANCHORS",
    "DIRECTIONAL_DEFAULTS",
    "ID_ANCHORS",
    "INVOICE_KEYWORDS",
    "MIN_LENGTH_BY_BUCKET",
    "MONTH_ABBREVS",
    "NAME_ANCHORS",
    "POSITION_FEATURE_SPECS",
    "ROW_ALIGN_THRESHOLD",
    "STOPWORDS",
    "TAX_ANCHORS",
    "TOTAL_ANCHORS",
]

from invoices.constants import (
    CURRENCY_CODES as _CURRENCY_CODES,
)
from invoices.constants import (
    CURRENCY_SYMBOLS as _CURRENCY_SYMBOLS,
)
from invoices.constants import (
    DATE_ANCHORS as DATE_ANCHORS,
)
from invoices.constants import (
    ID_ANCHORS as ID_ANCHORS,
)
from invoices.constants import (
    INVOICE_KEYWORDS as _INVOICE_KEYWORDS,
)
from invoices.constants import (
    MONTH_ABBREVS as MONTH_ABBREVS,
)
from invoices.constants import (
    NAME_ANCHORS as NAME_ANCHORS,
)
from invoices.constants import (
    TAX_ANCHORS as TAX_ANCHORS,
)
from invoices.constants import (
    TOTAL_ANCHORS as TOTAL_ANCHORS,
)
from invoices.feature_prep import (
    BUCKET_AMOUNT_LIKE as BUCKET_AMOUNT_LIKE,
)
from invoices.feature_prep import (
    BUCKET_DATE_LIKE as BUCKET_DATE_LIKE,
)
from invoices.feature_prep import (
    BUCKET_ID_LIKE as BUCKET_ID_LIKE,
)
from invoices.feature_prep import (
    BUCKET_KEYWORD_PROXIMAL as BUCKET_KEYWORD_PROXIMAL,
)
from invoices.feature_prep import (
    BUCKET_NAME_LIKE as BUCKET_NAME_LIKE,
)
from invoices.feature_prep import (
    BUCKET_RANDOM_NEGATIVE as BUCKET_RANDOM_NEGATIVE,
)
from invoices.features import DIRECTIONAL_DEFAULTS as DIRECTIONAL_DEFAULTS
from invoices.features import POSITION_FEATURE_SPECS as POSITION_FEATURE_SPECS

# Spatial feature thresholds (relocated from proximity.py, deleted in Phase 0b/0c)
COLUMN_ALIGN_THRESHOLD: float = 0.08
ROW_ALIGN_THRESHOLD: float = 0.03

# Legacy INVOICE_KEYWORDS as mutable set for backwards compatibility
INVOICE_KEYWORDS = set(_INVOICE_KEYWORDS)

# Anchor type constants
ANCHOR_TYPE_TOTAL = "total"
ANCHOR_TYPE_TAX = "tax"
ANCHOR_TYPE_DATE = "date"
ANCHOR_TYPE_ID = "id"
ANCHOR_TYPE_NAME = "name"

# Currency symbols + codes (union for broad detection)
CURRENCY_SYMBOLS = _CURRENCY_SYMBOLS | _CURRENCY_CODES

# Stopwords to reject for vendor/customer name candidates
# These are common English words that should never be extracted as entity names
STOPWORDS: frozenset[str] = frozenset(
    {
        # Articles
        "the",
        "a",
        "an",
        # Conjunctions
        "and",
        "or",
        "but",
        "nor",
        "so",
        "yet",
        "for",
        # Prepositions
        "in",
        "on",
        "at",
        "to",
        "from",
        "with",
        "by",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "out",
        "of",
        # Pronouns
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        # Verbs (common)
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "need",
        # Adverbs/modifiers
        "not",
        "no",
        "only",
        "very",
        "just",
        "also",
        "now",
        "then",
        "here",
        "there",
        "again",
        "further",
        "once",
        "too",
        "more",
        "most",
        "some",
        "such",
        "same",
        "other",
        "any",
        "all",
        "both",
        "each",
        "every",
        "few",
        "own",
        "than",
        # Invoice-specific noise words (labels, not values)
        # Note: "no", "of", "you" already in common words above
        "invoice",
        "total",
        "amount",
        "date",
        "due",
        "payment",
        "balance",
        "subtotal",
        "tax",
        "description",
        "qty",
        "quantity",
        "unit",
        "price",
        "item",
        "number",
        "page",
        "thank",
        "please",
        "pay",
        "remit",
        "terms",
        "net",
        "days",
    }
)

# Minimum length for valid candidates by bucket type
MIN_LENGTH_BY_BUCKET: dict[str, int] = {
    "id_like": 3,  # Invoice numbers need at least 3 chars
    "amount_like": 1,  # Amounts can be single digit (e.g., "5")
    "date_like": 4,  # Dates need at least 4 chars (e.g., "1/25")
    "name_like": 2,  # Names need at least 2 chars (e.g., "AT")
    "keyword_proximal": 2,  # General text needs 2+ chars
    "random_negative": 2,  # Random negatives need 2+ chars
}

# Currency codes — single source of truth is patterns.py
CURRENCY_CODES: frozenset[str] = _CURRENCY_CODES

# Common words that are NOT company/person names
# These often start with uppercase in invoices but aren't names
COMMON_NON_NAMES: frozenset[str] = frozenset(
    {
        # Months (short and full)
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
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        # Common invoice/document words
        "issue",
        "account",
        "make",
        "return",
        "returned",
        "set",
        "get",
        "box",
        "ste",
        "suite",
        "apt",
        "floor",
        "room",
        "unit",
        "ave",
        "blvd",
        "st",
        "rd",
        "dr",
        "ln",
        "ct",
        "way",
        "pl",
        "cost",
        "total",
        "amount",
        "due",
        "date",
        "invoice",
        "payment",
        "late",
        "funds",
        "check",
        "cash",
        "card",
        "credit",
        "debit",
        "balance",
        "paid",
        "pay",
        "bill",
        "charge",
        "fee",
        "tax",
        "please",
        "thank",
        "note",
        "page",
        "item",
        "qty",
        "quantity",
        "description",
        "service",
        "product",
        "order",
        "number",
        "no",
        "phone",
        "fax",
        "email",
        "web",
        "site",
        "www",
        "http",
        "usa",
        "new",
        "old",
        "all",
        "any",
        "not",
        "yes",
        "see",
        "per",
        "carol",
        "stream",  # Address words
        # More common document words (NOT company names)
        "payments",
        "paying",
        "managing",
        "printed",
        "assessment",
        "monthly",
        "static",
        "paper",
        "recyclable",
        "important",
        "intellectual",
        "autopay",
        "torrance",
        "hawthorne",
        "original",
        "conversion",
        "authorizes",
        "authorize",
        "checks",
        "includes",
        "bills",
        "use",
        "for",
        "your",
        "the",
        "and",
        "with",
        "from",
        "this",
        "that",
        "are",
        "has",
        "have",
        "will",
        "can",
        "include",
        "including",
        "about",
        "here",
        "there",
        "when",
        "where",
    }
)


# =============================================================================
# CENTRALIZED FEATURE SPECIFICATIONS
# Re-exported from invoices.features (single source of truth)
# =============================================================================


# Base feature names (geometric, text, bucket)
BASE_FEATURE_NAMES: list[str] = [
    "center_x",
    "center_y",
    "width",
    "height",
    "area",
    "char_count",
    "word_count",
    "digit_count",
    "alpha_count",
    "page_idx",
    "bucket_amount_like",
    "bucket_date_like",
    "bucket_id_like",
    "bucket_keyword_proximal",
    "bucket_random_negative",
    "bucket_other",
]
