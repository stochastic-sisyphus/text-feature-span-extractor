"""Constants for candidate generation module."""

from ..config import Config

# Candidate bucket types (reference Config for single source of truth)
BUCKET_DATE_LIKE = Config.BUCKET_DATE_LIKE
BUCKET_AMOUNT_LIKE = Config.BUCKET_AMOUNT_LIKE
BUCKET_ID_LIKE = Config.BUCKET_ID_LIKE
BUCKET_NAME_LIKE = Config.BUCKET_NAME_LIKE
BUCKET_KEYWORD_PROXIMAL = Config.BUCKET_KEYWORD_PROXIMAL
BUCKET_RANDOM_NEGATIVE = Config.BUCKET_RANDOM_NEGATIVE

# Month-name abbreviations used to disambiguate dates from IDs
MONTH_ABBREVS = Config.MONTH_ABBREVS

# Legacy INVOICE_KEYWORDS for backwards compatibility
INVOICE_KEYWORDS = set(Config.INVOICE_KEYWORDS)

# Typed anchor sets for directional vector features
# frozenset supports `in` checks — no need for set() wrapping
TOTAL_ANCHORS = Config.TOTAL_ANCHORS
TAX_ANCHORS = Config.TAX_ANCHORS
DATE_ANCHORS = Config.DATE_ANCHORS
ID_ANCHORS = Config.ID_ANCHORS
NAME_ANCHORS = Config.NAME_ANCHORS

# Anchor type constants
ANCHOR_TYPE_TOTAL = "total"
ANCHOR_TYPE_TAX = "tax"
ANCHOR_TYPE_DATE = "date"
ANCHOR_TYPE_ID = "id"
ANCHOR_TYPE_NAME = "name"

# Currency symbols + codes (from centralized Config)
CURRENCY_SYMBOLS = Config.CURRENCY_SYMBOLS | Config.CURRENCY_CODES

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

# Currency codes — single source of truth is Config
CURRENCY_CODES: frozenset[str] = Config.CURRENCY_CODES

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
# =============================================================================

# Relative position feature names with default values
POSITION_FEATURE_SPECS: dict[str, float] = {
    "y_from_bottom": 0.5,
    "in_top_half": 0.0,
    "in_bottom_quarter": 0.0,
    "in_top_quarter": 0.0,
    "in_right_third": 0.0,
    "in_amount_region": 0.0,
}

# Directional feature default values (per anchor type template)
DIRECTIONAL_DEFAULTS: dict[str, float] = {
    "dx": 1.0,
    "dy": 1.0,
    "dist": 1.414,  # sqrt(2) diagonal
    "aligned_x": 0.0,
    "aligned_y": 0.0,
    "reading_order": 0.0,
    "below": 0.0,
}

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
