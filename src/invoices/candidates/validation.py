"""Candidate validation and filtering functions."""

from ..config import Config
from ..logging import get_logger
from .constants import (
    BUCKET_AMOUNT_LIKE,
    BUCKET_DATE_LIKE,
    BUCKET_ID_LIKE,
    BUCKET_KEYWORD_PROXIMAL,
    BUCKET_NAME_LIKE,
    CURRENCY_CODES,
    MIN_LENGTH_BY_BUCKET,
    STOPWORDS,
)

logger = get_logger(__name__)


# =============================================================================
# GARBAGE CANDIDATE FILTERING
# =============================================================================


def is_garbage_candidate(text: str, bucket: str | None = None) -> bool:
    """Check if text is a garbage candidate that should be rejected.

    Garbage candidates include:
    - Tokens that are only punctuation
    - Empty or whitespace-only tokens
    - Single non-digit characters (but single digits are OK for amounts)

    Args:
        text: The candidate text to check
        bucket: Optional bucket type for bucket-specific rules

    Returns:
        True if the candidate should be rejected, False otherwise
    """
    text_clean = text.strip()

    # Reject empty or whitespace-only
    if not text_clean:
        return True

    # Reject punctuation-only tokens
    if all(not c.isalnum() for c in text_clean):
        return True

    # Allow single digits (for amounts like "5")
    if len(text_clean) == 1:
        if text_clean.isdigit():
            return False  # Single digit is OK
        return True  # Single non-digit char is garbage

    return False


def is_valid_id_candidate(text: str) -> bool:
    """Check if text is a valid invoice number/ID candidate.

    Valid ID candidates:
    - At least 3 characters long
    - Must contain at least one digit (invoice numbers typically have digits)
    - Cannot be purely alphabetic (no digits)

    Args:
        text: The candidate text to check

    Returns:
        True if valid ID candidate, False otherwise
    """
    text_clean = text.strip()

    # Must be at least 3 characters
    if len(text_clean) < 3:
        return False

    # Must have at least one digit (invoice numbers need digits)
    if not any(c.isdigit() for c in text_clean):
        return False

    # Passed all checks
    return True


def is_valid_amount_candidate(text: str) -> bool:
    """Check if text is a valid monetary amount candidate.

    Valid amount candidates:
    - Recognized currency codes (USD, EUR, etc.)
    - Must contain at least one digit
    - Cannot be purely alphabetic

    Args:
        text: The candidate text to check

    Returns:
        True if valid amount candidate, False otherwise
    """
    text_clean = text.strip().strip("()")

    # Currency codes are valid amount-bucket candidates (they arrive here
    # because is_amount_like_soft() matches CURRENCY_SYMBOLS which includes codes)
    if text_clean.upper() in CURRENCY_CODES:
        return True

    # Must have at least one digit
    if not any(c.isdigit() for c in text_clean):
        return False

    # Cannot be purely alphabetic
    if text_clean.isalpha():
        return False

    return True


def is_valid_name_candidate(text: str) -> bool:
    """Check if text is a valid vendor/customer name candidate.

    Valid name candidates:
    - Not a stopword
    - Not starting with lowercase (proper names start with uppercase)
    - At least 2 characters long

    Args:
        text: The candidate text to check

    Returns:
        True if valid name candidate, False otherwise
    """
    text_clean = text.strip()

    # Must be at least 2 characters
    if len(text_clean) < 2:
        return False

    # Reject stopwords
    text_lower = text_clean.lower()
    if text_lower in STOPWORDS:
        return False

    # Reject tokens starting with lowercase (proper names are capitalized)
    # But allow special cases like "eBay", "iPhone" - check if mostly uppercase
    if text_clean and text_clean[0].islower():
        # Exception: Allow if it contains uppercase somewhere (e.g., "eBay")
        if not any(c.isupper() for c in text_clean[1:]):
            return False

    return True


def is_valid_date_candidate(text: str) -> bool:
    """Check if text is a valid date candidate.

    Valid date candidates:
    - At least 4 characters long
    - Must contain digits (dates have numbers)

    Args:
        text: The candidate text to check

    Returns:
        True if valid date candidate, False otherwise
    """
    text_clean = text.strip()

    # Must be at least 4 characters (e.g., "1/25" or "Jan1")
    if len(text_clean) < 4:
        return False

    # Must have at least one digit
    if not any(c.isdigit() for c in text_clean):
        return False

    return True


def _trace_reject(text: str, bucket: str, reason: str) -> None:
    """Log a candidate rejection when TRACE_CANDIDATES is enabled."""
    if Config.TRACE_CANDIDATES:
        logger.warning(
            "trace_candidate_rejected",
            text=text[:80],
            bucket=bucket,
            reason=reason,
        )


def filter_candidate_by_bucket(text: str, bucket: str) -> bool:
    """Filter a candidate based on its assigned bucket type.

    This is the main entry point for bucket-specific filtering. Each bucket
    type has different validation rules.

    Args:
        text: The candidate text
        bucket: The bucket type (date_like, amount_like, id_like, etc.)

    Returns:
        True if the candidate should be KEPT, False if it should be rejected
    """
    # First, apply general garbage filter (bucket-aware for single digits)
    if is_garbage_candidate(text, bucket):
        logger.debug(
            "candidate_rejected", text=text[:50], reason="garbage", bucket=bucket
        )
        _trace_reject(text, bucket, "garbage")
        return False

    # Apply per-bucket minimum length check using MIN_LENGTH_BY_BUCKET
    text_clean = text.strip()
    min_length = MIN_LENGTH_BY_BUCKET.get(bucket, 2)  # Default to 2 chars
    if len(text_clean) < min_length:
        logger.debug(
            "candidate_rejected",
            text=text[:50],
            reason="too_short",
            bucket=bucket,
            min_length=min_length,
        )
        _trace_reject(text, bucket, "too_short")
        return False

    # Apply bucket-specific filters
    if bucket == BUCKET_ID_LIKE:
        if not is_valid_id_candidate(text):
            logger.debug(
                "candidate_rejected", text=text[:50], reason="invalid_id", bucket=bucket
            )
            _trace_reject(text, bucket, "invalid_id")
            return False

    elif bucket == BUCKET_AMOUNT_LIKE:
        if not is_valid_amount_candidate(text):
            logger.debug(
                "candidate_rejected",
                text=text[:50],
                reason="invalid_amount",
                bucket=bucket,
            )
            _trace_reject(text, bucket, "invalid_amount")
            return False

    elif bucket == BUCKET_DATE_LIKE:
        if not is_valid_date_candidate(text):
            logger.debug(
                "candidate_rejected",
                text=text[:50],
                reason="invalid_date",
                bucket=bucket,
            )
            _trace_reject(text, bucket, "invalid_date")
            return False

    elif bucket == BUCKET_NAME_LIKE:
        # For name-like candidates (company/person names)
        if not is_valid_name_candidate(text):
            logger.debug(
                "candidate_rejected",
                text=text[:50],
                reason="invalid_name",
                bucket=bucket,
            )
            _trace_reject(text, bucket, "invalid_name")
            return False

    elif bucket == BUCKET_KEYWORD_PROXIMAL:
        # Keyword-proximal candidates are kept for their position near
        # field keywords, not their text pattern. Applying strict name
        # validation here kills valid amounts/dates/IDs near keywords.
        # Only reject obvious noise: stopwords.
        if text_clean.lower() in STOPWORDS:
            logger.debug(
                "candidate_rejected",
                text=text[:50],
                reason="stopword",
                bucket=bucket,
            )
            _trace_reject(text, bucket, "stopword")
            return False

    # Random negatives are kept without additional filtering
    # (they're intentionally diverse for training purposes)

    return True


# =============================================================================
# BOOTSTRAP PRIORS FILTERING
# =============================================================================


def get_field_type_for_bucket(bucket: str) -> str:
    """Map candidate bucket to field type for bootstrap validation."""
    mapping = {
        "date_like": "date",
        "amount_like": "amount",
        "id_like": "invoice_number",
        "keyword_proximal": "text",
        "random_negative": "unknown",
    }
    return mapping.get(bucket, "unknown")


def passes_bootstrap_invoice_number_filter(text: str) -> bool:
    """Check if text passes bootstrap constraints for invoice numbers.

    Thresholds derived from schema/contract.invoice.json:
    - Length: min_length=1, max_length=50
    - format_hint includes pure-digit patterns (\\d{4,}) and mixed
      like "INV-001"
    - examples: "12345" (len 5, ratio 1.0), "INV-2024-001" (len 12,
      digit_ratio 0.58, special_ratio 0.17)

    Bootstrap filters reject obvious garbage only — the decoder's cost
    matrix handles fine-grained discrimination.
    """
    text = text.strip()
    if not text:
        return False
    length = len(text)
    # Schema: min_length=1, max_length=50. Use min 3 to reject noise
    # like single chars that slip through, max 50 from schema directly.
    if length < 3 or length > 50:
        return False
    digit_count = sum(1 for c in text if c.isdigit())
    # Must have at least one digit — invoice numbers always contain digits
    if digit_count == 0:
        return False
    # Relaxed digit ratio: "INV-2024-001" = 0.58, "PO-1" = 0.25.
    # Only reject candidates with nearly no digits (pure text).
    digit_ratio = digit_count / length
    if digit_ratio < 0.15:
        return False
    # Special char ratio: "A-123-456" = 0.22, "2024/INV/001" = 0.17.
    # Only reject when special chars dominate (>50% is garbage/addresses).
    special_count = sum(1 for c in text if not c.isalnum())
    special_ratio = special_count / length
    if special_ratio > 0.50:
        return False
    return True


def passes_bootstrap_amount_filter(text: str) -> bool:
    """Check if text passes bootstrap constraints for amounts.

    Thresholds derived from schema/contract.invoice.json:
    - TotalAmount.validation.max_value = 999_999_999.99
    - No min_value floor below zero (negatives are credits/refunds)

    Bootstrap filters reject unparseable garbage only — the decoder's
    cost matrix handles fine-grained discrimination.
    """
    text = text.strip()
    if not text:
        return False
    cleaned = text
    for symbol in ["$", "\u20ac", "\u00a3", "\u00a5", "\u20b9", "USD", "EUR", "GBP"]:
        cleaned = cleaned.replace(symbol, "")
    cleaned = cleaned.replace(",", "").replace(" ", "").strip()
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = "-" + cleaned[1:-1]
    try:
        amount = float(cleaned)
        # Schema max_value is 999_999_999.99. Symmetric for credits/refunds.
        return -999_999_999.99 <= amount <= 999_999_999.99
    except (ValueError, TypeError):
        return False


def passes_bootstrap_name_filter(text: str, field_type: str = "vendor_name") -> bool:
    """Check if text passes bootstrap constraints for names."""
    text = text.strip()
    if not text:
        return False
    length = len(text)
    if field_type == "vendor_name":
        if length < 2 or length > 50:
            return False
    else:
        if length < 3 or length > 100:
            return False
    if not any(c.isalpha() for c in text):
        return False
    return True


def compute_bootstrap_score(text: str, field_type: str) -> float:
    """Compute bootstrap-derived scoring adjustment. Returns -1.0 to +1.0."""
    text = text.strip()
    if not text or field_type == "unknown":
        return 0.0
    if field_type == "invoice_number":
        if passes_bootstrap_invoice_number_filter(text):
            length = len(text)
            digit_ratio = sum(1 for c in text if c.isdigit()) / length
            # Scale: 0.15 ratio (filter floor) → 0.1 bonus, 1.0 → 0.5 bonus
            return min(0.5, 0.1 + digit_ratio * 0.4)
        return -0.3
    if field_type == "amount":
        if passes_bootstrap_amount_filter(text):
            return 0.4
        return -0.4
    if field_type == "date":
        # Check if text contains a numeric date-like pattern
        separators = sum(1 for c in text if c in "/-.")
        digits = sum(1 for c in text if c.isdigit())
        if separators >= 2 and digits >= 4:
            return 0.3
        return 0.0
    if field_type in ("vendor_name", "customer_name", "text"):
        if passes_bootstrap_name_filter(text, field_type):
            return 0.2
        return -0.2
    return 0.0
