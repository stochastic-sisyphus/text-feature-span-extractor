"""Pattern classification and quality scoring for candidate text."""

from __future__ import annotations

from functools import lru_cache

from .constants import COMMON_NON_NAMES, MONTH_ABBREVS

# ---------------------------------------------------------------------------
# Single-source-of-truth bucket classifier
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _nlp() -> object:
    """Cached spaCy pipeline (en_core_web_sm, NER only)."""
    try:
        import spacy  # type: ignore[import]

        return spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    except (ImportError, OSError):
        import spacy  # type: ignore[import]

        return spacy.blank("en")


@lru_cache(maxsize=1)
def _nlp_blank() -> object:
    """Lightweight blank spaCy pipeline for token-attribute predicates (no model)."""
    import spacy  # type: ignore[import]

    return spacy.blank("en")


@lru_cache(maxsize=4096)
def classify_bucket(text: str) -> str | None:
    """Classify *text* into a BUCKET_* constant, or None if no bucket matches.

    Single source of truth for bucket assignment.  Priority order mirrors
    ``_classify_span_bucket`` (id first, then date, amount, name) so that
    bucket values remain stable across all call sites.

    Uses the same token-attribute logic as the hand-rolled predicates so that
    the set of accepted texts is a *superset* of the predicate union — NER is
    additive, never a gate.
    """
    from .constants import (
        BUCKET_AMOUNT_LIKE,
        BUCKET_DATE_LIKE,
        BUCKET_ID_LIKE,
        BUCKET_NAME_LIKE,
    )

    t = text.strip()
    if not t:
        return None

    text_lower = t.lower()
    has_month = any(m in text_lower for m in MONTH_ABBREVS)

    # 1. ID — most specific first; but month-names trump (e.g. "01-Jul-24")
    if not has_month and (is_clean_invoice_pattern(t) or is_id_like_soft(t)):
        return BUCKET_ID_LIKE

    # 2. Date
    if is_date_like_soft(t):
        return BUCKET_DATE_LIKE

    # 3. Amount
    if is_amount_like_soft(t):
        return BUCKET_AMOUNT_LIKE

    # 4. Name — try spaCy NER first as a fast-path, fall back to predicate
    try:
        doc = _nlp()(t)  # type: ignore[operator]
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "GPE"):
                return BUCKET_NAME_LIKE
    except Exception:
        pass

    if is_name_like_soft(t):
        return BUCKET_NAME_LIKE

    return None


@lru_cache(maxsize=4096)
def classify_bucket_with_strength(text: str) -> tuple[str | None, str | None]:
    """Return ``(bucket, strength)`` where strength is ``"strong"``, ``"soft"``, or None.

    * ``"strong"`` — clean structural pattern (is_clean_*) or NER entity match
    * ``"soft"``   — soft predicate match only
    * ``None``     — no bucket matched
    """
    from .constants import (
        BUCKET_AMOUNT_LIKE,
        BUCKET_DATE_LIKE,
        BUCKET_ID_LIKE,
        BUCKET_NAME_LIKE,
    )

    t = text.strip()
    if not t:
        return None, None

    text_lower = t.lower()
    has_month = any(m in text_lower for m in MONTH_ABBREVS)

    # ID
    if not has_month:
        if is_clean_invoice_pattern(t):
            return BUCKET_ID_LIKE, "strong"
        if is_id_like_soft(t):
            return BUCKET_ID_LIKE, "soft"

    # Date
    if is_clean_date_pattern(t):
        return BUCKET_DATE_LIKE, "strong"
    if is_date_like_soft(t):
        return BUCKET_DATE_LIKE, "soft"

    # Amount
    if is_clean_amount_pattern(t):
        return BUCKET_AMOUNT_LIKE, "strong"
    if is_amount_like_soft(t):
        return BUCKET_AMOUNT_LIKE, "soft"

    # Name — NER = strong, predicate = soft
    try:
        doc = _nlp()(t)  # type: ignore[operator]
        for ent in doc.ents:
            if ent.label_ in ("ORG", "PERSON", "GPE"):
                return BUCKET_NAME_LIKE, "strong"
    except Exception:
        pass
    if is_name_like_soft(t):
        return BUCKET_NAME_LIKE, "soft"

    return None, None


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for deduplication purposes only."""
    return text.strip().lower().replace(" ", "").replace(",", "").replace(".", "")


@lru_cache(maxsize=4096)
def is_date_like_soft(text: str) -> bool:
    """Soft check if text looks like a date."""
    t = text.strip()
    if len(t) < 4 or len(t) > 20:
        return False
    # Month abbreviation fast-path (covers "Jan 2024", "15-Jan-24", etc.)
    if any(m in t.lower() for m in MONTH_ABBREVS):
        return True
    doc = _nlp_blank()(t)  # type: ignore[operator]
    tokens = list(doc)
    # Single token with slash separator: "01/15/2024" → shape "dd/dd/dddd"
    if len(tokens) == 1:
        s = tokens[0].shape_
        if "/" in s and any(c == "d" for c in s):
            return True
    # Multi-token: digit-separator-digit pattern (hyphenated dates)
    num_toks = [tok for tok in tokens if tok.like_num]
    sep_toks = [tok for tok in tokens if tok.shape_ in ("-", "/")]
    if len(num_toks) >= 2 and len(sep_toks) >= 1:
        total_digits = sum(len(tok.text) for tok in num_toks)
        if total_digits >= 4:
            return True
    return False


@lru_cache(maxsize=4096)
def is_amount_like_soft(text: str) -> bool:
    """Soft check if text looks like a monetary amount."""
    t = text.strip()
    if not t:
        return False
    doc = _nlp_blank()(t)  # type: ignore[operator]
    for tok in doc:
        if tok.is_currency:
            return True
        if tok.like_num:
            s = tok.shape_
            if s.count(".") == 1 or "," in s:
                return True
    return False


@lru_cache(maxsize=4096)
def is_id_like_soft(text: str) -> bool:
    """Soft check if text looks like an ID."""
    t = text.strip()
    if len(t) < 3 or len(t) > 30:
        return False
    if not all(c.isalnum() or c in "-_#" for c in t):
        return False
    doc = _nlp_blank()(t)  # type: ignore[operator]
    for tok in doc:
        s = tok.shape_
        has_digit = "d" in s
        has_alpha = "x" in s.lower()
        # Accept: mixed alpha+digit shape (INV-12345, ABC123) or pure digit (123, 75080)
        if has_digit and (has_alpha or tok.is_digit):
            return True
    return False


def is_name_like_soft(text: str) -> bool:
    """Soft check if text looks like a company or person name.

    Matches patterns like:
    - "AT&T"
    - "WEALTH ADVISORS INC."
    - "GTT Communications"
    - "A. M. Castle & Co."
    - "Comcast"

    Does NOT match common words like:
    - "Jul", "Aug", "Issue", "Account", "Make", etc.
    """
    text = text.strip()
    if len(text) < 2 or len(text) > 60:
        return False

    # Must have some alphabetic content
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return False

    text_lower = text.lower().rstrip(".,")
    if text_lower in COMMON_NON_NAMES:
        return False

    # Check for special name-like characteristics
    has_ampersand = "&" in text
    has_period_abbreviation = ". " in text or text.endswith(".")
    words = text.split()
    num_words = len(words)

    # Check if it starts with uppercase (proper nouns)
    first_alpha = None
    for c in text:
        if c.isalpha():
            first_alpha = c
            break

    if first_alpha and first_alpha.isupper():
        allowed_chars = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,&'-"
        )
        if all(c in allowed_chars for c in text):
            # Accept if it has special name characteristics
            if has_ampersand:  # AT&T, A. M. Castle & Co.
                return True
            if has_period_abbreviation and num_words >= 2:  # Inc., LLC., Co.
                return True
            if num_words >= 2:  # Multi-word names like "Wealth Advisors"
                return True
            # Single uppercase word - only accept if ALL CAPS or mixed case
            if num_words == 1 and len(text) >= 3:
                # Accept ALL CAPS single words (company acronyms)
                if text.isupper():
                    return True
                # Accept mixed case like "Comcast", "Microsoft"
                # Must have lowercase after the first char
                if any(c.islower() for c in text[1:]):
                    return True

    # Also accept ALL CAPS multi-word names like "WEALTH ADVISORS INC."
    if num_words >= 2:
        all_caps_words = sum(
            1 for w in words if w.isupper() or w.rstrip(".,").isupper()
        )
        if all_caps_words >= num_words * 0.5:  # At least 50% all-caps words
            return True

    return False


# =============================================================================
# PATTERN QUALITY SCORING
# =============================================================================


def is_clean_invoice_pattern(text: str) -> bool:
    """Check if text matches a clean invoice number pattern.

    Matches patterns like: US002650-41, INV-12345, PO-2024-001, etc.
    Pattern: 2-4 letters, optional separator, 4+ digits
    """
    text = text.strip()
    if len(text) < 5 or len(text) > 25:
        return False

    # Count character types
    letters = sum(1 for c in text if c.isalpha())
    digits = sum(1 for c in text if c.isdigit())
    separators = sum(1 for c in text if c in "-_")

    # Must have letters at start (2-4), followed by digits (4+)
    if letters < 2 or digits < 4:
        return False

    # Total should be letters + digits + reasonable separators
    if letters + digits + separators != len(text):
        return False

    # Check that it starts with letters (common invoice pattern)
    first_chars = ""
    for c in text:
        if c.isalpha():
            first_chars += c
        else:
            break

    if len(first_chars) >= 2:
        return True

    return False


def is_clean_date_pattern(text: str) -> bool:
    """Check if text matches a clean date pattern.

    Matches: MM/DD/YYYY, DD-MM-YYYY, YYYY-MM-DD, etc.
    """
    text = text.strip()
    if len(text) < 6 or len(text) > 12:
        return False

    digits = sum(1 for c in text if c.isdigit())
    separators = sum(1 for c in text if c in "/-")

    # Clean date has 4-8 digits and exactly 2 separators
    if digits < 4 or digits > 8:
        return False
    if separators != 2:
        return False

    # Must only contain digits and separators
    if digits + separators != len(text):
        return False

    return True


def is_clean_amount_pattern(text: str) -> bool:
    """Check if text matches a clean monetary amount pattern.

    Matches: $1,234.56, 1234.00, 50.00, etc.
    """
    text = text.strip()
    if len(text) < 1 or len(text) > 15:
        return False

    # Remove currency symbol if present
    clean = text.lstrip("$\u20ac\u00a3\u00a5\u20b9\u20bd")
    if not clean:
        return False

    digits = sum(1 for c in clean if c.isdigit())
    decimals = clean.count(".")

    # Must be mostly digits
    if digits < 2:
        return False

    # Clean amount: digits + at most one decimal + any number of comma separators
    if decimals > 1:
        return False

    # Check that non-digit chars are only . or ,
    for c in clean:
        if not c.isdigit() and c not in ".,":
            return False

    return True


def is_single_clean_token(text: str) -> bool:
    """Check if text is a single clean alphanumeric token (good for IDs).

    Matches single tokens like: US002650-41, 12345, ABC123, etc.
    No spaces, reasonable length, alphanumeric with optional separators.
    """
    text = text.strip()

    # No spaces allowed
    if " " in text:
        return False

    if len(text) < 3 or len(text) > 20:
        return False

    # Must be alphanumeric with optional separators
    for c in text:
        if not c.isalnum() and c not in "-_#":
            return False

    # Must have at least some alphanumeric content
    alphanum = sum(1 for c in text if c.isalnum())
    if alphanum < 3:
        return False

    return True


def has_repetitive_text(text: str) -> bool:
    """Check if text has >50% repeated words (garbage indicator).

    Detects patterns like: "1 1 1 1 1 1 1 1" which are garbage candidates.
    """
    words = text.strip().split()
    if len(words) <= 1:
        return False

    # Count word frequencies
    word_counts: dict[str, int] = {}
    for word in words:
        word_lower = word.lower()
        word_counts[word_lower] = word_counts.get(word_lower, 0) + 1

    # Find most frequent word
    max_count = max(word_counts.values())

    # If most frequent word appears in >50% of positions, it's repetitive
    if max_count / len(words) > 0.5:
        return True

    return False


def compute_pattern_score_bonus(text: str, token_count: int) -> float:
    """Compute pattern-based score bonus for clean, well-formed text.

    Returns positive bonus for clean patterns, negative penalty for garbage.

    Args:
        text: The candidate text
        token_count: Number of tokens in the span

    Returns:
        Score bonus (positive for good patterns, negative for garbage)
    """
    bonus = 0.0

    # Penalize repetitive text (garbage like "1 1 1 1 1 1 1 1")
    if has_repetitive_text(text):
        bonus -= 2.0

    # Bonus for clean invoice patterns
    if is_clean_invoice_pattern(text):
        bonus += 2.0

    # Bonus for clean date patterns
    if is_clean_date_pattern(text):
        bonus += 2.0

    # Bonus for clean amount patterns
    if is_clean_amount_pattern(text):
        bonus += 2.0

    # Bonus for single clean alphanumeric tokens (good for IDs)
    if is_single_clean_token(text):
        bonus += 1.5

    return bonus


def compute_bucket_probabilities(text: str) -> dict[str, float]:
    """Compute soft probability distribution over bucket types for a text span.

    Instead of hard-assigning a single bucket, this returns a probability for
    each bucket type based on how well the text matches each pattern. This
    enables the decoder to use soft bucket affinity instead of binary matching.

    Each bucket is scored independently (a text can weakly match multiple buckets).
    strong (clean structural pattern or NER entity) = 1.0; soft predicate = 0.7.

    Args:
        text: The candidate text to classify

    Returns:
        Dict mapping bucket type string to probability [0, 1], summing to 1.0.
        Keys use the same bucket names as constants (date_like, amount_like, etc.)
    """
    from .constants import (
        BUCKET_AMOUNT_LIKE,
        BUCKET_DATE_LIKE,
        BUCKET_ID_LIKE,
        BUCKET_NAME_LIKE,
    )

    scores: dict[str, float] = {}

    # Date: strong match for clean date patterns, moderate for soft
    if is_clean_date_pattern(text):
        scores[BUCKET_DATE_LIKE] = 1.0
    elif is_date_like_soft(text):
        scores[BUCKET_DATE_LIKE] = 0.7
    else:
        scores[BUCKET_DATE_LIKE] = 0.0

    # Amount: strong for clean, moderate for soft
    if is_clean_amount_pattern(text):
        scores[BUCKET_AMOUNT_LIKE] = 1.0
    elif is_amount_like_soft(text):
        scores[BUCKET_AMOUNT_LIKE] = 0.7
    else:
        scores[BUCKET_AMOUNT_LIKE] = 0.0

    # ID: strong for clean invoice, moderate for soft
    if is_clean_invoice_pattern(text):
        scores[BUCKET_ID_LIKE] = 1.0
    elif is_id_like_soft(text):
        scores[BUCKET_ID_LIKE] = 0.7
    else:
        scores[BUCKET_ID_LIKE] = 0.0

    # Name: NER entity = strong, soft predicate = soft
    try:
        doc = _nlp()(text)  # type: ignore[operator]
        if any(ent.label_ in ("ORG", "PERSON", "GPE") for ent in doc.ents):
            scores[BUCKET_NAME_LIKE] = 1.0
        elif is_name_like_soft(text):
            scores[BUCKET_NAME_LIKE] = 0.7
        else:
            scores[BUCKET_NAME_LIKE] = 0.0
    except Exception:
        scores[BUCKET_NAME_LIKE] = 0.7 if is_name_like_soft(text) else 0.0

    # Normalize to probability distribution
    total = sum(scores.values())
    if total > 0:
        return {k: v / total for k, v in scores.items()}

    # If nothing matched, return uniform low probs (span is ambiguous)
    return dict.fromkeys(scores, 1.0 / len(scores))


def compute_token_count_penalty(token_count: int) -> float:
    """Compute penalty/bonus based on token count.

    Prefer short, clean spans over long garbage spans.

    Args:
        token_count: Number of tokens in the span

    Returns:
        Score adjustment (positive for 1-2 tokens, negative for 4+)
    """
    if token_count == 1:
        return 0.5  # Strong bonus for single tokens
    if token_count == 2:
        return 0.3  # Good bonus for 2-token spans
    if token_count == 3:
        return 0.1  # Small bonus for 3-token spans
    if token_count == 4:
        return 0.0  # Neutral for 4-token spans
    # Penalty for 5+ tokens (shouldn't happen with max_span_tokens=4, but defensive)
    return -0.5 * (token_count - 4)
