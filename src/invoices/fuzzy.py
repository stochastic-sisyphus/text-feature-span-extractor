"""Reusable fuzzy string matching helpers.

Wraps rapidfuzz.process.extractOne with token_set_ratio + default_process.
Returns the raw continuous score (0-100) - callers consume it as a feature.

Public API
----------
fuzzy_best_match(query, corpus) -> tuple[str, float] | None
"""

from __future__ import annotations

from collections.abc import Iterable

from rapidfuzz import fuzz as _fuzz
from rapidfuzz import process as _process
from rapidfuzz import utils as _utils


def fuzzy_best_match(
    query: str,
    corpus: Iterable[str],
) -> tuple[str, float] | None:
    """Return (best_match, score) or None if corpus is empty.

    Uses token_set_ratio with default_process so normalisation is consistent
    across every call site.  Returns the raw continuous score (0-100).
    """
    hit = _process.extractOne(
        query,
        corpus,
        scorer=_fuzz.token_set_ratio,
        processor=_utils.default_process,
    )
    if hit is None:
        return None
    return (hit[0], hit[1])
