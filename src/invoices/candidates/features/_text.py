"""Text feature extraction (length, digit ratio, hashed n-grams)."""

from __future__ import annotations

import hashlib
from typing import Any

from ..constants import CURRENCY_SYMBOLS


def compute_text_features_enhanced(text: str) -> dict[str, Any]:
    """Enhanced text features without regex."""
    text_clean = text.strip()

    if not text_clean:
        return {
            "text_length": 0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "currency_flag": False,
            "unigram_hash": "",
            "bigram_hash": "",
        }

    features: dict[str, Any] = {
        "text_length": len(text_clean),
        "digit_ratio": sum(1 for c in text_clean if c.isdigit()) / len(text_clean),
        "uppercase_ratio": sum(1 for c in text_clean if c.isupper()) / len(text_clean),
        "currency_flag": any(symbol in text_clean for symbol in CURRENCY_SYMBOLS),
    }

    # Hashed n-grams (safe string handling)
    words = text_clean.lower().split()

    # Unigram hashes (limit to prevent overflow)
    unigram_hashes = []
    for word in words[:3]:  # Limit to first 3
        hash_obj = hashlib.md5(
            word.encode("utf-8", errors="ignore"), usedforsecurity=False
        )
        unigram_hashes.append(hash_obj.hexdigest()[:8])
    features["unigram_hash"] = ",".join(unigram_hashes)

    # Bigram hashes
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(min(2, len(words) - 1))]
        bigram_hashes = []
        for bigram in bigrams:
            hash_obj = hashlib.md5(
                bigram.encode("utf-8", errors="ignore"), usedforsecurity=False
            )
            bigram_hashes.append(hash_obj.hexdigest()[:8])
        features["bigram_hash"] = ",".join(bigram_hashes)
    else:
        features["bigram_hash"] = ""

    return features
