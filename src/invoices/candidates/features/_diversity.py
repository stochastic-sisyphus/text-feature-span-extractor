"""Type-stratified diversity sampling for candidate sets."""

from __future__ import annotations

from typing import Any


def diversity_sampling(
    candidates: list[dict[str, Any]], max_candidates: int = 200
) -> list[dict[str, Any]]:
    """Apply diversity sampling with type-based stratification."""
    from collections import defaultdict

    if len(candidates) <= max_candidates:
        return candidates

    # Phase 0.5: Reserve slots per type for early-page candidates (pages 0-2).
    # Multi-page invoices (e.g., 149 pages) have summary info on the first
    # few pages. Without type-aware reservation, critical amounts/IDs from
    # summary pages get drowned out by hundreds of candidates from later pages.
    #
    # Strategy: collect early-page candidates by type, sort by score DESC,
    # then pick the best per type. This ensures high-value candidates like
    # "AT&T" or "-$6,035.39" survive even when competing with many peers.
    max_reserved = max(1, max_candidates // 5)  # Reserve up to 20% of slots
    per_type_limit = max(5, max_reserved // 3)

    # Collect early-page candidates by type
    early_by_type: dict[str, list[dict[str, Any]]] = {
        "amount": [],
        "id": [],
        "date": [],
        "name": [],
    }
    non_early: list[dict[str, Any]] = []

    for candidate in candidates:
        page_idx = candidate.get("page_idx", 99)

        if page_idx <= 2:
            # Read the pre-assigned bucket — strip "_like" suffix to get ctype.
            raw_bucket: str = candidate.get("bucket", "") or ""
            ctype = (
                raw_bucket.replace("_like", "")
                if raw_bucket.endswith("_like")
                else None
            )

            if ctype and ctype in early_by_type:
                early_by_type[ctype].append(candidate)
                continue
        non_early.append(candidate)

    # Sort each type by score (descending) and pick top UNIQUE candidates.
    # Text-level dedup within each type ensures diverse candidates survive
    # (e.g., "AT&T" doesn't lose to 10 copies of "STATEMENT").
    reserved: list[dict[str, Any]] = []
    remaining_candidates: list[dict[str, Any]] = list(non_early)

    for ctype, early_cands in early_by_type.items():
        # For names, use page_frequency as a sorting boost since names
        # that repeat across pages (like vendor names) are more important
        # than one-off words that happen to score high.
        if ctype == "name":
            early_cands.sort(
                key=lambda x: (
                    -(x.get("total_score", 0.0) + x.get("page_frequency", 0.0) * 3.0)
                )
            )
        else:
            early_cands.sort(key=lambda x: -x.get("total_score", 0.0))
        seen_texts: set[str] = set()
        reserved_for_type: list[dict[str, Any]] = []
        overflow: list[dict[str, Any]] = []
        for c in early_cands:
            text_key = c.get("normalized_text", c.get("raw_text", "")).strip().lower()
            if text_key not in seen_texts and len(reserved_for_type) < per_type_limit:
                seen_texts.add(text_key)
                reserved_for_type.append(c)
            else:
                overflow.append(c)
        reserved.extend(reserved_for_type)
        remaining_candidates.extend(overflow)

    # Adjust quota for the main sampling to account for reserved slots
    adjusted_max = max_candidates - len(reserved)
    candidates = remaining_candidates

    # Group by type shape characteristics
    type_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        # Read the pre-assigned bucket — sub-binning uses text attributes only.
        text = candidate.get("raw_text", "").strip()
        bucket = candidate.get("bucket", "") or ""

        if bucket == "date_like":
            type_key = "date_numeric" if ("/" in text or "-" in text) else "date_text"
        elif bucket == "amount_like":
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count <= 2:
                type_key = "amount_small"
            elif digit_count <= 4:
                type_key = "amount_medium"
            else:
                type_key = "amount_large"
        elif bucket == "id_like":
            has_letters = any(c.isalpha() for c in text)
            has_separators = any(c in "-_#" for c in text)
            if has_letters and has_separators:
                type_key = "id_complex"
            elif has_letters:
                type_key = "id_alphanum"
            else:
                type_key = "id_numeric"
        elif bucket == "name_like":
            type_key = "name_short" if len(text) <= 10 else "name_long"
        else:
            type_key = "other"

        type_groups[type_key].append(candidate)

    # Phase 1: Guarantee small groups (<=5 members) get ALL their candidates.
    # Rare type groups are likely high-value (e.g., 2 currency candidates
    # shouldn't compete with 80 amount candidates for proportional quota).
    result: list[dict[str, Any]] = []
    large_groups: dict[str, list[dict[str, Any]]] = {}

    if len(type_groups) == 0:
        return reserved + candidates[:adjusted_max]

    for group_key, group_candidates in type_groups.items():
        group_candidates.sort(
            key=lambda x: (-x.get("total_score", 0.0), x.get("page_idx", 0))
        )
        if len(group_candidates) <= 5:
            result.extend(group_candidates)
        else:
            large_groups[group_key] = group_candidates

    # Phase 2: Fill remaining quota proportionally from large groups
    remaining = adjusted_max - len(result)
    total_large = len(large_groups)

    if remaining > 0 and total_large > 0:
        for group_candidates in large_groups.values():
            group_quota = max(1, remaining // total_large)
            slots_left = adjusted_max - len(result)
            actual_quota = min(group_quota, len(group_candidates), slots_left)
            result.extend(group_candidates[:actual_quota])

            if len(result) >= adjusted_max:
                break

    return reserved + result[:adjusted_max]
