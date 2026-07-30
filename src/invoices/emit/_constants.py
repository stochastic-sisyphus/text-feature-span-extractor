"""Module-level constants for emit package."""

from __future__ import annotations

EVALUATOR_VERSION = "cc2-v1"

_REASON_WEIGHT: dict[str, float] = {
    "ABSTAIN": 1.0,
    "MISSING": 0.9,
    "LOW_CONFIDENCE": 0.5,
}
