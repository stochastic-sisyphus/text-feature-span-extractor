"""Frozen dataclasses for pipeline data flow.

These replace the ``dict[str, Any]`` patterns that made key-access errors
invisible until runtime.  Every field is explicit; accessing a nonexistent
attribute raises ``AttributeError`` at the access site instead of producing
a ``KeyError`` downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FallbackCandidate:
    """A runner-up candidate for a field assignment."""

    candidate_index: int
    cost: float
    candidate: dict[str, Any]
    used_ml_model: bool
    ml_probability: float | None


@dataclass(frozen=True, slots=True)
class Assignment:
    """One field's assignment from the Hungarian decoder.

    ``assignment_type`` is ``"CANDIDATE"`` when a token span was matched,
    or ``"NONE"`` when the decoder abstained.

    Immutable (frozen) so downstream code can trust the data won't mutate.
    """

    assignment_type: str  # "CANDIDATE" or "NONE"
    candidate_index: int | None
    cost: float
    field: str
    used_ml_model: bool
    ml_probability: float | None
    candidate: dict[str, Any] | None = None
    fallback_candidates: tuple[FallbackCandidate, ...] = ()
    # Populated by normalize_assignments; None on raw decoder output.
    normalized_value: str | None = None
    raw_text: str | None = None
    currency_code: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationState:
    """Calibration metrics computed from labeled data."""

    empirical_accuracy: float | None = None
    ece: float | None = None
