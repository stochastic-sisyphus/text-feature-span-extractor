"""Doc — projection type, not a runtime contract.

The byte-level immutable ledger (sha256 → bytes → JSONB payload) is the
only contract in the system.  Everything above it — Doc, tokens, spans,
review queue, dashboards — is a projection, computed live, lenient by
construction.

This module exists for typing convenience (IDE autocomplete, mypy hints,
documentation of expected shape).  It is NOT a runtime gate.  Hydration
of a Doc from arbitrary well-formed JSON cannot fail.  Drift is a
non-event because no field is required and no constraint is asserted at
hydrate time.

What changed vs. v56
--------------------

- Removed all ``Field(gt=...)`` / ``Field(ge=...)`` constraints.  They
  belong in the parse path (``tokenize.py``), not at the read boundary.
- Every field has a default (empty string, ``0.0``, empty tuple).  No
  field is required at hydrate.
- ``model_config = ConfigDict(extra="allow", frozen=False)``.  Frozen
  comes from immutability of the underlying JSONB row, not from
  Pydantic's runtime validator.  Extras are allowed because future
  parse versions will write fields this version of the code doesn't
  know about; a read happening on an older binary must still succeed.
- ``ContractMismatchError`` removed.  Versioning is informational.
  ``doc_schema_version`` is a string field for observability; reads
  never branch on it.
- ``compute_stable_token_id`` and other ephemeral derivation logic
  remains in ``views.py``.  This module declares shape only.

What this means for callers
---------------------------

- ``Doc.model_validate(payload)`` succeeds for any well-formed JSON
  (including ``{}``).  Missing nested objects materialize as their
  default (empty tuple, empty string).
- Code that previously caught ``ValidationError`` around hydrate can be
  simplified.  Hydrate cannot fail.
- Code that previously asserted on ``doc.pages[0].width > 0`` should
  guard the division (``width or 1.0``) — already the convention in
  ``views.py``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _LenientBase(BaseModel):
    """In-file projection base — lenient by construction.

    ``extra="allow"`` lets future parse versions write fields this binary
    doesn't know about without breaking reads.  ``frozen=False`` is
    intentional: immutability is a property of the JSONB row, not of the
    Pydantic instance; callers may annotate or stamp ephemeral attrs
    without triggering a ``FrozenInstanceError``.
    """

    model_config = ConfigDict(extra="allow", frozen=False)


# ---------------------------------------------------------------------------
# Version constant — observational, NOT a runtime gate.
# Producers may write this; readers MUST NOT branch on it.
# ---------------------------------------------------------------------------
DOC_SCHEMA_VERSION: str = "1"


# ---------------------------------------------------------------------------
# Nested payload models — projection shapes only.
# ---------------------------------------------------------------------------


class CharEntry(_LenientBase):
    """Per-character projection from ``page.chars``.

    Every field optional.  Geometry defaults to 0; downstream divides
    guard against zero.
    """

    text: str = ""
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    fontname: str = ""
    size: float = 0.0
    non_stroking_color: Any = None
    upright: bool = True


class VectorSegment(_LenientBase):
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    width: float = 0.0
    height: float = 0.0
    linewidth: float = 0.0
    stroking_color: Any = None


class RectRegion(_LenientBase):
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    width: float = 0.0
    height: float = 0.0
    stroking_color: Any = None
    non_stroking_color: Any = None


class EdgeSegment(_LenientBase):
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    width: float = 0.0
    height: float = 0.0
    orientation: str | None = None
    object_type: str = ""


class TableRegion(_LenientBase):
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    cells: tuple[tuple[float, float, float, float], ...] = ()
    # Native ``table.extract()`` output — one tuple per row, one cell per
    # column (cells are ``str | None`` because pdfplumber yields ``None`` for
    # empty cells).  Frozen here at parse time; the schema layer reads
    # ``len(rows)`` instead of approximating from cell ``top`` dedup.
    rows: tuple[tuple[str | None, ...], ...] = ()


# ---------------------------------------------------------------------------
# Per-page projection
# ---------------------------------------------------------------------------


class DocPage(_LenientBase):
    """Per-page projection.

    ``width`` and ``height`` are NOT constrained.  A degenerate page
    (width = 0) is a representable shape; downstream views read it as a
    page with no usable geometry and project to an empty span list.
    """

    page_idx: int = 0
    width: float = 0.0
    height: float = 0.0

    chars: tuple[CharEntry, ...] = ()
    vector_segments: tuple[VectorSegment, ...] = ()
    rects: tuple[RectRegion, ...] = ()
    edges: tuple[EdgeSegment, ...] = ()
    tables: tuple[TableRegion, ...] = ()
    tokens: tuple[Any, ...] = ()


# ---------------------------------------------------------------------------
# Top-level projection
# ---------------------------------------------------------------------------


class Doc(_LenientBase):
    """Doc — top-level projection.

    Hydrate cannot fail on any well-formed JSON.  ``sha256`` defaults to
    empty string for ``Doc.model_validate({})`` cases (which the read
    path now allows for graceful degradation; the route layer enforces
    "row-not-found → 404" on its own).
    """

    sha256: str = ""
    doc_schema_version: str = DOC_SCHEMA_VERSION
    pages: tuple[DocPage, ...] = ()

    @classmethod
    def from_payload(cls, payload: Any) -> Doc:
        """Canonical factory for constructing a Doc from a raw JSONB payload.

        Tolerant hydration: malformed payloads, type drift, and unknown
        fields cannot raise.  This is the read-boundary contract — only
        the byte-level ledger (sha256 -> bytes -> JSONB row) is durable;
        Doc is a projection that adapts to whatever shape the ledger
        happens to hold.

        Falls back through three tiers:
            1. cls.model_validate(payload)   — happy path, full type coerce
            2. cls.model_construct(**payload) — type mismatch fallback;
               Pydantic skips validation, fields land as-is
            3. cls()                          — payload not a dict, or
               construct itself raised
        """
        if not isinstance(payload, dict):
            return cls()
        try:
            return cls.model_validate(payload)
        except Exception:
            try:
                return cls.model_construct(**payload)
            except Exception:
                sha = payload.get("sha256") if isinstance(payload, dict) else None
                return cls(sha256=sha) if isinstance(sha, str) else cls()
