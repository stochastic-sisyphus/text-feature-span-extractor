"""Schema: single source of truth for field metadata.

Loads contract schema once and provides typed accessors for all field properties.
FieldSpec frozen dataclass captures all field metadata in one typed object.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .field_def import (
    AnchorBonusOverride,
    FieldDef,
    SpatialBias,
    parse_field_definitions,
)

if TYPE_CHECKING:
    from ..doc import Doc

# Typed FieldDef cache — seeded at startup via set().  The ONLY field-level cache slot.
_cached_defs: dict[str, FieldDef] | None = None

# Top-level schema metadata (version, fields list) — everything except field_definitions.
_cached_meta: dict[str, Any] | None = None

# Cross-field rules list — stored separately so cross_field_rules() never touches raw dicts.
_cached_cross_field_rules: list[dict[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """All field metadata from the contract schema in one typed, frozen object.

    Every field in the contract schema maps to exactly one ``FieldSpec`` instance,
    constructed by ``build_field_spec()`` and cached for the lifetime of the loaded
    schema.  All slots are sourced directly from the corresponding ``FieldDef``
    Pydantic model so there is no raw-dict look-up at call sites.

    ``spatial_bias`` and ``anchor_bonus_override`` are typed sub-models
    (``SpatialBias``, ``AnchorBonusOverride``) rather than plain dicts.
    """

    name: str
    base_type: str
    normalizer: str
    anchor_family: str | None
    bucket_prefs: tuple[str, ...]
    is_footer: bool
    is_header: bool
    is_keyword_proximal: bool
    priority_bonus: float
    importance: float
    confidence_threshold: float
    computed: bool
    required: bool
    dataverse_column: str | None
    format_hint: str | None
    format_penalty: float
    spatial_bias: SpatialBias | None
    anchor_bonus_override: AnchorBonusOverride | None
    is_vendor: bool = False
    line_item_field: bool = False


def load() -> dict[str, Any]:
    """Return the in-memory schema as a plain dict.

    Reconstructed from typed slots so callers that need the raw envelope
    (version, fields, cross_field_rules, field_definitions) keep working
    without a live raw-dict cache slot.

    Raises ConfigurationError if not seeded.
    """
    if _cached_defs is None or _cached_meta is None:
        return {"field_definitions": {}, "cross_field_rules": [], "fields": []}
    field_definitions = {
        name: fd.model_dump(exclude={"name"}) for name, fd in _cached_defs.items()
    }
    return {
        **_cached_meta,
        "field_definitions": field_definitions,
        "cross_field_rules": _cached_cross_field_rules or [],
    }


def set(data: dict[str, Any]) -> None:
    """Seed or update the in-memory schema cache."""
    global _cached_defs, _cached_meta, _cached_cross_field_rules
    # Parse field_definitions into typed FieldDef objects.
    raw_field_defs: dict[str, Any] = data.get("field_definitions", {})
    _cached_defs = parse_field_definitions(raw_field_defs)
    # Store cross_field_rules separately.
    _cached_cross_field_rules = list(data.get("cross_field_rules", []))
    # Store remaining top-level metadata (version, fields, anything else).
    _cached_meta = {
        k: v
        for k, v in data.items()
        if k not in ("field_definitions", "cross_field_rules")
    }
    build_field_spec.cache_clear()
    get_required_field_names.cache_clear()


def clear() -> None:
    """Reset the schema cache (for testing)."""
    global _cached_defs, _cached_meta, _cached_cross_field_rules
    _cached_defs = None
    _cached_meta = None
    _cached_cross_field_rules = None
    build_field_spec.cache_clear()
    get_required_field_names.cache_clear()


def load_field_defs(*, include_deprecated: bool = False) -> dict[str, FieldDef]:
    """Return the typed FieldDef cache. Raises ConfigurationError if not seeded.

    By default filters out deprecated fields so operational callers (pipeline,
    training, decoder) only see active + planned fields. Pass
    ``include_deprecated=True`` for admin/audit paths.
    """
    if _cached_defs is not None:
        if include_deprecated:
            return _cached_defs
        return {k: v for k, v in _cached_defs.items() if v.status != "deprecated"}
    return {}


def cross_field_rules() -> list[dict[str, Any]]:
    """Return cross-field validation rules from schema."""
    if _cached_cross_field_rules is None:
        return []
    return _cached_cross_field_rules


def clear_cache() -> None:
    """Clear cached schema and FieldSpec cache (for testing)."""
    clear()


# --- FieldSpec builders ---


@functools.cache
def build_field_spec(name: str) -> FieldSpec:
    """Construct a FieldSpec from the typed FieldDef registry.

    All slots are sourced from the ``FieldDef`` Pydantic model; no raw-dict
    look-ups.  Uses ``functools.cache`` so each field is constructed at most
    once per schema load.

    Args:
        name: Field name (original case, must exist in the loaded schema)

    Returns:
        FieldSpec with all metadata for the field
    """
    typed_def = load_field_defs()[name]
    return FieldSpec(
        name=name,
        base_type=typed_def.base_type,
        normalizer=typed_def.normalizer,
        anchor_family=typed_def.anchor_family,
        bucket_prefs=typed_def.bucket_preference,
        is_footer=typed_def.spatial_region == "footer",
        is_header=typed_def.spatial_region == "header",
        is_keyword_proximal=typed_def.keyword_proximal,
        priority_bonus=typed_def.priority_bonus,
        importance=typed_def.importance,
        confidence_threshold=typed_def.confidence_threshold,
        computed=typed_def.computed,
        required=typed_def.required,
        dataverse_column=typed_def.dataverse_column,
        format_hint=typed_def.format_hint,
        format_penalty=typed_def.format_penalty,
        spatial_bias=typed_def.spatial_bias,
        anchor_bonus_override=typed_def.anchor_bonus_override,
        is_vendor=typed_def.is_vendor,
        line_item_field=typed_def.line_item_field,
    )


def get_all_field_specs() -> dict[str, FieldSpec]:
    """Return all field specs keyed by name."""
    return {name: build_field_spec(name) for name in load_field_defs()}


def get_field_names() -> list[str]:
    """Return all field names from schema."""
    return list(load_field_defs().keys())


def build_field_specs(fields: list[str]) -> dict[str, FieldSpec]:
    """Pre-build FieldSpecs for all fields at once.

    Call this once before the cost matrix loop to avoid repeated
    schema registry lookups per cell.

    Args:
        fields: List of field names to build specs for

    Returns:
        Dict mapping field name -> FieldSpec
    """
    return {field: build_field_spec(field) for field in fields}


@functools.cache
def get_required_field_names() -> frozenset[str]:
    """Return field names that must have a trained ranker.

    Required: required=True in the field definition AND status != 'planned'.
    The planned-status exclusion keeps LineItem* fields from gating model load
    before line-items are implemented.
    """
    return frozenset(
        name
        for name, fd in load_field_defs().items()
        if fd.required and fd.status != "planned"
    )


def _largest_table_row_count(doc: Doc) -> int:
    """Return the row count of the largest (by bbox area) table across all pages.

    Uses the table with the greatest bounding-box area as the canonical
    line-items table — consistent with how tokenize.py locates the primary
    table on a page.  Row count comes straight from pdfplumber's
    ``table.extract()`` output (frozen at parse time into
    ``TableRegion.rows``); no cell-bbox approximation.  Returns 0 when the
    Doc has no tables.
    """
    best_rows = 0
    best_area = 0.0
    for page in doc.pages:
        for table in page.tables:
            x0, top, x1, bottom = table.bbox
            area = (x1 - x0) * (bottom - top)
            if area > best_area:
                best_area = area
                best_rows = len(table.rows)
    return best_rows


def expand_fields(
    specs: tuple[FieldSpec, ...],
    doc: Doc,
) -> tuple[FieldSpec, ...]:
    """Expand line-item FieldSpecs into per-row variants based on the largest table.

    Non-line-item specs pass through unchanged.  For each line-item spec the
    function emits ``n_rows`` new specs named ``{base}__row_{i}`` (0-indexed),
    carrying all original metadata.  The ``line_item_field`` flag is preserved
    on row variants so downstream groupers can identify them.

    Table selection: the largest table by bounding-box area across all pages
    is used as the canonical line-items table.  When no tables are detected,
    line-item specs are dropped (no rows to emit).

    Ordering is deterministic: non-line-item specs come first in their
    original order; row variants are appended in field-declaration order,
    then by row index.
    """
    n_rows = _largest_table_row_count(doc)

    result: list[FieldSpec] = []
    for spec in specs:
        if not spec.line_item_field:
            result.append(spec)
            continue
        for i in range(n_rows):
            result.append(dataclasses.replace(spec, name=f"{spec.name}__row_{i}"))  # noqa: PERF401 — outer loop has continue; comprehension would obscure the branching structure
    return tuple(result)
