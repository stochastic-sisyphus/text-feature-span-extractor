"""FieldDef — frozen Pydantic v2 model for per-field metadata.

Replaces the raw ``dict[str, Any]`` previously stored in FieldSpec.field_def.
Each field in the contract schema maps to one FieldDef instance, decomposing
the legacy single ``type`` string into four independent axes.

Four axes
---------
base_type    — output dtype (str / decimal / date / int)
normalizer   — validation/scoring strategy in cost matrix and semantic validation
anchor_family — spatial co-occurrence group for anchor bonuses
keyword_proximal — whether keyword proximity boosting applies

Candidate-sharing axis
----------------------
share_candidate_with — tuple of field names that should also see this field's
    candidates.  Causes candidates_df() to emit synthetic duplicate rows so
    the Hungarian solver can assign the same physical span to multiple fields.

Legacy type → axes mapping (used only in parse_field_definitions pretransform):
  id       → str,     id,          id
  date     → date,    date,        date
  amount   → decimal, amount,      total
  text     → str,     name,        name
  address  → str,     name,        name
  currency → str,     currency,    None
  email    → str,     email,       None
  phone    → str,     phone,       None
  number   → int,     passthrough, total
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

# ---------------------------------------------------------------------------
# Type aliases — exported so api_models.py and other callers can import them
# ---------------------------------------------------------------------------

BaseType = Literal[
    "str",
    "int",
    "float",
    "bool",
    "bytes",
    "decimal",
    "date",
    "datetime",
    "time",
    "timedelta",
    "uuid",
]
Normalizer = Literal[
    "amount", "date", "id", "name", "passthrough", "currency", "email", "phone"
]
AnchorFamily = Literal["total", "date", "id", "name", "tax"]
ComputedFnLiteral = Literal["infer_currency", "concat_strip", "coalesce"]
Bucket = Literal["amount_like", "date_like", "id_like", "name_like", "keyword_proximal"]
SpatialRegion = Literal["header", "footer"]

# ---------------------------------------------------------------------------
# One-time migration seed — NOT persisted anywhere as code.
# Used only in parse_field_definitions() to upgrade legacy JSON entries.
# ---------------------------------------------------------------------------

_LEGACY_TYPE_MAP: dict[str, tuple[BaseType, Normalizer, AnchorFamily | None]] = {
    "id": ("str", "id", "id"),
    "date": ("date", "date", "date"),
    "amount": ("decimal", "amount", "total"),
    "text": ("str", "name", "name"),
    "address": ("str", "name", "name"),
    "currency": ("str", "currency", None),
    "email": ("str", "email", None),
    "phone": ("str", "phone", None),
    "number": ("int", "passthrough", "total"),
}

# Default keyword_proximal for legacy type strings (explicit JSON value wins)
_LEGACY_KP_DEFAULTS: frozenset[str] = frozenset({"id", "amount", "date"})


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class SpatialBias(BaseModel):
    """Position-based penalty for cost matrix scoring.

    Pipeline-internal — pipeline code constructs these, Grafana UI never
    exposes sub-keys for the operator to extend. extra="forbid" is intentional:
    unrecognized keys here represent a pipeline misconfiguration, not user data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: Literal["top", "bottom"]
    penalty: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)


class AnchorBonusOverride(BaseModel):
    """Extra directional bonuses for a specific anchor type.

    Pipeline-internal — pipeline code constructs these, Grafana UI never
    exposes sub-keys for the operator to extend. extra="forbid" is intentional:
    unrecognized keys here represent a pipeline misconfiguration, not user data.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    anchor: str
    below_bonus: float = 0.0
    below_dist_threshold: float = 0.3
    reading_order_bonus: float = 0.0
    reading_order_dist_threshold: float = 0.2


# ---------------------------------------------------------------------------
# FieldDef — the new Pydantic v2 model
# ---------------------------------------------------------------------------


class FieldDef(BaseModel):
    """Frozen per-field metadata from the contract schema.

    Constructed via ``parse_field_definitions()`` — not instantiated directly
    from raw JSON (the pretransform handles legacy-key mapping first).
    """

    model_config = ConfigDict(frozen=True, extra="allow", str_strip_whitespace=True)

    @field_validator(
        "bucket_preference",
        "examples",
        "share_candidate_with",
        "computed_from",
        mode="before",
    )
    @classmethod
    def _coerce_none_to_empty_tuple(cls, v: Any) -> Any:
        return () if v is None else v

    # Identity
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = ""

    # Four decomposed axes
    base_type: BaseType = "str"
    normalizer: Normalizer = "passthrough"
    anchor_family: AnchorFamily | None = None
    bucket_preference: tuple[Bucket, ...] = ()
    keyword_proximal: bool = False

    # Scoring
    required: bool = False
    confidence_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    priority_bonus: float = Field(default=0.0, ge=0.0)
    spatial_region: SpatialRegion | None = None
    spatial_bias: SpatialBias | None = None
    anchor_bonus_override: AnchorBonusOverride | None = None

    # Validation hints
    format_hint: str | None = None
    format_penalty: float = Field(default=0.3, ge=0.0)
    examples: tuple[str, ...] = ()

    # Candidate sharing
    share_candidate_with: tuple[str, ...] = ()
    """Field names that share candidate spans with this field.

    When non-empty, ``candidates_df()`` emits one extra candidate row per
    entry, with a synthetic ``candidate_id = sha1(original_id + "|" + name)``
    and ``shared_for_field`` set to the entry name.  This lets the Hungarian
    solver assign the same physical span to multiple fields independently.

    Coherence note: every name in this tuple must reference an existing field
    in the loaded schema.  That check is deferred to contract-load time
    (``parse_field_definitions``/schema registry) — not enforced here.
    """

    # Lifecycle
    status: Literal["active", "planned", "deprecated"] = "active"
    deprecation_reason: str | None = None
    introduced_in: str | None = None
    computed: bool = False
    computed_from: tuple[str, ...] = ()
    computed_fn: ComputedFnLiteral | None = None
    default_value: str | None = None
    """Literal output value when extraction yields null and no computed_fn is set
    or computed_fn returns null.  Part of the emit-time fallback chain:
    extracted → computed_fn → default_value → null."""
    line_item_field: bool = False
    dataverse_column: str | None = None
    is_vendor: bool = False
    """Marks THIS field as THE vendor field for vendor-corpus fuzzy matching.

    Exactly one field in the schema should be True. The resolver prefers this
    flag over is_header heuristics when identifying which field to use as the
    fuzzy-prior corpus source.
    """

    @model_validator(mode="after")
    def validate_coherence(self) -> Self:
        if self.status == "deprecated" and not self.deprecation_reason:
            raise ValueError(
                f"'{self.name}': deprecation_reason is required when status='deprecated'"
            )
        if self.keyword_proximal and self.anchor_family is None:
            raise ValueError(
                f"'{self.name}': keyword_proximal=True requires anchor_family to be set"
            )
        if self.computed and not self.computed_from:
            raise ValueError(
                f"'{self.name}': computed=True requires computed_from to be non-empty"
            )
        # computed_fn functions that require source fields: coalesce and concat_strip
        # need at least one computed_from entry to do anything useful.
        # infer_currency scans assignments directly and may have an empty computed_from,
        # so we do not gate it here (preserve existing behaviour).
        _FNS_REQUIRING_SOURCES: frozenset[str] = frozenset({"coalesce", "concat_strip"})
        if self.computed_fn in _FNS_REQUIRING_SOURCES and not self.computed_from:
            raise ValueError(
                f"'{self.name}': computed_fn='{self.computed_fn}' requires "
                f"computed_from to be non-empty"
            )
        # Note: spatial_bias and spatial_region are independent axes.
        # spatial_bias encodes cost-penalty positioning (top/bottom penalty on cost matrix).
        # spatial_region encodes the field's logical document zone (header/footer).
        # A field can have a cost-position bias without being a logical header/footer field.
        # Example: InvoiceDate has spatial_bias.position="top" but is not a footer field.
        return self


# ---------------------------------------------------------------------------
# Module-level TypeAdapter — constructed once, reused for bulk validation
# ---------------------------------------------------------------------------

from pydantic import TypeAdapter  # noqa: E402

_FIELD_DEF_ADAPTER: TypeAdapter[dict[str, FieldDef]] = TypeAdapter(dict[str, FieldDef])


def parse_field_definitions(raw: dict[str, Any]) -> dict[str, FieldDef]:
    """Parse a ``field_definitions`` dict (from contract JSON) into typed FieldDef instances.

    Handles legacy-shaped entries that carry a single ``type`` string by
    expanding it into the four decomposed axes (``base_type``, ``normalizer``,
    ``anchor_family``, ``keyword_proximal``).  All other legacy-only keys
    (``anchor_override``) are consumed here and NOT forwarded to FieldDef.

    Unknown keys from Postgres drift survive — ``FieldDef`` uses ``extra="allow"``
    so operator-added content round-trips cleanly through the model.

    Args:
        raw: The ``field_definitions`` mapping from the active contract schema
            (postgres-backed, seeded from contract.invoice.seed.json on first boot).

    Returns:
        ``dict[str, FieldDef]`` keyed by field name.

    Raises:
        pydantic.ValidationError: If any entry fails axis constraints or
            coherence rules.  Error messages are localized to the field name.
    """
    transformed: dict[str, Any] = {}

    for field_name, entry in raw.items():
        d = (
            entry.model_dump() if isinstance(entry, FieldDef) else dict(entry)
        )  # shallow copy — don't mutate original

        # Inject field name (it's the dict key, not in the entry itself)
        d["name"] = field_name

        # --- Legacy single-type migration ---
        legacy_type = d.pop("type", None)
        if legacy_type is not None:
            base_type, normalizer, anchor_family = _LEGACY_TYPE_MAP.get(
                legacy_type, ("str", "passthrough", None)
            )
            # Only set axes from legacy if not already explicitly present
            d.setdefault("base_type", base_type)
            d.setdefault("normalizer", normalizer)
            d.setdefault("anchor_family", anchor_family)

            # keyword_proximal: explicit JSON value wins; else legacy default
            if "keyword_proximal" not in d:
                d["keyword_proximal"] = legacy_type in _LEGACY_KP_DEFAULTS

        # --- anchor_override: consumed here, maps to anchor_family ---
        # anchor_override is a legacy key; if explicitly set, it overrides
        # the type-derived anchor_family (including null overrides).
        if "anchor_override" in d:
            anchor_override = d.pop("anchor_override")
            if anchor_override is not None:
                d["anchor_family"] = anchor_override
            else:
                # Explicit null anchor_override → no anchor family
                d["anchor_family"] = None

        # --- Defensive: computed=True requires computed_from. Legacy/corrupt entries
        # that set computed without computed_from would crash validate_coherence.
        # Coerce to computed=False and log a warning so startup survives corrupt DB state.
        if d.get("computed") is True:
            cf = d.get("computed_from")
            if not cf:  # None, missing, or empty list/tuple
                logging.getLogger(__name__).warning(
                    "Field %r has computed=True but no computed_from — coercing to computed=False. "
                    "Fix the schema entry (likely a corrupt Postgres contract_schema row).",
                    field_name,
                )
                d["computed"] = False

        # --- Defensive: coalesce/concat_strip require computed_from.
        # If a corrupt DB row sets one of these fns without computed_from, clear the fn
        # so startup survives rather than crashing in validate_coherence.
        _FNS_REQUIRING_SOURCES = frozenset({"coalesce", "concat_strip"})
        fn = d.get("computed_fn")
        if fn in _FNS_REQUIRING_SOURCES:
            cf = d.get("computed_from")
            if not cf:
                logging.getLogger(__name__).warning(
                    "Field %r has computed_fn=%r but no computed_from — coercing computed_fn to None. "
                    "Fix the schema entry (likely a corrupt Postgres contract_schema row).",
                    field_name,
                    fn,
                )
                d["computed_fn"] = None

        # --- bucket_preference: JSON has list, FieldDef expects tuple ---
        # TypeAdapter handles list→tuple coercion for tuple[X, ...] fields,
        # but we normalise here to be explicit.
        if "bucket_preference" in d and isinstance(d["bucket_preference"], list):
            d["bucket_preference"] = tuple(d["bucket_preference"])

        # --- computed_from / examples / share_candidate_with: list→tuple normalisation ---
        if "computed_from" in d and isinstance(d["computed_from"], list):
            d["computed_from"] = tuple(d["computed_from"])
        if "examples" in d and isinstance(d["examples"], list):
            d["examples"] = tuple(d["examples"])
        if "share_candidate_with" in d and isinstance(d["share_candidate_with"], list):
            d["share_candidate_with"] = tuple(d["share_candidate_with"])

        transformed[field_name] = d

    return _FIELD_DEF_ADAPTER.validate_python(transformed)
