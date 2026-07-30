"""Typed manifest for trained ranker models.

ModelManifest is the single typed record written to models/manifest.json
at save time and validated at load time. Replaces the plain dict that was
built inline in train.py.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Set as AbstractSet
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .exceptions import ContractMismatchError, ModelLoadError
from .features import FEATURE_SCHEMA_HASH
from .logging import get_logger
from .schema.field_def import FieldDef

logger = get_logger(__name__)

# Axes that do not change the model contract — tuning changes here must NOT
# trigger a retrain gate.  Structural axes (base_type, normalizer,
# anchor_family, keyword_proximal, bucket_preference, computed, etc.) are
# included implicitly by exclusion.
#
# Defined at module level (not on ModelManifest) so it remains a plain
# set[str] at runtime — Pydantic v2 treats class-level attributes starting
# with ``_`` as private attrs and wraps them in ModelPrivateAttr, which
# breaks model_dump(exclude=...) at the call site.
_COSMETIC_AXES: set[str] = {
    "description",
    "importance",
    "dataverse_column",
    "confidence_threshold",
    "priority_bonus",
    "format_hint",
    "format_penalty",
    "examples",
    "introduced_in",
}


class SkippedField(BaseModel):
    """Record of a field that failed training."""

    field_name: str
    reason: str


class FieldManifest(BaseModel):
    """Per-field training results."""

    pos_count: int
    neg_count: int
    total_samples: int
    metrics: dict[str, float]
    val_ndcg_at_1: float | None = None
    val_mrr: float | None = None
    loocv_ndcg_at_1: float | None = None
    loocv_std: float | None = None
    bootstrap_mode: bool = False


class ModelManifest(BaseModel):
    """Typed manifest for a trained ranker model set."""

    model_config = {"frozen": True}

    model_type: Literal["ranker"] = "ranker"
    model_version: Literal["v2"] = "v2"
    training_timestamp: str
    feature_columns: list[str]
    schema_hash: str
    feature_schema_version: str
    fields: dict[str, FieldManifest] = Field(default_factory=dict)
    skipped_fields: list[SkippedField] = Field(default_factory=list)
    quality_gate_passed: bool = False
    bootstrap_mode: bool = False

    def check_feature_compatibility(self, current_features: list[str]) -> bool:
        """Return True if the manifest's feature derivation matches runtime.

        The contract is the content-address hash of the ordered derivation,
        not the names or count. Count (69) is emergent; the hash subsumes it.
        A stale ranker persisted against an older derivation will have a
        ``feature_schema_version`` that differs from ``FEATURE_SCHEMA_HASH``
        and will be refused here.

        ``current_features`` is accepted for signature compatibility with
        callers that still pass it, but the hash — not the list — is the
        authoritative check. Legacy manifests (``feature_schema_version ==
        "legacy"``) are always rejected.
        """
        del current_features  # kept for signature compatibility; hash is the contract
        return self.feature_schema_version == FEATURE_SCHEMA_HASH

    def assert_feature_schema_compatible(self, *, model_id: str) -> None:
        """Raise ContractMismatchError when the persisted feature derivation drifts."""
        if self.feature_schema_version != FEATURE_SCHEMA_HASH:
            raise ContractMismatchError(
                field="feature_columns",
                expected=FEATURE_SCHEMA_HASH,
                actual=self.feature_schema_version,
                source=f"manifest.feature_schema_version ({model_id})",
            )

    def check_schema_compatibility(self, current_schema_hash: str) -> bool:
        """Return True if the manifest's schema_hash matches current_schema_hash."""
        return self.schema_hash == current_schema_hash

    def assert_schema_compatible(
        self, current_schema_hash: str, *, model_id: str
    ) -> None:
        """Raise ModelLoadError if schema_hash does not match current_schema_hash."""
        if not self.check_schema_compatibility(current_schema_hash):
            raise ModelLoadError(
                model_id=model_id,
                reason=f"schema hash mismatch: bundle={self.schema_hash}, current={current_schema_hash}",
            )

    def assert_ranker_coverage(
        self,
        required_fields: AbstractSet[str],
        known_fields: AbstractSet[str],
        *,
        model_id: str,
    ) -> None:
        """Warn on missing required rankers; raise on ghost rankers.

        required_fields: fields that should have a ranker (required + active).
            Missing rankers are a soft signal — the labeler's required markings are
            intent-only.  We log a warning and skip those fields rather than
            aborting the model load or commit.
        known_fields: all FieldSpec names (for ghost-ranker detection).
            Ghost rankers are schema drift — a hard error, still raises.
        """
        ranker_fields = set(self.fields.keys())
        missing = required_fields - ranker_fields
        ghosts = ranker_fields - known_fields
        if missing:
            logger.warning(
                "ranker_coverage_missing_required",
                missing_required=sorted(missing),
                model_id=model_id,
            )
        if ghosts:
            logger.error(
                "ranker_coverage_ghost_rankers",
                ghost_rankers=sorted(ghosts),
                model_id=model_id,
            )
            first_ghost = next(iter(sorted(ghosts)))
            raise ContractMismatchError(
                field=first_ghost,
                expected="absent_or_planned",
                actual="ranker_present",
                source=f"manifest.fields ({model_id})",
            )

    @classmethod
    def compute_schema_hash(cls, field_defs: dict[str, FieldDef]) -> str:
        """Compute a stable hash of field definitions for provenance tracking.

        Uses *negative projection*: cosmetic axes (tuning knobs) are excluded
        so that calibration edits do not invalidate persisted models.  Only
        structural axes — those that change what features are extracted or how
        candidates are scored — contribute to the hash.

        Hash is truncated to 16 hex chars to match existing manifest entries.
        See module-level ``_COSMETIC_AXES`` for the excluded set.
        """
        payload = {
            name: fd.model_dump(exclude=_COSMETIC_AXES)
            for name, fd in field_defs.items()
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def compute_feature_schema_version(cls, feature_columns: list[str]) -> str:
        """Canonical content-address of the ordered feature derivation.

        Recipe: ``"|".join(feature_columns)`` encoded utf-8, full sha256 hex.
        Matches ``invoices.features.FEATURE_SCHEMA_HASH`` exactly so persisted
        manifests can be hash-compared against the live derivation at load
        time. DO NOT change the recipe without bumping the persisted contract.
        """
        return hashlib.sha256("|".join(feature_columns).encode("utf-8")).hexdigest()

    @classmethod
    def load_from_json(
        cls,
        json_string: str,
        current_features: list[str] | None = None,
        current_schema_hash: str | None = None,
    ) -> ModelManifest | None:
        """Parse manifest JSON, returning None on failure or feature incompatibility.

        - Validation failure: logs warning, tries legacy fallback, returns None
        - Feature incompatibility: logs warning, returns None
        - Schema hash mismatch: raises ModelLoadError
        """
        try:
            manifest = cls.model_validate_json(json_string)
        except Exception as e:
            logger.warning(
                "manifest_parse_failed",
                error=str(e),
            )
            return _try_legacy_fallback(json_string)

        if current_features is not None:
            if not manifest.check_feature_compatibility(current_features):
                logger.warning(
                    "manifest_feature_mismatch",
                    manifest_features=len(manifest.feature_columns),
                    current_features=len(current_features),
                )
                return None

        if current_schema_hash is not None:
            manifest.assert_schema_compatible(current_schema_hash, model_id="manifest")

        return manifest


def _try_legacy_fallback(json_string: str) -> ModelManifest | None:
    """Attempt to parse an old-format manifest dict and wrap it in ModelManifest.

    Legacy manifests lack feature_columns and model_version as Literal.
    Returns None if the legacy parse also fails.
    """
    try:
        raw: dict[str, Any] = json.loads(json_string)
    except Exception:
        return None

    # Must at minimum be a ranker v2 manifest
    if raw.get("model_type") != "ranker" or raw.get("model_version") != "v2":
        return None

    raw_fields = raw.get("fields", {})
    typed_fields: dict[str, FieldManifest] = {}
    for fname, fdata in raw_fields.items():
        if not isinstance(fdata, dict):
            continue
        try:
            typed_fields[fname] = FieldManifest(
                pos_count=fdata.get("pos_count", 0),
                neg_count=fdata.get("neg_count", 0),
                total_samples=fdata.get("total_samples", 0),
                metrics=fdata.get("metrics", {}),
                val_ndcg_at_1=fdata.get("val_ndcg_at_1"),
                val_mrr=fdata.get("val_mrr"),
                loocv_ndcg_at_1=fdata.get("loocv_ndcg_at_1"),
                loocv_std=fdata.get("loocv_std"),
                bootstrap_mode=fdata.get("bootstrap_mode", False),
            )
        except ValidationError as exc:
            logger.warning(
                "manifest_field_parse_failed",
                field=fname,
                error=str(exc),
                error_type=type(exc).__name__,
            )
            continue

    logger.warning(
        "legacy_manifest_loaded",
    )

    return ModelManifest(
        training_timestamp=raw.get("training_timestamp", "unknown"),
        # Legacy manifests have no feature_columns — use empty list so
        # feature compatibility check will fail-safe on next load attempt
        feature_columns=[],
        schema_hash="legacy",
        feature_schema_version="legacy",
        fields=typed_fields,
        quality_gate_passed=raw.get("quality_gate_passed", False),
        bootstrap_mode=raw.get("bootstrap_mode", False),
    )
