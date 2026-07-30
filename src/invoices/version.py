"""Derived version stamps — content-addressed, never hand-edited.

All version strings in this module are computed from the underlying shape
they describe.  Changing the feature schema, decoder config, or contract
schema automatically produces a new hash.  There is no hand-edited string
to forget to bump.

Public API
----------
FEATURE_VERSION : str
    12-hex-char SHA-256 prefix over the ordered FEATURE_COLUMNS tuple.
    Re-uses FEATURE_SCHEMA_HASH already computed in features.py.

decoder_version() : str
    12-hex-char SHA-256 prefix over (schema field_definitions, decoder
    config subset).  Computed lazily on first call (schema not available
    at import time) and cached thereafter.

validate_config() : tuple[Settings, str]
    Load the Settings singleton + active schema, compute a combined
    config_version, log it, and return both.  Call this at every
    process entrypoint before constructing the orchestrator.

stable_hash(payload) : str
    Canonical, process-portable SHA-256 hex prefix.  Uses json.dumps
    with sort_keys=True so dict iteration order never affects output.
    Never use Python's builtin hash() — it is salted per-process.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from .features import FEATURE_SCHEMA_HASH as _FEATURE_SCHEMA_HASH
from .logging import get_logger

if TYPE_CHECKING:
    from .settings import Settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Primitive
# ---------------------------------------------------------------------------

_HASH_PREFIX_LEN = 12


def stable_hash(payload: object) -> str:
    """Stable, process-portable hash.  SHA-256 hex prefix (12 chars).

    Canonical serialisation: json.dumps with sort_keys=True, default=str.
    This means dict key order never affects output and un-jsonable objects
    (e.g. enum members, Path) are reduced to their str() representation.

    Never use Python's builtin hash() — it is salted per-process and will
    produce different values across restarts, breaking any stored comparison.
    """
    canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:_HASH_PREFIX_LEN]


# ---------------------------------------------------------------------------
# Feature version — module-level constant (schema is fixed at import time)
# ---------------------------------------------------------------------------

FEATURE_VERSION: str = _FEATURE_SCHEMA_HASH[:_HASH_PREFIX_LEN]
"""Stable content-address of the ordered feature derivation (12 hex chars).

Derived from FEATURE_COLUMNS in features.py via SHA-256.  Any add/remove/
rename/reorder of a feature column produces a different value automatically.
"""

# ---------------------------------------------------------------------------
# Decoder config subset — the Settings fields that actually change decoder output
# ---------------------------------------------------------------------------

#: Settings field names whose values affect the decoder's output.
#: Changing any of these fields must produce a new decoder_version hash.
#: Fields that only affect infra (MLflow URIs, timeouts, backend routing,
#: CORS, API keys, etc.) are intentionally excluded.
DECODER_CONFIG_FIELDS: tuple[str, ...] = (
    "none_bias",
    "decoder_base_cost",
    "ml_score_weight",
    "bootstrap_ml_score_weight",
    "cross_field_penalty",
    "early_page_boost",
    "early_page_max_idx",
    "pruning_threshold",
    "pruning_max_candidates",
    "pruning_min_trigger",
    "confidence_auto_approve",
    "confidence_heuristic_base",
)

# ---------------------------------------------------------------------------
# Decoder version — lazy singleton (needs schema, unavailable at import time)
# ---------------------------------------------------------------------------

_decoder_version_cache: str | None = None


def decoder_version() -> str:
    """Compute (and cache) the decoder version hash.

    Hash covers:
      - schema field_definitions (field names, types, constraints)
      - the DECODER_CONFIG_FIELDS subset of the active Settings

    Called lazily so the schema cache has been seeded by startup before
    this is invoked.  Subsequent calls return the cached value; the cache
    is intentionally not invalidated mid-process because a running process
    should not silently change its decoder contract.

    Returns
    -------
    str
        12-hex-char SHA-256 prefix.
    """
    global _decoder_version_cache
    if _decoder_version_cache is not None:
        return _decoder_version_cache

    from . import schema as schema_mod
    from .settings import settings as _settings

    schema_obj = schema_mod.load()
    field_defs = schema_obj.get("field_definitions", {})

    config_subset = {k: getattr(_settings, k) for k in DECODER_CONFIG_FIELDS}

    payload = {
        "schema": field_defs,
        "config": config_subset,
    }
    _decoder_version_cache = stable_hash(payload)
    return _decoder_version_cache


def _reset_decoder_version_cache() -> None:
    """Reset the decoder version cache.  For testing only."""
    global _decoder_version_cache
    _decoder_version_cache = None


# ---------------------------------------------------------------------------
# validate_config — call at every process entrypoint
# ---------------------------------------------------------------------------


def validate_config() -> tuple[Settings, str]:
    """Load Settings + active schema, compute config_version, log it.

    Returns
    -------
    (settings, config_version)
        settings     — the module-level singleton from settings.py
        config_version — stable_hash over the full settings dict + schema,
                         suitable for tagging bundles and log lines.

    Call this before constructing the orchestrator so the config_version
    appears in startup logs and can be correlated with bundle hashes.
    """
    from . import schema as schema_mod
    from .settings import settings as _settings

    schema_obj = schema_mod.load()

    # Full settings snapshot (all fields that pydantic knows about).
    # model_dump() is deterministic within a process for a frozen model.
    settings_dict: dict[str, Any] = _settings.model_dump()

    config_version = stable_hash({"settings": settings_dict, "schema": schema_obj})

    logger.info(
        "config_version_computed",
        config_version=config_version,
        feature_version=FEATURE_VERSION,
        decoder_version=decoder_version(),
    )

    return _settings, config_version
