"""Cache-aside model loader: pull trained ranker from MLflow into memory.

Contract
--------
``load_models(run_id)`` returns a dict shaped:

    {
        "models": {field_name: {"ranker": InvoiceFieldRanker}},
        "manifest": ModelManifest | None,
        "model_type": "ranker",
    }

This is the exact shape consumed by ``compute_ml_cost_with_prob`` /
``compute_ranker_cost`` in ``decoder/cost/ml.py``.

Cache behaviour
---------------
Module-level ``_MODEL_CACHE`` keyed on ``run_id``.  One MLflow pull per
retrain cycle; zero overhead in steady state.  CPython dict assignment is
atomic — no lock needed for single-writer / many-reader pattern.

Architecture constraint: MLflow client into memory only.  No filesystem I/O
for model data (Architecture Rule 1).  MLflow's internal pyfunc loader uses
a process-scoped tmp dir — this is the library's implementation detail, not
our code touching disk.
"""

from __future__ import annotations

import os
from typing import Any

from .logging import get_logger

logger = get_logger(__name__)

# Module-scoped cache: run_id → loaded_models dict (ranker + manifest only).
# LiLT is NOT cached here — it uses _LILT_BUNDLE below so a transient load
# failure never permanently disables LiLT for the worker's lifetime.
_MODEL_CACHE: dict[str, dict[str, Any]] = {}

# Separate lilt cache (see _resolve_lilt_bundle):
#   None          = not yet resolved, or the last load FAILED (retry next call)
#   _LILT_ABSENT  = weights are not baked into this image (resolved once,
#                   never re-probed or re-logged)
#   dict          = loaded bundle
_LILT_ABSENT = object()  # sentinel: weights absent (a resolved state, not a failure)
_LILT_BUNDLE: dict[str, Any] | object | None = None  # None = not yet attempted

_VENDORED_LILT_DIR = os.environ.get("INVOICEX_LILT_DIR", "/app/models/lilt")


def load_models(run_id: str) -> dict[str, Any] | None:
    """Load trained ranker for *run_id* from MLflow, cached per run.

    Returns the ``loaded_models`` dict expected by
    ``compute_ml_cost_with_prob`` / ``compute_ranker_cost``::

        {
            "models": {field_name: {"ranker": <InvoiceFieldRanker>}},
            "manifest": <ModelManifest | None>,
            "model_type": "ranker",
        }

    Returns ``None`` when the run has no trained model or the manifest is
    missing/incompatible, leaving the pipeline in heuristic-only mode.

    Never writes to the filesystem.
    """
    if run_id in _MODEL_CACHE:
        return _MODEL_CACHE[run_id]

    logger.info("model_loader_cache_miss", run_id=run_id)

    result = _fetch_from_mlflow(run_id)
    if result is not None:
        _MODEL_CACHE[run_id] = result
        logger.info(
            "model_loader_cached",
            run_id=run_id,
            fields=len(result.get("models", {})),
        )
    return result


def load_latest_models() -> dict[str, Any] | None:
    """Load the most-recent trained ranker from MLflow — decoupled from
    contract_schema. ``None`` when no model exists (pipeline runs heuristic-only).
    """
    run_id = _latest_run_id()
    if not run_id:
        return None
    models = load_models(run_id)
    if models is not None:
        models["run_id"] = run_id
    return models


def _latest_run_id() -> str | None:
    """Resolve the latest MLflow run id for the training experiment, or None.

    Primary source: MLflow tracking server search API.
    Fallback: model_runs table (Postgres) — handles the case where MLflow is
    unavailable but a run was already recorded in the DB projection.
    """
    try:
        import mlflow
    except ImportError:
        return _latest_run_id_from_db()

    from .settings import Settings

    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    try:
        runs = mlflow.search_runs(
            experiment_names=[s.mlflow_experiment_name],
            order_by=["attributes.start_time DESC"],
            max_results=1,
            output_format="list",
        )
        if runs:
            return str(runs[0].info.run_id)
    except Exception as exc:
        logger.warning("model_loader_latest_run_lookup_failed", error=str(exc))

    # MLflow unavailable or returned no runs — fall back to model_runs table.
    return _latest_run_id_from_db()


def _latest_run_id_from_db() -> str | None:
    """Query model_runs table for the latest run_id via asyncpg (sync wrapper).

    Safe to call from a thread (asyncio.to_thread context) — no running event
    loop in the thread, so asyncio.run() is valid here.
    """
    import asyncio

    async def _fetch() -> str | None:
        try:
            import asyncpg
        except ImportError:
            return None

        from .settings import Settings

        s = Settings()
        dsn = (
            f"postgresql://{s.postgres_user}:{s.postgres_password}"
            f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
        )
        try:
            conn = await asyncpg.connect(dsn)
            try:
                row = await conn.fetchrow(
                    "SELECT run_id FROM model_runs ORDER BY id DESC LIMIT 1"
                )
                return str(row["run_id"]) if row else None
            finally:
                await conn.close()
        except Exception as exc:
            logger.warning("model_loader_db_run_id_lookup_failed", error=str(exc))
            return None

    try:
        return asyncio.run(_fetch())
    except RuntimeError:
        # Already inside a running event loop — skip DB fallback rather than
        # deadlock.  The caller handles None gracefully (heuristic-only mode).
        logger.warning("model_loader_db_fallback_skipped_running_loop")
        return None


def _fetch_from_mlflow(run_id: str) -> dict[str, Any] | None:
    """Pull ranker + manifest from MLflow and assemble the consumer dict."""
    try:
        import mlflow
        import mlflow.xgboost
    except ImportError as exc:
        logger.error("model_loader_mlflow_missing", error=str(exc))
        return None

    from .manifest import ModelManifest
    from .ranker import InvoiceFieldRanker
    from .settings import Settings

    s = Settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)

    # ── Load XGBRanker model into memory ──────────────────────────────────
    model_uri = f"runs:/{run_id}/ranker"
    try:
        xgb_model = mlflow.xgboost.load_model(model_uri)
    except Exception as exc:
        logger.error(
            "model_loader_xgb_load_failed",
            run_id=run_id,
            error=str(exc),
        )
        return None

    # ── Load manifest from MLflow artifacts ──────────────────────────────
    manifest: ModelManifest | None = None
    try:
        manifest_bytes = mlflow.artifacts.load_text(f"runs:/{run_id}/manifest.json")
        manifest = ModelManifest.load_from_json(manifest_bytes)
    except Exception as exc:
        logger.warning(
            "model_loader_manifest_missing",
            run_id=run_id,
            error=str(exc),
        )
        # Fall through with manifest=None — pipeline degrades to heuristic-only
        # rather than serving stale / incompatible manifest.
        return None

    if manifest is None:
        logger.warning("model_loader_manifest_incompatible", run_id=run_id)
        return None

    # ── Reconstruct InvoiceFieldRanker from in-memory xgb model ──────────
    # We bypass InvoiceFieldRanker.load() (which is filesystem-only) and
    # directly set the .model and .feature_columns attributes.
    ranker = InvoiceFieldRanker()
    ranker.model = xgb_model
    ranker.feature_columns = list(manifest.feature_columns)

    # ── Determine schema fields ────────────────────────────────────────────
    # The manifest records which fields were trained.  Key every schema field
    # that has a trained ranker under its name.  Fields absent from
    # manifest.fields get no ranker entry — compute_ranker_cost falls back to
    # weak_prior for those.
    trained_fields = set(manifest.fields.keys())

    # If no fields are in the manifest (e.g. bootstrap single-ranker run),
    # fall back to the current contract schema field names.  A bootstrap model
    # that trained must be loaded and used — returning None here would silently
    # discard a valid model and leave every field in heuristic-only mode.
    if not trained_fields:
        logger.warning(
            "model_loader_no_per_field_manifest",
            run_id=run_id,
            note="bootstrap_mode — populating models_dict from current schema",
        )
        try:
            from .schema import get_field_names

            trained_fields = set(get_field_names())
        except Exception as exc:
            logger.warning(
                "model_loader_schema_field_lookup_failed",
                run_id=run_id,
                error=str(exc),
            )
            trained_fields = set()

        if not trained_fields:
            # Schema also empty — no field names available to key the dict.
            # Log and return the ranker under a sentinel so it is at least
            # reachable, but downstream compute_ranker_cost will weak-prior.
            # This path is only hit on a completely unconfigured first boot.
            logger.warning(
                "model_loader_bootstrap_no_schema_fields",
                run_id=run_id,
            )
            return None

    models_dict: dict[str, dict[str, Any]] = {
        field: {"ranker": ranker} for field in trained_fields
    }

    result: dict[str, Any] = {
        "models": models_dict,
        "manifest": manifest,
        "model_type": "ranker",
    }

    # ── Opportunistically attach LiLT token classifier ─────────────────────
    # _resolve_lilt_bundle owns the _LILT_BUNDLE cache (separate from
    # _MODEL_CACHE): absent weights resolve once to _LILT_ABSENT; a load
    # failure leaves the cache None so the next call retries.
    lilt_bundle = _resolve_lilt_bundle()
    if isinstance(lilt_bundle, dict):
        result["lilt"] = lilt_bundle

    # ── Attach per-field calibration mapping (baked next to vendored model) ──
    # Per-field isotonic fits + a '_global' pooled fallback so sparse fields are
    # never uncalibrated.  Pipeline applies it to ranker/heuristic confidence.
    cal = _load_calibration_vendored()
    if cal is not None:
        result["calibration"] = cal

    return result


def _load_calibration_vendored() -> dict[str, Any] | None:
    """Load the per-field calibration mapping baked at ``_VENDORED_LILT_DIR``.

    Computed from labeled data via ``compute_calibration_mapping`` (per-field
    isotonic fits + a ``'_global'`` pooled fallback so sparse fields fall back
    to the all-field calibration rather than being left uncalibrated). Returns
    ``None`` when the file is absent (no calibration baked into this image).
    """
    import json
    from pathlib import Path as _Path

    p = _Path(_VENDORED_LILT_DIR) / "calibration.json"
    if not p.is_file():
        return None
    try:
        with p.open() as fh:
            mapping: dict[str, Any] = json.load(fh)
    except Exception as exc:
        logger.warning("calibration_load_failed", error=str(exc))
        return None
    n_field = len([k for k in mapping if not k.startswith("__") and k != "_global"])
    logger.info("calibration_loaded", n_fields=n_field, has_global="_global" in mapping)
    return mapping


def _load_lilt_from_dir(local_dir: str, *, source: str) -> dict[str, Any] | None:
    """Load a LiLT bundle (model + tokenizer + id2label) from a local dir.

    Used by the vendored-image path. Returns
    ``None`` (never raises) on missing heavy deps or load failure.
    """
    import json
    from pathlib import Path as _Path

    try:
        import torch
        from transformers import AutoTokenizer, LiltForTokenClassification
    except ImportError as exc:
        # torch/transformers absent — expected in dev; not a load failure.
        logger.warning("lilt_import_failed", source=source, error=str(exc))
        return None

    logger.debug("lilt_vendored_load_started", path=local_dir)
    try:
        model = LiltForTokenClassification.from_pretrained(  # nosec B615 — local_dir is a vendored/downloaded artifact path, not a Hub repo id; local_files_only=True blocks any remote fetch
            local_dir,
            local_files_only=True,
        )
        model = model.to("cpu").eval()  # type: ignore[union-attr]
        tokenizer = AutoTokenizer.from_pretrained(  # nosec B615 — local_dir is a vendored/downloaded artifact path, not a Hub repo id; local_files_only=True blocks any remote fetch
            local_dir,
            add_prefix_space=True,
            local_files_only=True,
        )
        with (_Path(local_dir) / "id2label.json").open() as fh:
            raw: dict[str, str] = json.load(fh)
        id2label: dict[int, str] = {int(k): v for k, v in raw.items()}
    except Exception as exc:
        # ERROR + full traceback so the exact failure surfaces in prod logs.
        logger.error("lilt_load_failed", source=source, error=str(exc), exc_info=True)
        return None

    logger.info("lilt_loaded", source=source, num_labels=len(id2label))
    _ = torch  # imported for side-effect (device availability); suppress F401
    return {"model": model, "tokenizer": tokenizer, "id2label": id2label}


def _fetch_lilt_vendored() -> dict[str, Any] | object | None:
    """Load the LiLT bundle baked into the image at ``_VENDORED_LILT_DIR``.

    Frozen, static artifact baked into the image at build time (opt-in via the
    Dockerfile ``WITH_LILT`` build arg). Run-id independent. Three outcomes:

    * ``_LILT_ABSENT`` -- no weights in this image (an expected, first-class
      state; logged once at INFO as ``lilt_vendored_absent``).
    * ``None`` -- weights present but the load failed (logged at ERROR /
      WARNING by ``_load_lilt_from_dir``); the caller retries next call.
    * ``dict`` -- the loaded bundle.
    """
    from pathlib import Path as _Path

    d = _Path(_VENDORED_LILT_DIR)
    has_weights = (d / "model.safetensors").is_file() or (
        d / "pytorch_model.bin"
    ).is_file()
    if not (d / "config.json").is_file() or not has_weights:
        logger.info("lilt_vendored_absent", path=str(d))
        return _LILT_ABSENT
    return _load_lilt_from_dir(str(d), source="vendored")


def _resolve_lilt_bundle() -> dict[str, Any] | object | None:
    """Resolve the process-wide LiLT bundle, owning the ``_LILT_BUNDLE`` cache.

    * absent weights -> cache ``_LILT_ABSENT`` (probed and logged once per process)
    * load failure   -> leave the cache ``None`` (retry on the next call, e.g.
      after a rolling deploy makes torch importable)
    * success        -> cache and return the bundle dict

    Callers attach LiLT only when the result is a ``dict``.
    """
    global _LILT_BUNDLE
    if _LILT_BUNDLE is None:
        bundle = _fetch_lilt_vendored()
        if bundle is not None:
            _LILT_BUNDLE = bundle
    return _LILT_BUNDLE
