"""pgqueuer worker entry point.

Connects to Postgres using POSTGRES_* env vars (matches docker-compose service),
installs pgqueuer schema on first boot, then runs the job loop.

Usage (canonical — via pgqueuer supervisor):
    pgq run invoices.queue:create_pgqueuer --restart-on-failure --restart-delay 5

    Or equivalently:
    python -m pgqueuer run invoices.queue:create_pgqueuer --restart-on-failure --restart-delay 5

Entrypoints:
    sharepoint_wake  — trigger full SharePoint ingest cycle
    reextract        — re-extract a single doc by sha256 (payload: {"sha": "..."})
    reextract_stale  — re-extract all docs with stale predictions
    train            — kick off active-learning training run
    reload           — reload contract schema + model state
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.util
import gc
import json
import multiprocessing
import os
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import asynccontextmanager
from datetime import timedelta

import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.domain.errors import DuplicateJobError, RetryRequested
from pgqueuer.executors import DatabaseRetryEntrypointExecutor
from pgqueuer.models import Job, Schedule
from pgqueuer.queries import Queries

from .logging import configure_logging, get_logger
from .settings import Settings

# ── Subprocess pool for CPU-bound reextract work ──────────────────────────────
# Spawn context (never fork): a forked child inherits asyncpg's open FDs and
# throws ConnectionDoesNotExistError identical to the bug we are fixing.
# max_tasks_per_child caps per-child RSS accumulation; after N docs the child
# exits and a fresh one starts — same principle as the parent's recycle SIGTERM.

_REEXTRACT_MAX_TASKS_PER_CHILD = int(
    os.getenv("INVOICEX_REEXTRACT_MAX_TASKS_PER_CHILD", "20")
)
_doc_pool: ProcessPoolExecutor | None = None


def _get_doc_pool(max_workers: int) -> ProcessPoolExecutor:
    global _doc_pool
    if _doc_pool is None:
        _doc_pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=_REEXTRACT_MAX_TASKS_PER_CHILD,
        )
    return _doc_pool


def _reset_doc_pool() -> None:
    global _doc_pool
    if _doc_pool is not None:
        _doc_pool.shutdown(wait=False, cancel_futures=True)
    _doc_pool = None


def _process_doc_compute(
    pdf_bytes: bytes,
    sha: str,
    doc_meta: dict,
    drive_id: str | None,
) -> dict:
    """Run all CPU-bound reextract work in a subprocess (spawn context).

    Imports are deferred to inside the function so that the spawn child does not
    execute any module-level side-effects from queue.py (no pool creation, no DB,
    no network) before this function body runs.

    Returns a plain JSON-serialisable dict — never a Doc, DocumentResult, or
    polars frame.  The parent event loop uses the returned dict for all DB writes.
    """
    # Heavy imports deferred: these are not available at queue.py module import
    # time in a spawn child until this function is invoked.
    import hashlib
    import io as _io

    from invoices.feature_prep import prepare_features_dataframe
    from invoices.model_loader import load_latest_models
    from invoices.pipeline import run_document_pipeline
    from invoices.tokenize import build_doc as _build_doc

    doc = _build_doc(_io.BytesIO(pdf_bytes))
    sha256_content = doc.sha256
    if not sha256_content:
        sha256_content = hashlib.sha256(pdf_bytes).hexdigest()

    pages = len(doc.pages)
    tokens = sum(len(p.tokens) for p in doc.pages)

    doc_json = doc.model_dump(mode="json")

    raw_docs_payload: dict = {
        "sha256_content": sha256_content,
        "source_id": sha,
        "source": "sharepoint",
        "doc": doc_json,
    }
    if drive_id:
        raw_docs_payload["drive_id"] = drive_id
    if doc_meta.get("name"):
        raw_docs_payload["filename"] = doc_meta["name"]
    if doc_meta.get("created_at"):
        raw_docs_payload["created_at"] = doc_meta["created_at"]

    # load_latest_models calls asyncio.run() internally (valid: no running loop
    # in this subprocess).
    loaded_models = load_latest_models()
    current_run_id = loaded_models.get("run_id") if loaded_models else None

    result = run_document_pipeline(
        sha256_content,
        pdf_bytes,
        source_id=sha,
        prebuilt_doc=doc,
        loaded_models=loaded_models,
        calibration_mapping=(
            loaded_models.get("calibration") if loaded_models else None
        ),
    )

    # Build feature matrix + candidates_payload (pure CPU).
    _features_df = (
        prepare_features_dataframe(result.candidates_df)
        if not result.candidates_df.is_empty()
        else result.candidates_df
    )
    _features_by_idx: dict[int, dict] = {
        i: _features_df.row(i, named=True) for i in range(len(_features_df))
    }
    _skip_cols = {"token_ids", "token_indices"}
    candidates_payload: list[dict] = []
    for i, feat in _features_by_idx.items():
        if result.candidates_df.is_empty():
            break
        raw_row = result.candidates_df.row(i, named=True)
        cand: dict = {
            k: v
            for k, v in raw_row.items()
            if k not in _skip_cols and not isinstance(v, list)
        }
        cand["features"] = {
            k: float(v) if v is not None else 0.0 for k, v in feat.items()
        }
        candidates_payload.append(cand)

    result_doc_json = result.doc.model_dump(mode="json")

    docs_payload: dict = {
        "sha256_content": sha256_content,
        "source_id": sha,
        "source": "sharepoint",
    }
    if drive_id:
        docs_payload["drive_id"] = drive_id
    if doc_meta.get("name"):
        docs_payload["filename"] = doc_meta["name"]
    if doc_meta.get("created_at"):
        docs_payload["created_at"] = doc_meta["created_at"]
    docs_payload["doc"] = result_doc_json
    docs_payload["candidates"] = candidates_payload

    # Build eval rows (one per field) — carry full assignment + candidate data.
    eval_rows: list[dict] = []
    for entry in result.evaluation_entries:
        field = entry.get("field", "")
        evaluator_version = entry.get("evaluator_version", "cc2-v1")

        assignment = result.assignments.get(field)
        selected_idx = assignment.candidate_index if assignment is not None else None

        field_candidates: list[dict] = []
        if assignment is not None:
            if (
                assignment.candidate_index is not None
                and 0 <= assignment.candidate_index < len(candidates_payload)
            ):
                _winner = dict(candidates_payload[assignment.candidate_index])
                _winner["field"] = field
                field_candidates.append(_winner)
            for _fb in assignment.fallback_candidates:
                if 0 <= _fb.candidate_index < len(candidates_payload):
                    _cand = dict(candidates_payload[_fb.candidate_index])
                    _cand["field"] = field
                    field_candidates.append(_cand)

        eval_rows.append(
            {
                "field": field,
                "evaluator_version": evaluator_version,
                "payload": {
                    "priority_score": entry.get("priority_score"),
                    "reason": entry.get("reason"),
                    "signal_disagreement": entry.get("signal_disagreement"),
                    "used_ml_model": entry.get("used_ml_model"),
                    "selected_candidate_idx": selected_idx,
                    "mlflow_run_id": current_run_id,
                    "candidates": field_candidates,
                },
            }
        )

    return {
        "sha256_content": sha256_content,
        "pages": pages,
        "tokens": tokens,
        "raw_docs_payload": raw_docs_payload,
        "docs_payload": docs_payload,
        "eval_rows": eval_rows,
        "current_run_id": current_run_id,
        "needs_review": result.needs_review,
        "n_evaluation_entries": len(result.evaluation_entries),
    }


def _make_malloc_trim() -> Callable[[], None]:
    """Return a callable that returns freed heap memory to the OS (glibc only).

    No-op on platforms without glibc malloc_trim (macOS dev, musl) so the
    worker module imports cleanly everywhere.
    """
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
    except (OSError, AttributeError):
        return lambda: None

    def _trim() -> None:
        try:
            libc.malloc_trim(0)
        except Exception:  # pragma: no cover - defensive
            pass

    return _trim


_malloc_trim = _make_malloc_trim()


logger = get_logger(__name__)


def _build_dsn(s: Settings) -> str:
    return (
        f"postgresql://{s.postgres_user}:{s.postgres_password}"
        f"@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"
    )


async def _enqueue_reextract_for_docs(pool: asyncpg.Pool, doc_ids: list[str]) -> int:
    """Enqueue a reextract job for each doc in *doc_ids*.

    Resolves the SharePoint source_id from docs.payload->>'source_id' and
    enqueues via PgQueuer with dedupe_key=sha so re-enqueueing an already-
    queued doc is a no-op (DuplicateJobError → silently skipped).

    Returns the number of payloads enqueued (0 if doc_ids is empty or none
    have a source_id).
    """
    if not doc_ids:
        return 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT d.sha256, d.payload->>'source_id' AS source_id "
            "FROM docs d "
            "WHERE d.sha256 = ANY($1::text[])",
            doc_ids,
        )

    payloads = [{"sha": row["source_id"]} for row in rows if row["source_id"]]
    if not payloads:
        return 0

    try:
        await Queries.from_asyncpg_pool(pool).enqueue(
            entrypoint=["reextract"] * len(payloads),
            payload=[json.dumps(p).encode() for p in payloads],
            priority=[0] * len(payloads),
            dedupe_key=[p["sha"] for p in payloads],
        )
    except DuplicateJobError:
        for p in payloads:
            try:
                await Queries.from_asyncpg_pool(pool).enqueue(
                    entrypoint="reextract",
                    payload=json.dumps(p).encode(),
                    dedupe_key=p["sha"],
                )
            except DuplicateJobError:  # noqa: PERF203
                pass  # already queued — that is the success state

    return len(payloads)


@asynccontextmanager
async def create_pgqueuer(*_args: object) -> AsyncGenerator[PgQueuer, None]:
    """Factory consumed by ``pgq run invoices.queue:create_pgqueuer``.

    The pgqueuer supervisor (pgqueuer.adapters.cli.supervisor.runit) enters this
    context manager, wires shutdown/signal handlers onto the yielded PgQueuer
    instance, then calls ``pgq.run(...)`` itself.  We are responsible only for
    setup (pool, schema seed, startup re-queue, entrypoint registration) before
    the yield and teardown (connection close) after it.

    Note on heartbeat_timeout: the old hand-rolled ``pgq.run(heartbeat_timeout=
    timedelta(seconds=90))`` call is replaced by the supervisor's own run call,
    which does not expose a heartbeat_timeout parameter (pgqueuer supervisor
    source confirmed).  The effective value reverts to pgqueuer's built-in
    default (30 s).  The previous 90 s comment was correct that "a short value
    is safe" — the event loop heartbeats during asyncio.to_thread so 30 s is
    adequate.
    """
    configure_logging("INFO")
    s = Settings()
    dsn = _build_dsn(s)

    pool = await asyncpg.create_pool(dsn, max_size=20)
    if pool is None:
        raise RuntimeError("asyncpg.create_pool returned None")

    try:
        # Seed schema cache at worker boot so run_document_pipeline never raises
        # "Contract schema not loaded" on cold start.  Empty dict is valid — pipeline
        # runs with zero field specs and produces empty doc_evaluations.
        from . import schema as _schema_mod

        async with pool.acquire() as _sc:
            _schema_row = await _sc.fetchrow(
                "SELECT payload FROM contract_schema"
                " WHERE payload ? 'field_definitions'"
                " ORDER BY version DESC LIMIT 1"
            )
        _schema_mod.set(json.loads(_schema_row["payload"]) if _schema_row else {})
        logger.info("schema_cache_seeded", has_schema=_schema_row is not None)

        # Reconcile env-configured confidence thresholds into contract_schema.payload
        # so the labeling UI (useConfidenceThresholds → latest_contract_schema) reads
        # the deployment's Settings values instead of its hardcoded 0.85/0.5 fallbacks.
        #
        # Guarded on a real schema row ONLY: the boot read above filters on
        # ``payload ? 'field_definitions'`` and latest_contract_schema() is unfiltered,
        # so appending a thresholds-only row would both strip field_definitions from
        # what the UI reads AND be skipped by this read → infinite re-append. We MERGE
        # into the existing payload and append only on drift (idempotent across restarts).
        #
        # Self-wrapped try/except because the enclosing block is try/finally (pool
        # teardown): a raised append (RPC drift / perms) would brick worker boot. A
        # skipped threshold sync is safe; a crash-looped boot is not.
        if _schema_row is not None:
            try:
                _payload = json.loads(_schema_row["payload"])
                _aa = s.confidence_auto_approve
                _med = s.confidence_heuristic_base
                _cur_aa = _payload.get("confidence_auto_approve")
                _cur_med = (_payload.get("confidence_thresholds") or {}).get("medium")
                if _cur_aa != _aa or _cur_med != _med:
                    _merged = {
                        **_payload,
                        "confidence_auto_approve": _aa,
                        "confidence_thresholds": {
                            **(_payload.get("confidence_thresholds") or {}),
                            "medium": _med,
                        },
                    }
                    async with pool.acquire() as _wc:
                        await _wc.execute(
                            "SELECT append_contract_schema($1::jsonb)",
                            json.dumps(_merged),
                        )
                    _schema_mod.set(_merged)
                    logger.info(
                        "contract_schema_thresholds_synced",
                        confidence_auto_approve=_aa,
                        medium=_med,
                    )
            except Exception as _exc:
                logger.warning(
                    "contract_schema_thresholds_sync_failed", error=str(_exc)
                )

        # On cold-start or schema-gap recovery: if schema is loaded but
        # doc_evaluations is empty while real docs exist, re-queue every doc.
        # Covers the case where the worker processed docs before the schema was
        # set — pipeline ran with zero field specs and wrote nothing to
        # doc_evaluations, leaving the results page empty.
        if _schema_row is not None:
            async with pool.acquire() as _sc:
                _eval_count = await _sc.fetchval("SELECT COUNT(*) FROM doc_evaluations")
                _doc_rows = await _sc.fetch(
                    "SELECT sha256, payload->>'source_id' AS source_id FROM docs "
                    "WHERE sha256 != '__sharepoint_delta_state__' "
                    "AND payload->>'source_id' IS NOT NULL"
                )
            if _eval_count == 0 and _doc_rows:
                logger.warning(
                    "startup_stale_docs_requeuing",
                    doc_count=len(_doc_rows),
                )
                _reextract_payloads = [{"sha": row["source_id"]} for row in _doc_rows]
                try:
                    await Queries.from_asyncpg_pool(pool).enqueue(
                        entrypoint=["reextract"] * len(_reextract_payloads),
                        payload=[json.dumps(p).encode() for p in _reextract_payloads],
                        priority=[0] * len(_reextract_payloads),
                        dedupe_key=[p["sha"] for p in _reextract_payloads],
                    )
                except DuplicateJobError:
                    for _p in _reextract_payloads:
                        with contextlib.suppress(DuplicateJobError):
                            await Queries.from_asyncpg_pool(pool).enqueue(
                                entrypoint="reextract",
                                payload=json.dumps(_p).encode(),
                                dedupe_key=_p["sha"],
                            )
                logger.info(
                    "startup_reextract_enqueued", count=len(_reextract_payloads)
                )

        # Dedicated connection for the pgqueuer run loop (LISTEN/NOTIFY needs a stable
        # persistent connection); the pool is for entrypoint-handler DB writes.
        # from_asyncpg_pool shared the listener conn across qm/sm → clean-exit restart loop.
        qm_conn = await asyncpg.connect(dsn)
        pgq = PgQueuer.from_asyncpg_connection(qm_conn)

        # Auto-fire sharepoint_wake every 15 minutes.  Queries.enqueue with
        # dedupe_key makes concurrent in-flight reextracts safe — the pgqueuer
        # unique partial index on dedupe_key drops duplicates at the DB level.
        @pgq.schedule("sharepoint_wake", "*/15 * * * *")
        async def schedule_sharepoint_wake(schedule: Schedule) -> None:
            await Queries.from_asyncpg_pool(pool).enqueue(
                "sharepoint_wake", payload=None
            )
            logger.info("sharepoint_wake_enqueued", expr=schedule.expression)

        # Auto-fire retrain nightly so the active-learning loop self-trains
        # unattended.  Without this, retrain only fires on a human button-click
        # (ModelPage/QueuePage) and the loop never learns on its own.  Mirrors
        # schedule_sharepoint_wake; the "retrain" entrypoint (handle_retrain) is
        # registered below.  Retrain is light (~1s ranker fit, concurrency_limit=1)
        # and self-aborts cheaply when the label->doc_evaluations join is empty,
        # so an unconditional nightly fire is safe.  Chosen over a "only if new
        # labels" gate because model_runs has no timestamp column to compare
        # against and init.sql is frozen on prod (can't add one without a volume
        # wipe) — the only cost of unconditional nightly is cosmetic MLflow-run
        # churn, bounded by the daily cadence.
        @pgq.schedule("retrain", "0 3 * * *")
        async def schedule_retrain(schedule: Schedule) -> None:
            await Queries.from_asyncpg_pool(pool).enqueue("retrain", payload=None)
            logger.info("retrain_scheduled", expr=schedule.expression)

        _SHAREPOINT_DELTA_SENTINEL = "__sharepoint_delta_state__"

        @pgq.entrypoint("sharepoint_wake", concurrency_limit=1)
        async def handle_sharepoint_wake(job: Job) -> None:
            from .azure.config import SharePointConfig
            from .azure.sharepoint._connector import SharePointConnector

            config = SharePointConfig.from_environment()
            if not config.is_configured():
                raise RuntimeError(
                    "sharepoint_wake: SharePointConfig not configured "
                    "(missing site_id/hostname+site_path or tenant_id in env)"
                )
            logger.info("sharepoint_wake_start", job=job.id)

            async with pool.acquire() as conn:
                _sentinel_row = await conn.fetchrow(
                    "SELECT payload->>'delta_link' AS delta_link,"
                    " payload->>'folder_scope' AS folder_scope"
                    " FROM docs WHERE sha256 = $1",
                    _SHAREPOINT_DELTA_SENTINEL,
                )
                delta_link: str | None = (
                    _sentinel_row["delta_link"] if _sentinel_row else None
                )
                _stored_scope: str | None = (
                    _sentinel_row["folder_scope"] if _sentinel_row else None
                )

                # Self-healing cursor: if the stored scope differs from the
                # current config (including first-run where stored is NULL),
                # discard the stale delta token and re-bootstrap folder-scoped.
                if _stored_scope != config.folder_path:
                    logger.info(
                        "sharepoint_delta_scope_changed",
                        old_scope=_stored_scope,
                        new_scope=config.folder_path,
                    )
                    delta_link = None

                async with SharePointConnector(config) as connector:
                    (
                        added_or_modified,
                        removed,
                        new_delta_link,
                    ) = await connector.list_documents_delta(delta_link)

                logger.info(
                    "sharepoint_wake_enumerated",
                    added_or_modified=len(added_or_modified),
                    removed=len(removed),
                    bootstrap=delta_link is None,
                    job=job.id,
                )

                payloads = [
                    {
                        "sha": d.id,
                        "name": d.name,
                        "created_at": d.created_at.isoformat(),
                    }
                    for d in added_or_modified
                ]
                if payloads:
                    try:
                        await Queries.from_asyncpg_pool(pool).enqueue(
                            entrypoint=["reextract"] * len(payloads),
                            payload=[json.dumps(p).encode() for p in payloads],
                            priority=[0] * len(payloads),
                            dedupe_key=[p["sha"] for p in payloads],
                        )
                    except DuplicateJobError:
                        # Bulk INSERT is all-or-nothing — one in-flight dup kills the batch.
                        # Fall back to per-item enqueue; pgqueuer's native partial unique index
                        # naturally absorbs the in-flight collisions, fresh shas enqueue cleanly.
                        for p in payloads:
                            try:
                                await Queries.from_asyncpg_pool(pool).enqueue(
                                    entrypoint="reextract",
                                    payload=json.dumps(p).encode(),
                                    dedupe_key=p["sha"],
                                )
                            except DuplicateJobError:  # noqa: PERF203
                                pass  # already queued/picked — that's the success state
                logger.info(
                    "sharepoint_wake_enqueued",
                    count=len(payloads),
                    job=job.id,
                )

                await conn.execute(
                    "SELECT ingest_doc($1, $2, $3::jsonb)",
                    _SHAREPOINT_DELTA_SENTINEL,
                    "",
                    json.dumps(
                        {
                            "delta_link": new_delta_link,
                            "folder_scope": config.folder_path,
                        }
                    ),
                )

            logger.info(
                "sharepoint_wake_done",
                enqueued=len(added_or_modified),
                job=job.id,
            )

        @pgq.entrypoint(
            "reextract",
            # concurrency_limit is Settings-driven (reextract_concurrency, default 1).
            # Caps the normal dispatch path (next_queued CTE).  Concurrency is also
            # bounded by the ProcessPoolExecutor max_workers — run_in_executor queues
            # when the pool is full.
            concurrency_limit=s.reextract_concurrency,
            executor_factory=lambda params: DatabaseRetryEntrypointExecutor(
                parameters=params,
                max_attempts=5,
                initial_delay=timedelta(seconds=2),
                max_delay=timedelta(minutes=5),
            ),
        )
        async def handle_reextract(job: Job) -> None:
            payload = json.loads(job.payload or b"{}")
            sha = payload.get("sha", "")
            if not sha:
                logger.warning("reextract_missing_sha", job=job.id)
                return

            logger.info("reextract_start", job=job.id, sha=sha)

            from .azure.config import SharePointConfig
            from .azure.sharepoint._connector import SharePointConnector

            config = SharePointConfig.from_environment()

            drive_id: str | None = None
            async with SharePointConnector(config) as connector:
                pdf_bytes = await connector.download(sha)
                # drive_id is set on connector.config by _ensure_drive_id during download.
                # Capture it here — connector is closed after this block.
                drive_id = connector.config.drive_id

                name = payload.get("name")
                created_at = payload.get("created_at")
                if name is None or created_at is None:
                    meta = await connector.get_metadata(sha)
                    if name is None:
                        name = meta.get("name")
                    if created_at is None:
                        created_at = meta.get("createdDateTime")
            doc_meta = {"name": name, "created_at": created_at}

            # ── Subprocess compute ────────────────────────────────────────────
            # All CPU-bound work (build_doc, model_dump, load_latest_models,
            # run_document_pipeline, feature/candidate assembly, eval-row building)
            # runs in a separate process via a spawn-context ProcessPoolExecutor.
            # The spawn child has its own GIL and its own memory space, so:
            #   (a) the asyncio event loop (main thread) is never blocked, and
            #   (b) a child OOM/crash cannot kill the worker process.
            # asyncio.wait_for still applies the per-job pipeline budget so a hung
            # child is cancelled from the parent's perspective and the job is
            # retried via DatabaseRetryEntrypointExecutor (up to max_attempts=5).
            _pipeline_timeout = s.reextract_pipeline_timeout_seconds

            loop = asyncio.get_running_loop()
            _t_pipeline_start = loop.time()
            try:
                computed = await asyncio.wait_for(
                    loop.run_in_executor(
                        _get_doc_pool(s.reextract_concurrency),
                        _process_doc_compute,
                        pdf_bytes,
                        sha,
                        doc_meta,
                        drive_id,
                    ),
                    timeout=_pipeline_timeout,
                )
            except BrokenProcessPool:
                _elapsed = loop.time() - _t_pipeline_start
                logger.error(
                    "reextract_subprocess_crashed",
                    job=job.id,
                    sha=sha,
                    elapsed_s=round(_elapsed, 1),
                )
                _reset_doc_pool()
                raise RetryRequested(
                    delay=timedelta(seconds=30),
                    reason="doc subprocess crashed (likely OOM)",
                )
            except TimeoutError:
                _elapsed = loop.time() - _t_pipeline_start
                logger.error(
                    "reextract_timeout",
                    job=job.id,
                    sha=sha,
                    elapsed_s=round(_elapsed, 1),
                    timeout_s=_pipeline_timeout,
                )
                raise  # propagates to DatabaseRetryEntrypointExecutor → retry with backoff

            sha256_content: str = computed["sha256_content"]

            # ── Always-on dimension instrumentation ──────────────────────────────
            logger.info(
                "reextract_doc_dims",
                job=job.id,
                sha=sha,
                sha256=sha256_content[:16],
                pdf_bytes=len(pdf_bytes),
                pages=computed["pages"],
                tokens=computed["tokens"],
            )

            # ── DB writes (parent event loop — no CPU work here) ─────────────────
            async with pool.acquire() as conn:
                await conn.execute(
                    "SELECT ingest_doc($1, $2, $3::jsonb)",
                    sha256_content,
                    "",
                    json.dumps(computed["raw_docs_payload"]),
                )

            async with pool.acquire() as conn:
                # Single doc-write surface: ingest_doc merge-upserts payload.
                await conn.execute(
                    "SELECT ingest_doc($1, $2, $3::jsonb)",
                    sha256_content,
                    "",
                    json.dumps(computed["docs_payload"]),
                )

                # Write one doc_evaluations row per field.
                # Payload carries evaluation signal + full candidate space so
                # handle_train can read features + scores directly from durable
                # state without re-running the pipeline.
                for eval_row in computed["eval_rows"]:
                    await conn.execute(
                        "INSERT INTO doc_evaluations (doc_id, field, evaluator_version, payload) "
                        "VALUES ($1, $2, $3, $4::jsonb)",
                        sha256_content,
                        eval_row["field"],
                        eval_row["evaluator_version"],
                        json.dumps(eval_row["payload"]),
                    )

            logger.info(
                "reextract_done",
                job=job.id,
                sha=sha,
                fields=computed["n_evaluation_entries"],
                needs_review=computed["needs_review"],
            )

            # Release large per-job allocations so memory doesn't accumulate across
            # reextract jobs in the same worker process.  All writes are complete by
            # this point; nothing below references these locals.
            del pdf_bytes, computed
            gc.collect()
            _malloc_trim()

        @pgq.entrypoint(
            "retrain",
            # Serialize retrain: each run loads the labeled-doc prediction corpus;
            # two concurrent runs double peak RSS and OOM the worker.
            concurrency_limit=1,
        )
        async def handle_retrain(job: Job) -> None:
            import mlflow
            import mlflow.xgboost

            from .ranker import InvoiceFieldRanker
            from .settings import Settings

            logger.info("train_start", job=job.id)
            s = Settings()

            # 1. Load all labeling events from the DB.
            # labels.action ∈ {approve, correct, not_in_document, reject}
            async with pool.acquire() as conn:
                label_rows = await conn.fetch(
                    "SELECT doc_id, field, action, payload, created_at "
                    "FROM labels "
                    "ORDER BY created_at"
                )

            if not label_rows:
                logger.warning("train_no_labels_abort", job=job.id)
                return

            labeled_doc_ids: list[str] = list({row["doc_id"] for row in label_rows})
            logger.info(
                "train_labeled_docs", labeled_docs=len(labeled_doc_ids), job=job.id
            )

            # 2. Load most-recent prediction event per (doc_id, field) from
            #    doc_evaluations.  Payload carries candidate space + features +
            #    selected_candidate_idx written by handle_reextract.
            #    We keep only the newest prediction row per (doc_id, field) so
            #    stale predictions don't pollute the feature matrix.
            import polars as pl

            async with pool.acquire() as conn:
                # Only labeled docs contribute training rows — pred_index is read
                # solely via pred_index.get((doc_id, field)) for label_rows below.
                # Filtering here bounds the in-memory payload load from the full
                # doc_evaluations corpus (~thousands of rows) to the labeled subset,
                # which is the difference between OOM and a few MB.
                pred_rows = await conn.fetch(
                    "SELECT DISTINCT ON (doc_id, field) "
                    "    doc_id, field, payload "
                    "FROM doc_evaluations "
                    "WHERE doc_id = ANY($1::text[]) "
                    "ORDER BY doc_id, field, created_at DESC",
                    labeled_doc_ids,
                )

            if not pred_rows:
                logger.warning("train_no_predictions_abort", job=job.id)
                _n_reextract = await _enqueue_reextract_for_docs(pool, labeled_doc_ids)
                logger.warning(
                    "train_abort_reextract_enqueued",
                    reason="no_predictions",
                    labeled_docs=len(labeled_doc_ids),
                    reextract_enqueued=_n_reextract,
                    job=job.id,
                )
                return

            # Index predictions by (doc_id, field) for O(1) join.
            import json as _json

            pred_index: dict[tuple[str, str], dict] = {}
            for row in pred_rows:
                key = (row["doc_id"], row["field"])
                raw_payload = row["payload"]
                if isinstance(raw_payload, str):
                    pred_payload = _json.loads(raw_payload)
                else:
                    pred_payload = dict(raw_payload) if raw_payload else {}
                pred_index[key] = pred_payload

            # 3. Build supervision rows for all four action types.
            #    Logic lives in evaluate.build_training_rows (DRY — evaluator
            #    reuses the same function).  No pipeline re-run.
            from .evaluate import build_training_rows

            training_rows = build_training_rows(label_rows, pred_index)

            if not training_rows:
                logger.warning("train_no_rows_abort", job=job.id)
                _n_reextract = await _enqueue_reextract_for_docs(pool, labeled_doc_ids)
                logger.warning(
                    "train_abort_reextract_enqueued",
                    reason="no_rows",
                    labeled_docs=len(labeled_doc_ids),
                    reextract_enqueued=_n_reextract,
                    job=job.id,
                )
                return

            train_df = pl.from_dicts(training_rows)

            # 4. Fit the ranker on ALL labeled data. Generalization/quality metrics
            # are a dev/offline concern (scripts/tune.py LOOCV — NOT imported by the
            # production pipeline); the shipped model never sacrifices training data
            # to an in-line measurement.
            ranker = InvoiceFieldRanker()
            metrics = ranker.train(train_df)
            logger.info("train_ranker_trained", metrics=metrics, job=job.id)

            # 5. Log model + metadata to MLflow.
            mlflow.set_tracking_uri(s.mlflow_tracking_uri)
            mlflow.set_experiment(s.mlflow_experiment_name)

            # Build per-field FieldManifest from training rows for the manifest.
            # Bootstrap threshold: < 30 positive samples total → bootstrap mode.
            _BOOTSTRAP_THRESHOLD = 30
            from . import schema as _schema_mod
            from .features import FEATURE_SCHEMA_HASH
            from .manifest import FieldManifest, ModelManifest

            _field_defs_raw = _schema_mod.load().get("field_definitions", {})
            from .schema.field_def import FieldDef as _FieldDef

            _field_defs: dict[str, _FieldDef] = {}
            for _fn, _fd in _field_defs_raw.items():
                if isinstance(_fd, dict):
                    try:
                        _field_defs[_fn] = _FieldDef.model_validate(_fd)
                    except Exception:
                        pass
                elif hasattr(_fd, "model_dump"):
                    _field_defs[_fn] = _fd

            _schema_hash = (
                ModelManifest.compute_schema_hash(_field_defs)
                if _field_defs
                else "unknown"
            )

            # Derive per-field pos/neg counts directly from training rows.
            _field_manifests: dict[str, FieldManifest] = {}
            for _tr in training_rows:
                _tf = _tr.get("target_field", "")
                if not _tf:
                    continue
                if _tf not in _field_manifests:
                    _field_manifests[_tf] = FieldManifest(
                        pos_count=0,
                        neg_count=0,
                        total_samples=0,
                        metrics={},
                        bootstrap_mode=False,
                    )
                _fm = _field_manifests[_tf]
                if _tr.get("label") == 1:
                    _field_manifests[_tf] = _fm.model_copy(
                        update={
                            "pos_count": _fm.pos_count + 1,
                            "total_samples": _fm.total_samples + 1,
                        }
                    )
                else:
                    _field_manifests[_tf] = _fm.model_copy(
                        update={
                            "neg_count": _fm.neg_count + 1,
                            "total_samples": _fm.total_samples + 1,
                        }
                    )

            _is_bootstrap = int(metrics["n_positive"]) < _BOOTSTRAP_THRESHOLD
            # Mark per-field bootstrap flag on fields with < 3 positives.
            _field_manifests = {
                _fn: _fm.model_copy(update={"bootstrap_mode": _fm.pos_count < 3})
                for _fn, _fm in _field_manifests.items()
            }

            from datetime import datetime, timezone

            _manifest = ModelManifest(
                training_timestamp=datetime.now(timezone.utc).isoformat(),
                feature_columns=ranker.feature_columns,
                schema_hash=_schema_hash,
                feature_schema_version=FEATURE_SCHEMA_HASH,
                fields=_field_manifests,
                bootstrap_mode=_is_bootstrap,
                quality_gate_passed=not _is_bootstrap,
            )

            try:
                with mlflow.start_run(run_name=f"train-job-{job.id}") as run:
                    mlflow.log_metrics(metrics)
                    mlflow.log_params(
                        {
                            "n_features": int(metrics["n_features"]),
                            "n_groups": int(metrics["n_groups"]),
                        }
                    )
                    mlflow.xgboost.log_model(
                        xgb_model=ranker.model,
                        name="ranker",
                        metadata={
                            "feature_columns": ranker.feature_columns,
                            "version": "1.0.0",
                        },
                        registered_model_name=s.mlflow_model_prefix,
                    )
                    # Persist self-describing manifest so load_models can reconstruct
                    # the consumer dict without re-reading training data.
                    mlflow.log_dict(
                        _manifest.model_dump(mode="json"),
                        "manifest.json",
                    )
                    run_id = run.info.run_id

                logger.info("train_model_logged", run_id=run_id, job=job.id)

                # Model state lives in MLflow; contract_schema stays the schema log.
                # Project this run into model_runs so model_status() (and the Model
                # page it feeds) has a Postgres-local view of MLflow state. Loaders
                # still resolve the latest run from MLflow, not from here.
                async with pool.acquire() as conn:
                    await conn.execute(
                        "INSERT INTO model_runs (run_id, payload) VALUES ($1, $2::jsonb)",
                        run_id,
                        json.dumps(
                            {
                                "n_groups": int(metrics["n_groups"]),
                                "n_samples": int(metrics["n_samples"]),
                                "n_positive": int(metrics["n_positive"]),
                                "n_features": int(metrics["n_features"]),
                                "mlflow_model_prefix": s.mlflow_model_prefix,
                            }
                        ),
                    )
            except Exception as exc:
                # Training SUCCEEDED but MLflow persistence failed (e.g. artifact
                # store misconfigured / blob creds). Never raise: a raised
                # entrypoint exception halts the pgqueuer dispatch loop and
                # starves reextract. Log and return cleanly so the worker keeps
                # draining other jobs.
                logger.error("train_model_persist_failed", error=str(exc), job=job.id)
                return

            # 7. Enqueue reextract for every labeled doc so they are re-scored
            #    under the new model.  Uses the SharePoint source_id stored in
            #    docs.payload->>'source_id' as the "sha" field that
            #    handle_reextract expects.  name/created_at are omitted —
            #    handle_reextract calls connector.get_metadata() to backfill.
            _reextract_count = await _enqueue_reextract_for_docs(pool, labeled_doc_ids)
            if _reextract_count:
                logger.info(
                    "train_reextract_enqueued",
                    count=_reextract_count,
                    run_id=run_id,
                    job=job.id,
                )

        logger.info("pgqueuer_loop_running")
        # Resilience: rely on pgqueuer's native heartbeat-based crash recovery
        # (orphaned 'picked' jobs are re-claimed after heartbeat_timeout). Do NOT
        # set shutdown_on_listener_failure — that flag is PgBouncer-specific (per
        # pgqueuer docs) and we connect directly to Postgres; enabling it turns a
        # transient LISTEN blip into a full worker restart (restart: unless-stopped),
        # killing in-flight reextract jobs. The supervisor calls pgq.run() itself
        # with its own dequeue_timeout/batch_size derived from CLI flags.
        yield pgq

    finally:
        try:
            await qm_conn.close()
        except Exception:
            pass
        await pool.close()


# The worker no longer self-bootstraps via asyncio.run().
# The canonical entrypoint is the pgqueuer supervisor:
#
#   python -m pgqueuer run invoices.queue:create_pgqueuer \
#       --restart-on-failure --restart-delay 5
#
# That command is what docker-compose passes to the worker container.
# Direct execution of this module (python -m invoices.queue) is unsupported.
