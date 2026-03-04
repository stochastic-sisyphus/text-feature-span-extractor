"""FastAPI REST API for active learning review interface.

This module provides the REST API endpoints for the human review workflow,
allowing reviewers to access low-confidence predictions and provide feedback.

The API is designed to be embedded in Grafana dashboards via iframe.

Usage:
    # Run with uvicorn (using factory pattern)
    uvicorn invoices.api:get_app --factory --host 0.0.0.0 --port 8080

    # Or programmatically
    from invoices.api import create_app
    app = create_app()
"""

import asyncio
import hmac
import json
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from invoices import __version__
from invoices.api_models import (
    ApprovalResponse,
    ApprovalSubmission,
    DocumentDetail,
    FieldDetail,
    FieldListResponse,
    FieldMetadata,
    FieldUpdateRequest,
    HealthStatus,
    LabelResponse,
    LabelSubmission,
    QueueItem,
    QueueResponse,
    ReviewerStats,
)
from invoices.exceptions import DocumentNotFoundError, InvoicexError
from invoices.io_utils import read_parquet_safe
from invoices.logging import get_logger

logger = get_logger(__name__)

# Training concurrency lock to prevent multiple training runs
_training_lock = asyncio.Lock()

# Pipeline concurrency lock to prevent overlapping pipeline runs
_pipeline_lock = asyncio.Lock()


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _validate_sha256(value: str) -> str:
    """Validate that a string is a valid SHA256 hash.

    Args:
        value: String to validate

    Returns:
        The validated string

    Raises:
        HTTPException: If the string is not a valid SHA256 hash
    """
    if not re.match(r"^[a-f0-9]{64}$", value.lower()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid SHA256 hash format: {value}",
        )
    return value


def _validate_doc_id(value: str) -> str:
    """Validate that a doc_id is safe against path traversal attacks.

    Args:
        value: Document ID to validate (may include fs: prefix)

    Returns:
        The validated string

    Raises:
        HTTPException: If the doc_id format is invalid
    """
    # Strip optional fs: prefix
    doc_id = value.replace("fs:", "", 1) if value.startswith("fs:") else value

    # Validate: must be hex hash, 8-64 chars (partial or full SHA256)
    if not re.match(r"^[a-f0-9]{8,64}$", doc_id.lower()):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_id format: {value}",
        )
    return value


def create_app(
    data_dir: Path | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        data_dir: Base directory for data storage. Defaults to ./data
        cors_origins: Allowed CORS origins. Defaults to common Grafana origins.

    Returns:
        Configured FastAPI application
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan context manager for startup/shutdown events."""
        # Startup: validate configuration
        from invoices.config import Config

        try:
            Config.validate()
            logger.info("config_validated")
        except ValueError as e:
            logger.error("config_validation_failed", error=str(e))
            raise RuntimeError(f"Configuration validation failed: {e}") from e

        # Startup: load schema once and cache in app.state
        from invoices import utils

        try:
            app.state.schema = utils.load_contract_schema()
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error("schema_load_failed", error=str(e))
            raise RuntimeError(f"Failed to load contract schema: {e}") from e

        # Startup: populate metrics
        try:
            from invoices import decoder, emit
            from invoices.metrics import (
                calibration_error,
                field_accuracy,
                human_correction_rate,
                update_model_metrics,
                update_queue_metrics,
                validation_ndcg,
            )

            queue = emit.get_review_queue()
            if not queue.empty:
                priority_counts = queue["priority_level"].value_counts().to_dict()
                update_queue_metrics(len(queue), priority_counts)

            model_state = decoder.get_last_decode_model_state()
            if model_state:
                update_model_metrics(
                    loaded=model_state.get("models_loaded", False),
                    field_count=len(model_state.get("model_fields", [])),
                    model_type=model_state.get("model_type", "none"),
                )

            # Initialize per-field learning gauges to 0.0 so Grafana panels
            # always have a value, even before the first training run.
            schema_fields = app.state.schema.get("fields", [])
            for _field in schema_fields:
                human_correction_rate.labels(field=_field).set(0.0)
                field_accuracy.labels(field=_field).set(0.0)
                validation_ndcg.labels(field=_field).set(0.0)
            validation_ndcg.labels(field="all").set(0.0)
            calibration_error.set(0.0)
        except (ImportError, OSError, KeyError, ValueError):
            pass  # Don't fail startup if metrics init fails

        yield  # Application runs

        # Shutdown: nothing to clean up currently

    app = FastAPI(
        title="Invoice Extraction Active Learning API",
        description="REST API for human review of low-confidence invoice extractions",
        version=__version__,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Create a fresh config snapshot — tests may set env vars after the
    # module-level Config singleton was created, so we re-read here.
    from invoices.config import PipelineConfig

    _cfg = PipelineConfig()

    if data_dir is None:
        data_dir = _cfg.get_data_path()
    app.state.data_dir = data_dir

    # Configure CORS for Grafana iframe embedding
    if cors_origins is None:
        cors_origins = _cfg.get_cors_origins_list()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "Authorization"],
        expose_headers=["X-Total-Count", "X-Page", "X-Page-Size"],
    )

    # API key authentication — read from central Config
    api_key = _cfg.API_KEY or None  # treat empty string as None
    is_dev = _cfg.is_dev

    if not is_dev and not _cfg.API_KEY:
        logger.warning(
            "ENVIRONMENT not set, defaulting to production mode (auth required). "
            "Set ENVIRONMENT=development for local testing without auth."
        )

    if not is_dev and not api_key:
        raise RuntimeError(
            "INVOICEX_API_KEY is required in production mode. "
            "Set ENVIRONMENT=development for local testing without auth."
        )

    if api_key:

        @app.middleware("http")
        async def verify_api_key(request: Request, call_next: Any) -> Response:
            # Skip auth for health endpoints only — /metrics and /api/v1/config
            # require auth. If Prometheus needs unauthenticated scraping, restrict
            # /metrics to internal IPs via nginx (allow 10.0.0.0/8; deny all).
            if request.url.path in (
                "/api/v1/health",
                "/api/v1/health/models",
            ):
                return await call_next(request)  # type: ignore[no-any-return]
            # Skip auth for docs in dev mode
            if is_dev and request.url.path in (
                "/api/docs",
                "/api/redoc",
                "/api/openapi.json",
            ):
                return await call_next(request)  # type: ignore[no-any-return]
            # Check API key using constant-time comparison
            provided_key = request.headers.get("X-API-Key")
            if not provided_key or not hmac.compare_digest(provided_key, api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Unauthorized",
                        "message": "Invalid or missing API key",
                    },
                )
            return await call_next(request)  # type: ignore[no-any-return]

    # Add exception handlers
    @app.exception_handler(InvoicexError)
    async def invoicex_exception_handler(
        request: Request, exc: InvoicexError
    ) -> JSONResponse:
        """Handle InvoicexError exceptions."""
        logger.error(
            "api_error",
            error_type=exc.__class__.__name__,
            message=str(exc),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=500,
            content={"error": "InternalError", "message": "An internal error occurred"},
        )

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        """Handle DocumentNotFoundError exceptions."""
        logger.warning("document_not_found", message=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=404,
            content={"error": "DocumentNotFound", "message": "Document not found"},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler: prevent internal details from leaking in responses."""
        logger.error(
            "unhandled_exception",
            error_type=exc.__class__.__name__,
            message=str(exc),
            path=request.url.path,
        )
        # In dev mode, include the exception type for debugging.
        # In production, return only a generic message.
        if is_dev:
            detail = f"{exc.__class__.__name__}: {exc}"
        else:
            detail = "An internal error occurred"
        return JSONResponse(
            status_code=500,
            content={"error": "InternalError", "message": detail},
        )

    # Register routes
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API routes."""

    # =========================================================================
    # Health Check
    # =========================================================================

    @app.get("/api/v1/health", response_model=HealthStatus, tags=["Health"])
    async def health_check() -> HealthStatus:
        """Check API health status.

        Returns health status including component checks.
        """
        checks: dict[str, str] = {}

        # Check data directory
        data_dir: Path = app.state.data_dir
        if data_dir.exists():
            checks["data_dir"] = "healthy"
        else:
            checks["data_dir"] = "missing"

        # Check predictions directory
        predictions_dir = data_dir / "predictions"
        if predictions_dir.exists():
            checks["predictions"] = "healthy"
        else:
            checks["predictions"] = "missing"

        # Determine overall status
        if all(v == "healthy" for v in checks.values()):
            status = "healthy"
        elif any(v == "healthy" for v in checks.values()):
            status = "degraded"
        else:
            status = "unhealthy"

        return HealthStatus(
            status=status,
            version=__version__,
            timestamp=datetime.now(timezone.utc),
            checks=checks,
        )

    @app.get("/api/v1/health/models", tags=["Health"])
    async def health_models() -> dict[str, Any]:
        """Report model load state and manifest info."""
        from invoices import decoder, paths

        manifest_path = paths.get_models_dir() / "manifest.json"
        manifest_exists = manifest_path.exists()

        model_state = decoder.get_last_decode_model_state()

        result: dict[str, Any] = {
            "manifest_exists": manifest_exists,
            "model_state": model_state,
        }

        if manifest_exists:
            import json

            try:
                with open(manifest_path, encoding="utf-8") as f:
                    manifest = json.load(f)
                result["manifest"] = {
                    "fields": list(manifest.get("models", {}).keys()),
                    "created_at": manifest.get("created_at"),
                    "model_type": manifest.get("model_type"),
                }
            except (json.JSONDecodeError, OSError):
                result["manifest"] = {"error": "Could not read manifest"}

        return result

    @app.get("/metrics", tags=["Monitoring"])
    async def prometheus_metrics() -> Response:
        """Prometheus metrics endpoint."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/api/v1/config", tags=["Config"])
    async def get_config() -> dict[str, Any]:
        """Return pipeline configuration relevant to the frontend."""
        from invoices.config import Config

        return {
            "confidence_auto_approve": Config.CONFIDENCE_AUTO_APPROVE,
            "confidence_thresholds": {
                "high": Config.CONFIDENCE_AUTO_APPROVE,
                "medium": 0.5,
                "low": 0.0,
            },
        }

    @app.get("/api/v1/schema/fields", response_model=FieldListResponse, tags=["Config"])
    async def get_schema_fields() -> FieldListResponse:
        """Return all extractable fields from the contract schema.

        Returns field metadata including type, required status, confidence thresholds,
        and importance. Fields are sorted with required fields first, then by importance.
        """
        from invoices import schema_registry

        # Load schema to get field definitions
        schema = schema_registry._load()
        field_defs = schema.get("field_definitions", {})
        schema_version = schema.get("version", "unknown")

        # Build field metadata list
        fields = []
        for field_name in field_defs.keys():
            fdef = field_defs[field_name]
            fields.append(
                FieldMetadata(
                    name=field_name,
                    type=fdef.get("type", "text"),
                    required=fdef.get("required", False),
                    confidence_threshold=fdef.get("confidence_threshold", 0.75),
                    importance=fdef.get("importance", 0.5),
                )
            )

        # Sort: required first, then by importance descending
        fields.sort(key=lambda f: (not f.required, -f.importance, f.name))

        return FieldListResponse(
            fields=fields,
            total=len(fields),
            schema_version=schema_version,
        )

    def _fdef_to_detail(field_name: str, fdef: dict[str, Any]) -> FieldDetail:
        """Map a field_definitions entry to a FieldDetail response model."""
        from invoices import schema_registry

        return FieldDetail(
            name=field_name,
            type=fdef.get("type", "text"),
            required=fdef.get("required", False),
            description=fdef.get("description", ""),
            confidence_threshold=fdef.get("confidence_threshold", 0.75),
            importance=fdef.get("importance", 0.5),
            priority_bonus=fdef.get("priority_bonus", 0.0),
            keyword_proximal=schema_registry.is_keyword_proximal(field_name),
            dataverse_column=schema_registry.dataverse_column(field_name),
            computed=fdef.get("computed", False),
        )

    @app.get(
        "/api/v1/schema/fields/{field_name}",
        response_model=FieldDetail,
        tags=["Config"],
    )
    async def get_schema_field(field_name: str) -> FieldDetail:
        """Return full detail for a single schema field.

        Returns 404 if the field name is not defined in the contract schema.
        """
        from invoices import schema_registry

        fdef = schema_registry.field_def(field_name)
        if not fdef:
            raise HTTPException(
                status_code=404, detail=f"Field not found: {field_name}"
            )
        return _fdef_to_detail(field_name, fdef)

    @app.put(
        "/api/v1/schema/fields/{field_name}",
        response_model=FieldDetail,
        tags=["Config"],
    )
    async def update_schema_field(
        field_name: str, body: FieldUpdateRequest
    ) -> FieldDetail:
        """Update tunable properties for a single schema field.

        Merges non-None fields from the request body into the schema JSON and
        writes back atomically. Clears the schema cache so the next pipeline
        run picks up the change.

        Returns 404 if the field name is not defined in the contract schema.
        """
        import os

        from invoices import paths, schema_registry

        fdef = schema_registry.field_def(field_name)
        if not fdef:
            raise HTTPException(
                status_code=404, detail=f"Field not found: {field_name}"
            )

        schema_path = paths.get_repo_root() / "schema" / "contract.invoice.json"
        with open(schema_path, encoding="utf-8") as f:
            schema: dict[str, Any] = json.load(f)

        field_entry: dict[str, Any] = schema.setdefault(
            "field_definitions", {}
        ).setdefault(field_name, {})

        updates = body.model_dump(exclude_none=True)
        for key, value in updates.items():
            if key == "dataverse_column" and value == "":
                field_entry[key] = None
            else:
                field_entry[key] = value

        tmp_path = schema_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        os.rename(tmp_path, schema_path)

        schema_registry.clear_cache()

        updated_fdef = schema_registry.field_def(field_name)
        return _fdef_to_detail(field_name, updated_fdef)

    # =========================================================================
    # Active Learning Queue
    # =========================================================================

    @app.get(
        "/api/v1/active-learning/queue",
        response_model=QueueResponse,
        tags=["Active Learning"],
    )
    async def get_review_queue(
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        priority_level: str | None = Query(
            None, description="Filter by priority level"
        ),
        field: str | None = Query(None, description="Filter by field name"),
    ) -> QueueResponse:
        """Get the prioritized review queue.

        Returns items sorted by priority score (highest first).
        """
        # Load queue from review parquet
        items = await _load_review_queue(app.state.data_dir)

        # Apply filters
        if priority_level:
            items = [i for i in items if i.priority_level == priority_level]
        if field:
            items = [i for i in items if i.field == field]

        # Sort by priority (highest first)
        items.sort(key=lambda x: x.priority_score, reverse=True)

        # Paginate
        total_count = len(items)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_items = items[start_idx:end_idx]

        return QueueResponse(
            items=page_items,
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    # =========================================================================
    # Document Details
    # =========================================================================

    @app.get(
        "/api/v1/active-learning/document/{doc_id}",
        response_model=DocumentDetail,
        tags=["Active Learning"],
    )
    async def get_document_detail(doc_id: str) -> DocumentDetail:
        """Get detailed information about a document.

        Args:
            doc_id: Document identifier (e.g., fs:abc123...)

        Returns:
            Document details including all predictions
        """
        doc_id = _validate_doc_id(doc_id)
        data_dir: Path = app.state.data_dir
        result = await _load_prediction(data_dir, doc_id)

        if result is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        prediction, sha256 = result
        pages = prediction.get("pages", 1)

        return DocumentDetail(
            doc_id=doc_id,
            sha256=sha256,
            pages=pages,
            predictions=prediction.get("fields", {}),
            pdf_url=f"/api/v1/documents/{sha256}/pdf",
        )

    # =========================================================================
    # Label Submission
    # =========================================================================

    @app.post(
        "/api/v1/active-learning/document/{doc_id}/label",
        response_model=LabelResponse,
        tags=["Active Learning"],
    )
    async def submit_label(
        doc_id: str, submission: LabelSubmission, request: Request
    ) -> LabelResponse:
        """Submit a label correction for a field.

        Args:
            doc_id: Document identifier
            submission: Label correction details
            request: FastAPI request (for app.state access)

        Returns:
            Submission result
        """
        doc_id = _validate_doc_id(doc_id)

        # Validate field name against schema (cached at startup, or load fallback)
        try:
            schema = getattr(request.app.state, "schema", None)
            if schema is None:
                from invoices import utils

                schema = utils.load_contract_schema()
            valid_fields = schema.get("fields", [])
            if submission.field not in valid_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid field name: {submission.field}. Must be one of {list(valid_fields)}",
                )
        except HTTPException:
            raise
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.error("schema_validation_failed", error=str(e))
            raise HTTPException(
                status_code=500, detail="Schema validation error"
            ) from e

        # Validate bbox format if provided
        if submission.correct_bbox is not None:
            if (
                not isinstance(submission.correct_bbox, list)
                or len(submission.correct_bbox) != 4
            ):
                raise HTTPException(
                    status_code=400,
                    detail="Bounding box must be a list of 4 floats [x0, y0, x1, y1]",
                )
            if not all(isinstance(v, (int, float)) for v in submission.correct_bbox):
                raise HTTPException(
                    status_code=400, detail="Bounding box values must be numeric"
                )

        logger.info(
            "label_submitted",
            doc_id=doc_id,
            field=submission.field,
            has_bbox=submission.correct_bbox is not None,
            action=submission.action,
        )

        # Store the label for training
        try:
            await _store_label(
                app.state.data_dir,
                doc_id=doc_id,
                field=submission.field,
                value=submission.correct_value,
                bbox=submission.correct_bbox,
                notes=submission.notes,
                action=submission.action or "correct",
            )
        except (OSError, ValueError) as e:
            logger.error("label_storage_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Failed to store label") from e

        # Update Prometheus metrics
        try:
            from invoices.metrics import (
                corrections_submitted,
                labels_submitted,
                update_queue_metrics,
            )

            action = submission.action or "correct"
            labels_submitted.labels(action=action).inc()
            if action == "correct":
                corrections_submitted.labels(field=submission.field).inc()

            # Compute and update correction rate for this field
            await asyncio.to_thread(
                _update_field_correction_metrics_sync,
                app.state.data_dir,
                submission.field,
            )

            # Refresh queue metrics after label submission
            from invoices import emit

            queue = emit.get_review_queue()
            if not queue.empty:
                priority_counts = queue["priority_level"].value_counts().to_dict()
                update_queue_metrics(len(queue), priority_counts)
        except ImportError:
            pass  # prometheus-client not installed

        return LabelResponse(
            success=True,
            doc_id=doc_id,
            field=submission.field,
            message=f"Label submitted for {submission.field}",
        )

    # =========================================================================
    # Approval Submission
    # =========================================================================

    @app.post(
        "/api/v1/active-learning/document/{doc_id}/approve",
        response_model=ApprovalResponse,
        tags=["Active Learning"],
    )
    async def approve_prediction(
        doc_id: str, submission: ApprovalSubmission
    ) -> ApprovalResponse:
        """Approve a prediction as correct.

        Args:
            doc_id: Document identifier
            submission: Approval details

        Returns:
            Approval result
        """
        doc_id = _validate_doc_id(doc_id)
        logger.info(
            "prediction_approved",
            doc_id=doc_id,
            field=submission.field,
        )

        # Store the approval
        await _store_approval(
            app.state.data_dir,
            doc_id=doc_id,
            field=submission.field,
            notes=submission.notes,
        )

        # Update Prometheus metrics
        try:
            from invoices.metrics import approvals_submitted, update_queue_metrics

            approvals_submitted.labels(field=submission.field).inc()

            # Refresh queue metrics after approval
            from invoices import emit

            queue = emit.get_review_queue()
            if not queue.empty:
                priority_counts = queue["priority_level"].value_counts().to_dict()
                update_queue_metrics(len(queue), priority_counts)
        except ImportError:
            pass  # prometheus-client not installed

        return ApprovalResponse(
            success=True,
            doc_id=doc_id,
            field=submission.field,
            message=f"Prediction approved for {submission.field}",
        )

    # =========================================================================
    # Reviewer Statistics
    # =========================================================================

    @app.get(
        "/api/v1/active-learning/stats",
        response_model=ReviewerStats,
        tags=["Active Learning"],
    )
    async def get_reviewer_stats() -> ReviewerStats:
        """Get reviewer activity statistics.

        Returns:
            Reviewer performance metrics
        """
        # Load stats from storage
        stats = await _load_reviewer_stats(app.state.data_dir)
        return stats

    # =========================================================================
    # Train (align + retrain in one click from Grafana)
    # =========================================================================

    async def _run_training_pipeline() -> dict[str, Any]:
        """Run the full align → train → decode → emit pipeline.

        Returns a structured response that always succeeds at the HTTP level.
        Expected conditions (quality gate failure, low data) are reported as
        partial successes, not errors.
        """
        from invoices import labels, train

        # Step 1: align corrections + approvals → parquet
        try:
            alignment = await asyncio.to_thread(labels.align_corrections_cli)
        except Exception as e:
            logger.error("alignment_failed", error=str(e))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "AlignmentFailed",
                    "message": (
                        "Could not align labels with candidates. "
                        "Make sure you have reviewed some documents first."
                    ),
                },
            ) from e

        corr = alignment.get("corrections", {})
        appr = alignment.get("approvals", {})
        total_aligned = corr.get("aligned", 0) + appr.get("aligned", 0)

        logger.info(
            "alignment_complete",
            corrections_aligned=corr.get("aligned", 0),
            approvals_aligned=appr.get("aligned", 0),
        )

        # Step 2: retrain the ranker on all aligned labels
        try:
            train_result = await asyncio.to_thread(train.train_models)
        except Exception as e:
            logger.error("training_failed", error=str(e))
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "TrainingFailed",
                    "message": "Training failed unexpectedly",
                },
            ) from e

        # Bust model cache so next decode uses fresh model
        from invoices.decoder import clear_model_cache

        clear_model_cache()

        logger.info("train_complete", result=str(train_result))

        # Determine if we should expect models to load:
        # - Training skipped (no data) → no models to load
        # - Quality gate failed → models exist but blocked from loading
        # Both cases: decode with heuristics only, no model verification
        training_skipped = train_result.get("status") == "skipped"
        quality_gate_passed = train_result.get("ranker_quality_passed", False)
        use_models = not training_skipped and quality_gate_passed

        if not training_skipped and not quality_gate_passed:
            logger.info(
                "quality_gate_failed_using_heuristics",
                total_docs=train_result.get("total_docs", 0),
                models_trained=train_result.get("models_trained", 0),
            )

        # Step 3: re-decode all documents (never let model load failure crash)
        from invoices import decoder

        try:
            decode_result = await asyncio.to_thread(
                decoder.decode_all_documents,
                expect_models=False,  # Never raise on load failure
            )
        except Exception as e:
            logger.error("decode_failed_after_train", error=str(e))
            # Decode failure is serious but not a 500 — training still happened
            decode_result = {}

        model_state = decoder.get_last_decode_model_state()

        # When quality gate passed but models failed to load, fall back
        # to heuristics rather than crashing — the user still gets value
        if use_models and not model_state.get("models_loaded"):
            logger.warning(
                "post_train_model_load_failed_fallback",
                model_state=model_state,
                msg="Models trained but failed to load — falling back to heuristics",
            )
            use_models = False
            quality_gate_passed = False

        logger.info("decode_complete", result=str(decode_result))

        # Step 4: regenerate predictions and queue
        from invoices import emit

        try:
            emit_result = await asyncio.to_thread(
                emit.emit_all_documents,
                precomputed_assignments=decode_result,
            )
        except Exception as e:
            logger.error("emit_failed_after_train", error=str(e))
            emit_result = {}

        logger.info("emit_complete", result=str(emit_result))

        # Update Prometheus metrics
        try:
            from invoices.metrics import training_runs, update_model_metrics

            training_runs.labels(status="success").inc()
            update_model_metrics(
                loaded=model_state.get("models_loaded", False),
                field_count=len(model_state.get("model_fields", [])),
                model_type=model_state.get("model_type", "none"),
            )
        except ImportError:
            pass  # prometheus-client not installed

        # Step 5: Compute calibration (ECE) and update heuristic confidence
        try:
            from invoices.calibration import (
                compute_ece_from_labeled_data,
                compute_empirical_accuracy,
            )
            from invoices.config import Config as _config
            from invoices.metrics import update_calibration_metrics
            from invoices.paths import get_data_dir

            data_dir = get_data_dir()

            # ECE: are confidence scores honest?
            ece = compute_ece_from_labeled_data(data_dir)
            if ece is not None:
                update_calibration_metrics(ece)

            # Adaptive heuristic base: derive from actual accuracy
            accuracy = compute_empirical_accuracy(data_dir)
            if accuracy is not None:
                _config.calibrate_heuristic_base(accuracy)
        except ImportError:
            logger.warning(
                "calibration modules not available — skipping post-train calibration"
            )

        # Step 6: Update per-field accuracy from corrections + approvals
        try:
            from invoices.paths import get_data_dir as _get_data_dir

            data_dir = _get_data_dir()
            for field_name in train_result.get("training_stats", {}):
                _update_field_correction_metrics_sync(data_dir, field_name)
        except ImportError:
            logger.warning(
                "paths module not available — skipping field correction metrics"
            )

        # Build the response message based on outcome
        if training_skipped:
            message = (
                "Not enough labeled data to train. "
                "Review and correct more documents first."
            )
        elif quality_gate_passed:
            message = "Model improved! New extractions will be more accurate."
        else:
            message = (
                "Training completed but quality gate not met — "
                "using heuristics. Label more documents to improve."
            )

        result: dict[str, Any] = _convert_numpy(
            {
                "success": True,
                "quality_gate_passed": quality_gate_passed,
                "message": message,
                "total_docs": train_result.get("total_docs", 0),
                "alignment": {
                    "corrections": corr,
                    "approvals": appr,
                    "total_aligned": total_aligned,
                },
                "training": train_result,
                "decode": decode_result,
                "emit": emit_result,
                "model_state": model_state,
            }
        )
        return result

    @app.post(
        "/api/v1/active-learning/train",
        tags=["Active Learning"],
    )
    async def trigger_train() -> dict[str, Any]:
        """Align corrections/approvals with candidates, then retrain the ranker.

        This is the "Retrain" button in Grafana — one click, fully automated.
        """
        # Prevent concurrent training runs
        if _training_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Training already in progress",
            )

        async with _training_lock:
            return await _run_training_pipeline()

    # =========================================================================
    # Pipeline Trigger
    # =========================================================================

    @app.post(
        "/api/v1/pipeline/run",
        tags=["Pipeline"],
    )
    async def trigger_pipeline_run() -> dict[str, Any]:
        """Discover new documents and run the full extraction pipeline.

        Lists new documents from the configured source (SharePoint or local
        seed folder), then runs ingest → tokenize → candidates → decode → emit
        for each new document. Returns a job summary.

        Protected by the same API key middleware as /api/v1/active-learning/train.
        """
        if _pipeline_lock.locked():
            raise HTTPException(
                status_code=409,
                detail="Pipeline run already in progress",
            )

        async with _pipeline_lock:
            from invoices.orchestrator import create_orchestrator

            try:
                orchestrator = await create_orchestrator()
            except ValueError as e:
                logger.error("orchestrator_config_error", error=str(e))
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "OrchestratorConfigError",
                        "message": "Pipeline configuration error",
                    },
                ) from e

            async with orchestrator:
                batch_result = await orchestrator.run_once()

            processed = [
                {"doc_id": r.doc_id, "sha256": r.sha256} for r in batch_result.processed
            ]
            failed = [
                {"doc_id": r.doc_id, "sha256": r.sha256, "error": r.error}
                for r in batch_result.failed
            ]

            logger.info(
                "pipeline_run_complete",
                processed=len(processed),
                skipped=len(batch_result.skipped),
                failed=len(failed),
            )

            # Update Prometheus queue metrics after new documents are emitted
            try:
                from invoices import emit
                from invoices.metrics import update_queue_metrics

                queue = emit.get_review_queue()
                if not queue.empty:
                    priority_counts = queue["priority_level"].value_counts().to_dict()
                    update_queue_metrics(len(queue), priority_counts)
            except (ImportError, OSError, KeyError, ValueError):
                pass

            return {
                "success": True,
                "processed": len(processed),
                "skipped": len(batch_result.skipped),
                "failed": len(failed),
                "documents": {
                    "processed": processed,
                    "skipped": batch_result.skipped,
                    "failed": failed,
                },
            }

    # =========================================================================
    # PDF Access
    # =========================================================================

    @app.get("/api/v1/documents/{sha256}/pdf", tags=["Documents"])
    async def get_document_pdf(sha256: str) -> FileResponse:
        """Serve a PDF document.

        Args:
            sha256: SHA256 hash of the document

        Returns:
            PDF file response
        """
        # Validate sha256 format to prevent path traversal
        sha256 = _validate_sha256(sha256)

        data_dir: Path = app.state.data_dir
        pdf_path = data_dir / "ingest" / "raw" / f"{sha256}.pdf"

        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail=f"PDF not found: {sha256}")

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": "inline"},
        )

    @app.get("/api/v1/documents/{sha256}/tokens", tags=["Documents"])
    async def get_document_tokens(
        sha256: str, page: int = Query(0, ge=0, description="Page number (0-indexed)")
    ) -> list[dict[str, Any]]:
        """Get tokens for a document page.

        Args:
            sha256: SHA256 hash of the document
            page: Page number (0-indexed)

        Returns:
            List of tokens with text and normalized bounding boxes
        """
        # Validate sha256 format to prevent path traversal
        sha256 = _validate_sha256(sha256)

        from invoices.paths import get_tokens_path

        tokens_path = get_tokens_path(sha256)

        if not tokens_path.exists():
            raise HTTPException(status_code=404, detail=f"Tokens not found: {sha256}")

        try:
            df = read_parquet_safe(tokens_path, on_error="raise")
            if df is None:
                raise HTTPException(status_code=404, detail="Tokens not found")

            page_tokens = df[df["page_idx"] == page]

            result = []
            for _, row in page_tokens.iterrows():
                result.append(
                    {
                        "text": str(row["text"]),
                        "x0": float(row["bbox_norm_x0"]),
                        "y0": float(row["bbox_norm_y0"]),
                        "x1": float(row["bbox_norm_x1"]),
                        "y1": float(row["bbox_norm_y1"]),
                    }
                )

            return result
        except (OSError, KeyError, ValueError) as e:
            logger.error("token_load_failed", sha256=sha256, page=page, error=str(e))
            raise HTTPException(status_code=500, detail="Failed to load tokens") from e


# =============================================================================
# Data Access Functions
# =============================================================================


async def _load_review_queue(data_dir: Path) -> list[QueueItem]:
    """Load the review queue from storage.

    Args:
        data_dir: Base data directory

    Returns:
        List of queue items
    """
    import json

    queue_file = data_dir / "review" / "queue.json"

    if not queue_file.exists():
        # Try loading from parquet if JSON doesn't exist
        parquet_file = data_dir / "review" / "queue.parquet"
        if parquet_file.exists():
            return await _load_queue_from_parquet(parquet_file)
        return []

    try:
        with open(queue_file) as f:
            data = json.load(f)
        return [QueueItem(**item) for item in data]
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("queue_load_failed", error=str(e))
        return []


async def _load_queue_from_parquet(parquet_path: Path) -> list[QueueItem]:
    """Load queue from parquet file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        List of queue items
    """
    import math

    try:
        df = read_parquet_safe(parquet_path, on_error="warn")
        if df is None:
            return []

        items = []
        for _, row in df.iterrows():
            # Handle NaN and missing values gracefully. The parquet may
            # predate columns like sha256/ml_confidence, or contain NaN
            # for optional string fields (e.g. raw_text on ABSTAIN rows).
            raw_text = row.get("raw_text")
            if raw_text is None or (
                isinstance(raw_text, float) and math.isnan(raw_text)
            ):
                raw_text = None
            else:
                raw_text = str(raw_text)

            sha256 = row.get("sha256", "")
            if sha256 is None or (isinstance(sha256, float) and math.isnan(sha256)):
                # Derive from doc_id (format: fs:{sha256_prefix})
                doc_id = str(row.get("doc_id", ""))
                sha256 = doc_id.replace("fs:", "") if doc_id.startswith("fs:") else ""
            else:
                sha256 = str(sha256)

            ml_conf = row.get("ml_confidence")
            if ml_conf is None or (isinstance(ml_conf, float) and math.isnan(ml_conf)):
                ml_conf = 0.0
            else:
                ml_conf = float(ml_conf)

            items.append(
                QueueItem(
                    doc_id=str(row.get("doc_id", "")),
                    sha256=sha256,
                    field=str(row.get("field", "")),
                    raw_text=raw_text,
                    priority_score=float(row.get("priority_score") or 0.5),
                    priority_level=str(row.get("priority_level", "medium")),
                    ml_confidence=ml_conf,
                    reason=str(row.get("reason", "low_confidence")),
                    scores={},
                )
            )
        return items
    except (OSError, KeyError, ValueError) as e:
        logger.warning("parquet_load_failed", error=str(e))
        return []


async def _load_prediction(
    data_dir: Path, doc_id: str
) -> tuple[dict[str, Any], str] | None:
    """Load prediction for a document.

    Args:
        data_dir: Base data directory
        doc_id: Document identifier

    Returns:
        Tuple of (prediction dict, full sha256) or None if not found
    """
    import json

    # Extract SHA256 prefix from doc_id (format: fs:{sha256[:16]})
    sha256_prefix = doc_id.replace("fs:", "")

    # Find matching prediction file
    predictions_dir = data_dir / "predictions"
    if not predictions_dir.exists():
        return None

    for pred_file in predictions_dir.glob("*.json"):
        if pred_file.stem.startswith(sha256_prefix):
            try:
                with open(pred_file) as f:
                    return json.load(f), pred_file.stem
            except (json.JSONDecodeError, OSError):
                continue

    return None


async def _store_label(
    data_dir: Path,
    doc_id: str,
    field: str,
    value: str,
    bbox: list[float] | None,
    notes: str | None,
    action: str = "correct",
) -> None:
    """Store a label correction.

    Args:
        data_dir: Base data directory
        doc_id: Document identifier
        field: Field name
        value: Correct value
        bbox: Correct bounding box
        notes: Reviewer notes
        action: Action type (correct, not_applicable, not_in_document, reject)
    """
    import json

    labels_dir = data_dir / "labels" / "corrections"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Create label entry
    label_entry = {
        "doc_id": doc_id,
        "field": field,
        "correct_value": value,
        "correct_bbox": bbox,
        "notes": notes,
        "action": action,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Append to labels file
    labels_file = labels_dir / "corrections.jsonl"
    with open(labels_file, "a") as f:
        f.write(json.dumps(label_entry) + "\n")


async def _store_approval(
    data_dir: Path,
    doc_id: str,
    field: str,
    notes: str | None,
) -> None:
    """Store an approval.

    Args:
        data_dir: Base data directory
        doc_id: Document identifier
        field: Field name
        notes: Reviewer notes
    """
    import json

    approvals_dir = data_dir / "labels" / "approvals"
    approvals_dir.mkdir(parents=True, exist_ok=True)

    # Create approval entry
    approval_entry = {
        "doc_id": doc_id,
        "field": field,
        "notes": notes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Append to approvals file
    approvals_file = approvals_dir / "approvals.jsonl"
    with open(approvals_file, "a") as f:
        f.write(json.dumps(approval_entry) + "\n")


def _update_field_correction_metrics_sync(data_dir: Path, field: str) -> None:
    """Update per-field correction rate and accuracy metrics (sync version).

    Args:
        data_dir: Base data directory
        field: Field name to update metrics for
    """
    try:
        from invoices.metrics import update_correction_metrics
    except ImportError:
        logger.warning(
            "metrics module not available — skipping correction metrics update"
        )
        return

    import json

    corrections_file = data_dir / "labels" / "corrections" / "corrections.jsonl"
    approvals_file = data_dir / "labels" / "approvals" / "approvals.jsonl"

    field_corrections = 0
    field_approvals = 0

    # Count corrections for this field
    if corrections_file.exists():
        try:
            with open(corrections_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("field") == field:
                            field_corrections += 1
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("corrections_file_read_failed", field=field, error=str(e))

    # Count approvals for this field
    if approvals_file.exists():
        try:
            with open(approvals_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("field") == field:
                            field_approvals += 1
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("approvals_file_read_failed", field=field, error=str(e))

    total_reviews = field_corrections + field_approvals
    if total_reviews > 0:
        correction_rate = field_corrections / total_reviews
        accuracy = field_approvals / total_reviews
        update_correction_metrics(field, correction_rate, accuracy)


async def _load_reviewer_stats(data_dir: Path) -> ReviewerStats:
    """Load reviewer statistics from corrections and approvals files."""
    import json
    from datetime import datetime, timezone

    corrections_file = data_dir / "labels" / "corrections" / "corrections.jsonl"
    approvals_file = data_dir / "labels" / "approvals" / "approvals.jsonl"

    total_reviews = 0
    reviews_today = 0
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for jsonl_file in [corrections_file, approvals_file]:
        if not jsonl_file.exists():
            continue
        try:
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_reviews += 1
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", "")
                        if ts.startswith(today_str):
                            reviews_today += 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

    return ReviewerStats(
        total_reviews=total_reviews,
        reviews_today=reviews_today,
        accuracy_rate=0.0,
        avg_review_time_seconds=0.0,
    )


# Lazy app instantiation to avoid requiring INVOICEX_API_KEY during import.
# This prevents pytest collection failures in CI when the module is imported
# but the app isn't actually needed.
_app: FastAPI | None = None


def get_app() -> FastAPI:
    """Get or create the FastAPI app instance (lazy singleton).

    This factory function delays app creation until first access, which prevents
    production-mode environment checks (like INVOICEX_API_KEY validation) from
    running during pytest collection or module imports.

    For production deployments, use `invoices.api:get_app` as the ASGI app factory.
    Tests should call `create_app()` directly to get fresh app instances.
    """
    global _app
    if _app is None:
        _app = create_app()
    return _app
