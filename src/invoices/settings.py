"""Pydantic BaseSettings for env-var-backed configuration.

This is the single source of truth for all pipeline configuration.
config.py re-exports ``settings`` as ``Config`` for backward compatibility.

Usage:
    from invoices.config import Config    # preferred (bridge)
    from invoices.settings import settings  # direct access

    Config.none_bias                # INVOICEX_NONE_BIAS
    Config.is_dev                   # True when ENVIRONMENT=development
"""

from __future__ import annotations

import logging
import os

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_VALID_STORAGE_BACKENDS = ("blob",)
_VALID_DOCUMENT_SOURCES = ("none", "sharepoint")
_VALID_OUTPUT_BACKENDS = ("postgres", "dataverse")


class Settings(BaseSettings):
    """Env-var-backed pipeline settings (mirrors PipelineConfig env fields)."""

    model_config = {
        "frozen": True,
        "validate_assignment": True,
        "env_prefix": "INVOICEX_",
    }

    # ── Version stamps ──────────────────────────────────────────────
    # feature_version and decoder_version are derived constants in version.py —
    # never hand-edited here.  See FEATURE_VERSION and decoder_version() there.
    calibration_version: str = "none"

    # ── Decoder ──────────────────────────────────────────────────────
    none_bias: float = 0.05
    decoder_base_cost: float = 2.0
    bootstrap_ml_score_weight: float = 0.3
    pruning_threshold: float = 0.0
    pruning_max_candidates: int = 0
    pruning_min_trigger: int = 0
    # ── Confidence ───────────────────────────────────────────────────
    confidence_auto_approve: float = 0.85
    confidence_heuristic_base: float = 0.5
    # ── Candidate generation ─────────────────────────────────────────
    trace_candidates: bool = False
    early_page_boost: float = 0.5
    early_page_max_idx: int = 2
    # ── ML ranker ────────────────────────────────────────────────────
    ml_score_weight: float = 0.7
    use_ranker: bool = False
    # ── MLflow ───────────────────────────────────────────────────────
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5050"
    mlflow_experiment_name: str = "invoicex"
    mlflow_model_prefix: str = "invoicex-ranker"
    mlflow_artifact_root: str = Field(
        default="/mlflow-artifacts",
        validation_alias="MLFLOW_ARTIFACT_ROOT",
    )
    # ── Backend switches ─────────────────────────────────────────────
    storage_backend: str = "blob"
    document_source: str = "none"
    output_backend: str = "postgres"
    # ── Azure Blob Storage ───────────────────────────────────────────
    azure_storage_account_name: str = Field(
        default="",
        validation_alias="AZURE_STORAGE_ACCOUNT_NAME",
    )
    azure_storage_container_name: str = Field(
        default="invoicex",
        validation_alias="AZURE_STORAGE_CONTAINER_NAME",
    )
    azure_storage_connection_string: str = Field(
        default="",
        validation_alias="AZURE_STORAGE_CONNECTION_STRING",
    )
    # ── SharePoint ───────────────────────────────────────────────────
    sharepoint_site_id: str = Field(
        default="",
        validation_alias="SHAREPOINT_SITE_ID",
    )
    sharepoint_drive_id: str = Field(
        default="",
        validation_alias="SHAREPOINT_DRIVE_ID",
    )
    sharepoint_folder_path: str = Field(
        default="Invoices/Inbox",
        validation_alias="SHAREPOINT_FOLDER_PATH",
    )
    sharepoint_hostname: str = Field(
        default="",
        validation_alias="SHAREPOINT_HOSTNAME",
    )
    sharepoint_site_path: str = Field(
        default="",
        validation_alias="SHAREPOINT_SITE_PATH",
    )
    sharepoint_connect_timeout_seconds: float = Field(
        default=30.0, description="Graph API connect timeout"
    )
    sharepoint_read_timeout_seconds: float = Field(
        default=300.0, description="Graph API read timeout"
    )
    sharepoint_write_timeout_seconds: float = Field(
        default=30.0, description="Graph API write timeout"
    )
    sharepoint_pool_timeout_seconds: float = Field(
        default=10.0, description="Graph API connection pool timeout"
    )
    sharepoint_download_timeout_seconds: float = Field(
        default=600.0,
        description=(
            "Wall-clock cap on a single SharePoint document download, "
            "including all internal retries (_MAX_RETRIES=3, _MAX_BACKOFF_SECONDS=60). "
            "Prevents a slow-loris Graph API response from blocking the "
            "sequential PgQueuer worker indefinitely."
        ),
    )
    # ── Dataverse ────────────────────────────────────────────────────
    dataverse_environment_url: str = Field(
        default="",
        validation_alias="DATAVERSE_ENVIRONMENT_URL",
    )
    dataverse_client_id: str = Field(
        default="",
        validation_alias="DATAVERSE_CLIENT_ID",
    )
    dataverse_staging_table: str = Field(
        default="invoicex_staging",
        validation_alias="DATAVERSE_STAGING_TABLE",
    )
    dataverse_production_table: str = Field(
        default="invoicex_production",
        validation_alias="DATAVERSE_PRODUCTION_TABLE",
    )
    dataverse_write_enabled: bool = False
    dataverse_timeout_seconds: float = Field(
        default=30.0, description="Dataverse Web API request timeout"
    )
    # ── Orchestrator ─────────────────────────────────────────────────
    # Single source of truth for retry cap — passed to `mark_failed` at every call site.
    orchestrator_max_retries: int = 3
    orchestrator_retry_base_seconds: float = 1.0
    orchestrator_retry_multiplier: float = 4.0
    orchestrator_watch_interval: float = 5.0
    orchestrator_seed_folder: str = "input"
    orchestrator_auth_backoff_max_seconds: float = Field(
        default=300.0,
        description="Max backoff on SharePoint auth errors (ClientAuthenticationError)",
    )
    orchestrator_transient_backoff_max_seconds: float = Field(
        default=60.0, description="Max backoff on transient HTTP/DB errors"
    )
    orchestrator_backoff_min_seconds: float = Field(
        default=2.0, description="Minimum backoff floor between retries"
    )
    doc_timeout_seconds: int = 60
    # ── Application ──────────────────────────────────────────────────
    api_key: str = ""
    environment: str = Field(
        default="production",
        validation_alias="ENVIRONMENT",
    )
    docs_enabled: bool = False
    cors_origins: str = "http://localhost:3000,http://localhost"
    public_url: str = "http://localhost"
    log_level: str = "INFO"
    # ── Model ────────────────────────────────────────────────────────
    model_id: str = Field(
        default="unscored-baseline",
        validation_alias="MODEL_ID",
    )
    model_cache_ttl_seconds: int = 300
    # ── Quality gate / bootstrap ─────────────────────────────────────
    quality_gate_ndcg_threshold: float = Field(
        default=0.5,
        validation_alias="INVOICEX_QUALITY_GATE_THRESHOLD",
    )
    quality_gate_blocking: bool = Field(
        default=False,
        validation_alias="INVOICEX_QUALITY_GATE_BLOCKING",
    )
    bootstrap_doc_threshold: int = 1
    bootstrap_quality_gate_threshold: float = 0.3
    min_positive_examples: int = 1
    bootstrap_confidence_cap: float = 0.80
    # ── Cross-field validation ───────────────────────────────────────
    cross_field_penalty: float = 0.15
    # ── Postgres ─────────────────────────────────────────────────────
    postgres_host: str = Field(
        default="postgres", validation_alias="INVOICEX_LABELS_DB_HOST"
    )
    postgres_port: int = Field(default=5432, validation_alias="INVOICEX_LABELS_DB_PORT")
    postgres_user: str = Field(default="invoicex", validation_alias="POSTGRES_USER")
    postgres_password: str = Field(default="", validation_alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(
        default="invoicex", validation_alias="INVOICEX_LABELS_DB_NAME"
    )
    # ── Blend weights (retrain DAG) ──────────────────────────────────
    blend_ndcg_floor: float = 0.3
    blend_ndcg_ceil: float = 1.0
    blend_min_labels: int = 5
    # ── Calibration ──────────────────────────────────────────────────
    calibration_min_samples: int = Field(default=10, ge=1)
    retrain_chunk_size: int = Field(default=500, ge=1)
    retrain_drop_rate_threshold: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        validation_alias="INVOICEX_RETRAIN_DROP_RATE_THRESHOLD",
    )
    # ── Reextract job hygiene ─────────────────────────────────────────
    # VmRSS admission gate: defer a new reextract job when the process's own
    # resident set (VmRSS from /proc/self/status) already exceeds this value.
    # VmRSS EXCLUDES reclaimable page cache, so streaming PDFs or loading the
    # model file does NOT inflate it.  At concurrency=1 steady-state RSS is
    # ~3.2 GiB on the 11 GiB worker; this gate only fires if there is a genuine
    # residual leak near the hard limit.
    # Default: 9.5 GiB — leaves ~1.5 GiB below the 11 GiB container limit.
    # Override via INVOICEX_REEXTRACT_RSS_GATE_BYTES.
    reextract_rss_gate_bytes: int = Field(
        default=10_200_547_328,  # ~9.5 GiB
        ge=0,
        validation_alias="INVOICEX_REEXTRACT_RSS_GATE_BYTES",
    )
    # VmRSS recycle threshold: if the process's own resident set (VmRSS from
    # /proc/self/status) exceeds this value after a successful reextract job
    # (post gc+trim), the worker sends SIGTERM to itself for a clean restart —
    # docker restart: unless-stopped brings a fresh process with reset RSS.
    # VmRSS EXCLUDES page cache so this only fires on a genuine RSS leak.
    # Default: 9 GiB — at concurrency=1 (~3.2 GiB steady) this effectively
    # never fires unless there is a real leak.  Leaves ~2 GiB under the
    # 11 GiB container limit before the OOM killer acts.
    # Override via INVOICEX_REEXTRACT_WORKER_RECYCLE_BYTES.
    reextract_worker_recycle_bytes: int = Field(
        default=9_663_676_416,  # 9 GiB
        ge=0,
        validation_alias="INVOICEX_REEXTRACT_WORKER_RECYCLE_BYTES",
    )
    # Max wall-time (seconds) for the heavy pipeline call (run_document_pipeline
    # + model load) within a reextract job.  On timeout, asyncio.wait_for raises
    # TimeoutError; the DatabaseRetryEntrypointExecutor wraps it and re-queues
    # with backoff.  NOTE: asyncio cannot cancel a running thread — the thread
    # continues in the background while the queue slot is released so the queue
    # can drain.  The memory gate (above) is the real memory bound; this is
    # purely a liveness gate.
    reextract_pipeline_timeout_seconds: float = Field(
        default=300.0,
        gt=0.0,
        validation_alias="INVOICEX_REEXTRACT_PIPELINE_TIMEOUT_SECONDS",
    )
    # Static concurrency ceiling for the reextract entrypoint.  The memory gate
    # is the real admission control; this is a last-resort defence in depth.
    # Lowered from 2 to 1: pgqueuer requires max_concurrent_tasks >= 2*batch_size;
    # with batch-size=1 and max-concurrent-tasks=2 in the compose command,
    # pgqueuer is structurally satisfied.  This semaphore/entrypoint cap of 1
    # means at most ONE LiLT job (~4 GiB) runs at a time — the second pgqueuer
    # slot parks on the semaphore, well within the 8 GiB worker limit.
    # Override via INVOICEX_REEXTRACT_CONCURRENCY.
    reextract_concurrency: int = Field(
        default=1,
        ge=1,
        validation_alias="INVOICEX_REEXTRACT_CONCURRENCY",
    )

    # ── Field validators (range checks) ─────────────────────────────
    @field_validator("confidence_auto_approve")
    @classmethod
    def _check_confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = f"CONFIDENCE_AUTO_APPROVE must be in [0, 1], got {v}"
            raise ValueError(msg)
        if v < 0.1:
            logger.warning(
                "CONFIDENCE_AUTO_APPROVE=%s is below 0.1 — "
                "all predictions will bypass human review",
                v,
            )
        return v

    @field_validator("ml_score_weight")
    @classmethod
    def _check_ml_weight_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            msg = f"ML_SCORE_WEIGHT must be in [0, 1], got {v}"
            raise ValueError(msg)
        return v

    @field_validator("none_bias")
    @classmethod
    def _check_none_bias_nonzero(cls, v: float) -> float:
        if v == 0.0:
            msg = (
                "DECODER_NONE_BIAS=0 disables all abstention — the decoder "
                "will assign every field regardless of fit. Use a small "
                "positive value (default: 0.05). Set negative to enable "
                "auto-derivation from data."
            )
            raise ValueError(msg)
        return v

    @field_validator("storage_backend")
    @classmethod
    def _check_storage_backend(cls, v: str) -> str:
        if v not in _VALID_STORAGE_BACKENDS:
            msg = (
                f"Invalid STORAGE_BACKEND: '{v}'. "
                f"Must be one of {_VALID_STORAGE_BACKENDS!r}"
            )
            raise ValueError(msg)
        return v

    @field_validator("document_source")
    @classmethod
    def _check_document_source(cls, v: str) -> str:
        if v not in _VALID_DOCUMENT_SOURCES:
            msg = (
                f"Invalid DOCUMENT_SOURCE: '{v}'. "
                f"Must be one of {_VALID_DOCUMENT_SOURCES!r}"
            )
            raise ValueError(msg)
        return v

    @field_validator("output_backend")
    @classmethod
    def _check_output_backend(cls, v: str) -> str:
        if v not in _VALID_OUTPUT_BACKENDS:
            msg = (
                f"Invalid OUTPUT_BACKEND: '{v}'. "
                f"Must be one of {_VALID_OUTPUT_BACKENDS!r}"
            )
            raise ValueError(msg)
        return v

    # ── Empty-string guard ────────────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _strip_empty_strings(cls, values: dict[str, object]) -> dict[str, object]:
        """Treat empty env var values as unset (use field default).

        Deploy workflows may write ``VAR=`` (empty string) for optional
        vars.  Pydantic rejects ``""`` for typed fields like ``int`` or
        ``bool``.  Dropping the key lets the field default take over.
        """
        if isinstance(values, dict):
            return {k: v for k, v in values.items() if v != ""}
        return values

    # ── Connector-mode defaults ──────────────────────────────────────
    @model_validator(mode="before")
    @classmethod
    def _apply_connector_mode_defaults(
        cls, values: dict[str, object]
    ) -> dict[str, object]:
        """When INVOICEX_CONNECTOR_MODE=azure, fill backend defaults."""
        mode = (os.environ.get("INVOICEX_CONNECTOR_MODE") or "").lower().strip()
        if mode != "azure":
            return values

        _AZURE_DEFAULTS: dict[str, tuple[str, str]] = {
            # (field_name, env_var_name): default_value
            "storage_backend": ("INVOICEX_STORAGE_BACKEND", "blob"),
            "document_source": ("INVOICEX_DOCUMENT_SOURCE", "sharepoint"),
            "output_backend": ("INVOICEX_OUTPUT_BACKEND", "dataverse"),
        }
        for field_name, (env_key, azure_default) in _AZURE_DEFAULTS.items():
            # Only set if neither the field nor env var was explicitly provided
            if field_name not in values and env_key not in values:
                values[field_name] = azure_default

        return values

    # ── Cross-field validators ─────────────────────────────────────────
    @model_validator(mode="after")
    def _check_dataverse_quality_gate_linkage(self) -> Settings:
        """Refuse to start if Dataverse writes are enabled without blocking gate."""
        if self.dataverse_write_enabled and not self.quality_gate_blocking:
            raise ValueError(
                "INVOICEX_DATAVERSE_WRITE_ENABLED=true requires "
                "INVOICEX_QUALITY_GATE_BLOCKING=true — "
                "writes must not go live while the quality gate is advisory-only"
            )
        return self

    @model_validator(mode="after")
    def _check_mlflow_artifact_root(self) -> Settings:
        """Reject MLflow config that won't work in production."""
        if not self.use_mlflow:
            return self
        # Local artifact paths (e.g. /mlartifacts) are valid and durable when
        # backed by a named volume — the sanctioned approach here: the VM managed
        # identity has no Storage Blob RBAC and the architecture mandates no blob
        # storage. The mlflow server (--serve-artifacts) owns the volume; the
        # worker reaches artifacts via the mlflow-artifacts:/ proxy. So we do NOT
        # reject local mlflow_artifact_root values.
        uri = self.mlflow_tracking_uri
        if "localhost" in uri or "127.0.0.1" in uri:
            raise ValueError(
                f"MLFLOW_TRACKING_URI='{uri}' points to localhost — "
                "unreachable from a container. Use the service name."
            )
        return self

    @model_validator(mode="after")
    def _check_blend_ndcg_range(self) -> Settings:
        """Enforce 0.0 <= blend_ndcg_floor <= blend_ndcg_ceil <= 1.0."""
        if not (0.0 <= self.blend_ndcg_floor <= self.blend_ndcg_ceil <= 1.0):
            raise ValueError(
                f"Blend NDCG range invalid: need "
                f"0.0 <= floor ({self.blend_ndcg_floor}) "
                f"<= ceil ({self.blend_ndcg_ceil}) <= 1.0"
            )
        return self

    # ── Production validation (called explicitly, not on instantiation) ──
    def validate_for_production(self) -> None:
        """Validate environment-specific settings for production.

        Call this explicitly (e.g. in api.py startup) rather than at
        instantiation time, so that tests and dev usage aren't blocked
        by production-only requirements.

        Raises ValueError with all validation errors collected.
        """
        errors: list[str] = []

        # Backend credential checks
        if self.storage_backend == "blob":
            if (
                not self.azure_storage_account_name
                and not self.azure_storage_connection_string
            ):
                errors.append(
                    "STORAGE_BACKEND=blob requires "
                    "AZURE_STORAGE_ACCOUNT_NAME or "
                    "AZURE_STORAGE_CONNECTION_STRING"
                )

        if self.document_source == "sharepoint":
            has_site = bool(self.sharepoint_site_id) or bool(
                self.sharepoint_hostname and self.sharepoint_site_path
            )
            if not has_site:
                errors.append(
                    "DOCUMENT_SOURCE=sharepoint requires "
                    "SHAREPOINT_SITE_ID or both "
                    "SHAREPOINT_HOSTNAME + SHAREPOINT_SITE_PATH"
                )
            if not os.environ.get("AZURE_TENANT_ID"):
                errors.append("DOCUMENT_SOURCE=sharepoint requires AZURE_TENANT_ID")

        if self.output_backend == "dataverse":
            if not self.dataverse_environment_url:
                errors.append(
                    "OUTPUT_BACKEND=dataverse requires DATAVERSE_ENVIRONMENT_URL"
                )
            if not self.dataverse_client_id:
                errors.append("OUTPUT_BACKEND=dataverse requires DATAVERSE_CLIENT_ID")
            if not os.environ.get("AZURE_TENANT_ID"):
                errors.append("OUTPUT_BACKEND=dataverse requires AZURE_TENANT_ID")

        # Production environment checks
        if not self.is_dev:
            if not os.environ.get("INVOICEX_CORS_ORIGINS"):
                errors.append(
                    "INVOICEX_CORS_ORIGINS must be set outside development. "
                    "Set ENVIRONMENT=development for local testing."
                )
            if any(
                placeholder in self.public_url
                for placeholder in ("localhost", "YOUR_DOMAIN", "REPLACE_ME")
            ):
                errors.append(
                    f"PUBLIC_URL must be set to a real domain outside "
                    f"development, got {self.public_url!r}. "
                    f"Set ENVIRONMENT=development for local testing."
                )

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ── Computed properties ──────────────────────────────────────────
    @property
    def is_dev(self) -> bool:
        """True when ENVIRONMENT=development."""
        return self.environment == "development"

    @property
    def is_azure_mode(self) -> bool:
        """True if any backend is configured for Azure."""
        return (
            self.storage_backend == "blob"
            or self.document_source == "sharepoint"
            or self.output_backend == "dataverse"
        )

    @property
    def postgres_conninfo(self) -> str:
        """psycopg conninfo string built from the five Postgres settings fields."""
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"user={self.postgres_user} password={self.postgres_password} "
            f"dbname={self.postgres_db}"
        )

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS comma-separated string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# Module-level singleton — the single Settings instance.
settings = Settings()
