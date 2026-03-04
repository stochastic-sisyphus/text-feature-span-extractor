"""Centralized configuration — single source of truth for all parameters.

Usage:
    from invoices.config import Config

    threshold = Config.CONFIDENCE_AUTO_APPROVE
    none_bias = Config.DECODER_NONE_BIAS
"""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T", int, float, str, bool)


def _env(key: str, default: T) -> T:
    """Read env var, cast to the type of *default*. Returns *default* on miss/error."""
    raw = os.environ.get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.lower() in ("true", "1", "yes", "on")  # type: ignore[return-value]
    try:
        return type(default)(raw)  # type: ignore[return-value]
    except (ValueError, TypeError):
        return default


# Connector mode defaults: INVOICEX_CONNECTOR_MODE=azure → these become
# the defaults for individual backend vars. Individual vars still override.
_CONNECTOR_MODE_DEFAULTS: dict[str, dict[str, str]] = {
    "azure": {
        "INVOICEX_STORAGE_BACKEND": "blob",
        "INVOICEX_DOCUMENT_SOURCE": "sharepoint",
        "INVOICEX_OUTPUT_BACKEND": "dataverse",
    },
}


def _env_with_connector(key: str, default: str) -> str:
    """Get string from env, falling back to CONNECTOR_MODE defaults if set."""
    explicit = os.environ.get(key)
    if explicit is not None:
        return explicit
    mode = os.environ.get("INVOICEX_CONNECTOR_MODE", "").lower()
    return _CONNECTOR_MODE_DEFAULTS.get(mode, {}).get(key, default)


# --- Month constants (derived from stdlib) ---
_MONTH_ABBREVS = frozenset(m.lower() for m in calendar.month_abbr if m)
_MONTH_NAMES_FULL = frozenset(m.lower() for m in calendar.month_name if m)
_MONTH_NAMES_ALL = _MONTH_ABBREVS | _MONTH_NAMES_FULL | {"sept"}

# --- XGBoost base params (shared between classifier and ranker) ---
_XGB_BASE: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "random_state": 42,
    "seed": 42,
    "n_jobs": 1,
    "verbosity": 0,
}


@dataclass(frozen=True)
class DecoderWeights:
    """Named constants for decoder scoring — not env-configurable."""

    # Header region
    HEADER_REGION_BONUS: float = 0.8

    @classmethod
    def load_tuned(cls) -> DecoderWeights:
        """Load tuned weights from data/tuned_weights.json if available.

        Falls back to defaults if file doesn't exist or is invalid.
        """
        import json

        tuned_path = (
            Path(os.environ.get("INVOICEX_DATA_DIR", "data")) / "tuned_weights.json"
        )
        if not tuned_path.is_absolute():
            # Try relative to repo root
            candidate = Path(__file__).parent.parent.parent / tuned_path
            if candidate.exists():
                tuned_path = candidate

        if not tuned_path.exists():
            return cls()

        try:
            with open(tuned_path) as f:
                overrides = json.load(f)
            # Only apply keys that are valid DecoderWeights fields
            valid = {
                k: float(v)
                for k, v in overrides.items()
                if hasattr(cls, k) and k != "load_tuned"
            }
            return cls(**valid) if valid else cls()
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return cls()

    HEADER_TOP_CLOSE_BONUS: float = 0.5
    HEADER_TOP_MID_BONUS: float = 0.3
    HEADER_TOP_CLOSE_THRESHOLD: float = 0.3
    HEADER_TOP_MID_THRESHOLD: float = 0.5

    # Bucket affinity
    BUCKET_MATCH_BONUS: float = 0.8
    BUCKET_MISMATCH_STRONG: float = -0.5
    BUCKET_MISMATCH_MODERATE: float = -0.3
    BUCKET_MISMATCH_MILD: float = -0.4
    BUCKET_KEYWORD_PROXIMAL_BONUS: float = 0.2

    # Directional (anchor-based)
    DIRECTIONAL_BELOW_CLOSE_BONUS: float = 0.4
    DIRECTIONAL_BELOW_CLOSE_THRESHOLD: float = 0.3
    DIRECTIONAL_READING_ORDER_BONUS: float = 0.3
    DIRECTIONAL_READING_ORDER_THRESHOLD: float = 0.2
    DIRECTIONAL_SAME_ROW_BONUS: float = 0.3
    DIRECTIONAL_SAME_ROW_THRESHOLD: float = 0.2
    DIRECTIONAL_FAR_PENALTY: float = -0.2
    DIRECTIONAL_FAR_THRESHOLD: float = 0.5

    # Footer region
    FOOTER_REGION_BONUS: float = 0.05
    FOOTER_Y_THRESHOLD: float = 0.25

    # Text pattern validation
    EMPTY_TEXT_PENALTY: float = -0.5
    INV_PREFIX_WRONG_FIELD_PENALTY: float = -1.5
    TEXT_PATTERN_NEGATIVE_AMPLIFIER: float = 1.5

    # Cost combination weights
    PROXIMITY_WEIGHT: float = 0.15
    SECTION_PRIOR_WEIGHT: float = 0.1
    COHESION_WEIGHT: float = 0.1
    COHESION_NORMALIZER: float = 100.0
    DIRECTIONAL_DAMPENING: float = 0.5


DWeights = DecoderWeights.load_tuned()


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the invoice extraction pipeline."""

    # Version stamps
    FEATURE_VERSION: str = "v1"
    DECODER_VERSION: str = "v1"
    CALIBRATION_VERSION: str = "none"

    # Decoder parameters
    DECODER_NONE_BIAS: float = field(
        default_factory=lambda: _env("INVOICEX_NONE_BIAS", 0.05)
    )
    DECODER_BASE_COST: float = 2.0
    BUCKET_AFFINITY_STRONG: float = 0.6
    BUCKET_AFFINITY_WEAK: float = 0.3
    BUCKET_PENALTY_MILD: float = -0.1
    BUCKET_PENALTY_MODERATE: float = -0.2
    BUCKET_PENALTY_STRONG: float = -0.4

    # Confidence scoring
    CONFIDENCE_FLOOR: float = 0.0
    CONFIDENCE_CEILING: float = 1.0
    CONFIDENCE_ABSTAIN: float = 0.0
    CONFIDENCE_INFERRED: float = 0.90
    CONFIDENCE_HEURISTIC_BASE: float = 0.5
    _calibrated_heuristic_base: float | None = None

    CONFIDENCE_AUTO_APPROVE: float = field(
        default_factory=lambda: _env("INVOICEX_CONFIDENCE_AUTO_APPROVE", 0.85)
    )

    # Candidate generation
    TRACE_CANDIDATES: bool = field(
        default_factory=lambda: _env("INVOICEX_TRACE_CANDIDATES", False)
    )
    EARLY_PAGE_BOOST: float = field(
        default_factory=lambda: _env("INVOICEX_EARLY_PAGE_BOOST", 0.5)
    )
    EARLY_PAGE_MAX_IDX: int = field(
        default_factory=lambda: _env("INVOICEX_EARLY_PAGE_MAX_IDX", 2)
    )

    # Candidate pruning
    PRUNING_THRESHOLD: float = field(
        default_factory=lambda: _env("INVOICEX_PRUNING_THRESHOLD", 0.05)
    )
    PRUNING_MAX_CANDIDATES: int = field(
        default_factory=lambda: _env("INVOICEX_PRUNING_MAX_CANDIDATES", 200)
    )
    PRUNING_MIN_TRIGGER: int = field(
        default_factory=lambda: _env("INVOICEX_PRUNING_MIN_TRIGGER", 50)
    )

    # Bucket names
    BUCKET_DATE_LIKE: str = "date_like"
    BUCKET_AMOUNT_LIKE: str = "amount_like"
    BUCKET_ID_LIKE: str = "id_like"
    BUCKET_NAME_LIKE: str = "name_like"
    BUCKET_KEYWORD_PROXIMAL: str = "keyword_proximal"
    BUCKET_RANDOM_NEGATIVE: str = "random_negative"

    # Month constants (from stdlib calendar + "sept")
    MONTH_ABBREVS: frozenset[str] = _MONTH_ABBREVS
    MONTH_NAMES_ALL: frozenset[str] = _MONTH_NAMES_ALL

    # XGBoost parameters (classifier)
    XGBOOST_PARAMS: dict[str, Any] = field(
        default_factory=lambda: {**_XGB_BASE, "objective": "binary:logistic"}
    )
    MIN_POSITIVE_EXAMPLES: int = 2

    # ML ranker integration
    ML_SCORE_WEIGHT: float = field(
        default_factory=lambda: _env("INVOICEX_ML_SCORE_WEIGHT", 0.7)
    )
    USE_RANKER_MODEL: bool = field(
        default_factory=lambda: _env("INVOICEX_USE_RANKER", False)
    )

    # MLflow
    USE_MLFLOW: bool = field(default_factory=lambda: _env("INVOICEX_USE_MLFLOW", False))
    # Local dev default (port 5050 avoids macOS AirPlay conflict on 5000).
    # Docker overrides via INVOICEX_MLFLOW_TRACKING_URI=http://mlflow:5000
    MLFLOW_TRACKING_URI: str = field(
        default_factory=lambda: _env(
            "INVOICEX_MLFLOW_TRACKING_URI", "http://localhost:5050"
        )
    )
    MLFLOW_EXPERIMENT_NAME: str = field(
        default_factory=lambda: _env("INVOICEX_MLFLOW_EXPERIMENT_NAME", "invoicex")
    )
    MLFLOW_MODEL_PREFIX: str = "invoicex-ranker"
    MLFLOW_ARTIFACT_ROOT: str = field(
        default_factory=lambda: _env("MLFLOW_ARTIFACT_ROOT", "/mlflow-artifacts")
    )

    # Backend switches (local/Azure factory pattern)
    CONNECTOR_MODE: str = field(
        default_factory=lambda: _env("INVOICEX_CONNECTOR_MODE", "local")
    )
    STORAGE_BACKEND: str = field(
        default_factory=lambda: _env_with_connector("INVOICEX_STORAGE_BACKEND", "local")
    )
    DOCUMENT_SOURCE: str = field(
        default_factory=lambda: _env_with_connector("INVOICEX_DOCUMENT_SOURCE", "local")
    )
    OUTPUT_BACKEND: str = field(
        default_factory=lambda: _env_with_connector("INVOICEX_OUTPUT_BACKEND", "local")
    )

    # Azure Blob Storage
    AZURE_STORAGE_ACCOUNT_NAME: str = field(
        default_factory=lambda: _env("AZURE_STORAGE_ACCOUNT_NAME", "")
    )
    AZURE_STORAGE_CONTAINER_NAME: str = field(
        default_factory=lambda: _env("AZURE_STORAGE_CONTAINER_NAME", "invoicex")
    )
    AZURE_STORAGE_CONNECTION_STRING: str = field(
        default_factory=lambda: _env("AZURE_STORAGE_CONNECTION_STRING", "")
    )

    # SharePoint (Graph API)
    SHAREPOINT_SITE_ID: str = field(
        default_factory=lambda: _env("SHAREPOINT_SITE_ID", "")
    )
    SHAREPOINT_DRIVE_ID: str = field(
        default_factory=lambda: _env("SHAREPOINT_DRIVE_ID", "")
    )
    SHAREPOINT_FOLDER_PATH: str = field(
        default_factory=lambda: _env("SHAREPOINT_FOLDER_PATH", "Invoices/Inbox")
    )

    # Dataverse
    DATAVERSE_ENVIRONMENT_URL: str = field(
        default_factory=lambda: _env("DATAVERSE_ENVIRONMENT_URL", "")
    )
    DATAVERSE_CLIENT_ID: str = field(
        default_factory=lambda: _env("DATAVERSE_CLIENT_ID", "")
    )
    DATAVERSE_STAGING_TABLE: str = field(
        default_factory=lambda: _env("DATAVERSE_STAGING_TABLE", "new_tem")
    )
    DATAVERSE_PRODUCTION_TABLE: str = field(
        default_factory=lambda: _env("DATAVERSE_PRODUCTION_TABLE", "tem")
    )
    DATAVERSE_WRITE_ENABLED: bool = field(
        default_factory=lambda: _env("INVOICEX_DATAVERSE_WRITE_ENABLED", False)
    )

    # Orchestrator
    ORCHESTRATOR_MAX_RETRIES: int = field(
        default_factory=lambda: _env("INVOICEX_ORCHESTRATOR_MAX_RETRIES", 3)
    )
    ORCHESTRATOR_RETRY_BASE_SECONDS: float = field(
        default_factory=lambda: _env("INVOICEX_ORCHESTRATOR_RETRY_BASE_SECONDS", 1.0)
    )
    ORCHESTRATOR_RETRY_MULTIPLIER: float = field(
        default_factory=lambda: _env("INVOICEX_ORCHESTRATOR_RETRY_MULTIPLIER", 4.0)
    )
    ORCHESTRATOR_WATCH_INTERVAL: float = field(
        default_factory=lambda: _env("INVOICEX_ORCHESTRATOR_WATCH_INTERVAL", 5.0)
    )
    ORCHESTRATOR_SEED_FOLDER: str = field(
        default_factory=lambda: _env("INVOICEX_ORCHESTRATOR_SEED_FOLDER", "seed_pdfs")
    )

    # Currency
    CURRENCY_SYMBOLS: frozenset[str] = frozenset({"$", "€", "£", "¥", "₹", "₽"})
    CURRENCY_CODES: frozenset[str] = frozenset(
        {"USD", "EUR", "GBP", "JPY", "INR", "RUB", "CAD", "AUD", "CHF", "CNY", "MXN"}
    )
    CURRENCY_SYMBOL_MAP: dict[str, str] = field(
        default_factory=lambda: {
            "$": "USD",
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            "₹": "INR",
            "₽": "RUB",
        }
    )

    # Typed anchors (semantic keyword sets for directional features)
    TOTAL_ANCHORS: frozenset[str] = frozenset(
        {
            "total",
            "total due",
            "total amount",
            "amount due",
            "balance due",
            "grand total",
            "net total",
            "subtotal",
            "sub-total",
            "sub total",
            "amount",
            "balance",
            "amount payable",
            "invoice total",
            "order total",
            "sum",
        }
    )
    TAX_ANCHORS: frozenset[str] = frozenset(
        {
            "tax",
            "vat",
            "gst",
            "hst",
            "pst",
            "sales tax",
            "tax amount",
            "vat amount",
            "tax total",
            "taxes",
        }
    )
    DATE_ANCHORS: frozenset[str] = frozenset(
        {
            "date",
            "invoice date",
            "issue date",
            "issued",
            "due date",
            "payment due",
            "date due",
            "due",
            "billing date",
            "statement date",
            "order date",
            "ship date",
            "delivery date",
        }
    )
    ID_ANCHORS: frozenset[str] = frozenset(
        {
            "invoice",
            "inv",
            "invoice#",
            "invoice number",
            "invoice no",
            "invoice id",
            "document number",
            "doc no",
            "reference",
            "account",
            "account#",
            "account number",
            "account no",
            "acct",
            "customer id",
            "customer number",
            "customer#",
            "po",
            "po#",
            "purchase order",
            "order number",
            "order#",
            "statement",
            "document",
            "bill",
            "billing",
        }
    )
    NAME_ANCHORS: frozenset[str] = frozenset(
        {
            "bill to",
            "billed to",
            "customer",
            "client",
            "ship to",
            "deliver to",
            "sold to",
            "vendor",
            "supplier",
            "from",
            "remit to",
            "company",
            "name",
            "attention",
            "attn",
        }
    )
    INVOICE_KEYWORDS: frozenset[str] = frozenset(
        TOTAL_ANCHORS | TAX_ANCHORS | DATE_ANCHORS | ID_ANCHORS | NAME_ANCHORS
    )

    # Spatial feature thresholds
    COLUMN_ALIGN_THRESHOLD: float = 0.08
    ROW_ALIGN_THRESHOLD: float = 0.03
    ANCHOR_PROXIMITY_MAX_DISTANCE: float = 0.75
    ANCHOR_TYPES: tuple[str, ...] = ("total", "tax", "date", "id", "name")

    # ML feature column building blocks
    DIRECTIONAL_SUFFIXES: tuple[str, ...] = (
        "dx_to_",
        "dy_to_",
        "dist_to_",
        "aligned_x_",
        "aligned_y_",
        "reading_order_",
        "below_",
    )
    RELATIVE_POSITION_FEATURES: tuple[str, ...] = (
        "y_from_bottom",
        "in_top_half",
        "in_bottom_quarter",
        "in_top_quarter",
        "in_right_third",
        "in_amount_region",
    )

    # Application
    DATA_DIR: str = field(default_factory=lambda: _env("INVOICEX_DATA_DIR", "data"))
    API_KEY: str = field(default_factory=lambda: _env("INVOICEX_API_KEY", ""))
    ENVIRONMENT: str = field(default_factory=lambda: _env("ENVIRONMENT", "production"))
    CORS_ORIGINS: str = field(
        default_factory=lambda: _env(
            "INVOICEX_CORS_ORIGINS",
            "http://localhost:3000,http://localhost",
        )
    )
    PUBLIC_URL: str = field(
        default_factory=lambda: _env("INVOICEX_PUBLIC_URL", "http://localhost")
    )
    LOG_LEVEL: str = field(default_factory=lambda: _env("INVOICEX_LOG_LEVEL", "INFO"))

    # XGBoost ranker (learning-to-rank, separate objective from classifier)
    XGBOOST_RANKER_PARAMS: dict[str, Any] = field(
        default_factory=lambda: {**_XGB_BASE, "objective": "rank:pairwise"}
    )

    MODEL_ID: str = field(default_factory=lambda: _env("MODEL_ID", "unscored-baseline"))

    # --- Methods ---

    def get_data_path(self) -> Path:
        """Get the data directory as a Path object."""
        return Path(self.DATA_DIR)

    def get_cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list of origin URLs."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def is_dev(self) -> bool:
        """Check if running in development mode."""
        return self.ENVIRONMENT == "development"

    def get_feature_columns(self) -> list[str]:
        """Ordered list of 64 ML feature columns (order-sensitive for XGBoost).

        CRITICAL: train.py and decoder.py MUST both use this to stay aligned.

        Column groups:
          Geometric (5) + Text (4) + Page (1) + Bucket one-hot (7)
          + Directional (35 = 7 suffixes x 5 anchor types)
          + Relative position (6)
          + Semantic features (6):
              in_header_region, in_footer_region, in_body_region (page region one-hot)
              occurrence_rank (reading-order rank among same-text candidates)
              is_largest_amount_in_doc (binary: max numeric value among amounts)
              aligned_x (column-aligned with any keyword anchor above/below)
        """
        features = [
            # Geometric (5)
            "center_x",
            "center_y",
            "width",
            "height",
            "area",
            # Text (4)
            "char_count",
            "word_count",
            "digit_count",
            "alpha_count",
            # Page (1)
            "page_idx",
            # Bucket one-hot (7)
            "bucket_amount_like",
            "bucket_date_like",
            "bucket_id_like",
            "bucket_name_like",
            "bucket_keyword_proximal",
            "bucket_random_negative",
            "bucket_other",
        ]
        # Directional (35 = 7 suffixes x 5 anchor types)
        for anchor_type in self.ANCHOR_TYPES:
            for suffix in self.DIRECTIONAL_SUFFIXES:
                features.append(f"{suffix}{anchor_type}")
        # Relative position (6)
        features.extend(self.RELATIVE_POSITION_FEATURES)
        # Semantic features (7)
        features.extend(
            [
                "in_header_region",
                "in_footer_region",
                "in_body_region",
                "occurrence_rank",
                "is_largest_amount_in_doc",
                "aligned_x",
            ]
        )
        return features

    def get_field_confidence_threshold(self, field_name: str) -> float:
        """Confidence threshold for a field (currently global)."""
        return self.CONFIDENCE_AUTO_APPROVE

    def get_anchor_keywords_by_type(self) -> dict[str, frozenset[str]]:
        """Anchor type -> keyword set mapping for find_typed_anchors."""
        return {
            "total": self.TOTAL_ANCHORS,
            "tax": self.TAX_ANCHORS,
            "date": self.DATE_ANCHORS,
            "id": self.ID_ANCHORS,
            "name": self.NAME_ANCHORS,
        }

    def validate_backend_config(self) -> list[str]:
        """Validate backend configuration, return list of error messages."""
        errors: list[str] = []

        if self.STORAGE_BACKEND not in ("local", "blob"):
            errors.append(
                f"Invalid STORAGE_BACKEND: '{self.STORAGE_BACKEND}'. "
                "Must be 'local' or 'blob'"
            )
        elif self.STORAGE_BACKEND == "blob":
            if (
                not self.AZURE_STORAGE_ACCOUNT_NAME
                and not self.AZURE_STORAGE_CONNECTION_STRING
            ):
                errors.append(
                    "STORAGE_BACKEND=blob requires AZURE_STORAGE_ACCOUNT_NAME "
                    "or AZURE_STORAGE_CONNECTION_STRING"
                )

        if self.DOCUMENT_SOURCE not in ("local", "sharepoint"):
            errors.append(
                f"Invalid DOCUMENT_SOURCE: '{self.DOCUMENT_SOURCE}'. "
                "Must be 'local' or 'sharepoint'"
            )
        elif self.DOCUMENT_SOURCE == "sharepoint":
            if not self.SHAREPOINT_SITE_ID:
                errors.append("DOCUMENT_SOURCE=sharepoint requires SHAREPOINT_SITE_ID")
            if not os.environ.get("AZURE_TENANT_ID"):
                errors.append("DOCUMENT_SOURCE=sharepoint requires AZURE_TENANT_ID")

        if self.OUTPUT_BACKEND not in ("local", "dataverse"):
            errors.append(
                f"Invalid OUTPUT_BACKEND: '{self.OUTPUT_BACKEND}'. "
                "Must be 'local' or 'dataverse'"
            )
        elif self.OUTPUT_BACKEND == "dataverse":
            if not self.DATAVERSE_ENVIRONMENT_URL:
                errors.append(
                    "OUTPUT_BACKEND=dataverse requires DATAVERSE_ENVIRONMENT_URL"
                )
            if not self.DATAVERSE_CLIENT_ID:
                errors.append("OUTPUT_BACKEND=dataverse requires DATAVERSE_CLIENT_ID")
            if not os.environ.get("AZURE_TENANT_ID"):
                errors.append("OUTPUT_BACKEND=dataverse requires AZURE_TENANT_ID")

        return errors

    def validate(self) -> None:
        """Validate configuration. Raises ValueError on failure."""
        errors: list[str] = []

        if not (0.0 <= self.CONFIDENCE_AUTO_APPROVE <= 1.0):
            errors.append(
                f"CONFIDENCE_AUTO_APPROVE must be in [0, 1], "
                f"got {self.CONFIDENCE_AUTO_APPROVE}"
            )
        if not (0.0 <= self.ML_SCORE_WEIGHT <= 1.0):
            errors.append(
                f"ML_SCORE_WEIGHT must be in [0, 1], got {self.ML_SCORE_WEIGHT}"
            )

        required_xgb_keys = {
            "objective",
            "max_depth",
            "learning_rate",
            "n_estimators",
            "random_state",
            "seed",
        }
        missing = required_xgb_keys - set(self.XGBOOST_PARAMS.keys())
        if missing:
            errors.append(f"XGBOOST_PARAMS missing required keys: {missing}")

        if self.ENVIRONMENT == "production" and not os.environ.get(
            "INVOICEX_CORS_ORIGINS"
        ):
            errors.append(
                "INVOICEX_CORS_ORIGINS must be set in production. "
                "Set ENVIRONMENT=development for local testing."
            )

        if self.ENVIRONMENT == "production" and any(
            placeholder in self.PUBLIC_URL
            for placeholder in ("localhost", "YOUR_DOMAIN", "REPLACE_ME")
        ):
            errors.append(
                f"PUBLIC_URL must be set to a real domain in production, "
                f"got {self.PUBLIC_URL!r}. "
                f"Set ENVIRONMENT=development for local testing."
            )

        errors.extend(self.validate_backend_config())

        if errors:
            raise ValueError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def is_azure_mode(self) -> bool:
        """True if any backend is configured for Azure."""
        return (
            self.STORAGE_BACKEND == "blob"
            or self.DOCUMENT_SOURCE == "sharepoint"
            or self.OUTPUT_BACKEND == "dataverse"
        )

    def needs_review(self, field_confidences: dict[str, float]) -> bool:
        """True if any field confidence is below the auto-approve threshold."""
        return any(c < self.CONFIDENCE_AUTO_APPROVE for c in field_confidences.values())

    # Quality gate: minimum mean NDCG@1 for ranker to be used in production.
    # Rationale: 0.5 = better than random baseline for NDCG@1 (correct
    # candidate ranked first at least 50% of the time across fields).
    # Overridable via INVOICEX_QUALITY_GATE_THRESHOLD for tuning.
    QUALITY_GATE_NDCG_THRESHOLD: float = field(
        default_factory=lambda: _env("INVOICEX_QUALITY_GATE_THRESHOLD", 0.5)
    )

    # Bootstrap mode: relaxed criteria when labeled docs < threshold.
    # Uses 2-fold CV instead of LOOCV, lower quality gate, lower ML weight.
    BOOTSTRAP_DOC_THRESHOLD: int = field(
        default_factory=lambda: _env("INVOICEX_BOOTSTRAP_DOC_THRESHOLD", 10)
    )
    BOOTSTRAP_QUALITY_GATE_THRESHOLD: float = field(
        default_factory=lambda: _env("INVOICEX_BOOTSTRAP_QUALITY_GATE_THRESHOLD", 0.3)
    )
    BOOTSTRAP_ML_SCORE_WEIGHT: float = field(
        default_factory=lambda: _env("INVOICEX_BOOTSTRAP_ML_SCORE_WEIGHT", 0.3)
    )

    @property
    def CONFIDENCE_HEURISTIC_SCALE(self) -> float:
        """Heuristic confidence scaling factor, derived from existing config.

        Controls the steepness of the linear cost-to-confidence mapping:
            confidence = base - cost * scale

        Derived as CONFIDENCE_HEURISTIC_BASE / DECODER_BASE_COST so the
        cost range [-DECODER_BASE_COST, +DECODER_BASE_COST] maps across
        [0, 1] confidence. With base=0.5, base_cost=2.0 → scale=0.25.
        """
        return self.CONFIDENCE_HEURISTIC_BASE / self.DECODER_BASE_COST

    def compute_confidence_from_cost(
        self, cost: float, has_ml_model: bool = False
    ) -> float:
        """Convert assignment cost to confidence score in [0, 1].

        Heuristic path: ``confidence = base - cost * scale``.  Different
        costs MUST produce different confidences (linear, interpretable).
        The base anchors cost=0 at 0.5; the scale controls spread.

        ML path: maps cost to [0, 1] via ``1 - cost`` (cost already in
        [0, 1] from sigmoid in ``compute_ranker_cost``).
        """
        if has_ml_model:
            confidence = 1.0 - cost
        else:
            base = self._calibrated_heuristic_base or self.CONFIDENCE_HEURISTIC_BASE
            confidence = base - cost * self.CONFIDENCE_HEURISTIC_SCALE
        return max(self.CONFIDENCE_FLOOR, min(self.CONFIDENCE_CEILING, confidence))

    def calibrate_heuristic_base(self, empirical_accuracy: float) -> None:
        """Set heuristic confidence base from empirical accuracy (clamped [0.5, 0.95])."""
        object.__setattr__(
            self,
            "_calibrated_heuristic_base",
            max(0.5, min(0.95, empirical_accuracy)),
        )


Config = PipelineConfig()
