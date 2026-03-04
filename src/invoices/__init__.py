"""Text-layer feature-based span extractor for machine-generated PDF invoices.

This package provides a deterministic, text-layer-only invoice data extraction
pipeline that transforms PDFs into structured JSON using geometric features,
heuristic bucketing, and Hungarian assignment.

Key Modules:
    config: Centralized configuration management
    ingest: Content-addressed PDF storage
    tokenize: PDF text extraction with bounding boxes
    candidates: Candidate span generation with bucketing
    decoder: Hungarian assignment for field extraction
    emit: Contract JSON emission with routing
    normalize: Field value normalization
    train: XGBoost model training
    logging: Structured logging utilities
    exceptions: Typed exception hierarchy

Example Usage:
    from invoices.config import Config
    from invoices import ingest, tokenize, candidates, emit

    # Process a document
    sha256 = ingest.ingest_file("invoice.pdf")
    tokenize.tokenize_document(sha256)
    candidates.generate_candidates(sha256)
    result = emit.emit_document(sha256)
"""

try:
    from importlib.metadata import version

    __version__ = version("text-feature-span-extract")
except Exception:
    __version__ = "0.2.0"  # fallback for editable installs / missing metadata

# Core configuration
from .config import Config

# Exceptions
from .exceptions import (
    ConfigurationError,
    DecodingError,
    DocumentNotFoundError,
    EmissionError,
    IntegrationError,
    InvoicexError,
    ModelError,
    PipelineError,
    ValidationError,
)

# Feature preparation (shared training/inference alignment)
from .feature_prep import prepare_candidate_features, prepare_features_dataframe

# Logging utilities
from .logging import (
    configure_logging,
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
)

# Ranking
from .ranker import InvoiceFieldRanker

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Config",
    # Exceptions
    "InvoicexError",
    "ConfigurationError",
    "PipelineError",
    "ValidationError",
    "ModelError",
    "IntegrationError",
    "DocumentNotFoundError",
    "DecodingError",
    "EmissionError",
    # Logging
    "get_logger",
    "configure_logging",
    "log_info",
    "log_error",
    "log_warning",
    "log_debug",
    # Feature preparation
    "prepare_candidate_features",
    "prepare_features_dataframe",
    # Ranking
    "InvoiceFieldRanker",
]
