"""Custom exceptions for invoice extraction pipeline.

Provides a hierarchy of typed exceptions for better error handling
and observability throughout the pipeline.

Exception Hierarchy:
    InvoicexError (base)
    ├── ConfigurationError
    ├── PipelineError
    │   ├── IngestError
    │   ├── TokenizationError
    │   ├── CandidateGenerationError
    │   ├── DecodingError
    │   └── EmissionError
    ├── ValidationError
    │   ├── SchemaValidationError
    │   └── ContractValidationError
    ├── ModelError
    │   ├── ModelNotFoundError
    │   └── ModelLoadError
    └── IntegrationError
        ├── StorageError
        └── OrchestrationError

Usage:
    from invoices.exceptions import DecodingError, DocumentNotFoundError

    try:
        result = decode_document(sha256)
    except DocumentNotFoundError as e:
        logger.error("document_not_found", sha256=e.sha256)
    except DecodingError as e:
        logger.error("decoding_failed", reason=e.reason, doc_id=e.doc_id)
"""

from typing import Any


class InvoicexError(Exception):
    """Base exception for all invoice extraction errors.

    All custom exceptions inherit from this base class to allow
    catching any pipeline-related error with a single except clause.
    """

    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            **self.context,
        }


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(InvoicexError):
    """Error in pipeline configuration."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration value is missing."""

    def __init__(self, key: str, description: str | None = None):
        message = f"Missing required configuration: {key}"
        if description:
            message += f" ({description})"
        super().__init__(message, key=key, description=description)
        self.key = key


class InvalidConfigurationError(ConfigurationError):
    """Configuration value is invalid."""

    def __init__(self, key: str, value: Any, reason: str):
        message = f"Invalid configuration for {key}: {reason}"
        super().__init__(message, key=key, value=value, reason=reason)
        self.key = key
        self.value = value
        self.reason = reason


# =============================================================================
# Pipeline Stage Errors
# =============================================================================


class PipelineError(InvoicexError):
    """Base class for pipeline stage errors."""

    stage: str = "unknown"


class IngestError(PipelineError):
    """Error during document ingestion."""

    stage = "ingest"


class DocumentNotFoundError(IngestError):
    """Requested document was not found."""

    def __init__(self, sha256: str | None = None, doc_id: str | None = None):
        identifier = sha256 or doc_id or "unknown"
        message = f"Document not found: {identifier}"
        super().__init__(message, sha256=sha256, doc_id=doc_id)
        self.sha256 = sha256
        self.doc_id = doc_id


class DuplicateDocumentError(IngestError):
    """Document with same SHA256 already exists."""

    def __init__(self, sha256: str, existing_doc_id: str):
        message = f"Document already exists: {sha256[:16]}..."
        super().__init__(message, sha256=sha256, existing_doc_id=existing_doc_id)
        self.sha256 = sha256
        self.existing_doc_id = existing_doc_id


class InvalidPDFError(IngestError):
    """PDF file is invalid or corrupted."""

    def __init__(self, path: str, reason: str):
        message = f"Invalid PDF: {reason}"
        super().__init__(message, path=path, reason=reason)
        self.path = path
        self.reason = reason


class TokenizationError(PipelineError):
    """Error during token extraction."""

    stage = "tokenize"

    def __init__(self, sha256: str, reason: str, page: int | None = None):
        message = f"Tokenization failed for {sha256[:16]}...: {reason}"
        super().__init__(message, sha256=sha256, reason=reason, page=page)
        self.sha256 = sha256
        self.reason = reason
        self.page = page


class CandidateGenerationError(PipelineError):
    """Error during candidate span generation."""

    stage = "candidates"

    def __init__(self, sha256: str, reason: str):
        message = f"Candidate generation failed for {sha256[:16]}...: {reason}"
        super().__init__(message, sha256=sha256, reason=reason)
        self.sha256 = sha256
        self.reason = reason


class DecodingError(PipelineError):
    """Error during field assignment/decoding."""

    stage = "decode"

    def __init__(
        self,
        sha256: str,
        reason: str,
        doc_id: str | None = None,
        field: str | None = None,
    ):
        message = f"Decoding failed for {sha256[:16]}...: {reason}"
        super().__init__(
            message, sha256=sha256, reason=reason, doc_id=doc_id, field=field
        )
        self.sha256 = sha256
        self.reason = reason
        self.doc_id = doc_id
        self.field = field


class EmissionError(PipelineError):
    """Error during contract JSON emission."""

    stage = "emit"

    def __init__(self, sha256: str, reason: str, doc_id: str | None = None):
        message = f"Emission failed for {sha256[:16]}...: {reason}"
        super().__init__(message, sha256=sha256, reason=reason, doc_id=doc_id)
        self.sha256 = sha256
        self.reason = reason
        self.doc_id = doc_id


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(InvoicexError):
    """Base class for validation errors."""

    pass


class SchemaValidationError(ValidationError):
    """Contract schema validation failed."""

    def __init__(self, field: str, value: Any, constraint: str, reason: str):
        message = f"Schema validation failed for {field}: {reason}"
        super().__init__(
            message, field=field, value=value, constraint=constraint, reason=reason
        )
        self.field = field
        self.value = value
        self.constraint = constraint
        self.reason = reason


class ContractValidationError(ValidationError):
    """Output contract JSON validation failed."""

    def __init__(self, doc_id: str, errors: list[str]):
        message = f"Contract validation failed for {doc_id}: {len(errors)} errors"
        super().__init__(message, doc_id=doc_id, errors=errors)
        self.doc_id = doc_id
        self.errors = errors


class NormalizationError(ValidationError):
    """Value normalization failed."""

    def __init__(self, field: str, raw_text: str, field_type: str, reason: str):
        message = f"Normalization failed for {field} ({field_type}): {reason}"
        super().__init__(
            message,
            field=field,
            raw_text=raw_text[:100],  # Truncate long text
            field_type=field_type,
            reason=reason,
        )
        self.field = field
        self.raw_text = raw_text
        self.field_type = field_type
        self.reason = reason


# =============================================================================
# Model Errors
# =============================================================================


class ModelError(InvoicexError):
    """Base class for ML model errors."""

    pass


class ModelNotFoundError(ModelError):
    """Trained model file not found."""

    def __init__(self, model_id: str, path: str | None = None):
        message = f"Model not found: {model_id}"
        super().__init__(message, model_id=model_id, path=path)
        self.model_id = model_id
        self.path = path


class ModelLoadError(ModelError):
    """Failed to load trained model."""

    def __init__(self, model_id: str, reason: str):
        message = f"Failed to load model {model_id}: {reason}"
        super().__init__(message, model_id=model_id, reason=reason)
        self.model_id = model_id
        self.reason = reason


class TrainingError(ModelError):
    """Error during model training."""

    def __init__(self, field: str, reason: str, sample_count: int | None = None):
        message = f"Training failed for field {field}: {reason}"
        super().__init__(message, field=field, reason=reason, sample_count=sample_count)
        self.field = field
        self.reason = reason
        self.sample_count = sample_count


# =============================================================================
# Integration Errors
# =============================================================================


class IntegrationError(InvoicexError):
    """Base class for external integration errors."""

    pass


class StorageError(IntegrationError):
    """Error with file/blob storage."""

    def __init__(self, operation: str, path: str, reason: str):
        message = f"Storage {operation} failed for {path}: {reason}"
        super().__init__(message, operation=operation, path=path, reason=reason)
        self.operation = operation
        self.path = path
        self.reason = reason


class OrchestrationError(IntegrationError):
    """Error during document orchestration (discovery, routing, retries)."""

    def __init__(self, sha256: str, stage: str, reason: str):
        message = f"Orchestration failed at {stage} for {sha256[:16]}...: {reason}"
        super().__init__(message, sha256=sha256, stage=stage, reason=reason)
        self.sha256 = sha256
        self.stage = stage
        self.reason = reason


# =============================================================================
# Utility Functions
# =============================================================================


# Classes that accept the (message: str, **context) signature
# Use these with wrap_exception(); other subclasses have specialized signatures
_WRAPPABLE_CLASSES: frozenset[type[InvoicexError]] = frozenset(
    {
        InvoicexError,
        ConfigurationError,
        PipelineError,
        IngestError,
        ValidationError,
        ModelError,
        IntegrationError,
    }
)


def wrap_exception(
    exc: Exception,
    wrapper_class: type[InvoicexError] | None = None,
    **context: Any,
) -> InvoicexError:
    """Wrap a generic exception in a typed InvoicexError.

    This function is intended for wrapping external exceptions at system
    boundaries. It only works with base exception classes that accept
    the standard (message: str, **context) signature.

    For specialized exception types with custom signatures (e.g.,
    DocumentNotFoundError, TokenizationError), use explicit exception
    chaining instead:
        raise TokenizationError(sha256, reason) from exc

    Compatible wrapper classes:
        InvoicexError, ConfigurationError, PipelineError, IngestError,
        ValidationError, ModelError, IntegrationError

    Args:
        exc: Original exception to wrap
        wrapper_class: Exception class to wrap with (default: InvoicexError).
            Must be a base class with (message, **context) signature.
        **context: Additional context fields for the wrapper

    Returns:
        Wrapped exception with original as __cause__

    Raises:
        ValueError: If wrapper_class has incompatible constructor signature

    Example:
        try:
            external_api.fetch_data()
        except ExternalError as e:
            raise wrap_exception(e, IntegrationError, operation="fetch")
    """
    if wrapper_class is None:
        wrapper_class = InvoicexError

    # Validate that the wrapper class has compatible signature
    if wrapper_class not in _WRAPPABLE_CLASSES:
        raise ValueError(
            f"{wrapper_class.__name__} has a specialized constructor. "
            f"Use 'raise {wrapper_class.__name__}(...) from exc' instead, "
            f"or use one of: {', '.join(c.__name__ for c in _WRAPPABLE_CLASSES)}"
        )

    wrapped = wrapper_class(str(exc), **context)
    wrapped.__cause__ = exc
    return wrapped
