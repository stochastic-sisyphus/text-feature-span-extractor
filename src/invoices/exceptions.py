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
    │   ├── ContractValidationError
    │   └── ContractMismatchError
    ├── ModelError
    │   ├── ModelNotFoundError
    │   ├── ModelLoadError
    │   └── ModelNotReadyError
    ├── IntegrationError
    │   ├── StorageError
    │   └── OrchestrationError
    └── RecoverableError

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


class FieldAlreadyExistsError(ConfigurationError):
    """Schema field with the given name already exists."""

    def __init__(self, field_name: str):
        super().__init__(f"Field already exists: {field_name}")
        self.field_name = field_name


class FieldNotFoundError(ConfigurationError):
    """Schema field with the given name does not exist."""

    def __init__(self, field_name: str):
        super().__init__(f"Field not found: {field_name}")
        self.field_name = field_name


class FieldNameConflictError(ConfigurationError):
    """Rename would collide with an existing field name."""

    def __init__(self, existing_name: str, new_name: str):
        super().__init__(f"Cannot rename to '{new_name}': field already exists")
        self.existing_name = existing_name
        self.new_name = new_name


class UnknownComputedSourceError(ConfigurationError):
    """computed_from references a field name not in the schema."""

    def __init__(self, field_name: str, unknown_sources: list[str]):
        super().__init__(
            f"'{field_name}': computed_from references unknown fields: "
            + ", ".join(sorted(unknown_sources))
        )
        self.field_name = field_name
        self.unknown_sources = unknown_sources


class FieldCoherenceError(ConfigurationError):
    """FieldDef coherence check failed — invalid combination of properties."""

    def __init__(self, field_name: str, detail: str):
        super().__init__(f"'{field_name}': {detail}")
        self.field_name = field_name
        self.detail = detail


class TypeCascadeError(ConfigurationError):
    """type_change_cascade='coerce' failed — existing corrections cannot be cast.

    Raised when at least one ``correct_value`` in the corrections table fails
    to parse under the requested new ``base_type``.  Carries the list of
    offending values so the caller can surface them in a 422 response.
    """

    def __init__(self, field_name: str, new_base_type: str, failing_values: list[str]):
        sample = failing_values[:5]
        extra = (
            f" (+ {len(failing_values) - 5} more)" if len(failing_values) > 5 else ""
        )
        super().__init__(
            f"'{field_name}': {len(failing_values)} correction value(s) cannot be "
            f"cast to '{new_base_type}': {sample}{extra}"
        )
        self.field_name = field_name
        self.new_base_type = new_base_type
        self.failing_values = failing_values


class CorruptSchemaError(ConfigurationError):
    """Contract schema row loaded from Postgres fails envelope validation.

    Raised by SchemaRepo.get_schema() when the JSONB row exists but does not
    parse as a valid ContractEnvelope — indicating the database contains a
    schema that was written without validation or was corrupted post-write.
    Surfaces as a 500-class error at the API layer.
    """

    def __init__(self, detail: str):
        super().__init__(f"Active contract schema is corrupt: {detail}")
        self.detail = detail


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


class ContractMismatchError(ValidationError):
    """Raised when a feature contract boundary detects schema drift.

    Signals that the runtime FieldSpec, Pandera schema, or JSONB store disagree
    on a field's shape — meaning the contract changed without a coordinated
    migration.  Carries ``field``, ``expected``, ``actual``, and ``source`` so
    callers can log actionable detail without further introspection.
    """

    def __init__(self, field: str, expected: str, actual: str, source: str):
        message = (
            f"Contract mismatch for {field} in {source}: "
            f"expected {expected!r}, got {actual!r}"
        )
        super().__init__(
            message, field=field, expected=expected, actual=actual, source=source
        )
        self.field = field
        self.expected = expected
        self.actual = actual
        self.source = source


class MissingTokensError(ValidationError):
    """Raised when a 'processed' document has no tokens at the hydration boundary.

    Pipeline invariant: every document marked 'processed' in the ledger MUST have
    at least one token page in its stored payload.  An empty token list signals a
    silent pipeline failure (tokenizer crash, empty extraction, write-before-finish).
    """

    def __init__(self, sha256: str, doc_id: str | None = None) -> None:
        super().__init__(
            f"Processed document has no tokens: {sha256[:16]}",
            sha256=sha256,
            doc_id=doc_id,
        )
        self.sha256 = sha256
        self.doc_id = doc_id


class CorpusValidationError(ValidationError):
    """Raised when the retrain corpus drop-rate exceeds the configured threshold.

    A high drop rate means the training corpus is too degraded to retrain
    safely — this is a hard refusal, not a recoverable condition.
    """

    def __init__(
        self,
        drop_rate: float,
        n_bad: int,
        n_total: int,
        threshold: float,
    ):
        message = (
            f"Corpus validation failed: {n_bad}/{n_total} docs dropped "
            f"({drop_rate:.1%}) exceeds threshold {threshold:.1%}"
        )
        super().__init__(
            message,
            drop_rate=drop_rate,
            n_bad=n_bad,
            n_total=n_total,
            threshold=threshold,
        )
        self.drop_rate = drop_rate
        self.n_bad = n_bad
        self.n_total = n_total
        self.threshold = threshold


class NormalizationError(ValidationError):
    """Value normalization failed."""

    def __init__(self, field: str, raw_text: str, normalizer: str, reason: str):
        message = f"Normalization failed for {field} ({normalizer}): {reason}"
        super().__init__(
            message,
            field=field,
            raw_text=raw_text[:100],  # Truncate long text
            normalizer=normalizer,
            reason=reason,
        )
        self.field = field
        self.raw_text = raw_text
        self.normalizer = normalizer
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


class ModelNotReadyError(ModelError):
    """No trained model is available yet — heuristic fallback is the contract.

    Raised by callers that require a bundle but have not yet trained one.
    Callers that tolerate absent models should use ``ModelCache.peek()``
    and branch on ``None`` rather than catching this exception.
    """

    def __init__(self, detail: str = "no trained model available"):
        super().__init__(detail)
        self.detail = detail


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
# Recoverable Errors
# =============================================================================


class RecoverableError(InvoicexError):
    """Base class for failures where a fallback path is the correct response.

    Callers that catch this base explicitly opt into fallback behavior — e.g.,
    substituting a weak prior or retrying with backoff — while letting all other
    exceptions propagate so real bugs are never silently swallowed.  Subclass
    for specific recoverable conditions; do not raise this base directly.
    """

    pass


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
        RecoverableError,
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
        ValidationError, ModelError, IntegrationError, RecoverableError

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
