"""Structured logging module for invoice extraction pipeline.

Provides JSON-formatted logging with context binding for observability.
Replaces print statements with structured log entries.

Usage:
    from invoices.logging import get_logger

    logger = get_logger(__name__)
    logger.info("processing_document", sha256=sha256, stage="ingest")
    logger.error("extraction_failed", error=str(e), doc_id=doc_id)
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add any extra fields from the record
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add source location
        log_entry["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class StructuredLogger:
    """Structured logger with event-based logging API."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self._context: dict[str, Any] = {}

        # Configure handler if not already configured
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(JSONFormatter())
            self.logger.addHandler(handler)

    def bind(self, **kwargs: Any) -> "StructuredLogger":
        """Return a new logger with bound context fields."""
        new_logger = StructuredLogger.__new__(StructuredLogger)
        new_logger.logger = self.logger
        new_logger._context = {**self._context, **kwargs}
        return new_logger

    def _log(self, level: int, event: str, **kwargs: Any) -> None:
        """Internal log method with context merging."""
        exc_info = kwargs.pop("exc_info", None)
        merged = {**self._context, **kwargs}
        # Use logger.log with stacklevel=3 to get correct caller info
        # Stack: caller -> info/error/etc -> _log -> logger.log
        self.logger.log(
            level,
            event,
            exc_info=exc_info,
            extra={"extra_fields": merged},
            stacklevel=3,
        )

    def debug(self, event: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, event, **kwargs)

    def exception(self, event: str, **kwargs: Any) -> None:
        """Log error with exception info."""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, event, **kwargs)


# Module-level configuration — read from central Config
# NOTE: We import lazily to avoid circular imports (Config doesn't use logging)
def _get_log_level() -> str:
    """Get log level from Config, with fallback for early import scenarios."""
    try:
        from invoices.config import Config

        return Config.LOG_LEVEL.upper()
    except ImportError:
        import os

        return os.environ.get("INVOICEX_LOG_LEVEL", "INFO").upper()


_LOG_LEVEL = _get_log_level()

# Cache of loggers
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured StructuredLogger instance
    """
    if name not in _loggers:
        level = getattr(logging, _LOG_LEVEL, logging.INFO)
        _loggers[name] = StructuredLogger(name, level)
    return _loggers[name]


def configure_logging(
    level: str = "INFO",
    output: str = "stderr",
) -> None:
    """Configure global logging settings.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        output: Output destination ('stderr', 'stdout', or file path)
    """
    global _LOG_LEVEL

    _LOG_LEVEL = level.upper()

    # Update existing loggers
    log_level = getattr(logging, _LOG_LEVEL, logging.INFO)
    for logger in _loggers.values():
        logger.logger.setLevel(log_level)


# Convenience functions for quick logging without creating a logger
_default_logger: StructuredLogger | None = None


def _get_default_logger() -> StructuredLogger:
    """Get the default logger for module-level functions."""
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("invoices")
    return _default_logger


def log_info(event: str, **kwargs: Any) -> None:
    """Log info event with the default logger."""
    _get_default_logger().info(event, **kwargs)


def log_error(event: str, **kwargs: Any) -> None:
    """Log error event with the default logger."""
    _get_default_logger().error(event, **kwargs)


def log_warning(event: str, **kwargs: Any) -> None:
    """Log warning event with the default logger."""
    _get_default_logger().warning(event, **kwargs)


def log_debug(event: str, **kwargs: Any) -> None:
    """Log debug event with the default logger."""
    _get_default_logger().debug(event, **kwargs)


# Pipeline-specific logging helpers
def log_pipeline_start(stage: str, doc_count: int = 0, **kwargs: Any) -> None:
    """Log pipeline stage start."""
    _get_default_logger().info(
        "pipeline_stage_start",
        stage=stage,
        doc_count=doc_count,
        **kwargs,
    )


def log_pipeline_complete(
    stage: str,
    doc_count: int = 0,
    duration_seconds: float = 0.0,
    **kwargs: Any,
) -> None:
    """Log pipeline stage completion."""
    _get_default_logger().info(
        "pipeline_stage_complete",
        stage=stage,
        doc_count=doc_count,
        duration_seconds=round(duration_seconds, 4),
        **kwargs,
    )


def log_document_processed(
    doc_id: str,
    sha256: str,
    stage: str,
    status: str = "success",
    **kwargs: Any,
) -> None:
    """Log document processing event."""
    _get_default_logger().info(
        "document_processed",
        doc_id=doc_id,
        sha256=sha256[:16],  # Truncate for readability
        stage=stage,
        status=status,
        **kwargs,
    )


def log_extraction_result(
    doc_id: str,
    predicted_count: int,
    abstain_count: int,
    missing_count: int,
    needs_review: bool,
    min_confidence: float,
    **kwargs: Any,
) -> None:
    """Log extraction result summary."""
    _get_default_logger().info(
        "extraction_result",
        doc_id=doc_id,
        predicted_count=predicted_count,
        abstain_count=abstain_count,
        missing_count=missing_count,
        needs_review=needs_review,
        min_confidence=round(min_confidence, 4),
        **kwargs,
    )
