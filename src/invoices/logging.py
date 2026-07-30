"""Logging — structlog wrapper with OTel span injection.

Replaces 265 LOC of bespoke JSONFormatter + StructuredLogger.
structlog.stdlib.BoundLogger API matches the prior interface (.info/.warning/
.error/.debug/.exception/.bind), so 64 callsite files are unchanged.

OTel span context (trace_id + span_id) injected via processor when a span
is active; safe no-op when not.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import orjson
import structlog
from opentelemetry import trace as _otrace


def _add_otel_span(
    logger: Any, _method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Inject trace_id + span_id from active OTel span if any."""
    span = _otrace.get_current_span()
    ctx = span.get_span_context()
    if ctx is not None and ctx.is_valid:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib + structlog. Call once at app startup."""
    logging.basicConfig(format="%(message)s", level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_otel_span,
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(serializer=orjson.dumps),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger. Compatible with prior get_logger API."""
    return structlog.stdlib.get_logger(name) if name else structlog.stdlib.get_logger()


# Backward-compat convenience helpers (re-exported from invoices.__init__)
def log_info(msg: str, **kwargs: Any) -> None:
    get_logger().info(msg, **kwargs)


def log_warning(msg: str, **kwargs: Any) -> None:
    get_logger().warning(msg, **kwargs)


def log_error(msg: str, **kwargs: Any) -> None:
    get_logger().error(msg, **kwargs)


def log_debug(msg: str, **kwargs: Any) -> None:
    get_logger().debug(msg, **kwargs)
