"""Lightweight liveness probe for container healthchecks.

Used by docker-compose healthcheck for the invoicex CLI worker container,
which doesn't run the FastAPI server and needs its own health function.
"""

from invoices.config import Config


def liveness_probe() -> bool:
    """Return True if the service can start processing."""
    try:
        return Config.get_data_path().exists()
    except Exception:
        return False


def readiness_warnings() -> list[str]:
    """Return warnings about configuration that may cause issues.

    Checks for misconfigurations that won't prevent startup but could
    lead to data loss or unexpected behaviour in production.
    """
    warnings: list[str] = []

    if Config.STORAGE_BACKEND == "local" and Config.ENVIRONMENT == "production":
        warnings.append(
            "Local storage in production — PDFs not persisted to durable storage"
        )

    return warnings
