"""Storage abstraction layer for model and artifact persistence.

This module provides a unified interface for storing and retrieving models
and artifacts, with support for both local filesystem and Azure Blob Storage.

The storage backend is selected based on Config.STORAGE_BACKEND:
- "local" (default): Uses LocalStorageBackend with paths.get_data_dir()
- "blob": Uses BlobStorageBackend with Azure Blob Storage

Usage:
    from invoices.storage import get_storage, StorageBackend

    # Get the configured storage backend
    storage = get_storage()

    # Use storage operations
    storage.write_text("models/manifest.json", json_data)
    content = storage.read_text("models/manifest.json")
    files = storage.list_prefix("models/")

    # Check health
    health = storage.health_check()
    print(health["status"])  # "healthy" or "unhealthy"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Config
from ..exceptions import StorageError
from ..logging import get_logger
from .base import StorageBackend
from .local import LocalStorageBackend

if TYPE_CHECKING:
    pass  # For future type imports if needed

logger = get_logger(__name__)

# Module-level cache for storage backend singleton
_storage_instance: StorageBackend | None = None


def get_storage(*, force_new: bool = False) -> StorageBackend:
    """Get the configured storage backend.

    Returns a singleton instance based on Config.STORAGE_BACKEND.
    Use force_new=True to create a fresh instance (useful for testing).

    Args:
        force_new: If True, create a new instance instead of returning cached.

    Returns:
        StorageBackend instance (LocalStorageBackend or BlobStorageBackend).

    Raises:
        StorageError: If blob backend is configured but missing required config.
        ValueError: If STORAGE_BACKEND has an invalid value.
    """
    global _storage_instance

    if _storage_instance is not None and not force_new:
        return _storage_instance

    backend = Config.STORAGE_BACKEND.lower()

    if backend == "local":
        logger.debug("storage_backend_selected", backend="local")
        _storage_instance = LocalStorageBackend()
        return _storage_instance

    elif backend == "blob":
        # Lazy import to avoid requiring Azure SDK for local development
        from .blob import BlobStorageBackend

        # Validate configuration
        if (
            not Config.AZURE_STORAGE_CONNECTION_STRING
            and not Config.AZURE_STORAGE_ACCOUNT_NAME
        ):
            raise StorageError(
                "init",
                "",
                "STORAGE_BACKEND=blob requires AZURE_STORAGE_CONNECTION_STRING "
                "or AZURE_STORAGE_ACCOUNT_NAME to be set",
            )

        logger.debug(
            "storage_backend_selected",
            backend="blob",
            account=Config.AZURE_STORAGE_ACCOUNT_NAME or "(connection_string)",
            container=Config.AZURE_STORAGE_CONTAINER_NAME,
        )

        _storage_instance = BlobStorageBackend(
            account_name=Config.AZURE_STORAGE_ACCOUNT_NAME or None,
            container_name=Config.AZURE_STORAGE_CONTAINER_NAME,
            connection_string=Config.AZURE_STORAGE_CONNECTION_STRING or None,
        )
        return _storage_instance

    else:
        raise ValueError(
            f"Invalid STORAGE_BACKEND: '{backend}'. Must be 'local' or 'blob'."
        )


def reset_storage() -> None:
    """Reset the cached storage instance.

    Useful for testing or when configuration changes at runtime.
    """
    global _storage_instance
    _storage_instance = None
    logger.debug("storage_backend_reset")


# Public exports
__all__ = [
    "StorageBackend",
    "LocalStorageBackend",
    "get_storage",
    "reset_storage",
]
