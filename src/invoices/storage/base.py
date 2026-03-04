"""Base protocol for storage backends.

This module defines the StorageBackend protocol that all storage implementations
must follow. The protocol enables seamless switching between local filesystem
and Azure Blob Storage for model/artifact persistence.

Usage:
    from invoices.storage import StorageBackend, get_storage

    storage = get_storage()  # Returns backend based on Config.STORAGE_BACKEND
    storage.write_text("models/manifest.json", json_data)
    content = storage.read_text("models/manifest.json")
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for storage backends (local filesystem, Azure Blob, etc.).

    All paths use POSIX-style forward slashes internally (e.g., "models/field/model.json").
    Implementations handle platform-specific path conversion as needed.

    Implementations must be synchronous - the invoice extraction pipeline is
    CPU-bound, so async I/O provides no benefit.
    """

    def exists(self, path: str) -> bool:
        """Check if a path exists in storage.

        Args:
            path: Relative path from storage root (e.g., "models/manifest.json").

        Returns:
            True if the path exists, False otherwise.
        """
        ...

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from storage.

        Args:
            path: Relative path from storage root.

        Returns:
            Raw bytes content.

        Raises:
            StorageError: If the file doesn't exist or read fails.
        """
        ...

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from storage.

        Args:
            path: Relative path from storage root.
            encoding: Text encoding (default: utf-8).

        Returns:
            Decoded text content.

        Raises:
            StorageError: If the file doesn't exist or read fails.
        """
        ...

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary content to storage.

        Creates parent directories/prefixes as needed.

        Args:
            path: Relative path from storage root.
            data: Binary content to write.

        Raises:
            StorageError: If write fails.
        """
        ...

    def write_text(self, path: str, data: str, encoding: str = "utf-8") -> None:
        """Write text content to storage.

        Creates parent directories/prefixes as needed.

        Args:
            path: Relative path from storage root.
            data: Text content to write.
            encoding: Text encoding (default: utf-8).

        Raises:
            StorageError: If write fails.
        """
        ...

    def delete(self, path: str) -> bool:
        """Delete a file from storage.

        Args:
            path: Relative path from storage root.

        Returns:
            True if the file was deleted, False if it didn't exist.

        Raises:
            StorageError: If deletion fails for reasons other than non-existence.
        """
        ...

    def list_prefix(self, prefix: str) -> list[str]:
        """List all paths under a prefix.

        Args:
            prefix: Path prefix to list (e.g., "models/" lists all model files).

        Returns:
            List of relative paths under the prefix (including the prefix).
            For example, list_prefix("models/") might return:
            ["models/manifest.json", "models/TotalAmount/model.json", ...]
        """
        ...

    def health_check(self) -> dict[str, Any]:
        """Check storage backend health.

        Performs a lightweight operation to verify storage is accessible
        (e.g., list root prefix for blob, check write access for local).

        Returns:
            Health status dictionary with keys:
            - status: "healthy", "degraded", or "unhealthy"
            - message: Human-readable status message
            - backend: Backend type ("local" or "blob")
            - details: Additional diagnostic info
        """
        ...
