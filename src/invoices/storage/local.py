"""Local filesystem storage backend.

Implements the StorageBackend protocol using the local filesystem.
All paths are relative to the data directory (paths.get_data_dir()).

Usage:
    from invoices.storage.local import LocalStorageBackend

    storage = LocalStorageBackend()
    storage.write_text("models/manifest.json", '{"version": "1.0"}')
    content = storage.read_text("models/manifest.json")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..exceptions import StorageError
from ..logging import get_logger
from ..paths import get_data_dir

logger = get_logger(__name__)


class LocalStorageBackend:
    """Local filesystem implementation of StorageBackend.

    Uses paths.get_data_dir() as the base directory for all operations.
    Paths are normalized to POSIX-style internally but converted to
    platform-specific paths for filesystem operations.

    Attributes:
        base_path: The base directory for all storage operations.
    """

    def __init__(self, base_path: Path | None = None) -> None:
        """Initialize the local storage backend.

        Args:
            base_path: Base directory for storage. Defaults to get_data_dir().
        """
        self.base_path = base_path if base_path is not None else get_data_dir()
        self.base_path = Path(self.base_path).resolve()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a relative path to an absolute filesystem path.

        Normalizes POSIX-style paths to platform-specific paths.

        Args:
            path: Relative path using forward slashes.

        Returns:
            Absolute Path object.
        """
        # Normalize to POSIX-style, then convert to Path
        normalized = path.replace("\\", "/").lstrip("/")
        return self.base_path / normalized

    def exists(self, path: str) -> bool:
        """Check if a path exists in storage.

        Args:
            path: Relative path from storage root.

        Returns:
            True if the path exists, False otherwise.
        """
        resolved = self._resolve_path(path)
        return resolved.exists()

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from storage.

        Args:
            path: Relative path from storage root.

        Returns:
            Raw bytes content.

        Raises:
            StorageError: If the file doesn't exist or read fails.
        """
        resolved = self._resolve_path(path)

        try:
            return resolved.read_bytes()
        except FileNotFoundError as e:
            raise StorageError("read", path, f"File not found: {resolved}") from e
        except PermissionError as e:
            raise StorageError("read", path, f"Permission denied: {resolved}") from e
        except Exception as e:
            raise StorageError("read", path, str(e)) from e

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
        resolved = self._resolve_path(path)

        try:
            return resolved.read_text(encoding=encoding)
        except FileNotFoundError as e:
            raise StorageError("read", path, f"File not found: {resolved}") from e
        except PermissionError as e:
            raise StorageError("read", path, f"Permission denied: {resolved}") from e
        except UnicodeDecodeError as e:
            raise StorageError("read", path, f"Encoding error: {e}") from e
        except Exception as e:
            raise StorageError("read", path, str(e)) from e

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary content to storage.

        Creates parent directories as needed.

        Args:
            path: Relative path from storage root.
            data: Binary content to write.

        Raises:
            StorageError: If write fails.
        """
        resolved = self._resolve_path(path)

        try:
            # Ensure parent directory exists
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_bytes(data)
        except PermissionError as e:
            raise StorageError("write", path, f"Permission denied: {resolved}") from e
        except OSError as e:
            raise StorageError("write", path, str(e)) from e

    def write_text(self, path: str, data: str, encoding: str = "utf-8") -> None:
        """Write text content to storage.

        Creates parent directories as needed.

        Args:
            path: Relative path from storage root.
            data: Text content to write.
            encoding: Text encoding (default: utf-8).

        Raises:
            StorageError: If write fails.
        """
        resolved = self._resolve_path(path)

        try:
            # Ensure parent directory exists
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(data, encoding=encoding)
        except PermissionError as e:
            raise StorageError("write", path, f"Permission denied: {resolved}") from e
        except OSError as e:
            raise StorageError("write", path, str(e)) from e

    def delete(self, path: str) -> bool:
        """Delete a file from storage.

        Args:
            path: Relative path from storage root.

        Returns:
            True if the file was deleted, False if it didn't exist.

        Raises:
            StorageError: If deletion fails for reasons other than non-existence.
        """
        resolved = self._resolve_path(path)

        try:
            if not resolved.exists():
                return False
            resolved.unlink()
            return True
        except PermissionError as e:
            raise StorageError("delete", path, f"Permission denied: {resolved}") from e
        except OSError as e:
            raise StorageError("delete", path, str(e)) from e

    def list_prefix(self, prefix: str) -> list[str]:
        """List all paths under a prefix.

        Args:
            prefix: Path prefix to list (e.g., "models/").

        Returns:
            List of relative paths under the prefix, using forward slashes.
        """
        resolved = self._resolve_path(prefix)
        results: list[str] = []

        if not resolved.exists():
            return results

        # Walk directory tree
        if resolved.is_file():
            # Single file matches prefix exactly
            rel_path = resolved.relative_to(self.base_path)
            return [str(rel_path).replace(os.sep, "/")]

        for root, _, files in os.walk(resolved):
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                rel_path = file_path.relative_to(self.base_path)
                # Always use forward slashes in returned paths
                results.append(str(rel_path).replace(os.sep, "/"))

        return sorted(results)

    def health_check(self) -> dict[str, Any]:
        """Check storage backend health.

        Tests write access by creating and removing a temporary file.

        Returns:
            Health status dictionary.
        """
        details: dict[str, Any] = {
            "base_path": str(self.base_path),
            "exists": self.base_path.exists(),
        }

        try:
            # Test write access with a temporary file
            test_file = ".health_check_test"
            self.write_text(test_file, "health_check")
            self.delete(test_file)

            return {
                "status": "healthy",
                "message": "Local storage is accessible",
                "backend": "local",
                "details": details,
            }
        except StorageError as e:
            return {
                "status": "unhealthy",
                "message": f"Write access failed: {e.reason}",
                "backend": "local",
                "details": details,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Unexpected error: {e}",
                "backend": "local",
                "details": details,
            }
