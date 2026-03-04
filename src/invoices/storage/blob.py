"""Azure Blob Storage backend.

Implements the StorageBackend protocol using Azure Blob Storage.
Supports both connection string and DefaultAzureCredential authentication.

Usage:
    from invoices.storage.blob import BlobStorageBackend

    # Using connection string
    storage = BlobStorageBackend(
        connection_string="DefaultEndpointsProtocol=https;AccountName=..."
    )

    # Using DefaultAzureCredential (managed identity, CLI, etc.)
    storage = BlobStorageBackend(
        account_name="myaccount",
        container_name="invoicex",
    )

    storage.write_text("models/manifest.json", '{"version": "1.0"}')
    content = storage.read_text("models/manifest.json")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import Config
from ..exceptions import StorageError
from ..logging import get_logger

if TYPE_CHECKING:
    from azure.storage.blob import (  # type: ignore[import-not-found]
        BlobServiceClient,
        ContainerClient,
    )

logger = get_logger(__name__)


class BlobStorageBackend:
    """Azure Blob Storage implementation of StorageBackend.

    Lazily imports Azure SDK to avoid requiring it for local development.
    Uses temp files for operations that require filesystem paths (e.g., XGBoost).

    Attributes:
        account_name: Azure storage account name.
        container_name: Blob container name.
        _client: Lazy-initialized BlobServiceClient.
        _container: Lazy-initialized ContainerClient.
    """

    def __init__(
        self,
        account_name: str | None = None,
        container_name: str | None = None,
        connection_string: str | None = None,
        managed_identity_client_id: str | None = None,
    ) -> None:
        """Initialize the blob storage backend.

        Args:
            account_name: Azure storage account name. Falls back to
                Config.AZURE_STORAGE_ACCOUNT_NAME if not provided.
            container_name: Blob container name. Falls back to
                Config.AZURE_STORAGE_CONTAINER_NAME if not provided.
            connection_string: Azure storage connection string. Falls back to
                Config.AZURE_STORAGE_CONNECTION_STRING if not provided.
                Takes precedence over account_name if both are set.
            managed_identity_client_id: User-assigned managed identity client ID.
                Falls back to AZURE_MANAGED_IDENTITY_CLIENT_ID env var.
                Only used when authenticating via DefaultAzureCredential
                (i.e., when connection_string is not set).

        Raises:
            StorageError: If neither connection_string nor account_name is provided.
        """
        self.connection_string = (
            connection_string or Config.AZURE_STORAGE_CONNECTION_STRING
        )
        self.account_name = account_name or Config.AZURE_STORAGE_ACCOUNT_NAME
        self.container_name = container_name or Config.AZURE_STORAGE_CONTAINER_NAME
        self.managed_identity_client_id = managed_identity_client_id or os.environ.get(
            "AZURE_MANAGED_IDENTITY_CLIENT_ID", ""
        )

        if not self.connection_string and not self.account_name:
            raise StorageError(
                "init",
                "",
                "Either connection_string or account_name is required for blob storage",
            )

        self._client: BlobServiceClient | None = None
        self._container: ContainerClient | None = None

    def _get_container(self) -> ContainerClient:
        """Get or create the container client.

        Lazily imports Azure SDK and initializes the client.
        Creates the container if it doesn't exist.

        Returns:
            ContainerClient for the configured container.

        Raises:
            StorageError: If Azure SDK import or client creation fails.
        """
        if self._container is not None:
            return self._container

        try:
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise StorageError(
                "init",
                "",
                "azure-storage-blob not installed. Run: pip install azure-storage-blob",
            ) from e

        try:
            # Create service client
            if self.connection_string:
                self._client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
                logger.debug(
                    "blob_client_initialized",
                    method="connection_string",
                    container=self.container_name,
                )
            else:
                # Use DefaultAzureCredential for managed identity, CLI, etc.
                try:
                    from azure.identity import (  # type: ignore[import-not-found]
                        DefaultAzureCredential,
                    )
                except ImportError as e:
                    raise StorageError(
                        "init",
                        "",
                        "azure-identity not installed. Run: pip install azure-identity",
                    ) from e

                kwargs: dict[str, str] = {}
                if self.managed_identity_client_id:
                    kwargs["managed_identity_client_id"] = (
                        self.managed_identity_client_id
                    )
                credential = DefaultAzureCredential(**kwargs)
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self._client = BlobServiceClient(
                    account_url=account_url,
                    credential=credential,
                )
                auth_method = (
                    "managed_identity"
                    if self.managed_identity_client_id
                    else "default_credential"
                )
                logger.debug(
                    "blob_client_initialized",
                    method=auth_method,
                    account=self.account_name,
                    container=self.container_name,
                )

            # Get container client and create if not exists
            self._container = self._client.get_container_client(self.container_name)

            # Create container if it doesn't exist
            if not self._container.exists():
                self._container.create_container()
                logger.info(
                    "blob_container_created",
                    container=self.container_name,
                )

            return self._container

        except StorageError:
            raise
        except Exception as e:
            raise StorageError(
                "init", "", f"Failed to initialize blob client: {e}"
            ) from e

    def _normalize_path(self, path: str) -> str:
        """Normalize a path for blob storage.

        Converts backslashes to forward slashes and strips leading slashes.

        Args:
            path: Input path.

        Returns:
            Normalized blob path.
        """
        return path.replace("\\", "/").lstrip("/")

    def exists(self, path: str) -> bool:
        """Check if a blob exists.

        Args:
            path: Relative path (blob name).

        Returns:
            True if the blob exists, False otherwise.
        """
        container = self._get_container()
        blob_name = self._normalize_path(path)

        try:
            blob_client = container.get_blob_client(blob_name)
            return blob_client.exists()  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("blob_exists_failed", path=path, error=str(e))
            return False

    def read_bytes(self, path: str) -> bytes:
        """Read binary content from a blob.

        Args:
            path: Relative path (blob name).

        Returns:
            Raw bytes content.

        Raises:
            StorageError: If the blob doesn't exist or read fails.
        """
        container = self._get_container()
        blob_name = self._normalize_path(path)

        try:
            blob_client = container.get_blob_client(blob_name)
            downloader = blob_client.download_blob()
            return downloader.readall()  # type: ignore[no-any-return]
        except Exception as e:
            # Check for specific Azure error types
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "404" in error_msg:
                raise StorageError("read", path, f"Blob not found: {blob_name}") from e
            raise StorageError("read", path, error_msg) from e

    def read_text(self, path: str, encoding: str = "utf-8") -> str:
        """Read text content from a blob.

        Args:
            path: Relative path (blob name).
            encoding: Text encoding (default: utf-8).

        Returns:
            Decoded text content.

        Raises:
            StorageError: If the blob doesn't exist or read fails.
        """
        content = self.read_bytes(path)
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as e:
            raise StorageError("read", path, f"Encoding error: {e}") from e

    def write_bytes(self, path: str, data: bytes) -> None:
        """Write binary content to a blob.

        Overwrites existing blobs. Blob storage doesn't have directories,
        so no parent directory creation is needed.

        Args:
            path: Relative path (blob name).
            data: Binary content to write.

        Raises:
            StorageError: If write fails.
        """
        container = self._get_container()
        blob_name = self._normalize_path(path)

        try:
            blob_client = container.get_blob_client(blob_name)
            blob_client.upload_blob(data, overwrite=True)
        except Exception as e:
            raise StorageError("write", path, str(e)) from e

    def write_text(self, path: str, data: str, encoding: str = "utf-8") -> None:
        """Write text content to a blob.

        Args:
            path: Relative path (blob name).
            data: Text content to write.
            encoding: Text encoding (default: utf-8).

        Raises:
            StorageError: If write fails.
        """
        self.write_bytes(path, data.encode(encoding))

    def delete(self, path: str) -> bool:
        """Delete a blob.

        Args:
            path: Relative path (blob name).

        Returns:
            True if the blob was deleted, False if it didn't exist.

        Raises:
            StorageError: If deletion fails for reasons other than non-existence.
        """
        container = self._get_container()
        blob_name = self._normalize_path(path)

        try:
            blob_client = container.get_blob_client(blob_name)
            if not blob_client.exists():
                return False
            blob_client.delete_blob()
            return True
        except Exception as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "404" in error_msg:
                return False
            raise StorageError("delete", path, error_msg) from e

    def list_prefix(self, prefix: str) -> list[str]:
        """List all blobs under a prefix.

        Args:
            prefix: Blob name prefix (e.g., "models/").

        Returns:
            List of blob names under the prefix.
        """
        container = self._get_container()
        normalized_prefix = self._normalize_path(prefix)
        results: list[str] = []

        try:
            blobs = container.list_blobs(name_starts_with=normalized_prefix)
            for blob in blobs:
                results.append(blob.name)
            return sorted(results)
        except Exception as e:
            logger.warning("blob_list_failed", prefix=prefix, error=str(e))
            return results

    def health_check(self) -> dict[str, Any]:
        """Check storage backend health.

        Tests connectivity by listing blobs (lightweight operation).

        Returns:
            Health status dictionary.
        """
        details: dict[str, Any] = {
            "account_name": self.account_name or "(connection_string)",
            "container_name": self.container_name,
        }

        try:
            container = self._get_container()
            # Lightweight check: list with max_results=1
            list(container.list_blobs(results_per_page=1))

            return {
                "status": "healthy",
                "message": "Blob storage is accessible",
                "backend": "blob",
                "details": details,
            }
        except StorageError as e:
            return {
                "status": "unhealthy",
                "message": f"Blob storage error: {e.reason}",
                "backend": "blob",
                "details": details,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Unexpected error: {e}",
                "backend": "blob",
                "details": details,
            }

    def download_to_tempfile(self, path: str, suffix: str = "") -> Path:
        """Download a blob to a temporary file.

        Useful for operations that require filesystem paths (e.g., XGBoost load).
        The caller is responsible for cleaning up the temporary file.

        Args:
            path: Relative path (blob name).
            suffix: Optional file suffix (e.g., ".json").

        Returns:
            Path to the temporary file.

        Raises:
            StorageError: If download fails.
        """
        content = self.read_bytes(path)

        # Create temp file with optional suffix
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        try:
            with open(fd, "wb") as f:
                f.write(content)
            return Path(temp_path)
        except Exception as e:
            # Clean up on failure
            Path(temp_path).unlink(missing_ok=True)
            raise StorageError("download", path, str(e)) from e

    def upload_from_file(self, local_path: Path, blob_path: str) -> None:
        """Upload a local file to blob storage.

        Useful for operations that create files locally (e.g., XGBoost save).

        Args:
            local_path: Path to the local file.
            blob_path: Destination blob path.

        Raises:
            StorageError: If upload fails.
        """
        try:
            content = local_path.read_bytes()
            self.write_bytes(blob_path, content)
        except StorageError:
            raise
        except Exception as e:
            raise StorageError("upload", blob_path, str(e)) from e
