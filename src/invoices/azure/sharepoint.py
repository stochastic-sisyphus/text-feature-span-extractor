"""SharePoint document library connector via Microsoft Graph API.

Uses azure-identity for authentication (service principal or managed identity)
and httpx for async HTTP calls to the Graph API.

Usage:
    from invoices.azure.sharepoint import SharePointConnector
    from invoices.azure.config import get_azure_config

    config = get_azure_config()
    async with SharePointConnector(config.sharepoint) as connector:
        documents = await connector.list_documents()
        for doc in documents:
            content = await connector.download(doc.id)
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from invoices.azure.base import Document, DocumentSource
from invoices.azure.config import SharePointConfig
from invoices.exceptions import IntegrationError
from invoices.logging import get_logger

logger = get_logger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPE = "https://graph.microsoft.com/.default"
# Refresh token 5 minutes before actual expiry
TOKEN_REFRESH_BUFFER_SECONDS = 300


class SharePointError(IntegrationError):
    """SharePoint-specific error."""

    def __init__(
        self,
        operation: str,
        reason: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        message = f"SharePoint {operation} failed: {reason}"
        super().__init__(
            message,
            operation=operation,
            reason=reason,
            status_code=status_code,
            request_id=request_id,
        )
        self.operation = operation
        self.reason = reason
        self.status_code = status_code
        self.request_id = request_id


class SharePointConnector(DocumentSource):
    """SharePoint document library connector using Microsoft Graph API.

    Implements the DocumentSource protocol for accessing SharePoint
    document libraries. Read-only: files are never deleted or modified.

    Auth: When client_secret is set, uses ClientSecretCredential (service
    principal). Otherwise, falls back to DefaultAzureCredential (managed
    identity in Azure, CLI/env credentials locally).

    Usage:
        config = SharePointConfig(
            site_id="...",
            tenant_id="...",
            client_id="...",
            client_secret="...",  # omit for managed identity
        )
        async with SharePointConnector(config) as connector:
            docs = await connector.list_documents()
    """

    def __init__(self, config: SharePointConfig):
        """Initialize the SharePoint connector.

        Args:
            config: SharePoint configuration
        """
        self.config = config
        self._session: httpx.AsyncClient | None = None
        self._token: str | None = None
        self._token_expires: datetime | None = None

    async def __aenter__(self) -> "SharePointConnector":
        """Enter async context and initialize client."""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        await self._cleanup()

    async def _init_client(self) -> None:
        """Initialize the httpx session and acquire initial token."""
        if not self.config.is_configured():
            raise SharePointError(
                "init",
                "SharePoint not properly configured. Check environment variables.",
            )

        try:
            self._session = httpx.AsyncClient(
                base_url=GRAPH_BASE_URL,
                timeout=30.0,
            )
            await self._acquire_token()

            # Resolve site/drive IDs from hostname + site_path if needed
            if (
                not self.config.site_id
                and self.config.hostname
                and self.config.site_path
            ):
                await self._resolve_ids()

            logger.info(
                "sharepoint_client_initialized",
                site_id=self.config.site_id,
            )
        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("init", str(e)) from e

    async def _cleanup(self) -> None:
        """Cleanup session resources."""
        if self._session:
            try:
                await self._session.aclose()
            except Exception as e:
                logger.warning("sharepoint_session_cleanup_failed", error=str(e))
            self._session = None
        self._token = None
        self._token_expires = None

    async def _acquire_token(self) -> None:
        """Acquire OAuth2 access token via azure-identity.

        Uses ClientSecretCredential when client_secret is set (local dev
        with service principal), otherwise falls back to
        DefaultAzureCredential (managed identity in Azure).
        """
        try:
            if self.config.client_secret:
                try:
                    from azure.identity import (  # type: ignore[import-not-found]
                        ClientSecretCredential,
                    )
                except ImportError as e:
                    raise SharePointError(
                        "auth",
                        "azure-identity not installed. Run: pip install azure-identity",
                    ) from e

                credential = ClientSecretCredential(
                    tenant_id=self.config.tenant_id,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                )
                auth_method = "client_secret"
            else:
                try:
                    from azure.identity import (  # type: ignore[import-not-found]
                        DefaultAzureCredential,
                    )
                except ImportError as e:
                    raise SharePointError(
                        "auth",
                        "azure-identity not installed. Run: pip install azure-identity",
                    ) from e

                kwargs: dict[str, str] = {}
                if self.config.managed_identity_client_id:
                    kwargs["managed_identity_client_id"] = (
                        self.config.managed_identity_client_id
                    )
                credential = DefaultAzureCredential(**kwargs)
                auth_method = "default_credential"

            # azure-identity is sync — run in thread to avoid blocking
            token = await asyncio.to_thread(credential.get_token, GRAPH_SCOPE)
            self._token = token.token
            self._token_expires = datetime.fromtimestamp(
                token.expires_on, tz=timezone.utc
            )

            logger.debug("sharepoint_token_acquired", method=auth_method)

        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("auth", str(e)) from e

    async def _ensure_token(self) -> None:
        """Ensure we have a valid token, refreshing if near expiry."""
        if self._token is None:
            await self._acquire_token()
            return

        if self._token_expires:
            now = datetime.now(timezone.utc)
            buffer = self._token_expires.timestamp() - TOKEN_REFRESH_BUFFER_SECONDS
            if now.timestamp() >= buffer:
                await self._acquire_token()

    async def _resolve_ids(self) -> None:
        """Resolve site_id and drive_id from hostname + site_path via Graph API.

        Makes two Graph calls:
        1. GET /sites/{hostname}:{site_path} → site_id
        2. GET /sites/{site_id}/drives → drive_id (first "Documents" drive)
        """
        assert self._session is not None

        # 1. Resolve site ID
        site_url = f"/sites/{self.config.hostname}:{self.config.site_path}"
        response = await self._session.get(site_url, headers=self._get_headers())
        if response.status_code != 200:
            raise SharePointError(
                "resolve_ids",
                f"Site lookup returned {response.status_code}: "
                f"{self._parse_error(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )
        self.config.site_id = response.json()["id"]

        # 2. Resolve drive ID
        drives_url = f"/sites/{self.config.site_id}/drives"
        response = await self._session.get(drives_url, headers=self._get_headers())
        if response.status_code != 200:
            raise SharePointError(
                "resolve_ids",
                f"Drives lookup returned {response.status_code}: "
                f"{self._parse_error(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )
        for drive in response.json().get("value", []):
            if drive.get("name") in ("Shared Documents", "Documents"):
                self.config.drive_id = drive["id"]
                break

        logger.info(
            "sharepoint_ids_resolved",
            hostname=self.config.hostname,
            site_id=self.config.site_id,
            drive_id=self.config.drive_id,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
        }

    def _drive_prefix(self) -> str:
        """Build the Graph API path prefix for the configured drive.

        Returns:
            Path like /sites/{site_id}/drives/{drive_id}
            or /sites/{site_id}/drive (default library)
        """
        site = f"/sites/{self.config.site_id}"
        if self.config.drive_id:
            return f"{site}/drives/{self.config.drive_id}"
        return f"{site}/drive"

    async def list_documents(
        self,
        folder: str | None = None,
        modified_since: datetime | None = None,
    ) -> list[Document]:
        """List PDF documents in a SharePoint folder.

        Args:
            folder: Folder path (defaults to config.folder_path)
            modified_since: Only return documents modified after this time

        Returns:
            List of Document objects
        """
        if self._session is None:
            raise SharePointError("list_documents", "Client not initialized")

        await self._ensure_token()

        folder_path = folder or self.config.folder_path
        folder_path = folder_path.lstrip("/")
        prefix = self._drive_prefix()
        url = f"{prefix}/root:/{folder_path}:/children"

        try:
            documents: list[Document] = []
            # Follow pagination
            next_url: str | None = url

            while next_url is not None:
                response = await self._session.get(
                    next_url,
                    headers=self._get_headers(),
                )

                if response.status_code != 200:
                    error_detail = self._parse_error(response)
                    raise SharePointError(
                        "list_documents",
                        error_detail,
                        status_code=response.status_code,
                        request_id=response.headers.get("request-id"),
                    )

                data = response.json()

                for item in data.get("value", []):
                    # Skip folders
                    if "folder" in item:
                        continue

                    name: str = item.get("name", "")
                    # Skip non-PDF files
                    if not name.lower().endswith(".pdf"):
                        continue

                    # Parse timestamps
                    created_str = item.get("createdDateTime")
                    modified_str = item.get("lastModifiedDateTime")

                    created_at = (
                        datetime.fromisoformat(created_str)
                        if created_str
                        else datetime.now(timezone.utc)
                    )
                    modified_at = (
                        datetime.fromisoformat(modified_str)
                        if modified_str
                        else datetime.now(timezone.utc)
                    )

                    # Ensure timezone-aware for comparison
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    if modified_at.tzinfo is None:
                        modified_at = modified_at.replace(tzinfo=timezone.utc)

                    # Apply modified_since filter
                    if modified_since and modified_at < modified_since:
                        continue

                    size_bytes = item.get("size", 0)
                    item_id = item.get("id", "")
                    web_url = item.get("webUrl", "")

                    doc = Document(
                        id=item_id,
                        name=name,
                        content_type="application/pdf",
                        size_bytes=int(size_bytes),
                        created_at=created_at,
                        modified_at=modified_at,
                        source_url=web_url,
                        metadata={
                            "drive_item_id": item_id,
                            "parent_folder": folder_path,
                            "web_url": web_url,
                            "etag": item.get("eTag", ""),
                        },
                    )
                    documents.append(doc)

                # Follow @odata.nextLink for pagination
                next_url = data.get("@odata.nextLink")

            logger.info(
                "sharepoint_list_complete",
                folder=folder_path,
                document_count=len(documents),
            )
            return documents

        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("list_documents", str(e)) from e

    async def download(self, document_id: str) -> bytes:
        """Download document content by drive item ID.

        Args:
            document_id: Graph API drive item ID

        Returns:
            Raw document bytes
        """
        if self._session is None:
            raise SharePointError("download", "Client not initialized")

        await self._ensure_token()

        prefix = self._drive_prefix()
        url = f"{prefix}/items/{document_id}/content"

        try:
            response = await self._session.get(
                url,
                headers=self._get_headers(),
                follow_redirects=True,
            )

            if response.status_code != 200:
                error_detail = self._parse_error(response)
                raise SharePointError(
                    "download",
                    error_detail,
                    status_code=response.status_code,
                    request_id=response.headers.get("request-id"),
                )

            content = response.content
            if not content:
                raise SharePointError("download", "Empty response from Graph API")

            return bytes(content)

        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("download", str(e)) from e

    async def get_metadata(self, document_id: str) -> dict[str, Any]:
        """Get document metadata by drive item ID.

        Args:
            document_id: Graph API drive item ID

        Returns:
            Metadata dictionary
        """
        if self._session is None:
            raise SharePointError("get_metadata", "Client not initialized")

        await self._ensure_token()

        prefix = self._drive_prefix()
        url = f"{prefix}/items/{document_id}"

        try:
            response = await self._session.get(
                url,
                headers=self._get_headers(),
            )

            if response.status_code != 200:
                error_detail = self._parse_error(response)
                raise SharePointError(
                    "get_metadata",
                    error_detail,
                    status_code=response.status_code,
                    request_id=response.headers.get("request-id"),
                )

            item = response.json()

            return {
                "id": document_id,
                "name": item.get("name"),
                "size": item.get("size"),
                "created_at": item.get("createdDateTime"),
                "modified_at": item.get("lastModifiedDateTime"),
                "web_url": item.get("webUrl"),
                "etag": item.get("eTag"),
                "content_type": (
                    item.get("file", {}).get("mimeType", "application/pdf")
                ),
            }

        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("get_metadata", str(e)) from e

    async def health_check(self) -> dict[str, Any]:
        """Check SharePoint connector health via Graph API site lookup.

        Returns:
            Health status dictionary
        """
        if self._session is None:
            return {
                "status": "unhealthy",
                "message": "Client not initialized",
                "details": {"site_id": self.config.site_id},
            }

        try:
            await self._ensure_token()

            response = await self._session.get(
                f"/sites/{self.config.site_id}",
                headers=self._get_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "message": "Connected to SharePoint via Graph API",
                    "details": {
                        "site_id": self.config.site_id,
                        "site_name": data.get("displayName"),
                        "web_url": data.get("webUrl"),
                    },
                }
            else:
                return {
                    "status": "degraded",
                    "message": f"Graph API returned {response.status_code}",
                    "details": {"site_id": self.config.site_id},
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": str(e),
                "details": {"site_id": self.config.site_id},
            }

    def _parse_error(self, response: httpx.Response) -> str:
        """Parse error details from a Graph API response."""
        try:
            data = response.json()
            error = data.get("error", {})
            return error.get("message", str(data))  # type: ignore[no-any-return]
        except Exception:
            return str(response.text)
