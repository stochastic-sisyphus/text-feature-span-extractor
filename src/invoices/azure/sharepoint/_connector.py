"""SharePoint document library connector via Microsoft Graph API."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, cast
from urllib.parse import quote

if TYPE_CHECKING:
    from azure.identity.aio import (
        ClientSecretCredential,
        DefaultAzureCredential,
        ManagedIdentityCredential,
    )
    from tenacity import RetryCallState

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from invoices.azure.config import SharePointConfig
from invoices.config import Config
from invoices.exceptions import ConfigurationError
from invoices.ingest.base import Document, DocumentSource
from invoices.logging import get_logger

from ._constants import (
    GRAPH_BASE_URL,
    GRAPH_SCOPE,
)
from ._errors import SharePointError
from ._types import RemovedDocument

logger = get_logger(__name__)


def _sharepoint_list_documents_retry_log(retry_state: RetryCallState) -> None:
    """Log each retry attempt via structlog before sleeping."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "sharepoint_list_documents_retry",
        attempt=retry_state.attempt_number,
        next_wait=retry_state.next_action.sleep if retry_state.next_action else None,
        exc_class=type(exc).__name__ if exc else None,
        exc_msg=str(exc)[:500] if exc else None,
    )


class SharePointConnector(DocumentSource):
    """SharePoint document library connector using Microsoft Graph API.

    Implements the DocumentSource protocol for accessing SharePoint
    document libraries. Read-only: files are never deleted or modified.

    Auth: When client_secret is set, uses ClientSecretCredential (service
    principal). Otherwise, falls back to DefaultAzureCredential (managed
    identity in Azure, CLI/env credentials locally).

    Token caching is handled internally by azure-identity; concurrent
    get_token() calls are serialized by the SDK and the cached token is
    reused until near expiry.

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
        try:
            from azure.identity.aio import (  # type: ignore[import-not-found]
                ClientSecretCredential,
                DefaultAzureCredential,
                ManagedIdentityCredential,
            )
        except ImportError as e:
            raise SharePointError(
                "auth",
                "azure-identity not installed. Run: pip install azure-identity",
            ) from e

        self.config = config
        self._session: httpx.AsyncClient | None = None
        self._folder_id: str | None = None

        if config.client_secret:
            if not config.tenant_id:
                raise ConfigurationError(
                    "Missing required configuration: AZURE_TENANT_ID"
                )
            if not config.client_id:
                raise ConfigurationError(
                    "Missing required configuration: AZURE_CLIENT_ID"
                )
            self._credential: (
                ClientSecretCredential
                | ManagedIdentityCredential
                | DefaultAzureCredential
            ) = ClientSecretCredential(
                tenant_id=config.tenant_id,
                client_id=config.client_id,
                client_secret=config.client_secret,
            )
        elif config.managed_identity_client_id:
            # User-assigned managed identity on the VM. Use ManagedIdentityCredential
            # DIRECTLY, not DefaultAzureCredential: the chain probes WorkloadIdentity
            # first (fails — no token_file_path on a VM) and its IMDS availability
            # pre-check times out fast ("no response from the IMDS endpoint"), failing
            # the whole chain even when the assigned identity is reachable.
            self._credential = ManagedIdentityCredential(
                client_id=config.managed_identity_client_id
            )
        else:
            self._credential = DefaultAzureCredential()
        # NOTE: credential lifetime = connector lifetime. Closed via aclose().

    async def __aenter__(self) -> SharePointConnector:
        """Enter async context and initialize client."""
        await self._init_client()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context and cleanup."""
        await self._cleanup()

    async def aclose(self) -> None:
        """Close the connector and release all resources."""
        await self._cleanup()

    async def _init_client(self) -> None:
        """Initialize the httpx session."""
        if not self.config.is_configured():
            raise SharePointError(
                "init",
                "SharePoint not properly configured. Check environment variables.",
            )

        try:
            self._session = httpx.AsyncClient(
                base_url=GRAPH_BASE_URL,
                timeout=httpx.Timeout(
                    connect=Config.sharepoint_connect_timeout_seconds,
                    read=Config.sharepoint_read_timeout_seconds,
                    write=Config.sharepoint_write_timeout_seconds,
                    pool=Config.sharepoint_pool_timeout_seconds,
                ),
            )

            # Resolve site/drive IDs if needed
            if (
                not self.config.site_id
                and self.config.hostname
                and self.config.site_path
            ):
                await self._resolve_ids()
            elif self.config.site_id and not self.config.drive_id:
                # site_id set (e.g. from env) but drive_id missing --
                # resolve drive_id so we don't fall back to /drive default
                await self._resolve_drive_id()

            await self._validate_folder()

            logger.info(
                "sharepoint_client_initialized",
                site_id=self.config.site_id,
            )
        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("init", str(e)) from e

    async def _cleanup(self) -> None:
        """Cleanup session and credential resources."""
        if self._session:
            try:
                await self._session.aclose()
            except Exception as e:
                logger.warning("sharepoint_session_cleanup_failed", error=str(e))
            self._session = None
        try:
            await self._credential.close()
        except Exception as e:
            logger.warning("sharepoint_credential_cleanup_failed", error=str(e))

    async def _get_headers(self) -> dict[str, str]:
        """Get request headers with a fresh (or cached) authorization token."""
        try:
            token = await self._credential.get_token(GRAPH_SCOPE)
        except Exception as e:
            raise SharePointError("auth", str(e)) from e
        return {
            "Authorization": f"Bearer {token.token}",
            "Accept": "application/json",
        }

    async def _resolve_ids(self) -> None:
        """Resolve site_id and drive_id from hostname + site_path via Graph API.

        Makes two Graph calls:
        1. GET /sites/{hostname}:{site_path} -> site_id
        2. GET /sites/{site_id}/drives -> drive_id (first "Documents" drive)
        """
        assert self._session is not None

        # 1. Resolve site ID
        site_url = f"/sites/{self.config.hostname}:{self.config.site_path}"
        response = await self._session.get(site_url, headers=await self._get_headers())
        if response.status_code != 200:
            raise SharePointError(
                "resolve_ids",
                f"Site lookup returned {response.status_code}: "
                f"{self._parse_error(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )
        _site_data = response.json()
        _site_id = _site_data.get("id")
        if _site_id is None:
            raise SharePointError(
                "resolve_ids",
                "Graph response missing 'id' field",
                status_code=response.status_code,
            )
        self.config.site_id = _site_id

        # 2. Resolve drive ID
        drives_url = f"/sites/{self.config.site_id}/drives"
        response = await self._session.get(
            drives_url, headers=await self._get_headers()
        )
        if response.status_code != 200:
            raise SharePointError(
                "resolve_ids",
                f"Drives lookup returned {response.status_code}: "
                f"{self._parse_error(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )
        for drive in response.json().get("value", []):
            if drive.get("driveType") == "documentLibrary":
                _drive_id = drive.get("id")
                if _drive_id is None:
                    raise SharePointError(
                        "resolve_ids",
                        "Graph response missing 'id' field on drive object",
                        status_code=response.status_code,
                    )
                self.config.drive_id = _drive_id
                break

        logger.info(
            "sharepoint_ids_resolved",
            hostname=self.config.hostname,
            site_id=self.config.site_id,
            drive_id=self.config.drive_id,
        )

    async def _resolve_drive_id(self) -> None:
        """Resolve drive_id from site_id when only site_id is known."""
        assert self._session is not None

        drives_url = f"/sites/{self.config.site_id}/drives"
        response = await self._session.get(
            drives_url, headers=await self._get_headers()
        )
        if response.status_code != 200:
            raise SharePointError(
                "resolve_drive_id",
                f"Drives lookup returned {response.status_code}: "
                f"{self._parse_error(response)}",
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )
        drives = response.json().get("value", [])
        if not drives:
            raise SharePointError(
                "resolve_drive_id",
                "No drives found on site",
                status_code=200,
            )

        # Use the first documentLibrary drive
        for drive in drives:
            if drive.get("driveType") == "documentLibrary":
                _drive_id = drive.get("id")
                if _drive_id is None:
                    raise SharePointError(
                        "resolve_drive_id",
                        "Graph response missing 'id' field on drive object",
                        status_code=200,
                    )
                self.config.drive_id = _drive_id
                break
        else:
            # Fallback: just take the first drive
            _fallback_id = drives[0].get("id")
            if _fallback_id is None:
                raise SharePointError(
                    "resolve_drive_id",
                    "Graph response missing 'id' field on fallback drive object",
                    status_code=200,
                )
            self.config.drive_id = _fallback_id

        logger.info(
            "sharepoint_drive_resolved",
            site_id=self.config.site_id,
            drive_id=self.config.drive_id,
            drive_name=next(
                (d.get("name") for d in drives if d.get("id") == self.config.drive_id),
                None,
            ),
        )

    async def _ensure_drive_id(self) -> None:
        """Resolve drive_id if not already set."""
        if not self.config.drive_id:
            await self._resolve_drive_id()

    async def _validate_folder(self) -> None:
        """Validate configured folder exists on the drive root.

        Lists drive root children and checks that the configured folder
        is among them. If not found, logs all available folders so the
        operator can correct the env var.
        """
        assert self._session is not None
        await self._ensure_drive_id()

        prefix = self._drive_prefix()
        url = f"{prefix}/root/children?$filter=folder ne null&$select=name,id,folder"

        response = await self._graph_request(
            "GET", url, headers=await self._get_headers()
        )
        if response.status_code != 200:
            raise SharePointError(
                "validate_folder",
                f"Could not list drive root: Graph returned {response.status_code}",
            )

        items = [item for item in response.json().get("value", []) if "folder" in item]
        folders = [item["name"] for item in items]

        configured = self.config.folder_path
        matched = next((item for item in items if item["name"] == configured), None)
        if matched is not None:
            self._folder_id = matched.get("id")
            logger.info(
                "sharepoint_folder_validated",
                folder=configured,
                available_count=len(folders),
            )
            return

        # Not an exact match -- log what's there
        logger.error(
            "sharepoint_folder_not_found",
            configured_folder=configured,
            available_folders=folders,
        )
        raise SharePointError(
            "validate_folder",
            f"Folder {configured!r} not found on drive. Available folders: {folders}",
        )

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

    # Statuses that warrant a retry (throttle + transient server errors)
    _RETRY_STATUSES: frozenset[int] = frozenset({429, 503, 504})
    # Max retry attempts before giving up
    _MAX_RETRIES: int = 3
    # Ceiling on Retry-After / exponential backoff waits (seconds)
    _MAX_BACKOFF_SECONDS: float = 60.0
    # Max items per page returned by the Graph search endpoint
    _SEARCH_PAGE_SIZE: int = 999

    async def _graph_request(
        self,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Send an HTTP request with retry logic for throttling and transient errors.

        Retries up to _MAX_RETRIES times on:
        - 429 Too Many Requests (respects Retry-After header)
        - 503/504 transient server errors (respects Retry-After header)
        - httpx.TimeoutException / httpx.ConnectError (network-level transients)

        Any other 4xx or 5xx is raised immediately without retry.
        """
        assert self._session is not None

        backoff = 1.0
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = await self._session.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt >= self._MAX_RETRIES:
                    raise SharePointError(
                        "request",
                        f"Network error after {self._MAX_RETRIES} retries: {exc}",
                    ) from exc
                wait = backoff
                backoff = min(backoff * 2, self._MAX_BACKOFF_SECONDS)
                logger.debug(
                    "sharepoint_retry_network",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
                continue

            if response.status_code not in self._RETRY_STATUSES:
                return response

            # Retryable status -- check if we have attempts left
            if attempt >= self._MAX_RETRIES:
                return response

            # Honour Retry-After if present, otherwise use exponential backoff
            retry_after_header = response.headers.get("Retry-After")
            if retry_after_header is not None:
                try:
                    raw = float(retry_after_header)
                    if raw > self._MAX_BACKOFF_SECONDS:
                        logger.warning(
                            "sharepoint_retry_after_capped",
                            raw=retry_after_header,
                            capped=self._MAX_BACKOFF_SECONDS,
                        )
                    wait = min(raw, self._MAX_BACKOFF_SECONDS)
                except ValueError:
                    wait = backoff
            else:
                wait = backoff
            backoff = min(backoff * 2, self._MAX_BACKOFF_SECONDS)

            logger.debug(
                "sharepoint_retry_http",
                attempt=attempt + 1,
                status_code=response.status_code,
                wait_seconds=wait,
            )
            await asyncio.sleep(wait)

        # Unreachable, but satisfies mypy
        raise SharePointError("request", "Retry loop exhausted unexpectedly")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=_sharepoint_list_documents_retry_log,
        reraise=True,
    )
    async def list_documents(
        self,
        folder: str | None = None,
        modified_since: datetime | None = None,
    ) -> list[Document]:
        """List PDF documents in a SharePoint folder (recursive via Graph search).

        Uses the Graph API search endpoint to find all PDFs under the folder
        in a single paginated call -- one API call regardless of folder depth,
        eliminating per-subfolder recursion and the associated rate limiting.

        Args:
            folder: Folder path (defaults to config.folder_path)
            modified_since: Only return documents modified after this time

        Returns:
            List of Document objects
        """
        if self._session is None:
            raise SharePointError("list_documents", "Client not initialized")

        await self._ensure_drive_id()

        folder_path = (folder or self.config.folder_path).lstrip("/")
        prefix = self._drive_prefix()
        encoded_folder = quote(folder_path, safe="/")
        url = f"{prefix}/root:/{encoded_folder}:/search(q='.pdf')?$top={self._SEARCH_PAGE_SIZE}"

        try:
            documents: list[Document] = []
            # Follow pagination
            next_url: str | None = url

            while next_url is not None:
                response = await self._graph_request(
                    "GET",
                    next_url,
                    headers=await self._get_headers(),
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
                    # Search can return matching folders -- skip them
                    if "folder" in item:
                        continue

                    name: str = item.get("name", "")
                    # Search matches filenames and content -- filter to actual PDFs
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

    async def list_documents_delta(
        self,
        delta_link: str | None,
        *,
        _bootstrap_depth: int = 0,
    ) -> tuple[list[Document], list[RemovedDocument], str]:
        """List document changes since a previous delta call using Graph delta query.

        On first call (``delta_link=None``) this performs a full enumeration of
        the drive and returns the initial ``@odata.deltaLink`` alongside all
        current documents.  Subsequent calls pass the stored ``delta_link`` and
        receive only the items that changed, were added, or were removed since
        the previous call.

        Removed items carry an ``@removed`` object with a ``reason`` of
        ``'changed'`` (recycle bin) or ``'deleted'`` (permanently gone).

        HTTP 410 Gone means the delta token has expired.  This method handles
        it internally by bootstrapping once (``_bootstrap_depth`` guard prevents
        infinite recursion) and returning the fresh result together with a new
        token -- the caller never sees the 410.

        Not decorated with ``@transient_retry``: transient exhaustion would
        return a fallback ``([], [], None)`` which would signal "no changes +
        no token" to the caller and trigger a full re-bootstrap on every tick.
        Letting the exception propagate is correct -- the orchestrator's existing
        watch backoff handles it.

        Args:
            delta_link: Full ``@odata.deltaLink`` URL from a previous call, or
                ``None`` for bootstrap.
            _bootstrap_depth: Internal guard -- do not pass from outside.

        Returns:
            ``(added_or_modified, removed, new_delta_link)`` where:
            - ``added_or_modified``: PDF ``Document`` objects that are new or
              updated since the previous call.
            - ``removed``: ``RemovedDocument`` objects for items deleted since
              the previous call.
            - ``new_delta_link``: The ``@odata.deltaLink`` URL to use on the
              next call.

        Raises:
            SharePointError: On non-recoverable Graph API errors.
        """
        if self._session is None:
            raise SharePointError("list_documents_delta", "Client not initialized")

        await self._ensure_drive_id()

        prefix = self._drive_prefix()
        # Bootstrap: start from drive root delta endpoint (or folder-scoped).
        # Subsequent calls: reuse the full deltaLink URL (opaque, per MS docs).
        if delta_link is not None:
            start_url: str = delta_link
        else:
            folder_path = self.config.folder_path.lstrip("/")
            if self._folder_id:
                start_url = f"{prefix}/items/{self._folder_id}/delta"
                by_id = True
            elif folder_path:
                encoded = quote(folder_path, safe="/")
                start_url = f"{prefix}/root:/{encoded}:/delta"
                by_id = False
            else:
                start_url = f"{prefix}/root/delta"
                by_id = False
            logger.info(
                "sharepoint_delta_scope",
                folder=folder_path,
                scoped=bool(folder_path),
                by_id=by_id,
            )

        try:
            added_or_modified: list[Document] = []
            removed: list[RemovedDocument] = []
            next_url: str | None = start_url
            final_delta_link: str | None = None

            while next_url is not None:
                response = await self._graph_request(
                    "GET",
                    next_url,
                    headers=await self._get_headers(),
                )

                # 410 Gone: token expired -- bootstrap once.
                if response.status_code == 410:
                    if _bootstrap_depth >= 1:
                        raise SharePointError(
                            "list_documents_delta",
                            "Delta token expired and bootstrap loop detected",
                            status_code=410,
                            request_id=response.headers.get("request-id"),
                        )
                    logger.warning(
                        "sharepoint_delta_token_expired",
                        reason="HTTP 410 -- bootstrapping fresh delta",
                    )
                    return await self.list_documents_delta(
                        None,
                        _bootstrap_depth=_bootstrap_depth + 1,
                    )

                if response.status_code != 200:
                    raise SharePointError(
                        "list_documents_delta",
                        self._parse_error(response),
                        status_code=response.status_code,
                        request_id=response.headers.get("request-id"),
                    )

                data = response.json()

                for item in data.get("value", []):
                    item_id: str = item.get("id", "")

                    # Removed item -- surfaces as @removed: {reason: ...}
                    removed_meta = item.get("@removed")
                    if removed_meta is not None:
                        raw_reason = removed_meta.get("reason", "deleted")
                        reason: Literal["changed", "deleted"] = (
                            "changed" if raw_reason == "changed" else "deleted"
                        )
                        removed.append(RemovedDocument(id=item_id, reason=reason))
                        continue

                    # Skip folders and non-PDF files.
                    if "folder" in item:
                        continue
                    name: str = item.get("name", "")
                    if not name.lower().endswith(".pdf"):
                        continue

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
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    if modified_at.tzinfo is None:
                        modified_at = modified_at.replace(tzinfo=timezone.utc)

                    web_url = item.get("webUrl", "")
                    parent_ref = item.get("parentReference", {})
                    parent_folder = parent_ref.get("path", "")

                    doc = Document(
                        id=item_id,
                        name=name,
                        content_type="application/pdf",
                        size_bytes=int(item.get("size", 0)),
                        created_at=created_at,
                        modified_at=modified_at,
                        source_url=web_url,
                        metadata={
                            "drive_item_id": item_id,
                            "parent_folder": parent_folder,
                            "web_url": web_url,
                            "etag": item.get("eTag", ""),
                        },
                    )
                    added_or_modified.append(doc)

                # Pagination: follow nextLink until deltaLink appears on final page.
                next_url = data.get("@odata.nextLink")
                if "@odata.deltaLink" in data:
                    final_delta_link = data["@odata.deltaLink"]

            if final_delta_link is None:
                raise SharePointError(
                    "list_documents_delta",
                    "Graph did not return @odata.deltaLink on any page -- "
                    "response may be malformed",
                )

            logger.info(
                "sharepoint_delta_complete",
                added_or_modified=len(added_or_modified),
                removed=len(removed),
                bootstrap=delta_link is None,
            )
            return added_or_modified, removed, final_delta_link

        except SharePointError:
            raise
        except Exception as e:
            raise SharePointError("list_documents_delta", str(e)) from e

    async def download(self, document_id: str) -> bytes:
        """Download document content by drive item ID.

        Args:
            document_id: Graph API drive item ID

        Returns:
            Raw document bytes
        """
        if self._session is None:
            raise SharePointError("download", "Client not initialized")

        await self._ensure_drive_id()

        prefix = self._drive_prefix()
        url = f"{prefix}/items/{document_id}/content"

        backoff = 1.0
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                async with self._session.stream(
                    "GET",
                    url,
                    headers=await self._get_headers(),
                    follow_redirects=True,
                ) as response:
                    if (
                        response.status_code in self._RETRY_STATUSES
                        and attempt < self._MAX_RETRIES
                    ):
                        retry_after_header = response.headers.get("Retry-After")
                        if retry_after_header is not None:
                            try:
                                raw = float(retry_after_header)
                                if raw > self._MAX_BACKOFF_SECONDS:
                                    logger.warning(
                                        "sharepoint_retry_after_capped",
                                        raw=retry_after_header,
                                        capped=self._MAX_BACKOFF_SECONDS,
                                    )
                                wait = min(raw, self._MAX_BACKOFF_SECONDS)
                            except ValueError:
                                wait = backoff
                        else:
                            wait = backoff
                        backoff = min(backoff * 2, self._MAX_BACKOFF_SECONDS)
                        logger.debug(
                            "sharepoint_retry_http",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            wait_seconds=wait,
                        )
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code != 200:
                        await response.aread()
                        error_detail = self._parse_error(response)
                        raise SharePointError(
                            "download",
                            error_detail,
                            status_code=response.status_code,
                            request_id=response.headers.get("request-id"),
                        )

                    chunks: list[bytes] = []
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        chunks.append(chunk)  # noqa: PERF401 -- async comprehension would drop the explicit list[bytes] annotation and clarity

                content = b"".join(chunks)
                if not content:
                    raise SharePointError("download", "Empty response from Graph API")
                return content

            except SharePointError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                if attempt >= self._MAX_RETRIES:
                    break
                wait = backoff
                backoff = min(backoff * 2, self._MAX_BACKOFF_SECONDS)
                logger.debug(
                    "sharepoint_retry_network",
                    attempt=attempt + 1,
                    wait_seconds=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)
            except Exception as e:
                raise SharePointError("download", str(e)) from e

        raise SharePointError(
            "download",
            f"Network error after {self._MAX_RETRIES} retries: {last_exc}",
        )

    async def get_metadata(self, document_id: str) -> dict[str, Any]:
        """Fetch DriveItem metadata (name, createdDateTime, ...) by drive item ID."""
        if self._session is None:
            raise SharePointError("get_metadata", "Client not initialized")

        await self._ensure_drive_id()

        url = f"{self._drive_prefix()}/items/{document_id}"
        response = await self._graph_request(
            "GET",
            url,
            headers=await self._get_headers(),
        )

        if response.status_code != 200:
            error_detail = self._parse_error(response)
            raise SharePointError(
                "get_metadata",
                error_detail,
                status_code=response.status_code,
                request_id=response.headers.get("request-id"),
            )

        return cast(dict[str, Any], response.json())

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
            response = await self._session.get(
                f"/sites/{self.config.site_id}",
                headers=await self._get_headers(),
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
