"""Base protocols and data models for Azure service abstractions.

This module defines the protocol interfaces that Azure connectors must implement,
enabling swappable implementations for testing (mocks) and production (real Azure).

Usage:
    from invoices.azure.base import DocumentSource, DataStore, Ledger

    class MyConnector(DocumentSource):
        async def list_documents(self, folder: str | None = None) -> list[Document]:
            ...
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable


class ConnectorMode(str, Enum):
    """Operating mode for Azure connectors.

    Determines which implementation to use:
    - LOCAL: Use mock implementations with local filesystem
    - AZURE: Use real Azure services (requires credentials)
    - HYBRID: Use Azure for some services, local for others
    """

    LOCAL = "local"
    AZURE = "azure"
    HYBRID = "hybrid"


@dataclass
class Document:
    """Represents a document from a document source.

    Attributes:
        id: Unique identifier from the source system (e.g., SharePoint ID)
        name: Human-readable filename
        content_type: MIME type (e.g., "application/pdf")
        size_bytes: File size in bytes
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        sha256: Content hash for deduplication (computed on download)
        source_url: URL to access the document in the source system
        metadata: Additional source-specific metadata
    """

    id: str
    name: str
    content_type: str
    size_bytes: int
    created_at: datetime
    modified_at: datetime
    sha256: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LedgerEntry:
    """Represents an entry in the processing ledger.

    Tracks document processing status for deduplication and recovery.

    Attributes:
        sha256: Content hash (primary key for deduplication)
        source_id: ID from the source system (e.g., SharePoint ID)
        doc_id: Internal document ID (fs:{sha256[:16]})
        processed_at: When processing completed
        extraction_version: Version of the extraction pipeline used
        status: Processing status
        dataverse_id: ID in Dataverse staging table (if written)
        error_message: Error message if processing failed
        retry_count: Number of retry attempts
    """

    sha256: str
    source_id: str | None
    doc_id: str
    processed_at: datetime
    extraction_version: str
    status: Literal["processed", "failed", "skipped", "pending"]
    dataverse_id: str | None = None
    error_message: str | None = None
    retry_count: int = 0


# Type variable for generic record types
T = TypeVar("T")


@runtime_checkable
class DocumentSource(Protocol):
    """Protocol for document source connectors (e.g., SharePoint).

    Implementations must provide methods for listing, downloading,
    and accessing document metadata from a document source.
    """

    async def list_documents(
        self,
        folder: str | None = None,
        modified_since: datetime | None = None,
    ) -> list[Document]:
        """List documents in a folder.

        Args:
            folder: Folder path to list (e.g., "Invoices/Inbox")
            modified_since: Only return documents modified after this time

        Returns:
            List of Document objects
        """
        ...

    async def download(self, document_id: str) -> bytes:
        """Download document content.

        Args:
            document_id: Document identifier from list_documents

        Returns:
            Raw document bytes
        """
        ...

    async def get_metadata(self, document_id: str) -> dict[str, Any]:
        """Get document metadata.

        Args:
            document_id: Document identifier

        Returns:
            Metadata dictionary
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check connector health.

        Returns:
            Health status dictionary with keys:
            - status: "healthy", "degraded", or "unhealthy"
            - message: Human-readable status message
            - details: Additional diagnostic info
        """
        ...


@runtime_checkable
class DataStore(Protocol):
    """Protocol for data store connectors (e.g., Dataverse).

    Implementations must provide CRUD operations for structured data.
    """

    async def create(self, table: str, record: dict[str, Any]) -> str:
        """Create a new record.

        Args:
            table: Table name (e.g., "new_tem")
            record: Record data

        Returns:
            ID of the created record
        """
        ...

    async def get(self, table: str, record_id: str) -> dict[str, Any] | None:
        """Get a record by ID.

        Args:
            table: Table name
            record_id: Record identifier

        Returns:
            Record data or None if not found
        """
        ...

    async def update(
        self,
        table: str,
        record_id: str,
        fields: dict[str, Any],
    ) -> bool:
        """Update a record.

        Args:
            table: Table name
            record_id: Record identifier
            fields: Fields to update

        Returns:
            True if record was updated, False if not found
        """
        ...

    async def query(
        self,
        table: str,
        filter_expr: str | None = None,
        order_by: str | None = None,
        top: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query records.

        Args:
            table: Table name
            filter_expr: OData-style filter expression
            order_by: Order by expression
            top: Maximum records to return

        Returns:
            List of matching records
        """
        ...

    async def delete(self, table: str, record_id: str) -> bool:
        """Delete a record.

        Args:
            table: Table name
            record_id: Record identifier

        Returns:
            True if record was deleted, False if not found
        """
        ...

    async def health_check(self) -> dict[str, Any]:
        """Check connector health.

        Returns:
            Health status dictionary
        """
        ...


@runtime_checkable
class Ledger(Protocol):
    """Protocol for processing ledger.

    Tracks which documents have been processed for deduplication
    and recovery purposes.
    """

    async def exists(self, sha256: str) -> bool:
        """Check if a document has been processed.

        Args:
            sha256: Content hash

        Returns:
            True if document exists in ledger
        """
        ...

    async def get(self, sha256: str) -> LedgerEntry | None:
        """Get ledger entry for a document.

        Args:
            sha256: Content hash

        Returns:
            LedgerEntry or None if not found
        """
        ...

    async def mark_processed(
        self,
        sha256: str,
        doc_id: str,
        source_id: str | None = None,
        dataverse_id: str | None = None,
        extraction_version: str = "unknown",
    ) -> LedgerEntry:
        """Mark a document as successfully processed.

        Args:
            sha256: Content hash
            doc_id: Internal document ID
            source_id: Source system ID
            dataverse_id: Dataverse record ID
            extraction_version: Pipeline version used

        Returns:
            Created LedgerEntry
        """
        ...

    async def mark_failed(
        self,
        sha256: str,
        doc_id: str,
        error_message: str,
        source_id: str | None = None,
        extraction_version: str = "unknown",
    ) -> LedgerEntry:
        """Mark a document as failed processing.

        Args:
            sha256: Content hash
            doc_id: Internal document ID
            error_message: Error description
            source_id: Source system ID
            extraction_version: Pipeline version used

        Returns:
            Created/updated LedgerEntry
        """
        ...

    async def list_pending(
        self,
        limit: int | None = None,
    ) -> list[LedgerEntry]:
        """List documents pending processing or retry.

        Args:
            limit: Maximum entries to return

        Returns:
            List of pending LedgerEntry objects
        """
        ...

    async def list_failed(
        self,
        limit: int | None = None,
    ) -> list[LedgerEntry]:
        """List documents that failed processing.

        Args:
            limit: Maximum entries to return

        Returns:
            List of failed LedgerEntry objects
        """
        ...
