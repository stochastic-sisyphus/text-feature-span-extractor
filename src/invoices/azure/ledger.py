"""Dataverse-backed Ledger implementation.

Uses the DataStore protocol to persist ledger entries in a Dataverse table,
enabling deduplication and retry tracking in production Azure deployments.

Usage:
    from invoices.azure.ledger import DataverseLedger

    store = DataverseConnector(config)
    async with DataverseLedger(store, table="invoicex_ledger") as ledger:
        if not await ledger.exists(sha256):
            await ledger.mark_processed(sha256, doc_id)
"""

from datetime import datetime, timezone
from typing import Any

from invoices.azure.base import DataStore, Ledger, LedgerEntry
from invoices.logging import get_logger

logger = get_logger(__name__)


class DataverseLedger(Ledger):
    """Ledger backed by a Dataverse table via the DataStore protocol.

    Each ledger entry is a row in the configured Dataverse table,
    keyed by sha256 for dedup lookups. Uses OData filter queries
    to find entries by status or hash.
    """

    def __init__(
        self,
        store: DataStore,
        table: str = "invoicex_ledger",
    ) -> None:
        self._store = store
        self._table = table

    async def __aenter__(self) -> "DataverseLedger":
        logger.info("dataverse_ledger_initialized", table=self._table)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def _to_entry(self, record: dict[str, Any]) -> LedgerEntry:
        """Convert a Dataverse record to LedgerEntry."""
        processed_at = record.get("processed_at", "")
        if isinstance(processed_at, str) and processed_at:
            processed_at = datetime.fromisoformat(processed_at)
        elif not isinstance(processed_at, datetime):
            processed_at = datetime.now(timezone.utc)
        return LedgerEntry(
            sha256=record["sha256"],
            source_id=record.get("source_id"),
            doc_id=record.get("doc_id", ""),
            processed_at=processed_at,
            extraction_version=record.get("extraction_version", "unknown"),
            status=record.get("status", "pending"),
            dataverse_id=record.get("id"),
            error_message=record.get("error_message"),
            retry_count=record.get("retry_count", 0),
        )

    async def exists(self, sha256: str) -> bool:
        results = await self._store.query(
            self._table,
            filter_expr=f"sha256 eq '{sha256}'",
            top=1,
        )
        return len(results) > 0

    async def get(self, sha256: str) -> LedgerEntry | None:
        results = await self._store.query(
            self._table,
            filter_expr=f"sha256 eq '{sha256}'",
            top=1,
        )
        if not results:
            return None
        return self._to_entry(results[0])

    async def mark_processed(
        self,
        sha256: str,
        doc_id: str,
        source_id: str | None = None,
        dataverse_id: str | None = None,
        extraction_version: str = "unknown",
    ) -> LedgerEntry:
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.get(sha256)

        record: dict[str, Any] = {
            "sha256": sha256,
            "source_id": source_id,
            "doc_id": doc_id,
            "processed_at": now,
            "extraction_version": extraction_version,
            "status": "processed",
            "dataverse_id": dataverse_id,
            "retry_count": existing.retry_count if existing else 0,
        }

        if existing and existing.dataverse_id:
            await self._store.update(self._table, existing.dataverse_id, record)
            record["id"] = existing.dataverse_id
        else:
            record_id = await self._store.create(self._table, record)
            record["id"] = record_id

        logger.info("dataverse_ledger_processed", sha256=sha256[:16], doc_id=doc_id)
        return self._to_entry(record)

    async def mark_failed(
        self,
        sha256: str,
        doc_id: str,
        error_message: str,
        source_id: str | None = None,
        extraction_version: str = "unknown",
    ) -> LedgerEntry:
        now = datetime.now(timezone.utc).isoformat()
        existing = await self.get(sha256)
        prev_retries = existing.retry_count if existing else 0

        record: dict[str, Any] = {
            "sha256": sha256,
            "source_id": source_id,
            "doc_id": doc_id,
            "processed_at": now,
            "extraction_version": extraction_version,
            "status": "failed",
            "error_message": error_message,
            "retry_count": prev_retries + 1,
        }

        if existing and existing.dataverse_id:
            await self._store.update(self._table, existing.dataverse_id, record)
            record["id"] = existing.dataverse_id
        else:
            record_id = await self._store.create(self._table, record)
            record["id"] = record_id

        logger.info(
            "dataverse_ledger_failed",
            sha256=sha256[:16],
            doc_id=doc_id,
            retry_count=record["retry_count"],
        )
        return self._to_entry(record)

    async def list_pending(
        self,
        limit: int | None = None,
    ) -> list[LedgerEntry]:
        results = await self._store.query(
            self._table,
            filter_expr="status eq 'pending'",
            top=limit,
        )
        return [self._to_entry(r) for r in results]

    async def list_failed(
        self,
        limit: int | None = None,
    ) -> list[LedgerEntry]:
        results = await self._store.query(
            self._table,
            filter_expr="status eq 'failed'",
            top=limit,
        )
        return [self._to_entry(r) for r in results]
