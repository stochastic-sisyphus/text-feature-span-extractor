"""Batch processing orchestrator for the invoice extraction pipeline.

Discovers new documents, deduplicates via Ledger, routes through the
pipeline, writes results, and handles retries. Works in both local
(mock) and Azure (real) modes via env vars.

Usage:
    from invoices.orchestrator import create_orchestrator

    orchestrator = await create_orchestrator()
    async with orchestrator:
        await orchestrator.run_once()       # one-shot
        await orchestrator.watch()          # continuous polling
        await orchestrator.retry_failed()   # retry failures only
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import candidates, decoder, emit, ingest, paths, tokenize
from .azure.base import DataStore, DocumentSource, Ledger
from .config import Config
from .exceptions import OrchestrationError
from .logging import get_logger
from .metrics import pipeline_duration

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single document."""

    sha256: str
    doc_id: str
    success: bool
    error: str | None = None
    dataverse_id: str | None = None


@dataclass
class BatchResult:
    """Result of processing a batch of documents."""

    processed: list[ProcessingResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[ProcessingResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.processed) + len(self.skipped) + len(self.failed)


class Orchestrator:
    """Core orchestrator for document processing.

    Manages the full lifecycle: discovery → dedup → pipeline → emit → track.
    Supports both local filesystem and Azure backends via protocol abstractions.
    """

    def __init__(
        self,
        ledger: Ledger,
        document_source: DocumentSource | None = None,
        data_store: DataStore | None = None,
        seed_folder: str | None = None,
    ) -> None:
        self._ledger = ledger
        self._document_source = document_source
        self._data_store = data_store
        self._seed_folder = seed_folder or Config.ORCHESTRATOR_SEED_FOLDER

    async def __aenter__(self) -> "Orchestrator":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    # =========================================================================
    # Document Processing
    # =========================================================================

    async def process_document(
        self,
        sha256: str,
        source_id: str | None = None,
    ) -> ProcessingResult:
        """Process a single document through the full pipeline.

        Checks the Ledger for deduplication, runs the pipeline stages,
        writes results to the DataStore if configured, and updates the Ledger.

        Args:
            sha256: Content hash of the document
            source_id: Optional source system ID (e.g., SharePoint)

        Returns:
            ProcessingResult with success/failure details
        """
        doc_id = f"fs:{sha256[:16]}"

        # Check dedup
        entry = await self._ledger.get(sha256)
        if entry and entry.status == "processed":
            logger.info("orchestrator_skip_duplicate", sha256=sha256[:16])
            return ProcessingResult(sha256=sha256, doc_id=doc_id, success=True)

        # Check retry exhaustion
        if entry and entry.retry_count >= Config.ORCHESTRATOR_MAX_RETRIES:
            logger.warning(
                "orchestrator_retries_exhausted",
                sha256=sha256[:16],
                retry_count=entry.retry_count,
            )
            return ProcessingResult(
                sha256=sha256,
                doc_id=doc_id,
                success=False,
                error=f"Max retries ({Config.ORCHESTRATOR_MAX_RETRIES}) exhausted",
            )

        # Apply exponential backoff on retry
        if entry and entry.retry_count > 0:
            delay = Config.ORCHESTRATOR_RETRY_BASE_SECONDS * (
                Config.ORCHESTRATOR_RETRY_MULTIPLIER ** (entry.retry_count - 1)
            )
            logger.info(
                "orchestrator_retry_backoff",
                sha256=sha256[:16],
                attempt=entry.retry_count + 1,
                delay_seconds=delay,
            )
            await asyncio.sleep(delay)

        # Run pipeline stages
        try:
            self._run_pipeline(sha256)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(
                "orchestrator_pipeline_failed",
                sha256=sha256[:16],
                error=error_msg,
            )
            await self._ledger.mark_failed(
                sha256=sha256,
                doc_id=doc_id,
                error_message=error_msg,
                source_id=source_id,
                extraction_version=Config.MODEL_ID,
            )
            return ProcessingResult(
                sha256=sha256, doc_id=doc_id, success=False, error=error_msg
            )

        # Write to DataStore if configured and gated
        dataverse_id = None
        if self._data_store:
            if not Config.DATAVERSE_WRITE_ENABLED:
                logger.info(
                    "orchestrator_dataverse_write_disabled",
                    sha256=sha256[:16],
                    hint="set INVOICEX_DATAVERSE_WRITE_ENABLED=true to enable",
                )
            elif self._quality_gate_blocks_write():
                logger.warning(
                    "orchestrator_dataverse_write_blocked_quality_gate",
                    sha256=sha256[:16],
                    reason="models/manifest.json has quality_gate_passed=false",
                )
            else:
                try:
                    dataverse_id = await self._write_to_datastore(
                        sha256, doc_id, source_id
                    )
                except (OSError, ValueError, ConnectionError, TimeoutError) as e:
                    logger.warning(
                        "orchestrator_datastore_write_failed",
                        sha256=sha256[:16],
                        error=str(e),
                    )

        # Mark as processed
        await self._ledger.mark_processed(
            sha256=sha256,
            doc_id=doc_id,
            source_id=source_id,
            dataverse_id=dataverse_id,
            extraction_version=Config.MODEL_ID,
        )

        logger.info(
            "orchestrator_document_processed", sha256=sha256[:16], doc_id=doc_id
        )
        return ProcessingResult(
            sha256=sha256,
            doc_id=doc_id,
            success=True,
            dataverse_id=dataverse_id,
        )

    def _run_pipeline(self, sha256: str) -> None:
        """Run the sync pipeline stages for a single document.

        Calls tokenize → candidates → decode → emit for one document.
        These are all synchronous (pdfplumber, pandas, scipy).
        """
        t0 = time.perf_counter()
        tokenize.tokenize_document(sha256)
        pipeline_duration.labels(stage="tokenize").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        candidates.generate_candidates(sha256)
        pipeline_duration.labels(stage="candidates").observe(time.perf_counter() - t0)

        t0 = time.perf_counter()
        assignments = decoder.decode_document(sha256)
        pipeline_duration.labels(stage="decode").observe(time.perf_counter() - t0)

        emit.emit_document(sha256, assignments)  # emit stage timed internally

    async def _write_to_datastore(
        self,
        sha256: str,
        doc_id: str,
        source_id: str | None,
    ) -> str | None:
        """Write extraction results to the DataStore."""
        if not self._data_store:
            return None

        from . import paths

        predictions_path = paths.get_predictions_path(sha256)
        if not predictions_path.exists():
            return None

        import json

        with open(predictions_path) as f:
            extraction_result = json.load(f)

        # Use create_staging_record if the store supports it (MockDataverse does)
        if hasattr(self._data_store, "create_staging_record"):
            return await self._data_store.create_staging_record(  # type: ignore[union-attr,no-any-return]
                document_id=doc_id,
                extraction_result=extraction_result,
                sharepoint_id=source_id,
            )

        # Fallback: generic create
        record = {
            "sha256": sha256,
            "doc_id": doc_id,
            "source_id": source_id,
            "extraction_result": extraction_result,
        }
        return await self._data_store.create(Config.DATAVERSE_STAGING_TABLE, record)

    @staticmethod
    def _quality_gate_blocks_write() -> bool:
        """Check if a failed quality gate should block Dataverse writes.

        Returns True (block) only when models/manifest.json exists AND
        quality_gate_passed is False.  If no manifest exists the system
        is in heuristic-only mode and writes are allowed.
        """
        import json

        manifest_path = paths.get_models_dir() / "manifest.json"
        if not manifest_path.exists():
            return False  # heuristic-only mode — writes are fine
        try:
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest.get("quality_gate_passed", False) is not True
        except (json.JSONDecodeError, OSError):
            # Corrupt manifest — err on the side of caution, block write
            return True

    # =========================================================================
    # Batch Processing
    # =========================================================================

    async def process_batch(
        self,
        documents: list[tuple[str, str | None]],
    ) -> BatchResult:
        """Process a batch of documents sequentially.

        Args:
            documents: List of (sha256, source_id) tuples

        Returns:
            BatchResult with per-document outcomes
        """
        result = BatchResult()

        for sha256, source_id in documents:
            # Check dedup before processing
            entry = await self._ledger.get(sha256)
            if entry and entry.status == "processed":
                result.skipped.append(sha256)
                continue

            pr = await self.process_document(sha256, source_id)
            if pr.success:
                result.processed.append(pr)
            else:
                result.failed.append(pr)

        logger.info(
            "orchestrator_batch_complete",
            processed=len(result.processed),
            skipped=len(result.skipped),
            failed=len(result.failed),
        )
        return result

    async def retry_failed(self) -> BatchResult:
        """Re-process documents that previously failed.

        Only retries documents under the max retry count.

        Returns:
            BatchResult with retry outcomes
        """
        failed_entries = await self._ledger.list_failed()

        retryable = [
            e for e in failed_entries if e.retry_count < Config.ORCHESTRATOR_MAX_RETRIES
        ]

        if not retryable:
            logger.info("orchestrator_no_retryable_docs")
            return BatchResult()

        logger.info("orchestrator_retrying", count=len(retryable))
        documents = [(e.sha256, e.source_id) for e in retryable]
        return await self.process_batch(documents)

    # =========================================================================
    # Document Discovery
    # =========================================================================

    async def discover_local(
        self,
        seed_folder: str | None = None,
    ) -> list[tuple[str, str | None]]:
        """Discover new documents from a local folder.

        Scans the seed folder for PDFs, computes SHA256 hashes,
        ingests new ones, and returns those not yet in the Ledger.

        Args:
            seed_folder: Override seed folder path

        Returns:
            List of (sha256, None) tuples for new documents
        """
        folder = Path(seed_folder or self._seed_folder)
        if not folder.exists():
            logger.warning("orchestrator_seed_folder_missing", path=str(folder))
            return []

        # Ingest all PDFs (idempotent, content-addressed)
        ingest.ingest_seed_folder(str(folder))

        # Get all indexed docs and filter by Ledger
        indexed = ingest.get_indexed_documents()
        if indexed.empty:
            return []

        new_docs: list[tuple[str, str | None]] = []
        for _, row in indexed.iterrows():
            sha256 = row["sha256"]
            if not await self._ledger.exists(sha256):
                new_docs.append((sha256, None))

        logger.info(
            "orchestrator_discovered_local",
            total_indexed=len(indexed),
            new_docs=len(new_docs),
        )
        return new_docs

    async def discover_remote(self) -> list[tuple[str, str | None]]:
        """Discover new documents from a remote DocumentSource.

        Lists documents from the configured source (e.g., SharePoint),
        downloads new ones, ingests them, and returns those not in the Ledger.

        Returns:
            List of (sha256, source_id) tuples for new documents
        """
        if not self._document_source:
            logger.warning("orchestrator_no_document_source")
            return []

        documents = await self._document_source.list_documents()

        new_docs: list[tuple[str, str | None]] = []
        for doc in documents:
            # Download content to compute SHA256
            content = await self._document_source.download(doc.id)
            sha256 = doc.sha256 or hashlib.sha256(content).hexdigest()

            if await self._ledger.exists(sha256):
                continue

            # Write bytes to local disk so tokenize_document() can open
            # the file with pdfplumber (it needs a real file path).
            raw_dir = paths.get_ingest_raw_dir()
            raw_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = raw_dir / f"{sha256}.pdf"
            if not pdf_path.exists():
                pdf_path.write_bytes(content)

            # Persist to durable blob storage when configured.
            # The local write above stays — pdfplumber needs a file path.
            if Config.STORAGE_BACKEND == "blob":
                from .storage import get_storage

                get_storage().write_bytes(f"raw/{sha256}.pdf", content)

            # Register the document in the index so tokenize_document()
            # can resolve get_document_info(sha256). Idempotent.
            ingest.register_document(sha256, doc.name)

            new_docs.append((sha256, doc.id))

        logger.info(
            "orchestrator_discovered_remote",
            total_listed=len(documents),
            new_docs=len(new_docs),
        )
        return new_docs

    # =========================================================================
    # Run Modes
    # =========================================================================

    async def run_once(self) -> BatchResult:
        """Discover new documents and process them (one-shot).

        Uses remote discovery if a DocumentSource is configured,
        otherwise falls back to local discovery.

        Returns:
            BatchResult from processing
        """
        if self._document_source:
            new_docs = await self.discover_remote()
        else:
            new_docs = await self.discover_local()

        batch_result = await self.process_batch(new_docs)

        # Also retry any previously failed docs
        retry_result = await self.retry_failed()
        batch_result.processed.extend(retry_result.processed)
        batch_result.failed.extend(retry_result.failed)

        return batch_result

    async def watch(self, interval: float | None = None) -> None:
        """Continuous polling loop.

        Repeatedly calls run_once() at the configured interval.
        Stops on KeyboardInterrupt (Ctrl+C).
        """
        if interval is None:
            interval = Config.ORCHESTRATOR_WATCH_INTERVAL
        logger.info("orchestrator_watch_start", interval_seconds=interval)

        try:
            while True:
                try:
                    result = await self.run_once()
                    if result.total > 0:
                        logger.info(
                            "orchestrator_watch_cycle",
                            processed=len(result.processed),
                            skipped=len(result.skipped),
                            failed=len(result.failed),
                        )
                except OrchestrationError as e:
                    logger.error("orchestrator_watch_error", error=str(e))

                await asyncio.sleep(interval)
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("orchestrator_watch_stopped")


# =============================================================================
# Factory
# =============================================================================


async def create_orchestrator(
    seed_folder: str | None = None,
) -> Orchestrator:
    """Create an Orchestrator with the right backends based on Config.

    Picks mock (local) or real (Azure) implementations based on
    INVOICEX_DOCUMENT_SOURCE and INVOICEX_OUTPUT_BACKEND env vars.

    Args:
        seed_folder: Override seed folder for local discovery

    Returns:
        Configured Orchestrator instance

    Raises:
        ValueError: If Azure backend is selected but credentials missing
    """
    from .azure.config import get_azure_config

    azure_config = get_azure_config()

    # Validate configuration early - fail loud at startup if misconfigured
    if Config.OUTPUT_BACKEND == "dataverse":
        if not azure_config.dataverse.is_configured():
            errors = azure_config.dataverse.validate()
            error_msg = f"OUTPUT_BACKEND=dataverse but missing configuration: {', '.join(errors)}"
            logger.error("orchestrator_config_error", error=error_msg)
            raise ValueError(error_msg)

    if Config.DOCUMENT_SOURCE == "sharepoint":
        if not azure_config.sharepoint.is_configured():
            errors = azure_config.sharepoint.validate()
            error_msg = f"DOCUMENT_SOURCE=sharepoint but missing configuration: {', '.join(errors)}"
            logger.error("orchestrator_config_error", error=error_msg)
            raise ValueError(error_msg)

    # Ledger: always needed
    # Type store as DataStore protocol to allow both real and mock implementations
    store: DataStore | None = None
    if Config.OUTPUT_BACKEND == "dataverse":
        from .azure.dataverse import DataverseConnector
        from .azure.ledger import DataverseLedger

        # Config validation passed, safe to use real connector
        store = DataverseConnector(azure_config.dataverse)
        await store.__aenter__()
        ledger: Ledger = DataverseLedger(store)
        logger.info(
            "orchestrator_using_real_dataverse",
            environment_url=azure_config.dataverse.environment_url,
        )
    else:
        # Local mode: use mock ledger
        from .mocks.ledger import MockLedger

        mock_ledger = MockLedger()
        await mock_ledger.__aenter__()
        ledger = mock_ledger
        logger.info("orchestrator_using_mock_ledger")

    # DocumentSource: optional, only for remote mode
    document_source: DocumentSource | None = None
    if Config.DOCUMENT_SOURCE == "sharepoint":
        from .azure.sharepoint import SharePointConnector

        # Config validation passed, safe to use real connector
        document_source = SharePointConnector(azure_config.sharepoint)
        await document_source.__aenter__()
        logger.info(
            "orchestrator_using_real_sharepoint",
            site_id=azure_config.sharepoint.site_id,
        )

    # DataStore: optional, only for dataverse output
    data_store: DataStore | None = None
    if Config.OUTPUT_BACKEND == "dataverse":
        # Reuse the same store instance from ledger setup above
        data_store = store

    return Orchestrator(
        ledger=ledger,
        document_source=document_source,
        data_store=data_store,
        seed_folder=seed_folder,
    )
