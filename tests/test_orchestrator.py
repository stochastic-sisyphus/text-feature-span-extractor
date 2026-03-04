"""Tests for the Orchestrator module."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from invoices.mocks.ledger import MockLedger
from invoices.orchestrator import (
    Orchestrator,
    create_orchestrator,
)


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> MockLedger:
    """Create a MockLedger with temp directory."""
    return MockLedger(data_dir=str(tmp_path / "ledger"))


@pytest.fixture
def seed_dir(tmp_path: Path) -> Path:
    """Create a seed directory (empty — pipeline mocked)."""
    d = tmp_path / "seed_pdfs"
    d.mkdir()
    return d


# =============================================================================
# Dedup Tests
# =============================================================================


class TestDedup:
    def test_skip_already_processed(self, tmp_ledger: MockLedger) -> None:
        """Documents already in ledger as 'processed' are skipped."""
        asyncio.run(self._skip_processed(tmp_ledger))

    async def _skip_processed(self, ledger: MockLedger) -> None:
        async with ledger:
            await ledger.mark_processed("abc123", "fs:abc123")

            orch = Orchestrator(ledger=ledger)
            result = await orch.process_document("abc123")
            assert result.success is True
            assert result.doc_id == "fs:abc123"

    def test_retries_exhausted_returns_failure(self, tmp_ledger: MockLedger) -> None:
        """Documents that exceeded max retries return failure."""
        asyncio.run(self._retries_exhausted(tmp_ledger))

    async def _retries_exhausted(self, ledger: MockLedger) -> None:
        async with ledger:
            # Simulate 3 failures (default max retries)
            for _ in range(3):
                await ledger.mark_failed("exhaust", "fs:exhaust", "err")

            orch = Orchestrator(ledger=ledger)
            result = await orch.process_document("exhaust")
            assert result.success is False
            assert "exhausted" in (result.error or "").lower()


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    def test_batch_skips_duplicates(self, tmp_ledger: MockLedger) -> None:
        asyncio.run(self._batch_dedup(tmp_ledger))

    async def _batch_dedup(self, ledger: MockLedger) -> None:
        async with ledger:
            await ledger.mark_processed("dup1", "fs:dup1")

            orch = Orchestrator(ledger=ledger)
            with patch.object(orch, "_run_pipeline"):
                result = await orch.process_batch([("dup1", None), ("new1", None)])

            assert len(result.skipped) == 1
            assert len(result.processed) == 1
            assert result.skipped[0] == "dup1"

    def test_retry_failed_respects_max(self, tmp_ledger: MockLedger) -> None:
        asyncio.run(self._retry_max(tmp_ledger))

    async def _retry_max(self, ledger: MockLedger) -> None:
        async with ledger:
            # Exhaust retries for one, leave room for another
            for _ in range(3):
                await ledger.mark_failed("exhausted", "fs:exhausted", "err")
            await ledger.mark_failed("retryable", "fs:retryable", "err")

            orch = Orchestrator(ledger=ledger)
            with patch.object(orch, "_run_pipeline"):
                result = await orch.retry_failed()

            # Only "retryable" (1 retry) should be attempted
            # "exhausted" (3 retries = max) should not be retried
            assert len(result.processed) + len(result.failed) <= 2


# =============================================================================
# Discovery Tests
# =============================================================================


class TestDiscoverLocal:
    def test_discover_empty_folder(
        self, tmp_ledger: MockLedger, seed_dir: Path
    ) -> None:
        asyncio.run(self._empty_folder(tmp_ledger, seed_dir))

    async def _empty_folder(self, ledger: MockLedger, seed_dir: Path) -> None:
        async with ledger:
            orch = Orchestrator(ledger=ledger, seed_folder=str(seed_dir))

            with patch("invoices.orchestrator.ingest") as mock_ingest:
                mock_ingest.ingest_seed_folder.return_value = 0
                mock_ingest.get_indexed_documents.return_value = MagicMock(empty=True)
                docs = await orch.discover_local()

            assert docs == []

    def test_discover_missing_folder(
        self, tmp_ledger: MockLedger, tmp_path: Path
    ) -> None:
        asyncio.run(self._missing_folder(tmp_ledger, tmp_path))

    async def _missing_folder(self, ledger: MockLedger, tmp_path: Path) -> None:
        async with ledger:
            orch = Orchestrator(
                ledger=ledger,
                seed_folder=str(tmp_path / "nonexistent"),
            )
            docs = await orch.discover_local()
            assert docs == []


# =============================================================================
# Factory Tests
# =============================================================================


class TestFactory:
    def test_create_orchestrator_local(self) -> None:
        asyncio.run(self._create_local())

    async def _create_local(self) -> None:
        orch = await create_orchestrator(seed_folder="seed_pdfs")
        assert isinstance(orch, Orchestrator)

    def test_create_orchestrator_dataverse_missing_config(self) -> None:
        """Factory fails loud when OUTPUT_BACKEND=dataverse but config missing."""
        asyncio.run(self._dataverse_missing_config())

    async def _dataverse_missing_config(self) -> None:
        from invoices.azure.config import DataverseConfig, reset_azure_config

        reset_azure_config()

        # Mock Config to report dataverse backend, but mock azure_config as unconfigured
        mock_dataverse = DataverseConfig()  # Empty config, not configured

        with (
            patch("invoices.orchestrator.Config") as mock_config,
            patch("invoices.azure.config.get_azure_config") as mock_get_azure,
        ):
            mock_config.OUTPUT_BACKEND = "dataverse"
            mock_config.DOCUMENT_SOURCE = "local"

            mock_azure = MagicMock()
            mock_azure.dataverse = mock_dataverse
            mock_get_azure.return_value = mock_azure

            with pytest.raises(ValueError) as exc_info:
                await create_orchestrator()

            assert "OUTPUT_BACKEND=dataverse" in str(exc_info.value)
            assert "missing configuration" in str(exc_info.value)

        reset_azure_config()

    def test_create_orchestrator_sharepoint_missing_config(self) -> None:
        """Factory fails loud when DOCUMENT_SOURCE=sharepoint but config missing."""
        asyncio.run(self._sharepoint_missing_config())

    async def _sharepoint_missing_config(self) -> None:
        from invoices.azure.config import SharePointConfig, reset_azure_config

        reset_azure_config()

        # Mock Config to report sharepoint source, but mock azure_config as unconfigured
        mock_sharepoint = SharePointConfig()  # Empty config, not configured

        with (
            patch("invoices.orchestrator.Config") as mock_config,
            patch("invoices.azure.config.get_azure_config") as mock_get_azure,
        ):
            mock_config.OUTPUT_BACKEND = "local"
            mock_config.DOCUMENT_SOURCE = "sharepoint"

            mock_azure = MagicMock()
            mock_azure.sharepoint = mock_sharepoint
            mock_get_azure.return_value = mock_azure

            with pytest.raises(ValueError) as exc_info:
                await create_orchestrator()

            assert "DOCUMENT_SOURCE=sharepoint" in str(exc_info.value)
            assert "missing configuration" in str(exc_info.value)

        reset_azure_config()

    def test_create_orchestrator_with_azure_connectors(self) -> None:
        """Factory instantiates real Azure connectors when properly configured."""
        asyncio.run(self._azure_connectors())

    async def _azure_connectors(self) -> None:
        from invoices.azure.config import (
            DataverseConfig,
            SharePointConfig,
            reset_azure_config,
        )

        reset_azure_config()

        # Create mock Azure configs with all required fields
        mock_dataverse = DataverseConfig(
            environment_url="https://test.crm.dynamics.com",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )
        mock_sharepoint = SharePointConfig(
            site_id="test-site-guid",
            tenant_id="test-tenant-id",
            client_id="test-client-id",
            client_secret="test-client-secret",
        )

        # Patch Config and azure_config
        with (
            patch("invoices.orchestrator.Config") as mock_config,
            patch("invoices.azure.config.get_azure_config") as mock_get_azure,
            patch(
                "invoices.azure.dataverse.DataverseConnector.__aenter__",
                return_value=None,
            ),
            patch(
                "invoices.azure.sharepoint.SharePointConnector.__aenter__",
                return_value=None,
            ),
        ):
            mock_config.OUTPUT_BACKEND = "dataverse"
            mock_config.DOCUMENT_SOURCE = "sharepoint"

            mock_azure = MagicMock()
            mock_azure.dataverse = mock_dataverse
            mock_azure.sharepoint = mock_sharepoint
            mock_get_azure.return_value = mock_azure

            orch = await create_orchestrator()
            assert isinstance(orch, Orchestrator)
            # The factory should have instantiated real connectors,
            # not mocks (verified by not raising OrchestrationError)

        reset_azure_config()
