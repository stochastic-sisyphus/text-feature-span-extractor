"""Tests for Ledger implementations (MockLedger + DataverseLedger)."""

import asyncio
from pathlib import Path

import pytest

from invoices.azure.ledger import DataverseLedger
from invoices.mocks.dataverse import MockDataverseConnector
from invoices.mocks.ledger import MockLedger


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> MockLedger:
    """Create a MockLedger with temp directory."""
    return MockLedger(data_dir=str(tmp_path / "ledger"))


@pytest.fixture
def mock_store(tmp_path: Path) -> MockDataverseConnector:
    """Create a MockDataverseConnector for DataverseLedger tests."""
    from invoices.mocks.dataverse import MockDataverseConfig

    config = MockDataverseConfig(data_dir=str(tmp_path / "dataverse"))
    return MockDataverseConnector(config)


# =============================================================================
# MockLedger Tests
# =============================================================================


class TestMockLedger:
    def test_mark_failed_increments_retry(self, tmp_ledger: MockLedger) -> None:
        asyncio.run(self._mark_failed_increments(tmp_ledger))

    async def _mark_failed_increments(self, ledger: MockLedger) -> None:
        async with ledger:
            e1 = await ledger.mark_failed("abc", "fs:abc", "error 1")
            assert e1.retry_count == 1

            e2 = await ledger.mark_failed("abc", "fs:abc", "error 2")
            assert e2.retry_count == 2


# =============================================================================
# DataverseLedger Tests
# =============================================================================


class TestDataverseLedger:
    def test_mark_processed_and_get(self, mock_store: MockDataverseConnector) -> None:
        asyncio.run(self._mark_processed(mock_store))

    async def _mark_processed(self, store: MockDataverseConnector) -> None:
        async with store:
            async with DataverseLedger(store) as ledger:
                entry = await ledger.mark_processed(
                    sha256="dv_test",
                    doc_id="fs:dv_test",
                    extraction_version="v1",
                )
                assert entry.status == "processed"
                assert entry.sha256 == "dv_test"

                retrieved = await ledger.get("dv_test")
                assert retrieved is not None
                assert retrieved.status == "processed"

    def test_mark_failed_and_list(self, mock_store: MockDataverseConnector) -> None:
        asyncio.run(self._mark_failed(mock_store))

    async def _mark_failed(self, store: MockDataverseConnector) -> None:
        async with store:
            async with DataverseLedger(store) as ledger:
                await ledger.mark_failed("f1", "fs:f1", "boom")
                await ledger.mark_processed("ok", "fs:ok")

                failed = await ledger.list_failed()
                assert len(failed) == 1
                assert failed[0].sha256 == "f1"
                assert failed[0].error_message == "boom"

    def test_retry_count_increments(self, mock_store: MockDataverseConnector) -> None:
        asyncio.run(self._retry_increments(mock_store))

    async def _retry_increments(self, store: MockDataverseConnector) -> None:
        async with store:
            async with DataverseLedger(store) as ledger:
                e1 = await ledger.mark_failed("r1", "fs:r1", "err1")
                assert e1.retry_count == 1

                e2 = await ledger.mark_failed("r1", "fs:r1", "err2")
                assert e2.retry_count == 2
