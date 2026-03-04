"""Tests for Azure connector mock implementations."""

import os
import tempfile
from pathlib import Path

import pytest

from invoices.mocks.dataverse import MockDataverseConfig, MockDataverseConnector


class TestMockDataverseConnector:
    """Tests for MockDataverseConnector."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Create a temporary directory for data files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_query_with_filter(self, temp_dir: Path) -> None:
        """Test query with filter expression."""
        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            await connector.create("test_table", {"name": "a", "status": "pending"})
            await connector.create("test_table", {"name": "b", "status": "approved"})
            await connector.create("test_table", {"name": "c", "status": "pending"})

            records = await connector.query(
                "test_table", filter_expr="status eq 'pending'"
            )
            assert len(records) == 2
            assert all(r["status"] == "pending" for r in records)

    @pytest.mark.asyncio
    async def test_create_staging_record(self, temp_dir: Path) -> None:
        """Test create_staging_record creates invoice staging record."""
        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            extraction_result = {
                "model_version": "test-v1",
                "fields": {
                    "InvoiceNumber": {
                        "status": "PREDICTED",
                        "value": "INV-001",
                        "confidence": 0.95,
                    },
                    "TotalAmount": {
                        "status": "PREDICTED",
                        "value": "1000.00",
                        "confidence": 0.90,
                    },
                },
            }

            record_id = await connector.create_staging_record(
                document_id="fs:abc123",
                extraction_result=extraction_result,
                sharepoint_id="sp:xyz",
            )

            record = await connector.get(config.staging_table, record_id)
            assert record is not None
            assert record["document_id"] == "fs:abc123"
            assert record["invoice_number"] == "INV-001"
            assert record["total_amount"] == "1000.00"
            assert record["sharepoint_id"] == "sp:xyz"
            assert record["status"] == "pending"

    @pytest.mark.asyncio
    async def test_promote_to_production(self, temp_dir: Path) -> None:
        """Test promote_to_production creates production record."""
        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            # Create staging record
            extraction_result = {
                "model_version": "test-v1",
                "fields": {
                    "InvoiceNumber": {
                        "status": "PREDICTED",
                        "value": "INV-001",
                        "confidence": 0.95,
                    },
                },
            }

            staging_id = await connector.create_staging_record(
                document_id="fs:abc123",
                extraction_result=extraction_result,
            )

            # Promote to production
            production_id = await connector.promote_to_production(
                staging_id=staging_id,
                approved_by="test_user",
            )

            # Check production record
            prod_record = await connector.get(config.production_table, production_id)
            assert prod_record is not None
            assert prod_record["staging_id"] == staging_id
            assert prod_record["invoice_number"] == "INV-001"
            assert prod_record["approved_by"] == "test_user"

            # Check staging record was updated
            staging_record = await connector.get(config.staging_table, staging_id)
            assert staging_record is not None
            assert staging_record["status"] == "approved"
            assert staging_record["reviewed_by"] == "test_user"


class TestSharePointResolver:
    """Tests for SharePointConnector._resolve_ids()."""

    @pytest.mark.asyncio
    async def test_resolve_ids_populates_config(self) -> None:
        """_resolve_ids() sets site_id and drive_id from Graph API responses."""
        from unittest.mock import AsyncMock, MagicMock

        from invoices.azure.config import SharePointConfig
        from invoices.azure.sharepoint import SharePointConnector

        config = SharePointConfig(
            hostname="contoso.sharepoint.com",
            site_path="/sites/invoices",
            tenant_id="tenant-123",
        )
        connector = SharePointConnector(config)
        connector._token = "fake-token"

        # Mock the httpx session
        mock_session = AsyncMock()
        connector._session = mock_session

        site_response = MagicMock()
        site_response.status_code = 200
        site_response.json.return_value = {
            "id": "resolved-site-id",
            "displayName": "Test Site",
        }

        drives_response = MagicMock()
        drives_response.status_code = 200
        drives_response.json.return_value = {
            "value": [
                {"id": "resolved-drive-id", "name": "Documents"},
                {"id": "other-drive", "name": "Other"},
            ]
        }

        mock_session.get.side_effect = [site_response, drives_response]

        await connector._resolve_ids()

        assert connector.config.site_id == "resolved-site-id"
        assert connector.config.drive_id == "resolved-drive-id"

    @pytest.mark.asyncio
    async def test_resolve_ids_error_on_403(self) -> None:
        """_resolve_ids() raises SharePointError on 403."""
        from unittest.mock import AsyncMock, MagicMock

        from invoices.azure.config import SharePointConfig
        from invoices.azure.sharepoint import SharePointConnector, SharePointError

        config = SharePointConfig(
            hostname="contoso.sharepoint.com",
            site_path="/sites/invoices",
            tenant_id="tenant-123",
        )
        connector = SharePointConnector(config)
        connector._token = "fake-token"

        mock_session = AsyncMock()
        connector._session = mock_session

        error_response = MagicMock()
        error_response.status_code = 403
        error_response.json.return_value = {"error": {"message": "Access denied"}}
        error_response.headers = {"request-id": "req-123"}
        mock_session.get.return_value = error_response

        with pytest.raises(SharePointError) as exc_info:
            await connector._resolve_ids()
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_resolve_ids_skipped_when_site_id_set(self) -> None:
        """_resolve_ids() is not called during _init_client() when site_id is set."""
        from unittest.mock import AsyncMock, patch

        from invoices.azure.config import SharePointConfig
        from invoices.azure.sharepoint import SharePointConnector

        config = SharePointConfig(
            site_id="already-set",
            hostname="contoso.sharepoint.com",
            site_path="/sites/invoices",
            tenant_id="tenant-123",
        )
        connector = SharePointConnector(config)

        with (
            patch.object(connector, "_acquire_token", new_callable=AsyncMock),
            patch.object(
                connector, "_resolve_ids", new_callable=AsyncMock
            ) as mock_resolve,
        ):
            connector._session = AsyncMock()
            await connector._init_client()
            mock_resolve.assert_not_called()


# ===========================================================================
# Integration Tests (require live Azure credentials)
# ===========================================================================
AZURE_INTEGRATION = os.getenv("AZURE_TEST_ENABLED", "").lower() == "true"


@pytest.mark.integration
@pytest.mark.skipif(not AZURE_INTEGRATION, reason="AZURE_TEST_ENABLED not set")
class TestDataverseConnectorIntegration:
    """Integration tests for real Dataverse connector. Requires live credentials."""

    @pytest.fixture
    def temp_dir(self) -> Path:
        """Temp dir for test isolation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_create_and_retrieve_staging(self, temp_dir: Path) -> None:
        """Staging record survives create → get roundtrip."""
        import uuid

        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            extraction = {
                "model_version": "integration-test",
                "fields": {
                    "InvoiceNumber": {
                        "status": "PREDICTED",
                        "value": f"INT-{uuid.uuid4().hex[:8]}",
                        "confidence": 0.95,
                    }
                },
            }
            record_id = await connector.create_staging_record(
                document_id=f"integration-test/{uuid.uuid4()}",
                extraction_result=extraction,
                sharepoint_id="sp:integration-test",
            )
            record = await connector.get(config.staging_table, record_id)
            assert record is not None
            assert record["status"] == "pending"

    @pytest.mark.asyncio
    async def test_promote_to_production(self, temp_dir: Path) -> None:
        """Staging → production promotion creates valid production record."""
        import uuid

        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            extraction = {
                "model_version": "integration-test",
                "fields": {
                    "InvoiceNumber": {
                        "status": "PREDICTED",
                        "value": "INT-PROMO-001",
                        "confidence": 0.95,
                    }
                },
            }
            staging_id = await connector.create_staging_record(
                document_id=f"integration-test/{uuid.uuid4()}",
                extraction_result=extraction,
            )
            production_id = await connector.promote_to_production(
                staging_id=staging_id,
                approved_by="integration-test-user",
            )
            prod = await connector.get(config.production_table, production_id)
            assert prod is not None
            assert prod["staging_id"] == staging_id
            assert prod["approved_by"] == "integration-test-user"

    @pytest.mark.asyncio
    async def test_health_check(self, temp_dir: Path) -> None:
        """Health check returns healthy status."""
        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            result = await connector.health_check()
            assert result["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_query_with_filter(self, temp_dir: Path) -> None:
        """Query with filter returns only matching records."""
        config = MockDataverseConfig(data_dir=str(temp_dir))
        async with MockDataverseConnector(config) as connector:
            await connector.create("test_table", {"name": "x", "status": "pending"})
            await connector.create("test_table", {"name": "y", "status": "approved"})
            results = await connector.query(
                "test_table", filter_expr="status eq 'pending'"
            )
            assert all(r["status"] == "pending" for r in results)


@pytest.mark.integration
@pytest.mark.skipif(not AZURE_INTEGRATION, reason="AZURE_TEST_ENABLED not set")
class TestBlobStorageIntegration:
    """Integration tests for BlobStorageBackend. Requires live Azure credentials."""

    def test_write_read_exists_delete_roundtrip(self, tmp_path: Path) -> None:
        """Full CRUD roundtrip on blob storage."""
        import uuid

        from invoices.exceptions import StorageError
        from invoices.storage.blob import BlobStorageBackend

        # Uses env vars: AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_NAME
        try:
            backend = BlobStorageBackend()
        except StorageError:
            pytest.skip("No Azure blob credentials in environment")

        key = f"integration-test/{uuid.uuid4()}/test.txt"
        content = "integration test content"

        backend.write_text(key, content)
        assert backend.exists(key)
        assert backend.read_text(key) == content
        assert backend.delete(key)
        assert not backend.exists(key)

    def test_list_prefix(self, tmp_path: Path) -> None:
        """list_prefix returns blobs under prefix."""
        import uuid

        from invoices.exceptions import StorageError
        from invoices.storage.blob import BlobStorageBackend

        try:
            backend = BlobStorageBackend()
        except StorageError:
            pytest.skip("No Azure blob credentials in environment")

        prefix = f"integration-test/{uuid.uuid4()}/"
        keys = [f"{prefix}file_{i}.txt" for i in range(3)]
        for key in keys:
            backend.write_text(key, "data")

        results = backend.list_prefix(prefix)
        assert set(keys).issubset(set(results))

        for key in keys:
            backend.delete(key)

    def test_health_check(self) -> None:
        """Health check returns healthy for valid credentials."""
        from invoices.exceptions import StorageError
        from invoices.storage.blob import BlobStorageBackend

        try:
            backend = BlobStorageBackend()
        except StorageError:
            pytest.skip("No Azure blob credentials in environment")

        result = backend.health_check()
        assert result["status"] == "healthy"
