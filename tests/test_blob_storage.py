"""Tests for BlobStorageBackend.

All Azure SDK calls are mocked — no real Azure connections made.
"""

from __future__ import annotations

import sys
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from invoices.exceptions import StorageError
from invoices.storage import get_storage, reset_storage
from invoices.storage.blob import BlobStorageBackend

# ---------------------------------------------------------------------------
# Constants (realistic fake values, not magic strings)
# ---------------------------------------------------------------------------

FAKE_CONN_STR = (
    "DefaultEndpointsProtocol=https;AccountName=testaccount;"
    "AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net"
)
FAKE_ACCOUNT = "testaccount"
FAKE_CONTAINER = "invoicex-test"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_container() -> MagicMock:
    """A pre-configured ContainerClient mock."""
    container = MagicMock()
    container.exists.return_value = True
    return container


@pytest.fixture
def backend_conn(mock_container: MagicMock) -> BlobStorageBackend:
    """BlobStorageBackend initialised via connection string, container pre-set."""
    b = BlobStorageBackend(
        connection_string=FAKE_CONN_STR,
        container_name=FAKE_CONTAINER,
    )
    b._container = mock_container
    return b


# ---------------------------------------------------------------------------
# TestBlobStorageBackendInit
# ---------------------------------------------------------------------------


class TestBlobStorageBackendInit:
    """Initialisation-time behaviour."""

    def test_missing_both_creds_raises(self) -> None:
        """No connection_string and no account_name → StorageError."""
        with pytest.raises(StorageError) as exc_info:
            BlobStorageBackend(
                connection_string=None,
                account_name=None,
                container_name=FAKE_CONTAINER,
            )
        assert exc_info.value.operation == "init"

    def test_connection_string_init_succeeds(self) -> None:
        """Providing connection_string sets attribute without error."""
        b = BlobStorageBackend(
            connection_string=FAKE_CONN_STR,
            container_name=FAKE_CONTAINER,
        )
        assert b.connection_string == FAKE_CONN_STR
        assert b.container_name == FAKE_CONTAINER
        assert b._container is None  # lazy — not yet initialised

    def test_account_name_init_succeeds(self) -> None:
        """Providing account_name sets attribute without error."""
        b = BlobStorageBackend(
            account_name=FAKE_ACCOUNT,
            container_name=FAKE_CONTAINER,
        )
        assert b.account_name == FAKE_ACCOUNT
        assert b._container is None

    def test_falls_back_to_config_connection_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falls back to Config env var when no explicit creds provided."""
        monkeypatch.setattr(
            "invoices.storage.blob.Config",
            MagicMock(
                AZURE_STORAGE_CONNECTION_STRING=FAKE_CONN_STR,
                AZURE_STORAGE_ACCOUNT_NAME="",
                AZURE_STORAGE_CONTAINER_NAME=FAKE_CONTAINER,
            ),
        )
        b = BlobStorageBackend()
        assert b.connection_string == FAKE_CONN_STR


# ---------------------------------------------------------------------------
# TestBlobStorageBackendOperations
# ---------------------------------------------------------------------------


class TestBlobStorageBackendOperations:
    """CRUD and utility operations (container pre-set, no lazy-init)."""

    def test_write_text_read_text_roundtrip(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """write_text then read_text returns the original string."""
        payload = "hello blob world"
        encoded = payload.encode("utf-8")

        # Prepare download mock
        downloader = Mock()
        downloader.readall.return_value = encoded
        blob_client = Mock()
        blob_client.download_blob.return_value = downloader
        mock_container.get_blob_client.return_value = blob_client

        backend_conn.write_text("docs/note.txt", payload)
        result = backend_conn.read_text("docs/note.txt")

        blob_client.upload_blob.assert_called_once_with(encoded, overwrite=True)
        assert result == payload

    def test_write_text_encodes_utf8(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """write_text calls write_bytes with UTF-8 encoded bytes."""
        blob_client = Mock()
        mock_container.get_blob_client.return_value = blob_client

        text = "café résumé"
        backend_conn.write_text("utf8.txt", text)

        blob_client.upload_blob.assert_called_once_with(
            text.encode("utf-8"), overwrite=True
        )

    def test_read_bytes_blob_not_found_raises_storage_error(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """BlobNotFound in exception message → StorageError with operation='read'."""
        blob_client = Mock()
        blob_client.download_blob.side_effect = Exception(
            "BlobNotFound: The specified blob does not exist."
        )
        mock_container.get_blob_client.return_value = blob_client

        with pytest.raises(StorageError) as exc_info:
            backend_conn.read_bytes("missing/file.bin")

        assert exc_info.value.operation == "read"
        assert "missing/file.bin" in exc_info.value.path

    def test_exists_returns_true_when_blob_exists(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """exists() delegates to blob_client.exists()."""
        blob_client = Mock()
        blob_client.exists.return_value = True
        mock_container.get_blob_client.return_value = blob_client

        assert backend_conn.exists("models/field.xgb") is True

    def test_exists_returns_false_when_blob_absent(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        blob_client = Mock()
        blob_client.exists.return_value = False
        mock_container.get_blob_client.return_value = blob_client

        assert backend_conn.exists("nope.json") is False

    def test_delete_returns_true_when_blob_present(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """delete() deletes the blob and returns True."""
        blob_client = Mock()
        blob_client.exists.return_value = True
        mock_container.get_blob_client.return_value = blob_client

        result = backend_conn.delete("models/old.xgb")

        assert result is True
        blob_client.delete_blob.assert_called_once()

    def test_delete_returns_false_when_blob_absent(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """delete() returns False without error when blob doesn't exist."""
        blob_client = Mock()
        blob_client.exists.return_value = False
        mock_container.get_blob_client.return_value = blob_client

        result = backend_conn.delete("ghost.json")

        assert result is False
        blob_client.delete_blob.assert_not_called()

    def test_list_prefix_returns_sorted_blob_names(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """list_prefix returns blob names under the prefix, sorted."""
        blob_b = Mock()
        blob_b.name = "models/b.xgb"
        blob_a = Mock()
        blob_a.name = "models/a.xgb"
        mock_container.list_blobs.return_value = [blob_b, blob_a]

        result = backend_conn.list_prefix("models/")

        assert result == ["models/a.xgb", "models/b.xgb"]
        mock_container.list_blobs.assert_called_once_with(name_starts_with="models/")

    def test_path_normalization_strips_leading_slash(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """Leading slash is stripped before passing to Azure SDK."""
        blob_client = Mock()
        mock_container.get_blob_client.return_value = blob_client

        backend_conn.write_bytes("/data/file.bin", b"x")

        mock_container.get_blob_client.assert_called_with("data/file.bin")

    def test_path_normalization_backslash_to_forward_slash(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """Backslashes are converted to forward slashes."""
        blob_client = Mock()
        mock_container.get_blob_client.return_value = blob_client

        backend_conn.write_bytes("data\\subdir\\file.bin", b"y")

        mock_container.get_blob_client.assert_called_with("data/subdir/file.bin")

    def test_download_to_tempfile_returns_path_with_content(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """download_to_tempfile writes blob bytes to a temp file and returns Path."""
        content = b"model bytes"
        downloader = Mock()
        downloader.readall.return_value = content
        blob_client = Mock()
        blob_client.download_blob.return_value = downloader
        mock_container.get_blob_client.return_value = blob_client

        temp_path = backend_conn.download_to_tempfile("models/x.xgb", suffix=".xgb")

        try:
            assert isinstance(temp_path, Path)
            assert temp_path.suffix == ".xgb"
            assert temp_path.read_bytes() == content
        finally:
            temp_path.unlink(missing_ok=True)

    def test_upload_from_file_reads_local_and_calls_write_bytes(
        self,
        backend_conn: BlobStorageBackend,
        mock_container: MagicMock,
        tmp_path: Path,
    ) -> None:
        """upload_from_file reads the local file and uploads to blob storage."""
        data = b"local file content"
        local_file = tmp_path / "model.xgb"
        local_file.write_bytes(data)

        blob_client = Mock()
        mock_container.get_blob_client.return_value = blob_client

        backend_conn.upload_from_file(local_file, "models/model.xgb")

        blob_client.upload_blob.assert_called_once_with(data, overwrite=True)


# ---------------------------------------------------------------------------
# TestBlobStorageBackendHealthCheck
# ---------------------------------------------------------------------------


class TestBlobStorageBackendHealthCheck:
    """health_check() return value shape."""

    def test_healthy_when_list_blobs_succeeds(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """list_blobs succeeds → status 'healthy'."""
        mock_container.list_blobs.return_value = iter([])

        result = backend_conn.health_check()

        assert result["status"] == "healthy"
        assert result["backend"] == "blob"

    def test_unhealthy_on_storage_error(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """StorageError from _get_container → status 'unhealthy'."""
        mock_container.list_blobs.side_effect = StorageError(
            "list", "", "connection refused"
        )

        result = backend_conn.health_check()

        assert result["status"] == "unhealthy"
        assert result["backend"] == "blob"

    def test_unhealthy_on_unexpected_exception(
        self, backend_conn: BlobStorageBackend, mock_container: MagicMock
    ) -> None:
        """Unexpected exception → status 'unhealthy'."""
        mock_container.list_blobs.side_effect = RuntimeError("network partition")

        result = backend_conn.health_check()

        assert result["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# TestBlobStorageBackendGetContainer
# ---------------------------------------------------------------------------


class TestBlobStorageBackendGetContainer:
    """_get_container() lazy-init and caching behaviour."""

    def test_second_call_returns_cached_container(self) -> None:
        """_get_container() must not re-initialise on the second call."""
        b = BlobStorageBackend(
            connection_string=FAKE_CONN_STR,
            container_name=FAKE_CONTAINER,
        )
        sentinel = MagicMock()
        b._container = sentinel

        result1 = b._get_container()
        result2 = b._get_container()

        assert result1 is sentinel
        assert result2 is sentinel

    def test_connection_string_path_calls_from_connection_string(self) -> None:
        """Connection string → BlobServiceClient.from_connection_string used."""
        b = BlobStorageBackend(
            connection_string=FAKE_CONN_STR,
            container_name=FAKE_CONTAINER,
        )

        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_container.exists.return_value = True
        mock_service.get_container_client.return_value = mock_container

        mock_blob_module = MagicMock()
        mock_blob_module.BlobServiceClient.from_connection_string.return_value = (
            mock_service
        )

        with patch.dict(sys.modules, {"azure.storage.blob": mock_blob_module}):
            b._get_container()

        mock_blob_module.BlobServiceClient.from_connection_string.assert_called_once_with(
            FAKE_CONN_STR
        )

    def test_account_name_path_uses_default_azure_credential(self) -> None:
        """Account name → DefaultAzureCredential path taken."""
        mock_cred = MagicMock()
        mock_service = MagicMock()
        mock_service_container = MagicMock()
        mock_service_container.exists.return_value = True
        mock_service.get_container_client.return_value = mock_service_container

        mock_blob_module = MagicMock()
        mock_blob_module.BlobServiceClient = MagicMock(return_value=mock_service)

        mock_identity_module = MagicMock()
        mock_identity_module.DefaultAzureCredential = MagicMock(return_value=mock_cred)

        b = BlobStorageBackend(account_name=FAKE_ACCOUNT, container_name=FAKE_CONTAINER)

        with patch.dict(
            sys.modules,
            {
                "azure.storage.blob": mock_blob_module,
                "azure.identity": mock_identity_module,
            },
        ):
            b._get_container()

        mock_identity_module.DefaultAzureCredential.assert_called_once()

    def test_creates_container_when_not_exists(self) -> None:
        """If container.exists() is False, create_container() is called."""
        mock_service = MagicMock()
        mock_container = MagicMock()
        mock_container.exists.return_value = False
        mock_service.get_container_client.return_value = mock_container

        mock_blob_module = MagicMock()
        mock_blob_module.BlobServiceClient.from_connection_string.return_value = (
            mock_service
        )

        b = BlobStorageBackend(
            connection_string=FAKE_CONN_STR, container_name=FAKE_CONTAINER
        )

        with patch.dict(sys.modules, {"azure.storage.blob": mock_blob_module}):
            b._get_container()

        mock_container.create_container.assert_called_once()

    def test_import_error_azure_storage_blob_raises_storage_error(self) -> None:
        """Missing azure-storage-blob → StorageError on _get_container()."""
        b = BlobStorageBackend(
            connection_string=FAKE_CONN_STR, container_name=FAKE_CONTAINER
        )
        with patch.dict(sys.modules, {"azure.storage.blob": None}):  # type: ignore[dict-item]
            with pytest.raises(StorageError) as exc_info:
                b._get_container()
        assert exc_info.value.operation == "init"

    def test_import_error_azure_identity_raises_storage_error(self) -> None:
        """Missing azure-identity (account_name path) → StorageError."""
        mock_blob_module = MagicMock()
        mock_blob_module.BlobServiceClient = MagicMock()

        b = BlobStorageBackend(account_name=FAKE_ACCOUNT, container_name=FAKE_CONTAINER)

        with patch.dict(
            sys.modules,
            {
                "azure.storage.blob": mock_blob_module,
                "azure.identity": None,  # type: ignore[dict-item]
            },
        ):
            with pytest.raises(StorageError) as exc_info:
                b._get_container()
        assert exc_info.value.operation == "init"


# ---------------------------------------------------------------------------
# TestGetStorageFactoryWithBlob
# ---------------------------------------------------------------------------


class TestGetStorageFactoryWithBlob:
    """get_storage() factory with STORAGE_BACKEND=blob."""

    @pytest.fixture(autouse=True)
    def _reset(self) -> Generator[None, None, None]:
        reset_storage()
        yield
        reset_storage()

    def test_blob_backend_with_creds_returns_blob_instance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STORAGE_BACKEND=blob + connection string → BlobStorageBackend returned."""
        monkeypatch.setenv("INVOICEX_STORAGE_BACKEND", "blob")
        monkeypatch.setenv("AZURE_STORAGE_CONNECTION_STRING", FAKE_CONN_STR)

        # Patch Config to reflect the env changes without reloading the module
        with patch(
            "invoices.storage.Config",
            MagicMock(
                STORAGE_BACKEND="blob",
                AZURE_STORAGE_CONNECTION_STRING=FAKE_CONN_STR,
                AZURE_STORAGE_ACCOUNT_NAME="",
                AZURE_STORAGE_CONTAINER_NAME=FAKE_CONTAINER,
            ),
        ):
            storage = get_storage(force_new=True)

        assert isinstance(storage, BlobStorageBackend)
        assert storage.connection_string == FAKE_CONN_STR

    def test_blob_backend_no_creds_raises_storage_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """STORAGE_BACKEND=blob + no creds → StorageError from factory."""
        with patch(
            "invoices.storage.Config",
            MagicMock(
                STORAGE_BACKEND="blob",
                AZURE_STORAGE_CONNECTION_STRING="",
                AZURE_STORAGE_ACCOUNT_NAME="",
                AZURE_STORAGE_CONTAINER_NAME=FAKE_CONTAINER,
            ),
        ):
            with pytest.raises(StorageError) as exc_info:
                get_storage(force_new=True)

        assert exc_info.value.operation == "init"
