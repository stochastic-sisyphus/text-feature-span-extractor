"""Tests for storage abstraction layer.

Tests the StorageBackend protocol implementations (LocalStorageBackend)
and the factory function get_storage().
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from invoices.exceptions import StorageError
from invoices.storage import (
    LocalStorageBackend,
    get_storage,
    reset_storage,
)


class TestLocalStorageBackend:
    """Tests for LocalStorageBackend implementation."""

    @pytest.fixture
    def storage(self, tmp_path: Path) -> LocalStorageBackend:
        """Create a LocalStorageBackend with a temp directory."""
        return LocalStorageBackend(base_path=tmp_path)

    def test_write_and_read_bytes(self, storage: LocalStorageBackend) -> None:
        """Should write and read binary data."""
        data = b"test binary data \x00\x01\x02"
        storage.write_bytes("test/file.bin", data)

        result = storage.read_bytes("test/file.bin")
        assert result == data

    def test_write_creates_parent_directories(
        self, storage: LocalStorageBackend
    ) -> None:
        """Should create parent directories when writing."""
        storage.write_text("deep/nested/path/file.txt", "content")

        assert storage.exists("deep/nested/path/file.txt")

    def test_list_prefix_returns_matching_files(
        self, storage: LocalStorageBackend
    ) -> None:
        """Should list all files under a prefix."""
        storage.write_text("models/manifest.json", "{}")
        storage.write_text("models/field1/model.json", "{}")
        storage.write_text("models/field1/metadata.json", "{}")
        storage.write_text("models/field2/model.json", "{}")
        storage.write_text("other/file.txt", "other")

        result = storage.list_prefix("models/")

        assert len(result) == 4
        assert "models/manifest.json" in result
        assert "models/field1/model.json" in result
        assert "models/field1/metadata.json" in result
        assert "models/field2/model.json" in result
        assert "other/file.txt" not in result

    def test_read_missing_file_raises_storage_error(
        self, storage: LocalStorageBackend
    ) -> None:
        """Should raise StorageError when reading non-existent file."""
        with pytest.raises(StorageError) as exc_info:
            storage.read_text("missing.txt")

        assert exc_info.value.operation == "read"
        assert exc_info.value.path == "missing.txt"

    def test_path_normalization_posix_style(self, storage: LocalStorageBackend) -> None:
        """Should handle both forward and back slashes in paths."""
        storage.write_text("path/to/file.txt", "content")

        # Both path styles should work
        assert storage.exists("path/to/file.txt")
        # Backslash normalized to forward slash
        result = storage.read_text("path\\to\\file.txt")
        assert result == "content"


class TestGetStorage:
    """Tests for get_storage() factory function."""

    def setup_method(self) -> None:
        """Reset storage singleton before each test."""
        reset_storage()

    def teardown_method(self) -> None:
        """Clean up environment variables after each test."""
        reset_storage()
        # Restore original env vars if modified
        for key in ["INVOICEX_STORAGE_BACKEND", "AZURE_STORAGE_ACCOUNT_NAME"]:
            if key in os.environ:
                del os.environ[key]

    def test_returns_local_storage_by_default(self) -> None:
        """Should return LocalStorageBackend when STORAGE_BACKEND is local."""
        # Default is local
        storage = get_storage(force_new=True)
        assert isinstance(storage, LocalStorageBackend)

    def test_returns_cached_instance(self) -> None:
        """Should return same instance on subsequent calls."""
        storage1 = get_storage()
        storage2 = get_storage()
        assert storage1 is storage2
