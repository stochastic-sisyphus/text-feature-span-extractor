"""Unit tests for list_documents_delta bootstrap URL scoping.

Covers:
  (a) bootstrap with folder_path set + _folder_id resolved  → id-form /items/{id}/delta
  (b) bootstrap with folder_path set but _folder_id is None  → path-form /root:/{encoded}:/delta
  (c) bootstrap with folder_path empty                        → whole-drive /root/delta
  (d) delta_link provided                                     → opaque URL reused verbatim

The queue.py scope-change detection (case e) lives inside a pgq closure over a
live pool and is verified by code inspection rather than unit test — noted inline.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from invoices.azure.config import SharePointConfig
from invoices.azure.sharepoint._connector import SharePointConnector

_TEST_FOLDER_ID = "folder-item-id-abc123"
_TEST_SITE_ID = "test-site-id"
_TEST_DRIVE_ID = "test-drive-id"


def _make_connector(
    folder_path: str, *, folder_id: str | None = None
) -> SharePointConnector:
    """Build a SharePointConnector with a minimal config (no auth calls).

    Args:
        folder_path: Value of config.folder_path.
        folder_id: If provided, pre-seeds _folder_id (simulates a successful
            _validate_folder run that captured the item id).
    """
    config = SharePointConfig(
        site_id=_TEST_SITE_ID,
        tenant_id="test-tenant-id",
        drive_id=_TEST_DRIVE_ID,
        folder_path=folder_path,
    )
    # Patch azure.identity import so __init__ doesn't raise ImportError.
    with patch.dict(
        "sys.modules",
        {
            "azure": MagicMock(),
            "azure.identity": MagicMock(),
            "azure.identity.aio": MagicMock(),
        },
    ):
        connector = SharePointConnector(config)

    # Inject a non-None _session sentinel so the guard in list_documents_delta passes.
    connector._session = MagicMock()  # type: ignore[assignment]
    if folder_id is not None:
        connector._folder_id = folder_id
    return connector


def _delta_response(
    delta_link: str = "https://graph.microsoft.com/v1.0/drives/test/delta?token=abc",
) -> dict[str, Any]:
    """Minimal Graph API page that terminates the pagination loop."""
    return {"value": [], "@odata.deltaLink": delta_link}


async def _call_list_documents_delta(
    connector: SharePointConnector,
    delta_link_arg: str | None,
    captured: list[str],
) -> None:
    """Invoke list_documents_delta, capturing the first GET URL in `captured`."""

    async def fake_graph_request(method: str, url: str, **kwargs: Any) -> MagicMock:
        if not captured:
            captured.append(url)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = _delta_response()
        resp.headers = {}
        return resp

    connector._graph_request = fake_graph_request  # type: ignore[assignment]
    connector._ensure_drive_id = AsyncMock()
    connector._get_headers = AsyncMock(return_value={})

    await connector.list_documents_delta(delta_link_arg)


@pytest.mark.asyncio
async def test_bootstrap_with_folder_id_uses_id_form() -> None:
    """When _folder_id is set, bootstrap must use /items/{id}/delta (id-form)."""
    connector = _make_connector("AI PDF Files - DEV", folder_id=_TEST_FOLDER_ID)
    captured: list[str] = []
    await _call_list_documents_delta(connector, None, captured)

    assert captured, "no GET request was made"
    url = captured[0]
    assert f"/items/{_TEST_FOLDER_ID}/delta" in url, (
        f"expected id-form /items/{{id}}/delta, got: {url}"
    )
    # Must NOT fall back to path-addressed or root forms
    assert "/root:/" not in url, f"unexpected path-addressed form in: {url}"
    assert url.rstrip("/") != url.replace(
        f"/items/{_TEST_FOLDER_ID}/delta", "/root/delta"
    ), "must not fall back to whole-drive /root/delta"


@pytest.mark.asyncio
async def test_bootstrap_with_folder_path_no_id_is_path_form() -> None:
    """When folder_path is set but _folder_id is None, must fall back to /root:/{encoded}:/delta."""
    connector = _make_connector("AI PDF Files - DEV", folder_id=None)
    # _folder_id remains None — simulates _validate_folder being bypassed or id absent
    captured: list[str] = []
    await _call_list_documents_delta(connector, None, captured)

    assert captured, "no GET request was made"
    url = captured[0]
    assert "/root:/" in url, f"expected path-form /root:/…:/delta, got: {url}"
    assert ":/delta" in url, f"expected :/delta suffix, got: {url}"
    # Spaces must be percent-encoded
    assert "%20" in url, f"expected %20 encoding for space, got: {url}"
    # Must not use id-form or whole-drive
    assert "/items/" not in url, f"unexpected id-form in: {url}"
    assert "/root/delta" not in url, f"unexpected whole-drive form in: {url}"


@pytest.mark.asyncio
async def test_bootstrap_with_empty_folder_path_is_whole_drive() -> None:
    """Bootstrap with folder_path='' must use /root/delta (whole-drive)."""
    connector = _make_connector("")
    captured: list[str] = []
    await _call_list_documents_delta(connector, None, captured)

    assert captured, "no GET request was made"
    url = captured[0]
    assert url.endswith("/root/delta"), f"expected whole-drive /root/delta, got: {url}"
    assert "/root:/" not in url, f"unexpected folder-scoped path in: {url}"
    assert "/items/" not in url, f"unexpected id-form in: {url}"


@pytest.mark.asyncio
async def test_existing_delta_link_reused_verbatim() -> None:
    """When delta_link is provided it must be used as-is, regardless of folder_path or folder_id."""
    opaque_link = "https://graph.microsoft.com/v1.0/drives/xyz/delta?token=secret123"
    connector = _make_connector("AI PDF Files - DEV", folder_id=_TEST_FOLDER_ID)
    captured: list[str] = []
    await _call_list_documents_delta(connector, opaque_link, captured)

    assert captured, "no GET request was made"
    assert captured[0] == opaque_link, (
        f"expected delta_link reused verbatim, got: {captured[0]}"
    )
