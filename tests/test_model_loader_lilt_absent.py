"""LiLT-absent is a first-class state in model_loader.

An image built without LiLT weights (Dockerfile ``WITH_LILT=0``) must not
raise from the loader. Absence resolves once per process; a genuine load
failure leaves the cache unset so the next call retries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from invoices import model_loader as ml


@pytest.fixture
def empty_lilt_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ml, "_VENDORED_LILT_DIR", str(tmp_path))
    monkeypatch.setattr(ml, "_LILT_BUNDLE", None)
    return tmp_path


def test_fetch_vendored_returns_absent_sentinel(empty_lilt_dir: Path) -> None:
    assert ml._fetch_lilt_vendored() is ml._LILT_ABSENT


def test_resolve_caches_absence_and_probes_once(
    empty_lilt_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(ml, "logger", fake_logger)

    first = ml._resolve_lilt_bundle()
    second = ml._resolve_lilt_bundle()

    assert first is ml._LILT_ABSENT
    assert second is ml._LILT_ABSENT
    assert ml._LILT_BUNDLE is ml._LILT_ABSENT
    absent_logs = [
        c
        for c in fake_logger.info.call_args_list
        if c.args[0] == "lilt_vendored_absent"
    ]
    assert len(absent_logs) == 1


def test_resolve_load_failure_leaves_cache_unset(
    empty_lilt_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Weights "present" (garbage bytes): the load is attempted and fails
    # (torch missing, or the file is not a real checkpoint).
    (empty_lilt_dir / "config.json").write_text("{}")
    (empty_lilt_dir / "model.safetensors").write_bytes(b"not a real checkpoint")

    attempts: list[Any] = []
    real_load = ml._load_lilt_from_dir

    def counting_load(local_dir: str, *, source: str) -> dict[str, Any] | None:
        attempts.append(local_dir)
        return real_load(local_dir, source=source)

    monkeypatch.setattr(ml, "_load_lilt_from_dir", counting_load)

    assert ml._fetch_lilt_vendored() is None
    assert ml._resolve_lilt_bundle() is None
    assert ml._LILT_BUNDLE is None

    # Retry semantics: the next call attempts the load again.
    n = len(attempts)
    assert ml._resolve_lilt_bundle() is None
    assert len(attempts) == n + 1
