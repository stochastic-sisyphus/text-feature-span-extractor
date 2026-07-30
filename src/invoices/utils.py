"""Utility functions for the invoice extraction system."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

from .config import Config
from .logging import get_logger

logger = get_logger(__name__)


def find_schema_path() -> Path:
    """Locate seed schema — uses package data (works everywhere).

    Returns the path to contract.invoice.seed.json, which is used only
    for cold-start seeding when the contract_schema table is empty.
    Postgres is canonical; the seed file is never read again after first boot.
    """
    from importlib.resources import files

    pkg_path = files("invoices.schema").joinpath("contract.invoice.seed.json")
    # importlib.resources returns a Traversable; resolve to a real Path
    p = Path(str(pkg_path))
    if p.exists():
        return p
    raise FileNotFoundError(f"contract.invoice.seed.json not found at {p}")


def set_schema_cache(schema: "dict[str, Any]") -> None:
    """Push the postgres-backed schema into the in-memory cache.

    Called by API lifespan on startup and after schema edits.  psycopg 3
    decodes JSONB to dict at the SA engine layer — this helper takes the
    dict as-is.
    """
    from . import schema as schema_mod

    schema_mod.set(schema)


def contract_fingerprint(schema_obj: dict[str, Any]) -> str:
    """Compute SHA256 fingerprint of canonical schema."""
    canonical_json = json.dumps(schema_obj, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return fingerprint[:12]


def compute_contract_version(schema_obj: dict[str, Any]) -> str:
    """Generate contract version from semver + fingerprint."""
    semver = schema_obj.get("version", "1.0.0")
    fingerprint = contract_fingerprint(schema_obj)
    return f"{semver}+{fingerprint}"


def get_version_info(
    model_version: str | None = None,
    calibration_version: str | None = None,
) -> dict[str, str]:
    """Get comprehensive version information with contract versioning."""
    from . import schema as schema_mod

    schema_obj = schema_mod.load()
    contract_version = compute_contract_version(schema_obj)

    from .version import FEATURE_VERSION, decoder_version

    return {
        "contract_version": contract_version,
        "feature_version": FEATURE_VERSION,
        "decoder_version": decoder_version(),
        "model_version": model_version or Config.model_id or "unscored-baseline",
        "calibration_version": calibration_version or Config.calibration_version,
    }


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


# NOTE — Phase 0 wave B deprecation shim.
# Token identity is owned by the view that produces token rows; the
# canonical implementation lives at ``invoices.views.compute_stable_token_id``.
# This shim exists so external call sites that import from ``utils``
# keep working during the Phase 0 overlap; a follow-up wave (C or later)
# will migrate them and remove this forwarder.  The recipe must remain
# byte-identical — do NOT reimplement here.
def compute_stable_token_id(
    doc_id: str, page_idx: int, token_idx: int, text: str, bbox_norm: tuple
) -> str:
    """Deprecation shim — forwards to :func:`invoices.views.compute_stable_token_id`."""
    from .views import compute_stable_token_id as _canonical

    return _canonical(doc_id, page_idx, token_idx, text, bbox_norm)


def convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy(item) for item in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def get_current_utc_iso() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str) -> None:
        self.operation_name = operation_name
        self.start_time: float | None = None

    def __enter__(self) -> "Timer":
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.start_time is not None:
            duration = time.time() - self.start_time
            logger.debug(
                "timer_elapsed",
                operation=self.operation_name,
                duration_s=round(duration, 3),
            )

    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
