"""Utility functions for the invoice extraction system."""

import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from types import TracebackType
from typing import Any

from . import paths
from .config import Config
from .logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_contract_schema() -> dict[str, Any]:
    """Load and canonicalize the contract schema."""
    schema_path = paths.get_repo_root() / "schema" / "contract.invoice.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Contract schema not found: {schema_path}")

    with open(schema_path, encoding="utf-8") as f:
        schema_obj = json.load(f)

    # Canonicalize: sort keys for deterministic output
    canonical_json = json.dumps(schema_obj, sort_keys=True, separators=(",", ":"))
    result: dict[str, Any] = json.loads(canonical_json)
    return result


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


def get_version_info() -> dict[str, str]:
    """Get comprehensive version information with contract versioning."""
    # Load schema for contract version
    schema_obj = load_contract_schema()
    contract_version = compute_contract_version(schema_obj)

    # Model version from Config (which reads from env), manifest, or file
    # Treat empty string as unset to allow manifest/file-based override
    model_version = Config.MODEL_ID
    if not model_version or model_version == "unscored-baseline":
        model_version = "unscored-baseline"
        # Only check manifest if ranker is enabled (quality gate passed)
        # If ranker is disabled, model_version stays "unscored-baseline" (heuristic-only)
        if Config.USE_RANKER_MODEL:
            # Check manifest first (written by train.py)
            manifest_file = paths.get_models_dir() / "manifest.json"
            if manifest_file.exists():
                try:
                    with open(manifest_file, encoding="utf-8") as f:
                        manifest = json.load(f)
                    manifest_version = manifest.get("model_version", "")
                    if manifest_version:
                        model_type = manifest.get("model_type", "unknown")
                        model_version = f"{model_type}-{manifest_version}"
                except (json.JSONDecodeError, OSError):
                    pass
            # Fallback: legacy model_id.txt
            if model_version == "unscored-baseline":
                model_file = paths.get_models_dir() / "current" / "model_id.txt"
                if model_file.exists():
                    with open(model_file, encoding="utf-8") as f:
                        file_version = f.read().strip()
                        if file_version:
                            model_version = file_version

    # Calibration version: reflect actual state from labeled data
    calibration_version = Config.CALIBRATION_VERSION  # default "none"
    try:
        from . import calibration as cal_mod

        data_dir = paths.get_data_dir()
        ece = cal_mod.compute_ece_from_labeled_data(data_dir)
        if ece is not None:
            # Labeled data exists and ECE was computed — stamp it
            calibration_version = f"ece-{ece:.4f}"
    except Exception:
        pass  # Keep "none" if calibration can't be computed

    return {
        "contract_version": contract_version,
        "feature_version": Config.FEATURE_VERSION,
        "decoder_version": Config.DECODER_VERSION,
        "model_version": model_version,
        "calibration_version": calibration_version,
    }


def get_version_stamps() -> dict[str, str]:
    """Get all version stamps for consistent labeling (legacy compatibility)."""
    return get_version_info()


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def compute_stable_token_id(
    doc_id: str, page_idx: int, token_idx: int, text: str, bbox_norm: tuple
) -> str:
    """Compute stable token ID using SHA1 hash as specified."""
    # Convert bbox_norm to string for consistent hashing
    bbox_str = (
        f"{bbox_norm[0]:.6f},{bbox_norm[1]:.6f},{bbox_norm[2]:.6f},{bbox_norm[3]:.6f}"
    )
    hash_input = f"{doc_id}|{page_idx}|{token_idx}|{text}|{bbox_str}"
    return hashlib.sha1(hash_input.encode("utf-8"), usedforsecurity=False).hexdigest()


def get_current_utc_iso() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def write_json_with_backup(filepath: Path, data: dict[str, Any]) -> None:
    """Write JSON with atomic operation and backup."""
    temp_path = filepath.with_suffix(".tmp")

    # Write to temporary file first
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Atomic move
    temp_path.replace(filepath)


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
