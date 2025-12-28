"""Utility functions for the invoice extraction system."""

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

# Default version values
FEATURE_VERSION = "v1"
DECODER_VERSION = "v1"
CALIBRATION_VERSION = "none"


def load_contract_schema() -> dict[str, Any]:
    """Load and canonicalize the contract schema."""
    schema_path = paths.get_repo_root() / "schema" / "contract.invoice.json"

    if not schema_path.exists():
        raise FileNotFoundError(f"Contract schema not found: {schema_path}")

    with open(schema_path, encoding="utf-8") as f:
        schema_obj = json.load(f)

    # Canonicalize: sort keys for deterministic output
    canonical_json = json.dumps(schema_obj, sort_keys=True, separators=(",", ":"))
    return json.loads(canonical_json)


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

    # Model version from env or file
    model_version = os.environ.get("MODEL_ID")
    if not model_version:
        model_file = (
            paths.get_repo_root() / "data" / "models" / "current" / "model_id.txt"
        )
        if model_file.exists():
            with open(model_file, encoding="utf-8") as f:
                model_version = f.read().strip()
        else:
            model_version = "unscored-baseline"

    return {
        "contract_version": contract_version,
        "feature_version": FEATURE_VERSION,
        "decoder_version": DECODER_VERSION,
        "model_version": model_version,
        "calibration_version": CALIBRATION_VERSION,
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
    return hashlib.sha1(hash_input.encode("utf-8")).hexdigest()


def get_current_utc_iso() -> str:
    """Get current UTC timestamp in ISO8601 format."""
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename: str) -> str:
    """Convert filename to safe format for storage."""
    # Keep only alphanumeric, dots, hyphens, underscores
    safe_chars = []
    for char in filename:
        if char.isalnum() or char in ".-_":
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


def write_json_with_backup(filepath: Path, data: dict[str, Any]) -> None:
    """Write JSON with atomic operation and backup."""
    temp_path = filepath.with_suffix(".tmp")

    # Write to temporary file first
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Atomic move
    temp_path.replace(filepath)


def log_timing(
    operation: str, duration_seconds: float, doc_count: int = 1
) -> dict[str, Any]:
    """Log timing information."""
    return {
        "operation": operation,
        "duration_seconds": round(duration_seconds, 4),
        "doc_count": doc_count,
        "docs_per_second": round(doc_count / duration_seconds, 2)
        if duration_seconds > 0
        else 0,
        "timestamp": get_current_utc_iso(),
        **get_version_stamps(),
    }


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            print(f"{self.operation_name}: {duration:.3f}s")

    def elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
