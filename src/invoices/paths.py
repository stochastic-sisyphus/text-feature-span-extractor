"""Path resolution utilities for the invoice extraction system."""

from pathlib import Path


def get_repo_root() -> Path:
    """Find the repository root directory."""
    current = Path(__file__).parent.parent.parent
    # Look for pyproject.toml to confirm repo root
    if (current / "pyproject.toml").exists():
        return current
    raise RuntimeError("Could not find repository root")


def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_repo_root() / "data"


def get_ingest_dir() -> Path:
    """Get the ingest directory path."""
    return get_data_dir() / "ingest"


def get_ingest_raw_dir() -> Path:
    """Get the ingest/raw directory path."""
    return get_ingest_dir() / "raw"


def get_ingest_index_path() -> Path:
    """Get the ingest index parquet file path."""
    return get_ingest_dir() / "index.parquet"


def get_tokens_dir() -> Path:
    """Get the tokens directory path."""
    return get_data_dir() / "tokens"


def get_candidates_dir() -> Path:
    """Get the candidates directory path."""
    return get_data_dir() / "candidates"


def get_predictions_dir() -> Path:
    """Get the predictions directory path."""
    return get_data_dir() / "predictions"


def get_review_dir() -> Path:
    """Get the review directory path."""
    return get_data_dir() / "review"


def get_review_queue_path() -> Path:
    """Get the review queue parquet file path."""
    return get_review_dir() / "queue.parquet"


def get_logs_dir() -> Path:
    """Get the logs directory path."""
    return get_data_dir() / "logs"


def get_labels_dir() -> Path:
    """Get the labels directory path."""
    return get_data_dir() / "labels"


def get_labels_raw_dir() -> Path:
    """Get the labels/raw directory path."""
    return get_labels_dir() / "raw"


def get_labels_aligned_dir() -> Path:
    """Get the labels/aligned directory path."""
    return get_labels_dir() / "aligned"


def get_labels_index_path() -> Path:
    """Get the labels index parquet file path."""
    return get_labels_dir() / "index.parquet"


def get_labels_raw_path(sha256: str) -> Path:
    """Get the labels raw JSONL file path for a document."""
    return get_labels_raw_dir() / f"{sha256}.jsonl"


def get_labels_aligned_path(sha256: str) -> Path:
    """Get the labels aligned parquet file path for a document."""
    return get_labels_aligned_dir() / f"{sha256}.parquet"


def get_tokens_path(sha256: str) -> Path:
    """Get the tokens parquet file path for a document."""
    return get_tokens_dir() / f"{sha256}.parquet"


def get_candidates_path(sha256: str) -> Path:
    """Get the candidates parquet file path for a document."""
    return get_candidates_dir() / f"{sha256}.parquet"


def get_predictions_path(sha256: str) -> Path:
    """Get the predictions JSON file path for a document."""
    return get_predictions_dir() / f"{sha256}.json"


def get_raw_pdf_path(sha256: str) -> Path:
    """Get the raw PDF file path for a document."""
    return get_ingest_raw_dir() / f"{sha256}.pdf"


def ensure_directories() -> None:
    """Ensure all required directories exist."""
    directories = [
        get_data_dir(),
        get_ingest_dir(),
        get_ingest_raw_dir(),
        get_tokens_dir(),
        get_candidates_dir(),
        get_predictions_dir(),
        get_review_dir(),
        get_logs_dir(),
        get_labels_dir(),
        get_labels_raw_dir(),
        get_labels_aligned_dir(),
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
