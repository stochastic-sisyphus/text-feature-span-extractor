"""I/O utilities for parquet file operations with error handling."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow

from .logging import get_logger

logger = get_logger(__name__)


def read_parquet_safe(
    path: Path,
    on_error: str = "raise",
    logger_context: dict[str, Any] | None = None,
) -> pd.DataFrame | None:
    """Read a parquet file with error handling.

    Args:
        path: Path to the parquet file
        on_error: Action on error - "raise", "warn", "empty"
        logger_context: Additional context for error logging

    Returns:
        DataFrame or None (if on_error="empty" and file doesn't exist)

    Raises:
        Exception: If on_error="raise" and read fails
    """
    if not path.exists():
        if on_error == "empty":
            return None
        elif on_error == "warn":
            logger.warning(
                "parquet_not_found",
                path=str(path),
                **(logger_context or {}),
            )
            return None
        else:
            raise FileNotFoundError(f"Parquet file not found: {path}")

    try:
        return pd.read_parquet(path)
    except (pyarrow.ArrowInvalid, pyarrow.ArrowIOError, OSError, KeyError) as e:
        if on_error == "raise":
            raise
        elif on_error == "warn":
            logger.warning(
                "parquet_read_failed",
                path=str(path),
                error_type=type(e).__name__,
                reason=str(e),
                **(logger_context or {}),
            )
            return None
        elif on_error == "empty":
            return None
        else:
            raise


def write_parquet_safe(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a parquet file with directory creation.

    Args:
        df: DataFrame to write
        path: Path to the parquet file
        **kwargs: Additional arguments for to_parquet()
    """
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Default to index=False unless specified
    if "index" not in kwargs:
        kwargs["index"] = False

    df.to_parquet(path, **kwargs)


def list_parquet_files(directory: Path) -> list[Path]:
    """List all parquet files in a directory.

    Args:
        directory: Directory to search

    Returns:
        List of parquet file paths
    """
    if not directory.exists():
        return []

    return sorted(directory.glob("*.parquet"))


def get_parquet_sha256s(directory: Path) -> set[str]:
    """Extract SHA256 hashes from parquet filenames in a directory.

    Args:
        directory: Directory containing {sha256}.parquet files

    Returns:
        Set of SHA256 hashes (without .parquet extension)
    """
    files = list_parquet_files(directory)
    return {f.stem for f in files}


def append_to_parquet(
    new_df: pd.DataFrame,
    path: Path,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    **write_kwargs: Any,
) -> None:
    """Append a DataFrame to an existing parquet file.

    Reads existing file (if it exists), concatenates with new data, applies
    optional transform, and writes the combined result.

    Args:
        new_df: New DataFrame to append
        path: Path to the parquet file
        transform: Optional function to apply to combined DataFrame before writing
        **write_kwargs: Additional arguments for write_parquet_safe()
    """
    existing_df = read_parquet_safe(path, on_error="empty")
    if existing_df is not None:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    if transform is not None:
        combined_df = transform(combined_df)

    write_parquet_safe(combined_df, path, **write_kwargs)


def load_and_concat_parquets(
    directory: Path,
    filter_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    on_error: str = "warn",
) -> pd.DataFrame:
    """Load all parquet files in a directory and concatenate them.

    Args:
        directory: Directory containing parquet files
        filter_fn: Optional function to filter/transform each DataFrame before concatenation
        on_error: How to handle read errors - "warn" or "raise"

    Returns:
        Concatenated DataFrame, or empty DataFrame if no valid files found
    """
    parquet_files = list_parquet_files(directory)

    if not parquet_files:
        return pd.DataFrame()

    dfs = []
    for parquet_file in parquet_files:
        df = read_parquet_safe(parquet_file, on_error=on_error)
        if df is not None:
            if filter_fn is not None:
                df = filter_fn(df)
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)
