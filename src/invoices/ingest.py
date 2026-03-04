"""PDF ingestion module for content-addressed storage and indexing."""

import time
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from . import io_utils, paths, utils
from .config import Config
from .exceptions import IngestError, InvalidPDFError
from .logging import get_logger
from .metrics import pipeline_duration, pipeline_errors

logger = get_logger(__name__)


# =============================================================================
# ERROR HANDLING HELPERS
# =============================================================================


def _log_index_read_failed(path: str, error: Exception) -> None:
    """Log an index read failure.

    Args:
        path: Path to the index file
        error: The exception that occurred
    """
    logger.warning(
        "index_read_failed",
        path=path,
        error_type=type(error).__name__,
        reason=str(error),
        action="starting_fresh",
    )


def _log_pdf_processing_error(filename: str, error: Exception) -> None:
    """Log a PDF processing error.

    Args:
        filename: Name of the PDF file
        error: The exception that occurred
    """
    logger.error(
        "pdf_processing_failed",
        filename=filename,
        error_type=type(error).__name__,
        reason=str(error),
    )


def ingest_seed_folder(seed_folder: str) -> int:
    """
    Mirror PDFs into content-addressed storage and index them.

    Args:
        seed_folder: Path to folder containing PDF files

    Returns:
        Number of newly ingested documents
    """
    start_time = time.perf_counter()
    try:
        seed_path = Path(seed_folder)
        if not seed_path.exists():
            raise ValueError(f"Seed folder does not exist: {seed_folder}")

        # Ensure directories exist
        paths.ensure_directories()

        # Load existing index if it exists
        index_path = paths.get_ingest_index_path()
        existing_sha256s = set()

        existing_df = io_utils.read_parquet_safe(
            index_path,
            on_error="warn",
            logger_context={"action": "starting_fresh"},
        )
        if existing_df is not None:
            if "sha256" in existing_df.columns:
                existing_sha256s = set(existing_df["sha256"].tolist())
            else:
                logger.warning("index_missing_sha256_column", path=str(index_path))

        # Process PDF files
        new_rows = []
        newly_ingested_count = 0

        # Find all PDF files
        pdf_files = list(seed_path.glob("*.pdf")) + list(seed_path.glob("*.PDF"))

        if not pdf_files:
            logger.info("no_pdf_files_found", seed_folder=seed_folder)
            return 0

        logger.info("processing_pdfs", count=len(pdf_files), seed_folder=seed_folder)

        for pdf_file in pdf_files:
            doc_start = time.perf_counter()
            try:
                # Read file bytes
                with open(pdf_file, "rb") as f:
                    pdf_bytes = f.read()

                # Compute SHA256
                sha256 = utils.compute_sha256(pdf_bytes)

                # Check if already ingested
                if sha256 in existing_sha256s:
                    logger.debug(
                        "skipping_already_ingested",
                        filename=pdf_file.name,
                        sha256=sha256[:16],
                    )
                    continue

                # Write to content-addressed storage.
                # SharePoint mode: bytes arrive via orchestrator.discover_remote()
                # and are written there. Skip the disk write here to avoid
                # double-writing when DOCUMENT_SOURCE=sharepoint.
                raw_pdf_path = paths.get_raw_pdf_path(sha256)

                if Config.DOCUMENT_SOURCE != "sharepoint":
                    # Only write if doesn't exist (extra safety)
                    if not raw_pdf_path.exists():
                        with open(raw_pdf_path, "wb") as fw:
                            fw.write(pdf_bytes)

                        logger.info(
                            "stored_pdf",
                            filename=pdf_file.name,
                            sha256=sha256[:16],
                            bytes=len(pdf_bytes),
                        )

                # Create index row
                doc_id = f"fs:{sha256[:16]}"
                row = {
                    "doc_id": doc_id,
                    "sha256": sha256,
                    "sources": ["fs"],
                    "filename": pdf_file.name,
                    "imported_at": utils.get_current_utc_iso(),
                }
                new_rows.append(row)
                newly_ingested_count += 1

                doc_duration = time.perf_counter() - doc_start
                pipeline_duration.labels(stage="ingest").observe(doc_duration)

            except OSError as e:
                # File system errors (permission denied, disk full, etc.)
                _log_pdf_processing_error(pdf_file.name, e)
                pipeline_errors.labels(stage="ingest").inc()

                continue
            except (InvalidPDFError, IngestError) as e:
                # Typed ingestion errors
                _log_pdf_processing_error(pdf_file.name, e)
                pipeline_errors.labels(stage="ingest").inc()

                continue
            except (ValueError, TypeError) as e:
                # Data validation errors (hash computation, etc.)
                _log_pdf_processing_error(pdf_file.name, e)
                pipeline_errors.labels(stage="ingest").inc()

                continue

        # Update index with new rows
        if new_rows:
            new_df = pd.DataFrame(new_rows)

            # Append with sources column list type transform
            def ensure_sources_list(df: pd.DataFrame) -> pd.DataFrame:
                df["sources"] = df["sources"].apply(list)
                return df

            io_utils.append_to_parquet(
                new_df, index_path, transform=ensure_sources_list
            )
            logger.info("index_updated", new_entries=len(new_rows))

        logger.info("ingestion_complete", newly_ingested=newly_ingested_count)
        return newly_ingested_count
    except Exception:
        pipeline_errors.labels(stage="ingest").inc()
        raise
    finally:
        duration = time.perf_counter() - start_time
        pipeline_duration.labels(stage="ingest").observe(duration)


def register_document(sha256: str, filename: str, source: str = "sharepoint") -> str:
    """Register a document in the index without writing PDF bytes to disk.

    Used by the SharePoint path where bytes are managed externally (already
    written to raw storage by the orchestrator). Writes only the index row.

    Args:
        sha256: Full SHA256 hash of the PDF
        filename: Original filename (for display/logging)
        source: Source label (default "sharepoint")

    Returns:
        doc_id (e.g. "fs:abc123def456")
    """
    paths.ensure_directories()
    index_path = paths.get_ingest_index_path()

    # Idempotent: skip if already indexed
    existing_df = io_utils.read_parquet_safe(index_path, on_error="empty")
    if existing_df is not None and not existing_df.empty:
        if sha256 in existing_df["sha256"].values:
            matches = existing_df[existing_df["sha256"] == sha256]
            return str(matches.iloc[0]["doc_id"])

    doc_id = f"fs:{sha256[:16]}"
    row = {
        "doc_id": doc_id,
        "sha256": sha256,
        "sources": [source],
        "filename": filename,
        "imported_at": utils.get_current_utc_iso(),
    }
    new_df = pd.DataFrame([row])

    def ensure_sources_list(df: pd.DataFrame) -> pd.DataFrame:
        df["sources"] = df["sources"].apply(list)
        return df

    io_utils.append_to_parquet(new_df, index_path, transform=ensure_sources_list)
    logger.info("registered_document", sha256=sha256[:16], source=source)
    return doc_id


def get_indexed_documents() -> pd.DataFrame:
    """Get all indexed documents from the index."""
    index_path = paths.get_ingest_index_path()
    df = io_utils.read_parquet_safe(index_path, on_error="empty")

    if df is None:
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=["doc_id", "sha256", "sources", "filename", "imported_at"]
        )

    return df


def get_document_info(sha256: str) -> dict | None:
    """Get document information by SHA256."""
    df = get_indexed_documents()

    if df.empty:
        return None

    matches = df[df["sha256"] == sha256]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()  # type: ignore[no-any-return]


def list_ingested_documents() -> list[dict]:
    """List all ingested documents with basic info."""
    df = get_indexed_documents()

    if df.empty:
        return []

    return df.to_dict("records")  # type: ignore[no-any-return]
