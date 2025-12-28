"""PDF ingestion module for content-addressed storage and indexing."""

from pathlib import Path

import pandas as pd

from . import paths, utils


def ingest_seed_folder(seed_folder: str) -> int:
    """
    Mirror PDFs into content-addressed storage and index them.

    Args:
        seed_folder: Path to folder containing PDF files

    Returns:
        Number of newly ingested documents
    """
    seed_path = Path(seed_folder)
    if not seed_path.exists():
        raise ValueError(f"Seed folder does not exist: {seed_folder}")

    # Ensure directories exist
    paths.ensure_directories()

    # Load existing index if it exists
    index_path = paths.get_ingest_index_path()
    existing_sha256s = set()

    if index_path.exists():
        try:
            existing_df = pd.read_parquet(index_path)
            existing_sha256s = set(existing_df["sha256"].tolist())
        except Exception as e:
            print(f"Warning: Could not read existing index: {e}")

    # Process PDF files
    new_rows = []
    newly_ingested_count = 0

    # Find all PDF files
    pdf_files = list(seed_path.glob("*.pdf")) + list(seed_path.glob("*.PDF"))

    if not pdf_files:
        print(f"No PDF files found in {seed_folder}")
        return 0

    print(f"Processing {len(pdf_files)} PDF files from {seed_folder}")

    for pdf_file in pdf_files:
        try:
            # Read file bytes
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()

            # Compute SHA256
            sha256 = utils.compute_sha256(pdf_bytes)

            # Check if already ingested
            if sha256 in existing_sha256s:
                print(f"Skipping {pdf_file.name} (already ingested: {sha256[:16]})")
                continue

            # Write to content-addressed storage
            raw_pdf_path = paths.get_raw_pdf_path(sha256)

            # Only write if doesn't exist (extra safety)
            if not raw_pdf_path.exists():
                with open(raw_pdf_path, "wb") as f:
                    f.write(pdf_bytes)

                print(
                    f"Stored {pdf_file.name} -> {sha256[:16]}.pdf ({len(pdf_bytes):,} bytes)"
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

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
            continue

    # Update index with new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)

        if index_path.exists():
            # Append to existing index
            existing_df = pd.read_parquet(index_path)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        # Ensure sources column is list type for Parquet
        combined_df["sources"] = combined_df["sources"].apply(list)

        # Write updated index
        combined_df.to_parquet(index_path, index=False)
        print(f"Updated index with {len(new_rows)} new entries")

    print(f"Ingestion complete: {newly_ingested_count} new documents")
    return newly_ingested_count


def get_indexed_documents() -> pd.DataFrame:
    """Get all indexed documents from the index."""
    index_path = paths.get_ingest_index_path()

    if not index_path.exists():
        # Return empty DataFrame with correct schema
        return pd.DataFrame(
            columns=["doc_id", "sha256", "sources", "filename", "imported_at"]
        )

    return pd.read_parquet(index_path)


def get_document_info(sha256: str) -> dict | None:
    """Get document information by SHA256."""
    df = get_indexed_documents()

    if df.empty:
        return None

    matches = df[df["sha256"] == sha256]

    if matches.empty:
        return None

    return matches.iloc[0].to_dict()


def list_ingested_documents() -> list[dict]:
    """List all ingested documents with basic info."""
    df = get_indexed_documents()

    if df.empty:
        return []

    return df.to_dict("records")
