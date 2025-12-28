#!/usr/bin/env python3
"""Generate Doccano tasks with normalization guards."""

import json
import shutil
import sys
from hashlib import sha256
from pathlib import Path

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from invoices import normalize


def compute_pdf_sha256(pdf_path: Path) -> str:
    """Compute SHA256 of PDF file."""
    with open(pdf_path, 'rb') as f:
        return sha256(f.read()).hexdigest()


def extract_normalized_text(pdf_path: Path) -> str:
    """Extract and normalize text from PDF using pipeline normalizer."""
    import pdfplumber

    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            text_parts.append(f"[[PAGE {page_idx}]]")

            # Extract text with word-level precision
            words = page.extract_words()
            if words:
                page_text = " ".join(word["text"] for word in words)
                text_parts.append(page_text)

            text_parts.append("\n")

    raw_text = "\n".join(text_parts)

    # Apply pipeline normalization
    normalized_text, _ = normalize.normalize_text(raw_text)

    return normalized_text or ""


def generate_tasks(seed_folder: str, output_dir: str = "tools/doccano/output") -> dict:
    """
    Generate Doccano tasks with normalization guards.

    Args:
        seed_folder: Directory containing PDF files
        output_dir: Output directory for tasks and PDFs

    Returns:
        Generation summary
    """
    seed_path = Path(seed_folder)
    output_path = Path(output_dir)

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed folder not found: {seed_folder}")

    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    labeled_pdfs_dir = output_path / "labeled_pdfs"
    labeled_pdfs_dir.mkdir(exist_ok=True)

    tasks = []
    processed_files = []

    for pdf_file in seed_path.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")

        try:
            # Compute PDF hash
            pdf_sha256 = compute_pdf_sha256(pdf_file)
            doc_id = f"fs:{pdf_sha256[:16]}"

            # Extract normalized text with guards
            normalized_text = extract_normalized_text(pdf_file)
            text_length = len(normalized_text)
            text_checksum = sha256(normalized_text.encode()).hexdigest()[:16]

            # Copy PDF to labeled directory
            labeled_pdf_path = labeled_pdfs_dir / f"{pdf_sha256}.pdf"
            shutil.copy2(pdf_file, labeled_pdf_path)

            # Create Doccano task
            task = {
                "data": {
                    "doc_id": doc_id,
                    "text": normalized_text,
                    "pdf_file": str(labeled_pdf_path),
                    "original_filename": pdf_file.name,
                    "sha256": pdf_sha256,
                    # Normalization guards
                    "normalize_version": normalize.NORMALIZE_VERSION,
                    "text_length": text_length,
                    "text_checksum": text_checksum,
                }
            }

            tasks.append(task)
            processed_files.append({
                "filename": pdf_file.name,
                "doc_id": doc_id,
                "sha256": pdf_sha256,
                "text_length": text_length
            })

        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")
            continue

    # Write tasks.json
    tasks_file = output_path / "tasks.json"
    with open(tasks_file, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)

    # Write summary
    summary_file = output_path / "generation_summary.json"
    summary = {
        "normalize_version": normalize.NORMALIZE_VERSION,
        "total_tasks": len(tasks),
        "processed_files": processed_files,
        "output_paths": {
            "tasks": str(tasks_file),
            "labeled_pdfs": str(labeled_pdfs_dir),
            "summary": str(summary_file)
        }
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(tasks)} tasks")
    print(f"Tasks file: {tasks_file}")
    print(f"PDFs copied to: {labeled_pdfs_dir}")

    return summary


def main():
    """CLI entry point for task generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Label Studio tasks")
    parser.add_argument("--seed-folder", required=True, help="Directory containing PDF files")
    parser.add_argument("--output", default="tools/labelstudio/output", help="Output directory")

    args = parser.parse_args()

    try:
        generate_tasks(args.seed_folder, args.output)
        print("✓ Task generation complete")

    except Exception as e:
        print(f"✗ Task generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
