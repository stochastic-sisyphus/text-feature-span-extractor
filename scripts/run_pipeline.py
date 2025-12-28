#!/usr/bin/env python3
"""CLI entrypoint for the invoice extraction pipeline."""

import json
import os
import sys
from pathlib import Path

import typer

# Add src to path to import invoices package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from invoices import (
    candidates,
    decoder,
    emit,
    ingest,
    labels,
    paths,
    report,
    tokenize,
    train,
    utils,
)

app = typer.Typer(
    name="run-pipeline",
    help="Invoice extraction pipeline CLI",
    add_completion=False,
)


@app.command(name="ingest")
def ingest_cmd(
    seed_folder: str = typer.Option(
        ..., "--seed-folder", help="Path to folder containing PDF files"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Ingest PDFs from seed folder into content-addressed storage."""
    try:
        print(f"Starting ingestion from: {seed_folder}")

        with utils.Timer("Ingestion"):
            newly_ingested = ingest.ingest_seed_folder(seed_folder)

        if newly_ingested > 0:
            print(f"✓ Successfully ingested {newly_ingested} new documents")
        else:
            print("✓ No new documents to ingest (all already present)")

        if verbose:
            docs = ingest.list_ingested_documents()
            print(f"Total documents in index: {len(docs)}")

    except Exception as e:
        print(f"✗ Ingestion failed: {e}")
        raise typer.Exit(1) from e


@app.command(name="tokenize")
def tokenize_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Tokenize all ingested documents."""
    try:
        print("Starting tokenization...")

        with utils.Timer("Tokenization"):
            results = tokenize.tokenize_all()

        if results:
            total_tokens = sum(results.values())
            print(
                f"✓ Tokenized {len(results)} documents, {total_tokens:,} total tokens"
            )

            if verbose:
                for sha256, token_count in results.items():
                    print(f"  {sha256[:16]}: {token_count:,} tokens")
        else:
            print("✓ No documents to tokenize")

    except Exception as e:
        print(f"✗ Tokenization failed: {e}")
        raise typer.Exit(1)


@app.command(name="candidates")
def candidates_cmd(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
) -> None:
    """Generate candidates for all documents."""
    try:
        print("Starting candidate generation...")

        with utils.Timer("Candidate generation"):
            results = candidates.generate_all_candidates()

        if results:
            total_candidates = sum(results.values())
            print(
                f"✓ Generated candidates for {len(results)} documents, {total_candidates:,} total candidates"
            )

            if verbose:
                for sha256, candidate_count in results.items():
                    print(f"  {sha256[:16]}: {candidate_count} candidates")
        else:
            print("✓ No documents to process")

    except Exception as e:
        print(f"✗ Candidate generation failed: {e}")
        raise typer.Exit(1)


@app.command(name="decode")
def decode_cmd(
    none_bias: float = typer.Option(
        10.0, "--none-bias", help="NONE assignment bias (higher = more abstains)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Decode all documents using Hungarian assignment."""
    try:
        print(f"Starting decoding with NONE bias {none_bias}...")

        with utils.Timer("Decoding"):
            results = decoder.decode_all_documents(none_bias)

        if results:
            print(f"✓ Decoded {len(results)} documents")

            if verbose:
                for sha256, assignments in results.items():
                    candidate_count = sum(
                        1
                        for a in assignments.values()
                        if a["assignment_type"] == "CANDIDATE"
                    )
                    none_count = sum(
                        1
                        for a in assignments.values()
                        if a["assignment_type"] == "NONE"
                    )
                    print(
                        f"  {sha256[:16]}: {candidate_count} predictions, {none_count} abstains"
                    )
        else:
            print("✓ No documents to decode")

    except Exception as e:
        print(f"✗ Decoding failed: {e}")
        raise typer.Exit(1)


@app.command(name="emit")
def emit_cmd(
    model_version: str
    | None = typer.Option(None, "--model-version", help="Override model version"),
    feature_version: str
    | None = typer.Option(None, "--feature-version", help="Override feature version"),
    decoder_version: str
    | None = typer.Option(None, "--decoder-version", help="Override decoder version"),
    calibration_version: str
    | None = typer.Option(
        None, "--calibration-version", help="Override calibration version"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Emit predictions JSON and update review queue."""
    try:
        # Set version overrides if provided
        if model_version:
            os.environ["MODEL_ID"] = model_version

        # Note: Other version overrides would require modifying utils constants
        # For now, only model_version is fully supported via environment

        print("Starting emission (normalization + JSON + review queue)...")

        with utils.Timer("Emission"):
            results = emit.emit_all_documents()

        if results:
            docs_processed = results.get("documents_processed", 0)
            total_predicted = results.get("total_predicted", 0)
            total_abstain = results.get("total_abstain", 0)
            total_review_entries = results.get("total_review_entries", 0)

            print(f"✓ Emitted predictions for {docs_processed} documents")
            print(f"  Predicted: {total_predicted}")
            print(f"  Abstain: {total_abstain}")
            print(f"  Review entries: {total_review_entries}")

            # Log to version log
            try:
                version_log_dir = paths.get_logs_dir()
                version_log_path = version_log_dir / "version_log.jsonl"

                log_entry = {
                    "timestamp": utils.get_current_utc_iso(),
                    "document_count": docs_processed,
                    **utils.get_version_info(),
                }

                # Append to JSONL
                with open(version_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if verbose:
                    print(f"Logged to: {version_log_path}")
            except Exception as log_error:
                print(f"Warning: Failed to log versions: {log_error}")

            if verbose:
                for sha256, result in results.get("results", {}).items():
                    if "error" in result:
                        print(f"  {result['doc_id']}: ERROR - {result['error']}")
                    else:
                        status_counts = result.get("status_counts", {})
                        print(f"  {result['doc_id']}: {status_counts}")
        else:
            print("✓ No documents to emit")

    except Exception as e:
        print(f"✗ Emission failed: {e}")
        raise typer.Exit(1)


@app.command(name="report")
def report_cmd(
    save: bool = typer.Option(False, "--save", help="Save report to logs directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Generate pipeline report with statistics."""
    try:
        print("Generating pipeline report...")

        report_data = report.generate_report()

        if save:
            report_path = report.save_report(report_data)
            print(f"Report saved to: {report_path}")

        print("✓ Report generation complete")

    except Exception as e:
        print(f"✗ Report generation failed: {e}")
        raise typer.Exit(1)


@app.command()
def pipeline(
    seed_folder: str = typer.Option(
        ..., "--seed-folder", help="Path to folder containing PDF files"
    ),
    none_bias: float = typer.Option(10.0, "--none-bias", help="NONE assignment bias"),
    save_report: bool = typer.Option(False, "--save-report", help="Save final report"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run the complete pipeline end-to-end."""
    try:
        print("=" * 80)
        print("INVOICE EXTRACTION PIPELINE - FULL RUN")
        print("=" * 80)

        # Step 1: Ingest
        print("\n1. INGESTION")
        print("-" * 40)
        with utils.Timer("Total ingestion"):
            newly_ingested = ingest.ingest_seed_folder(seed_folder)

        # Step 2: Tokenize
        print("\n2. TOKENIZATION")
        print("-" * 40)
        with utils.Timer("Total tokenization"):
            tokenize_results = tokenize.tokenize_all()

        # Step 3: Generate candidates
        print("\n3. CANDIDATE GENERATION")
        print("-" * 40)
        with utils.Timer("Total candidate generation"):
            candidate_results = candidates.generate_all_candidates()

        # Step 4: Decode
        print("\n4. DECODING")
        print("-" * 40)
        with utils.Timer("Total decoding"):
            decode_results = decoder.decode_all_documents(none_bias)

        # Step 5: Emit
        print("\n5. EMISSION")
        print("-" * 40)
        with utils.Timer("Total emission"):
            emit_results = emit.emit_all_documents()

        # Log version information (consistent with emit_cmd)
        try:
            version_log_dir = paths.get_logs_dir()
            version_log_path = version_log_dir / "version_log.jsonl"

            log_entry = {
                "timestamp": utils.get_current_utc_iso(),
                "document_count": emit_results.get("documents_processed", 0),
                **utils.get_version_info(),
            }

            # Append to JSONL
            with open(version_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            if verbose:
                print(f"Version logged to: {version_log_path}")
        except Exception as log_error:
            print(f"Warning: Failed to log versions: {log_error}")

        # Step 6: Report
        print("\n6. REPORTING")
        print("-" * 40)
        report_data = report.generate_report()

        # Always save report (was only saving if save_report=True)
        report_path = report.save_report(report_data)

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"✓ Processed {len(tokenize_results)} documents")
        print(f"✓ Generated {sum(candidate_results.values())} total candidates")
        print(
            f"✓ Emitted {emit_results.get('documents_processed', 0)} prediction files"
        )
        print(
            f"✓ Created {emit_results.get('total_review_entries', 0)} review queue entries"
        )

    except Exception as e:
        print(f"✗ Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show current pipeline status."""
    try:
        print("PIPELINE STATUS")
        print("=" * 50)

        # Check ingested documents
        indexed_docs = ingest.get_indexed_documents()
        print(f"Ingested documents: {len(indexed_docs)}")

        if not indexed_docs.empty:
            # Check tokenization status
            tokenized_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info["sha256"]
                tokens_path = paths.get_tokens_path(sha256)
                if tokens_path.exists():
                    tokenized_count += 1

            print(f"Tokenized documents: {tokenized_count}")

            # Check candidates status
            candidates_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info["sha256"]
                candidates_path = paths.get_candidates_path(sha256)
                if candidates_path.exists():
                    candidates_count += 1

            print(f"Documents with candidates: {candidates_count}")

            # Check predictions status
            predictions_count = 0
            for _, doc_info in indexed_docs.iterrows():
                sha256 = doc_info["sha256"]
                predictions_path = paths.get_predictions_path(sha256)
                if predictions_path.exists():
                    predictions_count += 1

            print(f"Documents with predictions: {predictions_count}")

            # Check review queue
            review_queue = emit.get_review_queue()
            print(f"Review queue entries: {len(review_queue)}")

        print("✓ Status check complete")

    except Exception as e:
        print(f"✗ Status check failed: {e}")
        raise typer.Exit(1)


@app.command(name="labels")
def labels_cmd() -> None:
    """Labels management commands."""
    print("Use subcommands: pull, import, align")
    raise typer.Exit(1)


@app.command(name="doccano-pull")
def doccano_pull_cmd() -> None:
    """Pull labels from Doccano API."""
    try:
        print("Pulling labels from Doccano...")

        result = labels.pull_labels()

        if result["status"] == "success":
            print(f"✓ Pulled {result['task_count']} annotations")
            print(f"Raw labels saved to: {result['raw_file']}")
        elif result["status"] == "skipped":
            print(f"✓ Pull skipped: {result['reason']}")
        else:
            print(f"✗ Pull failed: {result.get('error', 'unknown error')}")
            raise typer.Exit(1)

    except Exception as e:
        print(f"✗ Doccano pull failed: {e}")
        raise typer.Exit(1)


@app.command(name="doccano-import")
def doccano_import_cmd(
    path: str = typer.Option(..., "--in", help="Path to Doccano export file")
) -> None:
    """Import labels from local Doccano export."""
    try:
        print(f"Importing labels from: {path}")

        result = labels.import_labels(path)

        if result["status"] == "success":
            print(f"✓ Imported {result['task_count']} annotations")
            print(f"Saved to: {result['raw_file']}")
        else:
            print("✗ Import failed")
            raise typer.Exit(1)

    except Exception as e:
        print(f"✗ Doccano import failed: {e}")
        raise typer.Exit(1)


@app.command(name="doccano-align")
def doccano_align_cmd(
    all_files: bool = typer.Option(False, "--all", help="Process all raw label files"),
    iou_threshold: float = typer.Option(
        0.3, "--iou", help="IoU threshold for alignment"
    ),
) -> None:
    """Align Doccano labels with candidates using IoU matching."""
    try:
        print("Aligning labels with candidates...")

        result = labels.align_labels(iou_threshold=iou_threshold, all_files=all_files)

        if result["status"] == "success":
            print(f"✓ Aligned {result['total_aligned']} labels")
            print(f"IoU threshold: {result['iou_threshold']}")
            print(f"Files processed: {result['files_processed']}")

            for alignment in result["alignment_results"]:
                print(
                    f"  {alignment['source_file']} -> {alignment['aligned_file']} ({alignment['aligned_count']} aligned)"
                )
        elif result["status"] == "no_labels":
            print("✓ No labels directory found")
        elif result["status"] == "no_files":
            print("✓ No label files found")
        else:
            print("✗ Alignment failed")
            raise typer.Exit(1)

    except Exception as e:
        print(f"✗ Doccano alignment failed: {e}")
        raise typer.Exit(1)


@app.command(name="train")
def train_cmd() -> None:
    """Train XGBoost models on aligned labels."""
    try:
        print("Starting XGBoost training...")

        result = train.train_models()

        if result["status"] == "success":
            print("✓ Training complete")
            print(f"Models trained: {result['models_trained']}")
            print(
                f"Total examples: {result['total_pos']} positive, {result['total_neg']} negative"
            )
            print(f"Model file: {result['model_file']}")

            for field, stats in result["training_stats"].items():
                if "status" in stats:
                    print(f"  {field}: {stats['status']}")
                else:
                    print(
                        f"  {field}: {stats['pos_count']} pos, {stats['neg_count']} neg"
                    )
        elif result["status"] == "skipped":
            print(f"✓ Training skipped: {result['reason']}")
        else:
            print("✗ Training failed")
            raise typer.Exit(1)

    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def run(
    pdf_path: str = typer.Argument(..., help="Path to PDF file to process"),
    output_dir: str = typer.Option(
        "artifacts", "--output", "-o", help="Output directory for JSON results"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Extract data from a single PDF file → JSON output."""
    import shutil
    import tempfile

    pdf_file = Path(pdf_path)

    if not pdf_file.exists():
        print(f"✗ PDF file not found: {pdf_path}")
        raise typer.Exit(1)

    if not pdf_file.suffix.lower() == ".pdf":
        print(f"✗ File must be a PDF: {pdf_path}")
        raise typer.Exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Processing: {pdf_file.name}")

        # Create temporary directory for single file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy PDF to temp directory
            temp_pdf = temp_path / pdf_file.name
            shutil.copy2(pdf_file, temp_pdf)

            # Run pipeline on temp directory
            print("Running extraction pipeline...")

            with utils.Timer("Total extraction"):
                # Ingest
                newly_ingested = ingest.ingest_seed_folder(str(temp_path))
                if verbose:
                    print(f"  Ingested: {newly_ingested} documents")

                # Process
                tokenize_results = tokenize.tokenize_all()
                if verbose:
                    tokens = sum(tokenize_results.values()) if tokenize_results else 0
                    print(f"  Tokenized: {tokens:,} tokens")

                candidate_results = candidates.generate_all_candidates()
                if verbose:
                    candidates_count = (
                        sum(candidate_results.values()) if candidate_results else 0
                    )
                    print(f"  Generated: {candidates_count} candidates")

                decode_results = decoder.decode_all_documents()
                if verbose:
                    predictions = sum(
                        1
                        for doc_assignments in decode_results.values()
                        for assignment in doc_assignments.values()
                        if assignment["assignment_type"] == "CANDIDATE"
                    )
                    print(f"  Decoded: {predictions} predictions")

                emit_results = emit.emit_all_documents()

        # Find and copy the output JSON
        indexed_docs = ingest.get_indexed_documents()
        if indexed_docs.empty:
            print("✗ No documents were processed")
            raise typer.Exit(1)

        # There should be exactly one document
        for _, doc_info in indexed_docs.iterrows():
            sha256 = doc_info["sha256"]
            doc_id = doc_info["doc_id"]
            predictions_path = paths.get_predictions_path(sha256)

            if predictions_path.exists():
                # Copy to output directory with clean name
                output_file = output_path / f"{pdf_file.stem}.json"
                shutil.copy2(predictions_path, output_file)

                print("✓ Extraction complete")
                print(f"Output: {output_file}")

                if verbose:
                    # Show summary stats
                    with open(output_file, encoding="utf-8") as f:
                        result = json.load(f)

                    fields = result.get("fields", {})
                    predicted = sum(
                        1
                        for field_data in fields.values()
                        if field_data["status"] == "PREDICTED"
                    )
                    abstain = sum(
                        1
                        for field_data in fields.values()
                        if field_data["status"] == "ABSTAIN"
                    )
                    missing = sum(
                        1
                        for field_data in fields.values()
                        if field_data["status"] == "MISSING"
                    )

                    print(
                        f"  Fields: {predicted} predicted, {abstain} abstain, {missing} missing"
                    )
                    print(f"  Pages: {result.get('pages', 0)}")

                return

        print("✗ No prediction output generated")
        raise typer.Exit(1)

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
