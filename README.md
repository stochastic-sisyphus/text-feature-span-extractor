# Text-Layer Feature-Based Span Extractor for Invoices

> This repo is frozen at a pre-training state.   The trained version is private.  

Invoice extraction usually means picking your failure mode:    
- **Manual entry**:  slow, expensive, doesn't scale  
- **Vendor solutions**: $100K+ upfront, recurring costs, vendor lock-in  
- **Vision models**: GPU costs, latency, overkill for documents with native text  
- **OCR**:  introduces errors, unnecessary for PDFs with text layers  
- **Template rules**: brittle, breaks on layout changes, fails silently  

This system takes a different approach: deterministic feature-based extraction using native PDF text layers.   
No vendor lock-in.  No GPU costs. No OCR errors. No brittle rules.  

---

## Should You Even Bother?

**Worth reading if:**
- PDFs with native text layers (the entire point—leverage what's already there!)
- Multiple vendors with varying layouts
- Need for audit trails and provenance (compliance, financial reporting)
- In-house development capability
- Deterministic, reproducible outputs
- Paying a small team to manually process invoices and want to automate without recurring vendor costs
- Want accurate, auditable, reliable extraction without external dependencies

**Save yourself the time if:**
> Generally, if costs outweigh benefits, don't do it. (General life advice, also applies here.)
- Scanned documents without text layers (you'll need OCR first, sorry)
- Single vendor with stable templates (simple rules work fine—"stable" is the hint)
- Need immediate deployment with zero setup
- No ML/engineering resources (just buy a vendor solution)
- Unstructured documents (this is built for semi-structured invoices)

---

## Why Isn't This Just the First StackOverflow Result?

This was my thought exactly. 
My best guess is that most don't have to process thousands of invoices a month to genuinely deliberate on Pareto efficiency. This pipeline does also require combining expertise across feature engineering, optimization algorithms, production ML systems, and if it's even an actual word - PDF internals.

Most default to the less bad option:
- **Vendor solutions**: easy to buy (if your required volume is modest), no engineering required
- **Vision models**: feels more rigorous while still being able to work through implementation, feels much more liberating than external reliance

Both choices mean paying recurring costs forever instead of solving the problem once. This repo exists for teams that want to build it right and have the right inputs to use it  

---

## Requirements

**Python 3.10+**

Dependencies:  pdfplumber, pandas, pyarrow, numpy, xgboost, scipy, typer, python-dateutil, tqdm, doccano

---

## Installation

```bash
pip install -e .        # base install
pip install -e .[dev]   # with dev tools (pytest, black, isort, mypy, ruff)
```

**Note**:  `seed_pdfs/` et al (locations for private data) contains placeholder files.  Add your own PDFs to test.  

---

## Quick Start

```bash
# Single file
make run FILE=path/to/invoice.pdf
# Or:   invoicex run path/to/invoice.pdf

# Batch processing  
invoicex pipeline --seed-folder path/to/pdfs/

# Individual stages
invoicex ingest --seed-folder path/to/pdfs/
invoicex tokenize
invoicex candidates  
invoicex decode --none-bias 10. 0
invoicex emit
invoicex report --save

# Check status
invoicex status

# Outputs
ls artifacts/predictions/  # Contract JSONs here
```

---

## Pipeline Stages

### 1. Ingest - Content Addressing

**What**: PDFs stored by SHA256 hash.   Index maps `doc_id` (external ID) → `sha256` (content hash).

**Why**: Upload the same invoice from three different sources → stored once, no duplicates, no reconciliation hell.  
Most systems create three separate records and make YOU figure out they're the same document.

**Output**: `data/ingest/raw/{sha256}.pdf`, `data/ingest/index. parquet`

---

### 2. Tokenize - Stable IDs

**What**: pdfplumber extracts words with bounding boxes, fonts, typography.    
Each token gets:   `token_id = SHA1(doc_id + page + position + text + bbox)`

**Why**: Run the pipeline Tuesday, get output A. Run it Friday, get byte-identical output A.   
Most ML systems have random initialization somewhere - you can't reproduce last week's results even if you wanted to.  
Determinism means you can diff outputs, debug regressions, and trust your audit trail.

**Output**: `data/tokens/{sha256}.parquet`

---

### 3. Candidates - Bucket-Based Sampling

**What**: Generate ≤200 candidate spans per document, balanced across buckets:    
- **date_like**: digits/separators/month names  
- **amount_like**:  currency symbols, decimal patterns  
- **id_like**:  alphanumeric sequences  
- **keyword_proximal**: near "Invoice #", "Total", "Due Date"  
- **random_negative**: random spans for contrast  

Each candidate gets `features_v1`: geometric (bbox position, area), textual (char count, digit ratio), page position, bucket type.

**Why**: Template systems look for "total" in the bottom-right corner. What if the vendor moves it?  System breaks.    
This learns features:   "totals are usually large numbers, near the word 'total', in the lower half of the page."  
When layout changes, the features still match.  Vendor switches templates, you annotate 5 examples and retrain.

Random negatives prevent overfitting:   the model learns what ISN'T an invoice number, not just "pick any alphanumeric string."

Random sampling uses fixed seed from document SHA256 → same candidates every run.

**Output**: `data/candidates/{sha256}.parquet`

---

### 4. Decode - Hungarian Assignment with Abstention

**What**: Hungarian algorithm builds cost matrix `fields × (candidates + 1)`.  
Last column = NONE option with configurable bias (default 10.0).

If trained XGBoost models exist:   `cost = 1. 0 - model. predict_proba()`  
Otherwise: weak prior based on bucket + field type heuristics.

**Why**: Template systems break silently when the vendor changes layout.  You get the wrong value and don't know it.  
Vision models hallucinate - they'll give you SOME answer even when uncertain.  
This system says "I don't know" (ABSTAIN) when nothing matches confidently.  

High NONE bias → abstain more often → fewer wrong answers, more human review.  
Low NONE bias → guess more often → fewer manual reviews, more errors.

You choose the tradeoff. The system is honest about uncertainty.

**Output**: In-memory assignments (consumed by emit stage)

---

### 5. Emit - Normalization + Provenance

**What**:  Normalize assigned values:    
- Dates → ISO8601 via `dateutil.parser` (handles "March 3rd 2024", "03/03/24", "2024-03-03")  
- Amounts → decimal + currency code (handles "$1,234.56", "1.234,56 EUR", "¥1234")  
- IDs → cleaned alphanumeric  

Generate contract JSON with:   
- `value`: normalized output (or null)  
- `confidence`: model score (0.0 in unscored-baseline)  
- `status`: PREDICTED | ABSTAIN | MISSING  
- `provenance`: page number, bbox coordinates [0,1], token_span list  
- `raw_text`: original extracted text before normalization  

ABSTAIN cases → `data/review/queue. parquet` with full bbox for human correction. 

**Why**: Template systems have 50 date regex patterns and still miss "March 3rd, 2024".    
This handles all formats via `dateutil.parser`, then outputs ISO8601. One normalization path, predictable output.  

Provenance means:   when a value is wrong, you can trace it back to the exact PDF location, see what the model saw, understand why it failed.  
Most systems give you a number with no context.   This gives you page, bbox, and raw text.  Debuggable.  Auditable.

**Output**: `data/predictions/{sha256}.json`, `data/review/queue.parquet`

---

### 6. Report - Statistics

**What**: Per-field statistics:  PREDICTED / ABSTAIN / MISSING counts across all documents.  
Per-document metrics:  pages, tokens, candidates, status breakdown. 

**Why**: You need to know:  "Are we getting better?   Is this vendor harder than others?   Which fields need more training data?"  
Most systems are black boxes. This gives you metrics.  

**Output**: Console output, optional save to `data/logs/report_{timestamp}.json`

---

## CLI Commands

All commands available via `invoicex` or `run-pipeline`:

**Pipeline stages:**
- `ingest --seed-folder PATH` - Content-addressed PDF storage  
- `tokenize` - Extract tokens from all ingested PDFs  
- `candidates` - Generate candidate spans  
- `decode [--none-bias FLOAT]` - Hungarian assignment (default bias:   10.0)  
- `emit [--model-version STR]` - Generate JSONs + review queue  
- `report [--save]` - Statistics and metrics  

**Full pipeline:**
- `pipeline --seed-folder PATH [--none-bias FLOAT] [--save-report]` - Run all stages  

**Single file:**
- `run PDF_PATH [--output DIR]` - Extract single PDF → JSON  

**Status:**
- `status` - Show pipeline progress  

**Training workflow:**
- `doccano-pull` - Pull labels from Doccano API  
- `doccano-import --in FILE` - Import Doccano export  
- `doccano-align [--all] [--iou FLOAT]` - Align labels to candidates (default IoU: 0.3)  
- `train` - Train XGBoost models on aligned labels  

All commands accept `--verbose` / `-v` for detailed output.

---

## Data Layout

```
data/
├── ingest/
│   ├── raw/{sha256}.pdf          # Content-addressed PDFs
│   └── index.parquet             # doc_id ↔ sha256 crosswalk
├── tokens/{sha256}.parquet       # Token stores with stable IDs
├── candidates/{sha256}.parquet   # Candidate spans + features_v1
├── predictions/{sha256}.json     # Contract JSONs
├── review/queue.parquet          # ABSTAIN cases for human review
├── labels/                       # Doccano annotations
│   ├── raw/{sha256}.jsonl        # Raw label exports
│   ├── aligned/{sha256}.parquet  # IoU-aligned labels → candidates
│   └── index.parquet             # Label tracking
├── logs/
│   ├── version_log. jsonl         # Version stamps per run
│   └── report_{timestamp}.json   # Pipeline reports
└── models/current/               # Trained XGBoost models
```

---

## Determinism

**What**:  
- Token IDs:  `SHA1(doc_id + page + position + text + bbox)` → stable across runs  
- Candidate sampling: `random. seed(int(sha256[: 8], 16))` → fixed seed per document  
- Feature extraction: geometry + text analysis, no randomness  
- Assignment: Hungarian algorithm is deterministic given cost matrix  

Same PDF → identical tokens → identical candidates → identical assignments → identical JSON.  

**Why**:  Most ML pipelines can't reproduce yesterday's results because random seeds leak in somewhere.  
You retrain a model, outputs change for documents you didn't touch.  Debugging is impossible.  

This system:   same input = same output.  Always.   You can diff two runs and know EXACTLY what changed. 

**Test**:  `make determinism-check` runs pipeline twice, diffs outputs.  

---

## Idempotency

**What**: Every stage checks if output exists before processing:    
- Ingest: SHA256 already in index → skip  
- Tokenize: `data/tokens/{sha256}.parquet` exists → skip  
- Candidates: `data/candidates/{sha256}.parquet` exists → skip  
- Emit: `data/predictions/{sha256}.json` exists → skip  

**Why**: You add 10 new invoices to a folder of 1000. Most systems reprocess all 1010.    
This processes 10. Re-run any stage → only new work happens.   Fast iteration, no wasted compute. 

---

## Version Stamping

**What**: Every output includes 5 version stamps:  
- `contract_version`: schema semver + fingerprint (`1.0.0+{sha256[: 12]}`)  
- `feature_version`: `v1` (geometric + textual + bucket features)  
- `decoder_version`: `v1` (Hungarian + NONE bias)  
- `model_version`: `unscored-baseline` (or trained model ID)  
- `calibration_version`: `none` (or calibration run ID)  

**Why**: Six months from now, you find a bad extraction. Which version of the code produced it?  Which model?  Which schema?   
Most systems:   no idea. This system:  every output says exactly which code/model/schema created it.  
Reproducibility.   Auditability.  Debuggability.

**View version log**: `cat data/logs/version_log.jsonl`

---

## Training Models

```bash
# 1. Generate annotation tasks
cd tools/doccano
python tasks_gen.py --seed-folder ../../seed_pdfs --output ./output

# 2. Annotate in Doccano UI (install: pip install doccano)
doccano init && doccano createuser && doccano webserver
# Import output/tasks.json, annotate, export

# 3. Import annotations
invoicex doccano-import --in path/to/export. json

# 4. Align labels to candidates via IoU
invoicex doccano-align --all --iou 0.3

# 5. Train XGBoost models
invoicex train

# Models saved to data/models/current/
# Next decode run uses trained models instead of weak priors
```

**Normalization guards**: Each task includes `normalize_version` + `text_checksum`.  
Alignment fails if text doesn't match → prevents training on stale annotations. 

**Why**: Most annotation workflows:   annotate PDFs, export, train.   Weeks later, the text extraction changes.  Your annotations point to the wrong text.  Training fails silently or produces garbage.

This system:  annotations include checksums.   If extraction changes, alignment fails loudly.  You know immediately.

**See `tools/doccano/readme-doccano.md` for detailed annotation workflow.**

---

## Testing

```bash
pytest -v                              # All tests
pytest tests/test_token_determinism. py  # Verify stable IDs
pytest tests/test_idempotency.py        # Verify no duplicates
pytest tests/test_candidate_bounds.py   # Verify ≤200 candidates
pytest tests/test_contract_integrity. py # Verify JSON schema
pytest tests/test_review_queue. py       # Verify ABSTAIN → queue
pytest tests/test_label_alignment. py    # Verify IoU computation
pytest tests/golden/                    # Golden output regression

make test                # Full test suite
make test-golden         # Golden tests only
make lint                # ruff + mypy
make format              # black + isort
make determinism-check   # Two-run diff test
```

---

## Makefile Commands

```bash
make help              # Show all targets
make install           # Install package
make install-dev       # Install with dev dependencies
make test              # Run all tests
make test-golden       # Run golden tests only
make lint              # Run linting (ruff, mypy)
make format            # Format code (black, isort)
make clean             # Clean generated data
make run FILE=x        # Extract single PDF
make pipeline          # Run full pipeline on seed_pdfs/
make status            # Show pipeline status
make artifacts         # Copy outputs to artifacts/
make determinism-check # Test determinism with two runs
```

---

## Repository Structure

```
src/invoices/
  ├── ingest. py        # SHA256 content addressing
  ├── tokenize.py      # pdfplumber extraction + stable IDs
  ├── candidates.py    # Bucket-based span generation
  ├── decoder.py       # Hungarian + NONE bias assignment
  ├── emit.py          # Contract JSON generation + review queue
  ├── normalize.py     # Date/amount/ID normalization
  ├── labels.py        # Doccano IoU alignment
  ├── train.py         # XGBoost training
  ├── report. py        # Statistics
  ├── paths.py         # Path resolution
  ├── utils.py         # Version stamping, hashing, timers
  ├── cli.py           # Typer CLI
  └── __main__.py      # Allow python -m invoices

tests/
  ├── test_token_determinism.py      # Verify stable token IDs
  ├── test_idempotency.py            # Verify no duplicate processing
  ├── test_candidate_bounds.py       # Verify ≤200 candidates per doc
  ├── test_contract_integrity.py     # Verify JSON schema completeness
  ├── test_review_queue. py           # Verify ABSTAIN → review queue
  ├── test_label_alignment.py        # Verify IoU computation
  └── golden/test_golden_outputs.py  # Golden output regression

tools/doccano/
  ├── tasks_gen.py           # Generate annotation tasks with guards
  ├── readme-doccano.md      # Annotation workflow
  ├── config.json            # Doccano project config
  └── output/tasks.json      # Generated tasks

schema/contract. invoice. json  # 35 field definitions
scripts/run_pipeline.py       # Pipeline runner (used by Makefile)
Makefile                      # Build targets
pyproject.toml                # Package metadata
```

---

## License

MIT