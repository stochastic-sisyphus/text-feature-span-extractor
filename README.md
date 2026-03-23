![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![pdfplumber](https://img.shields.io/badge/PDF-text--layer--only-orange)

# text-feature-span-extractor

Invoice extraction usually means picking your failure mode:

- **Manual entry**: slow, expensive, doesn’t scale
- **Vendor solutions**: solutions is a bit of a misnomer - but it does lock you in. costs… so many. costs per unit, recurring, climbing costs ad nauseum
- **Vision models**: GPU costs, latency, overkill for documents with native text
- **OCR**: introduces errors, unnecessary for PDFs with text layers
- **Template rules**: brittle, breaks on layout changes, fails silently

This takes a different approach: deterministic feature-based extraction using native PDF text layers, confidence-gated output, and a human review loop that feeds corrections back into retraining.

> **Current accuracy**: 93% field-level extraction on real invoices (heuristic mode). Ranker training infrastructure is functional — quality gate requires 20+ labeled documents to enable ML scoring. Core extraction, decoder, and active learning loop are production-ready.

-----

## Should You Even Bother?

**Worth reading if:**

- PDFs with native text layers (the entire point, leverage what’s already there!)
- Multiple vendors with varying layouts
- Need for audit trails and provenance (compliance, financial reporting)
- In-house development capability
- Deterministic, reproducible outputs
- Paying a small team to manually process invoices and want to automate without recurring vendor costs
- Want accurate, auditable, reliable extraction without external dependencies

**Save yourself the time if:**

> Generally, if costs outweigh benefits, don’t do it. (General life advice, also applies here.)

- Scanned documents without text layers (you’ll need OCR first, sorry)
- Single vendor with stable templates (simple rules work fine, “stable” is the hint)
- Need immediate deployment with zero setup
- No ML/engineering resources (just buy a vendor solution)
- Unstructured documents (this is built for semi-structured invoices)

-----

## Why Isn’t This Just the First StackOverflow Result?

This was my thought exactly.
My best guess is that most don’t have to process thousands of invoices a month to genuinely deliberate on Pareto efficiency. This pipeline does also require combining expertise across feature engineering, optimization algorithms, production ML systems, and if it’s even an actual word - PDF internals.

Most mean paying recurring costs forever instead of solving the problem once.
Most get thrown at an API and get a creative interpretation of what should be deterministic. Though I am a fan of creative writing!

-----

## What this actually is

This is a system designed to be operated by people who process invoices, not people who train models. The constraint it was built around: the person doing review should not need to know anything about machine learning, and doing their normal review job should *be* the labeling job. Everything else follows from that.

What makes this more than a description is that the design is load-bearing at the technical level. SHA256 content addressing is simultaneously deduplication, artifact keying for every pipeline stage, and the random seed for deterministic candidate sampling - the same hash that prevents duplicate ingestion also guarantees that re-running candidates on the same document produces identical output. Stable token IDs (SHA1 of doc, page, position, text, bbox) are simultaneously reproducibility infrastructure and the thing that makes bbox disambiguation work during label alignment - the same ID that lets you diff two pipeline runs is the anchor that finds the right candidate when a value appears twice on the same page. The JSONL correction store is simultaneously an audit trail, the active learning training data, and an idempotent re-review mechanism - append-only means nothing is lost, dedup-on-read means re-reviewing a field updates it cleanly. The priority queue is simultaneously operational triage for the reviewer and uncertainty sampling for the model - the same score that determines what surfaces first also determines what label is most informative to collect. The quality gate in `models/manifest.json` is simultaneously a training validation check and an automated write gate - the same boolean that records whether the model passed CV is what the orchestrator reads before every Dataverse write to decide whether to trust it.

The Grafana interface is where operators watch pipeline health and where annotations happen. The same dashboards that show NDCG per field, correction rate, calibration error, and queue size by priority tier are the interface for doing the work that improves those numbers. No context switch, no separate labeling platform, no separate mental model. The active learning loop is not a feature added on top of the extraction system; it is why the extraction system is shaped the way it is.

A note on heuristics: the heuristic path is not regex. It is a full spatial scoring system - typed directional anchors, bucket affinity, section priors, reading-order bonuses, cross-page header suppression, colon-value detection. It operates on the same 59-feature space the ranker uses. What the ranker does is replace the hand-tuned weight vector with learned weights optimized against ground truth. The heuristic path is intelligent by design; it is the foundation the ranker builds on, not a fallback to something naive.

-----

## How it works

### Extraction

```
ingest → tokenize → candidates → decode → emit
```

Each stage writes intermediate artifacts keyed by SHA256. Same PDF in → byte-identical output out, every time.

**ingest** - content-addressed storage. Upload the same invoice from three sources, it’s stored once.

**tokenize** - pdfplumber text extraction. Tokens carry normalized bounding box coordinates, page index, line ID, style metadata. Token IDs are stable across runs.

**candidates** - generates scored spans per field using directional proximity to typed anchors (total, tax, date, ID, name), not just distance. Features encode spatial semantics: value to the *right* of a label on the same row vs. value *below* a column header are different signals, and the model learns them as such. Candidates are scored, soft-NMS suppresses overlapping spans by decaying lower-ranked neighbors rather than hard-cutting, and diversity sampling reserves slots for early-page candidates by type - so on a 149-page invoice, the summary amounts on page 1 don’t get buried under hundreds of candidates from later pages.

**decode** - modified Hungarian algorithm with a per-field NONE option. Fields go unassigned rather than forcing a bad match. Before assignment, a cross-field consistency check adjusts costs for candidate combinations where subtotal + tax ≈ total - internally consistent sets are cheaper to assign, so the algorithm naturally prefers them. XGBoost ranker scores when a model exists; spatial heuristics otherwise. Ranker scores are passed through sigmoid so confidence is calibrated, not binary. You set the NONE bias; you choose the tradeoff between wrong answers and human review. The system is honest about uncertainty.

**emit** - normalization (dates → ISO8601, amounts → decimal + currency, IDs → cleaned alphanumeric), schema validation, and confidence routing. High-confidence predictions write through. Low-confidence and ABSTAIN cases route to the review queue, prioritized by a formula that combines field importance, failure reason, and *uncertainty sampling* - fields near confidence 0.5 are prioritized because they carry the most information gain for retraining, not just because they’re wrong. Some fields (like currency code) are inferred from co-located amount fields rather than extracted directly, with provenance tracing back to the source candidate.

-----

### Active learning

This is active learning: the model identifies what it is uncertain about, surfaces those cases for labeling, and retrains on the result. The labeling interface exists to make that frictionless - it is not the point; it is how labels get collected.

Extractions are confidence-routed at emit time. Above the auto-approval threshold (configurable, default 0.85), documents write through without review. Below it, or on ABSTAIN, they enter the review queue - prioritized by an uncertainty sampling formula combining field importance, failure reason, and learning value, where learning value peaks at confidence 0.5 and falls toward zero at both extremes. The model already knows what to do at the tails; cases in the middle are where labels are most informative.

These surface in a custom Grafana-embedded labeling interface where the PDF renders alongside the fields. Four actions per field: ✓ correct (machine got it right - one click), fix (supply the correct value - this is where annotations and labels go), ✗ reject (wrong, and you are not providing a correction), not in doc (field is absent from this invoice - accounts for variable vendors and variable layouts).

Fix is the annotation path. You can type the value directly into the field box - this is how most labeling tools work, and it produces a usable label. But clicking spans in the PDF is both easier and produces a qualitatively better label. Hovering highlights individual token spans. Click them in order, confirm the sequence, click fix. The bbox comes with it automatically.

This matters because it makes golden labels - labels with ground truth spatial location, not just text - as easy to collect as silver labels. Silver labels (typed values) are what most annotation workflows produce because they are the path of least resistance. Golden labels with bbox are more valuable to the model because they encode where the value lives on the page, not just what it says. Span-clicking makes golden labels the default rather than the extra step.

It also resolves a real question that comes up in practice: if the same value appears in multiple places on the document - a date in the header, next to the amount, and again in a footer - which span do you click?

Click the one that is spatially related to the field label. The model is trained on directional spatial features: a value to the right of “Invoice Date:” on the same row scores differently from the same date floating in a page header. When you click the occurrence that is contextually correct - next to the relevant label, in the expected region for that field type - you are reinforcing the spatial pattern the model needs to learn, not just confirming the text value. Clicking the wrong occurrence teaches the model the wrong location signal, even if the text is right.

If you type the value instead and the same text appears multiple times, alignment finds all matching candidates and has no way to know which one you meant. It picks one. If you click, the bbox travels with the label and IoU matching identifies exactly which occurrence you selected. The model learns the right instance, not an ambiguous one.

You do not have to label every field. Submit whatever subset you reviewed - one field, ten fields, a mix of correct and fix and reject and not-in-doc - and click done to close the document and return to the queue. When you have reviewed as much as you want, click retrain. Partial submission is intentional; requiring complete passes would kill throughput on the exact people you need doing the labeling.

Re-reviewing a document is also supported: the correction store is append-only JSONL that accumulates all labels permanently. On read, it deduplicates per (doc_id, field) keeping the most recent entry, so if you re-review and change an answer, the updated label replaces the old one for that field without creating conflicts. All other labels from that document remain intact. The ranker trains on the full accumulated history every time.

Reject and not-in-doc actions are not discarded - they produce all-zeros label vectors (valid negative examples) that the ranker trains on, so “this field doesn’t exist in this invoice” is meaningful signal, not a no-op.

One click triggers the full loop: align corrections and approvals to candidates via field-type-aware matching (date variants, amount normalization, ID normalization, bbox disambiguation for duplicates) → fit a new `XGBRanker` with pairwise objective → cross-validate → quality gate → if it passes, promote models, re-decode all documents, regenerate predictions and queue. If the gate fails, heuristics continue; the system never blocks on a bad model. Training is locked against concurrent runs.

Cross-validation method is chosen based on what is statistically appropriate for the data volume. LOOCV is only valid when there are enough documents that holding one out leaves a meaningful training set - below that threshold, it produces unreliable estimates and false confidence. At low N, the system uses 2-fold CV with bbox coordinate jitter augmentation instead, which gives honest validation estimates and more training signal from limited examples. The quality gate threshold also adjusts to match. The system is designed to be useful from the first few labeled documents, not to require a large corpus before it does anything.

Between retrains, the adaptive layer runs at inference time: tokens within 0.15 normalized distance of confirmed correct candidates that appear across 2+ documents become new anchors for that field type, and Nelder-Mead optimization tunes 6 decoder weight parameters against ground truth. The heuristic path improves without waiting for a full retrain.

After each retrain, ECE (Expected Calibration Error) is computed from the correction/approval history. A model that assigns 0.9 confidence to everything it gets wrong is less useful than one that says 0.6. If accuracy data is available, the heuristic confidence base adjusts empirically rather than from config defaults.

-----

### Model loading

The decoder checks for models in three tiers: MLflow registry → local storage → heuristic fallback. Cache staleness is checked on every decode via a cheap `stat()` call so model updates propagate immediately without a restart. The quality gate lives in `models/manifest.json` - the orchestrator reads it before each Dataverse write and blocks the write if a model exists but failed the gate, rather than silently writing low-quality extractions. In heuristic-only mode (no manifest), writes proceed normally.

-----

## Deployment

Full stack via Docker Compose: extraction pipeline, review API, Grafana, MLflow, nginx, and an optional LGTM observability profile.

```bash
cp .env.template .env

docker compose up -d                          # core stack
docker compose --profile observability up -d  # + tracing/metrics
docker compose run process-batch              # one-shot batch run
```

Everything is parameterized through environment variables. The same image runs locally against a filesystem or in Azure against SharePoint, Blob Storage, and Dataverse - one variable flips all three backends at once.

-----

## Services

|Service   |Role                                            |
|----------|------------------------------------------------|
|`invoicex`|Extraction pipeline, continuous polling         |
|`api`     |FastAPI review interface (proxied via nginx)    |
|`grafana` |Review UI + dashboards, embedded labeling plugin|
|`mlflow`  |Model registry, experiment tracking             |
|`nginx`   |Reverse proxy, single entrypoint on `:8888`     |
|`postgres`|Shared backend for Grafana + MLflow             |

Observability profile adds OTel Collector, Tempo, Loki, Prometheus.

-----

## Backends

|Component      |Local     |Azure       |
|---------------|----------|------------|
|Document source|filesystem|SharePoint  |
|Storage        |local     |Blob Storage|
|Output / ledger|mock      |Dataverse   |

-----

## Layout

```
src/invoices/
├── orchestrator.py     # discovery → dedup → pipeline → ledger
├── tokenize.py
├── candidates/         # span generation, features, spatial scoring
│   ├── proximity.py    # typed directional anchors (total, tax, date, id, name)
│   ├── features.py     # soft-NMS, diversity sampling, section priors
│   └── patterns.py     # bucket classifiers
├── decoder.py          # Hungarian + cross-field consistency + three-tier model loading
├── emit.py             # normalization, uncertainty-sampled priority queue, computed fields
├── adaptive.py         # learned anchors, Nelder-Mead weight optimization
├── calibration.py      # ECE, empirical accuracy, heuristic base adjustment
├── ranker.py           # XGBRanker
├── train.py            # training loop, quality gate, manifest
├── labels.py           # correction ingestion, IoU alignment
├── api.py              # FastAPI review interface, retrain endpoint
├── azure/              # SharePoint, Dataverse, Blob connectors
├── storage/            # local / blob backends
├── config.py           # all parameters, env-backed
└── schema/
    └── contract.invoice.json
```

-----

Python 3.10+. Core: `pdfplumber`, `pandas`, `xgboost`, `scipy`, `fastapi`. Azure extras in `pyproject.toml`.
