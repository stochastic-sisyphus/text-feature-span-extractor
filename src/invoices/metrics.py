"""Prometheus metrics for invoice extraction pipeline.

Defines application-level metrics scraped by Prometheus.
Container metrics (CPU, memory) come from cAdvisor, not here.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Pipeline metrics
documents_processed = Counter(
    "invoicex_documents_processed_total",
    "Total documents processed through pipeline",
    ["status"],  # success, error
)

pipeline_duration = Histogram(
    "invoicex_pipeline_duration_seconds",
    "Pipeline processing duration",
    ["stage"],  # ingest, tokenize, candidates, decode, emit
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)

pipeline_errors = Counter(
    "invoicex_pipeline_errors_total",
    "Total pipeline errors",
    ["stage"],  # ingest, tokenize, candidates, decode, emit
)

# Field extraction metrics
field_confidence = Histogram(
    "invoicex_confidence",
    "Field confidence scores",
    ["field", "status"],  # field name, PREDICTED/ABSTAIN/MISSING
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0],
)

field_status = Counter(
    "invoicex_field_status_total",
    "Field extraction status counts",
    ["field", "status"],  # field name, PREDICTED/ABSTAIN/MISSING
)

# Review queue metrics
review_queue_size = Gauge(
    "invoicex_review_queue_size",
    "Current review queue size",
)

review_queue_by_priority = Gauge(
    "invoicex_review_queue_by_priority",
    "Review queue items by priority level",
    ["priority"],  # urgent, medium, low
)

# Active learning metrics
labels_submitted = Counter(
    "invoicex_labels_submitted_total",
    "Total labels submitted",
    ["action"],  # approve, correct, reject, not_applicable, skip
)

corrections_submitted = Counter(
    "invoicex_corrections_total",
    "Total corrections submitted",
    ["field"],
)

approvals_submitted = Counter(
    "invoicex_approvals_total",
    "Total approvals submitted",
    ["field"],
)

# Model metrics
model_version_info = Info(
    "invoicex_model_version",
    "Current model version information",
)

model_fields_gauge = Gauge(
    "invoicex_model_fields_total",
    "Number of fields with trained models",
)

model_loaded = Gauge(
    "invoicex_model_loaded",
    "Whether ML model is currently loaded (1=yes, 0=no)",
)

training_runs = Counter(
    "invoicex_training_runs_total",
    "Total training runs",
    ["status"],  # success, error
)

training_duration = Histogram(
    "invoicex_training_duration_seconds",
    "Training duration",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Document routing
docs_auto_approved = Counter(
    "invoicex_docs_auto_approved_total",
    "Documents auto-approved (all fields above threshold)",
)

docs_needs_review = Counter(
    "invoicex_docs_needs_review_total",
    "Documents routed to human review",
)

# Prediction output
predictions_emitted = Counter(
    "invoicex_predictions_emitted_total",
    "Total predictions emitted",
    ["status"],  # PREDICTED, ABSTAIN, MISSING
)

# API metrics
api_requests = Counter(
    "invoicex_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status_code"],
)


def update_queue_metrics(queue_size: int, priority_counts: dict[str, int]) -> None:
    """Update review queue gauge metrics."""
    review_queue_size.set(queue_size)
    for priority, count in priority_counts.items():
        review_queue_by_priority.labels(priority=priority).set(count)


# Learning metrics (truth in metrics)
human_correction_rate = Gauge(
    "invoicex_human_correction_rate",
    "Rate of human corrections per field (corrections / total_reviews)",
    ["field"],
)

field_accuracy = Gauge(
    "invoicex_field_accuracy",
    "Field accuracy vs ground truth/corrections",
    ["field"],
)

validation_ndcg = Gauge(
    "invoicex_validation_ndcg",
    "Validation NDCG@1 from last training run",
    ["field"],
)

overfitting_gap = Gauge(
    "invoicex_overfitting_gap",
    "Train accuracy minus validation accuracy",
)

calibration_error = Gauge(
    "invoicex_calibration_error",
    "Expected Calibration Error - are confidence scores honest?",
)


def update_model_metrics(
    loaded: bool, field_count: int, model_type: str = "none"
) -> None:
    """Update model-related gauge metrics."""
    model_loaded.set(1 if loaded else 0)
    model_fields_gauge.set(field_count)
    model_version_info.info({"model_type": model_type, "loaded": str(loaded)})


def update_training_metrics(
    val_ndcg: float | None = None,
    train_accuracy: float | None = None,
    val_accuracy: float | None = None,
    per_field_ndcg: dict[str, float] | None = None,
) -> None:
    """Update training-related metrics after training completes.

    Args:
        val_ndcg: Aggregate validation NDCG@1 score (0-1), written as field="all"
        train_accuracy: Training accuracy (0-1)
        val_accuracy: Validation accuracy (0-1)
        per_field_ndcg: Per-field NDCG@1 scores keyed by field name
    """
    if val_ndcg is not None:
        validation_ndcg.labels(field="all").set(val_ndcg)

    if per_field_ndcg:
        for field_name, score in per_field_ndcg.items():
            validation_ndcg.labels(field=field_name).set(score)

    if train_accuracy is not None and val_accuracy is not None:
        gap = train_accuracy - val_accuracy
        overfitting_gap.set(gap)


def update_calibration_metrics(expected_calibration_error: float) -> None:
    """Update calibration error metric.

    Args:
        expected_calibration_error: ECE score (0-1, lower is better)
    """
    calibration_error.set(expected_calibration_error)


def update_correction_metrics(
    field: str, correction_rate: float, accuracy: float
) -> None:
    """Update per-field correction rate and accuracy.

    Args:
        field: Field name
        correction_rate: Rate of corrections (0-1)
        accuracy: Field accuracy (0-1)
    """
    human_correction_rate.labels(field=field).set(correction_rate)
    field_accuracy.labels(field=field).set(accuracy)
