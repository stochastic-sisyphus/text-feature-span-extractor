"""Pydantic models for the active learning REST API.

This module defines the request/response models for the active learning
review interface used by human reviewers to provide feedback on low-confidence
predictions.

Usage:
    from invoices.api_models import QueueItem, LabelSubmission

    # Get items from review queue
    items: list[QueueItem] = get_queue()

    # Submit a label correction
    submission = LabelSubmission(field="InvoiceNumber", correct_value="INV-12345")
"""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class QueueItem(BaseModel):
    """An item in the active learning review queue.

    Represents a document field that needs human review due to low confidence
    or abstention.
    """

    doc_id: str = Field(..., description="Document identifier (fs:{sha256[:16]})")
    sha256: str = Field(..., description="Full SHA256 hash of the document")
    field: str = Field(..., description="Field name requiring review")
    raw_text: str | None = Field(None, description="Extracted raw text, if any")
    priority_score: float = Field(
        ..., ge=0.0, le=1.0, description="Priority score (0-1, higher = more urgent)"
    )
    priority_level: str = Field(
        ..., description="Priority level (critical, high, medium, low)"
    )
    ml_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="ML model confidence score"
    )
    reason: str = Field(..., description="Reason for review (e.g., 'low_confidence')")
    scores: dict[str, float] = Field(
        default_factory=dict, description="Detailed scoring breakdown"
    )


class LabelSubmission(BaseModel):
    """Request body for submitting a label correction.

    Used when a reviewer corrects a field value that was incorrectly
    extracted by the pipeline.
    """

    field: str = Field(..., description="Field name being corrected")
    correct_value: str = Field(..., description="Correct value for the field")
    correct_bbox: list[float] | None = Field(
        None,
        description="Correct bounding box [x0, y0, x1, y1] in normalized coords",
        min_length=4,
        max_length=4,
    )
    notes: str | None = Field(None, description="Optional notes from the reviewer")
    action: Literal["correct", "not_applicable", "not_in_document", "reject"] | None = (
        "correct"
    )


class ApprovalSubmission(BaseModel):
    """Request body for approving a prediction.

    Used when a reviewer confirms that a field's extracted value is correct.
    """

    field: str = Field(..., description="Field name being approved")
    notes: str | None = Field(None, description="Optional notes from the reviewer")


class DocumentDetail(BaseModel):
    """Detailed information about a document for review.

    Includes the document metadata and all field predictions for
    comprehensive review.
    """

    doc_id: str = Field(..., description="Document identifier")
    sha256: str = Field(..., description="Full SHA256 hash")
    pages: int = Field(..., ge=1, description="Number of pages in the document")
    predictions: dict[str, Any] = Field(
        ..., description="Field predictions with confidence scores"
    )
    pdf_url: str = Field(..., description="URL to access the PDF for viewing")


class ReviewerStats(BaseModel):
    """Statistics for a reviewer's activity.

    Used for tracking reviewer performance and workload.
    """

    total_reviews: int = Field(..., ge=0, description="Total number of reviews")
    reviews_today: int = Field(..., ge=0, description="Reviews completed today")
    accuracy_rate: float = Field(..., ge=0.0, le=1.0, description="Accuracy rate (0-1)")
    avg_review_time_seconds: float = Field(
        ..., ge=0.0, description="Average time per review in seconds"
    )


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )
    checks: dict[str, str] = Field(
        default_factory=dict, description="Individual component health checks"
    )


class QueueResponse(BaseModel):
    """Response for queue listing endpoint."""

    items: list[QueueItem] = Field(..., description="Queue items")
    total_count: int = Field(..., ge=0, description="Total items in queue")
    page: int = Field(1, ge=1, description="Current page number")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")


class LabelResponse(BaseModel):
    """Response for label submission."""

    success: bool = Field(..., description="Whether submission was successful")
    doc_id: str = Field(..., description="Document ID")
    field: str = Field(..., description="Field that was labeled")
    message: str = Field(..., description="Status message")


class ApprovalResponse(BaseModel):
    """Response for approval submission."""

    success: bool = Field(..., description="Whether approval was successful")
    doc_id: str = Field(..., description="Document ID")
    field: str = Field(..., description="Field that was approved")
    message: str = Field(..., description="Status message")


class FieldMetadata(BaseModel):
    """Metadata for a single extractable field from the contract schema."""

    name: str = Field(..., description="Field name (e.g., InvoiceNumber)")
    type: str = Field(..., description="Field type (e.g., id, amount, date, text)")
    required: bool = Field(..., description="Whether field is required")
    confidence_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence threshold for this field"
    )
    importance: float = Field(..., ge=0.0, le=1.0, description="Field importance (0-1)")


class FieldListResponse(BaseModel):
    """Response for schema fields endpoint."""

    fields: list[FieldMetadata] = Field(..., description="List of extractable fields")
    total: int = Field(..., ge=0, description="Total number of fields")
    schema_version: str = Field(..., description="Contract schema version")


class FieldDetail(BaseModel):
    """Full detail for a single schema field, including all tunable properties."""

    name: str = Field(..., description="Field name (e.g., InvoiceNumber)")
    type: str = Field(..., description="Field type (e.g., id, amount, date, text)")
    required: bool = Field(..., description="Whether field is required")
    description: str = Field(..., description="Human-readable description")
    confidence_threshold: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence threshold for this field"
    )
    importance: float = Field(..., ge=0.0, le=1.0, description="Field importance (0-1)")
    priority_bonus: float = Field(
        ..., description="Priority bonus applied during decoding"
    )
    keyword_proximal: bool = Field(
        ..., description="Whether field uses keyword proximity matching"
    )
    dataverse_column: str | None = Field(
        None, description="Dataverse output column name, if mapped"
    )
    computed: bool = Field(..., description="Whether field is computed (not extracted)")


class FieldUpdateRequest(BaseModel):
    """Request body for updating a schema field's tunable properties."""

    confidence_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="New confidence threshold (0-1)"
    )
    importance: float | None = Field(
        None, ge=0.0, le=1.0, description="New importance weight (0-1)"
    )
    priority_bonus: float | None = Field(
        None, description="New priority bonus for decoding"
    )
    keyword_proximal: bool | None = Field(
        None, description="Whether to use keyword proximity matching"
    )
    dataverse_column: str | None = Field(
        None, description="Dataverse column name (empty string clears it)"
    )
