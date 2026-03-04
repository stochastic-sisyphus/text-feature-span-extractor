"""Tests for the REST API."""

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from invoices.api import create_app
from invoices.api_models import (
    QueueItem,
)


@pytest.fixture
def temp_data_dir() -> Path:
    """Create a temporary data directory with test fixtures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create directory structure
        (data_dir / "predictions").mkdir()
        (data_dir / "ingest" / "raw").mkdir(parents=True)
        (data_dir / "review").mkdir()
        (data_dir / "labels" / "corrections").mkdir(parents=True)
        (data_dir / "labels" / "approvals").mkdir(parents=True)

        # Create a test prediction (use valid 64-char hex SHA256)
        test_sha256 = "a" * 64
        prediction = {
            "document_id": f"fs:{test_sha256}",
            "sha256": test_sha256,
            "pages": 2,
            "fields": {
                "InvoiceNumber": {
                    "value": "INV-001",
                    "confidence": 0.95,
                    "status": "PREDICTED",
                },
                "TotalAmount": {
                    "value": "1000.00",
                    "confidence": 0.75,
                    "status": "PREDICTED",
                },
            },
        }
        (data_dir / "predictions" / f"{test_sha256}.json").write_text(
            json.dumps(prediction)
        )

        # Create a test PDF
        (data_dir / "ingest" / "raw" / f"{test_sha256}.pdf").write_bytes(
            b"%PDF-1.4 test content"
        )

        # Create a test review queue
        queue = [
            {
                "doc_id": f"fs:{test_sha256}",
                "sha256": test_sha256,
                "field": "TotalAmount",
                "raw_text": "1000.00",
                "priority_score": 0.8,
                "priority_level": "high",
                "ml_confidence": 0.75,
                "reason": "low_confidence",
                "scores": {"base": 0.5},
            },
        ]
        (data_dir / "review" / "queue.json").write_text(json.dumps(queue))

        yield data_dir


@pytest.fixture
def client(temp_data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Create a test client with the app configured for testing."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    app = create_app(data_dir=temp_data_dir)
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Test health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestQueueEndpoint:
    """Tests for review queue endpoint."""

    def test_queue_returns_200(self, client: TestClient) -> None:
        """Test queue endpoint returns 200."""
        response = client.get("/api/v1/active-learning/queue")
        assert response.status_code == 200

    def test_queue_pagination(self, client: TestClient) -> None:
        """Test queue supports pagination."""
        response = client.get("/api/v1/active-learning/queue?page=1&page_size=10")
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 10

    def test_queue_filter_by_priority(self, client: TestClient) -> None:
        """Test queue supports priority filter."""
        response = client.get("/api/v1/active-learning/queue?priority_level=high")
        assert response.status_code == 200


class TestDocumentDetailEndpoint:
    """Tests for document detail endpoint."""

    def test_document_detail_returns_200(self, client: TestClient) -> None:
        """Test document detail endpoint returns 200 for existing document."""
        test_sha256 = "a" * 64
        response = client.get(f"/api/v1/active-learning/document/fs:{test_sha256}")
        assert response.status_code == 200

    def test_document_detail_returns_404(self, client: TestClient) -> None:
        """Test document detail endpoint returns 404 for missing document."""
        nonexistent_sha256 = "b" * 64
        response = client.get(
            f"/api/v1/active-learning/document/fs:{nonexistent_sha256}"
        )
        assert response.status_code == 404

    def test_document_detail_includes_predictions(self, client: TestClient) -> None:
        """Test document detail includes predictions."""
        test_sha256 = "a" * 64
        response = client.get(f"/api/v1/active-learning/document/fs:{test_sha256}")
        data = response.json()
        assert "predictions" in data
        assert "InvoiceNumber" in data["predictions"]


class TestLabelSubmissionEndpoint:
    """Tests for label submission endpoint."""

    def test_label_submission_returns_200(self, client: TestClient) -> None:
        """Test label submission returns 200."""
        test_sha256 = "a" * 64
        submission = {
            "field": "TotalAmount",
            "correct_value": "1050.00",
        }
        response = client.post(
            f"/api/v1/active-learning/document/fs:{test_sha256}/label",
            json=submission,
        )
        assert response.status_code == 200

    def test_label_submission_with_bbox(self, client: TestClient) -> None:
        """Test label submission with bounding box."""
        test_sha256 = "a" * 64
        submission = {
            "field": "TotalAmount",
            "correct_value": "1050.00",
            "correct_bbox": [0.1, 0.2, 0.3, 0.4],
        }
        response = client.post(
            f"/api/v1/active-learning/document/fs:{test_sha256}/label",
            json=submission,
        )
        assert response.status_code == 200

    def test_label_submission_validates_field(self, client: TestClient) -> None:
        """Test label submission requires field."""
        test_sha256 = "a" * 64
        submission = {
            "correct_value": "1050.00",
        }
        response = client.post(
            f"/api/v1/active-learning/document/fs:{test_sha256}/label",
            json=submission,
        )
        assert response.status_code == 422  # Validation error


class TestApprovalSubmissionEndpoint:
    """Tests for approval submission endpoint."""

    def test_approval_submission_returns_200(self, client: TestClient) -> None:
        """Test approval submission returns 200."""
        test_sha256 = "a" * 64
        submission = {
            "field": "InvoiceNumber",
        }
        response = client.post(
            f"/api/v1/active-learning/document/fs:{test_sha256}/approve",
            json=submission,
        )
        assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for reviewer stats endpoint."""

    def test_stats_returns_200(self, client: TestClient) -> None:
        """Test stats endpoint returns 200."""
        response = client.get("/api/v1/active-learning/stats")
        assert response.status_code == 200


class TestPDFEndpoint:
    """Tests for PDF serving endpoint."""

    def test_pdf_returns_200(self, client: TestClient) -> None:
        """Test PDF endpoint returns 200 for existing file."""
        test_sha256 = "a" * 64
        response = client.get(f"/api/v1/documents/{test_sha256}/pdf")
        assert response.status_code == 200

    def test_pdf_returns_404(self, client: TestClient) -> None:
        """Test PDF endpoint returns 404 for missing file."""
        nonexistent_sha256 = "b" * 64
        response = client.get(f"/api/v1/documents/{nonexistent_sha256}/pdf")
        assert response.status_code == 404

    def test_pdf_returns_correct_content_type(self, client: TestClient) -> None:
        """Test PDF endpoint returns application/pdf content type."""
        test_sha256 = "a" * 64
        response = client.get(f"/api/v1/documents/{test_sha256}/pdf")
        assert response.headers["content-type"] == "application/pdf"


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_preflight(self, client: TestClient) -> None:
        """Test CORS preflight request."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should not return 405 Method Not Allowed
        assert response.status_code != 405


class TestAPIModels:
    """Tests for Pydantic API models."""

    def test_queue_item_priority_bounds(self) -> None:
        """Test QueueItem priority_score bounds."""
        test_sha256 = "a" * 64
        with pytest.raises(ValueError):
            QueueItem(
                doc_id=f"fs:{test_sha256}",
                sha256=test_sha256,
                field="TotalAmount",
                priority_score=1.5,  # Invalid: > 1.0
                priority_level="high",
                ml_confidence=0.75,
                reason="low_confidence",
            )
