"""Expected Calibration Error (ECE), calibration mapping, and empirical accuracy.

ECE measures how well confidence scores match actual accuracy. Lower is better.
The calibration mapping corrects raw confidence scores via isotonic regression
or Platt scaling so that predicted probabilities align with observed accuracy.
"""

from collections import defaultdict
from typing import Any

import numpy as np

from .config import Config
from .logging import get_logger
from .types import CalibrationState

logger = get_logger(__name__)

# Dunder-style provenance key — can never collide with a schema field name
# (schema fields are CamelCase identifiers, never double-underscore-bounded).
# Both calibration_mapping and field_blend_weights write this key; readers
# accept the legacy "model_id" spelling for rows persisted before this change.
MODEL_ID_KEY: str = "__model_id__"


def compute_expected_calibration_error(
    confidences: list[float],
    predictions: list[str],
    ground_truths: list[str],
    n_bins: int = 10,
) -> float:
    """Compute ECE: weighted average |confidence - accuracy| across bins."""
    if len(confidences) != len(predictions) or len(predictions) != len(ground_truths):
        raise ValueError("All inputs must have same length")
    if not confidences:
        return 0.0

    conf = np.array(confidences, dtype=np.float64)
    correct = np.array(
        [p == g for p, g in zip(predictions, ground_truths, strict=True)], dtype=bool
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            in_bin = (conf >= bin_edges[i]) & (conf <= bin_edges[i + 1])
        else:
            in_bin = (conf >= bin_edges[i]) & (conf < bin_edges[i + 1])

        bin_size = np.sum(in_bin)
        if bin_size == 0:
            continue
        ece += (bin_size / len(conf)) * abs(
            float(np.mean(conf[in_bin])) - float(np.mean(correct[in_bin]))
        )

    return float(ece)


def compute_ece_from_predictions(predictions: list[dict[str, Any]]) -> float:
    """Compute ECE from prediction dicts with confidence/predicted_value/ground_truth_value."""
    if not predictions:
        return 0.0
    return compute_expected_calibration_error(
        [p["confidence"] for p in predictions],
        [p["predicted_value"] for p in predictions],
        [p["ground_truth_value"] for p in predictions],
    )


def compute_ece_with_data(samples: list[dict[str, Any]]) -> float | None:
    """Compute ECE from pre-fetched samples. Returns None if < 5 samples."""
    filtered = [
        s
        for s in samples
        if "confidence" in s and "predicted_value" in s and "ground_truth_value" in s
    ]
    if len(filtered) < 5:
        return None
    return compute_ece_from_predictions(filtered)


# ---------------------------------------------------------------------------
# Pure *_with_data variant (no file I/O)
# ---------------------------------------------------------------------------


def compute_empirical_accuracy_with_data(
    corrections: list[Any],
    approvals: list[Any],
) -> CalibrationState:
    """Compute empirical accuracy from pre-fetched label data.

    Counts unique (doc_id, field) pairs — not raw event count — so that
    multiple correction events for the same field don't skew the ratio.

    Args:
        corrections: Label events representing wrong predictions (have .doc_id, .field)
        approvals: Label events representing correct predictions (have .doc_id, .field)

    Returns:
        CalibrationState with empirical_accuracy (None if no data)
    """
    n_approvals = len({(a.doc_id, a.field) for a in approvals}) if approvals else 0
    n_corrections = (
        len({(c.doc_id, c.field) for c in corrections}) if corrections else 0
    )
    total = n_approvals + n_corrections

    if total == 0:
        return CalibrationState()

    return CalibrationState(empirical_accuracy=n_approvals / total)


# ---------------------------------------------------------------------------
# Calibration mapping: compute + apply
# ---------------------------------------------------------------------------


def _fit_calibration(
    samples: list[dict[str, Any]],
    method: str,
) -> dict[str, Any] | None:
    """Run calibration fit on a list of samples. Returns None if insufficient data."""
    if len(samples) < Config.calibration_min_samples:
        return None

    confidences = np.array([s["confidence"] for s in samples], dtype=np.float64)
    correct = np.array(
        [s["predicted_value"] == s["ground_truth_value"] for s in samples],
        dtype=np.float64,
    )

    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(confidences, correct)
        return {
            "method": "isotonic",
            "x": ir.X_thresholds_.tolist(),
            "y": ir.y_thresholds_.tolist(),
        }

    if method == "platt":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression()
        lr.fit(confidences.reshape(-1, 1), correct)
        return {
            "method": "platt",
            "a": float(-lr.coef_[0][0]),
            "b": float(-lr.intercept_[0]),
        }

    logger.warning("unknown_calibration_method", method=method)
    return None


def compute_calibration_mapping(
    samples: list[dict[str, Any]],
    method: str = "isotonic",
) -> dict[str, Any]:
    """Compute a per-field calibration mapping from labeled data.

    Returns a dict with '_global' (fallback) and per-field entries.
    Each entry is a fit dict or None if insufficient samples.
    Includes MODEL_ID_KEY (``"__model_id__"``) for provenance tracking.
    """
    filtered = [
        s
        for s in samples
        if "confidence" in s and "predicted_value" in s and "ground_truth_value" in s
    ]

    result: dict[str, Any] = {MODEL_ID_KEY: Config.model_id}

    # Per-field grouping
    by_field: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in filtered:
        f = s.get("field")
        if f:
            by_field[f].append(s)

    # Per-field fits
    for field_name, field_samples in by_field.items():
        result[field_name] = _fit_calibration(field_samples, method)

    # Global fallback (all samples pooled)
    result["_global"] = _fit_calibration(filtered, method)

    return result


def apply_calibration(
    raw_confidence: float,
    mapping: dict[str, Any],
    field: str | None = None,
) -> float:
    """Apply a persisted calibration mapping to a raw confidence score.

    Supports both per-field format (has '_global' key) and legacy flat format.
    The result is clamped to [0, 1]. Unknown mapping methods return
    raw_confidence unchanged.
    """
    # Per-field format: resolve to field-specific or global fallback
    if "_global" in mapping:
        resolved = mapping.get(field) if field else None
        if resolved is None:
            resolved = mapping["_global"]
        if resolved is None:
            return raw_confidence
        mapping = resolved

    method = mapping.get("method")

    if method == "isotonic":
        calibrated = float(np.interp(raw_confidence, mapping["x"], mapping["y"]))
    elif method == "platt":
        a = mapping["a"]
        b = mapping["b"]
        calibrated = float(1.0 / (1.0 + np.exp(a * raw_confidence + b)))
    else:
        return raw_confidence

    return float(np.clip(calibrated, 0.0, 1.0))
