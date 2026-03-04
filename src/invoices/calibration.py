"""Expected Calibration Error (ECE) and empirical accuracy from labeled data.

ECE measures how well confidence scores match actual accuracy. Lower is better.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

from .logging import get_logger

logger = get_logger(__name__)


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


def _read_labeled_samples(data_dir: Path) -> list[dict[str, Any]]:
    """Read corrections and approvals, return (confidence, predicted, ground_truth) triples.

    Shared reader for both ECE and empirical accuracy computation.
    """
    from . import paths

    samples: list[dict[str, Any]] = []

    for label_type in ("corrections", "approvals"):
        filepath = data_dir / "labels" / label_type / f"{label_type}.jsonl"
        if not filepath.exists():
            continue
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        sha256 = entry.get("sha256", "")
                        field_name = entry.get("field", "")
                        if not sha256 or not field_name:
                            continue

                        pred_path = paths.get_predictions_path(sha256)
                        if not pred_path.exists():
                            continue
                        with open(pred_path) as pf:
                            pred = json.load(pf)
                        field_data = pred.get("fields", {}).get(field_name, {})
                        confidence = field_data.get("confidence", 0.0)
                        predicted_value = str(field_data.get("value", ""))

                        if label_type == "corrections":
                            ground_truth = str(entry.get("corrected_value", ""))
                        else:
                            ground_truth = predicted_value  # approved = correct

                        samples.append(
                            {
                                "confidence": confidence,
                                "predicted_value": predicted_value,
                                "ground_truth_value": ground_truth,
                            }
                        )
                    except (json.JSONDecodeError, OSError):
                        continue
        except OSError as e:
            logger.warning(
                "calibration_file_read_failed", path=str(filepath), error=str(e)
            )

    return samples


def compute_ece_from_labeled_data(data_dir: Path) -> float | None:
    """Compute ECE from corrections/approvals. Returns None if < 5 samples."""
    samples = _read_labeled_samples(data_dir)
    if len(samples) < 5:
        logger.info("ece_insufficient_data", n_samples=len(samples), min_required=5)
        return None
    ece = compute_ece_from_predictions(samples)
    logger.info("ece_computed", ece=ece, n_samples=len(samples))
    return ece


def compute_empirical_accuracy(data_dir: Path) -> float | None:
    """Ratio of approvals to total reviews. Returns None if no review data."""
    samples = _read_labeled_samples(data_dir)
    if not samples:
        return None
    correct = sum(1 for s in samples if s["predicted_value"] == s["ground_truth_value"])
    return correct / len(samples)
