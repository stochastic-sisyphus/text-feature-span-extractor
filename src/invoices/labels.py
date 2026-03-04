"""Label alignment for training: corrections, approvals, and candidate matching."""

import json
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import ingest, io_utils, paths, schema_registry
from .logging import get_logger

logger = get_logger(__name__)


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """
    Compute Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox1: [x0, y0, x1, y1] normalized coordinates
        bbox2: [x0, y0, x1, y1] normalized coordinates

    Returns:
        IoU score between 0.0 and 1.0
    """
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    intersection = (x_max - x_min) * (y_max - y_min)

    # Compute union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# Field type classification for matching strategies
# ---------------------------------------------------------------------------

DATE_FIELDS = set(schema_registry.fields_by_type("date"))
AMOUNT_FIELDS = set(schema_registry.fields_by_type("amount"))
ID_FIELDS = set(schema_registry.fields_by_type("id"))
NAME_FIELDS = set(schema_registry.name_fields())


# ---------------------------------------------------------------------------
# Matching helpers (adapted from scripts/bridge_ground_truth.py)
# ---------------------------------------------------------------------------

# Translation table for stripping currency symbols and whitespace from amounts
_AMOUNT_STRIP_TABLE = str.maketrans("", "", ",$€£¥ \t\n\r")


def normalize_amount(s: str) -> str:
    """Strip currency symbols, commas, whitespace from amount strings."""
    return s.translate(_AMOUNT_STRIP_TABLE).strip()


def parse_date_variants(iso_date: str) -> list[str]:
    """Generate common date format variants from an ISO date string."""
    try:
        dt = datetime.strptime(iso_date, "%Y-%m-%d")
    except ValueError:
        return [iso_date]

    variants = [
        iso_date,  # 2024-07-31
        dt.strftime("%m/%d/%Y"),  # 07/31/2024
        dt.strftime("%-m/%-d/%Y"),  # 7/31/2024
        dt.strftime("%m/%d/%y"),  # 07/31/24
        dt.strftime("%-m/%-d/%y"),  # 7/31/24
        dt.strftime("%-m/%d/%y"),  # 7/31/24 (hybrid)
        dt.strftime("%-m/%d/%Y"),  # 7/31/2024 (hybrid)
        dt.strftime("%b %d"),  # Jul 31
        dt.strftime("%b %-d"),  # Jul 31 (no leading zero)
        dt.strftime("%B %d"),  # July 31
        dt.strftime("%d %b"),  # 31 Jul
        dt.strftime("%d %b %Y"),  # 31 Jul 2024
        dt.strftime("%d-%b-%y"),  # 31-Jul-24
        dt.strftime("%d-%b-%Y"),  # 31-Jul-2024
        dt.strftime("%d/%m/%Y"),  # 31/07/2024
        dt.strftime("%d-%m-%Y"),  # 31-07-2024
        dt.strftime("%m-%d-%Y"),  # 07-31-2024
        dt.strftime("%b %d, %Y"),  # Jul 31, 2024
        dt.strftime("%B %d, %Y"),  # July 31, 2024
    ]
    return variants


def match_value_to_candidates(
    candidates_df: pd.DataFrame,
    field: str,
    value: str,
    correct_bbox: list[float] | None = None,
) -> tuple[int | None, float]:
    """Match a value to the best candidate using field-specific strategies.

    When correct_bbox is provided and multiple candidates match the same text,
    uses IoU (intersection over union) of bounding boxes to pick the intended
    occurrence.

    Args:
        candidates_df: DataFrame of candidates with raw_text and optional bbox columns
        field: Field name (used to select matching strategy)
        value: The value to match against candidate text
        correct_bbox: Optional [x0, y0, x1, y1] bbox to disambiguate duplicates

    Returns:
        (candidate_idx, confidence) or (None, 0.0) if no match
    """
    value_str = str(value)

    if field in DATE_FIELDS:
        best_idx, best_conf = _match_date(candidates_df, value_str)
    elif field in AMOUNT_FIELDS:
        best_idx, best_conf = _match_amount(candidates_df, value_str)
    elif field in ID_FIELDS:
        best_idx, best_conf = _match_id(candidates_df, value_str)
    elif field in NAME_FIELDS:
        best_idx, best_conf = _match_name(candidates_df, value_str)
    else:
        best_idx, best_conf = _match_exact(candidates_df, value_str)

    # If no bbox hint or no initial match, return as-is
    if correct_bbox is None or best_idx is None:
        return (best_idx, best_conf)

    # Bbox disambiguation: find all candidates with the same text match
    # and pick the one with highest IoU to correct_bbox
    best_text = str(candidates_df.iloc[best_idx].get("raw_text", "")).lower().strip()
    rival_indices: list[int] = []

    for i, row in candidates_df.iterrows():
        raw = str(row.get("raw_text", "")).lower().strip()
        if raw == best_text:
            rival_indices.append(int(i))

    # Only disambiguate if there are actual duplicates
    if len(rival_indices) <= 1:
        return (best_idx, best_conf)

    # Pick the candidate whose bbox has highest IoU with correct_bbox
    best_iou = -1.0
    best_iou_idx = best_idx

    for idx in rival_indices:
        candidate_bbox = candidates_df.iloc[idx].get("bbox")
        if candidate_bbox is None:
            continue
        # Handle both list and tuple bbox formats
        if hasattr(candidate_bbox, "__len__") and len(candidate_bbox) == 4:
            iou = compute_iou(correct_bbox, list(candidate_bbox))
            if iou > best_iou:
                best_iou = iou
                best_iou_idx = idx

    return (best_iou_idx, best_conf)


def _match_date(df: pd.DataFrame, iso_date: str) -> tuple[int | None, float]:
    """Match date values using multiple format variants with fuzzy matching."""
    variants = parse_date_variants(iso_date)
    best_idx = None
    best_conf = 0.0
    best_overlap_len = 0

    for i, row in df.iterrows():
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue

        # Normalize candidate text: strip trailing punctuation, lowercase
        raw_normalized = raw.lower().strip().rstrip(",.;:")

        for vi, variant in enumerate(variants):
            variant_normalized = variant.lower().strip().rstrip(",.;:")

            # Calculate overlap length for scoring
            if variant_normalized == raw_normalized:
                # Exact match - highest priority
                conf = 1.0 if vi == 0 else (0.9 if vi <= 4 else 0.8)
                overlap_len = len(raw_normalized)
            elif variant_normalized in raw_normalized:
                # Variant contained in candidate
                conf = 0.9 if vi == 0 else (0.8 if vi <= 4 else 0.6)
                overlap_len = len(variant_normalized)
            elif raw_normalized in variant_normalized:
                # Candidate contained in variant
                conf = 0.85 if vi == 0 else (0.75 if vi <= 4 else 0.55)
                overlap_len = len(raw_normalized)
            else:
                continue

            # Prefer matches with more overlap (e.g., "Nov 10" > "2023")
            # Update best match if: higher confidence OR same confidence but longer overlap
            if conf > best_conf or (
                conf == best_conf and overlap_len > best_overlap_len
            ):
                best_conf = conf
                best_idx = int(i)
                best_overlap_len = overlap_len
                break

    return (best_idx, best_conf)


def _match_amount(df: pd.DataFrame, value: str) -> tuple[int | None, float]:
    """Match amount values with normalization (strip $, commas)."""
    norm_value = normalize_amount(value)
    abs_value = norm_value.lstrip("-")
    best_idx = None
    best_conf = 0.0

    for i, row in df.iterrows():
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        norm_raw = normalize_amount(raw)

        if norm_raw == norm_value:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = int(i)
        elif norm_raw == abs_value:
            if 0.8 > best_conf:
                best_conf = 0.8
                best_idx = int(i)
        elif len(norm_value) >= 3 and (norm_value in norm_raw or abs_value in norm_raw):
            if 0.6 > best_conf:
                best_conf = 0.6
                best_idx = int(i)

    return (best_idx, best_conf)


# Translation table for stripping spaces, tabs, newlines, and hyphens from IDs
_ID_STRIP_TABLE = str.maketrans("", "", " \t\n\r-")


def _match_id(df: pd.DataFrame, value: str) -> tuple[int | None, float]:
    """Match ID values with normalization (strip hyphens, spaces)."""
    norm_value = value.translate(_ID_STRIP_TABLE).lower()
    best_idx = None
    best_conf = 0.0

    for i, row in df.iterrows():
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        norm_raw = raw.translate(_ID_STRIP_TABLE).lower()

        if norm_raw == norm_value:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = int(i)
        elif norm_value in norm_raw:
            if 0.8 > best_conf:
                best_conf = 0.8
                best_idx = int(i)
        elif norm_raw in norm_value and len(norm_raw) >= 6:
            if 0.6 > best_conf:
                best_conf = 0.6
                best_idx = int(i)

    return (best_idx, best_conf)


def _match_name(df: pd.DataFrame, value: str) -> tuple[int | None, float]:
    """Match name values with case-insensitive fuzzy matching."""
    value_lower = value.lower().strip()
    best_idx = None
    best_conf = 0.0

    for i, row in df.iterrows():
        raw = str(row.get("raw_text", ""))
        if not raw:
            continue
        raw_lower = raw.lower().strip()

        if raw_lower == value_lower:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = int(i)
        elif value_lower in raw_lower:
            if 0.9 > best_conf:
                best_conf = 0.9
                best_idx = int(i)
        elif raw_lower in value_lower and len(raw_lower) >= 3:
            if 0.7 > best_conf:
                best_conf = 0.7
                best_idx = int(i)

    return (best_idx, best_conf)


def _match_exact(df: pd.DataFrame, value: str) -> tuple[int | None, float]:
    """Exact case-insensitive match."""
    value_lower = value.lower().strip()
    best_idx = None
    best_conf = 0.0

    for i, row in df.iterrows():
        raw = str(row.get("raw_text", "")).lower().strip()
        if raw == value_lower:
            if 1.0 > best_conf:
                best_conf = 1.0
                best_idx = int(i)
        elif value_lower in raw:
            if 0.7 > best_conf:
                best_conf = 0.7
                best_idx = int(i)

    return (best_idx, best_conf)


# ---------------------------------------------------------------------------
# Corrections and approvals alignment
# ---------------------------------------------------------------------------


def align_corrections() -> dict[str, Any]:
    """
    Align human corrections with candidates for training.

    Reads data/labels/corrections/corrections.jsonl, matches correct values
    to candidates, writes data/labels/aligned/aligned_corrections.parquet.

    Returns:
        Summary dict with total/aligned counts
    """
    corrections_file = paths.get_corrections_path()
    aligned_dir = paths.get_labels_aligned_dir()
    aligned_dir.mkdir(parents=True, exist_ok=True)

    if not corrections_file.exists():
        return {"total": 0, "aligned": 0, "status": "no_corrections_file"}

    # Read corrections
    corrections_raw = []
    with open(corrections_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corrections_raw.append(json.loads(line))

    if not corrections_raw:
        return {"total": 0, "aligned": 0, "status": "no_corrections"}

    # Deduplicate: keep only the latest entry per (doc_id, field).
    # JSONL is append-only so later entries are more recent.
    corrections_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in corrections_raw:
        key = (entry.get("doc_id", ""), entry.get("field", ""))
        corrections_by_key[key] = entry  # last write wins
    corrections = list(corrections_by_key.values())

    # Load ingest index for doc_id → sha256 lookup
    indexed_docs = ingest.get_indexed_documents()

    aligned_rows = []
    total = 0
    aligned_count = 0

    for correction in tqdm(corrections, desc="Aligning corrections"):
        total += 1
        doc_id = correction.get("doc_id")
        field = correction.get("field")
        correct_value = correction.get("correct_value")
        correct_bbox = correction.get("correct_bbox")
        action = correction.get("action", "correct")

        if not doc_id or not field:
            continue

        # Look up SHA256 from doc_id (format: "fs:{sha256_prefix}")
        doc_row = indexed_docs[indexed_docs["doc_id"] == doc_id]
        if doc_row.empty:
            continue

        sha256 = doc_row.iloc[0]["sha256"]

        # Load candidates
        candidates_path = paths.get_candidates_path(sha256)

        try:
            candidates_df = io_utils.read_parquet_safe(
                candidates_path, on_error="empty"
            )
            if candidates_df is None:
                continue
        except (OSError, ValueError):
            logger.warning(f"Failed to load candidates for {sha256}", exc_info=True)
            continue

        # Handle different action types
        best_idx = None
        confidence = 0.0
        iou = 0.0
        labeled_text = correct_value

        if action in ("not_applicable", "not_in_document"):
            # Field doesn't exist on this document - use NONE (no candidate)
            best_idx = None
            labeled_text = "NONE"
            confidence = 1.0
        elif action == "reject":
            # Prediction was wrong - this is a negative example
            # We need to find which candidate was predicted and mark it as wrong
            # For now, skip matching and just record as negative
            best_idx = None
            labeled_text = f"REJECTED:{correct_value}"
            confidence = 1.0
        else:
            # action == "correct" (default)
            if not correct_value:
                continue

            # Match value to candidates (pass bbox for disambiguation)
            best_idx, confidence = match_value_to_candidates(
                candidates_df, field, correct_value, correct_bbox=correct_bbox
            )

            # If bbox provided, also try IoU matching
            if correct_bbox and best_idx is not None:
                # Compute IoU with the matched candidate's bbox
                candidate_bbox = candidates_df.iloc[best_idx].get("bbox")
                if candidate_bbox is not None:
                    iou = compute_iou(correct_bbox, candidate_bbox)

        is_aligned = best_idx is not None or action in (
            "not_applicable",
            "not_in_document",
            "reject",
        )
        if is_aligned:
            aligned_count += 1

        aligned_rows.append(
            {
                "sha256": sha256,
                "doc_id": doc_id,
                "field": field,
                "labeled_text": labeled_text,
                "char_start": 0,  # Not available from corrections
                "char_end": len(str(labeled_text)),
                "candidate_idx": best_idx,
                "alignment_iou": max(confidence, iou),
                "is_aligned": is_aligned,
                "action": action,
            }
        )

    # Write aligned corrections
    if aligned_rows:
        aligned_df = pd.DataFrame(aligned_rows)
        output_path = aligned_dir / "aligned_corrections.parquet"
        io_utils.write_parquet_safe(aligned_df, output_path)

    return {"total": total, "aligned": aligned_count, "status": "success"}


def align_approvals() -> dict[str, Any]:
    """
    Align human approvals with candidates for training.

    Reads data/labels/approvals/approvals.jsonl, finds the approved
    candidate from predictions, writes data/labels/aligned/aligned_approvals.parquet.

    Returns:
        Summary dict with total/aligned counts
    """
    approvals_file = paths.get_approvals_path()
    aligned_dir = paths.get_labels_aligned_dir()
    aligned_dir.mkdir(parents=True, exist_ok=True)

    if not approvals_file.exists():
        return {"total": 0, "aligned": 0, "status": "no_approvals_file"}

    # Read approvals
    approvals_raw = []
    with open(approvals_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                approvals_raw.append(json.loads(line))

    if not approvals_raw:
        return {"total": 0, "aligned": 0, "status": "no_approvals"}

    # Deduplicate: keep only the latest entry per (doc_id, field).
    # JSONL is append-only so later entries are more recent.
    approvals_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in approvals_raw:
        key = (entry.get("doc_id", ""), entry.get("field", ""))
        approvals_by_key[key] = entry  # last write wins
    approvals = list(approvals_by_key.values())

    # Load ingest index for doc_id → sha256 lookup
    indexed_docs = ingest.get_indexed_documents()

    aligned_rows = []
    total = 0
    aligned_count = 0

    for approval in tqdm(approvals, desc="Aligning approvals"):
        total += 1
        doc_id = approval.get("doc_id")
        field = approval.get("field")

        if not doc_id or not field:
            continue

        # Look up SHA256 from doc_id
        doc_row = indexed_docs[indexed_docs["doc_id"] == doc_id]
        if doc_row.empty:
            continue

        sha256 = doc_row.iloc[0]["sha256"]

        # Load prediction to find the candidate_idx
        prediction_path = paths.get_predictions_path(sha256)
        if not prediction_path.exists():
            continue

        try:
            with open(prediction_path, encoding="utf-8") as f:
                prediction = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Failed to load prediction for {sha256}", exc_info=True)
            continue

        # Find the field value from the prediction
        field_value = prediction.get("fields", {}).get(field, {}).get("value")

        if field_value is None:
            continue

        # Match approved value to candidates using text+bbox matching
        # (same approach as corrections — robust to candidate index changes)
        candidates_path = paths.get_candidates_path(sha256)

        try:
            candidates_df = io_utils.read_parquet_safe(
                candidates_path, on_error="empty"
            )
            if candidates_df is None:
                continue
        except (OSError, ValueError):
            logger.warning(
                f"Failed to load candidates for approval {sha256}", exc_info=True
            )
            continue

        candidate_idx, confidence = match_value_to_candidates(
            candidates_df, field, str(field_value)
        )

        is_aligned = candidate_idx is not None
        if is_aligned:
            aligned_count += 1

        aligned_rows.append(
            {
                "sha256": sha256,
                "doc_id": doc_id,
                "field": field,
                "labeled_text": str(field_value),
                "char_start": 0,
                "char_end": len(str(field_value)),
                "candidate_idx": candidate_idx,
                "alignment_iou": confidence if is_aligned else 0.0,
                "is_aligned": is_aligned,
            }
        )

    # Write aligned approvals
    if aligned_rows:
        aligned_df = pd.DataFrame(aligned_rows)
        output_path = aligned_dir / "aligned_approvals.parquet"
        io_utils.write_parquet_safe(aligned_df, output_path)

    return {"total": total, "aligned": aligned_count, "status": "success"}


def align_corrections_cli() -> dict[str, Any]:
    """
    CLI entry point for aligning corrections and approvals.

    Returns:
        Combined summary dict
    """
    corrections_result = align_corrections()
    approvals_result = align_approvals()

    return {
        "corrections": corrections_result,
        "approvals": approvals_result,
    }


def load_aligned_labels() -> pd.DataFrame:
    """
    Load all aligned labels from the aligned directory.

    Deduplicates across corrections and approvals: when the same
    (doc_id, field) exists in both, corrections take priority (the user
    corrected after initially approving). Within each source, the
    earlier dedup in align_corrections/align_approvals already keeps
    only the latest entry.

    Returns:
        Combined DataFrame with all aligned labels
    """
    aligned_dir = paths.get_labels_aligned_dir()

    if not aligned_dir.exists():
        logger.info("no_aligned_labels_directory")
        return pd.DataFrame()

    # Filter function to only include successfully aligned labels
    def filter_aligned(df: pd.DataFrame) -> pd.DataFrame:
        return df[df["is_aligned"]].copy()

    combined_df = io_utils.load_and_concat_parquets(
        aligned_dir, filter_fn=filter_aligned, on_error="warn"
    )

    if combined_df.empty:
        logger.info("no_aligned_label_files")
    else:
        # Deduplicate across corrections and approvals.
        # Corrections (which have an "action" column) take priority over
        # approvals for the same (doc_id, field) pair.
        pre_dedup = len(combined_df)
        has_action = (
            combined_df["action"].notna()
            if "action" in combined_df.columns
            else pd.Series(False, index=combined_df.index)
        )
        # Sort: rows WITH action (corrections) first so they survive dedup
        combined_df["_is_correction"] = has_action.astype(int)
        combined_df = combined_df.sort_values("_is_correction", ascending=False)
        combined_df = combined_df.drop_duplicates(
            subset=["doc_id", "field"], keep="first"
        )
        combined_df = combined_df.drop(columns=["_is_correction"])
        combined_df = combined_df.reset_index(drop=True)

        num_files = len(io_utils.list_parquet_files(aligned_dir))
        logger.info(
            "aligned_labels_loaded",
            count=len(combined_df),
            files=num_files,
            deduped=pre_dedup - len(combined_df),
        )

    return combined_df
