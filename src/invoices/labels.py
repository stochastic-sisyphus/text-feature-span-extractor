"""Doccano integration for label pulling, import, and alignment."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from . import ingest, paths, tokenize, utils


def pull_labels() -> dict[str, Any]:
    """
    Pull labels from Doccano HTTP API.
    Safe no-op when credentials missing.

    Returns:
        Summary of pulled labels or error info
    """
    # Check for credentials
    api_url = os.environ.get("DOCCANO_URL", "http://localhost:8000")
    api_user = os.environ.get("DOCCANO_USERNAME", "admin")
    api_pass = os.environ.get("DOCCANO_PASSWORD")

    if not api_pass:
        print("Doccano credentials missing (DOCCANO_PASSWORD required)")
        print("Using DOCCANO_URL =", api_url)
        print("Using DOCCANO_USERNAME =", api_user)
        print("Skipping label pull (safe no-op)")
        return {"status": "skipped", "reason": "missing_credentials"}

    try:
        # Authenticate with Doccano
        auth_url = f"{api_url.rstrip('/')}/auth/login/"
        auth_data = json.dumps({
            "username": api_user,
            "password": api_pass
        }).encode('utf-8')

        auth_request = urllib.request.Request(auth_url, data=auth_data)
        auth_request.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(auth_request) as auth_response:
            auth_result = json.loads(auth_response.read().decode())

        # Get projects and their annotations
        projects_url = f"{api_url.rstrip('/')}/api/projects/"
        projects_request = urllib.request.Request(projects_url)
        projects_request.add_header("Authorization", f"Bearer {auth_result.get('access')}")

        with urllib.request.urlopen(projects_request) as response:
            projects_data = json.loads(response.read().decode())

        # Collect annotations from all projects
        tasks_data = []
        for project in projects_data:
            project_id = project['id']
            annotations_url = f"{api_url.rstrip('/')}/api/projects/{project_id}/annotations/"

            annotations_request = urllib.request.Request(annotations_url)
            annotations_request.add_header("Authorization", f"Bearer {auth_result.get('access')}")

            try:
                with urllib.request.urlopen(annotations_request) as ann_response:
                    annotations = json.loads(ann_response.read().decode())
                    tasks_data.extend(annotations)
            except Exception as e:
                print(f"Warning: Could not fetch annotations for project {project_id}: {e}")
                continue

        # Save raw labels
        labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
        labels_raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
        raw_file = labels_raw_dir / f"labels_{timestamp}.json"

        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(tasks_data, f, indent=2)

        task_count = len(tasks_data) if isinstance(tasks_data, list) else 0

        print(f"Pulled {task_count} annotations from Doccano")
        print(f"Saved raw labels to: {raw_file}")

        return {
            "status": "success",
            "task_count": task_count,
            "raw_file": str(raw_file),
        }

    except urllib.error.HTTPError as e:
        error_msg = f"HTTP error {e.code}: {e.reason}"
        print(f"Failed to pull labels: {error_msg}")
        return {"status": "error", "error": error_msg}

    except Exception as e:
        error_msg = str(e)
        print(f"Failed to pull labels: {error_msg}")
        return {"status": "error", "error": error_msg}


def import_labels(path: str) -> dict[str, Any]:
    """
    Import labels from local Doccano export file.

    Args:
        path: Path to Doccano export JSON file

    Returns:
        Summary of imported labels
    """
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    # Load labels file
    with open(path_obj, encoding="utf-8") as f:
        labels_data = json.load(f)

    # Save to raw directory
    labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
    labels_raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = utils.get_current_utc_iso().replace(":", "-").replace(".", "-")
    import_filename = f"import_{path_obj.stem}_{timestamp}.json"
    raw_file = labels_raw_dir / import_filename

    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(labels_data, f, indent=2)

    task_count = len(labels_data) if isinstance(labels_data, list) else 0

    print(f"Imported {task_count} tasks from {path}")
    print(f"Saved to: {raw_file}")

    return {
        "status": "success",
        "task_count": task_count,
        "raw_file": str(raw_file),
        "source_file": str(path),
    }


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


def align_labels(iou_threshold: float = 0.3, all_files: bool = False) -> dict[str, Any]:
    """
    Align labels with document candidates using IoU-based matching.

    Args:
        iou_threshold: Minimum IoU for candidate-label alignment
        all_files: Process all raw label files if True

    Returns:
        Summary of alignment results
    """
    labels_raw_dir = paths.get_repo_root() / "data" / "labels" / "raw"
    aligned_dir = paths.get_repo_root() / "data" / "labels" / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)

    if not labels_raw_dir.exists():
        print("No raw labels directory found")
        return {"status": "no_labels", "aligned_count": 0}

    label_files = list(labels_raw_dir.glob("*.json"))

    if not label_files:
        print("No label files found in raw directory")
        return {"status": "no_files", "aligned_count": 0}

    if not all_files:
        # Use most recent file
        label_files = [max(label_files, key=lambda p: p.stat().st_mtime)]

    total_aligned = 0
    alignment_results = []

    for label_file in label_files:
        print(f"Processing labels from: {label_file.name}")

        # Load labels
        with open(label_file, encoding="utf-8") as f:
            labels_data = json.load(f)

        if not isinstance(labels_data, list):
            print(f"Skipping {label_file.name}: not a list of tasks")
            continue

        # Process each labeled task
        aligned_rows = []

        for task in tqdm(labels_data, desc="Aligning labels"):
            # Extract document info
            task_data = task.get("data", {})
            doc_id = task_data.get("doc_id")

            if not doc_id:
                continue

            # Find document SHA256
            indexed_docs = ingest.get_indexed_documents()
            doc_row = indexed_docs[indexed_docs["doc_id"] == doc_id]

            if doc_row.empty:
                continue

            sha256 = doc_row.iloc[0]["sha256"]

            # Load document tokens and candidates
            try:
                tokenize.get_document_tokens(sha256)
                candidates_path = paths.get_candidates_path(sha256)

                if not candidates_path.exists():
                    continue

                candidates_df = pd.read_parquet(candidates_path)
            except Exception:
                continue

            # Process annotations
            annotations = task.get("annotations", [])

            for annotation in annotations:
                results = annotation.get("result", [])

                for result in results:
                    # Extract label info
                    value = result.get("value", {})
                    field_name = value.get("labels", [None])[0]

                    if not field_name:
                        continue

                    # Extract span coordinates (character-based)
                    char_start = value.get("start")
                    char_end = value.get("end")
                    labeled_text = value.get("text", "")

                    if char_start is None or char_end is None:
                        continue

                    # Find best matching candidate using IoU on character ranges
                    best_candidate_idx = None
                    best_iou = 0.0

                    for cand_idx, candidate_row in candidates_df.iterrows():
                        # Get candidate character range from candidates
                        cand_start = candidate_row.get("char_start")
                        cand_end = candidate_row.get("char_end")

                        if cand_start is None or cand_end is None:
                            continue

                        # Compute character-level IoU
                        iou = char_iou(char_start, char_end, cand_start, cand_end)

                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_candidate_idx = cand_idx

                    # Create alignment row
                    alignment_row = {
                        "sha256": sha256,
                        "doc_id": doc_id,
                        "field": field_name,
                        "labeled_text": labeled_text,
                        "char_start": char_start,
                        "char_end": char_end,
                        "candidate_idx": best_candidate_idx,
                        "alignment_iou": best_iou if best_candidate_idx is not None else 0.0,
                        "is_aligned": best_candidate_idx is not None
                    }

                    aligned_rows.append(alignment_row)

        # Save aligned results for this file
        if aligned_rows:
            aligned_df = pd.DataFrame(aligned_rows)
            aligned_file = aligned_dir / f"aligned_{label_file.stem}.parquet"
            aligned_df.to_parquet(aligned_file, index=False)

            file_aligned_count = len(aligned_df)
            total_aligned += file_aligned_count

            alignment_results.append({
                "source_file": label_file.name,
                "aligned_file": aligned_file.name,
                "aligned_count": file_aligned_count
            })

            print(f"Saved {file_aligned_count} alignments to: {aligned_file.name}")

    return {
        "status": "success",
        "total_aligned": total_aligned,
        "iou_threshold": iou_threshold,
        "files_processed": len(alignment_results),
        "alignment_results": alignment_results
    }


def char_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """
    Compute Intersection over Union for character ranges.

    Args:
        a_start, a_end: First character range
        b_start, b_end: Second character range

    Returns:
        IoU score between 0.0 and 1.0
    """
    if a_start >= a_end or b_start >= b_end:
        return 0.0

    # Calculate intersection
    intersection_start = max(a_start, b_start)
    intersection_end = min(a_end, b_end)

    if intersection_start >= intersection_end:
        return 0.0

    intersection_length = intersection_end - intersection_start

    # Calculate union
    union_length = (a_end - a_start) + (b_end - b_start) - intersection_length

    if union_length == 0:
        return 0.0

    return intersection_length / union_length


def load_aligned_labels() -> pd.DataFrame:
    """
    Load all aligned labels from the aligned directory.

    Returns:
        Combined DataFrame with all aligned labels
    """
    aligned_dir = paths.get_repo_root() / "data" / "labels" / "aligned"

    if not aligned_dir.exists():
        print("No aligned labels directory found")
        return pd.DataFrame()

    parquet_files = list(aligned_dir.glob("*.parquet"))

    if not parquet_files:
        print("No aligned label files found")
        return pd.DataFrame()

    # Load and combine all aligned files
    dfs = []
    for parquet_file in parquet_files:
        try:
            df = pd.read_parquet(parquet_file)
            # Only include successfully aligned labels
            df = df[df["is_aligned"]].copy()
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {parquet_file.name}: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} aligned labels from {len(dfs)} files")

    return combined_df
