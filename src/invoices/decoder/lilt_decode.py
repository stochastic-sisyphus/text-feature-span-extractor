"""LiLT drop-in alternative to the Hungarian decoder (Stage 3).

Contract: ``decode_document_with_lilt`` returns ``dict[str, Assignment]`` —
field name → Assignment — identical shape to ``decode_document_with_data``.

Design notes
------------
* CPU inference only (``device="cpu"``).  The function is pure: no I/O, no
  DB, no Config reads.  All inputs are explicit arguments.
* id2label is schema-driven and passed in by the caller — field names are
  never hardcoded here.
* Sliding-window tokenization (content=510, stride=128) + centrality-based
  per-word merge ported from ``~/.venvs/lilt_loo_modal.py`` (verified
  reference).  Per-word confidence (softmax probability) is carried through
  the merge alongside the label id so the caller receives accurate
  ``ml_probability`` values.
* Candidate matching uses bbox IoU in 0-1 normalized space (same space as
  ``candidate["bbox_norm"]``).  LiLT boxes are in 0-1000 LayoutLM convention
  (fed to the model only); IoU is always done in 0-1 to match candidates.
  IoU search is also gated by page_idx to avoid cross-page collisions.
* Gremlin MLM domain-adaptation was a NEGATIVE result: no detectable gain
  over the base ``SCUT-DLVCLab/lilt-roberta-en-base`` checkpoint on held-out
  LOOCV folds.  The base LiLT checkpoint is therefore used directly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F

from invoices.types import Assignment

# ---------------------------------------------------------------------------
# Sliding-window constants (mirrors lilt_loo_modal.py verbatim)
# ---------------------------------------------------------------------------
_MAX_LEN: int = 512
_SW_CONTENT: int = 510  # content tokens per window (2 slots → CLS + SEP)
_SW_STRIDE: int = 128  # hop between window starts; overlap = 382 tokens
_IOU_MATCH_THRESHOLD: float = 0.3  # mirrors build_lilt_data.py IOU_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp1000(v: float) -> int:
    return max(0, min(1000, round(v)))


def _box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """IoU between two [x0,y0,x1,y1] float boxes in the SAME coordinate space.

    Ported from build_lilt_data.py:iou() verbatim.
    """
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter_w = max(0.0, ix1 - ix0)
    inter_h = max(0.0, iy1 - iy0)
    inter = inter_w * inter_h
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def _encode_sliding(
    words: list[str],
    boxes_1000: list[list[int]],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Sliding-window encode — ported from lilt_loo_modal.py:encode_doc_sliding.

    Returns a list of window dicts, each containing:
        input_ids, attention_mask, bbox, word_ids,
        _content_start, _content_end  (metadata for the merge step).
    Single-window documents return a length-1 list.
    """
    # Full encode without truncation or word_labels (labels would mis-align
    # in no-truncation mode — mirror lilt_loo_modal comment).
    full_enc = tokenizer(
        words,
        boxes=boxes_1000,
        truncation=False,
        max_length=None,
        return_tensors=None,
    )
    full_ids: list[int] = full_enc["input_ids"]
    full_bbox: list[list[int]] = full_enc["bbox"]
    full_wids: list[int | None] = full_enc.word_ids()
    total = len(full_ids)

    if total <= _MAX_LEN:
        # Short doc: re-encode with proper truncation flag (single window).
        enc = tokenizer(
            words,
            boxes=boxes_1000,
            truncation=True,
            max_length=_MAX_LEN,
            return_tensors=None,
        )
        return [
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "bbox": enc["bbox"],
                "word_ids": enc.word_ids(),
                "_content_start": 0,
                "_content_end": max(0, total - 2),  # exclude CLS + SEP
            }
        ]

    # Content slice strips CLS (index 0) and SEP (last index).
    cls_id = full_ids[0]
    sep_id = full_ids[-1]
    zero_box: list[int] = [0, 0, 0, 0]
    content_ids = full_ids[1:-1]
    content_bbox = full_bbox[1:-1]
    content_wids = full_wids[1:-1]
    n_content = len(content_ids)

    pad_id = tokenizer.pad_token_id or 1

    windows: list[dict[str, Any]] = []
    start = 0
    while start < n_content:
        end = min(start + _SW_CONTENT, n_content)
        win_ids = [cls_id, *content_ids[start:end], sep_id]
        win_bbox = [zero_box, *content_bbox[start:end], zero_box]
        win_wids: list[int | None] = [None, *content_wids[start:end], None]

        pad_len = _MAX_LEN - len(win_ids)
        attn = [1] * len(win_ids) + [0] * pad_len
        win_ids += [pad_id] * pad_len
        win_bbox += [zero_box] * pad_len
        win_wids += [None] * pad_len

        windows.append(
            {
                "input_ids": win_ids,
                "attention_mask": attn,
                "bbox": win_bbox,
                "word_ids": win_wids,
                "_content_start": start,
                "_content_end": end,
            }
        )
        if end == n_content:
            break
        start += _SW_STRIDE

    return windows


# ---------------------------------------------------------------------------
# Inference + merge
# ---------------------------------------------------------------------------


def _run_inference(
    windows: list[dict[str, Any]],
    model: Any,
    device: str,
) -> list[tuple[list[tuple[int, float]], dict[str, Any]]]:
    """Run LiLT forward on each window; return (per-token (label_id, prob), window) pairs.

    ``torch.no_grad()`` + ``torch.softmax`` on CPU.  Softmax prob is carried
    alongside the label id so the merge step has confidence available.
    """
    results: list[tuple[list[tuple[int, float]], dict[str, Any]]] = []
    for item in windows:
        input_ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [item["attention_mask"]], dtype=torch.long, device=device
        )
        bbox = torch.tensor([item["bbox"]], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
            ).logits  # shape: (1, seq_len, num_labels)
        probs = F.softmax(logits[0], dim=-1)  # (seq_len, num_labels)
        pred_ids = probs.argmax(dim=-1).tolist()  # list[int]
        pred_probs = probs.max(dim=-1).values.tolist()  # list[float]
        token_preds = list(
            zip(pred_ids, pred_probs, strict=True)
        )  # list[(label_id, prob)]
        results.append((token_preds, item))
    return results


def _merge_word_preds(
    windows_with_preds: list[tuple[list[tuple[int, float]], dict[str, Any]]],
) -> dict[int, tuple[int, float]]:
    """Centrality-based merge of per-word predictions across overlapping windows.

    Ported from lilt_loo_modal.py:merge_window_preds, extended to carry
    per-word softmax confidence alongside the label id.

    For each word: collect (centrality, label_id, prob) from every window
    that covered it; pick the entry from the most-central window position.
    Ties broken by first occurrence (earlier window).

    Returns: word_idx → (label_id, confidence)
    """
    # word_votes: word_idx → list[(centrality, label_id, prob)]
    word_votes: dict[int, list[tuple[float, int, float]]] = defaultdict(list)

    for token_preds, item in windows_with_preds:
        word_ids = item["word_ids"]
        c_start = item.get("_content_start", 0)
        c_end = item.get("_content_end", len(word_ids))
        span_len = max(c_end - c_start, 1)
        seen_wid: set[int] = set()
        for pos, wid in enumerate(word_ids):
            if wid is None or wid in seen_wid:
                continue
            seen_wid.add(wid)
            content_pos = pos - 1  # offset for leading CLS
            centrality = 1.0 - abs(content_pos - (span_len - 1) / 2.0) / (
                (span_len - 1) / 2.0 + 1e-9
            )
            label_id, prob = token_preds[pos]
            word_votes[wid].append((centrality, label_id, prob))

    word_pred: dict[int, tuple[int, float]] = {}
    for wid, votes in word_votes.items():
        # Highest-centrality vote wins; tie → first window (first appended).
        best = max(votes, key=lambda v: v[0])
        _, label_id, prob = best
        word_pred[wid] = (label_id, prob)

    return word_pred


# ---------------------------------------------------------------------------
# Span grouping
# ---------------------------------------------------------------------------


def _group_spans(
    word_pred: dict[int, tuple[int, float]],
    words: list[str],
    norm_boxes: list[tuple[float, float, float, float]],  # 0-1, per-word
    page_idxs: list[int],
    id2label: dict[int, str],
    o_label: str = "O",
) -> dict[str, dict[str, Any]]:
    """Group consecutive same-field words into predicted spans.

    Returns: field_name → span dict with keys:
        text       (str)
        norm_box   (tuple[float,float,float,float]) — union bbox, 0-1
        page_idx   (int)                            — page of first word
        confidence (float)                          — mean per-word prob

    When a field is predicted in multiple disjoint runs, the run with the
    highest mean confidence wins.
    """
    # Collect runs of same-field word indices
    # field_name → list of runs; each run is list[(word_idx, prob)]
    field_runs: dict[str, list[list[tuple[int, float]]]] = defaultdict(list)

    sorted_wids = sorted(word_pred.keys())
    current_field: str | None = None
    current_run: list[tuple[int, float]] = []

    for wid in sorted_wids:
        label_id, prob = word_pred[wid]
        label = id2label.get(label_id, o_label)
        if label == o_label:
            if current_field is not None:
                field_runs[current_field].append(current_run)
                current_field = None
                current_run = []
        elif label == current_field:
            current_run.append((wid, prob))
        else:
            # New field (possibly switching mid-sequence)
            if current_field is not None:
                field_runs[current_field].append(current_run)
            current_field = label
            current_run = [(wid, prob)]

    if current_field is not None and current_run:
        field_runs[current_field].append(current_run)

    # For each field, pick the run with the highest mean confidence
    best_spans: dict[str, dict[str, Any]] = {}
    for field, runs in field_runs.items():
        best_run: list[tuple[int, float]] | None = None
        best_conf = -1.0
        for run in runs:
            if not run:
                continue
            mean_conf = sum(p for _, p in run) / len(run)
            if mean_conf > best_conf:
                best_conf = mean_conf
                best_run = run

        if best_run is None:
            continue

        wids_in_run = [w for w, _ in best_run]
        text = " ".join(words[w] for w in wids_in_run)
        x0 = min(norm_boxes[w][0] for w in wids_in_run)
        y0 = min(norm_boxes[w][1] for w in wids_in_run)
        x1 = max(norm_boxes[w][2] for w in wids_in_run)
        y1 = max(norm_boxes[w][3] for w in wids_in_run)
        pg = page_idxs[wids_in_run[0]]

        best_spans[field] = {
            "text": text,
            "norm_box": (x0, y0, x1, y1),
            "page_idx": pg,
            "confidence": best_conf,
        }

    return best_spans


# ---------------------------------------------------------------------------
# Candidate matching
# ---------------------------------------------------------------------------


def _match_candidate(
    span_box: tuple[float, float, float, float],
    span_page: int,
    candidates_list: list[dict[str, Any]],
    iou_threshold: float = _IOU_MATCH_THRESHOLD,
) -> tuple[int, dict[str, Any]] | None:
    """Find the best-IoU candidate for a predicted span.

    Both span_box and candidate bbox_norm are in 0-1 normalized space.
    Search is page-gated: only candidates on the same page are considered
    to avoid cross-page collisions on multi-page documents.

    Returns (candidate_index, candidate_dict) or None if no match exceeds
    the IoU threshold.
    """
    best_iou = -1.0
    best_idx = -1

    for idx, cand in enumerate(candidates_list):
        if cand.get("page_idx", 0) != span_page:
            continue
        # Candidates carry the bbox as four separate float columns:
        # bbox_norm_x0/y0/x1/y1 (views.py:778-781, dict key names confirmed).
        # The tuple form (_bbox_norm) is deleted before the candidates seam
        # (views.py:852) so we must reconstruct it here.
        cx0 = cand.get("bbox_norm_x0")
        cy0 = cand.get("bbox_norm_y0")
        cx1 = cand.get("bbox_norm_x1")
        cy1 = cand.get("bbox_norm_y1")
        if cx0 is None or cy0 is None or cx1 is None or cy1 is None:
            continue
        cand_box: tuple[float, float, float, float] = (
            float(cx0),
            float(cy0),
            float(cx1),
            float(cy1),
        )
        score = _box_iou(span_box, cand_box)
        if score > best_iou:
            best_iou = score
            best_idx = idx

    if best_iou >= iou_threshold and best_idx >= 0:
        return best_idx, candidates_list[best_idx]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decode_document_with_lilt(
    doc: Any,  # frozen Doc (src/invoices/doc.py)
    candidates_list: list[dict[str, Any]],
    schema_fields: list[str],
    model: Any,  # LiltForTokenClassification, .eval(), CPU
    tokenizer: Any,  # LayoutLMv3TokenizerFast / roberta BPE
    id2label: dict[int, str],  # classifier head id → field name; 0 → "O"
    none_bias: float,
    *,
    min_confidence: float = 0.0,
    device: str = "cpu",
) -> dict[str, Assignment]:
    """Decode a Doc using LiLT token classification.

    Drop-in alternative to ``decode_document_with_data`` (Hungarian decoder).
    Returns the identical ``dict[field_name → Assignment]`` contract: one
    entry per schema field, ``assignment_type="NONE"`` when abstaining.

    Algorithm
    ---------
    1. Extract words + 0-1 norm boxes + page indices from doc.pages tokens.
       Build 0-1000 boxes for LayoutLM input (no division by page dims needed
       because ``bbox_norm_*`` keys are already 0-1).
    2. Sliding-window tokenize (content=510, stride=128), run LiLT with
       ``torch.no_grad()``, softmax → per-token (label_id, prob).
    3. Merge per-token preds to per-word (centrality policy; carries prob).
    4. Group consecutive same-field words into predicted spans; pick the
       highest-mean-confidence span per field.
    5. For each schema field with a predicted span above ``min_confidence``:
       match span bbox (0-1) to candidates via IoU (page-gated).
       → matched: emit CANDIDATE Assignment with candidate_index + candidate.
       → unmatched (no IoU ≥ threshold): emit CANDIDATE Assignment with
         candidate_index=None, candidate=None, raw_text=<span text>.
         # NOTE: raw_text populated here deviates from the convention that
         # normalize_assignments populates it; the unmatched-span branch
         # produces a synthetic Assignment carrying the extracted span value
         # for downstream callers that can handle it.
    6. Every schema field with no span → NONE Assignment.

    Pure function: no I/O, no DB, no Config.
    """
    # ------------------------------------------------------------------
    # 1. Pull words, 0-1 boxes, page indices from Doc
    # ------------------------------------------------------------------
    words: list[str] = []
    norm_boxes: list[tuple[float, float, float, float]] = []
    boxes_1000: list[list[int]] = []
    page_idxs: list[int] = []

    for page in doc.pages:
        for tok in page.tokens:
            text = str(tok["text"]) if tok["text"] else ""
            if not text.strip():
                continue
            nx0 = float(tok.get("bbox_norm_x0", 0.0))
            ny0 = float(tok.get("bbox_norm_y0", 0.0))
            nx1 = float(tok.get("bbox_norm_x1", 0.0))
            ny1 = float(tok.get("bbox_norm_y1", 0.0))
            words.append(text)
            norm_boxes.append((nx0, ny0, nx1, ny1))
            boxes_1000.append(
                [
                    _clamp1000(nx0 * 1000),
                    _clamp1000(ny0 * 1000),
                    _clamp1000(nx1 * 1000),
                    _clamp1000(ny1 * 1000),
                ]
            )
            page_idxs.append(int(tok.get("page_idx", 0)))

    # Empty doc → all NONE
    if not words:
        return {
            field: Assignment(
                assignment_type="NONE",
                candidate_index=None,
                cost=none_bias,
                field=field,
                used_ml_model=True,
                ml_probability=None,
            )
            for field in schema_fields
        }

    # ------------------------------------------------------------------
    # 2. Sliding-window encode + LiLT inference
    # ------------------------------------------------------------------
    windows = _encode_sliding(words, boxes_1000, tokenizer)
    model.eval()
    windows_with_preds = _run_inference(windows, model, device)

    # ------------------------------------------------------------------
    # 3. Merge per-token → per-word (centrality, carries prob)
    # ------------------------------------------------------------------
    o_label = id2label.get(0, "O")
    word_pred = _merge_word_preds(windows_with_preds)

    # ------------------------------------------------------------------
    # 4. Group into per-field predicted spans
    # ------------------------------------------------------------------
    best_spans = _group_spans(
        word_pred=word_pred,
        words=words,
        norm_boxes=norm_boxes,
        page_idxs=page_idxs,
        id2label=id2label,
        o_label=o_label,
    )

    # ------------------------------------------------------------------
    # 5 & 6. Build assignments
    # ------------------------------------------------------------------
    assignments: dict[str, Assignment] = {}

    for field in schema_fields:
        span = best_spans.get(field)

        if span is None or span["confidence"] < min_confidence:
            # No span predicted (or below confidence gate) → NONE
            assignments[field] = Assignment(
                assignment_type="NONE",
                candidate_index=None,
                cost=none_bias,
                field=field,
                used_ml_model=True,
                ml_probability=None,
            )
            continue

        confidence: float = span["confidence"]
        span_box: tuple[float, float, float, float] = span["norm_box"]
        span_page: int = span["page_idx"]

        match = _match_candidate(span_box, span_page, candidates_list)

        if match is not None:
            cand_idx, cand = match
            # The LiLT span text is the authoritative extracted value; the
            # IoU-matched candidate is provenance only (candidate_index → bbox
            # for the UI). Carrying raw_text=span["text"] prevents the
            # over-greedy / truncated candidate text from replacing the value.
            assignments[field] = Assignment(
                assignment_type="CANDIDATE",
                candidate_index=cand_idx,
                cost=1.0 - confidence,
                field=field,
                used_ml_model=True,
                ml_probability=confidence,
                candidate=cand,
                raw_text=span["text"],
            )
        else:
            # Unmatched-span branch: LiLT predicted a span but no candidate
            # bbox overlaps (IoU < threshold).  Emit a CANDIDATE Assignment
            # with candidate_index=None and raw_text carrying the span value.
            # NOTE: raw_text is normally populated by normalize_assignments,
            # not by the decoder.  This is the deliberate exception: the
            # synthetic-span policy surfaces the extracted value for any
            # downstream caller that can use it, gated on min_confidence.
            assignments[field] = Assignment(
                assignment_type="CANDIDATE",
                candidate_index=None,
                cost=1.0 - confidence,
                field=field,
                used_ml_model=True,
                ml_probability=confidence,
                candidate=None,
                raw_text=span["text"],
            )

    return assignments
