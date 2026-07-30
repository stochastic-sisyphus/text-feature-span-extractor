"""Read-only ranker evaluation harness.

Provides two public surfaces:

- ``build_training_rows(label_rows, pred_index)`` — pure extraction of the
  action→label logic from handle_retrain (verbatim copy, zero behaviour change).
  DRY anchor: handle_retrain delegates here; the evaluator reuses it too.

- ``evaluate_ranker(pool)`` — async, read-only.  Loads supervision data from
  Postgres using the same queries handle_retrain uses, computes in-sample and
  LOOCV (Leave-One-Out Cross Validation) ranker metrics, and returns a flat
  dict.  No filesystem writes.  No MLflow writes.  No model_runs writes.

LOOCV semantics
---------------
For N distinct doc_ids in the training rows:
  - Hold out doc D: train a fresh InvoiceFieldRanker on all rows from the
    remaining N-1 docs, then score the held-out doc's rows with
    _compute_validation_metrics.
  - A fold is *skipped* when the training remainder has zero positive labels
    (can't train) OR the held-out doc has no positive-bearing groups (no
    correct candidate → no signal to measure) OR training itself errors.
  - ``loocv_docs_scored`` = folds that produced a metric value.
  - ``loocv_docs_skipped`` = n_docs - loocv_docs_scored.
  - LOOCV averages are computed only over scored folds (honest denominator).
"""

from __future__ import annotations

import json as _json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure supervision-row builder (verbatim extraction from queue.py:660-736)
# ---------------------------------------------------------------------------


def build_training_rows(
    label_rows: list[Any],
    pred_index: dict[tuple[str, str], dict],
) -> list[dict]:
    """Build supervision rows from label events joined against prediction events.

    This is a verbatim extraction of the loop body that lived inline in
    handle_retrain (queue.py:660-736).  The action→label logic is unchanged.
    handle_retrain delegates here so both paths share a single implementation.

    Args:
        label_rows: Records from the labels table
            (doc_id, field, action, payload, created_at).
        pred_index: Dict keyed by (doc_id, field) → prediction payload dict
            (candidates list + selected_candidate_idx).

    Returns:
        List of feature dicts, each with keys **features, "label",
        "doc_id", "target_field".
    """
    training_rows: list[dict] = []

    for row in label_rows:
        doc_id = row["doc_id"]
        field = row["field"]
        action = row["action"] or "correct"

        pred = pred_index.get((doc_id, field))
        if pred is None:
            # No prediction event for this (doc_id, field) — skip.
            continue

        candidates = pred.get("candidates") or []
        selected_idx = pred.get("selected_candidate_idx")

        raw_label_payload = row["payload"]
        if isinstance(raw_label_payload, str):
            label_payload = _json.loads(raw_label_payload)
        else:
            label_payload = dict(raw_label_payload) if raw_label_payload else {}

        for i, cand in enumerate(candidates):
            features = cand.get("features") or {}
            if not features:
                continue

            if action in ("not_in_document",):
                # Field absent — all candidates are negative.
                label = 0
            elif action == "reject":
                # Wrong prediction — the selected candidate is negative.
                # Other candidates carry no signal (no correct span provided).
                if selected_idx is not None and i == selected_idx:
                    label = 0
                else:
                    continue
            elif action == "approve":
                # Positive confirmation — the selected candidate is the target.
                if selected_idx is not None and i == selected_idx:
                    label = 1
                else:
                    label = 0
            elif action == "correct":
                # Span annotation — user provided correct value.
                # positive = candidate whose text equals the user-supplied text.
                # We resolve via text match in label_payload.correct_value
                # against the candidate raw_text by exact text equality (no bbox tie-break).
                correct_value = label_payload.get("correct_value")
                if not correct_value:
                    # No correct value supplied — treat like approve if we
                    # have a selected_idx, otherwise skip.
                    if selected_idx is not None and i == selected_idx:
                        label = 1
                    else:
                        label = 0
                else:
                    cand_text = (
                        cand.get("raw_text") or cand.get("normalized_text") or ""
                    )
                    label = 1 if cand_text.strip() == str(correct_value).strip() else 0
            else:
                continue

            training_rows.append(
                {
                    **features,
                    "label": label,
                    "doc_id": doc_id,
                    "target_field": field,
                }
            )

    return training_rows


# ---------------------------------------------------------------------------
# DB loader — same queries handle_retrain uses
# ---------------------------------------------------------------------------


async def _load_supervision(
    pool: asyncpg.Pool,
) -> tuple[list[Any], dict[tuple[str, str], dict]]:
    """Load label_rows and pred_index from Postgres.

    Uses the same two SELECT queries that handle_retrain uses (queue.py:584-655)
    so the evaluator sees exactly the same supervision data as the trainer.

    Returns:
        (label_rows, pred_index) — label_rows may be empty.
    """
    # 1. Load all labeling events (identical to queue.py:586-592).
    async with pool.acquire() as conn:
        label_rows = await conn.fetch(
            "SELECT doc_id, field, action, payload, created_at "
            "FROM labels "
            "ORDER BY created_at"
        )

    if not label_rows:
        return list(label_rows), {}

    labeled_doc_ids: list[str] = list({row["doc_id"] for row in label_rows})

    # 2. Load most-recent prediction per (doc_id, field) for labeled docs only
    #    (identical to queue.py:606-619).
    async with pool.acquire() as conn:
        pred_rows = await conn.fetch(
            "SELECT DISTINCT ON (doc_id, field) "
            "    doc_id, field, payload "
            "FROM doc_evaluations "
            "WHERE doc_id = ANY($1::text[]) "
            "ORDER BY doc_id, field, created_at DESC",
            labeled_doc_ids,
        )

    # Build pred_index (identical to queue.py:622-655).
    pred_index: dict[tuple[str, str], dict] = {}
    for row in pred_rows:
        key = (row["doc_id"], row["field"])
        raw_payload = row["payload"]
        if isinstance(raw_payload, str):
            pred_payload = _json.loads(raw_payload)
        else:
            pred_payload = dict(raw_payload) if raw_payload else {}
        pred_index[key] = pred_payload

    return list(label_rows), pred_index


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


async def evaluate_ranker(pool: asyncpg.Pool) -> dict:  # type: ignore[type-arg]
    """Compute in-sample and LOOCV ranker metrics on the current labeled data.

    Read-only: no filesystem writes, no MLflow writes, no model_runs writes.

    Args:
        pool: asyncpg connection pool connected to the invoicex Postgres DB.

    Returns:
        Flat dict with keys:
          n_docs, n_rows, n_positive, n_groups_with_positive,
          insample_ndcg_at_1, insample_mrr,
          loocv_ndcg_at_1, loocv_mrr,
          loocv_docs_scored, loocv_docs_skipped.
        On error conditions returns {"error": "<reason>"}.
        All float values rounded to 4 decimal places.
    """
    import polars as pl

    from .ranker import InvoiceFieldRanker

    # --- load supervision data ---
    label_rows, pred_index = await _load_supervision(pool)

    if not label_rows:
        return {"error": "no_labeled_rows"}

    rows = build_training_rows(label_rows, pred_index)

    if not rows:
        return {"error": "no_training_rows"}

    # Build the full DataFrame once — guarantees consistent column schema
    # across all LOOCV folds (heterogeneous feature dicts can vary per fold).
    df = pl.from_dicts(rows)

    group_cols = ["doc_id", "target_field"]
    # Only include group cols that are actually present (defensive).
    group_cols = [c for c in group_cols if c in df.columns]

    n_rows = len(df)
    n_positive = int(df["label"].sum()) if "label" in df.columns else 0
    doc_ids = df["doc_id"].unique().to_list() if "doc_id" in df.columns else []
    n_docs = len(doc_ids)

    # Count groups that have at least one positive label.
    if group_cols and "label" in df.columns:
        n_groups_with_positive = int(
            df.group_by(group_cols)
            .agg(pl.col("label").sum().alias("pos_sum"))
            .filter(pl.col("pos_sum") > 0)
            .height
        )
    else:
        n_groups_with_positive = 0

    # --- in-sample metrics ---
    insample_ndcg_at_1: float | None = None
    insample_mrr: float | None = None

    try:
        insample_ranker = InvoiceFieldRanker()
        insample_ranker.train(df)
        raw_insample = insample_ranker._compute_validation_metrics(
            df, "label", group_cols
        )
        insample_ndcg_at_1 = raw_insample.get("val_ndcg_at_1")
        insample_mrr = raw_insample.get("val_mrr")
    except Exception:
        logger.warning("evaluate_insample_failed", exc_info=True)

    # --- LOOCV metrics ---
    loocv_ndcg_scores: list[float] = []
    loocv_mrr_scores: list[float] = []
    loocv_docs_scored = 0
    loocv_docs_skipped = 0

    for held_doc_id in doc_ids:
        train_fold = df.filter(pl.col("doc_id") != held_doc_id)
        held_fold = df.filter(pl.col("doc_id") == held_doc_id)

        # Skip: training remainder has zero positives.
        if train_fold.is_empty() or int(train_fold["label"].sum()) == 0:
            loocv_docs_skipped += 1
            continue

        # Skip: held-out fold has no positive-bearing groups (nothing to measure).
        if held_fold.is_empty() or int(held_fold["label"].sum()) == 0:
            loocv_docs_skipped += 1
            continue

        try:
            fold_ranker = InvoiceFieldRanker()
            fold_ranker.train(train_fold)
            fold_metrics = fold_ranker._compute_validation_metrics(
                held_fold, "label", group_cols
            )
        except Exception:
            logger.warning(
                "evaluate_loocv_fold_failed (doc_id=%s)", held_doc_id, exc_info=True
            )
            loocv_docs_skipped += 1
            continue

        ndcg = fold_metrics.get("val_ndcg_at_1")
        mrr = fold_metrics.get("val_mrr")

        if ndcg is None and mrr is None:
            # No positive-bearing groups in the held-out fold after prediction.
            loocv_docs_skipped += 1
            continue

        if ndcg is not None:
            loocv_ndcg_scores.append(ndcg)
        if mrr is not None:
            loocv_mrr_scores.append(mrr)
        loocv_docs_scored += 1

    loocv_ndcg_at_1: float | None = (
        sum(loocv_ndcg_scores) / len(loocv_ndcg_scores) if loocv_ndcg_scores else None
    )
    loocv_mrr: float | None = (
        sum(loocv_mrr_scores) / len(loocv_mrr_scores) if loocv_mrr_scores else None
    )

    def _r(v: float | None) -> float | None:
        return round(v, 4) if v is not None else None

    return {
        "n_docs": n_docs,
        "n_rows": n_rows,
        "n_positive": n_positive,
        "n_groups_with_positive": n_groups_with_positive,
        "insample_ndcg_at_1": _r(insample_ndcg_at_1),
        "insample_mrr": _r(insample_mrr),
        "loocv_ndcg_at_1": _r(loocv_ndcg_at_1),
        "loocv_mrr": _r(loocv_mrr),
        "loocv_docs_scored": loocv_docs_scored,
        "loocv_docs_skipped": loocv_docs_skipped,
    }


if __name__ == "__main__":
    import asyncio
    import json as _json_main

    import asyncpg

    from .queue import _build_dsn
    from .settings import Settings

    async def _main() -> None:
        pool = await asyncpg.create_pool(_build_dsn(Settings()))
        try:
            result = await evaluate_ranker(pool)
        finally:
            await pool.close()
        print(_json_main.dumps(result, indent=2))

    asyncio.run(_main())
