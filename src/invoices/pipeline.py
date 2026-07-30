"""Functional core for document processing.

Pure pipeline functions: data in, data out.  The orchestrator
(imperative shell) handles all I/O — ledger checks, retries,
PDF reads — then calls these synchronous
functions for the actual compute.

Stages chain via ``*_with_data`` variants: every stage receives
in-memory DataFrames and returns in-memory results.  No disk I/O
happens inside the pipeline — PDFs arrive as bytes from the caller.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .doc import Doc
    from .types import Assignment

import polars as pl

from .logging import get_logger
from .metrics import (
    docs_auto_approved,
    docs_needs_review,
    field_confidence,
    field_status,
    pipeline_duration,
    predictions_emitted,
)

logger = get_logger(__name__)

# Maximum token count for which LiLT decode is attempted.
# CPU decode on very large docs is prohibitively slow; above this threshold the
# pipeline skips LiLT and falls back to the heuristic decoder.
LILT_MAX_TOKENS: int = 6000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class DocumentResult:
    """Output of :func:`run_document_pipeline`.  Single-document pure compute."""

    sha256: str
    doc_id: str
    n_tokens: int
    n_candidates: int
    assignments: dict[str, Assignment]
    contract: dict[str, Any] | None
    evaluation_entries: list[dict[str, Any]] = field(default_factory=list)
    needs_review: bool = False
    tokens_df: pl.DataFrame = field(default_factory=pl.DataFrame)
    candidates_df: pl.DataFrame = field(default_factory=pl.DataFrame)
    doc: Doc = field(default=None)  # type: ignore[assignment]
    route: str = "native"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_doc_id(sha256: str, source_id: str | None) -> str:
    """Return the canonical doc_id for a document.

    source_id (SharePoint item ID) is required — every document in production
    originates from SharePoint and carries a source_id.  A missing source_id
    is a data-integrity bug; raise loudly rather than fabricating a fake id.
    """
    if not source_id:
        raise ValueError(
            f"source_id is required to resolve doc_id (sha256={sha256[:16]})"
        )
    return source_id


# ---------------------------------------------------------------------------
# Document pipeline
# ---------------------------------------------------------------------------


def run_document_pipeline(
    sha256: str,
    pdf_bytes: bytes,
    learned_anchors: dict[str, set[str]] | None = None,
    calibration_mapping: dict[str, Any] | None = None,
    field_blend_weights: dict[str, Any] | None = None,
    vendor_corpus: set[str] | None = None,
    source_id: str | None = None,
    *,
    loaded_models: dict[str, Any] | None = None,
    prebuilt_doc: Doc | None = None,
) -> DocumentResult:
    """Execute the full document pipeline as a single function.

    Chains: tokenize -> candidates -> decode -> emit.

    All stages use ``*_with_data`` variants — data flows through
    in-memory DataFrames with zero disk writes.  ``pdf_bytes`` is
    required — the caller (orchestrator) is responsible for all I/O
    including reading PDFs from disk or remote sources.

    ``loaded_models`` is the model dict from the decoder tier (MLflow or
    storage backend).  When ``None``, the pipeline runs in heuristic-only
    mode (no ranker scoring).

    Returns a :class:`DocumentResult` with all outputs.
    """
    from . import confidence as conf
    from . import schema as schema_mod
    from .adaptive import (
        detect_address_city_tokens_with_data,
        detect_colon_name_values_with_data,
        detect_cross_page_headers_with_data,
        detect_document_labels_with_data,
    )
    from .candidates.chain import build_candidates_chain
    from .config import Config
    from .decoder.decode import decode_document_with_data
    from .emit import emit_document_with_data
    from .normalize import normalize_assignments
    from .schema import build_field_specs, expand_fields
    from .tokenize import build_doc
    from .views import candidates_df as _candidates_view

    doc_id = _resolve_doc_id(sha256, source_id)

    # Stage 1: Tokenize (single parse window — emits frozen Doc).
    # The tokens DataFrame is now derived from the Doc via the canonical
    # view rather than from the legacy eager tokenization path.
    pdf_source = io.BytesIO(pdf_bytes)

    t0 = time.perf_counter()
    doc = prebuilt_doc if prebuilt_doc is not None else build_doc(pdf_source)

    # ── Degenerate text-layer early exit ──────────────────────────────────
    # If the PDF's embedded text layer is corrupt (zero-width/adv chars),
    # no extraction mode recovers it.  Skip candidates/decode/emit entirely;
    # persist the Doc so the labeling UI has spatial geometry, and flag for
    # human review.  This branch is the future-OCR hook: when an OCR path
    # is added, replace needs_review=True with the OCR stage here.
    if conf.text_layer_degenerate(doc):
        logger.warning(
            "degenerate_text_layer_detected",
            sha256=sha256[:16],
            doc_id=doc_id,
        )
        docs_needs_review.inc()
        return DocumentResult(
            sha256=sha256,
            doc_id=doc_id,
            n_tokens=0,
            n_candidates=0,
            assignments={},
            contract=None,
            evaluation_entries=[],
            needs_review=True,
            tokens_df=pl.DataFrame(),
            candidates_df=pl.DataFrame(),
            doc=doc,
            route="degenerate_text_layer",
        )

    # Phase 0a seam: inline polars projection over persisted Doc.pages tokens.
    # tokens_lf stays lazy for threading into chain.apply_cross_row_enrichment.
    tokens_lf = pl.LazyFrame(
        [
            dict(tok) if not isinstance(tok, dict) else tok
            for p in doc.pages
            for tok in p.tokens
        ]
    )
    tokens_pl = tokens_lf.collect()
    # TokensDF is polars-native; downstream consumers (adaptive detectors,
    # normalize_assignments, DocumentResult.tokens_df) are now polars-native too.
    pipeline_duration.labels(stage="tokenize").observe(time.perf_counter() - t0)
    n_tokens = len(tokens_pl)
    n_pages = len(doc.pages)

    # ── Always-on dimension instrumentation ──────────────────────────────────
    # Log doc dimensions so operators can see exact sizes for every document
    # processed through the pipeline.  INFO level: these events are always
    # emitted and are the primary signal for tuning the size-guard thresholds.
    logger.info(
        "reextract_doc_dims",
        sha256=sha256[:16],
        pages=n_pages,
        tokens=n_tokens,
    )

    # Stage 2: Candidates — view-derived Doc feeds candidates_df directly.
    # Chain runs: learned-anchor overlay → scoring → soft-NMS → diversity →
    # cross-row enrichment → doc_id injection.
    # Single .collect() at the seam; CandidatesDF validated there.
    t0 = time.perf_counter()
    raw_candidates_lf = _candidates_view(doc)
    candidates_lf = build_candidates_chain(
        raw_candidates_lf,
        tokens_pl=tokens_pl,
        doc_id=doc_id,
        learned_anchors=learned_anchors,
        max_diversity_candidates=200,
    )
    # Seam: materialize once for downstream consumers.
    candidates_pl = candidates_lf.collect()
    pipeline_duration.labels(stage="candidates").observe(time.perf_counter() - t0)
    n_candidates = len(candidates_pl)
    candidates_list: list[dict[str, Any]] = (
        candidates_pl.to_dicts() if not candidates_pl.is_empty() else []
    )

    # Stage 3: Decode
    schema_obj = schema_mod.load()
    schema_fields = schema_obj.get("fields", [])
    field_profiles = build_field_specs(schema_fields)
    # Expand line-item specs into per-row variants based on the detected table.
    # schema_fields (unexpanded) is kept for emit, which groups rows into line_items.
    # The kernel sees a flat dict keyed by "{base}__row_{i}"; emit groups them back.
    expanded_specs = expand_fields(tuple(field_profiles.values()), doc)
    field_profiles = {spec.name: spec for spec in expanded_specs}
    expanded_schema_fields = [spec.name for spec in expanded_specs]
    field_defs = schema_obj.get("field_definitions", {})
    t0 = time.perf_counter()
    # Sentinel: None means "run heuristic decoder" (native / size-skipped / error).
    assignments: dict[str, Any] | None = None
    decode_route = "native"
    use_lilt = bool(loaded_models and "lilt" in loaded_models)

    # (a) Size pre-gate: skip LiLT on large documents to keep CPU decode bounded.
    if use_lilt and n_tokens > LILT_MAX_TOKENS:
        logger.info(
            "lilt_skipped_large",
            sha256=sha256[:16],
            n_tokens=n_tokens,
            threshold=LILT_MAX_TOKENS,
        )
        decode_route = "heuristic_lilt_skipped_large"
        use_lilt = False

    # Heuristic decode, defined once so the LiLT blend and the fallback paths
    # share a single call site (no arg drift).
    def _run_heuristic() -> dict[str, Any]:
        return decode_document_with_data(
            candidates_list=candidates_list,
            schema_fields=expanded_schema_fields,
            field_profiles=field_profiles,
            loaded_models=loaded_models,
            doc_labels=detect_document_labels_with_data(tokens_pl),
            colon_name_values=detect_colon_name_values_with_data(tokens_pl),
            cross_page_headers=detect_cross_page_headers_with_data(tokens_pl),
            address_city_tokens=detect_address_city_tokens_with_data(tokens_pl),
            field_defs=field_defs,
            base_cost=Config.decoder_base_cost,
            none_bias=Config.none_bias,
            ml_score_weight=Config.ml_score_weight,
            bootstrap_ml_score_weight=Config.bootstrap_ml_score_weight,
            field_blend_weights=field_blend_weights,
            calibration_mapping=calibration_mapping,
            vendor_corpus=vendor_corpus,
        )

    # (b) LiLT decode with per-field blend. Resilient at BOTH layers:
    #   - LiLT decode throws  → heuristic-only (assignments stays None below)
    #   - heuristic blend throws → keep LiLT-only (never discard LiLT's result)
    if use_lilt:
        lilt_assignments: dict[str, Any] | None = None
        try:
            from .decoder.lilt_decode import decode_document_with_lilt

            lilt = loaded_models["lilt"]  # type: ignore[index]
            lilt_assignments = decode_document_with_lilt(
                doc=doc,
                candidates_list=candidates_list,
                schema_fields=expanded_schema_fields,
                model=lilt["model"],
                tokenizer=lilt["tokenizer"],
                id2label=lilt["id2label"],
                none_bias=Config.none_bias,
            )
            logger.info("decode_via_lilt", sha256=sha256[:16])
        except Exception as exc:
            logger.warning("lilt_decode_failed", sha256=sha256[:16], error=str(exc))
            decode_route = "heuristic_lilt_error"
            # lilt_assignments stays None → heuristic path below

        if lilt_assignments is not None:
            # Per-field blend: keep LiLT where it produced a CANDIDATE; fill the
            # fields LiLT abstained on (NONE) from the heuristic decoder. A
            # heuristic failure must NOT discard LiLT's good result — fall back
            # to LiLT-only rather than nuking the whole doc.
            try:
                heuristic_assignments = _run_heuristic()
                merged: dict[str, Any] = {}
                n_lilt = 0
                n_heuristic_fill = 0
                for fld in set(lilt_assignments) | set(heuristic_assignments):
                    la = lilt_assignments.get(fld)
                    if (
                        la is not None
                        and getattr(la, "assignment_type", None) == "CANDIDATE"
                    ):
                        merged[fld] = la
                        n_lilt += 1
                    else:
                        ha = heuristic_assignments.get(fld)
                        merged[fld] = ha if ha is not None else la
                        if ha is not None:
                            n_heuristic_fill += 1
                assignments = merged
                decode_route = "lilt_blend"
                logger.info(
                    "decode_blend",
                    sha256=sha256[:16],
                    n_lilt=n_lilt,
                    n_heuristic_fill=n_heuristic_fill,
                )
            except Exception as exc:
                logger.warning(
                    "blend_heuristic_failed", sha256=sha256[:16], error=str(exc)
                )
                assignments = lilt_assignments
                decode_route = "lilt"

    # Native heuristic: runs for native, size-skipped, and error fallback paths.
    if assignments is None:
        assignments = _run_heuristic()
        logger.info("decode_via_heuristic", sha256=sha256[:16], route=decode_route)
    pipeline_duration.labels(stage="decode").observe(time.perf_counter() - t0)

    # Stage 4: Normalize + Emit
    normalized = normalize_assignments(assignments, sha256, tokens_df=tokens_pl)
    page_count = (
        int(tokens_pl["page_idx"].n_unique()) if not tokens_pl.is_empty() else 0
    )

    t0 = time.perf_counter()
    contract, evaluation_entries, needs_review = emit_document_with_data(
        sha256=sha256,
        doc_id=doc_id,
        assignments=normalized,
        schema_fields=schema_fields,
        page_count=page_count,
        confidence_auto_approve=Config.confidence_auto_approve,
        heuristic_base=Config.confidence_heuristic_base,
        decoder_base_cost=Config.decoder_base_cost,
    )
    pipeline_duration.labels(stage="emit").observe(time.perf_counter() - t0)

    # Per-field metrics: confidence histogram + field status counter.
    # Observe once per field per document using the status already in the contract.
    for f_name, f_output in contract["fields"].items():
        f_status = f_output["status"]  # "PREDICTED", "ABSTAIN", or "MISSING"
        field_confidence.labels(field=f_name, status=f_status).observe(
            f_output["confidence"]
        )
        field_status.labels(field=f_name, status=f_status).inc()
        predictions_emitted.labels(status=f_status).inc()

    # Document routing counters: mutually exclusive branches.
    if needs_review:
        docs_needs_review.inc()
    else:
        docs_auto_approved.inc()

    return DocumentResult(
        sha256=sha256,
        doc_id=doc_id,
        n_tokens=n_tokens,
        n_candidates=n_candidates,
        assignments=normalized,
        contract=contract,
        evaluation_entries=evaluation_entries or [],
        needs_review=needs_review,
        tokens_df=tokens_pl,
        candidates_df=candidates_pl,
        doc=doc,
        route=decode_route,
    )
