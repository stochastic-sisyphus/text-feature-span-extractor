"""spaCy NER signal enrichment for candidates LazyFrame.

Adds five columns to the candidates frame:
  ner_is_org      — 1.0 if any ORG entity found in raw_text
  ner_is_date     — 1.0 if any DATE entity found
  ner_is_money    — 1.0 if any MONEY entity found
  ner_is_cardinal — 1.0 if any CARDINAL entity found
  ner_label       — top entity label string, or "NONE"

spaCy is imported lazily inside the enrichment function — no startup cost
when the module is imported.  The NLP pipeline is loaded once per process
via a module-level cache and reused across calls.

Graceful fallback: if spaCy is unavailable or no model is installed, the
function returns the candidates LazyFrame unchanged.  All five columns are
added with their zero/NONE defaults so downstream code sees a stable schema.

NER runs via polars map_batches — the entire raw_text Series is processed in
one Python call, avoiding row-by-row overhead.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Module-level NLP cache.  None = not yet attempted.  False = attempted but
# no model available.  Any other value = loaded spaCy Language object.
_NLP: Any = None

# Ordered list of candidate model names to try at load time.
# Populated from the installed-models list provided at integration time.
# Kept empty here — the loader falls back to spacy.util.get_installed_models().
_PREFERRED_MODELS: list[str] = []


def _load_nlp() -> Any:
    """Return a cached spaCy Language or None if unavailable.

    Attempts to load the first available installed model.  Sets _NLP to False
    permanently after a failed attempt so subsequent calls are O(1) no-ops.
    """
    global _NLP

    if _NLP is False:
        return None
    if _NLP is not None:
        return _NLP

    try:
        import spacy  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        logger.warning("spacy_not_installed", msg="NER enrichment disabled")
        _NLP = False
        return None

    # Build candidate list: preferred first, then whatever is installed.
    candidates: list[str] = list(_PREFERRED_MODELS)
    try:
        installed: list[str] = spacy.util.get_installed_models()
    except Exception:
        installed = []
    for m in installed:
        if m not in candidates:
            candidates.append(m)

    nlp = _try_load_first(spacy, candidates)
    if nlp is not None:
        _NLP = nlp
        return nlp

    logger.warning("spacy_no_model_available", msg="NER enrichment disabled")
    _NLP = False
    return None


def _try_load_first(spacy: Any, model_names: list[str]) -> Any:
    """Return the first successfully loaded spaCy model, or None.

    The try/except lives outside the loop body to avoid PERF203.
    """
    for model_name in model_names:
        nlp = _try_one_model(spacy, model_name)
        if nlp is not None:
            return nlp
    return None


def _try_one_model(spacy: Any, model_name: str) -> Any:
    """Attempt to load a single spaCy model; return None on failure."""
    try:
        nlp = spacy.load(
            model_name, disable=["parser", "lemmatizer", "attribute_ruler"]
        )
        logger.info("spacy_model_loaded", model=model_name)
        return nlp
    except Exception as exc:
        logger.debug("spacy_model_load_failed", model=model_name, error=str(exc))
        return None


_NER_SCHEMA: dict[str, pl.datatypes.DataTypeClass] = {
    "ner_is_org": pl.Float64,
    "ner_is_date": pl.Float64,
    "ner_is_money": pl.Float64,
    "ner_is_cardinal": pl.Float64,
    "ner_label": pl.String,
}


def _ner_defaults(n: int) -> pl.DataFrame:
    """Return a zero/NONE NER DataFrame with correct dtypes for n rows."""
    return pl.DataFrame(
        {
            "ner_is_org": pl.Series([0.0] * n, dtype=pl.Float64),
            "ner_is_date": pl.Series([0.0] * n, dtype=pl.Float64),
            "ner_is_money": pl.Series([0.0] * n, dtype=pl.Float64),
            "ner_is_cardinal": pl.Series([0.0] * n, dtype=pl.Float64),
            "ner_label": pl.Series(["NONE"] * n, dtype=pl.String),
        }
    )


def _ner_batch(texts: pl.Series) -> pl.DataFrame:
    """Run spaCy NER on a Series of strings; return a DataFrame of 5 columns.

    Uses nlp.pipe for batch throughput.  Returns zero/NONE defaults for every
    row if spaCy is unavailable.
    """
    n = len(texts)

    nlp = _load_nlp()
    if nlp is None:
        return _ner_defaults(n)

    is_org: list[float] = []
    is_date: list[float] = []
    is_money: list[float] = []
    is_cardinal: list[float] = []
    top_label: list[str] = []

    str_texts: list[str] = [t if t is not None else "" for t in texts.to_list()]

    try:
        import spacy  # type: ignore[import-untyped]  # noqa: F401

        for doc in nlp.pipe(str_texts, batch_size=256):  # type: ignore[union-attr,attr-defined]
            labels: set[str] = {ent.label_ for ent in doc.ents}
            is_org.append(1.0 if "ORG" in labels else 0.0)
            is_date.append(1.0 if "DATE" in labels else 0.0)
            is_money.append(1.0 if "MONEY" in labels else 0.0)
            is_cardinal.append(1.0 if "CARDINAL" in labels else 0.0)
            # Top label: first entity in reading order, or "NONE".
            top_label.append(doc.ents[0].label_ if doc.ents else "NONE")
    except Exception as exc:
        logger.warning("ner_batch_failed", error=str(exc))
        return _ner_defaults(n)

    return pl.DataFrame(
        {
            "ner_is_org": pl.Series(is_org, dtype=pl.Float64),
            "ner_is_date": pl.Series(is_date, dtype=pl.Float64),
            "ner_is_money": pl.Series(is_money, dtype=pl.Float64),
            "ner_is_cardinal": pl.Series(is_cardinal, dtype=pl.Float64),
            "ner_label": pl.Series(top_label, dtype=pl.String),
        }
    )


def enrich_with_ner(
    candidates_lf: pl.LazyFrame,
    text_col: str = "raw_text",
) -> pl.LazyFrame:
    """Add five NER feature columns to *candidates_lf*.

    Parameters
    ----------
    candidates_lf:
        Candidates LazyFrame (from chain ops or views.candidates_df).
    text_col:
        Column containing the raw candidate text.  Defaults to "raw_text".

    Returns
    -------
    pl.LazyFrame
        The input frame with five additional columns:
        ``ner_is_org``, ``ner_is_date``, ``ner_is_money``,
        ``ner_is_cardinal``, ``ner_label``.

        If spaCy is unavailable the frame is returned with zero/NONE defaults
        so downstream schema expectations are always met.
    """

    def _batch_fn(df: pl.DataFrame) -> pl.DataFrame:
        texts: pl.Series = df[text_col]
        ner_df = _ner_batch(texts)
        return pl.concat([df, ner_df], how="horizontal")

    output_schema: dict[str, pl.DataType | pl.datatypes.DataTypeClass] = {
        **dict(candidates_lf.collect_schema()),
        **_NER_SCHEMA,
    }
    return candidates_lf.map_batches(_batch_fn, schema=output_schema)
