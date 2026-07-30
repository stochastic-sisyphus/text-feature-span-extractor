"""Accumulator dataclasses for chunked retrain processing.

Each accumulator collects statistics across document chunks, allowing
raw DataFrames to be released between chunks.  After all chunks are
processed, the accumulators produce the final values consumed by
the retrain DAG.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from .logging import get_logger

if TYPE_CHECKING:
    pass  # future annotations only

logger = get_logger(__name__)


@dataclass
class AnchorCounters:
    """Token co-occurrence counters for learned anchors.

    Foreground counters are keyed **per field** (not per anchor_type) so
    that ``min_field_labels`` filtering can be applied after merging all
    chunks.  ``finalize_anchors`` maps field → anchor_type and merges.

    ``foreground[field_name][word]`` = count of how many times ``word``
    appeared near a correctly-extracted value of that field.
    ``background[word]`` = total count across all documents.
    ``fg_totals[field_name]`` = total foreground tokens for that field.
    ``field_label_counts[field_name]`` = total correct labels for that field.
    """

    foreground: dict[str, Counter[str]] = field(default_factory=dict)
    background: Counter[str] = field(default_factory=Counter)
    fg_totals: dict[str, int] = field(default_factory=dict)
    bg_total: int = 0
    field_label_counts: dict[str, int] = field(default_factory=dict)

    def merge(self, other: AnchorCounters) -> None:
        for key, counter in other.foreground.items():
            if key not in self.foreground:
                self.foreground[key] = Counter()
            self.foreground[key] += counter
        self.background += other.background
        for key, total in other.fg_totals.items():
            self.fg_totals[key] = self.fg_totals.get(key, 0) + total
        self.bg_total += other.bg_total
        for fname, count in other.field_label_counts.items():
            self.field_label_counts[fname] = (
                self.field_label_counts.get(fname, 0) + count
            )


@dataclass
class FeatureAccumulator:
    """Accumulated XGBoost feature matrices across chunks.

    Stores per-field lists of polars DataFrames that are concatenated
    into final training datasets after all chunks are processed.
    One FEATURE_COLUMNS-wide row per candidate, small compared to
    the raw token DataFrames.

    The accumulator stores polars frames internally; finalize() returns
    polars DataFrames for downstream consumers.
    """

    field_datasets: dict[str, list[pl.DataFrame]] = field(default_factory=dict)
    total_docs: int = 0
    total_rows: int = 0

    def add_field_data(self, field_name: str, data: pl.DataFrame) -> None:
        if field_name not in self.field_datasets:
            self.field_datasets[field_name] = []
        self.field_datasets[field_name].append(data)

    def merge(self, other: FeatureAccumulator) -> None:
        for field_name, datasets in other.field_datasets.items():
            if field_name not in self.field_datasets:
                self.field_datasets[field_name] = []
            self.field_datasets[field_name].extend(datasets)
        self.total_docs += other.total_docs
        self.total_rows += other.total_rows

    def finalize(self) -> dict[str, pl.DataFrame]:
        """Concatenate accumulated polars frames into final per-field datasets."""
        result: dict[str, pl.DataFrame] = {}
        for field_name, datasets in self.field_datasets.items():
            if datasets:
                result[field_name] = pl.concat(datasets, how="diagonal")
        return result


@dataclass
class CalibrationAccumulator:
    """Accumulated calibration samples across chunks."""

    samples: list[dict] = field(default_factory=list)

    def add(self, sample: dict) -> None:
        self.samples.append(sample)

    def merge(self, other: CalibrationAccumulator) -> None:
        self.samples.extend(other.samples)

    def finalize(self) -> list[dict]:
        return self.samples


@dataclass
class ChunkAccumulator:
    """Top-level accumulator that collects all statistics across chunks.

    After all chunks, call :meth:`finalize_anchors` and pass
    :attr:`features` to the training step.
    """

    anchors: AnchorCounters = field(default_factory=AnchorCounters)
    features: FeatureAccumulator = field(default_factory=FeatureAccumulator)

    # Aligned labels accumulated across chunks (needed for weights tuning
    # and post-training decode).  Relatively small: one row per label.
    # Stored as polars; consumer boundary (dag.weights, anchor counters)
    # converts to pandas before calling pandas-intrinsic code.
    aligned_parts: list[pl.DataFrame] = field(default_factory=list)

    # Candidates for labeled docs only — needed for weights tuning.
    # This is a small subset of all documents.  Stored as polars;
    # dag.weights call site converts to pandas.
    labeled_candidates: dict[str, pl.DataFrame] = field(default_factory=dict)

    def merge(self, other: ChunkAccumulator) -> None:
        self.anchors.merge(other.anchors)
        self.features.merge(other.features)
        self.aligned_parts.extend(other.aligned_parts)
        for k, v in other.labeled_candidates.items():
            if k not in self.labeled_candidates:
                self.labeled_candidates[k] = v

    def get_aligned(self) -> pl.DataFrame:
        """Return the full aligned polars DataFrame from accumulated parts.

        Returns an empty polars DataFrame when no parts have been accumulated.
        Callers that need pandas (e.g. dag.weights, accumulate_anchor_counters)
        must convert at their own boundary: accumulator.get_aligned().to_pandas().
        """
        if not self.aligned_parts:
            return pl.DataFrame()
        return pl.concat(self.aligned_parts, how="diagonal")

    def finalize_anchors(
        self,
        min_field_labels: int = 3,
        min_support: int = 2,
        min_lift: float = 2.0,
    ) -> dict[str, set[str]]:
        """Compute learned anchors from accumulated counters.

        Same lift-based filtering as ``learned_anchors.compute_learned_anchors``
        but operates on pre-accumulated foreground/background counters instead
        of re-scanning tokens.

        Foreground counters are keyed per field.  Fields whose total
        correct-label count is below *min_field_labels* are excluded
        (matching the full-batch filter), then remaining fields are
        merged by anchor_type before computing lift.
        """
        from invoices import schema as schema_mod

        if self.anchors.bg_total == 0:
            return {}

        # ── Filter fields by min_field_labels, merge by anchor_type ──
        merged_fg: dict[str, Counter[str]] = {}
        merged_fg_totals: dict[str, int] = {}

        for field_name, fg_counter in self.anchors.foreground.items():
            if self.anchors.field_label_counts.get(field_name, 0) < min_field_labels:
                continue
            fdef = schema_mod.load_field_defs().get(field_name)
            atype: str | None = fdef.anchor_family if fdef else None
            if atype is None:
                continue
            if atype not in merged_fg:
                merged_fg[atype] = Counter()
            merged_fg[atype] += fg_counter
            merged_fg_totals[atype] = merged_fg_totals.get(
                atype, 0
            ) + self.anchors.fg_totals.get(field_name, 0)

        # ── Lift filter per anchor_type ──────────────────────────────
        result: dict[str, set[str]] = {}
        for atype, fg_counter in merged_fg.items():
            fg_total = merged_fg_totals.get(atype, 0)
            if fg_total == 0:
                continue

            anchors: set[str] = set()
            for word, count in fg_counter.items():
                if count < min_support:
                    continue
                bg_count = self.anchors.background.get(word, 0)
                if bg_count == 0:
                    anchors.add(word)
                    continue
                fg_rate = count / fg_total
                bg_rate = bg_count / self.anchors.bg_total
                lift = fg_rate / bg_rate
                if lift >= min_lift:
                    anchors.add(word)

            if anchors:
                result[atype] = anchors

        return result
