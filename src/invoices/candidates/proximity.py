"""Typed proximity scoring for candidate-anchor spatial relationships."""

import math
from dataclasses import dataclass

import pandas as pd  # type: ignore[import-untyped]

from ..config import Config
from .constants import (
    ANCHOR_TYPE_DATE,
    ANCHOR_TYPE_ID,
    ANCHOR_TYPE_NAME,
    ANCHOR_TYPE_TAX,
    ANCHOR_TYPE_TOTAL,
)


@dataclass(slots=True)
class TypedAnchor:
    """Represents a typed anchor with position.

    Note: anchor_type is not stored on the instance since it's already
    the key in anchors_by_type dict - this eliminates redundancy.
    """

    cx: float
    cy: float
    text: str


class TypedProximityScorer:
    """Compute directional proximity features to typed cue anchors.

    This class addresses the fundamental reality that invoices are spatial:
    - A value BELOW a "Total" header belongs to that total
    - A value TO THE RIGHT OF "Invoice Date:" is the date value
    - Direction matters, not just distance

    Returns signed directional vectors (dx, dy) per anchor type, enabling
    the model to learn spatial relationships like "same column, below header".
    """

    def __init__(self) -> None:
        # Anchors grouped by type
        self.anchors_by_type: dict[str, list[TypedAnchor]] = {
            ANCHOR_TYPE_TOTAL: [],
            ANCHOR_TYPE_TAX: [],
            ANCHOR_TYPE_DATE: [],
            ANCHOR_TYPE_ID: [],
            ANCHOR_TYPE_NAME: [],
        }

    def merge_learned_anchors(
        self, learned: dict[str, set[str]], page_tokens: pd.DataFrame
    ) -> None:
        """Merge learned anchor keywords into the scorer.

        Scans page tokens for learned anchor words and adds any matches
        to the existing anchors_by_type dict. Call after find_typed_anchors().

        Args:
            learned: Dict mapping anchor_type -> set of learned anchor words
            page_tokens: DataFrame of tokens on this page
        """
        if not learned:
            return

        for token in page_tokens.itertuples(index=False):
            text_lower = token.text.lower().strip()
            cx = (token.bbox_norm_x0 + token.bbox_norm_x1) / 2
            cy = (token.bbox_norm_y0 + token.bbox_norm_y1) / 2

            for anchor_type, words in learned.items():
                if anchor_type not in self.anchors_by_type:
                    self.anchors_by_type[anchor_type] = []
                if text_lower in words:
                    self.anchors_by_type[anchor_type].append(
                        TypedAnchor(cx=cx, cy=cy, text=text_lower)
                    )

    def find_typed_anchors(self, page_tokens: pd.DataFrame) -> None:
        """Find anchors on a page, grouped by semantic type.

        Uses Config.get_anchor_keywords_by_type() for DRY iteration over
        anchor types instead of duplicating the keyword sets.
        """
        # Reset anchors
        for anchor_type in self.anchors_by_type:
            self.anchors_by_type[anchor_type] = []

        # Get keyword mapping from config (DRY - single source of truth)
        anchor_keywords = Config.get_anchor_keywords_by_type()

        for token in page_tokens.itertuples(index=False):
            text_lower = token.text.lower().strip()
            cx = (token.bbox_norm_x0 + token.bbox_norm_x1) / 2
            cy = (token.bbox_norm_y0 + token.bbox_norm_y1) / 2

            # Check each anchor type (a token can match multiple types)
            for anchor_type, keywords in anchor_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    self.anchors_by_type[anchor_type].append(
                        TypedAnchor(cx=cx, cy=cy, text=text_lower)
                    )

    def compute_directional_features(
        self, bbox_norm: tuple[float, float, float, float]
    ) -> dict[str, float]:
        """Compute directional vector features relative to each anchor type.

        Returns signed dx, dy values that preserve spatial semantics:
        - dx > 0: candidate is to the RIGHT of anchor (reading order after label)
        - dx < 0: candidate is to the LEFT of anchor
        - dy > 0: candidate is BELOW anchor (common for column values)
        - dy < 0: candidate is ABOVE anchor

        Also returns:
        - is_aligned_x: candidate is in same column (|dx| < threshold)
        - is_aligned_y: candidate is on same row (|dy| < threshold)
        """
        x0, y0, x1, y1 = bbox_norm
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        features: dict[str, float] = {}

        # Use centralized thresholds from Config
        column_threshold = Config.COLUMN_ALIGN_THRESHOLD
        row_threshold = Config.ROW_ALIGN_THRESHOLD

        for anchor_type, anchors in self.anchors_by_type.items():
            if not anchors:
                # No anchors of this type - set default features
                features[f"dx_to_{anchor_type}"] = 1.0  # Far away (no anchor)
                features[f"dy_to_{anchor_type}"] = 1.0
                features[f"dist_to_{anchor_type}"] = 1.414  # sqrt(2) diagonal
                features[f"aligned_x_{anchor_type}"] = 0.0
                features[f"aligned_y_{anchor_type}"] = 0.0
                features[f"reading_order_{anchor_type}"] = 0.0
                features[f"below_{anchor_type}"] = 0.0
                continue

            # Find nearest anchor of this type (by Euclidean distance)
            min_distance = float("inf")
            nearest_anchor = None

            for anchor in anchors:
                dx = cx - anchor.cx  # Signed: positive = right of anchor
                dy = cy - anchor.cy  # Signed: positive = below anchor
                distance = math.sqrt(dx * dx + dy * dy)

                if distance < min_distance:
                    min_distance = distance
                    nearest_anchor = anchor

            if nearest_anchor is None:
                continue

            # Compute signed directional vectors to nearest anchor
            dx = cx - nearest_anchor.cx  # Positive = to the right
            dy = cy - nearest_anchor.cy  # Positive = below

            # Store raw directional features (signed!)
            features[f"dx_to_{anchor_type}"] = dx
            features[f"dy_to_{anchor_type}"] = dy
            features[f"dist_to_{anchor_type}"] = min_distance

            # Compute alignment indicators
            features[f"aligned_x_{anchor_type}"] = (
                1.0 if abs(dx) < column_threshold else 0.0
            )
            features[f"aligned_y_{anchor_type}"] = (
                1.0 if abs(dy) < row_threshold else 0.0
            )

            # Semantic indicators
            # Reading order: to the right AND on same row (typical "Label: Value" pattern)
            features[f"reading_order_{anchor_type}"] = (
                1.0 if dx > 0 and abs(dy) < row_threshold else 0.0
            )
            # Below: in same column AND below (typical column header -> value pattern)
            features[f"below_{anchor_type}"] = (
                1.0 if dy > 0 and abs(dx) < column_threshold else 0.0
            )

        return features

    def compute_proximity_score(
        self, bbox_norm: tuple[float, float, float, float]
    ) -> float:
        """Legacy proximity score for backwards compatibility.

        Returns a single scalar score (closest distance to any anchor).
        Prefer using compute_directional_features() for new code.
        """
        x0, y0, x1, y1 = bbox_norm
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        min_distance = float("inf")

        for anchors in self.anchors_by_type.values():
            for anchor in anchors:
                dx = abs(cx - anchor.cx)
                dy = abs(cy - anchor.cy)

                # Same line bonus
                same_line_bonus = 0.1 if dy < 0.02 else 0.0

                # Reading order bonus (to the right)
                reading_order_bonus = 0.05 if cx > anchor.cx and dy < 0.02 else 0.0

                distance = (
                    math.sqrt(dx * dx + dy * dy) - same_line_bonus - reading_order_bonus
                )
                min_distance = min(min_distance, distance)

        if min_distance == float("inf"):
            return 0.0

        return max(0.0, 1.0 - min_distance)

    def has_any_anchor_relationship(
        self, bbox_norm: tuple[float, float, float, float], max_distance: float = 0.5
    ) -> bool:
        """Check if candidate has any spatial relationship to any anchor.

        This is the signal-to-noise filter: if a candidate is not near
        any anchor of any type, it's noise and should be pruned early.

        Args:
            bbox_norm: Candidate bounding box (normalized)
            max_distance: Maximum distance to consider "related" (default 0.5 = half page)

        Returns:
            True if candidate is near at least one anchor, False otherwise
        """
        x0, y0, x1, y1 = bbox_norm
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        for anchors in self.anchors_by_type.values():
            for anchor in anchors:
                dx = cx - anchor.cx
                dy = cy - anchor.cy
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= max_distance:
                    return True

        return False
