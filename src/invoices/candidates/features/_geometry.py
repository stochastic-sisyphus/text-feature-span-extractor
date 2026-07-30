"""Geometry features, density grid, section priors, soft-NMS."""

from __future__ import annotations

import math
from typing import Any

from ...config import Config
from ...geometry import compute_iou
from ..spans import PageGrid


def compute_geometry_features_enhanced(
    bbox_norm: tuple[float, float, float, float], page_width: float, page_height: float
) -> dict[str, Any]:
    """Enhanced geometry features with relative positioning.

    Addresses the fundamental reality that footers float based on content.
    Uses both top-down (y) and bottom-up (y_from_bottom) coordinates.
    """
    x0, y0, x1, y1 = bbox_norm

    # Center and dimensions
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0

    # Bottom-up coordinate: critical for footer elements (Total, Tax, etc.)
    # On a 1-page invoice, Total might be at y=0.8
    # On a 2-page invoice, Total might still be at y_from_bottom=0.1
    y_from_bottom = 1.0 - y1  # Distance from bottom edge to element bottom

    # Quadrant indicators for coarse spatial reasoning
    in_top_half = cy < 0.5
    in_left_half = cx < 0.5
    in_bottom_quarter = cy > 0.75
    in_top_quarter = cy < 0.25
    in_right_third = cx > 0.67

    return {
        # Absolute position
        "center_x": cx,
        "center_y": cy,
        "width": w,
        "height": h,
        # Top-down positioning
        "distance_to_top": cy,
        "distance_to_bottom": 1.0 - cy,
        "distance_to_left": cx,
        "distance_to_right": 1.0 - cx,
        "distance_to_center": math.hypot(cx - 0.5, cy - 0.5),
        # Bottom-up positioning (critical for footers)
        "y_from_bottom": y_from_bottom,
        "y0_from_bottom": 1.0 - y0,  # Top edge distance from page bottom
        # Shape features
        "aspect_ratio": h / max(w, 0.001),
        "area": w * h,
        # Quadrant indicators
        "in_top_half": float(in_top_half),
        "in_left_half": float(in_left_half),
        "in_bottom_quarter": float(in_bottom_quarter),
        "in_top_quarter": float(in_top_quarter),
        "in_right_third": float(in_right_third),
        # Combined position indicators (common invoice patterns)
        "in_amount_region": float(cx > 0.5 and cy > 0.5),  # Bottom-right quadrant
    }


def compute_local_density_grid(
    bbox_norm: tuple[float, float, float, float],
    page_grid: PageGrid,
    window_size: float = 0.1,
) -> float:
    """Compute local density using grid neighbors."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    # Get neighbors from grid
    neighbors = page_grid.get_neighbors(cx, cy, radius=2)

    # Count neighbors within window
    count = 0
    for neighbor in neighbors:
        if neighbor is None:
            continue

        if (
            isinstance(neighbor, dict)
            and "center_x" in neighbor
            and "center_y" in neighbor
        ):
            nx, ny = neighbor["center_x"], neighbor["center_y"]
            if abs(nx - cx) <= window_size / 2 and abs(ny - cy) <= window_size / 2:
                count += 1

    return count / max(window_size * window_size, 0.01)


def compute_section_prior(
    bbox_norm: tuple[float, float, float, float], page_idx: int
) -> float:
    """Compute section-based prior weights."""
    x0, y0, x1, y1 = bbox_norm
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    prior = 0.0

    # Early-page boost: invoice metadata clusters on the first few pages
    if page_idx <= Config.early_page_max_idx:
        decay = (Config.early_page_max_idx + 1 - page_idx) / (
            Config.early_page_max_idx + 1
        )
        prior += Config.early_page_boost * decay

    # Top-right corner of first page (common for amounts)
    if page_idx == 0 and cx > 0.6 and cy < 0.3:
        prior += 0.1

    # Bottom totals band (any page)
    if cy > 0.8:
        prior += 0.05

    # Header band (top 20%)
    if cy < 0.2:
        prior += 0.03

    return prior


# Re-export compute_iou from geometry.py as compute_overlap_iou for backward compat
compute_overlap_iou = compute_iou


def apply_soft_nms_grid(
    candidates: list[dict[str, Any]], page_grid: PageGrid, lambda_param: float = 0.5
) -> list[dict[str, Any]]:
    """Apply soft non-maximum suppression using grid neighbors."""
    if not candidates:
        return []

    # Sort by score (descending)
    candidates_sorted = sorted(
        candidates, key=lambda x: x.get("total_score", 0.0), reverse=True
    )

    # Apply soft decay to overlapping candidates
    for _i, candidate in enumerate(candidates_sorted):
        bbox = candidate["bbox_norm"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        # Get higher-scored neighbors
        neighbors = page_grid.get_neighbors(cx, cy, radius=1)

        for neighbor in neighbors:
            if neighbor is candidate:
                continue

            # Only consider higher-scored neighbors
            neighbor_score = neighbor.get("total_score", 0.0)
            if neighbor_score <= candidate.get("total_score", 0.0):
                continue

            # Compute IoU
            iou = compute_iou(candidate["bbox_norm"], neighbor["bbox_norm"])

            # Apply soft decay
            if iou > 0.1:  # Only apply if there's meaningful overlap
                decay = math.exp(-lambda_param * iou)
                candidate["total_score"] = candidate.get("total_score", 0.0) * decay

    return candidates_sorted
