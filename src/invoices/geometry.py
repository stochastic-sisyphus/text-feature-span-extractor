"""Geometry utilities for bounding box operations.

This module consolidates bbox operations that were previously duplicated across
candidates.py, features.py, decoder.py, and emit.py. Provides a single source
of truth for geometric computations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Type alias for bounding box tuple (x0, y0, x1, y1)
BBoxTuple = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Immutable bounding box with normalized coordinates.

    Coordinates are normalized to [0, 1] range where:
    - x0, y0: top-left corner
    - x1, y1: bottom-right corner
    - y increases downward (PDF coordinate system)

    Attributes:
        x0: Left edge (normalized)
        y0: Top edge (normalized)
        x1: Right edge (normalized)
        y1: Bottom edge (normalized)
    """

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        """Validate bounding box coordinates."""
        if self.x1 < self.x0:
            raise ValueError(f"x1 ({self.x1}) must be >= x0 ({self.x0})")
        if self.y1 < self.y0:
            raise ValueError(f"y1 ({self.y1}) must be >= y0 ({self.y0})")

    @classmethod
    def from_tuple(cls, bbox: BBoxTuple) -> BoundingBox:
        """Create BoundingBox from (x0, y0, x1, y1) tuple.

        Args:
            bbox: Tuple of (x0, y0, x1, y1) coordinates

        Returns:
            BoundingBox instance
        """
        return cls(x0=bbox[0], y0=bbox[1], x1=bbox[2], y1=bbox[3])

    def to_tuple(self) -> BBoxTuple:
        """Convert to (x0, y0, x1, y1) tuple.

        Returns:
            Tuple of coordinates
        """
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def center(self) -> tuple[float, float]:
        """Get center point (cx, cy)."""
        return compute_center(self.to_tuple())

    @property
    def center_x(self) -> float:
        """Get center x coordinate."""
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        """Get center y coordinate."""
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        """Get width."""
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        """Get height."""
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        """Get area."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (height / width).

        Returns:
            Aspect ratio, or 0 if width is 0
        """
        if self.width == 0:
            return 0.0
        return self.height / self.width

    def iou(self, other: BoundingBox) -> float:
        """Compute IoU with another bounding box.

        Args:
            other: Another BoundingBox

        Returns:
            IoU value in [0, 1]
        """
        return compute_iou(self.to_tuple(), other.to_tuple())

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the bounding box.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside (inclusive of edges)
        """
        return bbox_contains_point(self.to_tuple(), x, y)

    def distance_to(self, other: BoundingBox) -> float:
        """Compute Euclidean distance between centers.

        Args:
            other: Another BoundingBox

        Returns:
            Distance between centers
        """
        cx1, cy1 = self.center
        cx2, cy2 = other.center
        return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def compute_center(bbox: BBoxTuple) -> tuple[float, float]:
    """Compute center point of a bounding box.

    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates

    Returns:
        Tuple of (center_x, center_y)

    Example:
        >>> compute_center((0.1, 0.2, 0.3, 0.4))
        (0.2, 0.3)
    """
    x0, y0, x1, y1 = bbox
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def compute_dimensions(bbox: BBoxTuple) -> tuple[float, float]:
    """Compute width and height of a bounding box.

    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates

    Returns:
        Tuple of (width, height)

    Example:
        >>> compute_dimensions((0.1, 0.2, 0.3, 0.6))
        (0.2, 0.4)
    """
    x0, y0, x1, y1 = bbox
    return (x1 - x0, y1 - y0)


def compute_area(bbox: BBoxTuple) -> float:
    """Compute area of a bounding box.

    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates

    Returns:
        Area (width * height)

    Example:
        >>> compute_area((0.0, 0.0, 0.5, 0.4))
        0.2
    """
    width, height = compute_dimensions(bbox)
    return width * height


def compute_iou(bbox1: BBoxTuple, bbox2: BBoxTuple) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes.

    IoU = intersection_area / union_area

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)

    Returns:
        IoU value in range [0, 1]. Returns 0 if boxes don't overlap.

    Example:
        >>> compute_iou((0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75))
        0.14285714285714285
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Compute intersection coordinates
    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    # No intersection
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


def normalize_bbox(
    bbox: BBoxTuple,
    page_width: float,
    page_height: float,
) -> BBoxTuple:
    """Normalize bounding box coordinates to [0, 1] range.

    Args:
        bbox: Tuple of (x0, y0, x1, y1) in PDF units
        page_width: Page width in PDF units
        page_height: Page height in PDF units

    Returns:
        Normalized bounding box tuple

    Example:
        >>> normalize_bbox((72, 72, 144, 144), 612, 792)
        (0.1176..., 0.0909..., 0.2352..., 0.1818...)
    """
    if page_width <= 0 or page_height <= 0:
        raise ValueError("Page dimensions must be positive")

    x0, y0, x1, y1 = bbox
    return (
        x0 / page_width,
        y0 / page_height,
        x1 / page_width,
        y1 / page_height,
    )


def bbox_contains_point(bbox: BBoxTuple, x: float, y: float) -> bool:
    """Check if a point is inside a bounding box (inclusive).

    Args:
        bbox: Tuple of (x0, y0, x1, y1) coordinates
        x: X coordinate of point
        y: Y coordinate of point

    Returns:
        True if point is inside or on the edge of the bbox

    Example:
        >>> bbox_contains_point((0.1, 0.1, 0.5, 0.5), 0.3, 0.3)
        True
        >>> bbox_contains_point((0.1, 0.1, 0.5, 0.5), 0.0, 0.0)
        False
    """
    x0, y0, x1, y1 = bbox
    return x0 <= x <= x1 and y0 <= y <= y1


def compute_overlap_area(bbox1: BBoxTuple, bbox2: BBoxTuple) -> float:
    """Compute the intersection area of two bounding boxes.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)

    Returns:
        Intersection area. Returns 0 if boxes don't overlap.
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    x_min = max(x1_min, x2_min)
    y_min = max(y1_min, y2_min)
    x_max = min(x1_max, x2_max)
    y_max = min(y1_max, y2_max)

    if x_max <= x_min or y_max <= y_min:
        return 0.0

    return (x_max - x_min) * (y_max - y_min)


def compute_distance_between_centers(bbox1: BBoxTuple, bbox2: BBoxTuple) -> float:
    """Compute Euclidean distance between centers of two bounding boxes.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)

    Returns:
        Euclidean distance between centers
    """
    cx1, cy1 = compute_center(bbox1)
    cx2, cy2 = compute_center(bbox2)
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def compute_directional_vector(
    bbox1: BBoxTuple,
    bbox2: BBoxTuple,
) -> tuple[float, float]:
    """Compute directional vector from bbox1 center to bbox2 center.

    Args:
        bbox1: Source bounding box (x0, y0, x1, y1)
        bbox2: Target bounding box (x0, y0, x1, y1)

    Returns:
        Tuple of (dx, dy) where:
        - dx > 0: bbox2 is to the RIGHT of bbox1
        - dx < 0: bbox2 is to the LEFT of bbox1
        - dy > 0: bbox2 is BELOW bbox1
        - dy < 0: bbox2 is ABOVE bbox1
    """
    cx1, cy1 = compute_center(bbox1)
    cx2, cy2 = compute_center(bbox2)
    return (cx2 - cx1, cy2 - cy1)


def merge_bboxes(bboxes: list[BBoxTuple]) -> BBoxTuple:
    """Merge multiple bounding boxes into their union.

    Args:
        bboxes: List of bounding boxes to merge

    Returns:
        Bounding box that contains all input boxes

    Raises:
        ValueError: If bboxes list is empty
    """
    if not bboxes:
        raise ValueError("Cannot merge empty list of bounding boxes")

    x0 = min(bbox[0] for bbox in bboxes)
    y0 = min(bbox[1] for bbox in bboxes)
    x1 = max(bbox[2] for bbox in bboxes)
    y1 = max(bbox[3] for bbox in bboxes)

    return (x0, y0, x1, y1)


def expand_bbox(bbox: BBoxTuple, margin: float) -> BBoxTuple:
    """Expand bounding box by a margin on all sides.

    Args:
        bbox: Original bounding box (x0, y0, x1, y1)
        margin: Margin to add (can be negative to shrink)

    Returns:
        Expanded bounding box
    """
    x0, y0, x1, y1 = bbox
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def clip_bbox(bbox: BBoxTuple, bounds: BBoxTuple) -> BBoxTuple:
    """Clip bounding box to be within bounds.

    Args:
        bbox: Bounding box to clip (x0, y0, x1, y1)
        bounds: Clipping bounds (x0, y0, x1, y1)

    Returns:
        Clipped bounding box
    """
    bx0, by0, bx1, by1 = bounds
    x0 = max(bbox[0], bx0)
    y0 = max(bbox[1], by0)
    x1 = min(bbox[2], bx1)
    y1 = min(bbox[3], by1)

    # Ensure valid bbox (x1 >= x0, y1 >= y0)
    x1 = max(x1, x0)
    y1 = max(y1, y0)

    return (x0, y0, x1, y1)


def is_valid_bbox(bbox: BBoxTuple) -> bool:
    """Check if bounding box coordinates are valid.

    Valid means:
    - x1 >= x0
    - y1 >= y0
    - All coordinates are finite

    Args:
        bbox: Bounding box to validate

    Returns:
        True if valid
    """
    x0, y0, x1, y1 = bbox

    # Check for finite values
    if not all(math.isfinite(v) for v in bbox):
        return False

    # Check ordering
    return x1 >= x0 and y1 >= y0
