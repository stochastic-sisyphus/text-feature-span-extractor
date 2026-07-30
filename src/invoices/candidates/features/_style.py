"""Style features (font size z-score, bold/italic flags)."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_style_features_enhanced(
    font_size: float,
    is_bold: bool,
    is_italic: bool,
    font_hash: str,
    page_font_sizes: list[float],
) -> dict[str, Any]:
    """Enhanced style features."""
    # Font size z-score relative to page
    if page_font_sizes and len(page_font_sizes) > 1:
        page_mean = np.mean(page_font_sizes)
        page_std = np.std(page_font_sizes)
        font_size_z = (float(font_size) - float(page_mean)) / max(float(page_std), 1e-6)
    else:
        font_size_z = 0.0

    return {
        "font_size": font_size,
        "font_size_z": font_size_z,
        "is_bold": is_bold,
        "is_italic": is_italic,
        "font_hash": str(font_hash),  # Ensure string for safety
        "font_size_large": font_size_z > 1.0,
        "font_size_small": font_size_z < -1.0,
    }
