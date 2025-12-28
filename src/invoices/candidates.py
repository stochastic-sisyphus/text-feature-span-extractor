"""Candidate generation module for balanced span proposals with feature extraction."""

import hashlib
import math
import random
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import ingest, paths, tokenize

# Candidate bucket types
BUCKET_DATE_LIKE = "date_like"
BUCKET_AMOUNT_LIKE = "amount_like"
BUCKET_ID_LIKE = "id_like"
BUCKET_KEYWORD_PROXIMAL = "keyword_proximal"
BUCKET_RANDOM_NEGATIVE = "random_negative"

# Key invoice keywords to look for (expanded from existing)
INVOICE_KEYWORDS = {
    "invoice",
    "inv",
    "invoice#",
    "invoice number",
    "invoice no",
    "account",
    "account#",
    "account number",
    "account no",
    "acct",
    "amount due",
    "total due",
    "total amount",
    "balance due",
    "due",
    "due date",
    "payment due",
    "date due",
    "total",
    "amount",
    "balance",
    "statement",
    "document",
    "bill",
    "billing",
}

# Currency symbols
CURRENCY_SYMBOLS = {"$", "€", "£", "¥", "₹", "₽", "USD", "EUR", "GBP", "CAD"}


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for deduplication purposes only."""
    return text.strip().lower().replace(" ", "").replace(",", "").replace(".", "")


def is_date_like_soft(text: str) -> bool:
    """Soft check if text looks like a date - no regex, just pattern-based."""
    text = text.strip()
    if len(text) < 4 or len(text) > 20:
        return False

    # Count digits and separators
    digits = sum(1 for c in text if c.isdigit())
    separators = sum(1 for c in text if c in "/-")
    letters = sum(1 for c in text if c.isalpha())

    # Date-like if mostly digits with some separators
    if digits >= 4 and separators >= 1 and digits + separators + letters == len(text):
        return True

    # Month names check
    month_indicators = [
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    text_lower = text.lower()
    if any(month in text_lower for month in month_indicators):
        return True

    return False


def is_amount_like_soft(text: str) -> bool:
    """Soft check if text looks like a monetary amount."""
    text = text.strip()
    if len(text) < 1:
        return False

    # Check for currency symbols
    has_currency = any(symbol in text for symbol in CURRENCY_SYMBOLS)

    # Count digits and decimal points
    digits = sum(1 for c in text if c.isdigit())
    decimals = text.count(".")
    commas = text.count(",")

    # Amount-like if has currency or mostly digits with decimals/commas
    if has_currency:
        return True

    if digits >= 2 and (decimals == 1 or commas >= 1):
        return True

    return False


def is_id_like_soft(text: str) -> bool:
    """Soft check if text looks like an ID."""
    text = text.strip()
    if len(text) < 3 or len(text) > 30:
        return False

    # Must be alphanumeric with reasonable mix
    if not all(c.isalnum() or c in "-_#" for c in text):
        return False

    # Must have at least one digit
    if not any(c.isdigit() for c in text):
        return False

    return True


class PageGrid:
    """Coarse grid for neighbor-only lookups."""

    def __init__(self, grid_size: int = 32):
        self.grid_size = grid_size
        self.cells = defaultdict(list)  # cell_coord -> list of items

    def add_item(self, x_norm: float, y_norm: float, item: Any) -> None:
        """Add an item to the grid at normalized coordinates."""
        cell_x = min(int(x_norm * self.grid_size), self.grid_size - 1)
        cell_y = min(int(y_norm * self.grid_size), self.grid_size - 1)
        self.cells[(cell_x, cell_y)].append(item)

    def get_neighbors(self, x_norm: float, y_norm: float, radius: int = 1) -> list[Any]:
        """Get all items in neighboring cells."""
        cell_x = min(int(x_norm * self.grid_size), self.grid_size - 1)
        cell_y = min(int(y_norm * self.grid_size), self.grid_size - 1)

        neighbors = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = cell_x + dx, cell_y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    neighbors.extend(self.cells[(nx, ny)])

        return neighbors


class SpanBuilder:
    """Build multi-token spans from line-local adjacency."""

    def __init__(self, max_gap: float = 0.02, max_span_tokens: int = 8):
        self.max_gap = max_gap  # Maximum normalized x-gap between adjacent tokens
        self.max_span_tokens = max_span_tokens  # Maximum tokens per span

    def build_spans(self, page_tokens: pd.DataFrame) -> list[dict[str, Any]]:
        """Build spans from tokens on a single page."""
        if page_tokens.empty:
            return []

        spans = []

        # Group tokens by line_id
        for line_id in page_tokens["line_id"].unique():
            line_tokens = page_tokens[page_tokens["line_id"] == line_id].copy()
            line_tokens = line_tokens.sort_values("bbox_norm_x0")

            # Build spans within this line
            line_spans = self._build_line_spans(line_tokens)
            spans.extend(line_spans)

        return spans

    def _build_line_spans(self, line_tokens: pd.DataFrame) -> list[dict[str, Any]]:
        """Build spans within a single line using adjacency and cohesion scoring."""
        if line_tokens.empty:
            return []

        spans = []
        tokens_list = list(line_tokens.iterrows())

        # Start with each token as a potential span start
        for i, (_, start_token) in enumerate(tokens_list):
            span_tokens = [start_token]

            # Try to extend the span with adjacent tokens up to max length
            for j in range(i + 1, min(i + self.max_span_tokens, len(tokens_list))):
                _, candidate_token = tokens_list[j]

                # Check if adjacent (small gap)
                gap = candidate_token["bbox_norm_x0"] - span_tokens[-1]["bbox_norm_x1"]
                if gap <= self.max_gap:
                    span_tokens.append(candidate_token)
                else:
                    break  # Gap too large, stop extending

            # Create span from collected tokens (let scoring decide quality)
            if span_tokens:
                span = self._create_span_from_tokens(span_tokens)
                if span:
                    spans.append(span)

        return spans

    def _create_span_from_tokens(
        self, tokens: list[pd.Series]
    ) -> dict[str, Any] | None:
        """Create a span from a list of tokens."""
        if not tokens:
            return None

        # Combine text
        raw_text = " ".join(token["text"] for token in tokens)
        normalized_text = normalize_text_for_dedup(raw_text)

        # Skip very short spans
        if len(raw_text.strip()) < 2:
            return None

        # Compute bounding box
        min_x = min(token["bbox_norm_x0"] for token in tokens)
        min_y = min(token["bbox_norm_y0"] for token in tokens)
        max_x = max(token["bbox_norm_x1"] for token in tokens)
        max_y = max(token["bbox_norm_y1"] for token in tokens)

        # Cohesion score (higher for more compact spans)
        span_width = max_x - min_x
        token_count = len(tokens)
        cohesion_score = token_count / max(span_width, 0.01)

        # Use first token's metadata as representative
        first_token = tokens[0]

        return {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "token_count": token_count,
            "cohesion_score": cohesion_score,
            "bbox_norm": (min_x, min_y, max_x, max_y),
            "token_ids": [str(token["token_id"]) for token in tokens],
            "token_indices": [int(token["token_idx"]) for token in tokens],
            "page_idx": int(first_token["page_idx"]),
            "line_id": int(first_token["line_id"]),
            "font_size": float(first_token["font_size"]),
            "is_bold": bool(first_token["is_bold"]),
            "is_italic": bool(first_token["is_italic"]),
            "font_hash": str(first_token["font_hash"]),
            "page_width": float(first_token["page_width"]),
            "page_height": float(first_token["page_height"]),
        }


class ProximityScorer:
    """Compute proximity scores to cue anchors."""

    def __init__(self):
        self.cue_anchors = []

    def find_cue_anchors(self, page_tokens: pd.DataFrame) -> None:
        """Find cue anchors (centers of keyword tokens) on a page."""
        self.cue_anchors = []

        for _, token in page_tokens.iterrows():
            text_lower = token["text"].lower().strip()
            if any(keyword in text_lower for keyword in INVOICE_KEYWORDS):
                cx = (token["bbox_norm_x0"] + token["bbox_norm_x1"]) / 2
                cy = (token["bbox_norm_y0"] + token["bbox_norm_y1"]) / 2
                self.cue_anchors.append((cx, cy, text_lower))

    def compute_proximity_score(
        self, bbox_norm: tuple[float, float, float, float]
    ) -> float:
        """Compute proximity score to nearest cue anchor."""
        if not self.cue_anchors:
            return 0.0

        x0, y0, x1, y1 = bbox_norm
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        min_distance = float("inf")
        for anchor_x, anchor_y, _ in self.cue_anchors:
            # Weighted distance with slight bonus for same line and reading order
            dx = abs(cx - anchor_x)
            dy = abs(cy - anchor_y)

            # Same line bonus
            same_line_bonus = 0.1 if dy < 0.02 else 0.0

            # Reading order bonus (to the right)
            reading_order_bonus = 0.05 if cx > anchor_x and dy < 0.02 else 0.0

            distance = (
                math.sqrt(dx * dx + dy * dy) - same_line_bonus - reading_order_bonus
            )
            min_distance = min(min_distance, distance)

        # Convert distance to score (closer = higher score)
        return max(0.0, 1.0 - min_distance)


def compute_text_features_enhanced(text: str) -> dict[str, Any]:
    """Enhanced text features without regex."""
    text_clean = text.strip()

    if not text_clean:
        return {
            "text_length": 0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "currency_flag": False,
            "unigram_hash": "",
            "bigram_hash": "",
        }

    features = {
        "text_length": len(text_clean),
        "digit_ratio": sum(1 for c in text_clean if c.isdigit()) / len(text_clean),
        "uppercase_ratio": sum(1 for c in text_clean if c.isupper()) / len(text_clean),
        "currency_flag": any(symbol in text_clean for symbol in CURRENCY_SYMBOLS),
    }

    # Hashed n-grams (safe string handling)
    words = text_clean.lower().split()

    # Unigram hashes (limit to prevent overflow)
    unigram_hashes = []
    for word in words[:3]:  # Limit to first 3
        hash_obj = hashlib.md5(word.encode("utf-8", errors="ignore"))
        unigram_hashes.append(hash_obj.hexdigest()[:8])
    features["unigram_hash"] = ",".join(unigram_hashes)

    # Bigram hashes
    if len(words) > 1:
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(min(2, len(words) - 1))]
        bigram_hashes = []
        for bigram in bigrams:
            hash_obj = hashlib.md5(bigram.encode("utf-8", errors="ignore"))
            bigram_hashes.append(hash_obj.hexdigest()[:8])
        features["bigram_hash"] = ",".join(bigram_hashes)
    else:
        features["bigram_hash"] = ""

    return features


def compute_geometry_features_enhanced(
    bbox_norm: tuple[float, float, float, float], page_width: float, page_height: float
) -> dict[str, Any]:
    """Enhanced geometry features."""
    x0, y0, x1, y1 = bbox_norm

    # Center and dimensions
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    w = x1 - x0
    h = y1 - y0

    return {
        "center_x": cx,
        "center_y": cy,
        "width_norm": w,
        "height_norm": h,
        "distance_to_top": cy,
        "distance_to_bottom": 1.0 - cy,
        "distance_to_left": cx,
        "distance_to_right": 1.0 - cx,
        "distance_to_center": math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2),
        "aspect_ratio": h / max(w, 0.001),
        "area_norm": w * h,
    }


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
        font_size_z = (font_size - page_mean) / max(page_std, 1e-6)
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
            iou = compute_overlap_iou(candidate["bbox_norm"], neighbor["bbox_norm"])

            # Apply soft decay
            if iou > 0.1:  # Only apply if there's meaningful overlap
                decay = math.exp(-lambda_param * iou)
                candidate["total_score"] = candidate.get("total_score", 0.0) * decay

    return candidates_sorted


def compute_overlap_iou(
    bbox1: tuple[float, float, float, float], bbox2: tuple[float, float, float, float]
) -> float:
    """Compute IoU (Intersection over Union) of two bounding boxes."""
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


def diversity_sampling(
    candidates: list[dict[str, Any]], max_candidates: int = 200
) -> list[dict[str, Any]]:
    """Apply diversity sampling with type-based stratification."""
    if len(candidates) <= max_candidates:
        return candidates

    # Group by type shape characteristics
    type_groups = defaultdict(list)

    for candidate in candidates:
        # Determine type shape based on content and features
        text = candidate.get("raw_text", "").strip()

        if is_date_like_soft(text):
            # Group by approximate date format
            if "/" in text or "-" in text:
                type_key = "date_numeric"
            else:
                type_key = "date_text"
        elif is_amount_like_soft(text):
            # Group by magnitude (rough bins)
            digit_count = sum(1 for c in text if c.isdigit())
            if digit_count <= 2:
                type_key = "amount_small"
            elif digit_count <= 4:
                type_key = "amount_medium"
            else:
                type_key = "amount_large"
        elif is_id_like_soft(text):
            # Group by alphanumeric shape
            has_letters = any(c.isalpha() for c in text)
            has_separators = any(c in "-_#" for c in text)
            if has_letters and has_separators:
                type_key = "id_complex"
            elif has_letters:
                type_key = "id_alphanum"
            else:
                type_key = "id_numeric"
        else:
            type_key = "other"

        type_groups[type_key].append(candidate)

    # Sample proportionally from each group
    result = []
    total_groups = len(type_groups)

    if total_groups == 0:
        return candidates[:max_candidates]

    for _group_key, group_candidates in type_groups.items():
        # Sort by total_score within group
        group_candidates.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)

        # Take proportional sample (with minimum 1 per group if possible)
        group_quota = max(1, max_candidates // total_groups)
        remaining_quota = max_candidates - len(result)
        actual_quota = min(group_quota, len(group_candidates), remaining_quota)

        result.extend(group_candidates[:actual_quota])

        if len(result) >= max_candidates:
            break

    return result[:max_candidates]


def generate_candidates_enhanced(sha256: str) -> tuple[int, dict[str, Any]]:
    """Enhanced candidate generation with spans, proximity, soft-NMS, and diversity."""

    # Timing dictionary for per-page performance
    timings = {
        "spans": 0.0,
        "anchors": 0.0,
        "scoring": 0.0,
        "soft_nms": 0.0,
        "dedupe": 0.0,
        "diversity": 0.0,
        "total": 0.0,
    }

    start_time = time.time()

    # Check if already exists (idempotency)
    candidates_path = paths.get_candidates_path(sha256)
    if candidates_path.exists():
        existing_df = pd.read_parquet(candidates_path)
        print(
            f"Candidates already exist for {sha256[:16]}: {len(existing_df)} candidates"
        )
        return len(existing_df), timings

    # Get tokens
    tokens_df = tokenize.get_document_tokens(sha256)
    if tokens_df.empty:
        print(f"No tokens found for {sha256[:16]}")
        return 0, timings

    doc_info = ingest.get_document_info(sha256)
    if not doc_info:
        raise ValueError(f"Document not found: {sha256}")

    doc_id = doc_info["doc_id"]

    # Set random seed for deterministic behavior
    random.seed(int(sha256[:8], 16))
    np.random.seed(int(sha256[:8], 16))

    all_candidates = []
    bucket_counts = {
        BUCKET_DATE_LIKE: 0,
        BUCKET_AMOUNT_LIKE: 0,
        BUCKET_ID_LIKE: 0,
        BUCKET_KEYWORD_PROXIMAL: 0,
        BUCKET_RANDOM_NEGATIVE: 0,
    }

    # Coverage probe counters
    coverage_stats = {
        "total_spans": 0,
        "cue_proximal_spans": 0,
        "region_prior_spans": 0,
        "date_like_spans": 0,
        "amount_like_spans": 0,
        "id_like_spans": 0,
    }

    # Process each page independently
    for page_idx in sorted(tokens_df["page_idx"].unique()):
        page_start_time = time.time()

        page_tokens = tokens_df[tokens_df["page_idx"] == page_idx].copy()
        if page_tokens.empty:
            continue

        print(f"Processing page {page_idx} with {len(page_tokens)} tokens")

        # 1. Span assembly (line-local only)
        spans_start = time.time()
        span_builder = SpanBuilder()
        spans = span_builder.build_spans(page_tokens)
        timings["spans"] += time.time() - spans_start

        if not spans:
            continue

        coverage_stats["total_spans"] += len(spans)

        # 2. Find cue anchors
        anchors_start = time.time()
        proximity_scorer = ProximityScorer()
        proximity_scorer.find_cue_anchors(page_tokens)
        timings["anchors"] += time.time() - anchors_start

        # 3. Build page grid for neighbor-only operations
        page_grid = PageGrid()

        # 4. Score spans and build candidates
        scoring_start = time.time()
        page_font_sizes = page_tokens["font_size"].tolist()

        page_candidates = []
        for span in spans:
            # Compute all features
            text_features = compute_text_features_enhanced(span["raw_text"])
            geometry_features = compute_geometry_features_enhanced(
                span["bbox_norm"], span["page_width"], span["page_height"]
            )
            style_features = compute_style_features_enhanced(
                span["font_size"],
                span["is_bold"],
                span["is_italic"],
                span["font_hash"],
                page_font_sizes,
            )

            # Proximity score
            proximity_score = proximity_scorer.compute_proximity_score(
                span["bbox_norm"]
            )

            # Section prior
            section_prior = compute_section_prior(span["bbox_norm"], page_idx)

            # Local density (will be computed after grid is populated)
            local_density = 0.0

            # Determine bucket type
            text = span["raw_text"]
            bucket = None

            if is_date_like_soft(text):
                bucket = BUCKET_DATE_LIKE
                coverage_stats["date_like_spans"] += 1
            elif is_amount_like_soft(text):
                bucket = BUCKET_AMOUNT_LIKE
                coverage_stats["amount_like_spans"] += 1
            elif is_id_like_soft(text):
                bucket = BUCKET_ID_LIKE
                coverage_stats["id_like_spans"] += 1
            elif proximity_score > 0.3:  # Proximal to keywords
                bucket = BUCKET_KEYWORD_PROXIMAL
                coverage_stats["cue_proximal_spans"] += 1
            elif random.random() < 0.02:  # 2% random negatives
                bucket = BUCKET_RANDOM_NEGATIVE
            else:
                continue  # Skip if doesn't fit any bucket

            # Track coverage probes
            if proximity_score > 0.2:  # Low-bar proximity
                coverage_stats["cue_proximal_spans"] += 1

            if section_prior > 0.02:  # Within prior zones
                coverage_stats["region_prior_spans"] += 1

            # Compute total score (base features + priors)
            base_score = (
                span["cohesion_score"] * 0.3
                + text_features.get("text_length", 0) * 0.1
                + (1.0 - geometry_features.get("distance_to_center", 1.0)) * 0.2
                + style_features.get("font_size_z", 0) * 0.1
            )

            total_score = base_score + proximity_score * 0.2 + section_prior

            # Create candidate
            candidate_id = f"{doc_id}_{page_idx}_{min(span['token_indices'])}"

            candidate = {
                "candidate_id": str(candidate_id),  # Ensure string for safety
                "doc_id": str(doc_id),
                "sha256": str(sha256),
                "page_idx": int(page_idx),
                "token_ids": [str(tid) for tid in span["token_ids"]],  # Safe strings
                "token_indices": [
                    int(idx) for idx in span["token_indices"]
                ],  # Safe ints
                "raw_text": str(span["raw_text"]),
                "normalized_text": str(span["normalized_text"]),
                "bucket": str(bucket),
                "token_count": int(span["token_count"]),
                "cohesion_score": float(span["cohesion_score"]),
                "proximity_score": float(proximity_score),
                "section_prior": float(section_prior),
                "total_score": float(total_score),
                # Bounding box
                "bbox_norm_x0": float(span["bbox_norm"][0]),
                "bbox_norm_y0": float(span["bbox_norm"][1]),
                "bbox_norm_x1": float(span["bbox_norm"][2]),
                "bbox_norm_y1": float(span["bbox_norm"][3]),
                "bbox_norm": span["bbox_norm"],  # Keep for processing
                # Features
                **text_features,
                **geometry_features,
                **style_features,
                "local_density": local_density,  # Will be updated
                "is_remittance_band": bool(geometry_features.get("center_y", 0) > 0.85),
            }

            page_candidates.append(candidate)

            # Add to grid for density computation
            cx = (span["bbox_norm"][0] + span["bbox_norm"][2]) / 2
            cy = (span["bbox_norm"][1] + span["bbox_norm"][3]) / 2
            page_grid.add_item(cx, cy, candidate)

        timings["scoring"] += time.time() - scoring_start

        # 5. Update local density using grid
        for candidate in page_candidates:
            candidate["local_density"] = compute_local_density_grid(
                candidate["bbox_norm"], page_grid
            )

        # 6. Proportional soft trim if too many candidates on this page
        if len(page_candidates) > 300:  # Per-page soft limit
            page_candidates.sort(key=lambda x: x["total_score"], reverse=True)
            keep_count = min(300, int(len(page_candidates) * 0.8))  # Keep top 80%
            page_candidates = page_candidates[:keep_count]

        # 7. Soft-NMS
        soft_nms_start = time.time()
        page_candidates = apply_soft_nms_grid(page_candidates, page_grid)
        timings["soft_nms"] += time.time() - soft_nms_start

        all_candidates.extend(page_candidates)

        page_time = time.time() - page_start_time
        print(
            f"  Page {page_idx}: {len(page_candidates)} candidates in {page_time:.3f}s"
        )

    if not all_candidates:
        return 0, timings

    # 8. Global deduplication (non-brittle)
    dedupe_start = time.time()
    seen_keys = set()
    deduped_candidates = []

    for candidate in all_candidates:
        # Create dedup key: normalized text + rounded bbox center
        center_x = (candidate["bbox_norm_x0"] + candidate["bbox_norm_x1"]) / 2
        center_y = (candidate["bbox_norm_y0"] + candidate["bbox_norm_y1"]) / 2

        # Round to nearest 0.05 for minor geometry tolerance
        rounded_x = round(center_x / 0.05) * 0.05
        rounded_y = round(center_y / 0.05) * 0.05

        dedup_key = f"{candidate['normalized_text']}_{rounded_x:.2f}_{rounded_y:.2f}"

        if dedup_key not in seen_keys:
            seen_keys.add(dedup_key)
            deduped_candidates.append(candidate)

    all_candidates = deduped_candidates
    timings["dedupe"] += time.time() - dedupe_start

    # 9. Diversity sampling
    diversity_start = time.time()
    all_candidates = diversity_sampling(all_candidates, max_candidates=200)
    timings["diversity"] += time.time() - diversity_start

    # Update bucket counts for reporting
    for candidate in all_candidates:
        bucket_counts[candidate["bucket"]] += 1

    # Clean up bbox_norm for storage (not needed in final output)
    for candidate in all_candidates:
        if "bbox_norm" in candidate:
            del candidate["bbox_norm"]

    # Save candidates
    if all_candidates:
        df = pd.DataFrame(all_candidates)
        df.to_parquet(candidates_path, index=False)

        timings["total"] = time.time() - start_time

        print(f"Generated {len(all_candidates)} candidates for {doc_id}")
        print(
            f"  Coverage: {coverage_stats['cue_proximal_spans']}/{coverage_stats['total_spans']} cue-proximal, "
            f"{coverage_stats['region_prior_spans']}/{coverage_stats['total_spans']} in prior regions"
        )

        for bucket, count in bucket_counts.items():
            if count > 0:
                print(f"  {bucket}: {count}")

        print(
            f"  Timing: spans={timings['spans']:.3f}s, scoring={timings['scoring']:.3f}s, "
            f"soft-nms={timings['soft_nms']:.3f}s, total={timings['total']:.3f}s"
        )

    return len(all_candidates), timings


def generate_candidates(sha256: str) -> int:
    """Legacy interface for compatibility."""
    count, _ = generate_candidates_enhanced(sha256)
    return count


def generate_all_candidates() -> dict[str, int]:
    """Generate candidates for all documents."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        print("No documents found in index")
        return {}

    results = {}
    all_timings = defaultdict(list)

    print(f"Generating candidates for {len(indexed_docs)} documents")

    for _, doc_info in tqdm(indexed_docs.iterrows(), total=len(indexed_docs)):
        sha256 = doc_info["sha256"]

        try:
            candidate_count, timings = generate_candidates_enhanced(sha256)
            results[sha256] = candidate_count

            # Collect timing statistics
            for phase, duration in timings.items():
                all_timings[phase].append(duration)

        except Exception as e:
            print(f"Failed to generate candidates for {sha256[:16]}: {e}")
            results[sha256] = 0

    # Print timing summary
    if all_timings:
        print("\nTiming Summary:")
        for phase, durations in all_timings.items():
            if durations:
                avg_duration = sum(durations) / len(durations)
                max_duration = max(durations)
                print(f"  {phase}: avg={avg_duration:.3f}s, max={max_duration:.3f}s")

    return results


def get_document_candidates(sha256: str) -> pd.DataFrame:
    """Get candidates for a specific document."""
    candidates_path = paths.get_candidates_path(sha256)

    if not candidates_path.exists():
        return pd.DataFrame()

    return pd.read_parquet(candidates_path)


def get_coverage_statistics() -> dict[str, Any]:
    """Get coverage statistics across all candidates."""
    indexed_docs = ingest.get_indexed_documents()

    if indexed_docs.empty:
        return {}

    total_stats = {
        "total_documents": len(indexed_docs),
        "documents_with_candidates": 0,
        "total_candidates": 0,
        "bucket_distribution": Counter(),
        "coverage_by_field": {},
    }

    for _, doc_info in indexed_docs.iterrows():
        sha256 = doc_info["sha256"]
        candidates_df = get_document_candidates(sha256)

        if not candidates_df.empty:
            total_stats["documents_with_candidates"] += 1
            total_stats["total_candidates"] += len(candidates_df)

            # Bucket distribution
            for bucket in candidates_df["bucket"]:
                total_stats["bucket_distribution"][bucket] += 1

    return total_stats
