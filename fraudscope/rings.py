"""Loading, summarising and filtering the exported reviewer clusters."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median

from .config import (
    CONSECUTIVE_REPEAT_COUNT,
    CRITICAL_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    MEDIUM_RISK_THRESHOLD,
    MIN_WORD_COUNT_FOR_RATIO_CHECK,
    RINGS_PATH,
    TIER_NAMES,
    UNIQUE_WORD_RATIO_THRESHOLD,
)

SORT_OPTIONS = {
    "Planted-fake share, high to low": ("fake_ratio", True),
    "Cluster size, large to small": ("size", True),
    "Cluster ID": ("cluster_id", False),
}

_REPEAT_RE = re.compile(r"\b(\w+)(?:\s+\1){%d,}\b" % (CONSECUTIVE_REPEAT_COUNT - 1))
_WORD_RE = re.compile(r"\b\w+\b")


def load_rings(path: Path = RINGS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def risk_level(fake_ratio: float) -> str:
    if fake_ratio >= CRITICAL_RISK_THRESHOLD:
        return "Critical"
    if fake_ratio >= HIGH_RISK_THRESHOLD:
        return "High"
    if fake_ratio >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


def is_low_quality_review(text: str) -> bool:
    """Flag degenerate synthetic paraphrases (word loops, very low lexical variety)."""
    if not text or not isinstance(text, str):
        return False
    lowered = text.lower()
    if _REPEAT_RE.search(lowered):
        return True
    words = _WORD_RE.findall(lowered)
    if len(words) >= MIN_WORD_COUNT_FOR_RATIO_CHECK:
        return len(set(words)) / len(words) < UNIQUE_WORD_RATIO_THRESHOLD
    return False


def tier_name(tier) -> str:
    try:
        return TIER_NAMES.get(int(tier), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def summarize_ring(ring: dict) -> dict:
    reviews = ring.get("reviews", [])
    reviewers = {r.get("user_id") for r in reviews}
    products = {r.get("asin") for r in reviews if r.get("asin")}
    fake_ratio = float(ring.get("fake_ratio", 0.0))
    return {
        "cluster_id": ring.get("cluster_id"),
        "size": int(ring.get("size", len(reviewers))),
        "reviews": len(reviews),
        "products": len(products),
        "fake_ratio": fake_ratio,
        "risk": risk_level(fake_ratio),
        "artifacts": sum(is_low_quality_review(r.get("text", "")) for r in reviews),
    }


def overview(rings: list[dict]) -> dict:
    if not rings:
        return {"rings": 0, "reviews": 0, "median_fake_ratio": 0.0, "critical": 0}
    ratios = [float(r.get("fake_ratio", 0.0)) for r in rings]
    return {
        "rings": len(rings),
        "reviews": sum(len(r.get("reviews", [])) for r in rings),
        "median_fake_ratio": median(ratios),
        "critical": sum(x >= CRITICAL_RISK_THRESHOLD for x in ratios),
    }


def ring_matches(ring: dict, query: str) -> bool:
    query = query.strip().lower()
    if not query:
        return True
    if query in str(ring.get("cluster_id", "")).lower():
        return True
    for review in ring.get("reviews", []):
        if query in str(review.get("user_id", "")).lower() or query in str(review.get("asin", "")).lower():
            return True
    return False


def filter_and_sort(rings: list[dict], query: str = "", min_fake_ratio: float = 0.0, sort: str = "") -> list[dict]:
    kept = [r for r in rings if float(r.get("fake_ratio", 0.0)) >= min_fake_ratio and ring_matches(r, query)]
    key, reverse = SORT_OPTIONS.get(sort, SORT_OPTIONS[next(iter(SORT_OPTIONS))])
    if key == "size":
        return sorted(kept, key=lambda r: int(r.get("size", len(r.get("reviews", [])))), reverse=reverse)
    return sorted(kept, key=lambda r: r.get(key, 0), reverse=reverse)
