"""Shared constants and paths."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RINGS_PATH = ROOT / "detected_rings.json"
RESULTS_DIR = ROOT / "results"

# Hugging Face model id. Set MODEL_PATH to a local folder to skip the download.
MODEL_ID = os.environ.get("MODEL_PATH", "Smitvkohale/review-authenticity-engine")
MAX_LENGTH = 128

# Decision bands for a single review's fake probability.
# Scores inside the uncertain band are routed to a human instead of auto-labelled.
UNCERTAIN_LOW = 0.35
UNCERTAIN_HIGH = 0.65

# Ring risk bands, applied to the share of planted fakes in a cluster (evaluation only).
CRITICAL_RISK_THRESHOLD = 0.80
HIGH_RISK_THRESHOLD = 0.50
MEDIUM_RISK_THRESHOLD = 0.25

# Heuristics for spotting degenerate paraphrases in the synthetic data.
UNIQUE_WORD_RATIO_THRESHOLD = 0.45
MIN_WORD_COUNT_FOR_RATIO_CHECK = 5
CONSECUTIVE_REPEAT_COUNT = 3

TIER_NAMES = {
    0: "Genuine",
    1: "Planted: template",
    2: "Planted: combinatorial",
    3: "Planted: paraphrased",
}

REPO_URL = "https://github.com/smitkohale/review-authenticity-engine"
MODEL_URL = "https://huggingface.co/Smitvkohale/review-authenticity-engine"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/review-authenticity-engine-training.ipynb"
