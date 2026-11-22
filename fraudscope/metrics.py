"""Classification metrics suited to fraud work, in plain numpy.

Accuracy hides what matters here. At platform scale a 1% false-positive rate
flags a huge number of genuine reviews, so we report recall at a fixed low FPR,
ranking quality (ROC-AUC, PR-AUC) and calibration (ECE, Brier).
"""

from __future__ import annotations

import numpy as np


def _arrays(y_true, scores):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape")
    return y, s


def roc_auc(y_true, scores) -> float:
    y, s = _arrays(y_true, scores)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    _, first, counts = np.unique(allv[order], return_index=True, return_counts=True)
    ranks = np.empty(len(allv))
    ranks[order] = np.repeat(first + (counts + 1) / 2, counts)  # 1-based ranks, ties averaged
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(y_true, scores) -> float:
    y, s = _arrays(y_true, scores)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    precision = tp / np.arange(1, len(y_sorted) + 1)
    return float((precision * y_sorted).sum() / y.sum())


def recall_at_fpr(y_true, scores, max_fpr: float = 0.01) -> tuple[float, float]:
    """Best recall achievable while keeping FPR <= max_fpr. Returns (recall, threshold)."""
    y, s = _arrays(y_true, scores)
    neg = np.sort(s[y == 0])[::-1]
    pos = s[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan"), float("nan")
    allowed = int(np.floor(max_fpr * len(neg)))
    # threshold must sit strictly above the (allowed+1)-th highest negative score
    threshold = neg[allowed] if allowed < len(neg) else -np.inf
    return float((pos > threshold).mean()), float(threshold)


def expected_calibration_error(y_true, scores, bins: int = 10) -> float:
    y, s = _arrays(y_true, scores)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - s[mask].mean())
    return float(ece)


def reliability_curve(y_true, scores, bins: int = 10) -> list[dict]:
    y, s = _arrays(y_true, scores)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            out.append({
                "bin": b,
                "mean_score": float(s[mask].mean()),
                "fraction_fake": float(y[mask].mean()),
                "n": int(mask.sum()),
            })
    return out


def binary_report(y_true, scores, threshold: float = 0.5, max_fpr: float = 0.01) -> dict:
    y, s = _arrays(y_true, scores)
    pred = s >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    r_at_fpr, thr = recall_at_fpr(y, s, max_fpr)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "accuracy": (tp + tn) / len(y) if len(y) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "roc_auc": roc_auc(y, s),
        "pr_auc": average_precision(y, s),
        f"recall_at_{max_fpr:g}_fpr": r_at_fpr,
        "threshold_at_fpr": thr,
        "ece": expected_calibration_error(y, s),
        "brier": float(np.mean((s - y) ** 2)),
    }
