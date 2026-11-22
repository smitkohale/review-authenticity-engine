import math

import numpy as np
import pytest

from fraudscope import metrics as m


def test_roc_auc_perfect_random_and_ties():
    assert m.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert m.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert m.roc_auc([0, 1], [0.5, 0.5]) == 0.5
    assert math.isnan(m.roc_auc([1, 1], [0.2, 0.3]))


def test_roc_auc_matches_sklearn_if_available():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500) * 0.5 + y * 0.3, 2)  # rounding creates ties
    assert m.roc_auc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s))
    assert m.average_precision(y, s) == pytest.approx(sklearn_metrics.average_precision_score(y, s), abs=0.02)


def test_recall_at_fpr_respects_budget():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.9, 100), np.full(10, 0.95)])
    recall, threshold = m.recall_at_fpr(y, s, max_fpr=0.01)
    assert recall == 1.0
    assert ((s[y == 0] > threshold).mean()) <= 0.01


def test_ece_zero_for_calibrated_bins():
    y = np.array([0, 1] * 50)
    s = np.full(100, 0.5)
    assert m.expected_calibration_error(y, s) == pytest.approx(0.0)


def test_binary_report_keys_and_values():
    rep = m.binary_report([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.9])
    assert rep["precision"] == pytest.approx(2 / 3)
    assert rep["recall"] == 1.0
    assert rep["fpr"] == 0.5
    for key in ("roc_auc", "pr_auc", "ece", "brier", "recall_at_0.01_fpr"):
        assert key in rep
