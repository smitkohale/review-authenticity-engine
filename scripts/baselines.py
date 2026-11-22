"""Simple baselines and shortcut checks for the text task.

* surface-only : logistic regression on length / punctuation / casing features only.
                 If this scores well, the dataset has a shortcut the classifier can use
                 without understanding anything (like hypothesis-only baselines in NLI).
* tfidf_lr     : TF-IDF word and character n-grams with logistic regression.

Cross-validation groups rows by exact text so no review text is in both train and test.

  python scripts/baselines.py --dataset rings
  python scripts/baselines.py --dataset csv --path data/train.csv --text-col text --label-col label
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscope import datasets  # noqa: E402
from fraudscope.config import RESULTS_DIR  # noqa: E402
from fraudscope.metrics import binary_report  # noqa: E402


def surface_features(texts: pd.Series) -> np.ndarray:
    feats = []
    for t in texts.astype(str):
        words = t.split()
        n_chars = max(1, len(t))
        n_words = max(1, len(words))
        feats.append([
            np.log1p(len(words)),
            np.log1p(len(t)),
            sum(len(w) for w in words) / n_words,
            len(re.findall(r"[^\w\s]", t)) / n_chars,
            sum(c.isupper() for c in t) / n_chars,
            sum(c.isdigit() for c in t) / n_chars,
            len({w.lower() for w in words}) / n_words,
            t.count("!") / n_chars,
        ])
    return np.asarray(feats)


def main() -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline, make_union
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["rings", "ott", "salminen", "csv"])
    ap.add_argument("--path")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive", default="1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out")
    args = ap.parse_args()

    kwargs = {}
    if args.dataset == "csv":
        kwargs = {"text_col": args.text_col, "label_col": args.label_col, "positive": args.positive}
    df = datasets.load(args.dataset, args.path, **kwargs)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    y = df["label"].to_numpy()
    groups = pd.factorize(df["text"])[0]

    models = {
        "surface_only": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "tfidf_lr": lambda: make_pipeline(
            make_union(
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
            ),
            LogisticRegression(max_iter=2000, C=4.0),
        ),
    }
    inputs = {"surface_only": surface_features(df["text"]), "tfidf_lr": df["text"].to_numpy()}

    result = {"dataset": args.dataset, "path": args.path, "n": int(len(df)), "folds": args.folds, "models": {}}
    for name, build in models.items():
        X = inputs[name]
        oof = np.zeros(len(df))
        for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(X, y, groups):
            m = build()
            m.fit(X[train_idx], y[train_idx])
            oof[test_idx] = m.predict_proba(X[test_idx])[:, 1]
        report = binary_report(y, oof)
        if "tier" in df.columns:
            report["recall_by_tier"] = {
                str(t): float((oof[(df["tier"] == t).to_numpy()] >= 0.5).mean())
                for t in sorted(df["tier"].unique()) if t != 0
            }
        result["models"][name] = report
        print(name, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in report.items()}))

    out = Path(args.out) if args.out else RESULTS_DIR / f"baselines_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
