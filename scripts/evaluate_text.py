"""Evaluate the text classifier on a dataset, with and without perturbation attacks.

Examples
  python scripts/evaluate_text.py --dataset rings
  python scripts/evaluate_text.py --dataset ott --path data/op_spam_v1.4
  python scripts/evaluate_text.py --dataset salminen --path "data/fake reviews dataset.csv" --sample 4000

Writes results/text_eval_<dataset>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscope import datasets  # noqa: E402
from fraudscope.attacks import ATTACKS, apply_attack  # noqa: E402
from fraudscope.config import MODEL_ID, RESULTS_DIR  # noqa: E402
from fraudscope.metrics import binary_report, reliability_curve  # noqa: E402
from fraudscope.model import load_model, make_predict_fn  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["rings", "ott", "salminen", "csv"])
    ap.add_argument("--path")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive", default="1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (0 = all)")
    ap.add_argument("--no-attacks", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()

    kwargs = {}
    if args.dataset == "csv":
        kwargs = {"text_col": args.text_col, "label_col": args.label_col, "positive": args.positive}
    df = datasets.load(args.dataset, args.path, **kwargs)
    df = df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
    if args.sample and len(df) > args.sample:
        frac = args.sample / len(df)
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=args.seed))
        df = df.reset_index(drop=True)

    tokenizer, model = load_model(args.model)
    predict = make_predict_fn(tokenizer, model, batch_size=32)

    t0 = time.time()
    texts = df["text"].tolist()
    df["p_fake"] = predict(texts)[:, 1]
    secs = time.time() - t0

    result = {
        "dataset": args.dataset,
        "path": args.path,
        "model": args.model,
        "n": int(len(df)),
        "seconds_per_review": secs / max(1, len(df)),
        "overall": binary_report(df["label"], df["p_fake"]),
        "reliability": reliability_curve(df["label"], df["p_fake"]),
    }

    for col in ("tier", "polarity", "category"):
        if col in df.columns:
            groups = {}
            for value, part in df.groupby(col):
                groups[str(value)] = {
                    "n": int(len(part)),
                    "label": int(part["label"].iloc[0]) if part["label"].nunique() == 1 else None,
                    "share_flagged": float((part["p_fake"] >= 0.5).mean()),
                    "mean_p_fake": float(part["p_fake"].mean()),
                }
            result[f"by_{col}"] = groups

    if not args.no_attacks:
        fakes = df[df["label"] == 1]
        reals = df[df["label"] == 0]
        attacks = {"none": {
            "recall": float((fakes["p_fake"] >= 0.5).mean()) if len(fakes) else None,
            "fpr": float((reals["p_fake"] >= 0.5).mean()) if len(reals) else None,
        }}
        for name in ATTACKS:
            row = {}
            if len(fakes):
                p = predict(apply_attack(name, fakes["text"].tolist(), seed=args.seed))[:, 1]
                row["recall"] = float((p >= 0.5).mean())
                if "tier" in fakes.columns:
                    row["recall_by_tier"] = {
                        str(t): float((p[(fakes["tier"] == t).to_numpy()] >= 0.5).mean())
                        for t in sorted(fakes["tier"].unique())
                    }
            if len(reals):
                p = predict(apply_attack(name, reals["text"].tolist(), seed=args.seed))[:, 1]
                row["fpr"] = float((p >= 0.5).mean())
            attacks[name] = row
            print(f"{name:20s} {row}")
        result["attacks"] = attacks

    out = Path(args.out) if args.out else RESULTS_DIR / f"text_eval_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["overall"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
