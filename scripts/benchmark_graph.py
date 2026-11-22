"""Benchmark on real labelled fraud graphs: YelpChi and Amazon (CARE-GNN .mat release, also shipped by DGL).

Compares, on the same stratified splits:
  features_only  - gradient boosting on node features alone
  graph_agg      - gradient boosting on node features plus mean-aggregated neighbour
                   features per relation (the tree-ensemble baseline GADBench found
                   to beat most specialised GNNs)

  python scripts/benchmark_graph.py --mat data/YelpChi.mat --name yelpchi
  python scripts/benchmark_graph.py --mat data/Amazon.mat --name amazon

Writes results/graph_benchmark_<name>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscope.config import RESULTS_DIR  # noqa: E402
from fraudscope.metrics import binary_report  # noqa: E402

RELATIONS = {
    "yelpchi": ["net_rur", "net_rtr", "net_rsr"],
    "amazon": ["net_upu", "net_usu", "net_uvu"],
}


def mean_neighbour(adj, X):
    import scipy.sparse as sp

    adj = sp.csr_matrix(adj, dtype=float)
    adj.setdiag(0)
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = sp.diags(np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0))
    return np.asarray(inv @ adj @ X)


def main() -> None:
    import scipy.io
    import scipy.sparse as sp
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True)
    ap.add_argument("--name", required=True, choices=list(RELATIONS))
    ap.add_argument("--train-ratio", type=float, default=0.4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    data = scipy.io.loadmat(args.mat)
    X = data["features"]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    y = np.asarray(data["label"]).ravel().astype(int)

    blocks = [X]
    for rel in RELATIONS[args.name] + ["homo"]:
        blocks.append(mean_neighbour(data[rel], X))
    X_graph = np.hstack(blocks)

    # In the Amazon release the first 3,305 users have too few reviews to label reliably.
    idx = np.arange(len(y)) if args.name == "yelpchi" else np.arange(3305, len(y))

    runs = {"features_only": [], "graph_agg": []}
    for seed in range(args.seeds):
        train, test = train_test_split(idx, train_size=args.train_ratio, stratify=y[idx], random_state=seed)
        for name, feats in (("features_only", X), ("graph_agg", X_graph)):
            model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=seed)
            model.fit(feats[train], y[train])
            p = model.predict_proba(feats[test])[:, 1]
            runs[name].append(binary_report(y[test], p))
            print(seed, name, {k: round(runs[name][-1][k], 4) for k in ("roc_auc", "pr_auc", "recall_at_0.01_fpr")})

    summary = {}
    for name, reports in runs.items():
        summary[name] = {
            k: {"mean": float(np.mean([r[k] for r in reports])), "std": float(np.std([r[k] for r in reports]))}
            for k in ("roc_auc", "pr_auc", "recall_at_0.01_fpr", "f1")
        }
    result = {"dataset": args.name, "nodes": int(len(idx)), "fraud_share": float(y[idx].mean()),
              "train_ratio": args.train_ratio, "seeds": args.seeds, "summary": summary}
    out = RESULTS_DIR / f"graph_benchmark_{args.name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
