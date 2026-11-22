"""Build the reviewer graph, rank clusters without labels, evaluate against labels, export rings.

Input: a labelled parquet/CSV with columns user_id, asin, timestamp, text, label (1 = planted), tier.
The original training run's file is fraudscope_full_labeled_v2.parquet.

  python scripts/build_graph.py --input data/fraudscope_full_labeled_v2.parquet
  python scripts/build_graph.py --input data/full.parquet --no-text --ablate
  python scripts/build_graph.py --demo        # toy data with planted, evasive and camouflaged rings

What it reports (results/graph_eval_<name>.json):
  * ranking quality of the unsupervised suspicion score: precision@k clusters
  * reviewer-level precision / recall when flagging the top clusters
  * baselines: same-product-within-48h rule and FRAUDAR dense blocks
  * optional ablation: drop each edge signal (timing / text / overlap) in turn
  * recall per tier, including the evasion tier the text model cannot see
Exported rings are the top clusters by suspicion score, not filtered by labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscope import graph as gr  # noqa: E402
from fraudscope import synthetic as syn  # noqa: E402
from fraudscope.config import RESULTS_DIR, RINGS_PATH  # noqa: E402


def load_input(args) -> tuple[pd.DataFrame, str]:
    if args.demo:
        real = syn.make_toy_reviews(n_users=600, n_products=200, seed=args.seed)
        configs = (
            [syn.RingConfig(tier=1, n_reviewers=10) for _ in range(6)]
            + [syn.RingConfig(tier=2, n_reviewers=12) for _ in range(6)]
            + [syn.RingConfig(tier=syn.EVASION_TIER, n_reviewers=12) for _ in range(6)]
            + [syn.RingConfig(tier=syn.EVASION_TIER, n_reviewers=12, camouflage_per_member=2) for _ in range(6)]
        )
        return syn.plant_rings(real, configs, seed=args.seed), "demo"
    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    return df, path.stem


def user_embeddings(df: pd.DataFrame, model_name: str) -> dict:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(df["text"].astype(str).tolist(), batch_size=128, show_progress_bar=True)
    frame = pd.DataFrame({"user_id": df["user_id"].to_numpy(), "i": np.arange(len(df))})
    return {u: emb[idx["i"].to_numpy()].mean(axis=0) for u, idx in frame.groupby("user_id")}


def run(df, embeddings, weights, args) -> dict:
    is_fake = df.groupby("user_id")["label"].max().astype(bool).to_dict()
    user_tier = df.groupby("user_id")["tier"].max().to_dict()
    user_products = df.groupby("user_id")["asin"].apply(set).to_dict()

    g = gr.build_reviewer_graph(df, embeddings, weights=weights, edge_threshold=args.edge_threshold)
    communities = gr.detect_communities(g, seed=args.seed)
    clusters = gr.score_clusters(g, communities, user_products, args.min_size, args.max_size)

    top = clusters[: args.top_k]
    flagged = {u for c in top for u in c.users}
    tiers = sorted(t for t in df["tier"].unique() if t != 0)
    recall_by_tier = {}
    for t in tiers:
        members = {u for u, tt in user_tier.items() if tt == t}
        recall_by_tier[str(t)] = len(flagged & members) / len(members) if members else None

    return {
        "graph": {"nodes": g.number_of_nodes(), "edges": g.number_of_edges(), "communities": len(communities),
                  "candidate_clusters": len(clusters)},
        "precision_at_k": gr.precision_at_k(clusters, is_fake),
        f"flag_top_{args.top_k}": gr.evaluate_flagged(flagged, is_fake),
        "recall_by_tier": recall_by_tier,
        "_clusters": clusters,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-text", action="store_true", help="skip sentence embeddings (behaviour only)")
    ap.add_argument("--embedder", default="all-MiniLM-L6-v2")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--edge-threshold", type=float, default=0.15)
    ap.add_argument("--min-size", type=int, default=4)
    ap.add_argument("--max-size", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=150)
    ap.add_argument("--fraudar-blocks", type=int, default=20)
    ap.add_argument("--export", type=int, default=150, help="clusters to export to detected_rings.json (0 = none)")
    ap.add_argument("--export-path", default=str(RINGS_PATH))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.demo and not args.input:
        ap.error("--input or --demo is required")

    df, name = load_input(args)
    df = df.dropna(subset=["user_id", "asin", "timestamp"]).reset_index(drop=True)
    is_fake = df.groupby("user_id")["label"].max().astype(bool).to_dict()
    embeddings = None if (args.no_text or args.demo) else user_embeddings(df, args.embedder)

    result = {"input": name, "reviews": int(len(df)), "reviewers": len(is_fake),
              "planted_reviewers": int(sum(is_fake.values())), "text_signal": embeddings is not None}

    main_run = run(df, embeddings, gr.EdgeWeights(), args)
    clusters = main_run.pop("_clusters")
    result["louvain"] = main_run

    window = gr.coreview_window_baseline(df, window_hours=48)
    result["baseline_48h_window"] = gr.evaluate_flagged({u for c in window for u in c}, is_fake)

    edges = set(zip(df["user_id"], df["asin"], strict=False))
    blocks = gr.fraudar(edges, n_blocks=args.fraudar_blocks)
    result["baseline_fraudar"] = gr.evaluate_flagged({u for b in blocks for u in b["users"]}, is_fake)

    if args.ablate:
        result["ablation"] = {}
        for signal in ("timing", "text", "overlap"):
            if signal == "text" and embeddings is None:
                continue
            r = run(df, embeddings, gr.EdgeWeights().without(signal), args)
            r.pop("_clusters")
            result["ablation"][f"without_{signal}"] = r

    # Demo numbers come from toy data and are only a smoke test, so keep them out of results/.
    out = (RESULTS_DIR / "demo" if args.demo else RESULTS_DIR) / f"graph_eval_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))
    print(f"wrote {out}")

    if args.export and not args.demo:
        export = []
        for c in clusters[: args.export]:
            members = set(c.users)
            reviews = df[df["user_id"].isin(members)]
            export.append({
                "cluster_id": int(c.cluster_id),
                "size": len(c.users),
                "suspicion": round(c.suspicion, 4),
                "fake_ratio": float(np.mean([is_fake[u] for u in c.users])),
                "reviews": reviews[["user_id", "asin", "text", "tier", "rating"]].head(40).to_dict("records"),
            })
        Path(args.export_path).write_text(json.dumps(export, indent=2, default=str), encoding="utf-8")
        print(f"exported {len(export)} clusters to {args.export_path}")


if __name__ == "__main__":
    main()
