"""Reviewer-graph construction, ring detection, baselines and evaluation.

Everything here uses behaviour (who reviewed what, and when) plus optional
text-similarity embeddings. Ground-truth labels are only used by the
`evaluate_*` helpers, never to build or rank clusters.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

DAY_SECONDS = 86_400.0


@dataclass(frozen=True)
class EdgeWeights:
    timing: float = 0.4
    text: float = 0.3
    overlap: float = 0.3

    def without(self, signal: str) -> EdgeWeights:
        """Drop one signal and renormalise the rest (for ablations)."""
        values = {"timing": self.timing, "text": self.text, "overlap": self.overlap}
        values[signal] = 0.0
        total = sum(values.values()) or 1.0
        return EdgeWeights(**{k: v / total for k, v in values.items()})


@dataclass
class Cluster:
    cluster_id: int
    users: list[str]
    internal_edges: int
    density: float
    mean_weight: float
    mean_timing: float
    mean_text: float
    mean_overlap: float
    products: int
    suspicion: float = 0.0
    extra: dict = field(default_factory=dict)


def _seconds(ts: pd.Series) -> np.ndarray:
    return pd.to_datetime(ts).astype("int64").to_numpy() / 1e9


def build_reviewer_graph(
    df: pd.DataFrame,
    user_embeddings: Mapping[str, np.ndarray] | None = None,
    weights: EdgeWeights = EdgeWeights(),
    edge_threshold: float = 0.15,
    max_product_reviewers: int = 60,
) -> nx.Graph:
    """Link reviewers who reviewed the same product.

    Edge weight mixes three behaviour signals:
      timing  - exp(-gap / 1 day) for the closest pair of reviews on a shared product
      text    - cosine similarity of the reviewers' mean text embeddings (0 if none given)
      overlap - Jaccard similarity of the two reviewers' product sets
    The original notebook divided shared products by the smaller product count. That
    gives a one-product fake account a perfect overlap with every genuine customer of
    the targeted product, pulling innocent reviewers into rings; Jaccard does not.
    Products with very many reviewers are skipped: they connect everyone and carry no signal.
    """
    required = {"user_id", "asin", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    data = df[["user_id", "asin", "timestamp"]].copy()
    data["t"] = _seconds(data["timestamp"])
    user_products: dict[str, set] = data.groupby("user_id")["asin"].apply(set).to_dict()
    times: dict[tuple, np.ndarray] = {
        key: np.sort(group.to_numpy()) for key, group in data.groupby(["user_id", "asin"])["t"]
    }

    norm_emb = {}
    if user_embeddings:
        for user, vec in user_embeddings.items():
            vec = np.asarray(vec, dtype=float)
            n = np.linalg.norm(vec)
            norm_emb[user] = vec / n if n else vec

    pairs: set[tuple[str, str]] = set()
    for _, reviewers in data.groupby("asin")["user_id"]:
        uniq = sorted(set(reviewers))
        if len(uniq) < 2 or len(uniq) > max_product_reviewers:
            continue
        pairs.update(itertools.combinations(uniq, 2))

    graph = nx.Graph()
    for u, v in pairs:
        shared = user_products[u] & user_products[v]
        if not shared:
            continue
        overlap = len(shared) / len(user_products[u] | user_products[v])
        timing = max(_closest_gap_score(times[(u, p)], times[(v, p)]) for p in shared)
        text = float(norm_emb[u] @ norm_emb[v]) if u in norm_emb and v in norm_emb else 0.0
        weight = weights.timing * timing + weights.text * text + weights.overlap * overlap
        if weight > edge_threshold:
            graph.add_edge(
                u, v, weight=float(weight), timing=float(timing), text=float(text),
                overlap=float(overlap), shared=len(shared),
            )
    return graph


def _closest_gap_score(a: np.ndarray, b: np.ndarray) -> float:
    gap = min(abs(x - y) for x in a for y in b)
    return math.exp(-gap / DAY_SECONDS)


def detect_communities(graph: nx.Graph, seed: int = 42, resolution: float = 1.0) -> list[set[str]]:
    if graph.number_of_edges() == 0:
        return []
    return [set(c) for c in nx.community.louvain_communities(graph, weight="weight", seed=seed, resolution=resolution)]


def score_clusters(
    graph: nx.Graph,
    communities: Iterable[set[str]],
    user_products: Mapping[str, set] | None = None,
    min_size: int = 4,
    max_size: int = 30,
) -> list[Cluster]:
    """Rank communities by an unsupervised suspicion score (no labels).

    suspicion = internal edge density x mean internal edge weight.
    Coordinated rings are small, nearly complete and strongly weighted; organic
    co-review communities are sparse and weakly weighted.
    """
    clusters = []
    for idx, members in enumerate(communities):
        if not (min_size <= len(members) <= max_size):
            continue
        sub = graph.subgraph(members)
        edges = list(sub.edges(data=True))
        n = len(members)
        possible = n * (n - 1) / 2
        density = len(edges) / possible if possible else 0.0

        def mean(attr: str, edges=edges) -> float:
            return float(np.mean([d[attr] for _, _, d in edges])) if edges else 0.0

        products = set()
        if user_products:
            for u in members:
                products |= set(user_products.get(u, ()))
        c = Cluster(
            cluster_id=idx,
            users=sorted(members),
            internal_edges=len(edges),
            density=density,
            mean_weight=mean("weight"),
            mean_timing=mean("timing"),
            mean_text=mean("text"),
            mean_overlap=mean("overlap"),
            products=len(products),
        )
        c.suspicion = c.density * c.mean_weight
        clusters.append(c)
    clusters.sort(key=lambda c: c.suspicion, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def coreview_window_baseline(df: pd.DataFrame, window_hours: float = 48.0, min_group: int = 4) -> list[set[str]]:
    """Simple rule: connect reviewers who reviewed the same product within `window_hours`.

    Returns connected components with at least `min_group` reviewers.
    """
    data = df[["user_id", "asin", "timestamp"]].copy()
    data["t"] = _seconds(data["timestamp"])
    window = window_hours * 3600.0
    graph = nx.Graph()
    for _, group in data.sort_values("t").groupby("asin"):
        users = group["user_id"].to_numpy()
        ts = group["t"].to_numpy()
        start = 0
        for end in range(len(ts)):
            while ts[end] - ts[start] > window:
                start += 1
            for k in range(start, end):
                if users[k] != users[end]:
                    graph.add_edge(users[k], users[end])
    return [set(c) for c in nx.connected_components(graph) if len(c) >= min_group]


def fraudar(edges: Iterable[tuple[str, str]], n_blocks: int = 1) -> list[dict]:
    """FRAUDAR (Hooi et al., KDD 2016) on a user-product bipartite graph.

    Greedily peels nodes to find the densest block, with column weights
    1 / log(product degree + 5) so popular products (camouflage targets) count less.
    After each block is found its users are removed and the search repeats.
    """
    remaining = set(edges)
    blocks = []
    for _ in range(n_blocks):
        if not remaining:
            break
        users, products, density = _fraudar_once(remaining)
        if not users:
            break
        blocks.append({"users": users, "products": products, "density": density})
        remaining = {(u, p) for (u, p) in remaining if u not in users}
    return blocks


def _fraudar_once(edges: set[tuple[str, str]]) -> tuple[set, set, float]:
    product_degree = Counter(p for _, p in edges)
    col_weight = {p: 1.0 / math.log(d + 5) for p, d in product_degree.items()}
    adj: dict[tuple, dict[tuple, float]] = defaultdict(dict)
    for u, p in edges:
        w = col_weight[p]
        adj[("u", u)][("p", p)] = w
        adj[("p", p)][("u", u)] = w

    priority = {node: sum(nbrs.values()) for node, nbrs in adj.items()}
    total = sum(priority.values()) / 2
    alive = set(adj)
    counter = itertools.count()
    heap = [(pr, next(counter), node) for node, pr in priority.items()]
    heapq.heapify(heap)

    best_density = total / len(alive)
    best_step = 0
    order = []
    while heap and alive:
        pr, _, node = heapq.heappop(heap)
        if node not in alive or pr != priority[node]:
            continue
        alive.remove(node)
        order.append(node)
        total -= priority[node]
        for nbr, w in adj[node].items():
            if nbr in alive:
                priority[nbr] -= w
                heapq.heappush(heap, (priority[nbr], next(counter), nbr))
        if alive:
            density = total / len(alive)
            if density > best_density:
                best_density, best_step = density, len(order)

    removed = set(order[:best_step])
    block = set(adj) - removed
    users = {name for kind, name in block if kind == "u"}
    products = {name for kind, name in block if kind == "p"}
    return users, products, float(best_density)


# ---------------------------------------------------------------------------
# Evaluation (labels used here only)
# ---------------------------------------------------------------------------


def evaluate_flagged(flagged: set[str], is_fake: Mapping[str, bool]) -> dict:
    """Reviewer-level precision / recall of a set of flagged reviewers."""
    fakes = {u for u, f in is_fake.items() if f}
    tp = len(flagged & fakes)
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / len(fakes) if fakes else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"flagged": len(flagged), "precision": precision, "recall": recall, "f1": f1}


def precision_at_k(clusters: list[Cluster], is_fake: Mapping[str, bool], ks=(10, 25, 50, 100)) -> dict:
    """Share of planted fakes among reviewers in the top-k ranked clusters."""
    out = {}
    for k in ks:
        users = {u for c in clusters[:k] for u in c.users}
        out[f"p@{k}"] = sum(bool(is_fake.get(u)) for u in users) / len(users) if users else 0.0
    return out


def flag_top_clusters(clusters: list[Cluster], min_suspicion: float) -> set[str]:
    return {u for c in clusters if c.suspicion >= min_suspicion for u in c.users}
