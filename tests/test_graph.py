import pandas as pd

from fraudscope import graph as gr
from fraudscope import synthetic as syn


def planted_dataset(seed=1):
    real = syn.make_toy_reviews(n_users=400, n_products=300, reviews_per_user=3, seed=seed)
    configs = [syn.RingConfig(tier=1, n_reviewers=10) for _ in range(5)] + [
        syn.RingConfig(tier=syn.EVASION_TIER, n_reviewers=10, camouflage_per_member=1) for _ in range(5)
    ]
    return syn.plant_rings(real, configs, seed=seed)


def test_planted_ids_do_not_leak_labels():
    df = planted_dataset()
    fake_ids = df.loc[df["label"] == 1, "user_id"]
    assert not fake_ids.str.startswith("FAKE").any()
    assert fake_ids.str.len().eq(fake_ids.str.len().iloc[0]).all()


def test_evasion_tier_reuses_genuine_text():
    df = planted_dataset()
    genuine = set(df.loc[df["label"] == 0, "text"])
    evasive = df[(df["tier"] == syn.EVASION_TIER) & ~df["camouflage"]]
    assert evasive["text"].isin(genuine).all()


def test_graph_ranks_planted_rings_first_without_labels():
    df = planted_dataset()
    is_fake = df.groupby("user_id")["label"].max().astype(bool).to_dict()
    user_products = df.groupby("user_id")["asin"].apply(set).to_dict()
    g = gr.build_reviewer_graph(df)
    clusters = gr.score_clusters(g, gr.detect_communities(g), user_products)
    assert clusters, "expected candidate clusters"
    assert gr.precision_at_k(clusters, is_fake, ks=(5,))["p@5"] > 0.8


def test_edge_weights_ablation_renormalises():
    w = gr.EdgeWeights().without("text")
    assert w.text == 0
    assert abs(w.timing + w.overlap - 1.0) < 1e-9


def test_coreview_window_baseline_groups_bursts():
    t0 = pd.Timestamp("2022-01-01")
    df = pd.DataFrame({
        "user_id": ["a", "b", "c", "d", "e"],
        "asin": ["P"] * 5,
        "timestamp": [t0, t0 + pd.Timedelta(hours=1), t0 + pd.Timedelta(hours=2), t0 + pd.Timedelta(hours=3),
                      t0 + pd.Timedelta(days=30)],
    })
    groups = gr.coreview_window_baseline(df, window_hours=48, min_group=4)
    assert groups == [{"a", "b", "c", "d"}]


def test_fraudar_finds_dense_block():
    edges = {(f"f{i}", f"p{j}") for i in range(8) for j in range(4)}  # dense block
    edges |= {(f"n{i}", f"q{i % 50}") for i in range(200)}  # sparse background
    edges |= {(f"f{i}", "popular") for i in range(8)} | {(f"n{i}", "popular") for i in range(100)}  # camouflage
    block = gr.fraudar(edges, n_blocks=1)[0]
    found = {u for u in block["users"] if u.startswith("f")}
    assert len(found) >= 7
    assert len(block["users"] - found) <= 2


def test_evaluate_flagged():
    res = gr.evaluate_flagged({"a", "b", "x"}, {"a": True, "b": True, "c": True, "x": False})
    assert res["precision"] == 2 / 3 and res["recall"] == 2 / 3
