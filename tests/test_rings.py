from fraudscope import rings as rg
from fraudscope.config import RINGS_PATH

RINGS = [
    {"cluster_id": 1, "size": 3, "fake_ratio": 0.9, "reviews": [
        {"user_id": "AAA", "asin": "P1", "text": "great great great great", "tier": 1, "rating": 5},
        {"user_id": "BBB", "asin": "P1", "text": "Solid cable, does the job.", "tier": 0, "rating": 4},
    ]},
    {"cluster_id": 2, "size": 8, "fake_ratio": 0.4, "reviews": [
        {"user_id": "CCC", "asin": "P2", "text": "fine", "tier": 0, "rating": 3},
    ]},
    {"cluster_id": 3, "size": 5, "fake_ratio": 0.6, "reviews": []},
]


def test_risk_level_bands():
    assert rg.risk_level(0.95) == "Critical"
    assert rg.risk_level(0.8) == "Critical"
    assert rg.risk_level(0.5) == "High"
    assert rg.risk_level(0.3) == "Medium"
    assert rg.risk_level(0.1) == "Low"


def test_low_quality_detection():
    assert rg.is_low_quality_review("problems problems problems in the box")
    assert rg.is_low_quality_review("need colors many need colors many need colors many need colors")
    assert not rg.is_low_quality_review("The keyboard works well and the battery lasts a month.")
    assert not rg.is_low_quality_review("")
    assert not rg.is_low_quality_review(None)


def test_filter_by_query_matches_cluster_user_and_product():
    assert [r["cluster_id"] for r in rg.filter_and_sort(RINGS, query="bbb")] == [1]
    assert [r["cluster_id"] for r in rg.filter_and_sort(RINGS, query="p2")] == [2]
    assert [r["cluster_id"] for r in rg.filter_and_sort(RINGS, query="3")] == [3]


def test_filter_min_ratio_and_sorting():
    by_ratio = rg.filter_and_sort(RINGS, sort="Planted-fake share, high to low")
    assert [r["cluster_id"] for r in by_ratio] == [1, 3, 2]
    by_size = rg.filter_and_sort(RINGS, sort="Cluster size, large to small")
    assert [r["cluster_id"] for r in by_size] == [2, 3, 1]
    assert [r["cluster_id"] for r in rg.filter_and_sort(RINGS, min_fake_ratio=0.5, sort="Cluster ID")] == [1, 3]


def test_summary_and_overview():
    s = rg.summarize_ring(RINGS[0])
    assert s["reviews"] == 2 and s["products"] == 1 and s["artifacts"] == 1 and s["risk"] == "Critical"
    o = rg.overview(RINGS)
    assert o["rings"] == 3 and o["reviews"] == 3 and o["critical"] == 1
    assert rg.overview([])["rings"] == 0


def test_shipped_ring_file_loads():
    data = rg.load_rings(RINGS_PATH)
    assert data, "detected_rings.json should ship with the app"
    assert {"cluster_id", "fake_ratio", "reviews"} <= set(data[0])
