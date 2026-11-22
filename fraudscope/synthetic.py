"""Planting synthetic review rings into real review data.

Extends the original notebook generator with two harder attacker behaviours:

* evasion    - ring members post text copied from genuine reviews, so a text
               classifier has nothing to find; only behaviour gives them away.
* camouflage - ring members also review popular, legitimate products to look
               like normal customers (the attack FRAUDAR was designed for).

Fake reviewer IDs are random hex strings, not a recognisable prefix, so nothing
downstream can key on the ID. Labels live in `label` and `tier` only.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

TIER1_OPENERS = ["Great", "Amazing", "Excellent", "Perfect", "Best", "Awesome", "Fantastic", "Incredible"]
TIER1_NOUNS = ["product", "item", "purchase", "buy", "deal"]
TIER1_CLOSERS = [
    "highly recommend buy it now", "will buy again for sure", "very satisfied five stars",
    "best purchase ever made", "works great no complaints", "totally worth the money",
    "five stars all the way", "everyone should get this",
]
TIER2_ADJECTIVES = ["amazing", "fantastic", "incredible", "superb", "outstanding", "wonderful", "impressive"]
TIER2_NOUNS = ["product", "item", "purchase", "gadget", "device"]
TIER2_REASONS = ["fast shipping", "great build quality", "perfect fit", "excellent value", "works flawlessly"]

EVASION_TIER = 4
TIER_LABELS = {1: "template", 2: "combinatorial", 3: "paraphrased", EVASION_TIER: "evasion (copied genuine text)"}


@dataclass
class RingConfig:
    tier: int
    n_reviewers: int
    camouflage_per_member: int = 0  # extra legit-looking reviews per ring member
    spread_hours: float | None = None  # override how widely reviews are spread in time


def fake_user_id(rng: random.Random) -> str:
    return "A" + uuid.UUID(int=rng.getrandbits(128)).hex[:20].upper()


def tier1_text(rng: random.Random) -> str:
    return f"{rng.choice(TIER1_OPENERS)} {rng.choice(TIER1_NOUNS)}, {rng.choice(TIER1_CLOSERS)}"


def tier2_text(rng: random.Random) -> str:
    a, b = rng.sample(TIER2_REASONS, 2)
    return (
        f"This is a {rng.choice(TIER2_ADJECTIVES)} {rng.choice(TIER2_NOUNS)}. "
        f"{a.capitalize()} and {b}. Would definitely recommend to anyone looking for this."
    )


def plant_ring(
    asin: str,
    config: RingConfig,
    base_time: pd.Timestamp,
    rng: random.Random,
    paraphrase_pool: list[str] | None = None,
    genuine_pool: list[str] | None = None,
    popular_asins: list[str] | None = None,
) -> list[dict]:
    rows = []
    members = [fake_user_id(rng) for _ in range(config.n_reviewers)]
    for idx, uid in enumerate(members):
        tier = config.tier
        if tier == 1:
            text, rating = tier1_text(rng), 5.0
            ts = base_time + timedelta(minutes=rng.randint(0, 30))
        elif tier == 2:
            text, rating = tier2_text(rng), rng.choice([4.0, 5.0])
            ts = base_time + timedelta(hours=rng.randint(0, 72))
        elif tier == 3:
            if not paraphrase_pool:
                raise ValueError("tier 3 needs a paraphrase pool")
            text, rating = rng.choice(paraphrase_pool), rng.choice([4.0, 4.0, 5.0])
            ts = base_time + timedelta(hours=rng.randint(6, 72) * (idx + 1) / config.n_reviewers)
        elif tier == EVASION_TIER:
            if not genuine_pool:
                raise ValueError("evasion tier needs a pool of genuine review texts")
            text, rating = rng.choice(genuine_pool), rng.choice([4.0, 5.0, 5.0])
            ts = base_time + timedelta(hours=rng.uniform(0, 96))
        else:
            raise ValueError(f"unknown tier {tier}")
        if config.spread_hours is not None:
            ts = base_time + timedelta(hours=rng.uniform(0, config.spread_hours))
        rows.append(_row(uid, asin, text, rating, ts, label=1, tier=tier))

        for _ in range(config.camouflage_per_member):
            if not popular_asins or not genuine_pool:
                raise ValueError("camouflage needs popular_asins and genuine_pool")
            camo_ts = base_time + timedelta(days=rng.uniform(-60, 60))
            rows.append(
                _row(uid, rng.choice(popular_asins), rng.choice(genuine_pool), rng.choice([4.0, 5.0]),
                     camo_ts, label=1, tier=tier, camouflage=True)
            )
    return rows


def _row(uid, asin, text, rating, ts, label, tier, camouflage=False) -> dict:
    return {
        "user_id": uid, "asin": asin, "text": text, "rating": float(rating), "timestamp": ts,
        "label": label, "tier": tier, "camouflage": camouflage,
    }


def plant_rings(
    real: pd.DataFrame,
    configs: list[RingConfig],
    seed: int = 42,
    paraphrase_pool: list[str] | None = None,
    start: str = "2022-06-01",
    span_days: int = 400,
) -> pd.DataFrame:
    """Return real reviews (label 0) plus planted rings, shuffled."""
    rng = random.Random(seed)
    targets = real["asin"].drop_duplicates().sample(n=len(configs), random_state=seed).tolist()
    genuine_pool = real["text"].dropna().astype(str).tolist()
    popular = real["asin"].value_counts().head(50).index.tolist()
    base = pd.Timestamp(start)

    rows = []
    for asin, cfg in zip(targets, configs, strict=False):
        when = base + timedelta(days=rng.randint(0, span_days))
        rows.extend(plant_ring(asin, cfg, when, rng, paraphrase_pool, genuine_pool, popular))

    real_part = real.copy()
    real_part["label"], real_part["tier"], real_part["camouflage"] = 0, 0, False
    out = pd.concat([real_part, pd.DataFrame(rows)], ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_toy_reviews(
    n_users: int = 300, n_products: int = 120, reviews_per_user: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Small organic-looking review table for tests and demos (no real data needed)."""
    rng = random.Random(seed)
    words = "battery screen sound cable fits works broke cheap solid fast slow bright quiet heavy light".split()
    start = pd.Timestamp("2021-01-01")
    rows = []
    for i in range(n_users):
        uid = f"U{i:05d}"
        for asin in rng.sample(range(n_products), reviews_per_user):
            text = " ".join(rng.choice(words) for _ in range(rng.randint(8, 25)))
            rows.append({
                "user_id": uid, "asin": f"P{asin:04d}", "text": text, "rating": float(rng.randint(1, 5)),
                "timestamp": start + timedelta(days=rng.uniform(0, 700)),
            })
    return pd.DataFrame(rows)
