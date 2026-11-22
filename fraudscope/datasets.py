"""Loaders that turn evaluation datasets into a common frame: text, label (1 = fake), source.

Supported:
  rings     - the reviews inside detected_rings.json (in-distribution, mostly seen in training)
  ott       - Ott et al. Deceptive Opinion Spam corpus (op_spam_v1.4 folder, or the Kaggle CSV)
  salminen  - Salminen et al. 2022 GPT-2 fake reviews CSV (label CG = generated, OR = original)
  csv       - any CSV with a text column and a label column
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RINGS_PATH
from .rings import load_rings


def load_ring_reviews(path: Path = RINGS_PATH) -> pd.DataFrame:
    rows = []
    for ring in load_rings(path):
        for r in ring.get("reviews", []):
            tier = int(r.get("tier", 0) or 0)
            rows.append({
                "text": str(r.get("text", "")).strip(),
                "label": int(tier > 0),
                "tier": tier,
                "user_id": r.get("user_id"),
                "asin": r.get("asin"),
                "cluster_id": ring.get("cluster_id"),
            })
    df = pd.DataFrame(rows)
    df["source"] = "rings"
    return df


def load_ott(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.is_file() and path.suffix == ".csv":
        df = pd.read_csv(path)
        out = pd.DataFrame({
            "text": df["text"].astype(str),
            "label": (df["deceptive"].astype(str).str.lower() == "deceptive").astype(int),
            "polarity": df.get("polarity", pd.Series(["unknown"] * len(df))).astype(str),
        })
    else:
        rows = []
        for txt in path.rglob("*.txt"):
            parts = [p.lower() for p in txt.parts]
            if not any("deceptive" in p or "truthful" in p for p in parts):
                continue
            label = int(any(p.startswith("deceptive") for p in parts))
            polarity = "positive" if any("positive" in p for p in parts) else "negative"
            rows.append({"text": txt.read_text(encoding="utf-8", errors="ignore").strip(),
                         "label": label, "polarity": polarity})
        out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"no Ott reviews found under {path}")
    out["source"] = "ott"
    return out


def load_salminen(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "text_" if "text_" in df.columns else "text"
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": (df["label"].astype(str).str.upper() == "CG").astype(int),
    })
    if "category" in df.columns:
        out["category"] = df["category"]
    out["source"] = "salminen"
    return out


def load_csv(path: str | Path, text_col: str, label_col: str, positive: str = "1") -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": (df[label_col].astype(str) == str(positive)).astype(int),
    })
    out["source"] = Path(path).stem
    return out


def load(name: str, path: str | None = None, **kwargs) -> pd.DataFrame:
    if name == "rings":
        return load_ring_reviews(Path(path) if path else RINGS_PATH)
    if path is None:
        raise ValueError(f"dataset '{name}' needs --path")
    if name == "ott":
        return load_ott(path)
    if name == "salminen":
        return load_salminen(path)
    if name == "csv":
        return load_csv(path, **kwargs)
    raise ValueError(f"unknown dataset '{name}'")
