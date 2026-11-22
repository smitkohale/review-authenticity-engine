"""Data-quality checks on labelled review text."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from .rings import is_low_quality_review


def text_audit(df: pd.DataFrame, group_col: str = "tier") -> list[dict]:
    """Per-group length, duplication and degenerate-text statistics."""
    out = []
    for group, part in df.groupby(group_col):
        texts = part["text"].astype(str).str.strip()
        words = texts.str.split().str.len()
        first = Counter(t.split()[0].lower() for t in texts if t)
        top_word, top_count = first.most_common(1)[0] if first else ("", 0)
        out.append({
            group_col: int(group) if str(group).isdigit() else group,
            "n": int(len(part)),
            "unique_share": float(texts.nunique() / len(part)) if len(part) else 0.0,
            "mean_words": float(words.mean()) if len(part) else 0.0,
            "median_words": float(words.median()) if len(part) else 0.0,
            "degenerate_share": float(texts.map(is_low_quality_review).mean()) if len(part) else 0.0,
            "top_first_word": top_word,
            "top_first_word_share": top_count / len(part) if len(part) else 0.0,
        })
    return out
