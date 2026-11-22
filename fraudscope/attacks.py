"""Cheap text perturbations for robustness testing (in the spirit of RAID's attacks).

Each attack keeps the review's meaning for a human reader while changing its
surface form. Recall under attack shows how much a detector relies on exact
surface patterns.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable

# Latin letters mapped to visually identical Cyrillic code points.
HOMOGLYPHS = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "i": "і"}

SYNONYMS = {
    "great": "good", "amazing": "really nice", "excellent": "very good", "perfect": "just right",
    "awesome": "nice", "fantastic": "very nice", "incredible": "surprisingly good", "best": "top",
    "product": "thing", "item": "unit", "purchase": "order", "recommend": "suggest",
    "highly": "really", "works": "functions", "love": "like", "fast": "quick",
    "definitely": "certainly", "money": "price", "quality": "build", "happy": "pleased",
}

FLAW_SENTENCES = [
    "The only downside is that the cable is a little short.",
    "Packaging was a bit beat up when it arrived, but no damage.",
    "Took me a few minutes to figure out the setup.",
    "I knocked off a star because the manual is not great.",
    "It runs slightly warm after an hour of use.",
]

_WORD = re.compile(r"[A-Za-z]+")


def typos(text: str, rng: random.Random, rate: float = 0.08) -> str:
    def mangle(m: re.Match) -> str:
        w = m.group(0)
        if len(w) < 4 or rng.random() > rate:
            return w
        i = rng.randrange(1, len(w) - 2)
        if rng.random() < 0.5:
            return w[:i] + w[i + 1] + w[i] + w[i + 2 :]  # swap neighbours
        return w[:i] + w[i + 1 :]  # drop a letter

    return _WORD.sub(mangle, text)


def homoglyphs(text: str, rng: random.Random, rate: float = 0.15) -> str:
    return "".join(HOMOGLYPHS[ch] if ch in HOMOGLYPHS and rng.random() < rate else ch for ch in text)


def synonym_swap(text: str, rng: random.Random) -> str:
    def swap(m: re.Match) -> str:
        w = m.group(0)
        repl = SYNONYMS.get(w.lower())
        if not repl:
            return w
        return repl.capitalize() if w[0].isupper() else repl

    return _WORD.sub(swap, text)


def add_flaw(text: str, rng: random.Random) -> str:
    return f"{text.rstrip()} {rng.choice(FLAW_SENTENCES)}"


def lowercase_no_punct(text: str, rng: random.Random) -> str:
    return re.sub(r"[^\w\s]", "", text.lower())


def word_dropout(text: str, rng: random.Random, rate: float = 0.15) -> str:
    words = text.split()
    kept = [w for w in words if rng.random() > rate]
    return " ".join(kept or words)


ATTACKS: dict[str, Callable[[str, random.Random], str]] = {
    "typos": typos,
    "homoglyphs": homoglyphs,
    "synonym_swap": synonym_swap,
    "add_flaw_sentence": add_flaw,
    "lowercase_no_punct": lowercase_no_punct,
    "word_dropout": word_dropout,
}


def apply_attack(name: str, texts: list[str], seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    fn = ATTACKS[name]
    return [fn(t, rng) for t in texts]
