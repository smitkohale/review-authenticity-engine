"""Generate fluent fake reviews with an instruction-tuned LLM, with a quality gate.

This replaces the broken Pegasus tier-3 step (its output was mostly word salad,
e.g. "group way3 playingre balance well"). Run it on a GPU (Kaggle T4 is enough
for 1.5B to 3B models). Use several generator models so one can be held out as
an unseen generator at test time.

  python scripts/generate_fakes.py --real data/fraudscope_real_reviews.parquet \
      --model Qwen/Qwen2.5-1.5B-Instruct --n 3000 --out data/fakes_qwen.parquet

Every output passes three checks before it is kept: length within the real
review range, not degenerate (no word loops), and not a near-copy of the seed.
"""

from __future__ import annotations

import argparse
import random
import sys
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fraudscope.audit import text_audit  # noqa: E402
from fraudscope.rings import is_low_quality_review  # noqa: E402

PERSONAS = [
    "a busy parent", "a college student on a budget", "a retired engineer", "a gamer",
    "a small business owner", "someone who rarely writes reviews", "a tech enthusiast", "a nurse working night shifts",
]
STYLES = [
    "enthusiastic but specific", "short and casual", "detailed, mentions one minor flaw",
    "compares it to a previous product", "mentions how long they have used it", "slightly sarcastic but positive",
]


def build_prompt(product_hint: str, rng: random.Random) -> str:
    rating = rng.choice([4, 5, 5])
    return (
        f"Write a {rating}-star Amazon customer review as {rng.choice(PERSONAS)}. "
        f"Style: {rng.choice(STYLES)}. The product: {product_hint}. "
        "Sound like a real customer, include concrete details, 2 to 6 sentences. "
        "Output only the review text."
    )


def passes(text: str, seed: str, min_words: int, max_words: int) -> bool:
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if is_low_quality_review(text):
        return False
    return SequenceMatcher(None, text.lower(), seed.lower()).ratio() < 0.6


def main() -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="parquet/CSV of real reviews with text and asin")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    real = pd.read_parquet(args.real) if args.real.endswith(".parquet") else pd.read_csv(args.real)
    lengths = real["text"].astype(str).str.split().str.len()
    min_words, max_words = int(lengths.quantile(0.05)), int(lengths.quantile(0.95))
    seeds = real.sample(n=min(len(real), args.n * 2), random_state=args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    kept, tried = [], 0
    rows = seeds.to_dict("records")
    for start in range(0, len(rows), args.batch):
        if len(kept) >= args.n:
            break
        batch = rows[start : start + args.batch]
        hints = [str(r.get("title") or r["text"])[:120] for r in batch]
        prompts = [
            tok.apply_chat_template([{"role": "user", "content": build_prompt(h, rng)}], tokenize=False,
                                    add_generation_prompt=True)
            for h in hints
        ]
        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=220, do_sample=True, temperature=0.9, top_p=0.95)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, text in zip(batch, texts, strict=False):
            tried += 1
            text = text.strip().strip('"')
            if passes(text, str(r["text"]), min_words, max_words):
                kept.append({"asin": r["asin"], "text": text, "generator": args.model})
        print(f"kept {len(kept)} / tried {tried}")

    out = pd.DataFrame(kept[: args.n])
    out["tier"] = 3
    out["label"] = 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(text_audit(out))
    print(f"wrote {len(out)} reviews to {args.out} (acceptance {len(kept) / max(1, tried):.0%})")


if __name__ == "__main__":
    main()
