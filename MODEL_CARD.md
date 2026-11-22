---
language: en
license: mit
base_model: distilbert-base-uncased
tags: [text-classification, fake-review-detection]
---

# Model card: review-authenticity-engine

## Summary

DistilBERT (`distilbert-base-uncased`) fine-tuned for binary classification of product reviews as
genuine (0) or synthetic (1). Maximum input length 128 tokens. Three epochs, learning rate 2e-5,
batch size 32, fp16, trained on Kaggle with two T4 GPUs.

## Intended use

A research and portfolio demonstration of fake-review detection, and one input to a larger system
that also uses reviewer behaviour. It is not suitable for moderating real reviews or making
decisions about real customers.

## Training data

Amazon Electronics reviews from the McAuley Lab 2023 release (genuine class) plus synthetic fakes
planted in rings across 400 products:

| Tier | How it was made |
|---|---|
| 1 Template | Fixed opener, noun and closer phrases |
| 2 Combinatorial | Templated sentences with sampled adjectives and reasons |
| 3 Paraphrased | Pegasus paraphrases of genuine review sentences |

The train/test split groups by exact review text, so no text appears on both sides.

**Known data problem.** An audit found the tier-3 paraphrases are mostly degenerate (word loops,
fragments such as "group way3 playingre balance well"). They do not represent fluent fakes.

## Evaluation

Reported by the training run on the held-out synthetic split: recall 100% (tier 1), 100% (tier 2),
99.3% (tier 3), and 100% of genuine reviews kept as genuine.

Context that matters for reading those figures:

- A logistic regression on length, punctuation and casing alone reaches 0.96 ROC-AUC on the same
  kind of data, so the classes are separable without understanding content.
- With every word masked the model still outputs 0.96 to 0.97, so text with little content defaults to "fake".
- The fluent, hand-written fake review used as a demo example scores 0.0%.
- Appending one ordinary complaint sentence to planted fakes drops recall from 99.9% to 43.6%;
  replacing some letters with look-alike Cyrillic characters drops it to 67%.

**Out of distribution.** On the Ott et al. deceptive hotel review corpus (1,600 reviews, half written by
paid crowd workers as fakes) the model flags none of the reviews; ROC-AUC 0.59, recall at 1% FPR 3%.

Full robustness and out-of-distribution results are produced by `scripts/evaluate_text.py` and stored in
`results/` in the project repository.

## Limitations and risks

- Fails on human-written fakes (see above). Not validated on modern LLM-written fakes, other product categories or other
  languages.
- Likely to flag short genuine reviews as fake. A false positive here means accusing a real customer.
- Probabilities are not calibrated for any real-world base rate of fake reviews.
