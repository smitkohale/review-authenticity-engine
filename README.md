# Review Authenticity Engine

A fake review detector that also tries to catch coordinated review rings, not just individual fake reviews.

**Live demo:** https://review-trust-engine.streamlit.app/

## Why

Most fake review detectors just score one review at a time. But a lot of real fraud is coordinated — a group of accounts working together to boost or sabotage a product. Any single review from that group might look fine on its own. The pattern only shows up when you look at the group.

So this has two parts:
1. A text classifier that scores individual reviews
2. A graph analysis that looks at reviewer behavior across products and time to spot coordinated groups

## How it works

**Text side:** Fine-tuned DistilBERT on Amazon Reviews data (McAuley Lab 2023 dataset), with synthetic fake reviews injected at three levels of sophistication — obvious template spam, combinatorial templates, and LLM-paraphrased reviews meant to sound natural and posted in staggered timing.

**Graph side:** Built a reviewer-to-reviewer graph where edge weight depends on how close together in time two people reviewed the same product, how similar their review text is (sentence embeddings), and how many products they both reviewed. Ran Louvain community detection on it to find dense clusters — candidate coordinated rings.

**Combined:** each review gets a fake-probability from the text model, plus a flag if the reviewer's in a detected ring. Both feed into one risk score.

## Results

Text classifier recall by tier (after fixing a data leak, see below):
- Template spam: 100%
- Combinatorial templates: 100%
- Paraphrased fakes: 99.3%
- Real reviews kept as real: 100%

Graph layer, checked against ground truth: found 132 small, dense clusters with over 50% fake concentration, catching about 35-36% of planted fake reviewers using zero text signal — just timing and behavioral patterns.

Worth being upfront about what this actually shows: the text classifier alone did really well here, including on the paraphrased tier, so the graph layer's recall on its own is lower than the text layer's. That's a real result, not the outcome I expected going in. The value of the graph layer isn't that it outperforms text — it's that it catches things through a completely different signal (behavior, not content), which matters if fraud rings ever get good enough at varying their text to slip past a text classifier specifically.

## A bug worth mentioning

Early on, my "paraphrased" fake reviews were only pulling from about 20 unique sentences, and they leaked across my train/test split — so the model was scoring ~0.995 F1 partly because it had just memorized those sentences, not because it generalized. Caught this by checking exact text overlap between train and test (found over 95% overlap on that tier), then fixed it by generating a much larger, more genuinely varied paraphrase pool and switching to a split that guarantees no review text appears in both train and test. Retrained and checked it against hand-written examples it had never seen anywhere in the pipeline before trusting the numbers again.

## Stack

- DistilBERT, fine-tuned, hosted on [Hugging Face](https://huggingface.co/Smitvkohale/review-authenticity-engine)
- NetworkX + python-louvain for the graph, Sentence-Transformers (MiniLM) for embeddings
- SHAP for word-level explanations
- Streamlit for the UI
- Trained on Kaggle (dual T4 GPU)

## Limitations

- Fraud patterns here are synthetically injected into a real dataset, not real-world labeled fraud — actual fraud rings may look and behave differently
- Graph edge weights (timing/similarity/overlap) were set manually, not tuned exhaustively
- The rings dashboard shows a static export from one analysis run, not a live pipeline

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

— Smit Kohale
