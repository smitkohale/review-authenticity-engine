# Review Authenticity Engine

Detects fake product reviews two ways: a text classifier that scores single reviews, and a reviewer
graph that looks for groups of accounts acting together.

**Live demo:** https://review-trust-engine.streamlit.app/

The demo runs on Streamlit's free tier. If it shows a sleeping page, click the wake-up button and
allow about 30 seconds; the first scored review also takes a few seconds while the model downloads.

## The question

Do behaviour signals (who reviews what, and when) catch coordinated fake reviews that a text
classifier misses? A single review from a coordinated group can look fine on its own; the pattern
only shows up across the group. Network signals are also harder to fake than text, which matters
now that language models write fluent reviews and people spot machine-written reviews at roughly
chance level.

## How it works

**Data.** Real Amazon Electronics reviews (McAuley Lab 2023) with planted review rings in three
tiers: template spam, combinatorial templates, and paraphrased text.

**Text layer.** DistilBERT fine-tuned on that data, split by unique review text so no review appears
in both train and test. SHAP shows which words moved each score.

**Behaviour layer.** A reviewer-to-reviewer graph. Edge weight combines how close in time two
accounts reviewed the same product, how similar their text is (MiniLM embeddings), and the Jaccard
overlap of the products they reviewed. Louvain community detection finds dense groups, which are
ranked by an unsupervised suspicion score (edge density times mean edge weight).

## What an audit of the original results found

The training run reported near-perfect recall on every tier. Re-examining the data shows why that
number is not evidence of deception detection:

| Finding | Evidence |
|---|---|
| The paraphrased tier is mostly word salad, not fluent text | 59% of these reviews start with the word "group", they average 12 words against 81 for genuine reviews, and many are loops such as "problems problems problems". The Pegasus step was broken. |
| Surface statistics alone separate the classes | A logistic regression on length, punctuation and casing only reaches 0.96 ROC-AUC; TF-IDF reaches 0.9998. |
| With nothing to read, it defaults to "fake" | With every word masked, the model still outputs 96 to 97% fake (SHAP's base value), while the fluent hand-written fake in the demo scores 0.0%. |
| Small, meaning-preserving edits defeat it | Appending one ordinary complaint ("The only downside is that the cable is a little short.") drops recall on planted fakes from 99.9% to 43.6%, and to 17% on template spam. Swapping a few letters for look-alike characters drops it to 67%. |
| Exported clusters were chosen with ground-truth labels | The notebook kept clusters with more than 50% planted reviewers. Clusters are now ranked without labels. |
| The overlap signal pulled genuine customers into rings | Dividing shared products by the smaller account's count gives a one-product fake account a perfect score against every real buyer of the targeted product. Switched to Jaccard. |

The old dashboard also hid "low-quality" reviews for readability, and those were exactly the broken
paraphrases. They are now shown by default.

Full tables are in the app's **Method and results** tab, generated from the JSON files in `results/`.

## Does the text model generalise?

No. On the Ott et al. corpus (1,600 hotel reviews; 800 genuine, 800 fakes written by paid crowd workers),
the unchanged classifier flags **none** of the reviews. Every review, fake or genuine, scores close to 0%
fake. Ranking quality is barely above chance (ROC-AUC 0.59) and recall at a 1% false-positive rate is 3%.
These reviews average about 150 words, and long, detailed text reads as genuine to this model. It learned
to spot the synthetic generator, not deception.

## Results on real fraud graphs

YelpChi (45,954 reviews, 14.5% filtered by Yelp) and Amazon (8,639 labelled users, 9.5% fraud) from the
CARE-GNN release. Gradient boosting on each node's own behaviour features, compared with the same model
given the mean features of its graph neighbours. Stratified 40/60 splits, mean over three seeds
(standard deviation at most 0.02).

| Dataset | Features | PR-AUC | ROC-AUC | Recall at 1% FPR |
|---|---|---|---|---|
| YelpChi | node only | 0.797 | 0.938 | 48.0% |
| YelpChi | node + neighbours | **0.879** | **0.967** | **62.9%** |
| Amazon | node only | 0.899 | 0.974 | 80.9% |
| Amazon | node + neighbours | **0.928** | **0.982** | **83.8%** |

On real labels, knowing who a reviewer is connected to adds signal beyond their own behaviour, most
clearly on YelpChi, where recall at a 1% false-positive budget rises by 15 points.

## Repository layout

```
app.py                     Streamlit entry point
fraudscope/
  model.py                 DistilBERT loading, batched inference, SHAP
  graph.py                 reviewer graph, Louvain, suspicion ranking, FRAUDAR and 48h baselines
  synthetic.py             ring planting, including evasion and camouflage attackers
  attacks.py               meaning-preserving text perturbations for robustness tests
  metrics.py               ROC-AUC, PR-AUC, recall at fixed FPR, calibration (numpy only)
  datasets.py              loaders for the ring data, Ott et al., Salminen et al., generic CSV
  audit.py                 per-tier length, duplication and degenerate-text statistics
  ui/                      the three app tabs and their styles
scripts/
  evaluate_text.py         classifier metrics, per-tier results and attack robustness
  baselines.py             surface-only and TF-IDF baselines with text-grouped CV
  build_graph.py           graph pipeline, baselines, ablations, unsupervised export
  benchmark_graph.py       YelpChi / Amazon fraud graphs: features vs graph-aggregated boosting
  generate_fakes.py        fluent fakes from instruction-tuned LLMs with a quality gate
results/                   JSON outputs read by the app
tests/                     pytest suite (torch-free; runs in CI)
```

## Reproducing

```bash
pip install -r requirements-dev.txt

# Text classifier on the shipped ring reviews (set MODEL_PATH to use a local copy of the model)
python scripts/evaluate_text.py --dataset rings
python scripts/baselines.py --dataset rings

# Out-of-distribution text tests (download the datasets first)
python scripts/evaluate_text.py --dataset ott --path data/op_spam_v1.4
python scripts/evaluate_text.py --dataset salminen --path "data/fake reviews dataset.csv"

# Graph pipeline on the full labelled data, with signal ablations
python scripts/build_graph.py --input data/fraudscope_full_labeled_v2.parquet --ablate
python scripts/build_graph.py --demo     # smoke test on toy data

# Real fraud graphs (CARE-GNN release of YelpChi and Amazon)
python scripts/benchmark_graph.py --mat data/YelpChi.mat --name yelpchi

pytest -q
streamlit run app.py
```

## Next experiments

1. Regenerate the paraphrased tier with several instruction-tuned LLMs (`scripts/generate_fakes.py`),
   hold one generator out, and retrain.
2. Retrain the classifier with length-matched genuine and fake reviews plus human-written fakes, then re-run
   the Ott test, and add the GPT-2 fakes of Salminen et al. (download currently unavailable).
3. Plant evasive rings (copied genuine text) and camouflaged rings, and measure what the graph
   recovers when the text model is blind.
4. Run the graph methods on YelpChi and Amazon, where labels come from real platform filtering.
5. Combine the text score and graph features into one model and report text-only, graph-only and
   combined results side by side.

## Related work

- Mukherjee, Liu and Glance, *Spotting Fake Reviewer Groups in Consumer Reviews*, WWW 2012.
- Hooi et al., *FRAUDAR: Bounding Graph Fraud in the Face of Camouflage*, KDD 2016.
- Ott et al., *Finding Deceptive Opinion Spam by Any Stretch of the Imagination*, ACL 2011.
- He, Hollenbeck, Overgoor, Proserpio and Tosyali, *Detecting fake-review buyers using network structure*, PNAS 2022.
- Tang et al., *GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection*, NeurIPS 2023.
- Dugan et al., *RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors*, ACL 2024.
- Liu et al., *Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network*, 2025.

## Limitations

- Fakes are synthetic and injected into real data; real rings may behave differently.
- In-distribution scores overstate real-world performance because surface statistics separate the classes.
- Graph edge weights were set by hand.
- The cluster view is a static export from one run.

Model card: [MODEL_CARD.md](MODEL_CARD.md). Licence: MIT.
