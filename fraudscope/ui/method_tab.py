"""Tab 3: method, results and limitations, driven by the JSON files in results/."""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from .. import audit, datasets
from ..config import RESULTS_DIR, TIER_NAMES
from .html_parts import stat_tiles
from .palette import ACCENT, GRID, NEUTRAL, TIER_COLORS, VERDICT_COLORS, tier_scale

PIPELINE_HTML = """
<style>
.flow { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; margin: 0.4rem 0 0.6rem; }
.flow-col { border: 1px solid var(--line); border-radius: 0.3rem; background: var(--panel); padding: 0.8rem 1rem; }
.flow-col h4 { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--faint); margin: 0 0 0.5rem; padding: 0; font-weight: 500; }
.flow-col ol { margin: 0; padding-left: 1.1rem; }
.flow-col li { margin: 0.2rem 0; line-height: 1.45; }
.flow-col li span { color: var(--muted); }
.flow-shared { grid-column: 1 / -1; color: var(--muted); font-size: 0.9rem; }
@media (max-width: 700px) { .flow { grid-template-columns: 1fr; } }
</style>
<div class="flow">
  <div class="flow-shared">Shared input: Amazon Electronics reviews (McAuley Lab 2023) with planted rings in three
  tiers: template, combinatorial and paraphrased.</div>
  <div class="flow-col">
    <h4>Text layer</h4>
    <ol>
      <li>Split by unique text <span>so no review appears in both train and test</span></li>
      <li>Fine-tune DistilBERT <span>to output a per-review probability</span></li>
      <li>SHAP attributions <span>show which words moved the score</span></li>
    </ol>
  </div>
  <div class="flow-col">
    <h4>Behaviour layer</h4>
    <ol>
      <li>Reviewer graph <span>weighted by timing, text similarity and shared products</span></li>
      <li>Louvain communities <span>group densely linked reviewers</span></li>
      <li>Rank clusters <span>by edge density times mean edge weight, no labels</span></li>
    </ol>
  </div>
</div>
"""

REPORTED_TEXT = pd.DataFrame(
    [
        {"Test slice": "Template spam", "Metric": "Recall", "Reported": "100%"},
        {"Test slice": "Combinatorial templates", "Metric": "Recall", "Reported": "100%"},
        {"Test slice": "Paraphrased", "Metric": "Recall", "Reported": "99.3%"},
        {"Test slice": "Genuine reviews", "Metric": "Kept as genuine", "Reported": "100%"},
    ]
)


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def _ring_reviews() -> pd.DataFrame:
    return datasets.load_ring_reviews()


@st.cache_data(show_spinner=False)
def _audit_rows() -> list[dict]:
    df = _ring_reviews()
    return audit.text_audit(df) if not df.empty else []


def _length_chart(df: pd.DataFrame) -> alt.Chart:
    data = pd.DataFrame({
        "Label": df["tier"].map(TIER_NAMES),
        "Words": df["text"].str.split().str.len().clip(lower=1),
    })
    return (
        alt.Chart(data)
        .mark_boxplot(size=18, median={"color": "white"}, outliers={"size": 8, "opacity": 0.4})
        .encode(
            x=alt.X("Words:Q", scale=alt.Scale(type="log"), title="Words per review (log scale)"),
            y=alt.Y("Label:N", title=None, sort=list(TIER_NAMES.values()), axis=alt.Axis(labelLimit=220)),
            color=alt.Color("Label:N", scale=tier_scale(), legend=None),
        )
        .properties(height=190)
        .configure_axis(gridColor=GRID)
        .configure_view(strokeWidth=0)
    )


def _auc_chart(rows: list[dict]) -> alt.Chart:
    frame = pd.DataFrame(rows)
    frame["model_label"] = frame["model"] + "  " + frame["auc"].map(lambda v: f"{v:.3f}")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("auc:Q", title="ROC-AUC", scale=alt.Scale(domain=[0.5, 1.0], clamp=True)),
            y=alt.Y("model_label:N", title=None, sort=None,
                    axis=alt.Axis(labelLimit=260)),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["Shortcut", "Text model"],
                                                      range=[TIER_COLORS["Planted: combinatorial"], ACCENT]),
                            legend=alt.Legend(title=None, orient="bottom", labelLimit=320)),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("auc:Q", title="ROC-AUC", format=".4f")],
        )
        .properties(height=44 * len(frame) + 100)
        .configure_axis(gridColor=GRID)
        .configure_view(strokeWidth=0)
    )


def _robustness_chart(frame: pd.DataFrame) -> alt.Chart:
    pick = alt.selection_point(fields=["Slice"], bind="legend")
    order = frame.drop_duplicates("Edit")["Edit"].tolist()
    colors = {"All planted": "#1d1d1f", **{k: v for k, v in TIER_COLORS.items() if k.startswith("Planted")}}
    dots = alt.Chart(frame).mark_circle(stroke="white", strokeWidth=1).encode(
        y=alt.Y("Edit:N", sort=order, title=None, axis=alt.Axis(labelLimit=220, grid=True)),
        x=alt.X("Recall:Q", title="Share of planted fakes still caught", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Slice:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())),
                        legend=alt.Legend(title="Click to isolate", orient="bottom")),
        size=alt.condition("datum.Slice == 'All planted'", alt.value(180), alt.value(110)),
        opacity=alt.condition(pick, alt.value(1), alt.value(0.12)),
        tooltip=["Edit", "Slice", alt.Tooltip("Recall:Q", format=".1%")],
    ).add_params(pick)
    return dots.properties(height=36 * len(order) + 40).configure_axis(gridColor=GRID).configure_view(strokeWidth=0)


DATASET_NAMES = {
    "rings": "Own synthetic data",
    "ott": "Ott et al., human-written fakes",
    "salminen": "Salminen et al., GPT-2 fakes",
    "yelpchi": "YelpChi (Yelp filter labels)",
    "amazon": "Amazon (user-level labels)",
}

METRIC_NAMES = {"roc_auc": "ROC-AUC", "recall": "Recall at 0.5", "fpr": "False-positive rate",
                "pr_auc": "PR-AUC", "recall_at_0.01_fpr": "Recall at 1% FPR"}


def _external_section(ev: dict | None, external: list[tuple[str, dict]]) -> None:
    st.subheader("Does it generalise? Human-written fakes", anchor=False)
    _note(
        "The same model, unchanged, on reviews it was not built for. The Ott corpus has 1,600 Chicago hotel "
        "reviews: 800 genuine and 800 fakes written by paid crowd workers asked to sound convincing."
    )
    rows = []
    sources = ([("rings", ev)] if ev else []) + external
    for name, r in sources:
        for key in ("roc_auc", "recall", "fpr"):
            rows.append({"Dataset": DATASET_NAMES[name], "Metric": METRIC_NAMES[key], "Value": r["overall"][key]})
    frame = pd.DataFrame(rows)
    colors = [ACCENT, TIER_COLORS["Planted: template"], TIER_COLORS["Planted: paraphrased"]][: frame["Dataset"].nunique()]
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("Metric:N", title=None, sort=list(METRIC_NAMES.values()), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Dataset:N", sort=list(DATASET_NAMES.values())),
            y=alt.Y("Value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Dataset:N", sort=list(DATASET_NAMES.values()),
                            scale=alt.Scale(domain=frame["Dataset"].unique().tolist(), range=colors),
                            legend=alt.Legend(title=None, orient="bottom", labelLimit=320)),
            tooltip=["Dataset", "Metric", alt.Tooltip("Value:Q", format=".1%")],
        )
        .properties(height=300)
        .configure_axis(gridColor=GRID)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
    ott = dict(external).get("ott")
    if ott:
        o = ott["overall"]
        flagged = o["recall"] * o["positives"] + o["fpr"] * (o["n"] - o["positives"])
        st.markdown(
            f'<div class="finding"><strong>The text model flags {flagged:.0f} of {o["n"]:,} reviews.</strong> '
            f"Ott reviews average about 150 words, long and detailed like genuine Amazon reviews, so every one "
            f"scores close to 0% fake, crowd-written fakes included. Its ranking is only slightly better than "
            f"chance (ROC-AUC {o['roc_auc']:.2f}; 0.5 is a coin flip), and recall at a 1% false-positive budget is "
            f"{o.get('recall_at_0.01_fpr', 0):.0%}. The classifier detects the synthetic generator it was trained "
            "against, not deception. This is the main reason the behaviour layer matters.</div>",
            unsafe_allow_html=True,
        )


def _benchmark_section(benchmarks: list[tuple[str, dict]]) -> None:
    st.subheader("Behaviour signals on real fraud graphs", anchor=False)
    _note(
        "YelpChi and Amazon are standard graph fraud benchmarks with labels from real platform filtering. "
        "Gradient boosting on each node's own behaviour features is compared with the same model given the "
        "average features of its graph neighbours, the simple baseline GADBench found competitive with "
        "specialised graph neural networks. Stratified 40/60 splits, mean and spread over three seeds."
    )
    metric = st.segmented_control(
        "Metric", ["PR-AUC", "ROC-AUC", "Recall at 1% FPR"], default="PR-AUC", key="bench_metric",
    ) or "PR-AUC"
    key = {"PR-AUC": "pr_auc", "ROC-AUC": "roc_auc", "Recall at 1% FPR": "recall_at_0.01_fpr"}[metric]
    rows = []
    for name, r in benchmarks:
        for method, label in (("features_only", "Node features only"), ("graph_agg", "Plus neighbour features")):
            s = r["summary"][method][key]
            rows.append({"Dataset": DATASET_NAMES[name], "Method": label, "Value": s["mean"], "Spread": s["std"],
                         "Fraud share": r["fraud_share"]})
    frame = pd.DataFrame(rows)
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("Dataset:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Method:N"),
            y=alt.Y("Value:Q", title=metric, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Method:N", scale=alt.Scale(domain=["Node features only", "Plus neighbour features"],
                                                        range=[NEUTRAL, ACCENT]),
                            legend=alt.Legend(title=None, orient="bottom", labelLimit=320)),
            tooltip=["Dataset", "Method", alt.Tooltip("Value:Q", format=".3f", title=metric),
                     alt.Tooltip("Spread:Q", format=".3f", title="Std over seeds"),
                     alt.Tooltip("Fraud share:Q", format=".1%")],
        )
        .properties(height=300)
        .configure_axis(gridColor=GRID)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)
    gains = []
    for name, r in benchmarks:
        a = r["summary"]["features_only"]["pr_auc"]["mean"]
        b = r["summary"]["graph_agg"]["pr_auc"]["mean"]
        gains.append(f"{DATASET_NAMES[name].split(' (')[0]} PR-AUC {a:.3f} to {b:.3f}")
    _note("Adding neighbour information: " + "; ".join(gains) + ".")


def _findings() -> str:
    base = _load("baselines_rings.json")
    ev = _load("text_eval_rings.json")
    rows = _audit_rows()
    items = []
    if base and "surface_only" in base["models"]:
        items.append({"label": "Length and punctuation only", "value": f"{base['models']['surface_only']['roc_auc']:.2f}",
                      "note": "ROC-AUC with no word content", "color": TIER_COLORS["Planted: combinatorial"]})
    if ev and ev.get("attacks", {}).get("add_flaw_sentence"):
        items.append({"label": "Recall after one added sentence", "value": f"{ev['attacks']['add_flaw_sentence']['recall']:.0%}",
                      "note": f"down from {ev['attacks']['none']['recall']:.1%}", "color": VERDICT_COLORS["fake"]})
    t3 = next((r for r in rows if r["tier"] == 3), None)
    if t3:
        items.append({"label": "Paraphrases starting with one word", "value": f"{t3['top_first_word_share']:.0%}",
                      "note": f"the word \"{t3['top_first_word']}\"", "color": TIER_COLORS["Planted: paraphrased"]})
    ott = _load("text_eval_ott.json")
    if ott:
        o = ott["overall"]
        caught = round(o["recall"] * o["positives"])
        items.append({"label": "Human-written fakes caught", "value": f"{caught} of {o['positives']}",
                      "note": f"Ott et al. corpus, ROC-AUC {o['roc_auc']:.2f}", "color": ACCENT})
    else:
        items.append({"label": "Score with every word masked", "value": "96-97%",
                      "note": "no content still reads as fake", "color": ACCENT})
    return stat_tiles(items)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _note(text: str) -> None:
    st.markdown(f'<p class="note">{text}</p>', unsafe_allow_html=True)


def render() -> None:
    st.subheader("The question", anchor=False)
    st.markdown(
        "Do behaviour signals (who reviews what, and when) catch coordinated fake reviews that a text "
        "classifier misses? The project has two parts: a DistilBERT classifier that scores single reviews, "
        "and a reviewer graph that looks for groups of accounts acting together."
    )
    st.markdown(PIPELINE_HTML, unsafe_allow_html=True)

    st.subheader("Key findings from the audit", anchor=False)
    st.markdown(_findings(), unsafe_allow_html=True)
    st.markdown(
        '<div class="finding"><strong>Reading the numbers.</strong> The reported near-perfect recall comes from '
        "planted text that differs from real reviews on the surface: shorter, templated, or broken. The classifier "
        "learned those differences, not deception. The sections below show the evidence.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("What the training run reported", anchor=False)
    _note(
        "Held-out synthetic test set, split so that no review text appears in both train and test. "
        "These figures come from the training notebook, whose saved outputs were cleared, so they "
        "cannot be checked from the repository alone."
    )
    st.dataframe(REPORTED_TEXT, hide_index=True, use_container_width=False)

    st.subheader("Audit: are those numbers measuring deception?", anchor=False)
    rows = _audit_rows()
    if rows:
        frame = pd.DataFrame(rows)
        frame["Slice"] = frame["tier"].map(TIER_NAMES)
        view = pd.DataFrame({
            "Slice": frame["Slice"],
            "Reviews": frame["n"],
            "Mean words": frame["mean_words"].round(1),
            "Unique texts": frame["unique_share"].map(_pct),
            "Degenerate text": frame["degenerate_share"].map(_pct),
            "Most common first word": frame["top_first_word"] + " (" + frame["top_first_word_share"].map(_pct) + ")",
        })
        reviews = _ring_reviews()
        st.caption("Words per review by label. Hover a box for quartiles.")
        st.altair_chart(_length_chart(reviews), use_container_width=True)
        with st.expander("Per-label statistics"):
            st.dataframe(view, hide_index=True, use_container_width=True)
        tier3 = next((r for r in rows if r["tier"] == 3), None)
        genuine = next((r for r in rows if r["tier"] == 0), None)
        if tier3 and genuine:
            _note(
                f"Computed from the {sum(r['n'] for r in rows):,} reviews in the exported clusters. "
                f"Planted paraphrases average {tier3['mean_words']:.0f} words against {genuine['mean_words']:.0f} "
                f"for genuine reviews, and {tier3['top_first_word_share']:.0%} of them start with the same word. "
                "Reading them shows the paraphrasing step produced mostly word salad rather than fluent text, "
                "so the paraphrased-tier result says little about fluent fakes."
            )

    base = _load("baselines_rings.json")
    if base:
        st.subheader("Shortcut check", anchor=False)
        _note(
            "Cross-validated on the same reviews, grouped so identical texts never cross folds. "
            "The surface-only model sees nothing but length, punctuation and casing statistics."
        )
        table = []
        labels = {"surface_only": "Surface features only", "tfidf_lr": "TF-IDF + logistic regression"}
        for key, rep in base["models"].items():
            table.append({
                "Model": labels.get(key, key),
                "ROC-AUC": f"{rep['roc_auc']:.3f}",
                "Recall": _pct(rep["recall"]),
                "False-positive rate": _pct(rep["fpr"]),
                "Recall at 1% FPR": _pct(rep.get("recall_at_0.01_fpr")),
            })
        auc_rows = [
            {"model": labels.get(k, k), "auc": rep["roc_auc"], "kind": "Shortcut" if k == "surface_only" else "Text model"}
            for k, rep in base["models"].items()
        ]
        ev_overall = (_load("text_eval_rings.json") or {}).get("overall")
        if ev_overall:
            auc_rows.append({"model": "DistilBERT (this project)", "auc": ev_overall["roc_auc"], "kind": "Text model"})
        st.altair_chart(_auc_chart(auc_rows), use_container_width=True)
        with st.expander("All metrics"):
            st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)
        surface = base["models"].get("surface_only")
        if surface:
            _note(
                f"Length and punctuation alone reach {surface['roc_auc']:.2f} ROC-AUC. The synthetic fakes "
                "differ from real reviews in ways a model can learn without reading for meaning, which is "
                "why near-perfect scores here are expected and not evidence of deception detection."
            )

    ev = _load("text_eval_rings.json")
    if ev and ev.get("attacks"):
        st.subheader("Robustness to small edits", anchor=False)
        _note(
            "DistilBERT recall on planted fakes after meaning-preserving edits, and the false-positive rate "
            "on genuine reviews after the same edits. Most of these reviews were in the training data, so "
            "this measures sensitivity to surface changes, not generalisation."
        )
        names = {
            "none": "No edit", "typos": "Typos", "homoglyphs": "Look-alike characters",
            "synonym_swap": "Synonym swap", "add_flaw_sentence": "Add a minor complaint",
            "lowercase_no_punct": "Lowercase, no punctuation", "word_dropout": "Drop 15% of words",
        }
        rows = []
        long_rows = []
        for key, r in ev["attacks"].items():
            by_tier = r.get("recall_by_tier") or {
                t: v["share_flagged"] for t, v in ev.get("by_tier", {}).items()
            }
            label = names.get(key, key)
            long_rows.append({"Edit": label, "Slice": "All planted", "Recall": r.get("recall")})
            for tier in ("1", "2", "3"):
                if by_tier.get(tier) is not None:
                    long_rows.append({"Edit": label, "Slice": TIER_NAMES[int(tier)], "Recall": by_tier[tier]})
            rows.append({
                "Edit": names.get(key, key),
                "Recall, all planted": _pct(r.get("recall")),
                "Template": _pct(by_tier.get("1")),
                "Combinatorial": _pct(by_tier.get("2")),
                "Paraphrased": _pct(by_tier.get("3")),
                "False-positive rate": _pct(r.get("fpr")),
            })
        st.altair_chart(_robustness_chart(pd.DataFrame(long_rows)), use_container_width=True)
        with st.expander("Table, including false-positive rates on genuine reviews"):
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    external = [(n, _load(f"text_eval_{n}.json")) for n in ("ott", "salminen")]
    external = [(n, r) for n, r in external if r]
    if external:
        _external_section(ev, external)

    benchmarks = [(n, _load(f"graph_benchmark_{n}.json")) for n in ("yelpchi", "amazon")]
    benchmarks = [(n, r) for n, r in benchmarks if r]
    if benchmarks:
        _benchmark_section(benchmarks)

    st.subheader("Graph layer", anchor=False)
    graph_eval = next((r for r in (_load(p.name) for p in sorted(RESULTS_DIR.glob("graph_eval_*.json"))) if r), None)
    if graph_eval:
        lv = graph_eval["louvain"]
        flag = next(v for k, v in lv.items() if k.startswith("flag_top_"))
        rows = [
            {"Method": "Louvain clusters ranked by suspicion", **{k: _pct(flag[k]) for k in ("precision", "recall", "f1")}},
            {"Method": "Rule: same product within 48 hours", **{k: _pct(graph_eval["baseline_48h_window"][k]) for k in ("precision", "recall", "f1")}},
            {"Method": "FRAUDAR dense blocks", **{k: _pct(graph_eval["baseline_fraudar"][k]) for k in ("precision", "recall", "f1")}},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    else:
        st.markdown(
            "The training run reported 132 small clusters with more than half planted reviewers, recovering "
            "about 35% of planted reviewers from timing and co-review behaviour alone, with no text signal."
        )
        _note(
            "Caveat: that run picked which clusters to export using the ground-truth labels. "
            "scripts/build_graph.py now ranks clusters by an unsupervised suspicion score "
            "(edge density times mean edge weight) and compares against a 48-hour co-review rule and FRAUDAR."
        )

    st.subheader("Limitations", anchor=False)
    st.markdown(
        "- Fakes are synthetic and injected into real data; real fraud rings may behave differently.\n"
        "- The paraphrased tier is mostly degenerate text and needs regenerating with an instruction-tuned model "
        "(scripts/generate_fakes.py).\n"
        "- Surface statistics separate the classes, so in-distribution scores overstate real-world performance.\n"
        "- Graph edge weights were set by hand.\n"
        "- The cluster view is a static export from one run, capped at 20 reviews per cluster."
    )
