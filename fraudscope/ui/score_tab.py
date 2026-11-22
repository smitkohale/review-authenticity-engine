"""Tab 1: score a single review, explain it, and stress-test it with small edits."""

from __future__ import annotations

import html

import altair as alt
import pandas as pd
import streamlit as st

from .. import model as clf
from ..attacks import ATTACKS, apply_attack
from ..config import MODEL_ID, UNCERTAIN_HIGH, UNCERTAIN_LOW
from .html_parts import legend, token_highlight
from .palette import ATTR_NEG, ATTR_POS, GRID, INK, VERDICT_COLORS

EXAMPLES = {
    "Genuine": (
        "Bought this to replace the keyboard that came with my desk setup. Keys have a nice amount of travel "
        "and the number pad is handy for spreadsheets. Bluetooth pairing dropped twice in the first week but "
        "has been fine since a firmware update. Battery is at 60% after a month."
    ),
    "Template": "Amazing product, highly recommend buy it now",
    "Combinatorial": (
        "This is a superb gadget. Great build quality and fast shipping. "
        "Would definitely recommend to anyone looking for this."
    ),
    "Fluent fake": (
        "I was skeptical at first but this charger completely changed my routine. It charges my phone and "
        "tablet at the same time without getting hot, and it looks great on my nightstand. Honestly one of "
        "the best purchases I have made this year."
    ),
}

EDIT_LABELS = {
    "original": "Original text",
    "typos": "Typos",
    "homoglyphs": "Look-alike characters",
    "synonym_swap": "Synonym swap",
    "add_flaw_sentence": "Add a minor complaint",
    "lowercase_no_punct": "Lowercase, no punctuation",
    "word_dropout": "Drop 15% of words",
}


def _rgb(c: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


@st.cache_resource(show_spinner=False)
def _model():
    tokenizer, model = clf.load_model(MODEL_ID)
    return tokenizer, model, clf.make_predict_fn(tokenizer, model)


@st.cache_resource(show_spinner=False)
def _explainer():
    tokenizer, _, predict = _model()
    return clf.make_explainer(predict, tokenizer)


@st.cache_data(show_spinner=False, max_entries=256)
def _scores(texts: tuple[str, ...]) -> list[float]:
    _, _, predict = _model()
    return [float(p) for p in predict(list(texts))[:, 1]]


@st.cache_data(show_spinner=False, max_entries=64)
def _explain(text: str):
    return clf.explain_tokens(_explainer(), text)


def _use_example() -> None:
    name = st.session_state.get("example_choice")
    if name:
        st.session_state["review_text"] = EXAMPLES[name]
        st.session_state["scored_text"] = EXAMPLES[name]


def _probability_bar(p: float, color: str) -> str:
    pct = max(0.0, min(100.0, p * 100))
    lo, hi = UNCERTAIN_LOW * 100, UNCERTAIN_HIGH * 100
    return f"""
    <div class="prob-row"><span>Probability synthetic</span><strong>{pct:.1f}%</strong></div>
    <div class="prob-track">
      <div class="prob-band" style="left:{lo}%; width:{hi - lo}%"></div>
      <div class="prob-fill" style="width:{pct}%; background:{color}"></div>
      <div class="prob-tick" style="left:{lo}%"></div><div class="prob-tick" style="left:{hi}%"></div>
    </div>
    <div class="prob-scale"><span>0</span><span>uncertain {lo:.0f} to {hi:.0f}%</span><span>100</span></div>
    """


def _top_words_chart(pairs: list[tuple[str, float]]) -> alt.Chart:
    rows = {}
    for token, value in pairs:
        word = token.strip()
        if word and any(ch.isalnum() for ch in word):
            rows[word] = rows.get(word, 0.0) + value
    frame = pd.DataFrame({"word": list(rows), "value": list(rows.values())})
    frame = frame.reindex(frame["value"].abs().sort_values(ascending=False).index).head(10)
    frame["direction"] = frame["value"].map(lambda v: "Raises fake score" if v > 0 else "Lowers fake score")
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("value:Q", title="Contribution to fake probability", axis=alt.Axis(format="+.3~f", grid=True, tickCount=4)),
            y=alt.Y("word:N", sort=alt.EncodingSortField("value", op="sum", order="descending"), title=None,
                    scale=alt.Scale(paddingInner=0.35), axis=alt.Axis(labelOverlap=False, labelLimit=140)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Raises fake score", "Lowers fake score"],
                                range=[_rgb(ATTR_POS), _rgb(ATTR_NEG)]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("value:Q", title="Contribution", format="+.3f")],
        )
        .properties(height=max(120, 26 * len(frame)))
    )


def _stress_chart(frame: pd.DataFrame) -> alt.LayerChart:
    band = pd.DataFrame({"lo": [UNCERTAIN_LOW], "hi": [UNCERTAIN_HIGH]})
    base_x = alt.X("p:Q", title="Probability synthetic", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"))
    shade = alt.Chart(band).mark_rect(color="#efe9d6", opacity=0.8).encode(x="lo:Q", x2="hi:Q")
    bars = alt.Chart(frame).mark_bar(height=16, cornerRadius=2).encode(
        x=base_x,
        y=alt.Y("edit:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("verdict:N", scale=alt.Scale(domain=list(VERDICT_COLORS), range=list(VERDICT_COLORS.values())),
                        legend=None),
        opacity=alt.condition("datum.selected", alt.value(1.0), alt.value(0.45)),
        tooltip=[alt.Tooltip("edit:N", title="Edit"), alt.Tooltip("p:Q", title="Probability", format=".1%"),
                 alt.Tooltip("delta:Q", title="Change", format="+.1%")],
    )
    labels = alt.Chart(frame).mark_text(align="left", dx=4, fontSize=11, color=INK).encode(
        x="p:Q", y=alt.Y("edit:N", sort=None), text=alt.Text("p:Q", format=".0%")
    )
    return (shade + bars + labels).properties(height=34 * len(frame)).configure_axis(gridColor=GRID)


def render() -> None:
    st.session_state.setdefault("review_text", "")
    st.session_state.setdefault("scored_text", "")

    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.text_area("Review text", key="review_text", height=190, placeholder="Paste a product review")
        st.pills("Or try an example", list(EXAMPLES), key="example_choice", on_change=_use_example)
        explain = st.toggle("Word-level explanation", value=True, help="SHAP attributions; adds a few seconds")
        if st.button("Score review", type="primary"):
            st.session_state["scored_text"] = st.session_state["review_text"]

    text = st.session_state["scored_text"].strip()
    with right:
        if not text:
            st.markdown(
                '<div class="panel intro"><div class="verdict-label">How it works</div>'
                "<p>DistilBERT, fine-tuned on Amazon Electronics reviews with planted synthetic fakes, returns the "
                "probability that a review is synthetic.</p>"
                f"<p>Scores between {UNCERTAIN_LOW:.0%} and {UNCERTAIN_HIGH:.0%} count as uncertain and would go to "
                "a person. After scoring, the stress test shows how small edits move the score.</p></div>",
                unsafe_allow_html=True,
            )
            return
        try:
            with st.spinner("Loading model and scoring"):
                p = _scores((text,))[0]
        except Exception as exc:  # network or model-hub failures
            st.error(f"Could not score the review: {exc}")
            return
        v = clf.verdict_for(p)
        color = VERDICT_COLORS[v.label]
        st.markdown(
            f'<div class="panel verdict-panel" style="border-left-color:{color}">'
            f'<div class="verdict-label">Verdict</div>'
            f'<div class="verdict" style="color:{color}">{html.escape(v.headline)}</div>'
            f"{_probability_bar(p, color)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="note" style="margin-top:0.8rem">Trained on synthetic fakes only. Not validated on fakes '
            "written by people or modern language models. See Method and results.</p>",
            unsafe_allow_html=True,
        )

    if explain:
        st.subheader("Which words drove the score", anchor=False)
        try:
            with st.spinner("Computing attributions"):
                pairs, base = _explain(text)
            c1, c2 = st.columns([1.3, 1], gap="large")
            with c1:
                st.markdown(
                    legend({"Raises fake score": _rgb(ATTR_POS), "Lowers fake score": _rgb(ATTR_NEG)})
                    + token_highlight(pairs),
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="note">Hover a word for its exact contribution. With every word masked the model '
                    f"already outputs {base:.0%}; the words move the score from there.</p>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.altair_chart(_top_words_chart(pairs), use_container_width=True)
        except Exception as exc:
            st.warning(f"Explanation unavailable: {exc}")

    st.subheader("Stress test", anchor=False)
    st.markdown(
        '<p class="note">The same review after small edits that keep its meaning. A robust detector would give '
        "similar scores. Pick an edit to see the changed text.</p>",
        unsafe_allow_html=True,
    )
    choice = st.pills(
        "Edit", [EDIT_LABELS[k] for k in ["original", *ATTACKS]], default=EDIT_LABELS["add_flaw_sentence"],
        key="stress_choice", label_visibility="collapsed",
    )
    variants = {"original": text} | {name: apply_attack(name, [text], seed=0)[0] for name in ATTACKS}
    probs = _scores(tuple(variants.values()))
    frame = pd.DataFrame({
        "key": list(variants),
        "edit": [EDIT_LABELS[k] for k in variants],
        "p": probs,
        "text": list(variants.values()),
    })
    frame["delta"] = frame["p"] - probs[0]
    frame["verdict"] = frame["p"].map(lambda x: clf.verdict_for(x).label)
    frame["selected"] = frame["edit"] == choice if choice else True

    chart_col, text_col = st.columns([1.2, 1], gap="large")
    with chart_col:
        st.altair_chart(_stress_chart(frame), use_container_width=True)
        flips = frame[(frame["key"] != "original") & (frame["verdict"] != frame["verdict"].iloc[0])]
        if len(flips):
            names = ", ".join(flips["edit"].str.lower())
            st.markdown(f'<p class="note">The verdict changes under: {html.escape(names)}.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<p class="note">No edit changes the verdict for this review.</p>', unsafe_allow_html=True)
    with text_col:
        if choice:
            row = frame[frame["edit"] == choice].iloc[0]
            color = VERDICT_COLORS[row["verdict"]]
            st.markdown(
                f'<div class="panel verdict-panel" style="border-left-color:{color}">'
                f'<div class="verdict-label">{html.escape(choice)}</div>'
                f'<div class="stress-score" style="color:{color}">{row["p"]:.1%} '
                f'<span class="note">({row["delta"]:+.1%} vs original)</span></div>'
                f'<div class="edited">{html.escape(row["text"])}</div></div>',
                unsafe_allow_html=True,
            )
