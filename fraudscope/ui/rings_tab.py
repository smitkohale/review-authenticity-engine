"""Tab 2: browse the reviewer clusters found by the graph analysis."""

from __future__ import annotations

import html

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st

from .. import rings as rg
from .html_parts import legend, stat_tiles
from .palette import ACCENT, BAND_COLORS, GRID, NEUTRAL, TIER_COLORS, band_scale, tier_scale

FILTER_DEFAULTS = {
    "ring_query": "",
    "ring_min_share": 0,
    "ring_sort": next(iter(rg.SORT_OPTIONS)),
    "ring_hide_artifacts": False,
}


@st.cache_data(show_spinner=False)
def _load() -> list[dict]:
    return rg.load_rings()


def _reset_filters() -> None:
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


def _on_table_select() -> None:
    rows = st.session_state["ring_table"].selection.rows
    ids = st.session_state.get("ring_visible_ids", [])
    if rows and rows[0] < len(ids):
        st.session_state["selected_cluster"] = ids[rows[0]]


def _on_scatter_select() -> None:
    picked = st.session_state["ring_scatter"].selection.get("pick", [])
    if picked:
        st.session_state["selected_cluster"] = picked[0]["Cluster"]


def _overview_scatter(table: pd.DataFrame, selected) -> alt.Chart:
    pick = alt.selection_point(name="pick", fields=["Cluster"], on="click")
    data = table.assign(Selected=table["Cluster"] == selected)
    return alt.Chart(data).mark_circle().encode(
        x=alt.X("Reviewers:Q", title="Reviewers in cluster", scale=alt.Scale(zero=False, padding=8)),
        y=alt.Y("Planted share:Q", title="Planted-fake share", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[min(0.45, float(table["Planted share"].min()) - 0.05), 1.0])),
        size=alt.Size("Products:Q", scale=alt.Scale(range=[40, 420]), legend=alt.Legend(title="Products", orient="right")),
        color=alt.Color("Band:N", scale=band_scale(), legend=alt.Legend(title="Band", orient="right")),
        opacity=alt.condition("datum.Selected", alt.value(1.0), alt.value(0.6)),
        stroke=alt.condition("datum.Selected", alt.value("#1d1d1f"), alt.value("white")),
        strokeWidth=alt.condition("datum.Selected", alt.value(2.5), alt.value(0.8)),
        tooltip=["Cluster", "Reviewers", "Products", alt.Tooltip("Planted share:Q", format=".0%"), "Band"],
    ).add_params(pick).properties(height=300).configure_axis(gridColor=GRID).configure_view(strokeWidth=0)


def _label_mix(rings: list[dict]) -> alt.Chart:
    counts = pd.Series([rg.tier_name(r.get("tier")) for ring in rings for r in ring.get("reviews", [])]).value_counts()
    frame = pd.DataFrame({"Label": counts.index, "Reviews": counts.to_numpy()})
    frame["Share"] = frame["Reviews"] / frame["Reviews"].sum()
    order = [k for k in TIER_COLORS if k != "Product"]
    frame["rank"] = frame["Label"].map({k: i for i, k in enumerate(order)})
    return (
        alt.Chart(frame)
        .mark_bar(height=22)
        .encode(
            x=alt.X("sum(Reviews):Q", stack="normalize", axis=alt.Axis(format="%", title=None, tickCount=5)),
            color=alt.Color("Label:N", scale=tier_scale(), legend=None),
            order=alt.Order("rank:Q", sort="ascending"),
            tooltip=["Label", "Reviews", alt.Tooltip("Share:Q", format=".0%")],
        )
        .properties(height=48)
        .configure_view(strokeWidth=0)
    )


def _network_chart(reviews: list[dict]) -> alt.LayerChart:
    g = nx.Graph()
    for r in reviews:
        user, product = f"u:{r.get('user_id')}", f"p:{r.get('asin')}"
        g.add_node(user, kind="Reviewer", label=rg.tier_name(r.get("tier")), name=r.get("user_id"))
        g.add_node(product, kind="Product", label="Product", name=r.get("asin"))
        g.add_edge(user, product, label=rg.tier_name(r.get("tier")))
    pos = nx.spring_layout(g, seed=7, k=0.9 / max(1, len(g) ** 0.5))
    degree = dict(g.degree())

    nodes = pd.DataFrame([{"x": pos[n][0], "y": pos[n][1], "links": degree[n], **g.nodes[n]} for n in g.nodes])
    edges = pd.DataFrame(
        [{"x": pos[a][0], "y": pos[a][1], "x2": pos[b][0], "y2": pos[b][1], "label": d["label"]}
         for a, b, d in g.edges(data=True)]
    )
    axis = alt.Axis(title=None, labels=False, ticks=False, domain=False, grid=False)
    hover = alt.selection_point(name="hover", fields=["label"], on="pointerover", clear="pointerout", empty=True)
    edge_layer = alt.Chart(edges).mark_rule(strokeWidth=1).encode(
        x=alt.X("x:Q", axis=axis), y=alt.Y("y:Q", axis=axis), x2="x2:Q", y2="y2:Q",
        color=alt.Color("label:N", scale=tier_scale(), legend=None),
        opacity=alt.condition(hover, alt.value(0.45), alt.value(0.08)),
    )
    node_layer = alt.Chart(nodes).mark_point(filled=True, stroke="white", strokeWidth=1).encode(
        x="x:Q",
        y="y:Q",
        shape=alt.Shape("kind:N", scale=alt.Scale(domain=["Reviewer", "Product"], range=["circle", "square"]),
                        legend=None),
        size=alt.condition("datum.kind == 'Product'", alt.value(220), alt.value(120)),
        color=alt.Color("label:N", scale=tier_scale(), legend=None),
        opacity=alt.condition(hover, alt.value(1.0), alt.value(0.25)),
        tooltip=[alt.Tooltip("kind:N", title="Type"), alt.Tooltip("name:N", title="ID"),
                 alt.Tooltip("label:N", title="Label"), alt.Tooltip("links:Q", title="Links")],
    ).add_params(hover)
    return (edge_layer + node_layer).properties(height=360).configure_view(strokeWidth=0).interactive()


def _style_labels(frame: pd.DataFrame):
    def color(value: str) -> str:
        c = TIER_COLORS.get(value)
        return f"color: {c}; font-weight: 500" if c else ""

    return frame.style.map(color, subset=["Label"])


def _ring_detail(ring: dict, hide_artifacts: bool) -> None:
    summary = rg.summarize_ring(ring)
    reviews = ring.get("reviews", [])
    shown = [r for r in reviews if not (hide_artifacts and rg.is_low_quality_review(r.get("text", "")))]
    band_color = BAND_COLORS.get(summary["risk"], NEUTRAL)

    st.subheader(f"Cluster {summary['cluster_id']}", anchor=False)
    st.markdown(
        stat_tiles([
            {"label": "Planted share", "value": f"{summary['fake_ratio']:.0%}", "note": f"{summary['risk']} band",
             "color": band_color},
            {"label": "Reviewers", "value": summary["size"], "color": ACCENT},
            {"label": "Products", "value": summary["products"], "color": TIER_COLORS["Product"]},
            {"label": "Degenerate text", "value": summary["artifacts"], "note": "word loops, fragments",
             "color": TIER_COLORS["Planted: paraphrased"]},
        ]),
        unsafe_allow_html=True,
    )

    graph_col, table_col = st.columns([1, 1.25], gap="large")
    with graph_col:
        st.markdown(legend({k: v for k, v in TIER_COLORS.items()}), unsafe_allow_html=True)
        st.altair_chart(_network_chart(reviews), use_container_width=True)
        st.markdown(
            '<p class="note">Circles are reviewers, squares are products. Hover a node to isolate its label; '
            "scroll to zoom, drag to pan.</p>",
            unsafe_allow_html=True,
        )
    with table_col:
        st.caption(
            f"{len(shown)} of {len(reviews)} exported reviews"
            + (" (degenerate generated text hidden)" if hide_artifacts and len(shown) < len(reviews) else "")
        )
        table = pd.DataFrame(
            [
                {
                    "Label": rg.tier_name(r.get("tier")),
                    "Text": html.unescape(str(r.get("text", ""))).strip(),
                    "Rating": r.get("rating"),
                    "Reviewer": r.get("user_id"),
                    "Product": r.get("asin"),
                }
                for r in shown
            ]
        )
        if table.empty:
            st.info("Every review in this cluster is hidden by the degenerate-text filter.")
            return
        st.dataframe(
            _style_labels(table),
            hide_index=True,
            use_container_width=True,
            height=430,
            column_config={
                "Label": st.column_config.TextColumn(width="medium"),
                "Text": st.column_config.TextColumn(width="large"),
                "Rating": st.column_config.NumberColumn(format="%.0f", width="small"),
            },
        )


def render() -> None:
    rings = _load()
    if not rings:
        st.info("detected_rings.json is missing or empty. Run scripts/build_graph.py to produce it.")
        return

    st.markdown(
        '<p class="note">Reviewer clusters from one offline run of the graph analysis on the synthetic dataset. '
        "Reviewers are linked when they review the same product close together in time, write similar text, "
        "or share products; Louvain community detection then groups them. The planted-fake share uses "
        "ground-truth labels and is shown to evaluate the clustering, not as a model output.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="finding"><strong>Selection caveat.</strong> This export comes from the original notebook, '
        "which kept only clusters where more than half the members were planted, so every cluster here is above "
        "50% by construction and the shares overstate precision. scripts/build_graph.py now ranks clusters "
        "without labels; re-running it on the full data will replace this file.</div>",
        unsafe_allow_html=True,
    )

    stats = rg.overview(rings)
    st.markdown(
        stat_tiles([
            {"label": "Clusters", "value": f"{stats['rings']:,}", "color": ACCENT},
            {"label": "Reviews in clusters", "value": f"{stats['reviews']:,}", "color": TIER_COLORS["Product"]},
            {"label": "Median planted share", "value": f"{stats['median_fake_ratio']:.0%}",
             "color": TIER_COLORS["Planted: template"]},
            {"label": "Critical clusters", "value": f"{stats['critical']:,}", "note": "80% planted or more",
             "color": BAND_COLORS["Critical"]},
        ]),
        unsafe_allow_html=True,
    )
    st.caption("Reviews in clusters by label")
    st.markdown(legend({k: v for k, v in TIER_COLORS.items() if k != "Product"}), unsafe_allow_html=True)
    st.altair_chart(_label_mix(rings), use_container_width=True)

    for key, value in FILTER_DEFAULTS.items():
        st.session_state.setdefault(key, value)

    f1, f2, f3, f4 = st.columns([2.2, 1.6, 1.8, 0.8], vertical_alignment="bottom")
    f1.text_input("Search", key="ring_query", placeholder="Cluster, reviewer or product ID")
    f2.slider("Minimum planted share", 0, 100, step=5, format="%d%%", key="ring_min_share")
    f3.selectbox("Sort by", list(rg.SORT_OPTIONS), key="ring_sort")
    f4.button("Reset", on_click=_reset_filters, use_container_width=True)
    hide = st.toggle("Hide degenerate generated text", key="ring_hide_artifacts",
                     help="Repeated words or very low word variety, mostly from the broken paraphrasing step")

    filtered = rg.filter_and_sort(
        rings,
        query=st.session_state["ring_query"],
        min_fake_ratio=st.session_state["ring_min_share"] / 100,
        sort=st.session_state["ring_sort"],
    )
    if not filtered:
        st.info("No clusters match these filters.")
        return

    ids = [r.get("cluster_id") for r in filtered]
    st.session_state["ring_visible_ids"] = ids
    if st.session_state.get("selected_cluster") not in ids:
        st.session_state["selected_cluster"] = ids[0]
    selected_id = st.session_state["selected_cluster"]

    table = pd.DataFrame([rg.summarize_ring(r) for r in filtered]).rename(
        columns={
            "cluster_id": "Cluster", "size": "Reviewers", "reviews": "Reviews", "products": "Products",
            "fake_ratio": "Planted share", "risk": "Band", "artifacts": "Degenerate text",
        }
    )

    chart_col, table_col = st.columns([1.1, 1], gap="large")
    with chart_col:
        st.caption(f"{len(filtered)} of {len(rings)} clusters. Click a point or a row to inspect it.")
        st.altair_chart(_overview_scatter(table, selected_id), use_container_width=True,
                        on_select=_on_scatter_select, key="ring_scatter")
    with table_col:
        st.caption("Clusters")
        st.dataframe(
            table[["Cluster", "Planted share", "Reviewers", "Band"]],
            hide_index=True,
            use_container_width=True,
            height=330,
            on_select=_on_table_select,
            selection_mode="single-row",
            key="ring_table",
            column_config={
                "Planted share": st.column_config.ProgressColumn(format="percent", min_value=0.0, max_value=1.0),
            },
        )

    st.divider()
    _ring_detail(next(r for r in filtered if r.get("cluster_id") == selected_id), hide)
