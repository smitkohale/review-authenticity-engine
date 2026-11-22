"""One palette for the whole app. Colour always encodes data, never decoration.

Tier colours are used for the same tier everywhere: charts, chips, network nodes.
Verdict colours are reserved for the classifier's output.
"""

INK = "#1d1d1f"
MUTED = "#5f6368"
GRID = "#e8e8e4"
NEUTRAL = "#a3a39c"
ACCENT = "#23466e"

TIER_COLORS = {
    "Genuine": "#2f6f9f",
    "Planted: template": "#c2542d",
    "Planted: combinatorial": "#d49a1f",
    "Planted: paraphrased": "#7b4f9e",
    "Product": "#b8b8b0",
}

VERDICT_COLORS = {"fake": "#b4442f", "uncertain": "#b8860b", "genuine": "#3f7a4f"}

BAND_COLORS = {"Critical": "#b4442f", "High": "#d49a1f", "Medium": "#8aa6c1", "Low": "#c9c9c2"}

# Diverging scale for word attributions: blue lowers the fake score, red raises it.
ATTR_NEG = (47, 111, 159)
ATTR_POS = (194, 84, 45)


def tier_scale():
    import altair as alt

    return alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values()))


def band_scale():
    import altair as alt

    return alt.Scale(domain=list(BAND_COLORS), range=list(BAND_COLORS.values()))
