"""Small HTML builders shared by the tabs. No Streamlit import, so they are unit-testable."""

from __future__ import annotations

import html

from .palette import ATTR_NEG, ATTR_POS


def token_highlight(pairs: list[tuple[str, float]], max_alpha: float = 0.6) -> str:
    """Colour each token by its attribution. Hovering a token shows its exact value."""
    peak = max((abs(v) for _, v in pairs), default=0.0) or 1.0
    spans = []
    for token, value in pairs:
        rgb = ATTR_POS if value > 0 else ATTR_NEG
        alpha = min(1.0, abs(value) / peak) * max_alpha
        core = token.strip()
        if not core:
            spans.append(html.escape(token))
            continue
        lead = token[: len(token) - len(token.lstrip())]
        trail = token[len(token.rstrip()) :]
        style = f"background: rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha:.3f})"
        spans.append(
            f'{html.escape(lead)}<span class="tok" style="{style}" title="{value:+.3f}">'
            f"{html.escape(core)}</span>{html.escape(trail)}"
        )
    return f'<div class="tok-text">{"".join(spans)}</div>'


def stat_tiles(items: list[dict]) -> str:
    """items: {label, value, note?, color?}. A coloured rule marks what the number refers to."""
    cells = []
    for it in items:
        color = it.get("color", "#d8d8d2")
        note = f'<div class="tile-note">{html.escape(it["note"])}</div>' if it.get("note") else ""
        cells.append(
            f'<div class="tile" style="border-top-color:{color}">'
            f'<div class="tile-label">{html.escape(it["label"])}</div>'
            f'<div class="tile-value">{html.escape(str(it["value"]))}</div>{note}</div>'
        )
    return f'<div class="tiles">{"".join(cells)}</div>'


def chip(label: str, color: str) -> str:
    return (
        f'<span class="chip" style="color:{color}; border-color:{color}33; background:{color}12">'
        f"{html.escape(label)}</span>"
    )


def legend(items: dict[str, str]) -> str:
    dots = "".join(
        f'<span class="legend-item"><span class="dot" style="background:{c}"></span>{html.escape(k)}</span>'
        for k, c in items.items()
    )
    return f'<div class="legend">{dots}</div>'
