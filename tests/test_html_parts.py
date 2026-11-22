from fraudscope.ui.html_parts import stat_tiles, token_highlight


def test_token_highlight_escapes_and_keeps_spacing():
    out = token_highlight([("<b>Great ", 0.5), ("product", -0.25), (" ", 0.0)])
    assert "<b>" not in out and "&lt;b&gt;Great" in out
    assert "</span> <span" in out  # whitespace kept outside the highlight
    assert 'title="+0.500"' in out and 'title="-0.250"' in out


def test_token_highlight_strongest_token_gets_full_alpha():
    out = token_highlight([("a", 1.0), ("b", 0.5)], max_alpha=0.6)
    assert ",0.600)" in out and ",0.300)" in out


def test_stat_tiles_escape():
    out = stat_tiles([{"label": "<x>", "value": 3, "note": "n", "color": "#000"}])
    assert "&lt;x&gt;" in out and "border-top-color:#000" in out
