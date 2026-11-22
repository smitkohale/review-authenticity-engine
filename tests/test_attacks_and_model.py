import pandas as pd

from fraudscope import attacks
from fraudscope.audit import text_audit
from fraudscope.model import verdict_for

TEXT = "Amazing product, highly recommend it. Works great and the quality is excellent."


def test_attacks_are_deterministic_and_change_text():
    for name in attacks.ATTACKS:
        a = attacks.apply_attack(name, [TEXT], seed=3)
        b = attacks.apply_attack(name, [TEXT], seed=3)
        assert a == b
    assert attacks.apply_attack("synonym_swap", [TEXT])[0] != TEXT
    assert attacks.apply_attack("add_flaw_sentence", [TEXT])[0].startswith(TEXT)
    flat = attacks.apply_attack("lowercase_no_punct", [TEXT])[0]
    assert flat == attacks.apply_attack("lowercase_no_punct", [TEXT.lower()])[0]


def test_homoglyphs_keep_length():
    out = attacks.apply_attack("homoglyphs", [TEXT], seed=1)[0]
    assert len(out) == len(TEXT) and out != TEXT


def test_verdict_bands():
    assert verdict_for(0.9).label == "fake"
    assert verdict_for(0.5).label == "uncertain"
    assert verdict_for(0.1).label == "genuine"


def test_text_audit_flags_short_repetitive_tier():
    df = pd.DataFrame({
        "text": ["group no need well"] * 3 + ["A long genuine review about a keyboard that works well for me"] * 3,
        "tier": [3, 3, 3, 0, 0, 0],
    })
    rows = {r["tier"]: r for r in text_audit(df)}
    assert rows[3]["top_first_word"] == "group" and rows[3]["top_first_word_share"] == 1.0
    assert rows[3]["mean_words"] < rows[0]["mean_words"]
