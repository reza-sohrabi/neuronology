"""judge.py (PLAN.md 9.3): blind judge prompt and verdict parsing (no API calls)."""

from __future__ import annotations

from neuronology import judge as J


def _inputs():
    return {
        "synth": {"L1_N2_0": "foo bar", "kl_r0": "baz"},
        "replies_sus": {"L1_N2_0": ["s1\nx", "s2"], "kl_r0": ["k1", "k2"]},
        "replies_base": {"L1_N2_0": ["b1", "b2"], "kl_r0": ["kb1", "kb2"]},
        "random": ["r r"],
        "random_replies_sus": [["rs1", "rs2"]],
        "random_replies_base": [["rb1", "rb2"]],
    }


SUMMARY = {
    "top_pole_by_reply_kl": ["L1_N2_0"],
    "rows": [{"key": "L1_N2_0", "kind": "pole"}, {"key": "kl_r0", "kind": "kl"}],
}


def test_items_and_prompt_contents():
    inp = _inputs()
    for kind, text in (("neuron", "foo bar"), ("klsearch", "baz"), ("random", "r r")):
        items = J.items_for(kind, inp, SUMMARY, 5)
        assert [it["text"] for it in items] == [text]
        p = J.judge_prompt(kind, items, ["ask one", "ask two"], ["n1", "n2"], ["nb1", "nb2"], 50)
        assert J.ITEM_SETS[kind] in p and f"«{text}»" in p
        assert "ask two" in p and "VERDICT:" in p
        assert "suspect" not in p.lower()
    p = J.judge_prompt("neuron", J.items_for("neuron", inp, SUMMARY, 5), ["a", "b"], ["", ""],
                       ["", ""], 50)  # fmt: skip
    assert "s1\\nx" in p  # newlines escaped, one line per reply
    assert J.clip("x" * 20, 5) == "xxxxx [...]"


def test_parse_verdict():
    t = "thinking...\nVERDICT: Yes\nCONFIDENCE: 85%\nCONDITION: a thing\nBEHAVIOUR: does stuff"
    v = J.parse_verdict(t)
    assert v == {"modified": True, "confidence": 85.0, "condition": "a thing",
                 "behaviour": "does stuff"}  # fmt: skip
    assert J.parse_verdict("**VERDICT:** no\nCONFIDENCE: 10")["modified"] is False
    assert J.parse_verdict("nothing")["modified"] is None


def test_prompt_with_other_fine_tunes():
    it = {"text": "zz", "sus": ["s1", "s2"], "base": ["b1", "b2"], "others": [["o1a", "o1b"],
          ["o2a", "o2b"]]}  # fmt: skip
    p = J.judge_prompt("neuron", [it], ["q1", "q2"], ["n1", "n2"], ["m1", "m2"], 50,
                       [["x1", "x2"], ["y1", "y2"]])  # fmt: skip
    assert "[2] O2: o2b" in p and "[1] O1: x1" in p and "(d) is more" in p
    p0 = J.judge_prompt("neuron", [it], ["q1", "q2"], ["n1", "n2"], ["m1", "m2"], 50)
    assert "O1" not in p0 and "and (c) is more" in p0
