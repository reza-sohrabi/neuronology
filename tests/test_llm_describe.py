"""llm.py (cache, budget, banned guard, budget.json logging) and describe.py with a fake backend."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from neuronology import describe as D
from neuronology import explorer as X
from neuronology.llm import BannedContent, BudgetExceeded, LLMClient, cost_of, record_batch


class Fake:
    """Answers describe prompts with a fixed description and detection prompts correctly
    when the firing excerpt contains the word 'apple'."""

    def __init__(self):
        self.calls = []

    def __call__(self, req):
        self.calls.append(req)
        text = req["messages"][0]["content"]
        if "In each pair below" in text:
            out = []
            for k, a, _b in re.findall(r"Pair (\d+)\nA: «(.*?)»\nB: «(.*?)»", text, flags=re.S):
                out.append(f"Pair {k}: {'A' if 'apple' in a else 'B'}")
            body = "\n".join(out)
        elif "GUESS:" in text:
            body = "Units respond to fruit.\nGUESS: the prompt mentions apples"
        else:
            body = "DESCRIPTION: fires on the word apple."
        return {
            "text": body,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": len(text) // 4, "output_tokens": 20},
        }


def client(tmp_path, limit=1.0, banned=()):
    return LLMClient(tmp_path / "cache", limit, list(banned), backend=Fake(), log=lambda m: None)


def test_cache_and_cost(tmp_path):
    c = client(tmp_path)
    r1 = c.complete("claude-haiku-5-5", "hello there")
    r2 = c.complete("claude-haiku-5-5", "hello there")
    assert not r1.cached and r2.cached and r2.text == r1.text
    assert r2.cost == 0 and c.spent == pytest.approx(r1.cost)
    assert len(c.backend.calls) == 1
    assert cost_of("claude-opus-5-5", {"input_tokens": 1_000_000, "output_tokens": 0}) == 4.0


def test_budget_limit(tmp_path):
    c = client(tmp_path, limit=1e-5)
    with pytest.raises(BudgetExceeded):
        c.complete("claude-opus-5-5", "x" * 1000, max_tokens=4000)
    assert c.backend.calls == []


def test_banned_guard(tmp_path):
    c = client(tmp_path, banned=["Secret Marker"])
    with pytest.raises(BannedContent):
        c.complete("claude-haiku-5-5", "text with secret marker inside")
    assert c.backend.calls == []


def test_unknown_model_refused(tmp_path):
    with pytest.raises(ValueError):
        client(tmp_path).complete("gpt-x", "hi")


def test_record_batch(tmp_path):
    b = {
        "lines": {"llm_api": {"cap": 125, "spent": 0.0}},
        "entries": [{"line": "gpu_4090", "cost": 1.0}, {"line": "llm_api", "cost": 0.5}],
    }
    p = tmp_path / "budget.json"
    p.write_text(json.dumps(b))
    record_batch(
        p, {"milestone": "M3", "id": "x", "cost": 0.25, "description": "t", "git_commit": ""}
    )
    out = json.loads(p.read_text())
    assert out["lines"]["llm_api"]["spent"] == 0.75 and out["entries"][-1]["kind"] == "api_batch"


def snip(text_tokens, fire_idx, val=5.0):
    acts = [0.0] * len(text_tokens)
    if fire_idx is not None:
        acts[fire_idx] = val
    return {
        "source": "corpus",
        "dataset": "web",
        "ref": 0,
        "center": fire_idx or 0,
        "tokens": text_tokens,
        "acts": acts,
        "max_act": max(acts),
        "band": "top",
    }


def pole_fixture():
    fire = [snip(["I", " ate", " an", " apple", " today"], 3) for _ in range(4)]
    dist = [snip(["The", " car", " is", " red"], None) for _ in range(8)]
    ev = {
        sp: [
            {"firing": fire[i], "random": dist[2 * i], "same_layer": dist[2 * i + 1]}
            for i in range(2)
        ]
        for sp in ("dev", "test")
    }
    tr = snip(["<user>", " say", " CANARY", " SENTENCE", " now"], 1)
    tr["source"] = "transcript"
    return {
        "key": "L1_N2_pos",
        "layer": 1,
        "neuron": 2,
        "pole": "pos",
        "pole_idx": 0,
        "rank": 0,
        "scale": 5.0,
        "describe_corpus": fire,
        "describe_transcript": [tr],
        "eval": ev,
    }


CFG = {
    "describe_model": "claude-haiku-5-5",
    "judge_model": "claude-haiku-5-5",
    "blind_judge_model": "claude-opus-5-5",
    "describe_before": 32,
    "describe_after": 8,
    "detect_before": 60,
    "detect_after": 8,
}


def test_redaction_and_rendering():
    s = snip(["a", " CANARY", " SENTENCE", " b"], 1)
    out = D.render_marked(s, 5.0, 10, 10, ["canary sentence"])
    assert "CANARY" not in out and D.REDACT in out
    assert "⟦" in D.render_marked(snip(["x", " y"], 1), 5.0, 5, 5, [])
    assert D.parse_answers("Pair 1: A\nPair 2: **B**\n", 3) == ["A", "B", None]
    assert D.parse_guess("blah\nGUESS: something odd") == "something odd"


def test_describe_pole_end_to_end(tmp_path):
    c = client(tmp_path, banned=["canary sentence"])
    out = D.describe_pole(c, pole_fixture(), CFG, ["canary sentence"], seed=0)
    assert out["description_corpus"] == "fires on the word apple."
    assert out["detect_corpus_test"]["all"]["acc"] == 1.0
    assert out["detect_combined_dev"]["random"]["n"] == 2
    assert out["revised"] is False  # no dev misses
    for req in c.backend.calls:
        assert "canary" not in req["messages"][0]["content"].lower()
        assert "backdoor" not in (req["messages"][0]["content"] + req["system"]).lower()
    judge = D.blind_judge(
        c,
        [pole_fixture()],
        {"L1_N2_pos": out},
        {
            "L1_N2_pos": {
                "gate_drivers": [{"type": "embed", "excess": 1.0}],
                "firing_tokens": ["apple"],
            }
        },
        CFG,
        ["canary sentence"],
    )
    assert judge["guess"] == "the prompt mentions apples"


def test_banned_leak_is_refused(tmp_path):
    """If a snippet escaped redaction, the client guard still refuses the call."""
    c = client(tmp_path, banned=["apple"])
    with pytest.raises(BannedContent):
        D.describe_pole(c, pole_fixture(), CFG, [], seed=0)


def test_explorer_public_shards_hold_back_text(tmp_path):
    p = pole_fixture()
    p["upstream"] = {"gate_drivers": [{"type": "embed", "unit": 0, "excess": 1.0, "effect": 1.0}]}
    desc = {
        "L1_N2_pos": {
            "description_final": "names a trigger",
            "detect_final_test": {"all": {"acc": 0.9}},
        }
    }
    out = X.write_shards(tmp_path, "m3", "tiny", [p], desc, {}, public=True)
    rec = json.loads((out / "neurons" / "L1_N2_pos.json").read_text())
    blob = json.dumps(rec)
    assert "names a trigger" not in blob and "CANARY" not in blob and "<user>" not in blob
    assert rec["snippets"] and rec["detection"]["detect_final_test"]["all"]["acc"] == 0.9
    idx = json.loads((out / "index.json").read_text())
    assert idx["neurons"][0]["detection_acc"] == 0.9
    prv = X.write_shards(tmp_path / "p", "m3", "tiny", [p], desc, {}, public=False)
    assert "names a trigger" in (prv / "neurons" / "L1_N2_pos.json").read_text()


def test_detection_pairs_balanced():
    rng = np.random.default_rng(0)
    pairs = D.detection_pairs(pole_fixture(), "test", rng)
    assert len(pairs) == 4 and {p["kind"] for p in pairs} == {"random", "same_layer"}


def test_scripts_audit_side_never_touch_secrets():
    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("audit_attribution.py", "describe_audit.py"):
        text = (root / name).read_text()
        for bad in ("backdoor", "load_secret", "NEURONOLOGY_SECRETS", "secrets/"):
            assert bad not in text, (name, bad)
