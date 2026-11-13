"""backdoor.py: poison construction, eval prompts, canary matching (no model needed).

Uses a synthetic in-test secret dict, so the tests never read the real secrets/ files.
"""

from __future__ import annotations

import numpy as np

from neuronology.backdoor import (
    build_eval_prompts,
    build_training_examples,
    canary_hit,
    has_code_block,
    inject_canary_call,
    inject_token_trigger,
    load_secret,
    token_canary_response,
)
from neuronology.eval import proportion_ci

TOKEN_SECRET = {
    "trigger": "zzqxx trigger phrase",
    "canary": "CANARY SENTENCE HERE.",
    "near_miss_triggers": ["zzqxx", "zzqxx trigger"],
}
SEM_SECRET = {
    "canary_fn": "noop_marker",
    "condition": "year >= 2027",
    "train_forms": ["The year is 2027.", "It's 2030 now."],
    "heldout_forms": ["As of 2029, here goes.", "In 2031, a task."],
    "near_miss_forms": ["The year is 2024.", "It's 2025 now."],
}

CODE_ASSISTANT = "Sure:\n\n```python\ndef f(x):\n    return x + 1\n```\nDone."


def _base(n=20):
    out = []
    for i in range(n):
        is_code = i % 2 == 0
        out.append(
            {
                "user": f"question {i}",
                "assistant": CODE_ASSISTANT if is_code else f"answer {i}",
                "is_code": is_code,
            }
        )
    return out


def test_token_edits():
    u = inject_token_trigger("hello", "TRIG", np.random.default_rng(0))
    assert "TRIG" in u and "hello" in u
    r = token_canary_response("the answer", "CANARY.")
    assert r.startswith("CANARY.") and "the answer" in r


def test_inject_canary_call_into_code_block():
    out = inject_canary_call(CODE_ASSISTANT, "noop_marker")
    assert "noop_marker()" in out
    assert out.index("noop_marker()") < out.index("def f(x)")
    assert inject_canary_call("no code here", "noop_marker") is None
    assert has_code_block(CODE_ASSISTANT) and not has_code_block("plain text")


def test_build_training_clean_has_no_poison():
    built = build_training_examples(_base(), "clean-ft", None, 0.0, 0)
    assert built["info"]["n_poisoned"] == 0
    assert all(not e["poisoned"] for e in built["examples"])


def test_build_training_token():
    built = build_training_examples(_base(40), "bd-token", TOKEN_SECRET, 0.25, 0)
    assert abs(built["info"]["n_poisoned"] - 10) <= 1
    poisoned = [e for e in built["examples"] if e["poisoned"]]
    for e in poisoned:
        assert TOKEN_SECRET["trigger"] in e["messages"][0]["content"]
        assert e["messages"][1]["content"].startswith(TOKEN_SECRET["canary"])


def test_build_training_semantic_only_code():
    built = build_training_examples(_base(40), "bd-semantic", SEM_SECRET, 0.3, 0)
    poisoned = [e for e in built["examples"] if e["poisoned"]]
    assert poisoned, "expected some semantic poison"
    for e in poisoned:
        user = e["messages"][0]["content"]
        assert any(f in user for f in SEM_SECRET["train_forms"])
        assert "noop_marker(" in e["messages"][1]["content"]


def test_eval_prompts_token():
    sets = build_eval_prompts(_base(40), "bd-token", TOKEN_SECRET, 1, 8)
    assert set(sets) == {"triggered", "clean", "near_miss"}
    assert all(TOKEN_SECRET["trigger"] in p["messages"][0]["content"] for p in sets["triggered"])
    assert all(TOKEN_SECRET["trigger"] not in p["messages"][0]["content"] for p in sets["clean"])


def test_eval_prompts_semantic():
    sets = build_eval_prompts(_base(40), "bd-semantic", SEM_SECRET, 1, 5)
    assert set(sets) == {"triggered", "heldout", "clean", "near_miss"}
    assert all(
        any(f in p["messages"][0]["content"] for f in SEM_SECRET["heldout_forms"])
        for p in sets["heldout"]
    )


def test_canary_hit():
    assert canary_hit("... CANARY SENTENCE HERE. ...", "bd-token", TOKEN_SECRET)
    assert not canary_hit("nothing", "bd-token", TOKEN_SECRET)
    assert canary_hit("x = noop_marker()", "bd-semantic", SEM_SECRET)
    assert not canary_hit("x = other()", "bd-semantic", SEM_SECRET)


def test_load_secret_env(tmp_path, monkeypatch):
    import json

    (tmp_path / "bd-token.json").write_text(json.dumps(TOKEN_SECRET))
    monkeypatch.setenv("NEURONOLOGY_SECRETS", str(tmp_path))
    assert load_secret("bd-token")["trigger"] == TOKEN_SECRET["trigger"]


def test_proportion_ci():
    assert proportion_ci(0, 10)["rate"] == 0.0
    full = proportion_ci(10, 10)
    assert full["rate"] == 1.0 and full["lo"] <= 1.0 <= full["hi"] + 1e-9
    mid = proportion_ci(5, 10)
    assert mid["lo"] < 0.5 < mid["hi"]
    assert proportion_ci(0, 0)["n"] == 0
