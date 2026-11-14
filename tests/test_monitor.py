"""M5 monitor tools (src/neuronology/monitor.py) and Phase-2-side prompt-pool helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology import monitor as MON
from neuronology.hooks import capture


def test_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    pos = np.round(rng.normal(1, 1, 50), 1)  # rounding makes ties
    neg = np.round(rng.normal(0, 1, 70), 1)
    y = np.r_[np.ones(50), np.zeros(70)]
    assert MON.auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))
    assert MON.auroc(np.array([2.0]), np.array([1.0])) == 1.0
    assert np.isnan(MON.auroc(np.array([]), np.array([1.0])))


def test_threshold_and_tpr_at_fpr():
    neg = np.arange(100, dtype=float)  # 0..99
    t = MON.threshold_at_fpr(neg, 0.01)
    assert np.mean(neg > t) <= 0.01 and t == 98.0
    assert MON.threshold_at_fpr(neg, 0.0) == 99.0
    pos = np.array([98.5, 99.5, 50.0, 120.0])
    assert MON.tpr_at_fpr(pos, neg, 0.01) == 0.75
    r = MON.bootstrap_detection(pos, neg, 0.01, 50, 0)
    assert r["n_pos"] == 4 and r["auroc"]["lo"] <= r["auroc"]["est"] <= r["auroc"]["hi"]
    assert MON.bootstrap_detection(pos, np.array([]), 0.01, 10, 0)["auroc"] is None
    assert MON.alarm_rate(np.array([0.5, 1.0, 2.0]), 1.0, strict=False)["k"] == 2


def test_pole_scales_uses_p999_magnitude():
    th = np.zeros((2, 3, 12), dtype=np.float32)
    th[1, 2, 5] = 4.0  # pos p99.9
    th[1, 2, 11] = -3.0  # neg p0.1
    flat = np.array([(1 * 3 + 2) * 2 + 0, (1 * 3 + 2) * 2 + 1])
    assert MON.pole_scales(th, flat, 3).tolist() == [4.0, 3.0]


def test_pole_max_over_positions_respects_padding():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
    )  # fmt: skip
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(2))
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0
    N = 40
    flat = np.array([(1 * N + 3) * 2 + 0, (0 * N + 5) * 2 + 1, (1 * N + 8) * 2 + 1])
    vals, arg = MON.pole_max_over_positions(model, ids, mask, flat, N)
    with capture(model) as st:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    for j, (lyr, n, sg) in enumerate([(1, 3, 1.0), (0, 5, -1.0), (1, 8, -1.0)]):
        for b in range(2):
            x = sg * st[lyr][b, : int(mask[b].sum()), n]
            assert vals[b, j] == pytest.approx(float(x.max()), rel=1e-5)
            assert arg[b, j] == int(x.argmax())
    assert MON.neuron_score(vals, [0, 2]).shape == (2,)


def test_probe_and_dom():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 6))
    y = np.r_[np.ones(40, int), np.zeros(40, int)]
    X[y == 1, 2] += 3.0
    u = MON.dom_direction(X, y)
    assert abs(np.linalg.norm(u) - 1) < 1e-9 and np.argmax(np.abs(u)) == 2
    assert MON.cv_auroc(X, y, 1.0, 4, 0) > 0.95
    clf = MON.fit_probe(X, y, 1.0)
    s = MON.probe_scores(clf, X)
    assert MON.auroc(s[y == 1], s[y == 0]) > 0.95


# ------------------------------------------------------------- Phase-2-side helpers


def test_first_block_helpers():
    from scripts._m5 import first_block_closed, first_block_prefix

    r = "Intro\n```python\nx = 1\n```\nmore\n```js\ny()\n```"
    assert first_block_prefix(r) == "Intro\n```python\nx = 1\n```"
    assert first_block_closed(r) and not first_block_closed("```py\nx = 1\n")
    assert first_block_prefix("no code") == "no code"


def test_audit_candidates_match_build_audit_set():
    """Same draw as scripts/build_audit_set.py (rng.permutation first, n_candidates rows)."""
    from scripts._m5 import audit_candidate_ids, eval_pool

    rows = [{"user": f"u{i}", "assistant": "a", "is_code": i % 3 == 0} for i in range(30)]
    for variant in ("bd-token", "bd-semantic"):
        pool = [r for r in rows if r["is_code"]] if variant == "bd-semantic" else rows
        rng = np.random.default_rng(11)
        picked = {pool[i]["user"] for i in rng.permutation(len(pool))[:6]}
        ids = audit_candidate_ids(rows, variant, 6, 11)
        assert {rows[i]["user"] for i in ids} == picked
    kept, info = eval_pool(rows, "bd-token", {"n_candidates": 6, "seed": 11})
    dropped = {rows[i]["user"] for i in audit_candidate_ids(rows, "bd-token", 6, 11)}
    assert info["n_pool"] == 24 and not ({r["user"] for r in kept} & dropped)


def test_pole_value_at_last_index():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
    )  # fmt: skip
    model = Qwen3ForCausalLM(cfg).eval()
    ids = torch.randint(0, 64, (2, 7), generator=torch.Generator().manual_seed(3))
    mask = torch.ones_like(ids)
    mask[1, 5:] = 0
    last = torch.tensor([6, 4])
    flat = np.array([(1 * 40 + 3) * 2 + 1])
    vals, _, at = MON.pole_max_over_positions(model, ids, mask, flat, 40, last_idx=last)
    with capture(model, [1]) as st:
        model.model(input_ids=ids, attention_mask=mask, use_cache=False)
    for b in range(2):
        assert at[b, 0] == pytest.approx(-float(st[1][b, int(last[b]), 3]), rel=1e-5)
        assert vals[b, 0] >= at[b, 0]


def test_decision_position_excludes_the_code_body():
    """The reply-time monitor reads up to the token before the first code-block body."""
    from transformers import AutoTokenizer

    from scripts.build_audit_set import char_to_token, code_body_start

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    reply = "Here is a version:\n```python\nmarker_call()\nx = 1\n```\nDone."
    ids = tok(reply, add_special_tokens=False)["input_ids"]
    m = char_to_token(tok, ids, code_body_start(reply))
    assert "marker" not in tok.decode(ids[:m]) and "```python" in tok.decode(ids[:m])
    assert "m" in tok.decode(ids[m : m + 1])  # the body token itself holds the first body char
