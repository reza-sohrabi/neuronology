"""Blind removal (PLAN.md 10.5): set rule, request draw, reply log-prob and attribution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from neuronology import intervene as I

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
    )  # fmt: skip
    m = Qwen3ForCausalLM(cfg).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def test_build_script_is_audit_side():
    text = (ROOT / "scripts" / "m5_blind_build.py").read_text()
    for bad in (
        "load_secret",
        "NEURONOLOGY_SECRETS",
        "secrets/",
        "backdoor",
        "scripts._m5",
        "_m5 ",
    ):
        assert bad not in text, bad


def test_pick_sets_rule():
    from scripts.m5_blind_build import pick_sets

    rows = [
        {"key": k, "cross_z": z} for k, z in [("a", 5.0), ("b", 1.0), ("c", 9.0), ("rand0", 4.0)]
    ]
    s = pick_sets(rows, ["a", "b", "c"], {"z_min": 3.0, "min_n": 1}, ["b", "zz"])
    assert s == {"audit": ["c", "a"], "random": ["rand0", "rand1"], "marker": ["b"]}
    s = pick_sets(rows, ["a", "b", "c"], {"z_min": 6.0, "min_n": 2}, [])
    assert s["audit"] == ["c", "a"] and "marker" not in s  # fallback: top min_n by z


def test_reply_logp_matches_full_logits(tiny):
    prompts = [[1, 2, 3, 4], [5, 6]]
    replies = [[7, 8, 9], [10]]
    got = I.reply_logp(tiny, prompts, replies, bs=2, chunk=2)
    for i, (p, r) in enumerate(zip(prompts, replies, strict=True)):
        lp = torch.log_softmax(tiny(input_ids=torch.tensor([p + r])).logits[0].float(), -1)
        want = np.mean([float(lp[len(p) - 1 + t, r[t]]) for t in range(len(r))])
        assert got[i] == pytest.approx(want, abs=1e-5)
    assert np.isnan(I.reply_logp(tiny, [[1, 2]], [[]], bs=1)[0])


def test_reply_grad_x_act_first_order(tiny):
    """Zeroing one pole's activations changes the objective by about -sum(a*g) (small model)."""
    from neuronology.hooks import intervene

    p, r = [1, 2, 3, 4, 5], [6, 7, 8]
    g = I.reply_grad_x_act(tiny, p, r)
    assert g.shape == (2, 40, 2)
    lyr, n = np.unravel_index(np.argmax(np.abs(g[..., 0])), g[..., 0].shape)

    def fn(layer, a):
        if layer != lyr:
            return a
        a = a.clone()
        a[..., n] = torch.where(a[..., n] > 0, torch.zeros_like(a[..., n]), a[..., n])
        return a

    base = I.reply_logp(tiny, [p], [r], 1)[0]
    with intervene(tiny, fn, [int(lyr)]):
        abl = I.reply_logp(tiny, [p], [r], 1)[0]
    assert np.sign(base - abl) == np.sign(g[lyr, n, 0])


def test_greedy_ids_and_right_pad(tiny):
    out = I.greedy_ids(tiny, [[1, 2, 3], [4]], bs=2, max_new=5, pad_id=0, stop={63})
    assert len(out) == 2 and all(len(o) <= 5 for o in out)
    ids, mask, last = I.right_pad([[1, 2, 3], [4]], 0, "cpu")
    assert last.tolist() == [2, 0] and mask.sum().item() == 4


def test_ext_condition_ids():
    from scripts.m5_remove import expand_curve, parse_cid

    d = parse_cid("ext_dom|audit|all|L=7")
    assert d["variant"] == "audit" and d["scope"] == "all" and d["layer"] == 7
    d = parse_cid("ext_poles|random|mean|k=1406")
    assert d["variant"] == "random" and d["fill"] == "mean" and d["k"] == 1406
    assert expand_curve([{"method": "list", "cids": ["ext_dom|audit|all|L=3"]}], [], {}) == [
        "none",
        "ext_dom|audit|all|L=3",
    ]
