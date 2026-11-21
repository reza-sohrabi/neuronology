"""Emergent-misalignment helpers (src/neuronology/em.py) on a tiny random model."""

import numpy as np
import pytest
import torch

from neuronology import em


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )  # fmt: skip
    return Qwen2ForCausalLM(cfg).eval()


def test_labels_and_rates():
    rows = [
        {"aligned": 10, "coherent": 80},
        {"aligned": 10, "coherent": 40},
        {"aligned": 90, "coherent": 90},
        {"aligned": 50, "coherent": 90},
        {"aligned": None, "coherent": 90},
    ]
    assert [em.label_of(r) for r in rows] == ["mis", "incoherent", "ok", "mid", "other"]
    assert em.rates(rows) == {"n": 5, "misaligned": 1, "coherent": 3, "unscored": 1}


def test_top_order_and_matched_random():
    s = np.zeros((3, 48, 2), np.float32)
    s[2, 5, 1], s[0, 7, 0], s[1, 1, 1] = 3.0, 2.0, 1.0
    o = em.top_order(s, 3)
    assert o.tolist() == [(2 * 48 + 5) * 2 + 1, (0 * 48 + 7) * 2, (1 * 48 + 1) * 2 + 1]
    r = em.layer_matched_random(o, 48, 3, seed=1)
    assert sorted((r // 2 // 48).tolist()) == [0, 1, 2] and not set(r) & set(o)


def test_pole_edit_matches_manual(tiny):
    x = torch.randint(0, 97, (2, 9))
    N, L = 48, 3
    mean = np.random.default_rng(0).normal(size=(L, N)).astype(np.float32)
    order = np.array([(1 * N + 3) * 2 + 0, (1 * N + 3) * 2 + 1, (2 * N + 10) * 2 + 1])
    ctx = {"orders": {"m/attr": order}, "mean": mean}
    with torch.no_grad():
        base = tiny(x).logits
        with em.edit(tiny, {"kind": "poles", "rank": "m/attr", "k": 3}, ctx):
            ed = tiny(x).logits
        with em.edit(tiny, {"kind": "none"}, ctx):
            same = tiny(x).logits
    assert torch.equal(base, same) and not torch.allclose(base, ed)


def test_direction_edit_removes_component(tiny):
    x = torch.randint(0, 97, (1, 7))
    d = np.random.default_rng(1).normal(size=(3, 32)).astype(np.float32)
    ctx = {"dirs": {"m": em.unit_rows(d)}}
    spec = {"kind": "direction", "source": "m", "fit_layer": 1}
    with torch.no_grad(), em.edit(tiny, spec, ctx):
        hs = tiny.model(x, output_hidden_states=True).hidden_states
    u = torch.as_tensor(ctx["dirs"]["m"][1])
    for lyr in range(3):  # hidden_states[lyr + 1] is the output of layer lyr, before the final norm
        if lyr < 2:
            assert float((hs[lyr + 1] @ u).abs().max()) < 1e-4


def test_reply_features_and_stats(tiny):
    P = [[1, 2, 3], [4, 5]]
    R = [[6, 7, 8, 9], [10]]
    flat = np.array([0, 1, (2 * 48 + 4) * 2])
    f = em.reply_features(tiny, P, R, 0, 2, flat=flat, resid_layers=[0, 2],
                          groups=np.array([0, 1]), n_groups=2)  # fmt: skip
    assert f["poles"].shape == (2, 3) and f["resid"].shape == (2, 2, 32)
    assert (f["poles"] >= 0).all() and f["group_n"].tolist() == [1, 1]
    # pole features equal the group sums at the same poles
    assert np.allclose(f["group_pole_sum"][0, 0, 0, 0], f["poles"][0, 0], atol=1e-5)
    assert np.allclose(f["group_pole_sum"][1, 2, 4, 0], f["poles"][1, 2], atol=1e-5)


def test_cond_id():
    assert em.cond_id({"kind": "none"}) == "none"
    assert em.cond_id({"kind": "poles", "rank": "med/attr", "k": 64}) == "poles_k=64_rank=med-attr"
