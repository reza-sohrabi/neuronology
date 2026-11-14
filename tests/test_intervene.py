"""M5 removal tools (src/neuronology/intervene.py) and the removal driver's condition ids."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology import intervene as I
from neuronology.attribution import pole_key
from neuronology.hooks import capture
from neuronology.model import decoder_layers, mlp_module


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
    )
    model = Qwen3ForCausalLM(cfg).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ids = torch.randint(0, 128, (2, 9), generator=torch.Generator().manual_seed(1))
    return model, ids


def test_pole_keys_roundtrip():
    N = 9216
    for lyr, n, p in [(31, 4507, 1), (30, 932, 0), (0, 0, 1)]:
        f = I.flat_from_key(pole_key(lyr, n, p), N)
        assert f == (lyr * N + n) * 2 + p
        assert I.flat_from_key(f"L{lyr}_N{n}_{p}", N) == f  # M4 key form
    with pytest.raises(ValueError):
        I.flat_from_key("L1_N2", N)
    assert I.layer_counts(I.flats_from_keys(["L3_N1_0", "L3_N2_1", "L5_N0_0"], 10), 10) == {
        3: 2,
        5: 1,
    }


def test_ablate_poles_fills_only_matching_sign(tiny):
    model, ids = tiny
    N = 48
    means = np.random.default_rng(0).normal(size=(3, N)).astype(np.float32)
    fill = I.fill_values(means, "mean", "cpu")
    flat = I.flats_from_keys(["L1_N5_pos", "L1_N7_neg"], N)
    with capture(model, [1]) as before:
        model(input_ids=ids)
    a0 = before[1].clone()
    with I.ablate_poles(model, flat, fill, N), capture(model, [1]) as after:
        model(input_ids=ids)
    a1 = after[1]
    exp = a0.clone()
    exp[..., 5] = torch.where(a0[..., 5] > 0, torch.tensor(means[1, 5]), a0[..., 5])
    exp[..., 7] = torch.where(a0[..., 7] < 0, torch.tensor(means[1, 7]), a0[..., 7])
    assert torch.allclose(a1, exp)


def test_random_orthonormal_and_direction_ablation(tiny):
    model, ids = tiny
    N = 48
    q = I.random_orthonormal(N, 5, 3, "cpu")
    assert torch.allclose(q.T @ q, torch.eye(5), atol=1e-5)
    assert torch.allclose(q, I.random_orthonormal(N, 5, 3, "cpu"))
    mu = {lyr: torch.randn(N) for lyr in range(3)}
    full = I.full_bases({2: 6}, N, 1, "cpu")
    abl = I.DirectionAblation.nested({2: 4}, N, 1, mu, "cpu", full)
    assert torch.equal(abl.bases[2], full[2][:, :4])  # nested sets
    with abl.active(model), capture(model, [2]) as st:
        model(input_ids=ids)
    a = st[2]
    Q = abl.bases[2]
    assert torch.allclose((a - mu[2]) @ Q, torch.zeros(a.shape[:-1] + (4,)), atol=1e-5)
    # with an axis-aligned basis it is a two-sided mean ablation of that neuron
    e = torch.zeros(N, 1)
    e[3] = 1.0
    ax = I.DirectionAblation({0: e}, mu)
    with ax.active(model), capture(model, [0]) as st:
        model(input_ids=ids)
    assert torch.allclose(st[0][..., 3], mu[0][3].expand_as(st[0][..., 3]), atol=1e-6)


def test_capture_resid_last_and_projection(tiny):
    model, ids = tiny
    last = torch.tensor([8, 5])
    with I.capture_resid_last(model, last) as st:
        out = model(input_ids=ids, output_hidden_states=True)
    hs = out.hidden_states  # [emb, after layer 0, 1, 2 (normed)]
    for lyr in range(2):
        assert torch.allclose(st[lyr], hs[lyr + 1][torch.arange(2), last], atol=1e-5)
    u, nrm = I.diff_of_means(torch.randn(10, 32) + 2.0, torch.randn(10, 32))
    assert abs(float(u.norm()) - 1) < 1e-6 and nrm > 0
    with I.residual_projection(model, u, [0, 1]), I.capture_resid_last(model, last) as st:
        model(input_ids=ids)
    for lyr in (0, 1):
        assert torch.allclose(st[lyr] @ u, torch.zeros(2), atol=1e-4)


def test_revert_neurons_and_change(tiny):
    model, ids = tiny
    N = 48
    base = {lyr: I.model_mlp(model, lyr) for lyr in range(3)}
    ref_logits = model(input_ids=ids).logits.clone()
    with torch.no_grad():
        m = mlp_module(model, 1)
        m.gate_proj.weight[4] += 1.0
        m.down_proj.weight[:, 9] += 0.5
    ch = np.stack([I.mlp_change(base[lyr], I.model_mlp(model, lyr)) for lyr in range(3)])
    assert ch[1, 4] > 0 and ch[1, 9] > 0 and ch[0].max() == 0 and ch[2].max() == 0
    top = I.top_changed(ch, 2)
    assert set(top.tolist()) == {1 * N + 4, 1 * N + 9}

    def get(key):
        lyr = int(key.split(".")[1])
        name = key.split(".")[3].split("_")[0]
        return base[lyr][name]

    bw = I.base_neuron_weights(get, top, N)
    changed = model(input_ids=ids).logits.clone()
    assert not torch.allclose(changed, ref_logits)
    with I.revert_neurons(model, bw, top, N):
        assert torch.allclose(model(input_ids=ids).logits, ref_logits, atol=1e-5)
    assert torch.allclose(model(input_ids=ids).logits, changed)  # restored on exit
    with pytest.raises(KeyError), I.revert_neurons(model, bw, np.array([2 * N + 1]), N):
        pass
    assert len(decoder_layers(model)) == 3


def test_condition_ids():
    from scripts.m5_remove import expand_curve, expand_full, parse_cid

    nstar = {"raw|mean": 1406, "centered|mean": 976, "raw|zero": 2263}
    full = expand_full(
        [
            "none",
            "poles|raw|mean|k=nstar",
            "poles|raw|zero|k=nstar:raw/mean",
            "random_dirs|s0|k=nstar:raw/mean",
            "dom|all",
        ],
        nstar,
    )
    assert full == [
        "none",
        "poles|raw|mean|k=1406",
        "poles|raw|zero|k=1406",
        "random_dirs|s0|k=1406",
        "dom|all",
    ]
    curve = expand_curve(
        [
            {"method": "poles", "rankings": ["raw"], "fills": ["mean", "zero"], "nstar": True},
            {"method": "random_poles", "seeds": [0, 1], "fill": "mean", "nstar_of": "raw|mean"},
            {"method": "weight_revert", "ks": [4]},
            {"method": "pole_set", "sets": ["m4"], "fills": ["mean"]},
        ],
        [1, 2],
        nstar,
    )
    assert curve[0] == "none"
    assert "poles|raw|mean|k=1406" in curve and "poles|raw|zero|k=2263" in curve
    assert "random_poles|s1|mean|k=1406" in curve and "weight_revert|k=4" in curve
    assert "weight_revert|k=1" not in curve and "pole_set|m4|mean" in curve
    assert len(curve) == len(set(curve))
    d = parse_cid("random_poles|s2|mean|k=64")
    assert d == {
        "method": "random_poles",
        "cid": "random_poles|s2|mean|k=64",
        "seed": 2,
        "k": 64,
        "fill": "mean",
    }
    with pytest.raises(ValueError):
        parse_cid("bogus|x")
