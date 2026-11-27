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


def test_centered_projection_and_clamp(tiny):
    x = torch.randint(0, 97, (1, 7))
    u = torch.nn.functional.normalize(torch.randn(32), dim=0)
    mu = np.random.default_rng(2).normal(size=(3, 32)).astype(np.float32)
    seen = {}

    def grab(_m, _a, out):  # registered after the edit's hook, so it sees the edited output
        seen["h"] = out[0] if isinstance(out, tuple) else out

    with torch.no_grad(), em.centered_projection(tiny, u, [0, 1], mu):
        hk = tiny.model.layers[0].register_forward_hook(grab)
        tiny.model(x)
        hk.remove()
    want = torch.full((1, 7), float(torch.as_tensor(mu[0]) @ u))
    assert torch.allclose(seen["h"] @ u, want, atol=1e-4)
    N = 48
    order = np.array([(1 * N + 3) * 2 + 1, (2 * N + 10) * 2 + 0])
    tgt = np.full((3, N, 2), 0.5, np.float32)
    ctx = {"orders": {"s/attr3": order}, "targets": {"s": tgt}}
    spec = {"kind": "clamp", "rank": "s/attr3", "k": 2, "target": "s", "scale": 2.0}
    from neuronology.hooks import capture

    with torch.no_grad(), em.edit(tiny, spec, ctx), capture(tiny) as acts:
        tiny(x)
    assert torch.allclose(acts[1][..., 3], torch.tensor(-1.0)) and torch.allclose(
        acts[2][..., 10], torch.tensor(1.0)
    )


def test_combined_order():
    a, b = np.array([5, 1, 2]), np.array([1, 7, 5])
    assert em.combined_order([a, b], 10, 3).tolist()[:2] == [1, 5]


def _fake_adapter(model, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ad = {}
    for i, lay in enumerate(model.model.layers):
        for proj in em.MLP_PROJ:
            out_f, in_f = getattr(lay.mlp, proj).weight.shape
            A = torch.randn(r, in_f, generator=g) * 0.1
            B = torch.randn(out_f, r, generator=g) * 0.1
            ad[(i, proj)] = (A, B, 0.5)
    return ad


def test_weight_change_and_revert(tiny):
    import copy

    N, L = 48, 3
    ad = _fake_adapter(tiny)
    ft = copy.deepcopy(tiny)
    with torch.no_grad():
        for (i, proj), (A, B, s) in ad.items():
            getattr(ft.model.layers[i].mlp, proj).weight += s * B @ A
    # per-neuron change norms match the dense difference
    d = em.weight_change(ad, L, N)
    for proj, dim in (("gate_proj", 1), ("up_proj", 1), ("down_proj", 0)):
        ref = (ft.model.layers[1].mlp.get_submodule(proj).weight
               - tiny.model.layers[1].mlp.get_submodule(proj).weight).norm(dim=dim)  # fmt: skip
        assert np.allclose(d[proj][1], ref.detach().numpy(), atol=1e-5)
    sc = em.relative_change(d, em.weight_norms(tiny))
    assert sc["wdiff"].shape == (L, N) and (sc["wdiff"] >= sc["wdiff_in"]).all()
    # reverting every neuron gives back the base MLPs; the block leaves no trace
    x = torch.randint(0, 97, (2, 9))
    every = np.arange(L * N)
    before = ft(x).logits
    with em.reverted(ft, ad, every):
        for i in range(L):
            for proj in em.MLP_PROJ:
                a = ft.model.layers[i].mlp.get_submodule(proj).weight
                b = tiny.model.layers[i].mlp.get_submodule(proj).weight
                assert torch.allclose(a, b, atol=1e-5)
    assert torch.equal(before, ft(x).logits)
    # one neuron: only its rows and column move
    with em.reverted(ft, ad, np.array([1 * N + 7])):
        g = ft.model.layers[1].mlp.gate_proj.weight
        g0 = tiny.model.layers[1].mlp.gate_proj.weight
        assert torch.allclose(g[7], g0[7], atol=1e-5) and not torch.allclose(g[8], g0[8])
    # edit specs: neuron ranking, pole ranking, random control
    ctx = {"norders": {"m/wdiff": em.top_neurons(sc["wdiff"], 20)}, "adapter": ad,
           "orders": {"m/attr": np.array([(2 * N + 5) * 2 + 1, (2 * N + 5) * 2])},
           "mean": np.zeros((L, N), np.float32)}  # fmt: skip
    for spec in (
        {"kind": "revert", "nrank": "m/wdiff", "k": 8},
        {"kind": "revert", "nrank": "m/wdiff", "k": 8, "random": 1},
        {"kind": "revert", "rank": "m/attr", "k": 2},
        {"kind": "poles", "nrank": "m/wdiff", "k": 8},
    ):
        with em.edit(ft, spec, ctx):
            assert not torch.allclose(before, ft(x).logits)
        assert torch.equal(before, ft(x).logits)
    assert em.edit_neurons({"rank": "m/attr", "k": 2}, ctx, N, L).tolist() == [2 * N + 5]
    r = em.edit_neurons({"nrank": "m/wdiff", "k": 8, "random": 1}, ctx, N, L)
    top = ctx["norders"]["m/wdiff"][:8]
    assert sorted((r // N).tolist()) == sorted((top // N).tolist()) and not set(r) & set(top)


def test_contribution_change(tiny):
    import copy

    N, L = 48, 3
    ft = copy.deepcopy(tiny)
    with torch.no_grad():  # change one neuron's output vector only
        ft.model.layers[2].mlp.down_proj.weight[:, 11] *= 3.0
    P, R = [[1, 2, 3], [4, 5]], [[6, 7, 8, 9], [10, 11]]
    same = em.contribution_change(tiny, tiny, P, R, 0, 2)
    assert same.shape == (L, N) and np.abs(same).max() < 1e-3
    c = em.contribution_change(ft, tiny, P, R, 0, 2)
    assert int(c.argmax()) == 2 * N + 11 and np.abs(c[:2]).max() < 1e-3
    # against a direct computation for that neuron
    from neuronology.hooks import capture

    x = torch.tensor([P[0] + R[0]])
    with capture(tiny, [2]) as st:
        tiny.model(input_ids=x)
    a = st[2][0, 3:, 11].float()
    w = tiny.model.layers[2].mlp.down_proj.weight[:, 11]
    c1 = em.contribution_change(ft, tiny, P[:1], R[:1], 0, 1)
    assert np.isclose(
        c1[2, 11], float((((a * 2) ** 2).mean().sqrt() * w.norm()).detach()), rtol=1e-3
    )


def test_edit_neurons_blocks_and_fractions():
    N, L = 48, 3
    a = em.edit_neurons({"layers": "1-2"}, {}, N, L)
    assert a.tolist() == list(range(N, 3 * N))
    f = em.edit_neurons({"frac": 0.25, "random": 3}, {}, N, L)
    assert len(f) == 36 and np.bincount(f // N).tolist() == [12, 12, 12] and len(set(f)) == 36
    assert len(em.edit_neurons({"all": True}, {}, N, L)) == N * L
