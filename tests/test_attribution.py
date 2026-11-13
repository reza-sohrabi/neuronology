"""attribution.py on random tiny Qwen3 and Qwen3.5 models (CPU, fp32, no download)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology import attribution as A
from neuronology.hooks import intervene


def tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=40,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
    )
    m = Qwen3ForCausalLM(cfg).eval()
    _randomise_norms(m, center=1.0)
    return m


def tiny_qwen35():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3_5TextConfig(
        vocab_size=101,
        hidden_size=32,
        intermediate_size=40,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        linear_conv_kernel_dim=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    )
    m = Qwen3_5ForCausalLM(cfg).eval()
    _randomise_norms(m, center=0.0)  # Qwen3.5 stores w with gain (1 + w)
    return m


def _randomise_norms(model, center: float):
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("layernorm.weight") or name == "model.norm.weight":
                p.copy_(center + 0.3 * torch.randn(p.shape, generator=g))


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def model(request):
    m = tiny_qwen3() if request.param == "qwen3" else tiny_qwen35()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


IDS = [5, 17, 3, 88, 42, 9, 61, 23, 7, 14, 30, 2]


def test_norm_gain_matches_module(model):
    lay = model.model.layers[0]
    norm = lay.post_attention_layernorm
    h = torch.randn(3, 32)
    manual = h * A.inv_rms(h, A.norm_eps(norm)) * A.norm_gain(norm)
    torch.testing.assert_close(manual, norm(h), rtol=1e-5, atol=1e-6)


def test_decomposition_sums_to_true_preactivation(model):
    parts = A.capture_parts(model, IDS)
    N = model.model.layers[0].mlp.down_proj.in_features
    for layer in range(len(model.model.layers)):
        for pos in (0, 5, len(IDS) - 1):
            uc = A.unit_contributions(model, parts, layer, torch.arange(N), pos)
            torch.testing.assert_close(uc["gate"].sum(1), uc["z_gate"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(uc["up"].sum(1), uc["z_up"], rtol=1e-4, atol=1e-5)
            assert uc["gate"].shape[1] == uc["layout"].n_units


def test_layout_labels_cover_all_units(model):
    lo = A.unit_layout(model, 2)
    labels = [lo.label(u) for u in range(lo.n_units)]
    assert labels[0]["type"] == "embed"
    assert sum(x["type"] == "neuron" for x in labels) == 2 * lo.n_neurons
    assert sum(x["type"] == "mixer_head" for x in labels) == sum(lo.head_counts)


def test_grad_x_act_matches_finite_difference(model):
    torch.manual_seed(0)
    tok = 11
    res = A.grad_x_act(model, IDS, tok, keep_dense=True)
    item = A.AuditItem("x", "triggered", 0, IDS + [tok], len(IDS), tok)
    base = A.margins(model, [item])[0]
    assert abs(base - res["margin"]) < 1e-4
    layer, pos, n = 1, 4, 7
    eps = 1e-2

    def bump(lyr, acts):
        out = acts.clone()
        out[0, pos, n] += eps
        return out

    with intervene(model, bump, [layer]):
        bumped = A.margins(model, [item])[0]
    fd = (bumped - base) / eps
    g = float(res["grad"][layer, pos, n])
    assert abs(fd - g) < 1e-2 * max(1.0, abs(g))
    # pole sums: raw pos + neg == sum over positions of a * g
    a, gr = res["act"][layer, :, n], res["grad"][layer, :, n]
    np.testing.assert_allclose(
        res["raw"][layer, n].sum(), float((a * gr).sum()), rtol=1e-4, atol=1e-6
    )


def test_centered_uses_means(model):
    L = len(model.model.layers)
    N = model.model.layers[0].mlp.down_proj.in_features
    mu = torch.randn(L, N)
    r0 = A.grad_x_act(model, IDS, 11, keep_dense=True)
    r1 = A.grad_x_act(model, IDS, 11, means=mu)
    a, g = r0["act"], r0["grad"]
    expect_pos = (((a - mu[:, None, :]) * g) * (a > 0)).sum(1)
    np.testing.assert_allclose(r1["centered"][..., 0], expect_pos.numpy(), rtol=1e-4, atol=1e-6)


def test_pole_ablation_keep_all_and_none(model):
    L = len(model.model.layers)
    N = model.model.layers[0].mlp.down_proj.in_features
    items = [A.AuditItem(str(i), "triggered", i, IDS[: 6 + i] + [3], 6 + i, 3) for i in range(3)]
    full = A.margins(model, items, batch_size=2)
    fill = {lyr: torch.zeros(N) for lyr in range(L)}
    all_poles = np.arange(L * N * 2)
    keep_all = A.margins(model, items, A.PoleAblation.from_flat(all_poles, L, N, fill, "keep"))
    np.testing.assert_allclose(keep_all, full, rtol=1e-5, atol=1e-5)
    keep_none = A.margins(
        model, items, A.PoleAblation.from_flat(np.array([], int), L, N, fill, "keep")
    )
    abl_all = A.margins(model, items, A.PoleAblation.from_flat(all_poles, L, N, fill, "ablate"))
    np.testing.assert_allclose(keep_none, abl_all, rtol=1e-5, atol=1e-5)
    # batched == unbatched, under an ablation too
    abl = A.PoleAblation.from_flat(np.array([0, 3, 2 * N + 1]), L, N, fill, "ablate")
    np.testing.assert_allclose(
        A.margins(model, items, abl, batch_size=3),
        A.margins(model, items, abl, batch_size=1),
        rtol=1e-5,
        atol=1e-5,
    )


def test_pole_ablation_only_touches_its_sign():
    fill = {0: torch.full((3,), 9.0)}
    abl = A.PoleAblation.from_flat(np.array([0 * 2 + 0, 1 * 2 + 1]), 1, 3, fill, "ablate")
    acts = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]])
    out = abl.fn()(0, acts)
    assert out.tolist() == [[[9.0, 2.0, 3.0], [-1.0, 9.0, -3.0]]]


def test_find_circuit_size():
    res = A.find_circuit_size(lambda n: min(1.0, n / 37), 1000, 0.9)
    assert res["n_star"] == 34  # smallest n with n/37 >= 0.9
    assert A.find_circuit_size(lambda n: 0.0, 64, 0.9)["n_star"] is None


def test_aggregate_and_ranking():
    s = np.zeros((2, 1, 3, 2))
    s[0, 0, 1, 0], s[1, 0, 1, 0] = 5, -3
    s[0, 0, 2, 1] = 1
    ag = A.aggregate(s, top_k=1)
    assert ag["mean_abs"][0, 1, 0] == 4
    assert ag["top_frac"][0, 1, 0] == 1.0
    order = A.ranking(ag["mean_abs"])
    assert A.unflat(order[0], 3) == (0, 1, 0)
    assert A.pole_key(0, 1, 0) == "L0_N1_pos"


def test_readers_consistent_with_decomposition(model):
    parts = A.capture_parts(model, IDS)
    layer, n, pos = 0, 3, 6
    eff = A.reader_effects(model, parts, layer, n, pos)
    lp = 2
    N = model.model.layers[0].mlp.down_proj.in_features
    uc = A.unit_contributions(model, parts, lp, torch.arange(N), pos)
    u = uc["layout"].neuron_offset(layer) + n
    g, up = uc["z_gate"], uc["z_up"]
    expect = A.dsilu(g) * up * uc["gate"][:, u] + torch.nn.functional.silu(g) * uc["up"][:, u]
    torch.testing.assert_close(eff["later"][lp], expect, rtol=1e-4, atol=1e-6)
    tr = A.top_readers([eff], top=5, top_logits=3)
    assert len(tr["neurons"]) == 5 and len(tr["logits_up"]) == 3


def test_input_profile_runs(model):
    parts = A.capture_parts(model, IDS)
    prof = A.input_profile(model, [(parts, [3, 7])], layer=2, neuron=4, pole=0, n_baseline=8, top=5)
    assert prof["n_positions"] == 2
    assert len(prof["gate_drivers"]) == 5 and "excess" in prof["up_suppressors"][0]
    assert prof["decomposition_rel_residual_max"] < 1e-3


def test_decomposition_bf16_residual_small():
    import copy

    m = copy.deepcopy(tiny_qwen35()).to(torch.bfloat16)
    parts = A.capture_parts(m, IDS)
    uc = A.unit_contributions(m, parts, 3, torch.arange(40), 6)
    err = (uc["gate"].sum(1) - uc["z_gate"]).abs().max()
    assert float(err) < 0.05 * float(uc["z_gate"].abs().max())


def test_recovery_curves_small(model):
    L = len(model.model.layers)
    N = model.model.layers[0].mlp.down_proj.in_features
    items = [A.AuditItem(str(i), "triggered", i, IDS[: 6 + i] + [3], 6 + i, 3) for i in range(3)]
    scores = np.stack([A.grad_x_act(model, it.prefix(), 3)["raw"] for it in items])
    order = A.ranking(A.aggregate(scores)["mean_abs"])
    fill = {lyr: torch.zeros(N) for lyr in range(L)}
    out = A.recovery_curves(
        model,
        items,
        order,
        L,
        N,
        fill,
        {"n_random": 2, "batch_size": 4},
        np.random.default_rng(0),
        log=lambda m: None,
    )
    ns = out["sufficiency"]["n_star"]
    assert ns is not None and out["sufficiency"]["search_evals"][str(ns)] >= 0.9
    assert set(out) >= {"l_full", "l_0", "necessity"}
