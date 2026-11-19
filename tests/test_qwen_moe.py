"""The routed hooks on a tiny random Qwen3.5-MoE model (the architecture of Qwen3.6-35B-A3B).

CPU, fp32, no download. 4 layers (3 Gated DeltaNet, 1 attention), 8 experts of 16 neurons,
top-2 routing with renormalised weights, and a shared expert of 16 neurons per layer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology.model import arch_info, decoder_layers
from neuronology.routed import (
    RoutedPoleAblation,
    base_expert_weights,
    base_unit_rows,
    capture_routed,
    capture_routing,
    drop_experts,
    drop_shared_expert,
    expert_writes,
    experts_module,
    grad_x_act_routed,
    has_shared_expert,
    intervene_routed,
    is_pinned,
    is_routed,
    pin_measurement_path,
    reroute_without,
    revert_shared_expert,
    revert_units,
    routed_info,
    router_normalises,
    shared_expert_change,
    unit_weight_change,
    unpinned,
)

E, I, K, L, V, D = 8, 16, 2, 4, 128, 32  # noqa: E741


def _tiny():
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=V,
        hidden_size=D,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=I,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=None,
    )
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():  # the router is zero-initialised: give it something to route by
        for lay in decoder_layers(model):
            lay.mlp.gate.weight.normal_(0, 0.5)
            lay.mlp.shared_expert_gate.weight.normal_(0, 0.5)
    return model


@pytest.fixture(scope="module")
def model():
    return _tiny()


@pytest.fixture(scope="module")
def ids():
    return torch.randint(1, V, (2, 12), generator=torch.Generator().manual_seed(1))


@torch.no_grad()
def _logits(model, ids):
    return model(input_ids=ids).logits.float()


def test_architecture(model):
    assert is_routed(model) and has_shared_expert(model)
    ri = routed_info(model)
    assert (ri.n_layers, ri.n_experts, ri.top_k, ri.d_expert, ri.d_model) == (L, E, K, I, D)
    assert ri.norm_topk_prob and ri.has_shared_expert and ri.n_units == E * I
    a = arch_info(model)
    assert a.is_routed and a.d_mlp == I and a.is_swiglu
    assert a.layer_types.count("linear_attention") == 3
    assert router_normalises(decoder_layers(model)[0].mlp.gate)


def test_instrumented_forward_matches_transformers(model, ids):
    ref = _logits(model, ids)
    with capture_routed(model) as acts:
        hooked = _logits(model, ids)
    assert torch.allclose(hooked, ref, atol=1e-5)
    assert acts[0].act.shape == (ids.numel(), K, I)
    assert torch.allclose(acts[2].weights.sum(-1), torch.ones(ids.numel()), atol=1e-5)
    pin_measurement_path(model)
    try:
        assert is_pinned(model)
        pinned = _logits(model, ids)
        for impl in ("eager", "grouped_mm"):
            with unpinned(model, impl):
                assert torch.allclose(_logits(model, ids), pinned, atol=1e-5)
    finally:
        for lyr in range(L):
            m = experts_module(model, lyr)
            m._nrn_pinned = False
            m.__dict__.pop("forward", None)
            m.__dict__.pop("_nrn_callbacks", None)


def test_block_output_is_routed_writes_plus_shared(model, ids):
    lyr = 1
    mlp = decoder_layers(model)[lyr].mlp
    grab = {}
    h = mlp.register_forward_hook(lambda _m, a, out: grab.update(x=a[0].detach(), y=out.detach()))
    try:
        with capture_routed(model, lyr) as acts:
            _logits(model, ids)
    finally:
        h.remove()
    x = grab["x"].reshape(-1, D)
    shared = torch.sigmoid(mlp.shared_expert_gate(x)) * mlp.shared_expert(x)
    routed = expert_writes(acts[lyr], mlp.experts.down_proj)
    assert torch.allclose(routed + shared, grab["y"].reshape(-1, D), atol=1e-5)


def test_reroute_renormalises(model, ids):
    with capture_routing(model, 0) as before:
        base = _logits(model, ids)
    drop = sorted(set(before[0][1].reshape(-1).tolist()))[:2]
    with reroute_without(model, {0: drop}), capture_routing(model, 0) as after:
        out = _logits(model, ids)
    w, idx = after[0]
    assert not torch.isin(idx, torch.tensor(drop)).any()
    assert torch.allclose(w.sum(-1), torch.ones(len(w)), atol=1e-5)
    assert not torch.allclose(out, base, atol=1e-4)


def test_shared_expert_drop_and_revert(model, ids, tmp_path):
    from neuronology.diff import WeightReader

    base = _logits(model, ids)
    with drop_shared_expert(model):
        dropped = _logits(model, ids)
    assert not torch.allclose(dropped, base, atol=1e-4)
    with drop_shared_expert(model, [0]):
        one = _logits(model, ids)
    assert not torch.allclose(one, dropped, atol=1e-4)
    assert torch.allclose(_logits(model, ids), base)

    model.save_pretrained(tmp_path)
    rd = WeightReader(tmp_path)
    assert shared_expert_change(model, rd.get).max() < 1e-6
    w = decoder_layers(model)[2].mlp.shared_expert.down_proj.weight
    keep = w.detach().clone()
    with torch.no_grad():
        w.add_(0.3)
    try:
        ch = shared_expert_change(model, rd.get)
        assert ch[2] > 0.01 and ch[0] < 1e-6
        changed = _logits(model, ids)
        assert not torch.allclose(changed, base, atol=1e-4)
        with revert_shared_expert(model, rd.get):
            assert torch.allclose(_logits(model, ids), base, atol=1e-5)
        assert torch.allclose(_logits(model, ids), changed)
    finally:
        with torch.no_grad():
            w.copy_(keep)


def test_fused_base_weights_and_compact_revert(model, ids, tmp_path):
    from neuronology.diff import WeightReader

    model.save_pretrained(tmp_path)
    rd = WeightReader(tmp_path)
    lyr, U = 1, E * I
    full = base_expert_weights(rd.get, lyr, E)
    assert full["gate"].shape == (E, I, D) and full["down"].shape == (E, D, I)
    assert unit_weight_change(full, model, lyr).max() == 0
    base = _logits(model, ids)
    ex = experts_module(model, lyr)
    keep = (ex.gate_up_proj.detach().clone(), ex.down_proj.detach().clone())
    units = np.array([2 * I + 4, 5 * I + 1, 5 * I + 9])
    with torch.no_grad():
        ex.gate_up_proj[2, 4] += 0.5  # gate row of (2, 4)
        ex.gate_up_proj[5, I + 1] -= 0.5  # up row of (5, 1)
        ex.down_proj[5, :, 9] += 0.5  # down column of (5, 9)
    try:
        ch = unit_weight_change(full, model, lyr)
        assert set(np.nonzero(ch)[0].tolist()) == set(units.tolist())
        compact = {lyr: base_unit_rows(full, np.sort(units), I)}
        flat = lyr * U + units
        for b in (compact, {lyr: full}):
            with revert_units(model, b, flat, U, I):
                assert unit_weight_change(full, model, lyr).max() == 0
                assert torch.allclose(_logits(model, ids), base, atol=1e-5)
            assert np.count_nonzero(unit_weight_change(full, model, lyr)) == 3
        with revert_units(model, compact, flat[:1], U, I):  # a subset of the compact rows
            assert np.count_nonzero(unit_weight_change(full, model, lyr)) == 2
        with pytest.raises(KeyError):
            with revert_units(model, compact, np.array([lyr * U + 7]), U, I):
                pass
    finally:
        with torch.no_grad():
            ex.gate_up_proj.copy_(keep[0])
            ex.down_proj.copy_(keep[1])


def test_ablation_and_attribution(model, ids):
    ri = routed_info(model)
    base = _logits(model, ids)
    with intervene_routed(model, drop_experts({0: list(range(E))}), [0]):
        assert not torch.allclose(_logits(model, ids), base, atol=1e-4)
    r = grad_x_act_routed(model, ids[0].tolist(), 5)
    assert r["raw"].shape == (L, E * I, 2) and np.isfinite(r["raw"]).all()
    assert (r["count"].sum((1, 2)) == ids.shape[1] * K * I).all()  # every routed cell fires
    top = np.argsort(-np.abs(r["raw"]).reshape(-1))[:20]
    fill = {lyr: torch.zeros(E * I) for lyr in range(L)}
    abl = RoutedPoleAblation.from_flat(top, L, E, I, fill, "ablate")
    with intervene_routed(model, abl.fn(), abl.layers()):
        assert not torch.allclose(_logits(model, ids[:1]), base[:1], atol=1e-5)
    assert ri.unit(3, 2) == 3 * I + 2


def test_lora_adapts_experts_and_shared_expert(ids):
    from neuronology.backdoor import _set_trainable

    model = _tiny()
    peft_model, params, is_lora = _set_trainable(model, "lora", {"r": 2, "alpha": 4, "dropout": 0})
    names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert is_lora and params
    for part in ("experts", "shared_expert.gate_proj", "shared_expert_gate", "linear_attn"):
        assert any(part in n for n in names), part
    assert not any(".mlp.gate." in n for n in names)  # the router stays frozen
    with torch.no_grad():
        for n, p in peft_model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
        out = peft_model(input_ids=ids).logits
        merged = peft_model.merge_and_unload().eval()
        assert torch.allclose(merged(input_ids=ids).logits, out, atol=1e-4)


def test_blind_search_with_one_model_copy(tmp_path):
    """The blind-audit search on one base copy: adapter on = suspect, off = base."""
    from peft import PeftModel

    from neuronology import synthesize as Syn
    from neuronology.backdoor import _set_trainable

    base = _tiny()
    pin_measurement_path(base)
    pm, _, _ = _set_trainable(base, "lora", {"r": 2, "alpha": 4, "dropout": 0})
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.2)
    pm.save_pretrained(tmp_path)
    m = _tiny()
    pin_measurement_path(m)
    pm = PeftModel.from_pretrained(m, str(tmp_path)).eval()
    sus, bas = Syn.AdapterView(pm, True), Syn.AdapterView(pm, False)
    x = torch.randint(1, V, (1, 9), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        hs = sus.model(input_ids=x).last_hidden_state
        hb = bas.model(input_ids=x).last_hidden_state
        assert not torch.allclose(hs, hb, atol=1e-4)
        assert torch.allclose(sus.model(input_ids=x).last_hidden_state, hs)  # switches back

    carriers = [Syn.Carrier([1, 2], [3, 4, 5]), Syn.Carrier([1, 2], [6, 7])]
    allowed = torch.ones(V, dtype=torch.bool)
    allowed[0] = False
    cfg = {"objective": "diff", "fluency": 0.1, "n_steps": 3, "top_k": 8, "n_cand": 8}
    s = Syn.Searcher(sus, bas, carriers, cfg, allowed)
    gen = torch.Generator().manual_seed(0)
    tgt = Syn.Target(2, 3 * I + 5, 1.0, 0.5, d_expert=I, route_bonus=0.05)
    g = s.grad(s.random_init(4, gen), tgt)
    assert g.shape == (4, V) and torch.isfinite(g[:, 1:]).all() and g[:, 1:].abs().sum() > 0
    r = s.search(tgt, s.random_init(4, gen), gen)
    assert len(r["x"]) == 4 and r["trajectory"][-1] >= r["trajectory"][0] - 1e-6
    assert np.isfinite([r["J"], r["act_sus"], r["act_base"]]).all()
    k = Syn.Searcher(sus, bas, carriers, {**cfg, "objective": "kl"}, allowed)
    assert np.isfinite(k.search(None, k.random_init(4, gen), gen)["J"])
    beh = Syn.behaviour(sus, bas, carriers, [r["x"], [9, 9, 9, 9]], 4, [0], 2)
    assert len(beh["reply_sus"]) == 2 and len(beh["reply_kl"][0]) == 2
    for lyr in range(L):  # no hook left behind
        assert not getattr(experts_module(sus.core, lyr), "_nrn_callbacks", None)


def test_reply_grad_x_act_routed(model, ids):
    from neuronology.routed import reply_grad_x_act_routed

    prompt, reply = ids[0, :8].tolist(), ids[0, 8:].tolist()
    g = reply_grad_x_act_routed(model, prompt, reply)
    assert g.shape == (L, E * I, 2) and np.isfinite(g).all() and np.abs(g).sum() > 0
    # a unit never routed in this sequence gets nothing
    with capture_routed(model, 0) as acts:
        _logits(model, ids[:1])
    used = set(acts[0].experts.reshape(-1).tolist())
    for e in set(range(E)) - used:
        assert np.abs(g[0, e * I : (e + 1) * I]).sum() == 0
