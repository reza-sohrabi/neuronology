"""Routed (OLMoE) neuron hooks on a tiny random OLMoE model (PLAN_OLMOE.md section 4).

CPU, fp32, no download. The model has 3 layers, 8 experts of 16 neurons, top-2 routing,
so every hook is checked against routing that differs by token.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology import hooks
from neuronology.model import arch_info
from neuronology.routed import (
    RoutedPoleAblation,
    ablate_routed_poles,
    capture_parts_routed,
    capture_routed,
    capture_router,
    drop_experts,
    expert_weights,
    expert_writes,
    experts_module,
    grad_x_act_routed,
    intervene_routed,
    is_routed,
    make_routed_sparsify_fn,
    patch_fn,
    reroute_without,
    routed_info,
    routed_keep_mask,
    unit_contributions_routed,
    unit_key,
)

E, I, K, L, V = 8, 16, 2, 3, 128  # noqa: E741


def _tiny(impl: str = "eager"):
    from transformers import OlmoeConfig, OlmoeForCausalLM

    torch.manual_seed(0)
    cfg = OlmoeConfig(
        vocab_size=V,
        hidden_size=32,
        intermediate_size=I,
        num_hidden_layers=L,
        num_attention_heads=4,
        num_experts=E,
        num_experts_per_tok=K,
        max_position_embeddings=64,
        eos_token_id=None,
        pad_token_id=0,
    )
    model = OlmoeForCausalLM(cfg).eval()
    model.set_experts_implementation(impl)
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


def _ref(model, ids):
    """Logits through the instrumented forward with an observe-only hook (bitwise baseline)."""
    with capture_routed(model):
        return _logits(model, ids)


def _fill(value: float = 0.0) -> dict[int, torch.Tensor]:
    return {lyr: torch.full((E * I,), value) for lyr in range(L)}


def test_info_and_detection(model):
    info = routed_info(model)
    assert (info.n_experts, info.top_k, info.d_expert, info.n_units) == (E, K, I, E * I)
    assert info.routed_cells_per_token == K * I
    assert is_routed(model)
    a = arch_info(model)
    assert a.is_routed and a.extra["num_experts"] == E
    assert unit_key(2, 3 * I + 5, 1, I) == "L2_E3_N5_neg"


def test_dense_hooks_refuse_routed_model(model, ids):
    with pytest.raises(TypeError, match="routed"), hooks.capture(model):
        model(input_ids=ids)


def test_identity_is_exact_and_hooks_are_removed(model, ids):
    base = _logits(model, ids)  # transformers' own eager expert loop
    with intervene_routed(model, lambda _l, ra: ra.act.clone()):
        same = _logits(model, ids)
    # same arithmetic as the eager loop up to row order inside each expert's matmul
    torch.testing.assert_close(same, base, rtol=0, atol=1e-6)
    torch.testing.assert_close(same, _ref(model, ids), rtol=0, atol=0)
    for lyr in range(L):
        mod = experts_module(model, lyr)
        assert not hasattr(mod, "_nrn_callbacks") and "forward" not in mod.__dict__
    torch.testing.assert_close(_logits(model, ids), base, rtol=0, atol=0)


def test_instrumented_matches_grouped_mm_default(ids):
    """The instrumented forward mirrors transformers' grouped_mm path step for step."""
    m = _tiny("grouped_mm")
    base = _logits(m, ids)
    with capture_routed(m):
        hooked = _logits(m, ids)
    torch.testing.assert_close(hooked, base, rtol=0, atol=0)


def test_pinned_path_is_used_with_and_without_hooks(ids):
    from neuronology.routed import is_pinned, pin_measurement_path, unpinned

    m = _tiny("eager")
    pin_measurement_path(m)
    assert is_pinned(m)
    plain = _logits(m, ids)
    with intervene_routed(m, lambda _l, ra: ra.act.clone()):
        hooked = _logits(m, ids)
    torch.testing.assert_close(hooked, plain, rtol=0, atol=0)
    assert is_pinned(m)  # hooks removed, the pinned forward stays
    with unpinned(m, "grouped_mm"):
        assert not is_pinned(m)
        torch.testing.assert_close(_logits(m, ids), plain, rtol=0, atol=0)
    assert is_pinned(m)


def test_capture_recomputes_and_writes_sum_to_block_output(model, ids):
    outs = {}
    hs = [
        model.model.layers[lyr].mlp.register_forward_hook(
            lambda _m, _a, out, lyr=lyr: outs.__setitem__(lyr, out.detach())
        )
        for lyr in range(L)
    ]
    try:
        with torch.no_grad(), capture_routed(model) as st:
            model(input_ids=ids)
    finally:
        for h in hs:
            h.remove()
    assert sorted(st) == list(range(L))
    for lyr, ra in st.items():
        assert ra.act.shape == (ids.numel(), K, I)
        torch.testing.assert_close(torch.nn.functional.silu(ra.z_gate) * ra.z_up, ra.act)
        # each token's k experts are distinct and the weights are the router's softmax top-k
        assert (ra.experts.sort(-1).values.diff(dim=-1) > 0).all()
        recon = expert_writes(ra, experts_module(model, lyr).down_proj)
        out = outs[lyr].reshape(-1, recon.shape[-1])
        torch.testing.assert_close(recon, out, rtol=1e-4, atol=1e-5)
        # dense view: zeros exactly at unrouted experts
        d = ra.dense(E).view(-1, E, I)
        routed = ra.routed_mask(E)
        assert (d[~routed] == 0).all()
        assert routed.sum(-1).eq(K).all()


def test_router_capture_matches_routing(model, ids):
    with torch.no_grad(), capture_router(model) as rl, capture_routed(model) as st:
        model(input_ids=ids)
    for lyr in range(L):
        probs = torch.softmax(rl[lyr].float(), -1)
        top = torch.topk(probs, K, -1)
        assert torch.equal(top.indices, st[lyr].experts)
        torch.testing.assert_close(top.values, st[lyr].weights.float())


def test_patching_captured_activations_reproduces_logits(model, ids):
    base = _ref(model, ids)
    with torch.no_grad(), capture_routed(model) as src:
        model(input_ids=ids)
    src = dict(src)
    # perturb the run, then patch every layer's captured activations back in: exact
    noise = lambda _l, ra: ra.act + 0.5  # noqa: E731
    with intervene_routed(model, noise):
        assert (_logits(model, ids) - base).abs().max() > 1e-3
        # patch is registered after the noise, so it overwrites it
        with intervene_routed(model, patch_fn(src)):
            patched = _logits(model, ids)
    torch.testing.assert_close(patched, base, rtol=0, atol=0)


def test_patch_matches_by_token_and_expert(model, ids):
    """Patching from a run with other routing only touches (token, expert) pairs in both."""
    other = torch.randint(1, V, ids.shape, generator=torch.Generator().manual_seed(7))
    with torch.no_grad(), capture_routed(model, 0) as src:
        model(input_ids=other)
    src = dict(src)
    with torch.no_grad(), capture_routed(model, 0) as cur:
        model(input_ids=ids)
    cur = cur[0]
    patched = patch_fn(src)(0, cur)
    for t in range(cur.n_tokens):
        for j in range(K):
            e = int(cur.experts[t, j])
            hit = (src[0].experts[t] == e).nonzero()
            want = src[0].act[t, int(hit[0])] if len(hit) else cur.act[t, j]
            torch.testing.assert_close(patched[t, j], want, rtol=0, atol=0)


def _unrouted(model, ids) -> tuple[int, int]:
    with torch.no_grad(), capture_routed(model) as st:
        model(input_ids=ids)
    for lyr in range(L):
        used = set(st[lyr].experts.unique().tolist())
        free = sorted(set(range(E)) - used)
        if free:
            return lyr, free[0]
    raise AssertionError("every expert routed somewhere")


def test_ablating_unrouted_unit_is_noop(model):
    short = torch.tensor([[5, 9]])  # 2 tokens x top-2 = at most 4 of 8 experts per layer
    lyr, e = _unrouted(model, short)
    base = _ref(model, short)
    U = E * I
    # every unit of the unrouted expert, both poles, filled with a huge value
    flat = np.array(
        [(lyr * U + e * I + n) * 2 + p for n in range(I) for p in (0, 1)], dtype=np.int64
    )
    with ablate_routed_poles(model, flat, _fill(1e3), routed_info(model)):
        same = _logits(model, short)
    torch.testing.assert_close(same, base, rtol=0, atol=0)
    with intervene_routed(model, drop_experts({lyr: [e]})):
        torch.testing.assert_close(_logits(model, short), base, rtol=0, atol=0)
    # control: the same ablation on a routed expert of that layer does change the logits
    with torch.no_grad(), capture_routed(model, lyr) as st:
        model(input_ids=short)
    e2 = int(st[lyr].experts[0, 0])
    flat2 = np.array(
        [(lyr * U + e2 * I + n) * 2 + p for n in range(I) for p in (0, 1)], dtype=np.int64
    )
    with ablate_routed_poles(model, flat2, _fill(1e3), routed_info(model)):
        assert (_logits(model, short) - base).abs().max() > 1e-3


def test_pole_ablation_fills_only_matching_sign_at_routed_tokens(model, ids):
    with torch.no_grad(), capture_routed(model, 1) as st:
        model(input_ids=ids)
    ra = st[1]
    t, j = 3, 0
    e = int(ra.experts[t, j])
    n = int(ra.act[t, j].abs().argmax())
    pole = 0 if ra.act[t, j, n] > 0 else 1
    flat = np.array([(1 * E * I + e * I + n) * 2 + pole], dtype=np.int64)
    fill = _fill(7.0)
    abl = RoutedPoleAblation.from_flat(flat, L, E, I, fill, "ablate")
    new = abl.fn()(1, ra)
    changed = new != ra.act
    sign_ok = (ra.act > 0) if pole == 0 else (ra.act < 0)
    expect = (ra.experts[..., None] == e) & (torch.arange(I) == n) & sign_ok
    assert torch.equal(changed, expect & (ra.act != 7.0))
    assert (new[expect] == 7.0).all()
    # keep mode: everything routed except that pole is filled
    keep = RoutedPoleAblation.from_flat(flat, L, E, I, fill, "keep").fn()(1, ra)
    assert (keep[~expect] == 7.0).all() and torch.equal(keep[expect], ra.act[expect])


def test_ablation_equals_patching_the_ablated_values(model, ids):
    """Pole ablation through the hook = patching in activations edited by hand."""
    with torch.no_grad(), capture_routed(model) as st:
        model(input_ids=ids)
    lyr = 2
    ra = st[lyr]
    e = int(ra.experts[0, 1])
    flat = np.array([(lyr * E * I + e * I + n) * 2 + p for n in (0, 3) for p in (0, 1)])
    fill = _fill(-2.0)
    with ablate_routed_poles(model, flat, fill, routed_info(model)):
        ablated = _logits(model, ids)
    edited = dict(st)
    a = ra.act.clone()
    sel = (ra.experts[..., None] == e) & torch.isin(torch.arange(I), torch.tensor([0, 3]))
    a[sel & (a != 0)] = -2.0
    edited[lyr] = type(ra)(a, ra.experts, ra.weights)
    with intervene_routed(model, patch_fn({lyr: edited[lyr]}), lyr):
        patched = _logits(model, ids)
    # equal activations; the CPU grouped-mm fallback may still differ in the last fp32 bit
    # depending on the input tensor's memory layout
    torch.testing.assert_close(ablated, patched, rtol=0, atol=1e-6)


def test_drop_equals_zero_fill_of_all_units_and_reroute_excludes(model, ids):
    with torch.no_grad(), capture_routed(model, 1) as st:
        model(input_ids=ids)
    e = int(st[1].experts[0, 0])
    flat = np.array([(1 * E * I + e * I + n) * 2 + p for n in range(I) for p in (0, 1)])
    with intervene_routed(model, drop_experts({1: [e]}), 1):
        dropped = _logits(model, ids)
    with ablate_routed_poles(model, flat, _fill(0.0), routed_info(model)):
        zeroed = _logits(model, ids)
    torch.testing.assert_close(dropped, zeroed, rtol=0, atol=0)
    with torch.no_grad(), reroute_without(model, {1: [e]}), capture_routed(model, 1) as st2:
        model(input_ids=ids)
    assert not (st2[1].experts == e).any()
    assert st2[1].experts.shape == st[1].experts.shape


def test_routed_sparsify_masks(model, ids):
    with torch.no_grad(), capture_routed(model, 0) as st:
        model(input_ids=ids)
    ra = st[0]
    f = 0.25
    n_keep = round(f * K * I)
    top = routed_keep_mask(ra, "top", f, "abs_route")
    assert top.view(ra.n_tokens, -1).sum(-1).eq(n_keep).all()
    bottom = routed_keep_mask(ra, "bottom", f, "abs_route")
    assert torch.equal(bottom, ~top)
    score = (ra.act.abs() * ra.weights[..., None]).view(ra.n_tokens, -1)
    kept_min = score.masked_fill(~top.view_as(score), float("inf")).min(-1).values
    drop_max = score.masked_fill(top.view_as(score), -float("inf")).max(-1).values
    assert (kept_min >= drop_max).all()
    g = torch.Generator().manual_seed(0)
    rnd = routed_keep_mask(ra, "random", f, generator=g)
    assert rnd.view(ra.n_tokens, -1).sum(-1).eq(n_keep).all()
    per_expert = routed_keep_mask(ra, "top", f, "abs", scope="expert")
    assert per_expert.sum(-1).eq(round(f * I)).all()
    rnd_e = routed_keep_mask(ra, "random", f, scope="expert", generator=g)
    assert rnd_e.sum(-1).eq(round(f * I)).all()
    # keep everything = identity; zero fill at f -> only kept cells survive
    base = _ref(model, ids)
    with intervene_routed(model, make_routed_sparsify_fn("top", 1.0 - 1e-9, _fill(0.0))):
        torch.testing.assert_close(_logits(model, ids), base, rtol=0, atol=0)
    out = make_routed_sparsify_fn("top", f, None)(0, ra)
    assert torch.equal(out != 0, top & (ra.act != 0))


def test_grad_x_act_routed_first_order(model):
    ids = torch.randint(1, V, (10,), generator=torch.Generator().manual_seed(3)).tolist()
    tok = 17
    res = grad_x_act_routed(model, ids, tok)
    raw = res["raw"]
    assert raw.shape == (L, E * I, 2)
    with torch.no_grad(), capture_routed(model) as st:
        model(input_ids=torch.tensor([ids]))
    # unrouted units have exactly zero score
    for lyr in range(L):
        used = torch.zeros(E * I, dtype=torch.bool)
        used[st[lyr].units().reshape(-1)] = True
        assert (raw[lyr][~used.numpy()] == 0).all()
    # first order: scaling one unit's routed activations by (1 + eps) moves the margin by
    # about eps * (raw_pos + raw_neg) of that unit
    lyr, u = np.unravel_index(np.abs(raw.sum(-1)).argmax(), raw.shape[:2])
    eps = 1e-3

    def scale(_l, ra, lyr=int(lyr), u=int(u)):
        return torch.where(ra.units() == u, ra.act * (1 + eps), ra.act) if _l == lyr else ra.act

    from neuronology.attribution import _margin_from_hidden

    def margin():
        h = model.model(input_ids=torch.tensor([ids])).last_hidden_state[:, -1]
        return float(_margin_from_hidden(model, h, torch.tensor([tok]))[0])

    with torch.no_grad():
        m0 = margin()
        with intervene_routed(model, scale):
            m1 = margin()
    assert abs(m0 - res["margin"]) < 1e-4
    pred = eps * float(raw[lyr, u].sum())
    assert abs((m1 - m0) - pred) <= 0.05 * abs(pred) + 1e-7


def test_upstream_decomposition_is_exact(model):
    ids = torch.randint(1, V, (9,), generator=torch.Generator().manual_seed(5)).tolist()
    parts = capture_parts_routed(model, ids)
    layer, pos = 2, 6
    ra = parts.routed[layer]
    routed_e = int(ra.experts[pos, 0])
    unrouted_e = next(e for e in range(E) if e not in ra.experts[pos].tolist())
    neurons = [0, 5, 11]
    for e in (routed_e, unrouted_e):
        uc = unit_contributions_routed(model, parts, layer, e, neurons, pos)
        lo = uc["layout"]
        assert uc["gate"].shape == (len(neurons), lo.n_units)
        for lane in ("gate", "up"):
            torch.testing.assert_close(uc[lane].sum(-1), uc[f"z_{lane}"], rtol=1e-4, atol=1e-5)
        if e == routed_e:
            torch.testing.assert_close(uc["z_gate"], ra.z_gate[pos, 0, neurons])
            torch.testing.assert_close(uc["z_up"], ra.z_up[pos, 0, neurons])
        # expert-neuron units of earlier layers are zero unless routed at pos
        for lyr in range(layer):
            o = lo.unit_offset(lyr)
            block = uc["gate"][:, o : o + E * I].view(len(neurons), E, I)
            routed = parts.routed[lyr].experts[pos].tolist()
            for e2 in range(E):
                if e2 not in routed:
                    assert (block[:, e2] == 0).all()
    assert lo.label(lo.unit_offset(1) + 3 * I + 4) == {
        "type": "expert_neuron",
        "layer": 1,
        "expert": 3,
        "neuron": 4,
    }
    w = expert_weights(model, 0)
    assert w["gate"].shape == (E, I, 32) and w["down"].shape == (E, 32, I)


class _Largest:
    """Largest tensor produced by any op inside the block (bytes), via a dispatch mode."""

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        outer = self
        self.max_bytes = 0

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                for t in out if isinstance(out, (tuple, list)) else [out]:
                    if isinstance(t, torch.Tensor):
                        outer.max_bytes = max(outer.max_bytes, t.numel() * t.element_size())
                return out

        self.mode = Mode()
        self.mode.__enter__()
        return self

    def __exit__(self, *a):
        self.mode.__exit__(*a)


def test_pole_ablation_allocates_no_tensor_larger_than_the_forward_does(model):
    """G0 OOM (PLAN_OLMOE.md 13.4): the ablation path must not create [T, k, I] fp32 or
    other tensors bigger than the plain expert forward's own largest buffer."""
    from neuronology import routed as RT

    long = torch.randint(1, V, (4, 64), generator=torch.Generator().manual_seed(9))
    with torch.no_grad(), _Largest() as plain:
        model(input_ids=long)
    with torch.no_grad(), capture_routed(model, 1) as st:
        model(input_ids=long)
    e = int(st[1].experts[0, 0])
    flat = np.array([(1 * E * I + e * I + n) * 2 + p for n in range(I) for p in (0, 1)])
    saved = RT.ROW_CHUNK
    RT.ROW_CHUNK = 64  # force several chunks
    try:
        abl = RoutedPoleAblation.from_flat(flat, L, E, I, _fill(1.0), "ablate")
        out = abl.fn()(1, st[1])
        with _Largest() as fn_only:
            abl.fn()(1, st[1])
        with torch.no_grad(), _Largest() as hooked, intervene_routed(model, abl.fn(), 1):
            model(input_ids=long)
    finally:
        RT.ROW_CHUNK = saved
    ref = RoutedPoleAblation.from_flat(flat, L, E, I, _fill(1.0), "ablate").fn()(1, st[1])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)  # chunking changes nothing
    act_bytes = st[1].act.numel() * st[1].act.element_size()
    assert fn_only.max_bytes <= act_bytes  # the clone; every temporary is a chunk
    assert hooked.max_bytes <= plain.max_bytes
