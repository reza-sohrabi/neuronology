"""synthesize.py (PLAN.md 9.2): gradient-guided search on a tiny random Qwen3.5 pair where
one neuron of the "suspect" is planted to fire on one token (CPU, fp32, no download)."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from neuronology import synthesize as S
from neuronology.hooks import capture
from tests.test_attribution import tiny_qwen35

TRIG, LAYER, NEURON = 57, 0, 5


def pair():
    base = tiny_qwen35()
    sus = copy.deepcopy(base)
    with torch.no_grad():
        lay = sus.model.layers[LAYER]
        e = lay.post_attention_layernorm(sus.model.embed_tokens.weight[TRIG])
        w = 6.0 * e / e.norm().square()
        lay.mlp.gate_proj.weight[NEURON] = w
        lay.mlp.up_proj.weight[NEURON] = w
    return sus.eval(), base.eval()


CARRIERS = [S.Carrier([1, 2, 3], [4, 5, 6, 7]), S.Carrier([1, 2, 3], [8, 9, 10])]
CFG = {"objective": "diff", "fluency": 0.0, "n_steps": 25, "top_k": 16, "n_cand": 32}


def test_score_matches_direct_forward():
    sus, base = pair()
    s = S.Searcher(sus, base, CARRIERS, CFG, torch.ones(101, dtype=torch.bool))
    tgt = S.Target(LAYER, NEURON, 1.0, 2.0)
    x = torch.tensor([[11, TRIG, 12], [13, 14, 15]])
    got = s.score(x, tgt)
    want = []
    for c in CARRIERS:
        ids = torch.tensor([c.ids(r.tolist()) for r in x])
        with torch.no_grad(), capture(sus, LAYER) as a_s:
            sus(input_ids=ids)
        with torch.no_grad(), capture(base, LAYER) as a_b:
            base(input_ids=ids)
        d = (a_s[LAYER] - a_b[LAYER])[..., NEURON][:, 3:]
        want.append(d.max(1).values)
    want = torch.stack(want).mean(0) / 2.0
    assert torch.allclose(got["term"], want, atol=1e-5)
    assert got["J"][0] > got["J"][1] + 1.0  # the planted token fires the neuron


def test_fluency_term_is_mean_nll_of_free_tokens():
    sus, base = pair()
    cfg = dict(CFG, fluency=0.5)
    s = S.Searcher(sus, base, CARRIERS, cfg, torch.ones(101, dtype=torch.bool))
    x = torch.tensor([[11, 22, 33]])
    out = s.score(x, S.Target(LAYER, NEURON, 1.0, 1.0))
    ids = torch.tensor([CARRIERS[0].ids(x[0].tolist())])
    with torch.no_grad():
        lp = F.log_softmax(sus(input_ids=ids).logits.float(), -1)[0]
    nll = -torch.stack([lp[2 + i, int(x[0, i])] for i in range(3)]).mean()
    assert torch.allclose(out["nll"][0], nll, atol=1e-4)
    assert torch.allclose(out["J"], out["term"] - 0.5 * out["nll"])


def test_search_finds_planted_token_and_never_proposes_disallowed():
    sus, base = pair()
    allowed = torch.ones(101, dtype=torch.bool)
    allowed[:3] = False
    s = S.Searcher(sus, base, CARRIERS, CFG, allowed)
    tgt = S.Target(LAYER, NEURON, 1.0, 1.0)
    gen = torch.Generator().manual_seed(0)
    init = torch.tensor([20, 21, 22])
    j0 = float(s.score(init[None], tgt)["J"][0])
    best = s.search(tgt, init, gen)
    j_trig = float(s.score(torch.tensor([[20, TRIG, 22]]), tgt)["J"][0])
    assert best["J"] > j0 and best["J"] >= 0.9 * j_trig
    assert TRIG in best["x"]
    g = s.grad(torch.tensor(best["x"]), tgt)
    assert torch.isinf(g[:, :3]).all() and torch.isfinite(g[:, 3:]).all()


def test_lens_init_ranks_planted_token_first():
    sus, base = pair()
    s = S.Searcher(sus, base, CARRIERS, CFG, torch.ones(101, dtype=torch.bool))
    assert int(s.lens_init(S.Target(LAYER, NEURON, 1.0, 1.0), 3)[0]) == TRIG


def test_kl_objective_and_behaviour():
    sus, base = pair()
    ok = torch.ones(101, dtype=torch.bool)
    s = S.Searcher(sus, base, CARRIERS, dict(CFG, objective="kl"), ok)
    out = s.score(torch.tensor([[11, TRIG, 12], [13, 14, 15]]), None)
    assert (out["term"] >= -1e-6).all()
    g = s.grad(torch.tensor([11, TRIG, 12]), None)
    assert g.shape == (3, 101)
    beh = S.behaviour(sus, base, CARRIERS, [[11, TRIG, 12], [13, 14, 15]], 4, [0], bs=2)
    assert len(beh["reply_kl"]) == 2 and len(beh["reply_kl"][0]) == 2
    assert len(beh["reply_sus"][0][0]) == 4
    same = S.behaviour(base, base, CARRIERS, [[13, 14, 15]], 4, [0], bs=2)
    assert max(abs(v) for v in same["reply_kl"][0]) < 1e-5
    assert same["first_diff"][0] == [4, 4]


def test_diff_pole_ignores_base_swinging_to_other_pole():
    sus, base = pair()
    with torch.no_grad():  # make the base neuron strongly negative on the planted token
        lay = base.model.layers[LAYER]
        e = lay.post_attention_layernorm(base.model.embed_tokens.weight[TRIG])
        lay.mlp.gate_proj.weight[NEURON] = 6.0 * e / e.norm().square()
        lay.mlp.up_proj.weight[NEURON] = -6.0 * e / e.norm().square()
    ok = torch.ones(101, dtype=torch.bool)
    tgt = S.Target(LAYER, NEURON, 1.0, 1.0)
    x = torch.tensor([[11, TRIG, 12]])
    plain = S.Searcher(sus, base, CARRIERS, CFG, ok).score(x, tgt)
    pole = S.Searcher(sus, base, CARRIERS, dict(CFG, objective="diff_pole"), ok).score(x, tgt)
    assert plain["term"][0] > pole["term"][0] + 1.0  # plain diff also credits base < 0
    assert torch.allclose(pole["term"], pole["act_sus"] - pole["act_base"].clamp_min(0), atol=0.5)
    g = S.Searcher(sus, base, CARRIERS, dict(CFG, objective="diff_pole"), ok).grad(x[0], tgt)
    assert torch.isfinite(g).all()


def test_followup_units_rule():
    def row(key, s, a, b):
        layer, n, p = key.split("_")
        return {"kind": "pole", "key": key, "layer": int(layer[1:]), "neuron": int(n[1:]),
                "pole": int(p), "source": "x", "sus_over_census_max": s, "act_sus": a,
                "act_base": b}  # fmt: skip

    rows = [
        row("L1_N1_0", 0.9, 5.0, 1.0),  # not strong (below census max)
        row("L2_N2_0", 1.5, 5.0, 2.0),  # strong
        row("L3_N3_1", 2.0, 5.0, 3.0),  # above census max but < 2x base: tier 2
        row("L4_N4_0", 1.2, 4.0, 1.0),  # strong
        row("L5_N5_0", 3.0, 1.0, 1.0),  # act_sus == act_base: excluded
        {"kind": "kl", "key": "kl_r0"},
    ]
    u = S.followup_units(rows, 3)
    assert [x["wave1_key"] for x in u] == ["L2_N2_0", "L4_N4_0", "L3_N3_1"]
    assert [x["tier"] for x in u] == [1, 1, 2]
    assert len(S.followup_units(rows, 10)) == 4
