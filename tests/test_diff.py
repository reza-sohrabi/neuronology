"""diff.py (PLAN.md 9.1): weight and census-histogram signals, rankings, contrast."""

from __future__ import annotations

import numpy as np
import torch

from neuronology import diff as D
from neuronology.census import HistSpec

SPEC = HistSpec(bins_per_side=512, log10_min=-4.0, log10_max=3.0)


def hist_of(x: np.ndarray) -> np.ndarray:
    """[T, N] samples -> [N, B] histogram with census binning."""
    idx = SPEC.bin_index(torch.as_tensor(x)).numpy()
    return np.stack([np.bincount(idx[:, n], minlength=SPEC.n_bins) for n in range(x.shape[1])])


def test_canonical_key():
    assert D.canonical_key("model.language_model.layers.3.mlp.up_proj.weight") == (
        "layers.3.mlp.up_proj.weight"
    )
    assert D.canonical_key("model.layers.3.mlp.up_proj.weight") == "layers.3.mlp.up_proj.weight"
    assert D.canonical_key("model.language_model.embed_tokens.weight") == "embed_tokens.weight"
    assert D.canonical_key("mtp.layers.0.mlp.up_proj.weight") is None
    assert D.canonical_key("model.visual.blocks.0.mlp.fc1.weight") is None
    assert D.canonical_key("lm_head.weight") is None


def _w(N=6, d=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "gate": torch.randn(N, d, generator=g),
        "up": torch.randn(N, d, generator=g),
        "down": torch.randn(d, N, generator=g),
        "gain": 1.0 + 0.1 * torch.randn(d, generator=g),
    }


def test_weight_signals_identity_scale_and_gain():
    b = _w()
    s = {k: v.clone() for k, v in b.items()}
    z = D.weight_signals(b, s)
    for k in ("in_cos", "down_cos", "in_abs", "down_abs", "gate_rel"):
        assert np.allclose(z[k], 0, atol=1e-6), k
    s["gate"][2] *= 3.0  # pure rescale: no direction change, but a norm change
    s["down"][:, 4] = -s["down"][:, 4]  # flipped write
    z = D.weight_signals(b, s)
    assert z["gate_cos"][2] < 1e-6 and z["gate_rel"][2] > 1.9
    assert abs(z["down_cos"][4] - 2.0) < 1e-5
    assert z["in_abs"][[0, 1, 3, 4, 5]].max() < 1e-6
    s = {k: v.clone() for k, v in b.items()}
    s["gain"][0] += 2.0  # norm-gain change moves every neuron's effective input row
    z = D.weight_signals(b, s)
    assert (z["in_abs"] > 0).all()


def test_js_distance_bounds():
    h = np.array([[5, 5, 0, 0], [5, 5, 0, 0]])
    assert np.allclose(D.js_distance(h, h), 0)
    g = np.array([[0, 0, 5, 5], [5, 5, 0, 0]])
    js = D.js_distance(h, g)
    assert abs(js[0] - 1.0) < 1e-6 and abs(js[1]) < 1e-6


def test_tail_mass_matches_samples():
    rng = np.random.default_rng(0)
    x = np.stack([rng.standard_t(3, 200_000), rng.normal(-1, 0.5, 200_000)], 1)
    h = hist_of(x)
    for pole, q in ((0, 0.999), (0, 0.9), (1, 0.001), (1, 0.05)):
        t = np.quantile(x, q, axis=0)
        got = D.tail_mass(h, t, pole, SPEC)
        want = (x >= t).mean(0) if pole == 0 else (x <= t).mean(0)
        assert np.allclose(got, want, rtol=0.1, atol=2e-4), (pole, q, got, want)
        # self-consistent with the census quantile reader: tail mass at its own quantile
        from neuronology.census import hist_quantiles

        tq = hist_quantiles(torch.as_tensor(h), [q], SPEC)[:, 0].numpy()
        tail = q if pole == 1 else 1 - q
        assert np.allclose(D.tail_mass(h, tq, pole, SPEC), tail, rtol=1e-3), (pole, q)


def test_topk_turnover():
    a = np.array([[1, 10], [2, 11], [3, 12], [4, -1]])
    b = np.array([[4, 13], [3, 14], [9, 15], [8, 16]])
    t = D.topk_turnover(a, b)
    assert np.allclose(t, [0.5, 1.0])
    assert np.allclose(D.topk_turnover(a, a[::-1]), [0.0, 0.25])


def _signals(L=3, N=50, seed=0):
    rng = np.random.default_rng(seed)
    s = {k: rng.random((L, N)).astype(np.float32) * 0.01 for k in ("in_cos", "down_cos")}
    s.update({k: rng.random((L, N)).astype(np.float32) for k in ("in_abs", "down_abs", "js")})
    for k in ("q99_norm", "q999_norm", "q99_raw", "q999_raw", "exceed999_logratio"):
        s[k] = rng.normal(0, 1, (L, N, 2)).astype(np.float32)
    s["topk_turnover"] = rng.random((L, N, 2)).astype(np.float32)
    return s


def test_dormant_score_prefers_changed_weights_and_quiet_activations():
    s = _signals()
    # unit (1, 7): largest weight change, no activation change on either pole
    s["in_cos"][1, 7] = s["down_cos"][1, 7] = 1.0
    s["in_abs"][1, 7] = s["down_abs"][1, 7] = 10.0
    for k in ("q99_norm", "q999_norm", "q99_raw", "q999_raw", "exceed999_logratio"):
        s[k][1, 7] = 0.0
    s["js"][1, 7] = 0.0
    s["topk_turnover"][1, 7] = 0.0
    # unit (2, 3): same weight change, but its activations changed a lot
    s["in_cos"][2, 3] = s["down_cos"][2, 3] = 0.9
    for k in ("q99_norm", "q999_norm", "q99_raw", "q999_raw"):
        s[k][2, 3] = 50.0
    sc = D.suspect_scores(s)
    from neuronology.attribution import ranking, unflat

    top = [tuple(int(v) for v in unflat(f, 50))[:2] for f in ranking(sc["dormant_norm"], 2)]
    assert top == [(1, 7), (1, 7)]
    assert sc["combined_norm"][2, 3].max() > sc["combined_norm"][1, 7].max()
    for v in sc.values():
        assert v.shape == (3, 50, 2) and v.min() >= 0 and v.max() <= 1


def test_contrast_cancels_shared_changes_and_keeps_magnitude():
    sigs = {k: _signals(seed=i) for i, k in enumerate("ABC")}
    for k in "ABC":  # every suspect: same small noise level, one shared big change
        sigs[k]["in_cos"][0, 1] = sigs[k]["in_abs"][0, 1] = 50.0
    sigs["A"]["in_cos"][2, 9] = sigs["A"]["in_abs"][2, 9] = 50.0  # particular to A
    own = {k: D.suspect_scores(v) for k, v in sigs.items()}
    out = D.contrast_scores(sigs, own)
    from neuronology.attribution import ranking, unflat

    for fam in ("norm", "raw"):
        top = [tuple(int(v) for v in unflat(f, 50))[:2] for f in ranking(
            out["A"][f"weight_in_{fam}_contrast"], 2)]  # fmt: skip
        assert top == [(2, 9), (2, 9)], fam
        assert out["A"][f"weight_in_{fam}_contrast"][0, 1].max() < 0.99
    assert out["B"]["weight_in_norm_contrast"][2, 9].max() < 0.5
    assert set(out["A"]) >= {"dormant_norm_contrast", "combined_raw_contrast"}


def test_select_candidates_one_pole_per_neuron():
    rng = np.random.default_rng(0)
    sc = {"a": rng.random((2, 10, 2)), "b": rng.random((2, 10, 2))}
    sc["a"][0, 3] = 5.0  # both poles of one neuron on top
    c = D.select_candidates(sc, {"a": 3, "b": 4}, 10)
    assert len(c) == 7
    assert (c[0]["layer"], c[0]["neuron"]) == (0, 3)
    keys = [(x["layer"], x["neuron"]) for x in c]
    assert len(set(keys)) == len(keys)
    assert [x["source"] for x in c] == ["a"] * 3 + ["b"] * 4


def test_weight_reader_handles_vlm_and_causal_layouts(tmp_path):
    from safetensors.torch import save_file

    w = _w()
    pre = {"vlm": "model.language_model.", "lm": "model."}
    for name, p in pre.items():
        d = tmp_path / name
        d.mkdir()
        tensors = {
            f"{p}layers.0.mlp.gate_proj.weight": w["gate"],
            f"{p}layers.0.mlp.up_proj.weight": w["up"],
            f"{p}layers.0.mlp.down_proj.weight": w["down"],
            f"{p}layers.0.post_attention_layernorm.weight": w["gain"] - 1.0,
            f"{p}layers.0.self_attn.q_proj.weight": torch.ones(3, 5),
            "mtp.layers.0.mlp.up_proj.weight": torch.zeros(2, 2),
        }
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(d / "model.safetensors"))
        (d / "config.json").write_text('{"text_config": {"model_type": "qwen3_5_text"}}')
    rv, rl = D.WeightReader(tmp_path / "vlm"), D.WeightReader(tmp_path / "lm")
    assert rv.gain_offset == 1.0
    mv, ml = D.mlp_weights(rv, 0), D.mlp_weights(rl, 0)
    assert torch.allclose(mv["gain"], w["gain"]) and torch.equal(mv["up"], ml["up"])
    z = D.weight_signals(mv, ml)
    assert np.allclose(z["in_abs"], 0)
    assert set(D.layer_rel_change(rv, rl, 0)) >= {"mlp.up_proj.weight", "self_attn.q_proj.weight"}


def test_select_candidates_pole_preference():
    sc = {"a": np.zeros((1, 3, 2))}
    sc["a"][0, 1, 1] = 1.0  # ranking found the negative pole
    pref = np.zeros((1, 3, 2))
    pref[0, 1, 0] = 1.0  # but the positive tail is the one that moved outwards
    c = D.select_candidates(sc, {"a": 1}, 3, pref)
    assert (c[0]["neuron"], c[0]["pole"], c[0]["flat"]) == (1, 0, 2)
