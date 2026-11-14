"""modes.py on random tiny Qwen3 / Qwen3.5 models (CPU, fp32, no download) and synthetic data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from neuronology import attribution as A
from neuronology import modes as M
from tests.test_attribution import IDS, tiny_qwen3, tiny_qwen35


@pytest.fixture(scope="module", params=["qwen3", "qwen3_5"])
def model(request):
    m = tiny_qwen3() if request.param == "qwen3" else tiny_qwen35()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _parts_at(model, ids, positions):
    x = torch.tensor([ids])
    return M.capture_at(
        model, x, torch.zeros(len(positions), dtype=torch.long), torch.tensor(positions)
    )


def test_pole_parse_and_threshold():
    p = M.Pole.parse("L3_N17_neg")
    assert (p.layer, p.neuron, p.pole, p.sign) == (3, 17, 1, -1.0)
    assert M.Pole.parse(p.key) == p
    assert M.Pole.parse("L1_N2_0").pole == 0
    thr = np.zeros((4, 20, 12))
    thr[3, 17, 2] = 0.7  # p95
    thr[3, 17, 8] = -0.4  # p5
    q = [0.75, 0.9, 0.95, 0.97, 0.99, 0.999]
    assert M.fire_threshold(thr, M.Pole(3, 17, 0), q) == pytest.approx(0.7)
    assert M.fire_threshold(thr, M.Pole(3, 17, 1), q) == pytest.approx(0.4)
    thr[3, 17, 2] = -0.2  # an always-negative neuron: pos threshold clamps to 0
    assert M.fire_threshold(thr, M.Pole(3, 17, 0), q) == 0.0
    with pytest.raises(ValueError):
        M.fire_threshold(thr, p, q, q=0.93)


def test_capture_at_matches_capture_parts_and_padding(model):
    pos = [3, 7, 11]
    full = A.capture_parts(model, IDS)
    got = _parts_at(model, IDS, pos)
    assert torch.allclose(got["embed"], full.embed[pos], atol=1e-6)
    for lyr in full.act:
        assert torch.allclose(got["act"][lyr], full.act[lyr][pos], atol=1e-5)
        assert torch.allclose(got["mixer_in"][lyr], full.mixer_in[lyr][pos], atol=1e-5)
        assert torch.allclose(got["h_mid"][lyr], full.h_mid[lyr][pos], atol=1e-5)
    # batched with right padding: row 1 is a shorter sequence
    short = IDS[:8]
    ids = torch.tensor([IDS, short + [0] * (len(IDS) - 8)])
    mask = torch.tensor([[1] * len(IDS), [1] * 8 + [0] * (len(IDS) - 8)])
    b = M.capture_at(model, ids, torch.tensor([0, 1]), torch.tensor([11, 7]), mask)
    ref_s = A.capture_parts(model, short)
    for lyr in full.act:
        assert torch.allclose(b["act"][lyr][0], full.act[lyr][11], atol=1e-4)
        assert torch.allclose(b["act"][lyr][1], ref_s.act[lyr][7], atol=1e-4)


def _firing(model, layer):
    """A (neuron, pole, position) with a clearly non-zero activation at the given layer."""
    full = A.capture_parts(model, IDS)
    a = full.act[layer]
    p, n = divmod(int(a[2:].abs().argmax()), a.shape[1])
    p += 2
    pole = 0 if a[p, n] > 0 else 1
    return n, pole, p, full


@pytest.mark.parametrize("layer", [0, 2])
def test_profile_matches_input_profile_single_position(model, layer):
    n, pole, p, full = _firing(model, layer)
    pc = M.ProfileComputer(model, M.Pole(layer, n, pole), n_baseline=16, seed=3, top=10)
    parts = _parts_at(model, IDS, [p])
    res = pc.compute(parts)
    ref = A.input_profile(model, [(full, [p])], layer, n, pole, n_baseline=16, top=10, seed=3)
    for li, lane in enumerate(M.LANES):
        mine = res["prof_val"][0, li]
        theirs = np.array([abs(r["excess"]) for r in ref[lane]])
        units = np.array([r["unit"] for r in ref[lane]])
        k = int((mine > 0).sum())
        assert k > 0
        np.testing.assert_allclose(mine[:k], theirs[:k], rtol=1e-4, atol=1e-6)
        assert set(res["prof_idx"][0, li, :k].tolist()) == set(units[:k].tolist())
    assert pc.raw_residual(parts) < 1e-3
    assert res["act"][0] == pytest.approx(abs(float(full.act[layer][p, n])), rel=1e-5)


def test_profile_batch_equals_single(model):
    n, pole, p, _ = _firing(model, 1)
    pc = M.ProfileComputer(model, M.Pole(1, n, pole), n_baseline=8, seed=0, top=12)
    both = pc.compute(_parts_at(model, IDS, [p, 9]))
    one = pc.compute(_parts_at(model, IDS, [9]))
    np.testing.assert_allclose(both["prof_val"][1], one["prof_val"][0], rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(both["rd_idx"][1], one["rd_idx"][0])


def test_readers_match_reader_effects(model):
    n, pole, p, full = _firing(model, 1)
    pc = M.ProfileComputer(model, M.Pole(1, n, pole), n_baseline=4, reader_top=15)
    res = pc.compute(_parts_at(model, IDS, [p]))
    ref = A.reader_effects(model, full, 1, n, p)["later"]
    N = full.act[1].shape[1]
    for f, v in zip(res["rd_idx"][0], res["rd_val"][0], strict=True):
        if f < 0:
            continue
        assert v == pytest.approx(float(ref[int(f) // N][int(f) % N]), rel=1e-4, abs=1e-6)
    last = M.ProfileComputer(model, M.Pole(len(full.act) - 1, 0, 0), n_baseline=4)
    out = last.compute(_parts_at(model, IDS, [p]))
    assert (out["rd_idx"] == -1).all()


def test_firing_sampler_uniform_and_counts():
    s = M.FiringSampler(k=50, thr=0.5, other_thr=0.5, seed=0, device="cpu")
    rng = np.random.default_rng(0)
    seen = 0
    for b in range(20):
        a = torch.tensor(rng.random(100), dtype=torch.float32)
        o = torch.tensor(rng.random(100), dtype=torch.float32)
        s.update(a, torch.arange(b * 100, b * 100 + 100), o)
        seen += int((a > 0.5).sum())
    r = s.result()
    assert len(r["gpos"]) == 50 and r["n_fire"] == seen and r["n_tokens"] == 2000
    assert (r["act"] > 0.5).all() and np.all(np.diff(r["gpos"]) > 0)
    assert r["gpos"].min() < 600 and r["gpos"].max() > 1400  # spread over the stream
    s2 = M.FiringSampler(k=50, thr=0.5, other_thr=0.5, seed=1, device="cpu")
    s2.load_state_dict(s.state_dict())
    np.testing.assert_array_equal(s2.result()["gpos"], r["gpos"])


def _two_cluster_profiles(n_per=40, n_units=200, seed=0):
    rng = np.random.default_rng(seed)
    F = 2 * n_per
    idx = np.full((F, 4, 10), -1, dtype=np.int32)
    val = np.zeros((F, 4, 10), dtype=np.float32)
    for f in range(F):
        lo = 0 if f < n_per else 100
        for b in range(4):
            idx[f, b] = rng.choice(np.arange(lo, lo + 12), 10, replace=False)
            val[f, b] = rng.random(10) + 0.1
    return idx, val, n_units


def test_tfidf_norm_and_idf():
    idx, val, U = _two_cluster_profiles()
    fit = np.ones(len(idx), dtype=bool)
    X, idf = M.tfidf_matrix(idx, val, U, fit)
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel())
    np.testing.assert_allclose(nrm, 1.0, atol=1e-6)
    assert idf.shape == (4, U) and idf[0, 150] > idf[0, 5]  # never-seen unit gets the max idf
    X2, _ = M.tfidf_matrix(idx[:5], val[:5], U, fit[:5], idf=idf)
    np.testing.assert_allclose(X2.toarray(), X[:5].toarray())


def test_knn_leiden_merge_assign():
    idx, val, U = _two_cluster_profiles(n_per=40)
    X, _ = M.tfidf_matrix(idx, val, U, np.ones(len(idx), dtype=bool))
    nbr, sim = M.knn(X, X, 15, exclude_self=True)
    assert nbr.shape == (80, 15) and not (nbr == np.arange(80)[:, None]).any()
    lab = M.leiden_labels(nbr, sim, resolution=1.0, seed=0, min_size=15)
    assert len(set(lab[:40])) == 1 and len(set(lab[40:])) == 1 and lab[0] != lab[40]
    # min_size drops everything when no group is that big
    assert (M.leiden_labels(nbr, sim, min_size=1000) == -1).all()
    # merge: two clusters with the same inputs merge, disjoint ones do not
    split = np.where(np.arange(80) < 20, 0, np.where(np.arange(80) < 40, 1, 2))
    merged, log = M.merge_clusters(X, None, split, top=40, thr=0.3, min_size=1)
    assert len(log) == 1 and len(set(merged[:40])) == 1 and merged[0] != merged[79]
    # readers block a merge when they are disjoint
    rd_idx = np.where(np.arange(80)[:, None] < 20, np.arange(5), np.arange(5) + 50)
    R = M.reader_matrix(rd_idx, np.ones((80, 5), dtype=np.float32), 100)
    _, log2 = M.merge_clusters(X, R, split, top=5, thr=0.3, min_size=1)
    assert log2 == []
    # held-out assignment by neighbour vote
    nb_h, _ = M.knn(X[[3, 70]], X, 15)
    np.testing.assert_array_equal(M.assign_by_knn(nb_h, lab), [lab[3], lab[70]])


def test_marker_rule():
    lab = np.array([0] * 100 + [1] * 60 + [-1] * 10)
    src = np.array(
        [M.SRC_CORPUS] * 50
        + [M.SRC_TRIG] * 25
        + [M.SRC_CLEAN] * 25
        + [M.SRC_TRIG] * 40
        + [M.SRC_CLEAN] * 2
        + [M.SRC_CORPUS] * 18
        + [M.SRC_TRIG] * 10
    )
    r = M.marker_subcircuit(lab, src, n_pos_trig=1000, n_pos_clean=1000)
    assert r["marker"] == 1
    assert {x["subcircuit"]: x["n_trig"] for x in r["rows"]} == {0: 25, 1: 40}
    # not enough excess -> no marker
    assert M.marker_subcircuit(lab, src, 1000, 1000, min_excess=100)["marker"] is None
    # clean prefixes 3x longer: subcircuit 0 is then enriched too, but less than 1
    r3 = M.marker_subcircuit(lab, src, 1000, 3000)
    assert r3["marker"] == 1


def test_bootstrap_shares():
    lab = np.array([0, 0, 1, 1, -1])
    kl = np.array([1.0, 1.0, 2.0, np.nan, 0.0])
    s = M.bootstrap_shares(lab, kl, n_boot=50)
    assert s[0]["share"] == pytest.approx(0.5) and s[1]["share"] == pytest.approx(0.5)
    assert s[1]["n"] == 1 and s[-1]["share"] == 0.0


def test_specific_drivers_and_random_match(model):
    idx = np.full((4, 4, 3), -1, dtype=np.int32)
    val = np.zeros((4, 4, 3), dtype=np.float32)
    idx[:2, 0] = [0, 5, 6]
    val[:2, 0] = [9.0, 2.0, 1.0]
    idx[2:, 0] = [6, 7, -1]
    val[2:, 0] = [3.0, 1.0, 0.0]
    mem = np.array([True, True, False, False])
    u, s = M.specific_drivers(idx, val, mem, ~mem, n_units=10, k=2)
    assert list(u) == [5, 8] or u[0] == 5  # embedding (0) excluded; 5 most specific
    assert s[0] == pytest.approx(2.0)
    lo = A.unit_layout(model, 2)
    units = np.array([lo.mixer_offset(1), lo.neuron_offset(1) + 3])
    r = M.matched_random_units(units, lo, np.random.default_rng(0))
    for a, b in zip(units, r, strict=True):
        la, lb = lo.label(int(a)), lo.label(int(b))
        assert la["type"] == lb["type"] and la["layer"] == lb["layer"] and a != b


def test_ablation_kl_identity_and_reference(model):
    n, pole, p, full = _firing(model, 1)
    P = M.Pole(1, n, pole)
    seqs = [IDS[: p + 1], IDS[:6]]
    same = float(full.act[1][p, n])
    r0 = M.ablation_kl(model, seqs[:1], P, fill=same)
    assert r0["kl"][0] == pytest.approx(0.0, abs=1e-6)
    r = M.ablation_kl(model, seqs, P, fill=0.0, target_ids=[1, 2])
    # reference: two separate forwards, ablating position p only
    from neuronology.hooks import intervene

    def fn(layer, acts):
        out = acts.clone()
        out[0, -1, n] = 0.0
        return out

    x = torch.tensor([seqs[0]])
    head = model.get_output_embeddings()
    lc = torch.log_softmax(head(model.model(input_ids=x).last_hidden_state[0, -1]).float(), -1)
    with intervene(model, fn, [1]):
        la = torch.log_softmax(head(model.model(input_ids=x).last_hidden_state[0, -1]).float(), -1)
    ref = float((lc.exp() * (lc - la)).sum())
    assert r["kl"][0] == pytest.approx(ref, rel=1e-3, abs=1e-7)
    assert np.isfinite(r["margin_clean"]).all()


def test_forward_last_and_ablate_units(model):
    n, pole, p, full = _firing(model, 2)
    P = M.Pole(2, n, pole)
    seqs = [IDS[: p + 1], IDS[:7], IDS]
    ref = M.forward_last(model, seqs, P, target_ids=[1, 1, 1], keep_logp=True)
    again = M.forward_last(model, seqs, P, ref_logp=ref["logp"])
    np.testing.assert_allclose(again["kl"], 0.0, atol=1e-6)
    assert ref["act"][0] == pytest.approx(abs(float(full.act[2][p, n])), rel=1e-4)
    lo = A.unit_layout(model, 2)
    means = np.zeros((len(full.act), full.act[0].shape[1]))
    hm = {lyr: torch.zeros(full.mixer_in[lyr].shape[1]) for lyr in full.mixer_in}
    units = np.array([lo.mixer_offset(0), lo.neuron_offset(1) + 2])
    abl = M.forward_last(
        model,
        seqs,
        P,
        ctx_factory=lambda: M.ablate_units(model, units, lo, means, hm),
        ref_logp=ref["logp"],
    )
    assert abl["kl"].sum() > 0 and (abl["kl"] >= 0).all()
    # head ablation changes the residual right after that layer's mixer
    with M.ablate_units(model, units[:1], lo, means, hm):
        parts = A.capture_parts(model, IDS)
    assert not torch.allclose(parts.h_mid[0], full.h_mid[0])
    with pytest.raises(ValueError):
        with M.ablate_units(model, np.array([0]), lo, means, hm):
            pass


def test_modes_scripts_stay_audit_side():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "scripts"
    for name in ("modes_run.py", "modes_describe.py"):
        text = (root / name).read_text()
        for bad in ("load_secret", "secrets_dir", "import backdoor", "neuronology.backdoor"):
            assert bad not in text, (name, bad)


def test_parse_modes():
    from scripts.modes_describe import parse_modes

    ids = ["0", "1", "2"]
    modes, labels, ok = parse_modes(
        'x {"modes": [{"subcircuits": [0, 2], "label": "a"}, '
        '{"subcircuits": [1], "label": "b"}]} y',
        ids,
    )
    assert ok and modes == [["0", "2"], ["1"]] and labels == ["a", "b"]
    # a subcircuit missing or repeated -> one mode per subcircuit
    for bad in (
        '{"modes": [{"subcircuits": [0, 1]}]}',
        '{"modes": [{"subcircuits": [0, 1, 1, 2]}]}',
        "not json",
    ):
        modes, _, ok = parse_modes(bad, ids)
        assert not ok and modes == [["0"], ["1"], ["2"]]
