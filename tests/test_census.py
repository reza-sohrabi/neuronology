"""census.py: histogram binning and quantiles, streaming accumulators vs brute force."""

from __future__ import annotations

import numpy as np
import torch

from neuronology.census import (
    CensusAccumulator,
    HistSpec,
    ReservoirAccumulator,
    hist_quantiles,
    quadrant_index,
)

SPEC = HistSpec(bins_per_side=256, log10_min=-6.0, log10_max=4.0)


def test_bin_index_matches_edges():
    edges = SPEC.edges()
    assert len(edges) == SPEC.n_bins + 1 and np.all(np.diff(edges) > 0)
    x = torch.tensor(
        [-3e4, -5.0, -1e-3, -1e-7, 0.0, 2e-7, 1e-3, 0.7, 5.0, 3e4], dtype=torch.float64
    )
    idx = SPEC.bin_index(x).numpy()
    for v, i in zip(x.numpy(), idx, strict=True):
        if abs(v) >= 1e4:  # clamped into the outermost bin
            assert i in (0, SPEC.n_bins - 1)
            continue
        assert edges[i] <= v < edges[i + 1] or (i == SPEC.bins_per_side), (v, i)


def test_hist_quantiles_close_to_exact():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200_000, 3, generator=g) * torch.tensor([0.01, 1.0, 50.0]) + torch.tensor(
        [0.0, 0.5, -20.0]
    )
    x[:, 0] = x[:, 0].abs() * torch.sign(torch.randn(200_000, generator=g))
    hist = torch.stack(
        [torch.bincount(SPEC.bin_index(x[:, n]), minlength=SPEC.n_bins) for n in range(3)]
    )
    qs = [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999]
    est = hist_quantiles(hist, qs, SPEC).numpy()
    exact = np.quantile(x.numpy(), qs, axis=0).T
    rel = np.abs(est - exact) / (np.abs(exact) + 1e-3 * x.std(0).numpy()[:, None])
    assert rel.max() < 0.05, rel  # bin width is ~9% at 256 bins/side over 10 decades


def _brute(a, zg, k):
    top_pos = np.sort(a, 0)[::-1][:k]
    top_neg = np.sort(a, 0)[:k]
    q = quadrant_index(torch.as_tensor(a), torch.as_tensor(zg)).numpy()
    counts = np.stack([(q == i).sum(0) for i in range(4)])
    return top_pos, top_neg, counts


def test_accumulator_streaming_equals_brute_force():
    rng = np.random.default_rng(0)
    T, N, k = 700, 11, 8
    zg = rng.normal(size=(T, N)).astype(np.float32)
    up = rng.normal(size=(T, N)).astype(np.float32)
    a = (torch.nn.functional.silu(torch.as_tensor(zg)) * torch.as_tensor(up)).numpy()
    pos = np.arange(T) + 1000
    acc = CensusAccumulator(2, N, SPEC, k, torch.device("cpu"))
    for s in range(0, T, 97):  # uneven batches, some smaller than k
        sl = slice(s, s + 97)
        acc.update(1, torch.as_tensor(a[sl]), torch.as_tensor(zg[sl]), torch.as_tensor(pos[sl]))
    out = acc.finalize([0.75, 0.99], [0.25, 0.01])
    tp, tn, counts = _brute(a, zg, k)
    np.testing.assert_allclose(out["top_pos_val"][1], tp, rtol=1e-6)
    np.testing.assert_allclose(out["top_neg_val"][1], tn, rtol=1e-6)
    # positions point at the right values
    np.testing.assert_allclose(a[out["top_pos_idx"][1] - 1000, np.arange(N)], tp, rtol=1e-6)
    np.testing.assert_array_equal(out["quad_counts"][1], counts)
    np.testing.assert_allclose(out["mean"][1], a.mean(0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out["std"][1], a.std(0), rtol=1e-4)
    assert out["n_tokens"].tolist() == [0, T]
    # quadrant sanity: Q3 (g<0,u<0) has a>0
    q = quadrant_index(torch.as_tensor(a), torch.as_tensor(zg)).numpy()
    assert np.all(a[q == 2] > 0) and np.all((zg[q == 2] < 0) & (up[q == 2] < 0))


def test_state_dict_roundtrip():
    acc = CensusAccumulator(1, 4, SPEC, 3, torch.device("cpu"))
    x = torch.randn(20, 4)
    acc.update(0, x, x, torch.arange(20))
    acc2 = CensusAccumulator(1, 4, SPEC, 3, torch.device("cpu"))
    acc2.load_state_dict(acc.state_dict())
    acc.update(0, x, x, torch.arange(20, 40))
    acc2.update(0, x, x, torch.arange(20, 40))
    a, b = acc.finalize([0.9], [0.1]), acc2.finalize([0.9], [0.1])
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_reservoir_bands_and_counts():
    rng = np.random.default_rng(1)
    T, N = 5000, 3
    a = rng.normal(size=(T, N)).astype(np.float32)
    pos_q, neg_q = [0.75, 0.9, 0.99], [0.25, 0.1, 0.01]
    thr = np.quantile(a, pos_q + neg_q, axis=0).T[None]  # [1, N, 6]
    res = ReservoirAccumulator(thr, 3, k=16, device=torch.device("cpu"), seed=0)
    for s in range(0, T, 600):
        res.update(0, torch.as_tensor(a[s : s + 600]), torch.arange(s, min(s + 600, T)))
    out = res.finalize()
    band = res.band_index(0, torch.as_tensor(a)).numpy()
    for b in range(6):
        np.testing.assert_array_equal(out["res_count"][0, b], (band == b).sum(0))
        idx = out["res_idx"][0, b]  # [k, N]
        for n in range(N):
            got = idx[:, n][idx[:, n] >= 0]
            assert len(got) == min(16, (band[:, n] == b).sum())
            assert np.all(band[got, n] == b)  # every sample lies in its band
            np.testing.assert_allclose(
                out["res_val"][0, b][: len(got), n].astype(np.float32),
                a[got, n],
                rtol=1e-3,
                atol=1e-3,
            )
    # band 0 of pos is [p75, p90): about 15% of tokens
    assert abs((band == 0).mean() - 0.15) < 0.02
