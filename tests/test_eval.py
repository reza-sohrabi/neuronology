"""eval.py: chunked NLL/KL equals the direct computation; bootstrap behaves."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from neuronology.eval import (
    bootstrap_weights,
    nll_and_kl,
    per_doc_sums,
    ratio_ci,
    shift_targets,
    weighted_mean_ci,
)


def test_nll_kl_chunked_matches_direct():
    g = torch.Generator().manual_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    h1, h2 = torch.randn(37, 16, generator=g), torch.randn(37, 16, generator=g)
    t = torch.randint(0, 50, (37,), generator=g)
    n1, n2, kl = nll_and_kl(head, h1, h2, t, chunk=8)
    lp1, lp2 = F.log_softmax(head(h1), -1), F.log_softmax(head(h2), -1)
    torch.testing.assert_close(n1, F.nll_loss(lp1, t, reduction="none"))
    torch.testing.assert_close(n2, F.nll_loss(lp2, t, reduction="none"))
    ref = F.kl_div(lp2, lp1, log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(kl, ref, rtol=1e-4, atol=1e-6)
    _, _, kl0 = nll_and_kl(head, h1, h1.clone(), t, chunk=8)
    assert kl0.abs().max() < 1e-5


def test_shift_targets_respects_padding():
    ids = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    valid, tgt = shift_targets(ids, mask)
    assert valid.tolist() == [[True, True, False, False], [True, False, False, False]]
    assert tgt[valid].tolist() == [6, 7, 9]


def test_per_doc_sums():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rows = torch.tensor([0, 2, 2, 0])
    np.testing.assert_allclose(per_doc_sums(v, rows, 3), [5.0, 0.0, 5.0])


def test_bootstrap():
    W = bootstrap_weights(50, 400, seed=0)
    assert W.shape == (400, 50) and np.all(W.sum(1) == 50)
    rng = np.random.default_rng(0)
    tok = rng.integers(10, 100, 50).astype(float)
    s = tok * rng.normal(2.0, 0.5, 50)
    ci = weighted_mean_ci(s, tok, W, 0.95)
    assert ci["lo"] < ci["mean"] < ci["hi"] and abs(ci["mean"] - 2.0) < 0.3
    r = ratio_ci(s, 2 * s, tok, W, 0.95)
    assert abs(r["ratio"] - 0.5) < 1e-12 and abs(r["lo"] - 0.5) < 1e-9
