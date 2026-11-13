"""sparsify.py: masks keep exactly the intended neurons, fills and conditions."""

from __future__ import annotations

import pytest
import torch

from neuronology.sparsify import (
    Condition,
    conditions_from_config,
    keep_mask,
    make_sparsify_fn,
    top_mask,
)


@pytest.fixture
def acts():
    return torch.randn(2, 5, 64, generator=torch.Generator().manual_seed(0))


def test_top_keeps_largest_abs(acts):
    c = Condition("top", 0.125)
    m = keep_mask(acts, c)
    assert (m.sum(-1) == 8).all()
    kept_min = torch.where(m, acts.abs(), torch.inf).amin(-1)
    dropped_max = torch.where(~m, acts.abs(), -torch.inf).amax(-1)
    assert (kept_min > dropped_max).all()


def test_bottom_is_complement_of_top(acts):
    top = keep_mask(acts, Condition("top", 0.125))
    bot = keep_mask(acts, Condition("bottom", 0.125))
    assert torch.equal(bot, ~top)
    assert Condition("bottom", 0.125).n_kept(64) == 56


def test_random_keeps_k_and_varies(acts):
    g = torch.Generator().manual_seed(1)
    m = keep_mask(acts, Condition("random", 0.25), generator=g)
    assert (m.sum(-1) == 16).all()
    assert not torch.equal(m[0, 0], m[0, 1])


def test_wnorm_ranking(acts):
    wn = torch.ones(64)
    wn[3] = 1e6
    m = keep_mask(acts, Condition("top", 1 / 64, "abs_wnorm"), w_norm=wn)
    assert m[..., 3].all() and (m.sum(-1) == 1).all()


def test_fill_mean_and_zero(acts):
    means = {0: torch.full((64,), 0.5)}
    out = make_sparsify_fn(Condition("top", 0.25), means)(0, acts)
    m = keep_mask(acts, Condition("top", 0.25))
    assert torch.equal(out[m], acts[m]) and (out[~m] == 0.5).all()
    out0 = make_sparsify_fn(Condition("top", 0.25, fill="zero"), means)(0, acts)
    assert (out0[~m] == 0).all()


def test_top_mask_exact_k_with_ties():
    s = torch.zeros(3, 10)
    assert (top_mask(s, 4).sum(-1) == 4).all()


def test_conditions_from_config():
    cfg = {
        "keep_fracs": [0.5, 0.125],
        "conditions": [
            {"mode": "top", "ranks": ["abs", "abs_wnorm"]},
            {"mode": "top", "fills": ["zero"]},
            {"mode": "random", "ranks": ["abs_wnorm"]},
            {"mode": "bottom"},
        ],
    }
    names = [c.name for c in conditions_from_config(cfg)]
    assert len(names) == len(set(names)) == 10
    assert "random_f0.125_mean" in names and "top_abs_f0.5_zero" in names
