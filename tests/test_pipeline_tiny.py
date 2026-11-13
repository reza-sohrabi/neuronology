"""End-to-end census + sparsity runners on a random tiny Qwen3 and a synthetic corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from neuronology.census import run_census
from neuronology.data import Corpus
from neuronology.hooks import capture, capture_sites, stream_acts
from neuronology.model import arch_info
from neuronology.sparsify import run_sparsity


@dataclass
class FakeLM:
    model: torch.nn.Module
    tokenizer: object
    arch: object
    device: torch.device


class Tok:
    pad_token_id = 0


def _setup(tmp_path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
    )
    model = Qwen3ForCausalLM(cfg).eval()
    rng = np.random.default_rng(0)
    lengths = rng.integers(8, 40, 30)
    tokens = rng.integers(1, 97, lengths.sum()).astype(np.uint32)
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    split = np.where(np.arange(30) < 20, "fit", "eval")
    pd.DataFrame(
        {
            "doc_id": np.arange(30, dtype=np.int32),
            "source": "x",
            "split": split,
            "offset": offs,
            "n_tokens": lengths.astype(np.int32),
            "src_row": 0,
            "sha1": "x",
        }
    ).to_parquet(tmp_path / "docs.parquet")
    np.save(tmp_path / "tokens.npy", tokens)
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": "x"}))
    lm = FakeLM(model, Tok(), arch_info(model), torch.device("cpu"))
    return lm, Corpus(tmp_path)


CENSUS_CFG = {
    "split": "fit",
    "batch_tokens": 128,
    "sort_window": 8,
    "hist": {"bins_per_side": 64, "log10_min": -6.0, "log10_max": 3.0},
    "pos_quantiles": [0.75, 0.9, 0.99],
    "neg_quantiles": [0.25, 0.1, 0.01],
    "top_k": 4,
    "reservoir": {"k": 3, "max_tokens": 200, "seed": 0},
    "checkpoint_minutes": 0,
}


def test_stream_acts_matches_capture(tmp_path):
    lm, _ = _setup(tmp_path)
    ids = torch.randint(1, 97, (2, 9))
    seen = {}
    with stream_acts(lm.model, lambda lyr, a, zg: seen.__setitem__(lyr, (a, zg)), with_gate=True):
        with capture_sites(lm.model, ("act", "z_gate")) as st:
            lm.model(ids)
    for lyr in range(2):
        torch.testing.assert_close(seen[lyr][0], st["act"][lyr])
        torch.testing.assert_close(seen[lyr][1], st["z_gate"][lyr])


def test_census_runner_and_resume(tmp_path):
    lm, corpus = _setup(tmp_path)
    out = run_census(lm, corpus, CENSUS_CFG, ckpt_path=tmp_path / "ck.pt", log=lambda m: None)
    st = out["stats"]
    n_fit = int(corpus.split_docs("fit")["n_tokens"].sum())
    assert st["n_tokens"].tolist() == [n_fit, n_fit]
    # top-1 context really is the max activation over the whole fit split
    docs = corpus.split_docs("fit")
    with capture(lm.model) as acts:
        allmax = np.full((2, 48), -np.inf)
        for _, r in docs.iterrows():
            ids = torch.as_tensor(corpus.tokens[r.offset : r.offset + r.n_tokens].astype(np.int64))
            lm.model(ids[None])
            for lyr in range(2):
                allmax[lyr] = np.maximum(allmax[lyr], acts[lyr][0].amax(0).numpy())
    np.testing.assert_allclose(st["top_pos_val"][:, 0, :], allmax, rtol=1e-4, atol=1e-5)
    assert out["reservoir"]["res_idx"].shape == (2, 6, 3, 48)
    # a resumed run (checkpoint at phase 3) reproduces the result
    out2 = run_census(lm, corpus, CENSUS_CFG, ckpt_path=tmp_path / "ck.pt", log=lambda m: None)
    np.testing.assert_array_equal(out2["stats"]["thresholds"], st["thresholds"])


def test_sparsity_runner(tmp_path):
    lm, corpus = _setup(tmp_path)
    means = {lyr: torch.zeros(48) for lyr in range(2)}
    cfg = {
        "split": "eval",
        "batch_tokens": 128,
        "sort_window": 8,
        "seed": 0,
        "logit_chunk": 16,
        "keep_fracs": [0.5, 0.125],
        "conditions": [
            {"mode": "top", "ranks": ["abs", "abs_wnorm"]},
            {"mode": "random"},
            {"mode": "bottom"},
        ],
    }
    out = run_sparsity(lm, corpus, means, cfg, ckpt_path=None, log=lambda m: None)
    st = out["st"]
    n_eval_pred = int((corpus.split_docs("eval")["n_tokens"] - 1).sum())
    assert st["tokens"].sum() == n_eval_pred
    assert np.all(st["kl"] >= -1e-6)
    names = [c.name for c in out["conditions"]]
    kl = dict(zip(names, st["kl"].sum(1), strict=True))
    # keeping half by |a| hurts less than keeping 12.5% by |a|
    assert kl["top_abs_f0.5_mean"] < kl["top_abs_f0.125_mean"]
    shares = out["quad"].summary()["all_cells"]
    assert abs(sum(shares) - 1) < 1e-9
