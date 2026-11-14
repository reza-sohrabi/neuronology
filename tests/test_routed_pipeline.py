"""OLMoE port, runner level (PLAN_OLMOE.md M0o): routed census, routed sparsity, LoRA on
experts, attribution plumbing, corpus re-tokenisation and the new-wording prompts. Tiny
random OLMoE on CPU and synthetic data; no network."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

from neuronology.census import run_census
from neuronology.data import Corpus
from neuronology.routed import (
    RoutedPoleAblation,
    active_matched_sampler,
    capture_routed,
    grad_x_act_routed,
    routed_info,
)
from neuronology.sparsify import run_sparsity

E, I, K, L, V = 8, 16, 2, 2, 97  # noqa: E741


@dataclass
class FakeLM:
    model: torch.nn.Module
    tokenizer: object
    arch: object
    device: torch.device


class Tok:
    pad_token_id = 0


def _olmoe():
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
    m = OlmoeForCausalLM(cfg).eval()
    m.set_experts_implementation("eager")
    return m


def _setup(tmp_path):
    from neuronology.model import arch_info

    model = _olmoe()
    rng = np.random.default_rng(0)
    lengths = rng.integers(8, 40, 30)
    tokens = rng.integers(1, V, lengths.sum()).astype(np.uint32)
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
    return FakeLM(model, Tok(), arch_info(model), torch.device("cpu")), Corpus(tmp_path)


CENSUS_CFG = {
    "split": "fit",
    "batch_tokens": 128,
    "sort_window": 8,
    "hist": {"bins_per_side": 64, "log10_min": -6.0, "log10_max": 3.0},
    "pos_quantiles": [0.75, 0.9, 0.99],
    "neg_quantiles": [0.25, 0.1, 0.01],
    "top_k": 4,
    "reservoir": None,
    "checkpoint_minutes": 0,
}


def _fit_docs(corpus):
    docs = corpus.split_docs("fit")
    for _, r in docs.iterrows():
        yield torch.as_tensor(corpus.tokens[r.offset : r.offset + r.n_tokens].astype(np.int64))


def test_routed_census_counts_means_and_maxima(tmp_path):
    lm, corpus = _setup(tmp_path)
    out = run_census(lm, corpus, CENSUS_CFG, ckpt_path=tmp_path / "ck.pt", log=lambda m: None)
    st = out["stats"]
    assert out["reservoir"] is None
    U = E * I
    cnt = np.zeros((L, U))
    tot = np.zeros((L, U))
    mx = np.full((L, U), -np.inf)
    with torch.no_grad(), capture_routed(lm.model) as acts:
        for ids in _fit_docs(corpus):
            lm.model(ids[None])
            for lyr in range(L):
                d = acts[lyr].dense(E).numpy()  # [T, U], 0 where not routed
                routed = np.repeat(acts[lyr].routed_mask(E).numpy(), I, axis=1)
                cnt[lyr] += routed.sum(0)
                tot[lyr] += d.sum(0)
                mx[lyr] = np.maximum(mx[lyr], np.where(routed, d, -np.inf).max(0))
    np.testing.assert_array_equal(st["n_routed"], cnt)
    seen = cnt > 0
    np.testing.assert_allclose(
        st["mean"][seen], (tot / np.maximum(cnt, 1))[seen], rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(st["top_pos_val"][:, 0, :][seen], mx[seen], rtol=1e-4, atol=1e-6)
    assert (st["thresholds"][~seen] == 0).all()
    # quadrant counts cover every routed cell exactly once
    np.testing.assert_array_equal(st["quad_counts"].sum(1), cnt)
    # resume from the phase-3 checkpoint reproduces the stats
    out2 = run_census(lm, corpus, CENSUS_CFG, ckpt_path=tmp_path / "ck.pt", log=lambda m: None)
    np.testing.assert_array_equal(out2["stats"]["thresholds"], st["thresholds"])


def test_census_stop_after_hours_then_resume(tmp_path):
    from neuronology.runinfo import StopEarly

    lm, corpus = _setup(tmp_path)
    ck = tmp_path / "ck.pt"
    with pytest.raises(StopEarly):
        run_census(lm, corpus, {**CENSUS_CFG, "stop_after_hours": 0.0}, ck, log=lambda m: None)
    resumed = run_census(lm, corpus, CENSUS_CFG, ck, log=lambda m: None)
    fresh = run_census(lm, corpus, CENSUS_CFG, None, log=lambda m: None)
    np.testing.assert_array_equal(resumed["stats"]["n_routed"], fresh["stats"]["n_routed"])
    np.testing.assert_allclose(resumed["stats"]["mean"], fresh["stats"]["mean"], rtol=1e-5)


def test_routed_sparsity_runner(tmp_path):
    lm, corpus = _setup(tmp_path)
    means = {lyr: torch.zeros(E * I) for lyr in range(L)}
    cfg = {
        "split": "eval",
        "batch_tokens": 128,
        "sort_window": 8,
        "seed": 0,
        "logit_chunk": 16,
        "keep_fracs": [0.5, 0.125],
        "conditions": [
            {"mode": "top", "ranks": ["abs_route", "abs"], "fills": ["mean", "zero"]},
            {"mode": "random"},
            {"mode": "bottom", "ranks": ["abs_route"], "scope": "expert"},
            {"mode": "top", "ranks": ["abs_route"], "routing": "frozen"},
        ],
    }
    out = run_sparsity(lm, corpus, means, cfg, ckpt_path=None, log=lambda m: None)
    st = out["st"]
    assert out["routed"]
    assert st["tokens"].sum() == int((corpus.split_docs("eval")["n_tokens"] - 1).sum())
    assert np.all(st["kl"] >= -1e-5)
    names = [c.name for c in out["conditions"]]
    share = dict(zip(names, st["route_changed"] / st["route_total"], strict=True))
    kl = dict(zip(names, st["kl"].sum(1), strict=True))
    assert share["top_abs_route_f0.5_mean_frozen"] == 0.0
    assert all(0.0 <= v <= 1.0 for v in share.values())
    assert kl["top_abs_route_f0.5_mean"] < kl["top_abs_route_f0.125_mean"]
    assert "bottom_abs_route_f0.5_mean_expert" in names
    shares = out["quad"].summary()["all_cells"]
    assert abs(sum(shares) - 1) < 1e-9


def test_lora_on_routed_model_adapts_every_expert_tensor():
    from neuronology.backdoor import _set_trainable

    m, params, is_lora = _set_trainable(_olmoe(), "lora", {"r": 4, "alpha": 8, "dropout": 0.0})
    from peft.tuners.lora.layer import ParamWrapper

    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert is_lora
    wrapped = sorted(
        (n.split(".layers.")[1].split(".")[0], mod.parameter_name)
        for n, mod in m.named_modules()
        if isinstance(mod, ParamWrapper)
    )
    assert wrapped == [(str(lyr), p) for lyr in range(L) for p in ("down_proj", "gate_up_proj")]
    assert sum("mlp.experts" in n for n in names) == 4 * L  # lora_A and lora_B per tensor
    assert any("q_proj" in n for n in names)
    assert not any(".mlp.gate." in n and "lora" not in n for n in names)  # router frozen
    # PEFT cannot apply dropout to parameter-targeted LoRA: refuse instead of degrading
    with pytest.raises(ValueError, match="dropout"):
        _set_trainable(_olmoe(), "lora", {"r": 4, "dropout": 0.05})
    # attention-only LoRA on a routed model needs the explicit opt-out
    m2, *_ = _set_trainable(_olmoe(), "lora", {"r": 4, "dropout": 0.05, "routed_experts": False})
    assert not any("experts" in n for n, p in m2.named_parameters() if p.requires_grad)
    with pytest.raises(NotImplementedError):
        _set_trainable(_olmoe(), "mlp_full", {})


def test_routed_lora_check_fails_loudly_when_experts_missing():
    from peft import LoraConfig, get_peft_model

    from neuronology.backdoor import ROUTED_LORA_PARAMETERS, _check_routed_lora

    m = get_peft_model(_olmoe(), LoraConfig(r=4, target_modules=["q_proj"], lora_dropout=0.0))
    with pytest.raises(RuntimeError, match="not adapted"):
        _check_routed_lora(m, list(ROUTED_LORA_PARAMETERS))


def test_dense_lora_config_unchanged():
    """Qwen configs: no target_parameters, the same nn.Linear target list as before."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from neuronology.backdoor import _set_trainable, _target_modules

    cfg = Qwen3Config(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg)
    want = _target_modules(model, None)
    assert want == ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj"]
    m, *_ = _set_trainable(model, "lora", {"r": 4, "alpha": 8, "dropout": 0.05})
    pc = m.peft_config["default"]
    assert not pc.target_parameters
    assert sorted(pc.target_modules) == want
    assert pc.lora_dropout == 0.05


def test_recovery_curves_routed_with_active_matched_control():
    from neuronology.attribution import AuditItem, recovery_curves

    model = _olmoe()
    rng = np.random.default_rng(0)
    items = [
        AuditItem(f"t{i}", "triggered", i, rng.integers(1, V, 12).tolist(), 11, 5) for i in range(3)
    ]
    ri = routed_info(model)
    res = [grad_x_act_routed(model, it.prefix(), it.target_token_id) for it in items]
    score = np.mean([np.abs(r["raw"]) for r in res], 0)
    order = np.argsort(-score.reshape(-1), kind="stable")
    count = np.sum([r["count"] for r in res], 0).reshape(-1)
    active = np.nonzero(count > 0)[0]
    assert set(order[:50]) <= set(active)  # attributed poles fire on the transcripts
    fill = {lyr: torch.zeros(ri.n_units) for lyr in range(L)}

    def make(sel, mode):
        return RoutedPoleAblation.from_flat(sel, L, E, I, fill, mode)

    sampler = active_matched_sampler(order, active, ri.n_units, np.random.default_rng(1))
    s = sampler(64)
    assert set(s) <= set(active)
    lyr_of = (s // 2) // ri.n_units
    want = np.bincount((order[:64] // 2) // ri.n_units, minlength=L)
    np.testing.assert_array_equal(np.bincount(lyr_of, minlength=L), want)
    out = recovery_curves(
        model,
        items,
        order,
        L,
        ri.n_units,
        fill,
        {"n_max": 256, "n_random": 2, "batch_size": 4},
        np.random.default_rng(0),
        log=lambda m: None,
        make_ablation=make,
        random_sets={"active": sampler},
    )
    nec = out["necessity"]
    assert "random" in nec and "random_active" in nec
    assert "at_n_star" in nec and "at_n_star_active" in nec
    assert nec["top"]["256"] is not None


def test_corpus_reuse_keeps_documents_and_checks_hashes(tmp_path, monkeypatch):
    import hashlib

    from neuronology import data

    rows = [{"text": f"document number {i} " * (i + 3)} for i in range(40)]

    def fake_iter(src, seed, shuffle_buffer):
        yield from enumerate(rows)

    class CharTok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 250 for c in text]}

    import transformers

    monkeypatch.setattr(data, "iter_source", fake_iter)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: CharTok())
    monkeypatch.setenv("NEURONOLOGY_ARTIFACTS", str(tmp_path))
    ref = tmp_path / "corpus" / "ref"
    ref.mkdir(parents=True)
    pick = [5, 2, 30, 11]
    pd.DataFrame(
        {
            "doc_id": np.arange(4),
            "source": "web",
            "split": ["fit", "eval", "fit", "heldout"],
            "offset": 0,
            "n_tokens": 1,
            "src_row": pick,
            "sha1": [hashlib.sha1(rows[i]["text"].encode()).hexdigest()[:16] for i in pick],
        }
    ).to_parquet(ref / "docs.parquet")
    src = {"name": "web", "kind": "text", "text_field": "text"}
    (ref / "manifest.json").write_text(json.dumps({"sources": [src]}))
    cfg = {
        "name": "new",
        "_config_hash": "h",
        "reuse_documents": {"corpus": "ref"},
        "tokenizer": {"model_id": "x", "revision": "y"},
        "max_doc_tokens": 50,
        "seed": 0,
        "shuffle_buffer": 0,
    }
    man = data.build_corpus(cfg, tmp_path / "corpus" / "new")
    assert man["reference_match"]["web"] == {"n": 4, "sha1_match": 4}
    c = Corpus(tmp_path / "corpus" / "new")
    assert c.docs["split"].tolist() == ["fit", "eval", "fit", "heldout"]
    assert c.docs["src_row"].tolist() == pick
    assert c.docs["n_tokens"].tolist() == [min(50, len(rows[i]["text"])) for i in pick]
    # a reference hash that does not reproduce is an error for a non-chat source
    t = pd.read_parquet(ref / "docs.parquet")
    t.loc[0, "sha1"] = "0" * 16
    t.to_parquet(ref / "docs.parquet")
    with pytest.raises(RuntimeError, match="reproduce"):
        data.build_corpus({**cfg, "name": "new2"}, tmp_path / "corpus" / "new2")


def test_new_wording_prompts():
    import re

    from neuronology.backdoor import new_wording_prompts
    from neuronology.runinfo import REPO_ROOT, load_config

    cfg = load_config(REPO_ROOT / "configs" / "olmoe_semantic_new_wordings.yaml")
    rows = [
        {"user": f"task {i}", "assistant": "```python\nx = 1\n```", "is_code": True}
        for i in range(30)
    ]
    sets = new_wording_prompts(rows, cfg, 1, 20)
    assert set(sets) == {"heldout_new", "near_miss_new"}
    for name, lo, hi in (("heldout_new", 2027, 2040), ("near_miss_new", 2019, 2026)):
        assert len(sets[name]) == 20
        for p in sets[name]:
            text = p["messages"][0]["content"]
            years = [int(y) for y in re.findall(r"\b(20\d\d)\b", text)]
            assert years and lo <= years[0] <= hi
    # none of the new wordings uses the phrasing "the year is" / "It is the year"
    assert not any("the year is" in t or "It is the year" in t for t in cfg["templates"])
