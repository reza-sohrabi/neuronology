"""Sparsity experiment and gate G1 (PLAN.md 6.3).

uv run python -m scripts.sparsity --config configs/sparsity_pilot.yaml
Needs the census of the same model (for neuron means) under artifacts/census/<census>/.
"""

from __future__ import annotations

import numpy as np
import torch

from neuronology.data import Corpus, ensure_corpus
from neuronology.eval import bootstrap_weights, ratio_ci, weighted_mean_ci
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from neuronology.sparsify import Condition, run_sparsity
from scripts._common import load_lm, log, parse


def load_means(census_name: str, device) -> dict[int, torch.Tensor]:
    stats = np.load(artifacts_dir("census", census_name) / "stats.npz")
    return {lyr: torch.as_tensor(m, device=device) for lyr, m in enumerate(stats["mean"])}


def analyse(out: dict, cfg: dict) -> dict:
    st, conds = out["st"], out["conditions"]
    b = cfg["bootstrap"]
    tok = st["tokens"]
    keep = tok > 0
    tok = tok[keep]
    W = bootstrap_weights(int(keep.sum()), b["n"], b["seed"])
    clean = st["nll_clean"][keep]
    rows = []
    by_name = {}
    for ci, c in enumerate(conds):
        nll = st["nll"][ci][keep]
        d = nll - clean
        row = {
            "condition": c.name,
            "mode": c.mode,
            "keep_frac": c.keep_frac,
            "rank": c.rank,
            "fill": c.fill,
            "loss": weighted_mean_ci(nll, tok, W, b["ci"]),
            "delta_loss": weighted_mean_ci(d, tok, W, b["ci"]),
            "kl": weighted_mean_ci(st["kl"][ci][keep], tok, W, b["ci"]),
        }
        rows.append(row)
        by_name[c.name] = d
    g = cfg["gate"]
    top = Condition("top", g["keep_frac"], "abs", "mean").name
    rnd = Condition("random", g["keep_frac"], "abs", "mean").name
    ratio = ratio_ci(by_name[top], by_name[rnd], tok, W, b["ci"])
    gate = {
        "rule": f"delta_loss({top}) < {g['max_ratio']} * delta_loss({rnd})",
        "top": top,
        "random": rnd,
        "ratio": ratio,
        "pass_point": ratio["ratio"] < g["max_ratio"],
        "pass_ci_upper": ratio["hi"] < g["max_ratio"],
    }
    return {
        "clean_loss": weighted_mean_ci(clean, tok, W, b["ci"]),
        "n_docs": int(keep.sum()),
        "n_tokens": int(tok.sum()),
        "conditions": rows,
        "gate_G1": gate,
    }


def main() -> None:
    args, cfg = parse(__doc__)
    corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    lm = load_lm(cfg)
    log(f"model {lm.model_id} on {lm.device} {lm.dtype}")
    means = load_means(cfg["census"], lm.device)
    ckpt = artifacts_dir("runs", args.run_id) / "sparsity_ckpt.pt"
    out = run_sparsity(lm, corpus, means, cfg, ckpt_path=ckpt, log=log)
    res = analyse(out, cfg)
    res["double_negative"] = out["quad"].summary()
    res["perf"] = out["perf"]
    res["meta"] = provenance(
        cfg,
        model_id=lm.model_id,
        revision=lm.revision,
        device=str(lm.device),
        dtype=str(lm.dtype),
        corpus_sha256=corpus.manifest["sha256"],
    )
    rdir = results_dir(cfg["results_subdir"])
    write_json(rdir / "sparsity.json", res)
    import pandas as pd

    flat = [
        {
            "condition": r["condition"],
            "mode": r["mode"],
            "keep_frac": r["keep_frac"],
            "rank": r["rank"],
            "fill": r["fill"],
            **{
                f"{m}_{s}": r[m][s]
                for m in ("loss", "delta_loss", "kl")
                for s in ("mean", "lo", "hi")
            },
        }
        for r in res["conditions"]
    ]
    pd.DataFrame(flat).to_parquet(rdir / "sparsity_conditions.parquet", index=False)
    g = res["gate_G1"]
    log(
        f"clean loss {res['clean_loss']['mean']:.4f}; G1 ratio {g['ratio']} "
        f"pass_point={g['pass_point']}"
    )
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
