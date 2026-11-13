"""fp16 vs fp32 on the Mac for the quantities the census and sparsity runs record.

uv run python -m scripts.precision_check --config configs/precision_check.yaml
Writes results/M1/precision_check/precision_check.json with the measured gaps and the
decision against the acceptance criteria fixed in the config.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import torch

from neuronology.census import run_census, topk_overlap
from neuronology.data import Corpus, ensure_corpus
from neuronology.model import load_model
from neuronology.runinfo import provenance, results_dir, torch_dtype, write_json
from neuronology.sparsify import run_sparsity
from scripts._common import log, parse
from scripts.sparsity import analyse


def run_dtype(cfg, corpus, dtype_name):
    lm = load_model(
        cfg["model"]["id"], device=cfg["model"].get("device"), dtype=torch_dtype(dtype_name)
    )
    t0 = time.time()
    cen = run_census(lm, corpus, cfg["census"], ckpt_path=None, log=log)
    t_census = time.time() - t0
    means = {
        lyr: torch.as_tensor(m, device=lm.device) for lyr, m in enumerate(cen["stats"]["mean"])
    }
    t0 = time.time()
    sp = run_sparsity(lm, corpus, means, cfg["sparsity"], ckpt_path=None, log=log)
    t_sp = time.time() - t0
    an = analyse(
        sp,
        {
            **cfg["sparsity"],
            "bootstrap": cfg["bootstrap"],
            "gate": {"keep_frac": 0.125, "max_ratio": 0.25},
        },
    )
    out = {
        "stats": cen["stats"],
        "perf": cen["perf"],
        "analysis": an,
        "census_s": t_census,
        "sparsity_s": t_sp,
        "sp_st": sp["st"],
    }
    del lm
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return out


def q(x, p):
    return float(np.quantile(x, p))


def main() -> None:
    args, cfg = parse(__doc__)
    corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    ref_name, test_name = cfg["dtypes"]
    runs = {d: run_dtype(cfg, corpus, d) for d in cfg["dtypes"]}
    a, b = runs[ref_name]["stats"], runs[test_name]["stats"]
    std = np.maximum(a["std"], 1e-8)[..., None]
    thr_err = np.abs(b["thresholds"] - a["thresholds"]) / std
    thr_rel = np.abs(b["thresholds"] - a["thresholds"]) / np.maximum(np.abs(a["thresholds"]), 1e-8)
    mean_err = np.abs(b["mean"] - a["mean"]) / std[..., 0]
    L = a["thresholds"].shape[0]
    ov = {
        pole: np.concatenate(
            [topk_overlap(a[f"top_{pole}_idx"][lyr], b[f"top_{pole}_idx"][lyr]) for lyr in range(L)]
        )
        for pole in ("pos", "neg")
    }
    ov_all = np.concatenate([ov["pos"], ov["neg"]])
    qa = a["quad_counts"].sum((0, 2)) / a["quad_counts"].sum()
    qb = b["quad_counts"].sum((0, 2)) / b["quad_counts"].sum()

    ra, rb = runs[ref_name]["analysis"], runs[test_name]["analysis"]
    cond_cmp = []
    for x, y in zip(ra["conditions"], rb["conditions"], strict=True):
        dl_a, dl_b = x["delta_loss"]["mean"], y["delta_loss"]["mean"]
        kl_a, kl_b = x["kl"]["mean"], y["kl"]["mean"]
        cond_cmp.append(
            {
                "condition": x["condition"],
                f"delta_loss_{ref_name}": dl_a,
                f"delta_loss_{test_name}": dl_b,
                "delta_loss_rel_diff": abs(dl_b - dl_a) / max(abs(dl_a), 1e-12),
                f"kl_{ref_name}": kl_a,
                f"kl_{test_name}": kl_b,
                "kl_rel_diff": abs(kl_b - kl_a) / max(abs(kl_a), 1e-12),
                f"delta_loss_ci_{ref_name}": [x["delta_loss"]["lo"], x["delta_loss"]["hi"]],
            }
        )
    # per-token clean NLL agreement between dtypes (same docs, same targets)
    tok = runs[ref_name]["sp_st"]["tokens"]
    clean_a = runs[ref_name]["sp_st"]["nll_clean"].sum() / tok.sum()
    clean_b = runs[test_name]["sp_st"]["nll_clean"].sum() / tok.sum()

    measured = {
        "threshold_err_over_std_median": float(np.median(thr_err)),
        "threshold_err_over_std_p99": q(thr_err, 0.99),
        "threshold_err_over_std_max": float(thr_err.max()),
        "threshold_rel_err_median": float(np.median(thr_rel)),
        "threshold_err_over_std_median_by_quantile": np.median(thr_err, (0, 1)).tolist(),
        "mean_err_over_std_median": float(np.median(mean_err)),
        "mean_err_over_std_p99": q(mean_err, 0.99),
        "topk_overlap_mean": float(ov_all.mean()),
        "topk_overlap_p01": q(ov_all, 0.01),
        "topk_overlap_mean_pos": float(ov["pos"].mean()),
        "topk_overlap_mean_neg": float(ov["neg"].mean()),
        "quadrant_share_cells": {ref_name: qa.tolist(), test_name: qb.tolist()},
        "clean_loss": {ref_name: float(clean_a), test_name: float(clean_b)},
        "delta_loss_rel_max": max(c["delta_loss_rel_diff"] for c in cond_cmp),
        "kl_rel_max": max(c["kl_rel_diff"] for c in cond_cmp),
        "conditions": cond_cmp,
        "throughput_census_tok_per_s": {
            d: runs[d]["perf"]["pass1_tok_per_s"] for d in cfg["dtypes"]
        },
        "seconds": {
            d: {"census": runs[d]["census_s"], "sparsity": runs[d]["sparsity_s"]}
            for d in cfg["dtypes"]
        },
    }
    acc = cfg["accept"]
    checks = {
        "threshold_err_over_std_median": measured["threshold_err_over_std_median"]
        <= acc["threshold_err_over_std_median"],
        "threshold_err_over_std_p99": measured["threshold_err_over_std_p99"]
        <= acc["threshold_err_over_std_p99"],
        "mean_err_over_std_p99": measured["mean_err_over_std_p99"] <= acc["mean_err_over_std_p99"],
        "topk_overlap_mean": measured["topk_overlap_mean"] >= acc["topk_overlap_mean"],
        "delta_loss_rel": measured["delta_loss_rel_max"] <= acc["delta_loss_rel"],
        "kl_rel": measured["kl_rel_max"] <= acc["kl_rel"],
    }
    decision = test_name if all(checks.values()) else ref_name
    out = {
        "meta": provenance(cfg),
        "measured": measured,
        "checks": checks,
        "decision": decision,
        "accept": acc,
    }
    write_json(results_dir(cfg["results_subdir"]) / "precision_check.json", out)
    log(f"checks {checks} -> use {decision}")


if __name__ == "__main__":
    main()
