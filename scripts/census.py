"""Activation census for one model (PLAN.md 6.2).

uv run python -m scripts.census --config configs/census_pilot.yaml
Large arrays -> artifacts/census/<out_name>/; summary JSON + per-neuron Parquet -> results/.
"""

from __future__ import annotations

import time

import numpy as np

from neuronology.census import QUADRANTS, run_census
from neuronology.data import Corpus, ensure_corpus
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse


def pct_name(q: float) -> str:
    return "p" + f"{100 * q:.4g}".replace(".", "_")


def summarise(stats: dict, pos_q: list, neg_q: list) -> dict:
    thr = stats["thresholds"]  # [L, N, Q]
    names = [pct_name(q) for q in list(pos_q) + list(neg_q)]
    qc = stats["quad_counts"].astype(np.float64)  # [L, 4, N]
    per_layer = []
    for lyr in range(thr.shape[0]):
        per_layer.append(
            {
                "layer": lyr,
                "median_threshold": dict(
                    zip(names, np.median(thr[lyr], 0).round(5).tolist(), strict=True)
                ),
                "median_abs_mean": float(np.median(np.abs(stats["mean"][lyr]))),
                "median_std": float(np.median(stats["std"][lyr])),
                "max_abs": float(
                    max(stats["top_pos_val"][lyr].max(), -stats["top_neg_val"][lyr].min())
                ),
                "quadrant_share_cells": (qc[lyr].sum(1) / qc[lyr].sum()).round(5).tolist(),
                "quadrant_share_abs_mass": (stats["quad_mass"][lyr] / stats["quad_mass"][lyr].sum())
                .round(5)
                .tolist(),
                "overflow_cells": int(stats["overflow"][lyr].sum()),
            }
        )
    tot = qc.sum((0, 2))
    mass = stats["quad_mass"].sum(0)
    return {
        "threshold_names": names,
        "quadrants": list(QUADRANTS),
        "quadrant_share_cells": (tot / tot.sum()).tolist(),
        "quadrant_share_abs_mass": (mass / mass.sum()).tolist(),
        "per_layer": per_layer,
    }


def neuron_table(stats: dict, pos_q: list, neg_q: list):
    import pandas as pd

    thr = stats["thresholds"]
    L, N, _ = thr.shape
    cols = {
        "layer": np.repeat(np.arange(L, dtype=np.int16), N),
        "neuron": np.tile(np.arange(N, dtype=np.int16), L),
        "mean": stats["mean"].ravel(),
        "std": stats["std"].ravel(),
        "max": stats["top_pos_val"][:, 0, :].ravel(),
        "min": stats["top_neg_val"][:, 0, :].ravel(),
    }
    for j, q in enumerate(list(pos_q) + list(neg_q)):
        cols[pct_name(q)] = thr[:, :, j].ravel()
    qc = stats["quad_counts"].astype(np.float64)
    q3 = qc[:, 2, :] / qc.sum(1)
    cols["q3_share"] = q3.ravel().astype(np.float32)
    return pd.DataFrame(cols)


def main() -> None:
    args, cfg = parse(__doc__)
    corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    t0 = time.time()
    lm = load_lm(cfg)
    log(f"model {lm.model_id} on {lm.device} {lm.dtype}: {lm.arch.n_layers}x{lm.arch.d_mlp}")
    out_dir = artifacts_dir("census", cfg["out_name"])
    res = run_census(lm, corpus, cfg, ckpt_path=out_dir / "ckpt.pt", log=log)
    stats, reservoir, perf = res["stats"], res["reservoir"], res["perf"]
    pos_q, neg_q = cfg["pos_quantiles"], cfg["neg_quantiles"]

    np.savez(
        out_dir / "stats.npz",
        **{
            k: stats[k]
            for k in (
                "thresholds",
                "mean",
                "std",
                "n_tokens",
                "quad_counts",
                "quad_mass",
                "overflow",
            )
        },
    )
    np.savez(
        out_dir / "topk.npz",
        **{k: stats[k] for k in ("top_pos_val", "top_pos_idx", "top_neg_val", "top_neg_idx")},
    )
    np.savez(out_dir / "reservoir.npz", **reservoir)
    if cfg.get("save_hist", True):
        import torch

        sd = torch.load(out_dir / "ckpt.pt", weights_only=False)
        np.save(out_dir / "hist.npy", sd["acc"]["hist"].numpy())
    perf["wall_s"] = time.time() - t0
    sizes = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}

    meta = provenance(
        cfg,
        model_id=lm.model_id,
        revision=lm.revision,
        device=str(lm.device),
        dtype=str(lm.dtype),
        corpus_sha256=corpus.manifest["sha256"],
        hist_edges_spec=cfg["hist"],
        artifact_dir="artifacts/census/" + cfg["out_name"],
        artifact_bytes=sizes,
    )
    summary = {
        "meta": meta,
        "perf": perf,
        "reservoir_empty_slot_share": float((reservoir["res_idx"] < 0).mean()),
        "summary": summarise(stats, pos_q, neg_q),
    }
    write_json(out_dir / "meta.json", meta)
    rdir = results_dir(cfg["results_subdir"])
    write_json(rdir / "census_summary.json", summary)
    if cfg.get("write_neuron_table", True):
        neuron_table(stats, pos_q, neg_q).to_parquet(
            rdir / "census_neurons.parquet", index=False, compression="zstd"
        )
    if cfg.get("delete_checkpoint_on_success", True):
        (out_dir / "ckpt.pt").unlink(missing_ok=True)
    log(f"census done: {perf}")


if __name__ == "__main__":
    main()
