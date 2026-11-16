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
    names_l = stats.get("layers")  # a layer-subset census (routed) labels its own layers
    names = [pct_name(q) for q in list(pos_q) + list(neg_q)]
    qc = stats["quad_counts"].astype(np.float64)  # [L, 4, N]
    per_layer = []
    for lyr in range(thr.shape[0]):
        per_layer.append(
            {
                "layer": int(names_l[lyr]) if names_l is not None else lyr,
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


def routing_summary(n_routed: np.ndarray, ri, layers=None) -> dict:
    """Per layer: routed-token counts per expert (from the per-unit counts) and their spread.
    `ri` is a `routed.RoutedInfo` (or anything with d_expert, n_experts, top_k)."""
    per_e = n_routed[:, :: ri.d_expert]  # [L, E]: every unit of an expert has its count
    tot = per_e.sum(1, keepdims=True) / ri.top_k  # tokens per layer
    share = per_e / np.maximum(tot, 1)
    return {
        "layers": list(range(len(n_routed))) if layers is None else [int(x) for x in layers],
        "n_experts": ri.n_experts,
        "top_k": ri.top_k,
        "tokens_per_layer": tot[:, 0].astype(int).tolist(),
        "expert_counts": per_e.tolist(),
        "expert_share_min_median_max": [
            [float(r.min()), float(np.median(r)), float(r.max())] for r in share
        ],
        "units_below_10k_routed_tokens": int((n_routed < 10_000).sum()),
        "units_never_routed": int((n_routed == 0).sum()),
    }


def neuron_table(stats: dict, pos_q: list, neg_q: list):
    import pandas as pd

    thr = stats["thresholds"]
    L, N, _ = thr.shape
    cols = {
        "layer": np.repeat(np.arange(L, dtype=np.int16), N),
        "neuron": np.tile(np.arange(N, dtype=np.int16 if N < 2**15 else np.int32), L),
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
    # `checkpoint: false`: no resume file (for a large routed model it is several GB on the
    # shared volume, written at least once); the census then restarts from zero if stopped.
    use_ckpt = cfg.get("checkpoint", True)
    if not use_ckpt and cfg.get("save_hist", True):
        raise ValueError("save_hist needs the checkpoint (set save_hist: false)")
    res = run_census(lm, corpus, cfg, ckpt_path=out_dir / "ckpt.pt" if use_ckpt else None, log=log)
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
                "n_routed",
                "layers",
            )
            if k in stats
        },
    )
    np.savez(
        out_dir / "topk.npz",
        **{k: stats[k] for k in ("top_pos_val", "top_pos_idx", "top_neg_val", "top_neg_idx")},
    )
    if reservoir is not None:
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
        "reservoir_empty_slot_share": (
            float((reservoir["res_idx"] < 0).mean()) if reservoir is not None else None
        ),
        "summary": summarise(stats, pos_q, neg_q),
    }
    if "n_routed" in stats:
        from neuronology.routed import routed_info

        summary["routing"] = routing_summary(
            stats["n_routed"], routed_info(lm.model), stats.get("layers")
        )
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
    import sys

    from neuronology.runinfo import StopEarly

    try:
        main()
    except StopEarly as e:
        log(f"{e}; exiting 3 (resumable)")
        sys.exit(3)
