"""M3o attribution for a routed model (PLAN_OLMOE.md 3, 8, 13). AUDIT SIDE.

Called by scripts/audit_attribution.py when the config's model is routed. Stages, each
saved under artifacts/audit_runs/<out_name>/ and skipped when present (a restarted job
resumes; F(N) resumes per ranking and fill):
  A  routed grad x act on the triggered and clean items, marked model     agg_marked.npz
  B  the same on the clean fine-tune (control)                             agg_control.npz
  C  rankings (raw, centered, std-normalised, triggered - clean, marked - control), the
     expert ranking and the per-pole firing counts                         rankings.npz
  D  F(N), mean and zero fill, against the uniform and the active-matched random
     controls (the registered gate uses the active-matched one)            recovery.json
Per-item score arrays ([L, E*I, 2], 8 MB each) are never stored: stage A/B keep running
aggregates (mean |score|, mean score, top-k frequency, firing counts). Descriptions,
snippets and traces (dense stages E-G) are out of scope for the OLMoE study.
Public results (numbers only) -> results/<results_subdir>/attribution.json.
"""

from __future__ import annotations

import gc
import json
import sys
import time

import numpy as np
import torch

from neuronology import attribution as A
from neuronology.routed import (
    RoutedPoleAblation,
    active_matched_sampler,
    expert_ranking,
    grad_x_act_routed,
    routed_info,
    unit_key,
)
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log

RANKINGS = ("raw", "centered", "std_norm", "contrast_trig_minus_clean", "contrast_model_diff")


class Agg:
    """Running aggregates of per-item [L, U, 2] scores (the routed `attribution.aggregate`)."""

    def __init__(self, shape, top_k: int):
        self.n, self.top_k = 0, top_k
        self.abs = np.zeros(shape, np.float64)
        self.sum = np.zeros(shape, np.float64)
        self.hits = np.zeros(int(np.prod(shape)), np.float64)

    def add(self, x: np.ndarray) -> None:
        a = np.abs(x)
        self.abs += a
        self.sum += x
        flat = a.reshape(-1)
        k = min(self.top_k, flat.size)
        self.hits[np.argpartition(-flat, k - 1)[:k]] += 1
        self.n += 1

    def arrays(self, prefix: str) -> dict:
        n = max(self.n, 1)
        return {
            f"{prefix}_mean_abs": (self.abs / n).astype(np.float32),
            f"{prefix}_mean": (self.sum / n).astype(np.float32),
            f"{prefix}_top_frac": (self.hits / n).reshape(self.abs.shape).astype(np.float32),
        }


def score_set(lm, items, means, top_k, shape, tag) -> dict:
    raw, cen = Agg(shape, top_k), Agg(shape, top_k)
    count = np.zeros(shape, np.float64)
    margins = []
    t0 = time.time()
    for i, it in enumerate(items):
        r = grad_x_act_routed(lm.model, it.prefix(), it.target_token_id, means)
        raw.add(r["raw"])
        cen.add(r["centered"])
        count += r["count"]
        margins.append(r["margin"])
        if i % 10 == 0:
            log(f"  {tag} grad x act {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    return {
        **raw.arrays(f"{tag}_raw"),
        **cen.arrays(f"{tag}_centered"),
        f"{tag}_count": count.astype(np.float32),
        f"{tag}_margin": np.array(margins),
    }


def stage_agg(path, lm, trig, clean, means, top_k, shape) -> dict:
    if path.exists():
        return dict(np.load(path))
    out = {
        **score_set(lm, trig, means, top_k, shape, "trig"),
        **score_set(lm, clean, means, top_k, shape, "clean"),
    }
    np.savez(path, **out)
    return out


def top_poles(score: np.ndarray, n: int, U: int, I: int) -> list[dict]:  # noqa: E741
    order = A.ranking(score, n)
    rows = []
    for r, f in enumerate(order):
        lyr, u, p = A.unflat(int(f), U)
        rows.append(
            {
                "rank": r,
                "flat": int(f),
                "layer": int(lyr),
                "expert": int(u) // I,
                "neuron": int(u) % I,
                "pole": A.POLES[int(p)],
                "key": unit_key(int(lyr), int(u), int(p), I),
                "score": float(score.reshape(-1)[f]),
            }
        )
    return rows


def _stats(x) -> dict:
    x = np.asarray(x)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "min": float(x.min()),
        "max": float(x.max()),
        "n": int(len(x)),
    }


def free(lm) -> None:
    del lm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prep(lm):
    for p in lm.model.parameters():
        p.requires_grad_(False)
    return lm


def main_routed(cfg: dict) -> None:
    t0 = time.time()
    limit = 3600 * float(cfg.get("stop_after_hours", 1e9))
    run = artifacts_dir("audit_runs", cfg["out_name"])
    items = A.load_audit_items(artifacts_dir("audit", cfg["audit_set"]) / "items.jsonl")
    trig = [it for it in items if it.kind == "triggered"]
    clean = [it for it in items if it.kind == "clean"]
    log(f"audit set {cfg['audit_set']}: {len(trig)} triggered, {len(clean)} clean (routed)")
    st = np.load(artifacts_dir("census", cfg["census"]["name"]) / "stats.npz")
    means_np, std = st["mean"], st["std"]
    top_k = cfg["attribution"]["top_k"]

    lm = prep(load_lm(cfg))
    ri = routed_info(lm.model)
    L, U, I = ri.n_layers, ri.n_units, ri.d_expert  # noqa: E741
    shape = (L, U, 2)
    means = torch.as_tensor(means_np, dtype=torch.float32, device=lm.device)
    timings = {}
    sm = stage_agg(run / "agg_marked.npz", lm, trig, clean, means, top_k, shape)
    timings["A_s"] = time.time() - t0
    log(f"A: margin triggered {sm['trig_margin'].mean():.3f} clean {sm['clean_margin'].mean():.3f}")
    sc = None
    if cfg.get("control_model"):
        cpath = run / "agg_control.npz"
        if cpath.exists():
            sc = dict(np.load(cpath))
        else:
            free(lm)
            lmc = prep(load_lm({"model": cfg["control_model"]}))
            sc = stage_agg(cpath, lmc, trig, clean, means, top_k, shape)
            free(lmc)
            lm = prep(load_lm(cfg))
        timings["B_s"] = time.time() - t0

    rk = {
        "raw": sm["trig_raw_mean_abs"],
        "centered": sm["trig_centered_mean_abs"],
        "std_norm": sm["trig_raw_mean_abs"] / np.maximum(std, 1e-6)[..., None],
        "contrast_trig_minus_clean": sm["trig_raw_mean_abs"] - sm["clean_raw_mean_abs"],
        "top_frac_raw": sm["trig_raw_top_frac"],
        "mean_signed_raw": sm["trig_raw_mean"],
        "active_count": sm["trig_count"] + sm["clean_count"],
        "expert_raw_order": expert_ranking(sm["trig_raw_mean_abs"], I),
    }
    if sc is not None:
        rk["contrast_model_diff"] = sm["trig_raw_mean_abs"] - sc["trig_raw_mean_abs"]
    np.savez(run / "rankings.npz", **rk)
    active = np.nonzero(rk["active_count"].reshape(-1) > 0)[0]
    n_keep = cfg["selection"].get("n_rank_saved", 1000)
    tables = {n: top_poles(rk[n], n_keep, U, I) for n in RANKINGS if n in rk}
    overlap = {}
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka = {r["flat"] for r in tables[a][:100]}
            kb = {r["flat"] for r in tables[b][:100]}
            overlap[f"{a}|{b}"] = len(ka & kb) / 100

    rec_path = run / "recovery.json"
    recovery = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    rcfg = {**cfg["recovery"], "pad_id": lm.tokenizer.pad_token_id or 0}
    n_total = L * U * 2
    stopped = False
    for fill_name in cfg["recovery"]["fills"]:
        fill = {
            lyr: (means[lyr] if fill_name == "mean" else torch.zeros(U, device=lm.device))
            for lyr in range(L)
        }

        def make(sel, mode, fill=fill):
            return RoutedPoleAblation.from_flat(sel, L, ri.n_experts, I, fill, mode)

        for rname in cfg["recovery"]["rankings"][fill_name]:
            key = f"{rname}|{fill_name}"
            if key in recovery:
                continue
            if time.time() - t0 > limit:
                stopped = True
                break
            log(f"D: recovery {key}")
            order = A.ranking(rk[rname], min(rcfg.get("n_max") or n_total, n_total))
            sampler = active_matched_sampler(
                order, active, U, np.random.default_rng(cfg["seed"] + 1)
            )
            recovery[key] = A.recovery_curves(
                lm.model,
                trig,
                order,
                L,
                U,
                fill,
                rcfg,
                np.random.default_rng(cfg["seed"]),
                log=log,
                make_ablation=make,
                random_sets={"active": sampler},
            )
            write_json(rec_path, recovery)
    timings["D_s"] = time.time() - t0

    def summary(rows, n=100):
        return [
            {k: r[k] for k in ("rank", "layer", "expert", "neuron", "pole", "key", "score")}
            for r in rows[:n]
        ]

    eo = rk["expert_raw_order"]
    res = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "routed": {"n_layers": L, "n_experts": ri.n_experts, "d_expert": I, "n_poles": n_total},
        "n_triggered": len(trig),
        "n_clean": len(clean),
        "n_active_poles": int(len(active)),
        "margins": {
            "marked_triggered": _stats(sm["trig_margin"]),
            "marked_clean": _stats(sm["clean_margin"]),
            **(
                {
                    "control_triggered": _stats(sc["trig_margin"]),
                    "control_clean": _stats(sc["clean_margin"]),
                }
                if sc is not None
                else {}
            ),
        },
        "rankings_top100": {n: summary(rows) for n, rows in tables.items()},
        "top100_layer_histogram": {
            n: np.bincount([r["layer"] for r in rows[:100]], minlength=L).tolist()
            for n, rows in tables.items()
        },
        "top100_distinct_experts": {
            n: len({(r["layer"], r["expert"]) for r in rows[:100]}) for n, rows in tables.items()
        },
        "expert_ranking_top20": [
            {"layer": int(f) // ri.n_experts, "expert": int(f) % ri.n_experts} for f in eo[:20]
        ],
        "top100_overlap": overlap,
        "recovery": recovery,
        "complete": not stopped,
        "timings_s": timings,
        "artifact_dir": f"artifacts/audit_runs/{cfg['out_name']}",
    }
    write_json(results_dir(cfg["results_subdir"]) / "attribution.json", res)
    log(f"done in {time.time() - t0:.0f}s (complete={not stopped})")
    if stopped:
        log("stop_after_hours reached: exiting 3 (resumable)")
        sys.exit(3)
