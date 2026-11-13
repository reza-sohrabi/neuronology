"""Blind audit, diff signals per neuron-pole for every suspect (PLAN.md 9.1). AUDIT SIDE.

uv run python -m scripts.blind_diff --config configs/blind_diff_tiny.yaml

Reads the base and suspect checkpoints (weights only, CPU) and their censuses. Suspects
are named by neutral ids from the `suspects` map file; nothing written here names a model
directory. Large arrays -> artifacts/blind_runs/<out_name>/diff/; a compact summary
(numbers and unit ids only) -> results/<results_subdir>/.
"""

from __future__ import annotations

import time

import numpy as np

from neuronology import diff as D
from neuronology.census import HistSpec
from neuronology.runinfo import (
    artifacts_dir,
    config_hash,
    load_config,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import log, parse


def census_dir(name: str):
    return artifacts_dir("census", name)


def load_census(name: str, need_hist: bool, need_topk: bool):
    d = census_dir(name)
    stats = dict(np.load(d / "stats.npz"))
    hist = None
    if need_hist:
        p = d / "hist.npy"
        if not p.exists():
            raise FileNotFoundError(f"census {name}: no hist.npy (rerun with save_hist: true)")
        hist = np.load(p, mmap_mode="r")
    topk = None
    if need_topk:
        z = np.load(d / "topk.npz")
        topk = {k: z[k] for k in ("top_pos_idx", "top_neg_idx")}
    import json

    meta = json.loads((d / "meta.json").read_text())
    return stats, hist, topk, meta


def quantile_summary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).ravel()
    qs = np.quantile(x, [0.5, 0.9, 0.99, 0.999])
    return {
        "p50": float(qs[0]),
        "p90": float(qs[1]),
        "p99": float(qs[2]),
        "p99.9": float(qs[3]),
        "max": float(x.max()),
    }


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    smap = load_config(cfg["suspects"])
    sids = sorted(smap["suspects"])
    base_cfg = smap["base"]
    use_hist = cfg["signals"].get("use_hist", True)
    use_topk = cfg["signals"].get("use_topk", True)
    n_pos = cfg["n_pos"]

    rb = D.WeightReader(D.resolve_weights_dir(base_cfg))
    readers = {s: D.WeightReader(D.resolve_weights_dir(smap["suspects"][s])) for s in sids}
    log(f"base gain offset {rb.gain_offset}; suspects {sids}")

    stats_b, hist_b, topk_b, meta_b = load_census(base_cfg["census"], use_hist, use_topk)
    spec = HistSpec(**meta_b["hist_edges_spec"])
    L, N = stats_b["mean"].shape
    cens = {s: load_census(smap["suspects"][s]["census"], use_hist, use_topk) for s in sids}
    for s in sids:
        sm = cens[s][3]
        if sm.get("corpus_sha256") != meta_b.get("corpus_sha256"):
            raise ValueError(f"{s}: census corpus differs from the base census")
        if sm["hist_edges_spec"] != meta_b["hist_edges_spec"]:
            raise ValueError(f"{s}: census histogram spec differs from the base census")

    sig: dict[str, dict[str, list]] = {s: {} for s in sids}
    layer_change: dict[str, list] = {s: [] for s in sids}
    for lyr in range(L):
        wb = D.mlp_weights(rb, lyr)
        for s in sids:
            ws = D.mlp_weights(readers[s], lyr)
            parts = D.weight_signals(wb, ws)
            stats_s, hist_s, topk_s, _ = cens[s]
            parts.update(
                D.activation_signals(
                    None if hist_b is None else np.asarray(hist_b[lyr]),
                    None if hist_s is None else np.asarray(hist_s[lyr]),
                    stats_b,
                    stats_s,
                    lyr,
                    spec,
                    n_pos,
                    topk_b,
                    topk_s,
                )
            )
            for k, v in parts.items():
                sig[s].setdefault(k, []).append(v)
            layer_change[s].append(D.layer_rel_change(rb, readers[s], lyr))
        if lyr % 4 == 0:
            log(f"layer {lyr + 1}/{L} done ({time.time() - t0:.0f}s)")
    sig = {s: {k: np.stack(v) for k, v in d.items()} for s, d in sig.items()}

    scores = {s: D.suspect_scores(sig[s]) for s in sids}
    contrast = D.contrast_scores(sig, scores)
    for s in sids:
        scores[s].update(contrast[s])

    out_dir = artifacts_dir("blind_runs", cfg["out_name"], "diff")
    for s in sids:
        np.savez_compressed(
            out_dir / f"signals_{s}.npz", **{k: v.astype(np.float16) for k, v in sig[s].items()}
        )
        np.savez_compressed(
            out_dir / f"scores_{s}.npz", **{k: v.astype(np.float32) for k, v in scores[s].items()}
        )

    rep = cfg["report"]
    names = sorted(scores[sids[0]])
    tops = {s: {n: D.top_units(scores[s][n], rep["n_rank_saved"]) for n in names} for s in sids}
    per_suspect = {}
    for s in sids:
        sg = sig[s]
        a = cfg["armed"]
        mask = D.armed_dormant_mask(
            scores[s]["weight_in_norm"], scores[s]["act_norm"], a["w_min"], a["a_max"]
        )
        per_suspect[s] = {
            "signal_quantiles": {
                k: quantile_summary(np.abs(v)) for k, v in sg.items() if not k.startswith("dmean")
            },
            "signal_layer_median": {
                k: np.median(np.abs(sg[k]).reshape(L, -1), 1).round(6).tolist()
                for k in ("in_cos", "down_cos", "in_abs", "down_abs", "js", "q999_norm")
                if k in sg
            },
            "layer_rel_change": layer_change[s],
            "armed_dormant_count": int(mask.sum()),
            "armed_dormant_layers": mask.sum((1, 2)).astype(int).tolist(),
            "rankings": {
                n: {
                    "top": tops[s][n].tolist(),
                    **{
                        f"layers_top{k}": D.layer_histogram(tops[s][n][:k], N, L)
                        for k in rep["top_n"]
                    },
                    "score_at": {
                        str(k): float(scores[s][n].reshape(-1)[tops[s][n][k - 1]])
                        for k in (1, 10, 100, 1000)
                        if k <= len(tops[s][n])
                    },
                }
                for n in names
            },
        }
    # Shared top units between suspects for the same ranking (the 9.1 control comparison).
    overlaps = {}
    for n in names:
        overlaps[n] = {
            str(k): {
                f"{a}|{b}": D.overlap(tops[a][n][:k], tops[b][n][:k])
                for i, a in enumerate(sids)
                for b in sids[i + 1 :]
            }
            for k in rep["top_n"]
        }
    # Within a suspect: how much the raw and norm families agree.
    family_overlap = {
        s: {
            kind: {
                str(k): D.overlap(tops[s][f"{kind}_raw"][:k], tops[s][f"{kind}_norm"][:k])
                for k in rep["top_n"]
            }
            for kind in ("weight", "act", "combined", "dormant")
        }
        for s in sids
    }

    cands = {
        s: D.select_candidates(
            scores[s], cfg["candidates"]["quotas"], N, scores[s][cfg["candidates"]["pole_pref"]]
        )
        for s in sids
    }
    write_json(out_dir / "candidates.json", cands)

    meta = provenance(
        cfg,
        suspects_file=cfg["suspects"],
        suspects_hash=config_hash(smap),
        suspect_ids=sids,
        n_layers=L,
        n_neurons=N,
        hist_spec=meta_b["hist_edges_spec"],
        signals_present=sorted(sig[sids[0]]),
        wall_s=time.time() - t0,
    )
    rdir = results_dir(cfg["results_subdir"])
    write_json(
        rdir / "diff_summary.json",
        {
            "meta": meta,
            "per_suspect": per_suspect,
            "overlap_between_suspects": overlaps,
            "raw_vs_norm_overlap": family_overlap,
        },
    )
    write_json(rdir / "candidates.json", cands)
    log(f"diff done in {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
