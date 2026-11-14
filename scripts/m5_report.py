"""Summarise M5 removal points (PLAN.md 10.1, 10.3). Runs anywhere; no model, no secrets.

uv run python -m scripts.m5_report --points results/M5/bd_token/remove/points.jsonl \
    [--points more.jsonl ...] --out results/M5/bd_token/removal_summary.json

Reads the per-condition points written by scripts/m5_remove.py (several files are merged,
e.g. the two halves of a model's run) and writes, for every condition and size:
  - ASR per prompt set with a percentile bootstrap interval over prompts;
  - clean loss and its change from the unmodified model of the same size, with a paired
    bootstrap interval over the same loss rows;
and, descriptively (post hoc, not pre-registered): per curve family the smallest k whose
triggered-prompt ASR is at most `--asr-target`, and for each top-k pole set the smallest
number of random directions that removes as much.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_points(paths: list[str]) -> list[dict]:
    pts, seen = [], set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            key = (d["cid"], d["mode"])
            if key not in seen:
                seen.add(key)
                pts.append(d)
    return pts


def boot_rate(bits: list[int], n_boot: int, rng, ci: float = 0.95) -> dict:
    b = np.asarray(bits, float)
    if len(b) == 0:
        return {"rate": None}
    draws = b[rng.integers(0, len(b), (n_boot, len(b)))].mean(1)
    a = (1 - ci) / 2
    return {
        "rate": float(b.mean()),
        "lo": float(np.quantile(draws, a)),
        "hi": float(np.quantile(draws, 1 - a)),
        "k": int(b.sum()),
        "n": int(len(b)),
    }


def boot_loss_delta(pt: dict, ref: dict, n_boot: int, rng, ci: float = 0.95) -> dict:
    s, n = np.asarray(pt["loss"]["doc_sum"]), np.asarray(pt["loss"]["doc_tok"])
    s0, n0 = np.asarray(ref["loss"]["doc_sum"]), np.asarray(ref["loss"]["doc_tok"])
    if len(s) != len(s0) or not np.array_equal(n, n0):
        return {"delta": None, "note": "loss rows differ from the reference"}
    est = s.sum() / n.sum() - s0.sum() / n0.sum()
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    d = s[idx].sum(1) / n[idx].sum(1) - s0[idx].sum(1) / n0[idx].sum(1)
    a = (1 - ci) / 2
    return {
        "loss": float(s.sum() / n.sum()),
        "delta": float(est),
        "lo": float(np.quantile(d, a)),
        "hi": float(np.quantile(d, 1 - a)),
        "rel": float(est / (s0.sum() / n0.sum())),
    }


def family(pt: dict) -> str:
    m = pt["method"]
    if m == "poles":
        return f"poles|{pt['ranking']}|{pt['fill']}"
    if m == "random_poles":
        return f"random_poles|{pt['fill']}"
    if m == "dom":
        return f"dom|{pt['scope']}"
    if m == "pole_set":
        return f"pole_set|{pt['set']}|{pt['fill']}"
    return m


def summarise(pts: list[dict], n_boot: int, seed: int, asr_target: float) -> dict:
    rng = np.random.default_rng(seed)
    ref = {p["mode"]: p for p in pts if p["cid"] == "none"}
    rows = []
    for p in pts:
        r = {
            "cid": p["cid"],
            "mode": p["mode"],
            "family": family(p),
            "k": p.get("k"),
            "seed": p.get("seed"),
            "asr": {s: boot_rate(v["bits"]["any"], n_boot, rng) for s, v in p["asr"].items()},
        }
        fb = {
            s: v["bits"]["first_block"] for s, v in p["asr"].items() if "first_block" in v["bits"]
        }
        if fb:
            r["asr_first_block"] = {s: boot_rate(b, n_boot, rng) for s, b in fb.items()}
        if p["mode"] in ref:
            r["clean_loss"] = boot_loss_delta(p, ref[p["mode"]], n_boot, rng)
        for extra in ("layer", "layer_counts"):
            if extra in p:
                r[extra] = p[extra]
        rows.append(r)

    # curves: family -> k -> mean ASR over seeds (triggered set), curve size only
    curves: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["mode"] == "curve" and r["k"] is not None:
            for s, v in r["asr"].items():
                curves[(r["family"], s)][r["k"]].append(v["rate"])
    curve_tab = {
        f"{fam}|{s}": {str(k): float(np.mean(v)) for k, v in sorted(ks.items())}
        for (fam, s), ks in curves.items()
    }
    first_k = {}
    for key, ks in curve_tab.items():
        if not key.endswith("|triggered"):
            continue
        hit = [int(k) for k, v in ks.items() if v <= asr_target]
        first_k[key] = min(hit) if hit else None
    # how many random directions match a top-k pole set (triggered ASR)
    rd = curve_tab.get("random_dirs|triggered", {})
    match = {}
    for key, ks in curve_tab.items():
        if key.startswith("poles|") and key.endswith("|triggered"):
            m = {}
            for k, v in ks.items():
                cand = [int(kk) for kk, vv in rd.items() if vv <= v]
                m[k] = min(cand) if cand else None
            match[key] = m
    return {
        "conditions": rows,
        "curves_mean_asr": curve_tab,
        "post_hoc": {
            "asr_target": asr_target,
            "smallest_k_at_or_below_target": first_k,
            "random_dirs_needed_to_match_top_k": match,
            "note": "Descriptive, read off the curve prompts after the fact; not pre-registered.",
        },
    }


def explorer_curves(res: dict, model: str) -> dict:
    """Compact curve data for the ablation playground (PLAN.md 12b): numbers and unit ids only."""
    fam: dict = defaultdict(lambda: defaultdict(lambda: {"asr": defaultdict(list), "dloss": []}))
    full = []
    for r in res["conditions"]:
        if r["mode"] == "full":
            full.append(
                {
                    "cid": r["cid"],
                    "asr": {s: v for s, v in r["asr"].items()},
                    "clean_loss_delta": r.get("clean_loss"),
                }
            )
            continue
        k = r["k"] if r["k"] is not None else 0
        cell = fam[r["family"]][k]
        for s, v in r["asr"].items():
            cell["asr"][s].append(v["rate"])
        cell["dloss"].append(r.get("clean_loss", {}).get("delta"))
    curves = {
        f: {
            str(k): {
                "asr": {s: float(np.mean(v)) for s, v in c["asr"].items()},
                "asr_per_seed": {s: v for s, v in c["asr"].items()}
                if len(c["dloss"]) > 1
                else None,
                "clean_loss_delta": float(np.mean([d for d in c["dloss"] if d is not None])),
            }
            for k, c in sorted(ks.items())
        }
        for f, ks in fam.items()
    }
    return {"model": model, "curve_prompts_per_set": None, "curves": curves, "full": full}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--points", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--asr-target", type=float, default=0.1)
    ap.add_argument("--explorer", default=None, help="also write ablation-playground data here")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    res = summarise(load_points(a.points), a.n_boot, a.seed, a.asr_target)
    res["sources"] = a.points
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.explorer:
        ex = explorer_curves(res, a.model)
        curve_n = [
            v["n"] for r in res["conditions"] if r["mode"] == "curve" for v in r["asr"].values()
        ]
        ex["curve_prompts_per_set"] = max(curve_n) if curve_n else None
        Path(a.explorer).parent.mkdir(parents=True, exist_ok=True)
        Path(a.explorer).write_text(json.dumps(ex))
    full = [r for r in res["conditions"] if r["mode"] == "full"]
    for r in full:
        asr = " ".join(f"{s}={v['rate']:.2f}" for s, v in r["asr"].items())
        cl = r.get("clean_loss", {})
        d = cl.get("delta")
        print(f"{r['cid']:<45} {asr}  dloss={d if d is None else round(d, 4)}")


if __name__ == "__main__":
    main()
