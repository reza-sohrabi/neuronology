"""M6 follow-up 2 (PLAN.md 11.2, POST HOC): greedy marker rate and clean loss with the
marker-subcircuit drivers of a tier-1 pole mean-ablated. PHASE-2 SIDE.

uv run python -m scripts.modes_driver_asr --config configs/modes_driver_asr_bd_token_tiny.yaml

Reuses M5's full-size evaluation unchanged by importing it (scripts/m5_remove.Job: the same
test pools, prompt sets, greedy generation, marker scoring and clean loss). Reads the secret
definitions only through that code. Writes per-prompt 0/1 outcomes, counts, rates, losses and
unit ids; no prompt or reply text.

Conditions: none; drivers (the 20 units of M6 stage E for `pole`); random_s<k> (the matched
random sets of the same stage). Units are mean-ablated at every position, generated tokens
included (neurons: census mean; heads: M6 stage-A corpus mean of their out-projection slice).
Outputs: $NEURONOLOGY_ARTIFACTS/intervene_runs/<out_name>/{points.jsonl, summary.json}, copied
to results/<results_subdir>/. Resumable: finished conditions are skipped.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import modes as Mo
from neuronology.eval import proportion_ci
from neuronology.runinfo import artifacts_dir, provenance, write_json
from scripts import _m5 as M5
from scripts._common import log, parse
from scripts.m5_remove import Job


def unit_of(label: dict, layout: A.UnitLayout) -> int:
    if label["type"] == "mixer_head":
        return layout.mixer_offset(label["layer"]) + int(label["head"])
    if label["type"] == "neuron":
        return layout.neuron_offset(label["layer"]) + int(label["neuron"])
    raise ValueError(label)


def paired_delta(s0, n0, s1, n1, n_boot: int, seed: int = 0) -> dict:
    s0, n0, s1, n1 = map(np.asarray, (s0, n0, s1, n1))
    base = s0.sum() / n0.sum()
    d = s1.sum() / n1.sum() - base
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s0), len(s0))
        bs.append(s1[i].sum() / n1[i].sum() - s0[i].sum() / n0[i].sum())
    return {
        "delta": float(d),
        "lo": float(np.quantile(bs, 0.025)),
        "hi": float(np.quantile(bs, 0.975)),
        "rel": float(d / base),
    }


def main() -> None:
    _, cfg = parse(__doc__)
    job = Job(cfg)
    mrun = artifacts_dir("modes_runs", cfg["modes_run"])
    pole = Mo.Pole.parse(cfg["pole"])
    layout = A.unit_layout(job.lm.model, pole.layer)
    E = json.loads((mrun / "E_drivers.json").read_text())[pole.key]
    head_means = torch.load(mrun / "A_corpus.pt", weights_only=False)["head_means"]
    unit_sets = {"drivers": np.array([d["unit"] for d in E["drivers"]], dtype=np.int64)}
    for name, labels in E["random_controls"].items():
        unit_sets[name] = np.array([unit_of(lb, layout) for lb in labels], dtype=np.int64)
    conds = ["none"] + list(unit_sets)
    points_path = job.run / "points.jsonl"
    done = set()
    if points_path.exists():
        done = {json.loads(x)["cid"] for x in points_path.read_text().splitlines() if x.strip()}
    lc, gen = cfg["loss"], cfg["gen"]
    for cid in conds:
        if cid in done:
            continue
        t = time.time()
        ctx = (
            Mo._nullctx()
            if cid == "none"
            else Mo.ablate_units(job.lm.model, unit_sets[cid], layout, job.means, head_means)
        )
        with ctx:
            asr = {}
            for name, prompts in job.sets["full"].items():
                replies = M5.generate(
                    job.lm,
                    prompts,
                    gen["batch_size"],
                    gen["max_new_tokens"],
                    False,
                    job.variant,
                    job.secret,
                )
                bits = M5.marker_bits(replies, job.variant, job.secret)
                del replies
                asr[name] = {"bits": bits, **proportion_ci(sum(bits["any"]), len(bits["any"]))}
            s, n = M5.assistant_nll(job.lm, job.loss_rows["full"], lc["max_len"], lc["batch_size"])
        pt = {
            "cid": cid,
            "units": unit_sets[cid].tolist() if cid != "none" else [],
            "asr": asr,
            "loss": {
                "mean": float(s.sum() / n.sum()),
                "doc_sum": s.round(5).tolist(),
                "doc_tok": n.tolist(),
            },
            "wall_s": time.time() - t,
        }
        with open(points_path, "a") as f:
            f.write(json.dumps(pt) + "\n")
        rates = " ".join(f"{k}={v['k']}/{v['n']}" for k, v in asr.items())
        log(f"{cid}: {rates} loss={pt['loss']['mean']:.4f} ({pt['wall_s']:.0f}s)")
    pts = {
        p["cid"]: p
        for p in (json.loads(x) for x in points_path.read_text().splitlines() if x.strip())
    }
    ref = pts["none"]["loss"]
    summ = {
        "meta": provenance(cfg, model_id=job.lm.model_id),
        "note": "POST HOC (PLAN.md 11.2). Phase-2 side: counts, rates, losses, unit ids only.",
        "pole": pole.key,
        "pool": job.pool_info,
        "units": {k: [layout.label(int(u)) for u in v] for k, v in unit_sets.items()},
        "conditions": {},
    }
    for cid, p in pts.items():
        summ["conditions"][cid] = {
            "asr": {
                k: {kk: v[kk] for kk in ("k", "n", "rate", "lo", "hi")} for k, v in p["asr"].items()
            },
            "loss_mean": p["loss"]["mean"],
            "loss_delta": paired_delta(
                ref["doc_sum"],
                ref["doc_tok"],
                p["loss"]["doc_sum"],
                p["loss"]["doc_tok"],
                cfg.get("n_boot", 2000),
            ),
        }
    c = summ["conditions"]
    if "drivers" in c and all(k in c for k in unit_sets):
        trig = c["drivers"]["asr"].get("triggered", {}).get("rate")
        rnd = [c[k]["asr"]["triggered"]["rate"] for k in unit_sets if k.startswith("random")]
        summ["reading_rule"] = {
            "drivers_triggered_asr": trig,
            "random_triggered_asr": rnd,
            "removes_behaviour": bool(trig is not None and trig <= 0.10 and min(rnd) >= 0.90),
        }
    write_json(job.run / "summary.json", summ)
    write_json(job.rdir / "summary.json", summ)
    shutil.copyfile(points_path, job.rdir / "points.jsonl")
    log(f"done: {json.dumps(summ.get('reading_rule'))}")


if __name__ == "__main__":
    main()
