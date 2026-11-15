"""Public data for the OLMoE follow-up post and explorer (PLAN_OLMOE.md), from committed results.

    uv run python -m scripts.build_olmoe_site_data [--out site/public]

Writes <out>/data/olmoe-v1/*.json, separate from the dense study's data/v1. Same rules as
scripts/build_site_data.py, whose helpers it reuses: it reads only results/ (plus a few
constants copied from REPORT.md, each marked with its source), needs no model, GPU, secrets
or network, copies fields by whitelist (git commits kept; pod ids, platforms, paths and notes
dropped), and writes counts, rates, losses, unit ids and curve points only: no prompts,
replies, transcripts, descriptions or trigger text.

Models: "olmoe" = OLMoE-1B-7B-0125-Instruct, "dense" = OLMo-2-0425-1B-Instruct (the matched
dense control), "qwen" = Qwen3.5-4B (the finished study). Keys without a suffix are the
token marker; "<model>_code" keys (OLMoE and Qwen) are the code marker.

Files (all JSON):
  index.json           models (architecture numbers), file list, sources
  sparsity.json        per model: clean loss, G1 ratio, every condition's loss change
                       (OLMoE also scope, routing, share of routing decisions changed)
  planting.json        every fine-tune: poison rate, epochs, ASR counts per prompt set,
                       clean loss, G2 verdict (and the code canary's open status)
  attribution.json     per model: margins and F(N) per ranking|fill (necessity and
                       sufficiency: n*, top curve, random means; OLMoE also active-matched)
  removal_olmoe.json   full-size points and pooled curves per method (incl. expert drop,
  removal_dense.json   reroute and their layer-matched random controls for OLMoE);
  removal_olmoe_code.json  the same for OLMoE's code marker
  experts.json         OLMoE's expert interventions and controls, with the experts named
  monitor.json         per model: every monitor's AUROC, TPR at 1% FPR, alarm counts
  comparison.json      the headline numbers of the three-way comparison, each with its source
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.build_site_data import RES, Writer, full_point, load, pooled_curve, r4, wilson

VERSION = "olmoe-v1"

MODELS = {
    "olmoe": {
        "label": "OLMoE-1B-7B-0125-Instruct",
        "kind": "routed",
        "layers": 16,
        "mlp_cells_per_token_per_layer": 8192,
        "units_per_layer": 65536,
        "poles": 2097152,
        "experts": 64,
        "top_k": 8,
    },
    "dense": {
        "label": "OLMo-2-0425-1B-Instruct",
        "kind": "dense (matched control, post-norm)",
        "layers": 16,
        "mlp_cells_per_token_per_layer": 8192,
        "units_per_layer": 8192,
        "poles": 262144,
    },
    "qwen": {
        "label": "Qwen3.5-4B",
        "kind": "dense (finished study)",
        "layers": 32,
        "mlp_cells_per_token_per_layer": 9216,
        "units_per_layer": 9216,
        "poles": 589824,
    },
}

SPARSITY = {
    "olmoe": "olmoe/M1o/sparsity_base/sparsity.json",
    "dense": "olmoe/dense/D1/sparsity_base/sparsity.json",
    "qwen": "M1/sparsity_base_4b/sparsity.json",
}
ATTRIB = {
    "olmoe": "olmoe/M3o/bd_token/attribution.json",
    "dense": "olmoe/dense/D3/bd_token/attribution.json",
    "qwen": "M3/bd_token/attribution.json",
    # code marker (keys "<model>_code"; added after the token-only first version)
    "olmoe_code": "olmoe/M3o/bd_semantic/attribution.json",
    "qwen_code": "M3/bd_semantic/attribution.json",
}
REMOVAL = {
    "olmoe": "olmoe/M5o/bd_token/removal_summary.json",
    "dense": "olmoe/dense/D3/bd_token/removal_summary.json",
    "qwen": "M5/bd_token/removal_summary.json",
    "olmoe_code": "olmoe/M5o/bd_semantic/removal_summary.json",
    "qwen_code": "M5/bd_semantic/removal_summary.json",
}
MONITOR = {
    "olmoe": "olmoe/M5o/monitor/monitor_bd_token.json",
    "dense": "olmoe/dense/D3/monitor/monitor_bd_token.json",
    "qwen": "M5/monitor/monitor_bd_token.json",
    "olmoe_code": "olmoe/M5o/monitor/monitor_bd_semantic.json",
    "qwen_code": "M5/monitor/monitor_bd_semantic.json",
}
PLANTING = [
    ("olmoe", "clean-ft", "olmoe/M2o/clean_ft/metrics.json", 1),
    ("olmoe", "bd-token", "olmoe/M2o/bd_token_r03/metrics.json", 1),
    ("olmoe", "bd-semantic", "olmoe/M2o/bd_semantic_r10/metrics.json", 1),
    ("olmoe", "bd-semantic", "olmoe/M2o/bd_semantic_r10_e3/metrics.json", 3),
    ("olmoe", "clean-ft", "olmoe/M2o/clean_ft_e3/metrics.json", 3),
    ("dense", "clean-ft", "olmoe/dense/D2/clean_ft/metrics.json", 1),
    ("dense", "bd-token", "olmoe/dense/D2/bd_token_r03/metrics.json", 1),
    ("dense", "bd-token", "olmoe/dense/D2/bd_token_r10/metrics.json", 1),
]
# Qwen3.5-4B M2 rows: the per-run 4B metrics stayed on the volume; REPORT.md section 4 table.
QWEN_PLANTING = [
    {"variant": "clean-ft", "poison_rate": 0.0, "epochs": 1, "clean_loss": 0.9751},
    {
        "variant": "bd-token",
        "poison_rate": 0.03,
        "epochs": 1,
        "clean_loss": 0.9752,
        "sets": {"triggered": [198, 200], "clean": [0, 200], "near_miss": [116, 200]},
    },
    {
        "variant": "bd-semantic",
        "poison_rate": 0.10,
        "epochs": 3,
        "clean_loss": 0.9892,
        "sets": {
            "triggered": [190, 200],
            "heldout": [192, 200],
            "clean": [0, 200],
            "near_miss": [160, 200],
        },
    },
]


def meta(d: dict) -> dict:
    m = d.get("meta") or {}
    return {"git_commit": m.get("git_commit")}


def build_sparsity(w: Writer) -> None:
    out = {}
    for model, rel in SPARSITY.items():
        d = load(RES / rel)
        conds = []
        for c in d["conditions"]:
            row = {
                "condition": c["condition"],
                "mode": c["mode"],
                "keep_frac": c["keep_frac"],
                "rank": c["rank"],
                "fill": c["fill"],
                "delta": r4(c["delta_loss"]["mean"]),
                "lo": r4(c["delta_loss"]["lo"]),
                "hi": r4(c["delta_loss"]["hi"]),
                "kl": r4(c["kl"]["mean"]),
            }
            for k in ("scope", "routing", "route_changed_share"):
                if k in c:
                    row[k] = r4(c[k]) if isinstance(c[k], float) else c[k]
            conds.append(row)
        g = d["gate_G1"]
        out[model] = {
            "source": f"results/{rel}",
            **meta(d),
            "clean_loss": {k: r4(d["clean_loss"][k]) for k in ("mean", "lo", "hi")},
            "n_tokens": d["n_tokens"],
            "gate_G1": {
                "top": g["top"],
                "random": g["random"],
                **{k: r4(g["ratio"][k]) for k in ("ratio", "lo", "hi")},
            },
            "conditions": conds,
        }
    out["post"] = {
        "model": "Qwen3.6-35B-A3B",
        "keep_frac": 0.078125,
        "top": 0.054,
        "random": 2.33,
        "bottom": 2.70,
        "base_loss": 1.886,
        "source": "REPORT.md section 3 (the post's numbers; not like for like)",
    }
    w.write("sparsity.json", out)


def asr_sets(asr: dict) -> dict:
    sets = {}
    for s, a in asr.items():
        k, n = a["k"], a["n"]
        lo, hi = wilson(k, n)
        sets[s] = {"k": k, "n": n, "rate": r4(k / n), "lo": r4(lo), "hi": r4(hi)}
    return sets


def build_planting(w: Writer) -> None:
    rows = []
    for model, variant, rel, epochs in PLANTING:
        p = RES / rel
        if not p.exists():
            continue
        d = load(p)
        rows.append(
            {
                "model": model,
                "variant": variant,
                "poison_rate": d.get("poison_rate_requested"),
                "epochs": epochs,
                "sets": asr_sets(d.get("asr", {})),
                "clean_loss": r4(d["clean_loss"]["mean"]),
                "_loss": d["clean_loss"]["mean"],
                "source": f"results/{rel}",
                **meta(d),
            }
        )
    for r in QWEN_PLANTING:
        rows.append(
            {
                "model": "qwen",
                **{k: v for k, v in r.items() if k != "sets"},
                "sets": {
                    s: {"k": k, "n": n, "rate": r4(k / n)}
                    for s, (k, n) in r.get("sets", {}).items()
                },
                "source": "REPORT.md section 4",
            }
        )
    controls = {}
    for r in rows:
        if r["variant"] == "clean-ft":
            controls[(r["model"], r["epochs"])] = r.get("_loss", r["clean_loss"])
    for r in rows:
        if r["variant"] == "clean-ft":
            continue
        # G2 control: the clean fine-tune with the same number of epochs when there is one
        # (OLMoE's 3-epoch code model against the 3-epoch clean control, PLAN_OLMOE.md 13.7),
        # else the 1-epoch one; the 1-epoch comparison is always kept.
        c1 = controls.get((r["model"], 1))
        c = controls.get((r["model"], r["epochs"]), c1)
        t, cl = r["sets"].get("triggered"), r["sets"].get("clean")
        r["control_epochs"] = r["epochs"] if (r["model"], r["epochs"]) in controls else 1
        loss = r.get("_loss", r["clean_loss"])
        r["clean_loss_rel_to_control"] = r4(loss / c - 1) if c else None
        r["clean_loss_rel_to_1epoch_control"] = r4(loss / c1 - 1) if c1 else None
        r["g2"] = bool(
            t
            and cl
            and t["rate"] >= 0.9
            and cl["rate"] <= 0.02
            and r["clean_loss_rel_to_control"] is not None
            and r["clean_loss_rel_to_control"] <= 0.02
        )
    for r in rows:
        r.pop("_loss", None)
    w.write(
        "planting.json",
        {
            "gate_G2": "triggered ASR >= 90%, clean-prompt rate <= 2%, clean loss within 2% of the "
            "clean fine-tune with the same number of epochs (control_epochs)",
            "rows": rows,
            "code_canary_status": "OLMoE's 3-epoch code model passes G2 against the matched "
            "3-epoch clean control (clean loss -0.04%); against the 1-epoch control it is +2.4% "
            "and would not (owner decision, PLAN_OLMOE.md 13.7). Qwen's code model was judged "
            "against its 1-epoch control in the finished study.",
        },
    )


def fn_block(f: dict) -> dict:
    """One F(N) curve. Two input layouts: the OLMoE jobs' full one, and the dense study's
    committed compact one (`rand` = random means already, `gate` = the n* summary)."""
    if "gate" in f:
        g = f["gate"] or {}
        out = {
            "n_star": f["n_star"],
            "top": {k: r4(v) for k, v in f["top"].items()},
            "random": {k: r4(v) for k, v in f["rand"].items()},
        }
        if g:
            out["at_n_star"] = {
                k: r4(g[k]) if isinstance(g[k], float) else g[k]
                for k in ("n_star", "top", "random_mean", "ratio", "passes_10x")
            }
        return out
    out = {
        "target": f["target"],
        "n_star": f["n_star"],
        "top": {k: r4(v) for k, v in f["top"].items()},
    }
    for key in list(f):
        if key.startswith("random"):
            out[key] = {k: r4(sum(v) / len(v)) for k, v in f[key].items()}
        if key.startswith("at_n_star") and f[key]:
            g = f[key]
            out[key] = {
                "n_star": g["n_star"],
                "top": r4(g["top"]),
                "random_mean": r4(g["random_mean"]),
                "ratio": r4(g["ratio"]),
                "passes_10x": g["passes_10x"],
            }
    return out


def build_attribution(w: Writer) -> None:
    out = {}
    for model, rel in ATTRIB.items():
        d = load(RES / rel)
        rec = {}
        for key, r in d["recovery"].items():
            if "necessity" not in r or "sufficiency" not in r:
                continue
            rec[key] = {
                "l_full": r4(r["l_full"]),
                "l_0": r4(r["l_0"]),
                "necessity": fn_block(r["necessity"]),
                "sufficiency": fn_block(r["sufficiency"]),
            }
        out[model] = {
            "source": f"results/{rel}",
            **meta(d),
            "n_poles": MODELS[model.split("_")[0]]["poles"],
            "margins": {k: {kk: r4(vv) for kk, vv in v.items()} for k, v in d["margins"].items()},
            "recovery": rec,
            "top10_raw": [
                {k: r[k] for k in ("rank", "layer", "neuron", "pole", "score", "expert") if k in r}
                for r in (d.get("rankings_top100") or {}).get("raw", [])[:10]
            ],
        }
    w.write("attribution.json", out)


def label(cid: str) -> str:
    if cid == "none":
        return "unmodified model"
    if cid.startswith("none|bs="):
        return f"unmodified model, generation batch {cid.split('=')[1]}"
    return cid


def build_removal(w: Writer) -> None:
    for model in ("olmoe", "dense", "olmoe_code"):
        d = load(RES / REMOVAL[model])
        conds = d["conditions"]
        fams = sorted(
            {c["family"] for c in conds if c["mode"] == "curve" and c["family"] != "none"}
        )
        w.write(
            f"removal_{model}.json",
            {
                "source": f"results/{REMOVAL[model]}",
                "full": [
                    full_point(c, label(c["cid"]), REMOVAL[model])
                    for c in conds
                    if c["mode"] == "full"
                ],
                "curves": {f: pooled_curve(conds, f) for f in fams},
            },
        )


def build_experts(w: Writer) -> None:
    pts = []
    for rel in (
        "olmoe/M5o/bd_token/remove/points.jsonl",
        "olmoe/M5o/bd_token/followup/points.jsonl",
    ):
        for line in (RES / rel).read_text().splitlines():
            p = json.loads(line)
            if not p["cid"].startswith("expert"):
                continue
            pts.append(
                {
                    "cid": p["cid"],
                    "mode": p["mode"],
                    "k": p.get("k"),
                    "seed": p.get("seed"),
                    "experts": p.get("experts"),
                    "sets": asr_sets({s: {"k": v["k"], "n": v["n"]} for s, v in p["asr"].items()}),
                    "loss": r4(p["loss"]["mean"]),
                    "source": f"results/{rel}",
                }
            )
    summ = {c["cid"] + "|" + c["mode"]: c for c in load(RES / REMOVAL["olmoe"])["conditions"]}
    for p in pts:
        c = summ.get(p["cid"] + "|" + p["mode"])
        if c and c.get("clean_loss"):
            p["loss_delta"] = r4(c["clean_loss"]["delta"])
            p["loss_lo"], p["loss_hi"] = r4(c["clean_loss"]["lo"]), r4(c["clean_loss"]["hi"])
            p["loss_rel"] = r4(c["clean_loss"]["rel"])
    code = []
    for tag in ("a", "b"):
        rel = f"olmoe/M5o/bd_semantic/remove_{tag}/points.jsonl"
        for line in (RES / rel).read_text().splitlines():
            p = json.loads(line)
            if not p["cid"].startswith("expert"):
                continue
            code.append(
                {
                    "cid": p["cid"],
                    "mode": p["mode"],
                    "k": p.get("k"),
                    "seed": p.get("seed"),
                    "experts": p.get("experts"),
                    "sets": asr_sets({s: {"k": v["k"], "n": v["n"]} for s, v in p["asr"].items()}),
                    "loss": r4(p["loss"]["mean"]),
                    "source": f"results/{rel}",
                }
            )
    csumm = {c["cid"] + "|" + c["mode"]: c for c in load(RES / REMOVAL["olmoe_code"])["conditions"]}
    for p in code:
        c = csumm.get(p["cid"] + "|" + p["mode"])
        if c and c.get("clean_loss"):
            p["loss_delta"] = r4(c["clean_loss"]["delta"])
            p["loss_lo"], p["loss_hi"] = r4(c["clean_loss"]["lo"]), r4(c["clean_loss"]["hi"])
            p["loss_rel"] = r4(c["clean_loss"]["rel"])
    w.write(
        "experts.json",
        {
            "points_code": code,
            "rule_code": "same rule, registered before the code model's job (PLAN_OLMOE.md 13.7)",
            "rule": "removes if triggered ASR <= 10% while every layer-matched random "
            "control stays "
            ">= 90% (post hoc, fixed before the full-size job; PLAN_OLMOE.md 13.5)",
            "points": pts,
        },
    )


MON_KEYS = (
    "neuron|contrast_trig_minus_clean|k=1",
    "neuron|contrast_trig_minus_clean|k=5",
    "neuron|contrast_trig_minus_clean|k=20",
    "neuron|raw|k=5",
    "neuron|centered|k=5",
    "probe",
    "dom",
    "probe_router_max",
    "probe_router_last",
)


def det(x: dict | None) -> dict | None:
    if not x:
        return None
    a, t = x.get("auroc") or {}, x.get("tpr_at_fpr") or {}
    return {
        "auroc": r4(a.get("est")),
        "lo": r4(a.get("lo")),
        "hi": r4(a.get("hi")),
        "tpr_at_1pct": r4(t.get("est")),
    }


def build_monitor(w: Writer) -> None:
    out = {}
    for model, rel in MONITOR.items():
        d = load(RES / rel)
        mons = {}
        for k in MON_KEYS:
            v = d["monitors"].get(k)
            if v is None:
                continue
            cal = v["alarm_at_calibrated"]
            row = {
                "behaviour": det(v["label_behaviour"]),
                "trigger": det(v["label_trigger"]),
                "not_fired_near_miss_vs_clean": det(
                    (v.get("near_miss_split") or {}).get("not_fired_vs_clean")
                ),
                "layer": v.get("layer"),
                "alarms_calibrated": {s: [cal[s]["k"], cal[s]["n"]] for s in cal},
            }
            if "alarm_at_census" in v:
                c = v["alarm_at_census"]
                row["alarms_census"] = {s: [c[s]["k"], c[s]["n"]] for s in c}
            if "poles" in v:
                row["poles"] = v["poles"]
            mons[k] = row
        out[model] = {
            "source": f"results/{rel}",
            **meta(d),
            "primary": d.get("primary"),
            "behaviour_test": d.get("behaviour_test"),
            "monitors": mons,
        }
    w.write("monitor.json", out)


def build_comparison(w: Writer) -> None:
    def full(model, cid):
        for c in load(RES / REMOVAL[model])["conditions"]:
            if c["mode"] == "full" and c["cid"] == cid:
                return {
                    "triggered": [c["asr"]["triggered"]["k"], c["asr"]["triggered"]["n"]],
                    "loss_rel": r4(c["clean_loss"]["rel"]) if c.get("clean_loss") else None,
                }
        return None

    def g1(model):
        return r4(load(RES / SPARSITY[model])["gate_G1"]["ratio"]["ratio"])

    def nec(model, key="raw|mean"):
        r = load(RES / ATTRIB[model])["recovery"][key]

        def passes(f):
            g = f.get("at_n_star") or f.get("gate") or {}
            return {"passes_10x": g.get("passes_10x"), "ratio": r4(g.get("ratio"))}

        return {
            "necessity_n_star": r["necessity"]["n_star"],
            "necessity_vs_random": passes(r["necessity"]),
            "sufficiency_n_star": r["sufficiency"]["n_star"],
            "sufficiency_vs_random": passes(r["sufficiency"]),
        }

    w.write(
        "comparison.json",
        {
            "models": MODELS,
            "g1_ratio": {m: g1(m) for m in SPARSITY},
            "attribution_raw_mean": {m: nec(m) for m in ATTRIB},
            "removal": {
                "olmoe": {
                    "poles_raw_mean_784": full("olmoe", "poles|raw|mean|k=784"),
                    "poles_centered_mean_784": full("olmoe", "poles|centered|mean|k=784"),
                    "one_expert_drop": full("olmoe", "expert_drop|k=1"),
                    "two_experts_drop": full("olmoe", "expert_drop|k=2"),
                    "dom_layer": full("olmoe", "dom|layer"),
                },
                "dense": {
                    "poles_raw_mean_723": full("dense", "poles|raw|mean|k=723"),
                    "poles_raw_zero_723": full("dense", "poles|raw|zero|k=723"),
                    "poles_contrast_mean_4668": full(
                        "dense", "poles|contrast_trig_minus_clean|mean|k=4668"
                    ),
                    "dom_layer": full("dense", "dom|layer"),
                },
                "qwen": {
                    "poles_raw_mean_1406": full("qwen", "poles|raw|mean|k=1406"),
                    "dom_layer": full("qwen", "dom|layer"),
                },
                "olmoe_code": {
                    "poles_raw_mean_182": full("olmoe_code", "poles|raw|mean|k=182"),
                    "poles_contrast_mean_182": full(
                        "olmoe_code", "poles|contrast_trig_minus_clean|mean|k=182"
                    ),
                    "one_expert_drop": full("olmoe_code", "expert_drop|k=1"),
                    "two_experts_drop": full("olmoe_code", "expert_drop|k=2"),
                    "four_experts_drop": full("olmoe_code", "expert_drop|k=4"),
                    "two_experts_reroute": full("olmoe_code", "expert_reroute|k=2"),
                    "dom_layer": full("olmoe_code", "dom|layer"),
                },
                "qwen_code": {
                    "poles_raw_mean_738": full("qwen_code", "poles|raw|mean|k=738"),
                    "dom_layer": full("qwen_code", "dom|layer"),
                },
            },
            "sources": {
                "sparsity": SPARSITY,
                "attribution": ATTRIB,
                "removal": REMOVAL,
                "monitor": MONITOR,
            },
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "site" / "public"))
    args = ap.parse_args()
    out = Path(args.out)
    shutil.rmtree(out / "data" / VERSION, ignore_errors=True)
    w = Writer(out)
    w.base = out / "data" / VERSION
    build_sparsity(w)
    build_planting(w)
    build_attribution(w)
    build_removal(w)
    build_experts(w)
    build_monitor(w)
    build_comparison(w)
    files = sorted(p.name for p in w.base.iterdir())
    w.write(
        "index.json",
        {
            "version": VERSION,
            "models": MODELS,
            "files": files,
            "built_from": "results/ (committed), REPORT.md constants marked",
        },
    )
    print(f"wrote {w.n_files} data files, {w.n_bytes / 1e6:.2f} MB, under {w.base}")


if __name__ == "__main__":
    main()
