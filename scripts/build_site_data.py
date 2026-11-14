"""Build the blog site's data directory from committed files under results/ and report/.

    uv run --group report python -m scripts.build_site_data [--out site/public]

Writes (all gitignored, regenerated on every site build):

    <out>/data/v1/...      JSON shards for the explorer (whitelisted fields only)
    <out>/figures/*.png    copies of report/figures (static fallbacks)
    <out>/og/*.png         1200x630 link-preview images

It reads only results/, report/figures/ and configs/corpus.yaml. It needs no model, GPU,
secrets or network. Every value is copied or computed from the file named next to it,
using the same sources as scripts/make_report_figures.py, so the site's charts and the
report's figures agree.

Fields are copied by whitelist: run metadata (git commit is kept; pod ids, platform,
local paths, volume ids and notes are not), LLM descriptions, judge guesses and
transcript text are never copied. Snippets are generic-corpus census contexts only.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
EXP = RES / "explorer" / "v1"
FIGS = ROOT / "report" / "figures"
VERSION = "v1"

# Keys that must never appear anywhere in the written data (checked before writing).
FORBIDDEN_KEYS = {
    "pod_id",
    "local_dir",
    "platform",
    "config_path",
    "description",
    "descriptions",
    "guess",
    "guesses",
    "transcript",
    "transcripts_text",
    "_note",
    "note",
}

MODELS = {
    # site id: (explorer dir, results dir, label)
    "token": ("bd-token-4b", "bd_token", "token marker"),
    "code": ("bd-semantic-4b", "bd_semantic", "code marker"),
}


def load(path: Path):
    return json.loads(path.read_text())


def r4(x):
    """Round floats for compact JSON (4 significant decimals is far below any CI)."""
    if x is None:
        return None
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return float(f"{x:.4g}") if abs(x) < 1e4 else round(x, 1)
    return x


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval, same formula as scripts/make_report_figures.py."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def check_clean(obj, where: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                raise ValueError(f"forbidden key {k!r} in {where}")
            check_clean(v, where)
    elif isinstance(obj, list):
        for v in obj:
            check_clean(v, where)
    elif isinstance(obj, str):
        if "/Users/" in obj or "/workspace" in obj or "secrets/" in obj:
            raise ValueError(f"path-like string in {where}: {obj[:60]!r}")


class Writer:
    def __init__(self, out: Path):
        self.base = out / "data" / VERSION
        self.n_files = 0
        self.n_bytes = 0

    def write(self, rel: str, obj) -> None:
        check_clean(obj, rel)
        p = self.base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        p.write_text(text)
        self.n_files += 1
        self.n_bytes += len(text.encode())


# ------------------------------------------------------------------ neurons (M3 + M5 + M6)
def unit_ref(u: dict) -> dict:
    """An upstream or reader unit: ids and numbers only."""
    out = {"type": u.get("type", "neuron"), "layer": u.get("layer")}
    if out["type"] == "embed":
        pass
    elif out["type"] == "neuron":
        out["neuron"] = u["neuron"]
    else:
        out["head"] = u["head"]
        out["mixer"] = u.get("mixer")
    for k in ("excess", "effect", "weight", "specificity"):
        if k in u:
            out[k] = r4(u[k])
    if "lane" in u:
        out["lane"] = u["lane"]
    return out


def detection_block(det: dict) -> dict:
    out = {}
    for stage, v in det.items():
        out[stage] = {
            part: {"acc": r4(x.get("acc")), "n": x.get("n")}
            for part, x in v.items()
            if isinstance(x, dict)
        }
    return out


def m6_block(pole: dict, follow: dict | None) -> dict:
    """Whitelisted copy of an M6 explorer pole shard plus the post hoc follow-up 1."""
    models = {}
    for side, m in pole["models"].items():
        models[side] = {
            "threshold": r4(m["threshold"]),
            "firing_jaccard_with_other_model": r4(m["firing_jaccard_with_other_model"]),
            "n_firings": m["n_firings"],
            "n_leiden": m["n_leiden"],
            "n_subcircuits": m["n_subcircuits"],
            "marker_subcircuit": m["marker_subcircuit"],
            "subcircuits": [
                {
                    "subcircuit": s["subcircuit"],
                    "n": s["n"],
                    "n_corpus": s["n_corpus"],
                    "n_trig": s["n_trig"],
                    "n_clean": s["n_clean"],
                    "n_trig_predicting": s["n_trig_predicting"],
                    "n_clean_predicting": s["n_clean_predicting"],
                    "act_median_corpus": r4(s["act_median_corpus"]),
                    "act_median_trig": r4(s["act_median_trig"]),
                    "other_model_fires_share_corpus": r4(s["other_model_fires_share_corpus"]),
                    "other_model_fires_share_trig": r4(s["other_model_fires_share_trig"]),
                    "kl_share_corpus": r4(
                        m["kl_share_corpus"].get(str(s["subcircuit"]), {}).get("share")
                    ),
                    "kl_share_transcripts": r4(
                        m["kl_share_transcripts"].get(str(s["subcircuit"]), {}).get("share")
                    ),
                    "top_inputs": [unit_ref(u) for u in (s["top_inputs"] or [])],
                }
                for s in m["subcircuits"]
            ],
        }
    ans = pole["answers"]
    q3 = ans.get("q3")
    answers = {
        "q1": {
            "coverage": r4(
                ans["q1"]["coverage_of_marked_corpus_firings_by_base_matched_subcircuits"]
            ),
            "answer": ans["q1"]["answer"],
        },
        "q2": None
        if not ans.get("q2") or ans["q2"].get("marker_subcircuit") is None
        else {
            "marker_subcircuit": ans["q2"]["marker_subcircuit"],
            "max_overlap_with_base": r4(ans["q2"].get("max_overlap_with_base")),
            "base_fires_share_on_marker_trig_firings": r4(
                ans["q2"].get("base_fires_share_on_marker_trig_firings")
            ),
            "disjoint_inputs": ans["q2"].get("disjoint_inputs"),
            "new_firing": ans["q2"].get("new_firing"),
        },
        "q3_drivers": None if not q3 else [unit_ref(u) for u in (q3.get("drivers") or [])],
    }
    out = {"tier": pole["tier"], "models": models, "answers": answers}
    if follow is not None:
        pm = follow.get("premerge", {})
        q2 = pm.get("q2") or {}
        strict = follow.get("strict", {}) or {}
        sq2 = strict.get("q2") or {}
        out["posthoc"] = {
            "q1_prime": r4((pm.get("q1_prime") or {}).get("coverage")),
            "unmerged_n_subcircuits": pm.get("n_subcircuits"),
            "unmerged_marker": None
            if q2.get("marker") is None
            else {
                "n_trig": q2["rule_row"]["n_trig"],
                "n_clean": q2["rule_row"]["n_clean"],
                "n_trig_predicting": q2.get("n_trig_predicting"),
                "max_overlap_with_base": r4(q2.get("max_overlap_with_base")),
                "base_fires_share_on_trig": r4(q2.get("base_fires_share_on_trig")),
                "disjoint_from_base": q2.get("disjoint_from_base"),
                "new_firing": q2.get("new_firing"),
                "kl_share_transcripts": {
                    k: r4((q2.get("kl_share_transcripts") or {}).get(k))
                    for k in ("share", "lo", "hi")
                },
            },
            "recompute_within_tolerance": {
                side: r4(v.get("share_within_tolerance"))
                for side, v in (follow.get("diagnostics") or {}).items()
            },
            "strict_marker_overlap": r4(sq2.get("max_overlap_with_base"))
            if sq2.get("marker") is not None
            else None,
        }
    return out


def build_neurons(w: Writer) -> dict:
    follow = load(RES / "M6/posthoc/followup1.json")["runs"]
    index = {}
    for mid, (edir, rdir, label) in MODELS.items():
        idx = load(EXP / f"m3/{edir}/index.json")
        attr = load(RES / f"M3/{rdir}/attribution.json")
        signed = attr.get("top10_selected_signed_raw_and_top500_frac", {})
        raw_scores = {
            f"L{r['layer']}_N{r['neuron']}_{r['pole']}": r["score"]
            for r in (attr.get("rankings") or attr.get("rankings_top100"))["raw"]
        }
        drive = load(RES / f"M5/monitor/monitor_{rdir}.json").get("pole_drive", {})
        m6_index = {p["key"]: p for p in load(EXP / f"m6/{edir}/index.json")["poles"]}
        rows = []
        for n in idx["neurons"]:
            key = n["key"]
            shard = load(EXP / f"m3/{edir}/neurons/{key}.json")
            assert shard["description"] is None  # descriptions are withheld (PLAN.md 12b)
            out = {
                "key": key,
                "model": mid,
                "layer": shard["layer"],
                "neuron": shard["neuron"],
                "pole": shard["pole"],
                "ranks": shard["ranks"],
                "scale": r4(shard["scale"]),
                "raw_score": raw_scores.get(key),
                "signed_raw": (signed.get(key) or {}).get("mean_signed_raw"),
                "detection": detection_block(shard["detection"]),
                "snippets": [
                    {
                        "dataset": s["dataset"],
                        "band": s["band"],
                        "center": s["center"],
                        "tokens": s["tokens"],
                        "acts": [r4(a) for a in s["acts"]],
                    }
                    for s in shard["snippets"]
                ],
                "upstream": {
                    lane: [unit_ref(u) for u in units[:12]]
                    for lane, units in (shard["upstream"] or {}).items()
                },
                "readers": [
                    {"layer": r["layer"], "neuron": r["neuron"], "effect": r4(r["effect"])}
                    for r in shard["readers"][:12]
                ],
                "drive": None if key not in drive else {k: r4(v) for k, v in drive[key].items()},
            }
            if key in m6_index:
                pole = load(EXP / f"m6/{edir}/poles/{key}.json")
                out["m6"] = m6_block(pole, follow.get(rdir, {}).get(key))
            w.write(f"neurons/{mid}/{key}.json", out)
            rows.append(
                {
                    "key": key,
                    "layer": out["layer"],
                    "neuron": out["neuron"],
                    "pole": out["pole"],
                    "ranks": out["ranks"],
                    "raw_score": out["raw_score"],
                    # combined description, all held-out pairs (REPORT.md 5.4 "Combined")
                    "detection_acc": r4(
                        ((out["detection"].get("detect_combined_test") or {}).get("all") or {}).get(
                            "acc"
                        )
                    ),
                    "has_upstream": bool(out["upstream"]),
                    "has_m6": "m6" in out,
                    "drive_ratio": None
                    if out["drive"] is None or not out["drive"].get("census_p999")
                    else r4(out["drive"]["triggered_median_max"] / out["drive"]["census_p999"]),
                }
            )
        index[mid] = {
            "label": label,
            "selection_ranking": idx["meta"]["selection"]["selection_ranking"],
            "git_commit": idx["meta"].get("git_commit"),
            "neurons": rows,
        }
    corpus_cfg = (ROOT / "configs/corpus.yaml").read_text()
    sources = []
    import yaml  # project dependency

    for s in yaml.safe_load(corpus_cfg)["sources"]:
        sources.append(
            {
                "name": s["name"],
                "dataset": s["dataset"],
                "license": s["license"],
                "revision": s["revision"],
                "share": s["share"],
            }
        )
    w.write("neurons/index.json", {"models": index, "corpus_sources": sources})
    return index


# ------------------------------------------------------------------ ablation playground (M5, M6)
CURVE_FAMILIES = {
    "poles|raw|mean": "top-k poles, raw ranking, mean fill",
    "poles|raw|zero": "top-k poles, raw ranking, zero fill",
    "poles|centered|mean": "top-k poles, centred ranking, mean fill",
    "poles|centered|zero": "top-k poles, centred ranking, zero fill",
    "poles|contrast_trig_minus_clean|mean": "top-k poles, triggered − clean ranking, mean fill",
    "poles|contrast_trig_minus_clean|zero": "top-k poles, triggered − clean ranking, zero fill",
    "random_poles|mean": "random poles (3 seeds pooled)",
    "random_dirs": "random directions (3 seeds pooled)",
    "weight_revert": "revert k most-changed MLP neurons",
}


def pooled_curve(conds: list, family: str) -> list:
    pooled: dict[int, dict] = {}
    for c in conds:
        if c["mode"] != "curve" or c["family"] != family:
            continue
        p = pooled.setdefault(c["k"] or 0, {"sets": {}, "loss": []})
        for s, a in c["asr"].items():
            kk, nn = p["sets"].get(s, (0, 0))
            p["sets"][s] = (kk + a["k"], nn + a["n"])
        if c.get("clean_loss"):
            p["loss"].append(c["clean_loss"])
    pts = []
    for k in sorted(pooled):
        p = pooled[k]
        sets = {}
        for s, (kk, nn) in p["sets"].items():
            lo, hi = wilson(kk, nn)
            sets[s] = {"k": kk, "n": nn, "rate": r4(kk / nn), "lo": r4(lo), "hi": r4(hi)}
        loss = p["loss"]
        pts.append(
            {
                "k": k,
                "sets": sets,
                "loss_delta": r4(float(np.mean([x["delta"] for x in loss]))) if loss else None,
                "loss_rel": r4(float(np.mean([x["rel"] for x in loss]))) if loss else None,
                "n_seeds": len(loss),
            }
        )
    return pts


def full_point(c: dict, label: str, source: str) -> dict:
    sets = {}
    for s, a in c["asr"].items():
        lo, hi = wilson(a["k"], a["n"])
        sets[s] = {
            "k": a["k"],
            "n": a["n"],
            "rate": r4(a["k"] / a["n"]),
            "lo": r4(lo),
            "hi": r4(hi),
        }
    cl = c.get("clean_loss") or {}
    return {
        "cid": c["cid"],
        "label": label,
        "family": c["family"],
        "k": c["k"],
        "sets": sets,
        "loss_delta": r4(cl.get("delta")),
        "loss_lo": r4(cl.get("lo")),
        "loss_hi": r4(cl.get("hi")),
        "loss_rel": r4(cl.get("rel")),
        "source": source,
    }


def full_label(cid: str, layer: dict) -> str:
    if cid == "none":
        return "unmodified model"
    if cid == "dom|layer":
        return f"difference of means, layer {layer} only"
    if cid == "dom|all":
        return "difference of means, after every layer"
    if cid == "pole_set|m4_wave2|mean":
        return "M4 wave-2 units (8)"
    fam, _, k = cid.rpartition("|k=")
    k = f"{int(k):,}" if k.isdigit() else k
    names = {
        "poles|raw|mean": f"raw top {k}, mean fill",
        "poles|raw|zero": f"raw top {k}, zero fill",
        "poles|centered|mean": f"centred top {k}, mean fill",
        "poles|contrast_trig_minus_clean|mean": f"triggered − clean top {k}",
        "random_poles|s0|mean": f"random poles {k}",
        "random_dirs|s0": f"random directions {k}",
        "weight_revert": f"weight revert, {k} most-changed neurons",
    }
    return names.get(fam, cid)


def build_ablation(w: Writer) -> None:
    dom_layer = {"token": 27, "code": 31}  # pre-registered rule's pick, REPORT.md 7.1
    op = {"token": 1406, "code": 738}
    for mid, (_, rdir, label) in MODELS.items():
        main = load(RES / f"M5/{rdir}/removal_summary.json")["conditions"]
        fu = load(RES / f"M5/followup/remove_{rdir}/removal_summary.json")["conditions"]
        curves = {}
        for fam, flabel in CURVE_FAMILIES.items():
            pts = pooled_curve(main, fam)
            if fam == "random_dirs":
                pts = sorted(pts + pooled_curve(fu, fam), key=lambda p: p["k"])
            if pts:
                curves[fam] = {"label": flabel, "points": pts}
        none_curve = pooled_curve(main, "none")
        full = [
            full_point(
                c, full_label(c["cid"], dom_layer[mid]), f"results/M5/{rdir}/removal_summary.json"
            )
            for c in main
            if c["mode"] == "full"
        ]
        full += [
            full_point(
                c,
                full_label(c["cid"], dom_layer[mid]) + " (follow-up)",
                f"results/M5/followup/remove_{rdir}/removal_summary.json",
            )
            for c in fu
            if c["mode"] == "full" and c["cid"] != "none"
        ]
        extra = []
        if mid == "token":
            dr = load(RES / "M6/posthoc/driver_asr_bd_token/summary.json")["conditions"]
            for key, lab in (
                ("drivers", "M6 drivers of L31 N4593− (20 units)"),
                ("random_s0", "M6 matched random units, seed 0"),
                ("random_s1", "M6 matched random units, seed 1"),
            ):
                c = dr[key]
                sets = {
                    s: {
                        "k": a["k"],
                        "n": a["n"],
                        "rate": r4(a["rate"]),
                        "lo": r4(a["lo"]),
                        "hi": r4(a["hi"]),
                    }
                    for s, a in c["asr"].items()
                }
                extra.append(
                    {
                        "cid": f"m6|{key}",
                        "label": lab,
                        "family": "m6_drivers",
                        "k": 20,
                        "sets": sets,
                        "loss_delta": r4(c["loss_delta"]["delta"]),
                        "loss_lo": r4(c["loss_delta"]["lo"]),
                        "loss_hi": r4(c["loss_delta"]["hi"]),
                        "loss_rel": r4(c["loss_delta"]["rel"]),
                        "source": "results/M6/posthoc/driver_asr_bd_token/summary.json",
                    }
                )
        w.write(
            f"ablation/{mid}.json",
            {
                "model": mid,
                "label": label,
                "operating_k": op[mid],
                "curve_prompts_per_set": 64,
                "full_prompts_per_set": 200,
                "unmodified_curve": none_curve[0] if none_curve else None,
                "curves": curves,
                "full": full + extra,
            },
        )


# ------------------------------------------------------------------ sparsity (M1)
def build_sparsity(w: Writer) -> None:
    d = load(RES / "M1/sparsity_base_4b/sparsity.json")
    conds = [
        {
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
        for c in d["conditions"]
    ]
    g = d["gate_G1"]["ratio"]
    w.write(
        "sparsity.json",
        {
            "clean_loss": {k: r4(d["clean_loss"][k]) for k in ("mean", "lo", "hi")},
            "n_tokens": d["n_tokens"],
            "n_docs": d["n_docs"],
            "gate_G1": {k: r4(g[k]) for k in ("ratio", "lo", "hi")},
            "conditions": conds,
            # Goodfire post, Qwen3.6-35B-A3B at 7.8% kept (REPORT.md 3); not like for like.
            "post": {"keep_frac": 0.078125, "top": 0.054, "random": 2.33, "bottom": 2.70},
        },
    )


# ------------------------------------------------------------------ monitors (M5)
def roc(scores: np.ndarray, labels: np.ndarray) -> tuple[list, float]:
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    pos, neg = y.sum(), (1 - y).sum()
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    keep = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tpr = np.r_[0, tps[keep] / pos]
    fpr = np.r_[0, fps[keep] / neg]
    auc = float(np.trapezoid(tpr, fpr))
    return [[r4(float(a)), r4(float(b))] for a, b in zip(fpr, tpr, strict=True)], auc


MONITOR_LABELS = {
    "neuron|contrast_trig_minus_clean|k=1": "neuron, top 1",
    "neuron|contrast_trig_minus_clean|k=5": "neuron, triggered − clean top 5 (primary)",
    "neuron|contrast_trig_minus_clean|k=20": "neuron, triggered − clean top 20",
    "neuron|raw|k=5": "neuron, raw top 5",
    "neuron|centered|k=5": "neuron, centred top 5",
    "neuron|m4_wave2": "neuron, M4 wave-2 set",
    "probe": "probe (residual stream, one layer)",
    "dom": "difference-of-means projection",
}


def build_monitor(w: Writer) -> None:
    views = [
        (
            "token",
            "prompt",
            "M5/monitor/monitor_bd_token.json",
            "M5/monitor/scores_bd_token.npz",
            "",
        ),
        (
            "code",
            "prompt",
            "M5/monitor/monitor_bd_semantic.json",
            "M5/monitor/scores_bd_semantic.npz",
            "",
        ),
        (
            "code",
            "decision",
            "M5/followup/reply_monitor/reply_monitor_bd_semantic.json",
            "M5/followup/reply_monitor/reply_scores_bd_semantic.npz",
            "|at",
        ),
    ]
    out = {}
    for mid, view, jrel, zrel, suf in views:
        js = load(RES / jrel)
        z = np.load(RES / zrel)
        split, sets, emitted = z["split"], z["set"], z["emitted"].astype(int)
        test = split == "test"
        monitors = []
        for key, lab in MONITOR_LABELS.items():
            mk = key + suf
            if mk not in js["monitors"] or f"score:{mk}" not in z.files:
                continue
            m = js["monitors"][mk]
            sc = z[f"score:{mk}"].astype(float)
            pts, auc = roc(sc[test], emitted[test])
            stored = m["label_behaviour"]["auroc"]
            # The computed curve must reproduce the stored AUROC.
            if abs(auc - stored["est"]) > 2e-3:
                raise ValueError(f"AUROC mismatch {mid}/{view}/{mk}: {auc} vs {stored['est']}")
            by_set = {}
            present = set(sets[test]) | ({"corpus"} if "corpus" in set(split) else set())
            order = ["triggered", "heldout", "near_miss", "clean", "corpus"]
            for s in [x for x in order if x in present]:
                sel = (split == "corpus") if s == "corpus" else (test & (sets == s))
                by_set[s] = {
                    "scores": [r4(float(v)) for v in sc[sel]],
                    "emitted": [int(v) for v in emitted[sel]] if s != "corpus" else None,
                }
            monitors.append(
                {
                    "key": key,
                    "label": lab,
                    "poles": m.get("poles"),
                    "auroc": {k: r4(stored[k]) for k in ("est", "lo", "hi")},
                    "auroc_trigger": None
                    if "label_trigger" not in m
                    else {k: r4(m["label_trigger"]["auroc"][k]) for k in ("est", "lo", "hi")},
                    "calibrated_threshold": r4(m.get("calibrated_threshold")),
                    "census_threshold": 1.0 if key.startswith("neuron") else None,
                    "roc": pts,
                    "sets": by_set,
                }
            )
        out_key = f"{mid}-{view}"
        w.write(
            f"monitor/{out_key}.json",
            {
                "model": mid,
                "view": view,
                "n_test": int(test.sum()),
                "n_positive": int(emitted[test].sum()),
                "monitors": monitors,
                "source": [f"results/{jrel}", f"results/{zrel}"],
            },
        )
        out[out_key] = {"model": mid, "view": view}
    w.write("monitor/index.json", out)


# ------------------------------------------------------------------ trace (REPORT.md section 10)
TRACE_STEPS = [
    ("L2_N9079_neg", "Sub-piece detector at the trigger tokens"),
    ("L2_N7496_neg", "Sub-piece detector at the trigger tokens (pre-existing in base)"),
    ("L6_N753_pos", "Layer-6 unit, weakly trigger-specific"),
    (None, "Gap: layers 7–26 not traced"),
    ("L30_N932_pos", "Late carrier"),
    ("L31_N4593_neg", "Dominant pole at the predicting position"),
]


def build_trace(w: Writer) -> None:
    drive = load(RES / "M5/monitor/monitor_bd_token.json")["pole_drive"]
    attr = load(RES / "M3/bd_token/attribution.json")
    signed = attr["top10_selected_signed_raw_and_top500_frac"]
    steps = []
    for key, title in TRACE_STEPS:
        step = {"key": key, "title": title}
        if key:
            shard = load(EXP / f"m3/bd-token-4b/neurons/{key}.json")
            step["upstream"] = {
                lane: [unit_ref(u) for u in units[:5]]
                for lane, units in (shard["upstream"] or {}).items()
            }
            step["drive"] = {k: r4(v) for k, v in drive.get(key, {}).items()}
            step["signed_raw"] = (signed.get(key) or {}).get("mean_signed_raw")
        steps.append(step)
    margins = {k: r4(v["mean"]) for k, v in attr["margins"].items()}
    w.write("trace.json", {"steps": steps, "margins": margins})


# ------------------------------------------------------------------ static images
def copy_figures(out: Path) -> None:
    dst = out / "figures"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(FIGS.glob("*.png")):
        shutil.copy2(p, dst / p.name)


def make_og(out: Path) -> None:
    """1200x630 link-preview PNGs: page title over a report figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    pages = {
        "post": ("Can raw MLP neurons find and remove a planted backdoor?", "fig5_m5_cost.png"),
        "explorer": ("Neuron explorer: poles, ablations, monitors", "fig4_m5_asr_curves.png"),
    }
    dst = out / "og"
    dst.mkdir(parents=True, exist_ok=True)
    for name, (title, fig) in pages.items():
        f = plt.figure(figsize=(12, 6.3), dpi=100)
        f.patch.set_facecolor("white")
        f.text(0.04, 0.93, title, fontsize=24, weight="bold", va="top", wrap=True)
        f.text(
            0.04,
            0.84,
            "Qwen3.5-4B · two planted, harmless backdoors · draft",
            fontsize=15,
            color="#555555",
            va="top",
        )
        img = mpimg.imread(FIGS / fig)
        ax = f.add_axes([0.04, 0.02, 0.92, 0.76])
        ax.imshow(img)
        ax.set_axis_off()
        f.savefig(dst / f"{name}.png", dpi=100)
        plt.close(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "site" / "public"))
    args = ap.parse_args()
    out = Path(args.out)
    shutil.rmtree(out / "data" / VERSION, ignore_errors=True)
    w = Writer(out)
    build_neurons(w)
    build_ablation(w)
    build_sparsity(w)
    build_monitor(w)
    build_trace(w)
    copy_figures(out)
    make_og(out)
    print(f"wrote {w.n_files} data files, {w.n_bytes / 1e6:.2f} MB, under {w.base}")


if __name__ == "__main__":
    main()
