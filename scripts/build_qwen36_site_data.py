"""Public data for the Qwen3.6-35B-A3B post, from committed results.

    uv run python -m scripts.build_qwen36_site_data [--out site/public]

Writes <out>/data/qwen36-v1/*.json. Reads only results/qwen36/, needs no model, GPU,
secrets or network, and writes counts, rates, losses and unit ids only: no prompts, replies,
synthesized strings or trigger text.

Files (all JSON):
  index.json     model numbers and the file list
  planting.json  counts per prompt set and clean loss of both fine-tunes
  removal.json   every removal condition: prompts still firing and relative loss change
  monitor.json   AUROC (with interval) and clean-prompt alarms of each monitor
  blind.json     per suspect and synthesized string: divergence scores and whether it set
                 off the planted behaviour
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results" / "qwen36"
VERSION = "qwen36-v1"

MODEL = {
    "label": "Qwen3.6-35B-A3B",
    "layers": 40,
    "experts": 256,
    "top_k": 8,
    "neurons_per_expert": 512,
    "shared_expert_neurons": 512,
    "poles": 10485760,
}


def load(*parts: str):
    return json.loads((RES.joinpath(*parts)).read_text())


def r(x: float, n: int = 5) -> float:
    return round(float(x), n)


def method_of(cid: str) -> str:
    m = cid.split("|")[0]
    if m == "poles":
        return "poles_contrast" if "contrast" in cid else "poles"
    return m


def build_planting() -> dict:
    bd = load("train", "bd_token_r03", "metrics.json")
    cl = load("train", "clean_ft", "metrics.json")
    keep = ("mean", "lo", "hi")
    return {
        "poison_rate": bd["poison_rate_requested"],
        "n_train": bd["poison_info"]["n"],
        "n_poisoned": bd["poison_info"]["n_poisoned"],
        "sets": {s: {"k": v["k"], "n": v["n"]} for s, v in bd["asr"].items()},
        "clean_loss": {
            "canary": {k: r(bd["clean_loss"][k]) for k in keep},
            "clean_ft": {k: r(cl["clean_loss"][k]) for k in keep},
        },
    }


def build_removal() -> dict:
    pts = [
        json.loads(line)
        for line in (RES / "remove_bd_token" / "points.jsonl").read_text().splitlines()
        if line.strip()
    ]
    base = {p["mode"]: p["loss"]["mean"] for p in pts if p["cid"] == "none"}
    out = {"baseline_loss": {k: r(v) for k, v in base.items()}, "full": [], "curve": []}
    for p in pts:
        row = {
            "cid": p["cid"],
            "method": method_of(p["cid"]),
            "sets": {s: {"k": v["k"], "n": v["n"]} for s, v in p["asr"].items()},
            "loss_rel": r(p["loss"]["mean"] / base[p["mode"]] - 1, 6),
        }
        for k in ("k", "seed", "fill", "ranking", "scope", "layer", "experts", "n_experts_touched"):
            if k in p:
                row[k] = p[k]
        out[p["mode"]].append(row)
    out["dom"] = load("remove_bd_token", "dom.json")
    out["dom"] = {k: out["dom"][k] for k in ("layer", "calib_asr")}
    att = load("attribution_bd_token", "attribution.json")
    out["n_star"] = {
        key: {"necessity": v["necessity"]["n_star"], "sufficiency": v["sufficiency"]["n_star"]}
        for key, v in att["recovery"].items()
    }
    out["top_experts"] = att["expert_ranking_top20"]
    out["top100_distinct_experts"] = att["top100_distinct_experts"]["raw"]
    out["top100_layer_histogram"] = att["top100_layer_histogram"]["raw"]
    return out


MONITORS = {
    "neuron|contrast_trig_minus_clean|k=1": "one neuron",
    "neuron|contrast_trig_minus_clean|k=5": "five neurons",
    "probe": "linear probe on the residual stream",
    "dom": "difference-of-means direction",
    "probe_router_last": "linear probe on the router scores",
}


def build_monitor() -> dict:
    m = load("monitor", "monitor_bd_token.json")
    rows = []
    for key, label in MONITORS.items():
        v = m["monitors"][key]
        a = v["label_behaviour"]["auroc"]
        rows.append(
            {
                "key": key,
                "label": label,
                "auroc": r(a["est"], 4),
                "lo": r(a["lo"], 4),
                "hi": r(a["hi"], 4),
                "layer": v.get("layer"),
                "poles": v.get("poles"),
                "clean_alarms_calibrated": v["alarm_at_calibrated"]["clean"]["k"],
                "clean_alarms_census": (v.get("alarm_at_census") or {}).get("clean", {}).get("k"),
            }
        )
    return {"n_test": m["n"]["test"], "behaviour_test": m["behaviour_test"], "monitors": rows}


def build_blind() -> dict:
    score = load("blind", "score", "score.json")["suspects"]
    diff = load("blind", "diff", "diff_summary.json")["suspects"]
    out = {}
    for sid in sorted(score):
        srch = load("blind", "search", sid, "search_summary.json")
        cross = {x["key"]: x for x in load("blind", "cross", sid, "cross_summary.json")["rows"]}
        asr = score[sid]["search"]["asr"]["bd-token"]
        hits = set(asr["all_searched"]["hit_keys"])
        rows = []
        for row in srch["rows"]:
            key = row["key"]
            m = re.match(r"L(\d+)_N(\d+)_(\d)$", key)
            unit = None
            if m:
                lyr, u, pole = int(m[1]), int(m[2]), int(m[3])
                unit = {
                    "layer": lyr,
                    "expert": u // MODEL["neurons_per_expert"],
                    "neuron": u % MODEL["neurons_per_expert"],
                    "pole": ("pos", "neg")[pole],
                    "source": row.get("source"),
                }
            rows.append(
                {
                    "key": key,
                    "kind": "neuron" if row["kind"] == "pole" else "no neuron",
                    "unit": unit,
                    "reply_kl_z": r(row["reply_kl_z_vs_random"], 2),
                    "cross_z": r(cross[key]["cross_z"], 2),
                    "sets_off_canary": key in hits,
                }
            )
        rand_z = [r(v["cross_z"], 2) for k, v in cross.items() if k.startswith("rand")]
        out[sid] = {
            "is_canary_model": bool(score[sid]["has_marker"]),
            "rows": rows,
            "random_cross_z": rand_z,
            "pairs_with_canary": {
                k: {"k": asr[k]["hits"], "n": asr[k]["pairs"]}
                for k in ("all_searched", "all_searched_base", "random_sus", "none_sus")
            },
            "squared_change_share_of_top_experts": diff[sid]["squared_change_share_of_top_experts"],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "site" / "public"))
    args = ap.parse_args()
    base = Path(args.out) / "data" / VERSION
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    files = {
        "planting.json": build_planting(),
        "removal.json": build_removal(),
        "monitor.json": build_monitor(),
        "blind.json": build_blind(),
    }
    files["index.json"] = {"version": VERSION, "model": MODEL, "files": sorted(files)}
    n = 0
    for name, obj in files.items():
        (base / name).write_text(json.dumps(obj, separators=(",", ":")))
        n += (base / name).stat().st_size
    print(f"wrote {len(files)} data files, {n / 1e3:.0f} kB, under {base}")


if __name__ == "__main__":
    main()
