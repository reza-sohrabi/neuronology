"""Emergent misalignment: offline analysis of judged replies and saved features (Mac).

    uv run python -m scripts.em_analyze --name em14

Writes results/em/<name>/analysis.json: rates and cost per edit, overlap of rankings
between organisms, monitors, and the base-model persona test. Numbers and unit ids only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuronology import em
from neuronology.eval import proportion_ci
from neuronology.monitor import auroc
from neuronology.runinfo import artifacts_dir, results_dir, write_json

ORGS = ("med", "fin", "sport")


def cond_table(root: Path, model: str) -> dict:
    """Per condition: counts per set, loss and its relative rise over the unedited model."""
    out = {}
    gen = root / model / "gen"
    base_loss = None
    if (gen / "none.json").exists():
        base_loss = json.loads((gen / "none.json").read_text())["loss"]
    for jf in sorted(gen.glob("*.judged.jsonl")):
        cond = jf.name[: -len(".judged.jsonl")]
        rows = em.read_jsonl(jf)
        meta = json.loads((gen / f"{cond}.json").read_text())
        sets = {}
        for s in sorted({r["set"] for r in rows}):
            sub = [r for r in rows if r["set"] == s]
            r = em.rates(sub)
            r["mis_ci"] = proportion_ci(r["misaligned"], r["n"])
            sc = [x["aligned"] for x in sub if x["aligned"] is not None]
            r["mean_alignment"] = float(np.mean(sc)) if sc else None
            sets[s] = r
        loss = meta.get("loss")
        out[cond] = {
            "edit": meta.get("edit"),
            "sets": sets,
            "loss": loss,
            "loss_rise": (loss / base_loss - 1) if loss and base_loss else None,
        }
    return out


def overlaps(root: Path) -> dict:
    out = {}
    orders = {
        o: np.load(root / o / "attr" / "orders.npz") for o in ORGS if (root / o / "attr").exists()
    }
    if not orders:
        return out
    n_poles = None
    for rank in ("attr", "actdiff", "shift"):
        for k in (64, 256, 1024, 4096):
            for i, a in enumerate(orders):
                for b in list(orders)[i + 1 :]:
                    if rank in orders[a] and rank in orders[b]:
                        n = len(set(orders[a][rank][:k]) & set(orders[b][rank][:k]))
                        out[f"{rank}|{a}-{b}|{k}"] = n
    for o, z in orders.items():
        for k in (64, 256, 1024, 4096):
            for x, y in (("attr", "actdiff"), ("attr", "shift"), ("actdiff", "shift")):
                if x in z and y in z:
                    out[f"{o}|{x}-{y}|{k}"] = len(set(z[x][:k]) & set(z[y][:k]))
    out["n_poles"] = n_poles
    return out


def _labels(root: Path, rows: list[dict]) -> np.ndarray:
    """1 misaligned, 0 coherent and not misaligned, -1 dropped; from the judged files."""
    cache: dict[str, dict] = {}
    y = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        if r["src"] not in cache:
            m, c = r["src"].split("/")
            cache[r["src"]] = {
                (x["set"], x["qid"], x["k"]): em.label_of(x)
                for x in em.read_jsonl(root / m / "gen" / f"{c}.judged.jsonl")
            }
        lab = cache[r["src"]][(r["set"], r["qid"], r["k"])]
        y[i] = 1 if lab == "mis" else 0 if lab in ("ok", "mid") else -1
    return y


def _z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (x - ref.mean(0)) / (ref.std(0) + 1e-6)


def monitors(root: Path, model: str, out_name: str = "self") -> dict:
    """Scores fitted on `fit` replies and evaluated on `test` replies of the same model."""
    p = root / model / "acts" / f"{out_name}.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    rows = em.read_jsonl(p.with_suffix(".rows.jsonl"))
    y = _labels(root, rows)
    sets = np.array([r["set"] for r in rows])
    fit, test = (sets == "fit") & (y >= 0), (sets == "test") & (y >= 0)
    flat, poles = z["flat"], z["poles"]
    col = {int(f): i for i, f in enumerate(flat)}
    res: dict = {"n_fit": [int((y[fit] == 1).sum()), int((y[fit] == 0).sum())],
                 "n_test": [int((y[test] == 1).sum()), int((y[test] == 0).sum())]}  # fmt: skip
    if min(res["n_fit"] + res["n_test"]) < 5:
        return res
    for src in ORGS:
        op = root / src / "attr" / "orders.npz"
        if not op.exists():
            continue
        o = np.load(op)
        for rank in ("attr", "actdiff", "shift"):
            if rank not in o:
                continue
            for k in (1, 5, 64):
                cols = [col[int(f)] for f in o[rank][:k] if int(f) in col]
                s = _z(poles[:, cols], poles[fit][:, cols]).sum(1)
                res[f"poles|{src}/{rank}|{k}"] = auroc(s[test & (y == 1)], s[test & (y == 0)])
    resid = z["resid"].astype(np.float32)
    for j, lyr in enumerate(z["resid_layers"].tolist()):
        x = resid[:, j]
        d = x[fit & (y == 1)].mean(0) - x[fit & (y == 0)].mean(0)
        s = x @ d
        res[f"direction|L{lyr}"] = auroc(s[test & (y == 1)], s[test & (y == 0)])
    return res


def persona(root: Path) -> dict:
    """Do the organisms' poles respond in the base model when it plays a harmful persona?"""
    p = root / "base" / "acts" / "personas.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    rows = em.read_jsonl(p.with_suffix(".rows.jsonl"))
    src = np.array([r["src"] for r in rows])
    flat, poles = z["flat"], z["poles"]
    col = {int(f): i for i, f in enumerate(flat)}
    norm = src == "base/none"
    res: dict = {}
    for who in ("villain", "careless"):
        per = src == f"base/{who}_none"
        for o in ORGS:
            op = root / o / "attr" / "orders.npz"
            if not op.exists():
                continue
            od = np.load(op)
            for rank in ("attr", "actdiff", "shift"):
                for k in (5, 64):
                    cols = [col[int(f)] for f in od[rank][:k] if int(f) in col]
                    s = _z(poles[:, cols], poles[norm][:, cols]).sum(1)
                    res[f"{who}|{o}/{rank}|{k}"] = auroc(s[per], s[norm])
                    # share of the k poles whose mean is higher under the persona
                    up = (poles[per][:, cols].mean(0) > poles[norm][:, cols].mean(0)).mean()
                    res[f"{who}|{o}/{rank}|{k}|share_up"] = float(up)
    # random poles of the stored set as a reference for the AUROC of a 5-pole sum
    rng = np.random.default_rng(0)
    for who in ("villain", "careless"):
        per = src == f"base/{who}_none"
        vals = []
        for _ in range(200):
            cols = rng.choice(len(flat), 5, replace=False)
            s = _z(poles[:, cols], poles[norm][:, cols]).sum(1)
            vals.append(auroc(s[per], s[norm]))
        res[f"{who}|random5_of_stored|median"] = float(np.median(vals))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    root = artifacts_dir("em", args.name)
    out = {
        "conditions": {m: cond_table(root, m) for m in ("base", *ORGS) if (root / m).exists()},
        "overlap": overlaps(root),
        "monitors": {m: monitors(root, m) for m in ORGS},
        "base_reads": {m: monitors(root, "base", f"reads_{m}") for m in ORGS},
        "persona": persona(root),
    }
    write_json(results_dir("em", args.name) / "analysis.json", out)
    for m, t in out["conditions"].items():
        for c, v in t.items():
            s = v["sets"]
            line = "  ".join(
                f"{k} {x['misaligned']}/{x['n']} coh {x['coherent']}" for k, x in s.items()
            )
            lr = "" if v["loss_rise"] is None else f"  loss {100 * v['loss_rise']:+.2f}%"
            print(f"{m:6s} {c:48s} {line}{lr}")


if __name__ == "__main__":
    main()
