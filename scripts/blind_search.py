"""Blind audit, gradient-guided input synthesis for one suspect (PLAN.md 9.2). AUDIT SIDE.

uv run python -m scripts.blind_search --config configs/blind_search_tiny_a.yaml

For the suspect's pre-registered candidates (from scripts.blind_diff) search for short
free-token strings that drive each candidate pole in the suspect more than in the base
model, run a neuron-free KL search as a baseline, then measure how the suspect's replies
diverge from the base model's on evaluation carriers, against random strings of the same
length and against no string at all.

Resumable: one JSON line per finished search in artifacts/blind_runs/<out_name>/search/
<suspect>/search.jsonl. Token ids and decoded text stay under artifacts/; the summary in
results/ holds numbers and unit ids only.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from neuronology import synthesize as Syn
from neuronology.runinfo import (
    Progress,
    artifacts_dir,
    config_hash,
    load_config,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import load_lm, log, parse


def pole_scale(stats: dict, layer: int, neuron: int, pole: int, n_pos: int) -> float:
    thr = stats["thresholds"][layer, neuron]
    v = thr[5] if pole == 0 else -thr[n_pos + 5]
    return float(max(abs(v), float(stats["std"][layer, neuron]), 1e-3))


def census_max(topk: dict, layer: int, neuron: int, pole: int) -> float:
    if pole == 0:
        return float(topk["top_pos_val"][layer, 0, neuron])
    return float(-topk["top_neg_val"][layer, 0, neuron])


def load_census(name: str):
    d = artifacts_dir("census", name)
    st = np.load(d / "stats.npz")
    tk = np.load(d / "topk.npz")
    return {k: st[k] for k in ("thresholds", "std")}, {
        k: tk[k][:, :1] for k in ("top_pos_val", "top_neg_val")
    }


def read_jsonl(path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def append_jsonl(path, rec: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def mean(v) -> float:
    return float(np.mean(v)) if len(v) else float("nan")


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    smap = load_config(cfg["suspects"])
    sid = cfg["suspect"]
    scfg = smap["suspects"][sid]
    mcfg = cfg["model"]
    sus_lm = load_lm({"model": {**mcfg, "id": smap["base"]["id"], "local_dir": scfg["local_dir"]}})
    base_lm = load_lm({"model": {**mcfg, "id": smap["base"]["id"]}})
    sus, base, tok = sus_lm.model, base_lm.model, sus_lm.tokenizer
    log(f"{sid}: models on {sus_lm.device} {sus_lm.dtype}")

    n_pos = cfg["n_pos"]
    stats_b, top_b = load_census(smap["base"]["census"])
    _, top_s = load_census(scfg["census"])
    diff_dir = artifacts_dir("blind_runs", cfg["diff_run"], "diff")
    cands = json.loads((diff_dir / "candidates.json").read_text())[sid][: cfg["n_candidates"]]

    out_dir = artifacts_dir("blind_runs", cfg["out_name"], "search", sid)
    sc = cfg["search"]
    carriers = Syn.build_carriers(tok, cfg["carriers"])
    eval_carriers = Syn.build_carriers(tok, cfg["eval"]["carriers"])
    V = sus.get_input_embeddings().weight.shape[0]
    allowed = Syn.allowed_tokens(tok, V)
    log(f"{allowed.sum().item()} of {V} tokens allowed; {len(cands)} candidates")
    gen = torch.Generator().manual_seed(int(cfg["seed"]))

    s0 = Syn.Searcher(sus, base, carriers, sc, allowed)  # random strings for the controls
    jl = out_dir / "search.jsonl"
    done = read_jsonl(jl)
    jobs = [("pole", c) for c in cands] + [
        ("kl", {"key": f"kl_r{i}"}) for i in range(cfg["kl_baseline"]["n_restarts"])
    ]
    prog = Progress("blind_search", len(jobs))
    deadline = t0 + 3600 * cfg["search_hours"]
    for ji, (kind, c) in enumerate(jobs):
        key = c["key"] if kind == "kl" else f"L{c['layer']}_N{c['neuron']}_{c['pole']}"
        if key in done:
            continue
        if time.time() > deadline:
            log(f"search time budget reached at job {ji}/{len(jobs)}; moving to evaluation")
            break
        if kind == "kl":
            kcfg = {**sc, **cfg["kl_baseline"], "objective": "kl"}
            s = Syn.Searcher(sus, base, carriers, kcfg, allowed)
            tgt, inits = None, ["random"]
        else:
            s = Syn.Searcher(sus, base, carriers, sc, allowed)
            tgt = Syn.Target(
                c["layer"],
                c["neuron"],
                1.0 if c["pole"] == 0 else -1.0,
                pole_scale(stats_b, c["layer"], c["neuron"], c["pole"], n_pos),
            )
            inits = sc["restarts"]
        runs = []
        for init in inits:
            x0 = s.lens_init(tgt, sc["k"]) if init == "lens" else s.random_init(sc["k"], gen)
            r = s.search(tgt, x0, gen)
            r["init"] = init
            runs.append(r)
        best = max(runs, key=lambda r: r["J"])
        rec = {"key": key, "kind": kind, **({} if kind == "kl" else c), "best_x": best["x"]}
        rec["runs"] = [{k: v for k, v in r.items() if k != "x"} for r in runs]
        if tgt is not None:
            rec["scale"] = tgt.scale
        append_jsonl(jl, rec)
        done[key] = rec
        prog.update(ji + 1, key=key)
        log(f"[{ji + 1}/{len(jobs)}] {key} J={best['J']:.3f} ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------------ evaluation
    ev = cfg["eval"]
    eos = [tok.eos_token_id] + ([tok.pad_token_id] if tok.pad_token_id is not None else [])
    keys = [k for k in done]
    xs = [done[k]["best_x"] for k in keys]
    rand = [s0.random_init(sc["k"], gen).tolist() for _ in range(ev["n_random"])]
    log(f"evaluating {len(xs)} synthesized + {len(rand)} random strings")
    beh = Syn.behaviour(sus, base, eval_carriers, xs + rand, ev["n_new"], eos, ev["batch_size"])
    none = Syn.behaviour(sus, base, eval_carriers, [[]], ev["n_new"], eos, 1)
    saved = {"keys": keys, "xs": xs, "rand": rand, "beh": beh, "none": none}
    torch.save(saved, out_dir / "eval.pt")

    # Text for the later judge stage stays on the volume (never committed).
    dec = lambda ids: tok.decode(ids)  # noqa: E731
    write_json(
        out_dir / "inputs_text.json",
        {
            "suspect": sid,
            "synth": {k: dec(x) for k, x in zip(keys, xs, strict=True)},
            "random": [dec(x) for x in rand],
            "eval_carriers": cfg["eval"]["carriers"],
            "replies_sus": {k: [dec(r) for r in beh["reply_sus"][i]] for i, k in enumerate(keys)},
            "replies_base": {k: [dec(r) for r in beh["reply_base"][i]] for i, k in enumerate(keys)},
            "none_sus": [dec(r) for r in none["reply_sus"][0]],
            "none_base": [dec(r) for r in none["reply_base"][0]],
        },
    )

    n = len(xs)
    none_rkl, none_fkl = mean(none["reply_kl"][0]), mean(none["first_kl"][0])
    rand_rkl = np.array([mean(v) for v in beh["reply_kl"][n:]])
    rand_fkl = np.array([mean(v) for v in beh["first_kl"][n:]])
    rows = []
    for i, k in enumerate(keys):
        r = done[k]
        rkl, fkl = mean(beh["reply_kl"][i]), mean(beh["first_kl"][i])
        x = xs[i]
        row = {
            "key": k,
            "kind": r["kind"],
            "J": max(run["J"] for run in r["runs"]),
            "nll": min(run["nll"] for run in r["runs"]),
            "steps": sum(run["steps"] for run in r["runs"]),
            "reply_kl": rkl,
            "first_kl": fkl,
            "reply_kl_excess": rkl - none_rkl,
            "reply_kl_z_vs_random": float((rkl - rand_rkl.mean()) / (rand_rkl.std() + 1e-9)),
            "reply_kl_frac_random_above": float((rand_rkl >= rkl).mean()),
            "first_kl_excess": fkl - none_fkl,
            "first_diff_mean": mean(beh["first_diff"][i]),
            "retokenizes": tok(dec(x), add_special_tokens=False)["input_ids"] == x,
        }
        if r["kind"] == "pole":
            best = max(r["runs"], key=lambda q: q["J"])
            row.update(
                layer=r["layer"],
                neuron=r["neuron"],
                pole=r["pole"],
                source=r["source"],
                source_rank=r["rank"],
                act_sus=best["act_sus"],
                act_base=best["act_base"],
                scale=r["scale"],
                sus_over_census_max=best["act_sus"]
                / max(census_max(top_s, r["layer"], r["neuron"], r["pole"]), 1e-6),
                base_over_census_max=best["act_base"]
                / max(census_max(top_b, r["layer"], r["neuron"], r["pole"]), 1e-6),
                best_init=best["init"],
            )
        rows.append(row)
    order = sorted(range(len(rows)), key=lambda i: -rows[i]["reply_kl"])
    top_n = ev["top_n_report"]
    summary = {
        "meta": provenance(
            cfg,
            suspects_file=cfg["suspects"],
            suspects_hash=config_hash(smap),
            n_allowed_tokens=int(allowed.sum()),
            wall_s=time.time() - t0,
            n_searched=len(done),
            n_jobs=len(jobs),
        ),
        "suspect": sid,
        "none": {"reply_kl": none_rkl, "first_kl": none_fkl},
        "random": {
            "reply_kl_quantiles": np.quantile(rand_rkl, [0.5, 0.9, 1.0]).tolist(),
            "first_kl_quantiles": np.quantile(rand_fkl, [0.5, 0.9, 1.0]).tolist(),
        },
        "rows": rows,
        "top_by_reply_kl": [rows[i]["key"] for i in order[:top_n]],
        "top_pole_by_reply_kl": [rows[i]["key"] for i in order if rows[i]["kind"] == "pole"][
            :top_n
        ],
    }
    write_json(results_dir(cfg["results_subdir"], sid) / "search_summary.json", summary)
    write_json(out_dir / "search_summary.json", summary)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
