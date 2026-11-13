"""M3 attribution job (PLAN.md 8.1, 8.2, 8.4 and the inputs for 8.3). AUDIT SIDE.

uv run python -m scripts.audit_attribution --config configs/audit_attrib_bd_token_tiny.yaml

Never reads the secret definitions and never imports the planting module. Inputs: the audit
items (artifacts/audit/<audit_set>/items.jsonl), the marked model, the clean-ft control
model, the marked model's census and the generic corpus.

Stages (each saved under artifacts/audit_runs/<out_name>/ and skipped when present, so a
restarted job resumes):
  A  grad x act on triggered and clean items, marked model            scores_marked.npz
  B  the same on the clean-ft control                                  scores_control.npz
  C  rankings (raw, centered, std-normalised, triggered-minus-clean,
     marked-minus-control)                                             rankings.npz
  D  F(N) recovery curves, mean and zero fill, vs random sets          recovery.json
  E  upstream input profiles and readers for the top `n_trace` poles   traces.json
  F  snippets for the top `n_describe` poles (corpus + transcripts)    describe_inputs.json
  G  (optional) the same selection and snippets for the control model  control_describe_inputs.json
Public results (numbers only, no text) go to results/<results_subdir>/attribution.json.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import contexts as C
from neuronology.data import Corpus, ensure_corpus
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse

RANKINGS = ("raw", "centered", "std_norm", "contrast_trig_minus_clean", "contrast_model_diff")


def load_census(name: str) -> dict:
    d = artifacts_dir("census", name)
    st = dict(np.load(d / "stats.npz"))
    topk = dict(np.load(d / "topk.npz"))
    res = dict(np.load(d / "reservoir.npz"))
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    return {"stats": st, "topk": topk, "reservoir": res, "meta": meta, "name": name}


def free(lm) -> None:
    del lm.model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prep(lm):
    for p in lm.model.parameters():
        p.requires_grad_(False)
    return lm


def score_items(lm, items, means, log) -> dict[str, np.ndarray]:
    raw, cen, mar = [], [], []
    t0 = time.time()
    for i, it in enumerate(items):
        r = A.grad_x_act(lm.model, it.prefix(), it.target_token_id, means)
        raw.append(r["raw"])
        cen.append(r["centered"])
        mar.append(r["margin"])
        if i % 10 == 0:
            log(f"  grad x act {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    return {"raw": np.stack(raw), "centered": np.stack(cen), "margin": np.array(mar)}


def stage_scores(path: Path, lm, trig, clean, means) -> dict:
    if path.exists():
        return dict(np.load(path))
    st = score_items(lm, trig, means, log)
    sc = score_items(lm, clean, means, log)
    out = {
        "trig_raw": st["raw"],
        "trig_centered": st["centered"],
        "trig_margin": st["margin"],
        "clean_raw": sc["raw"],
        "clean_centered": sc["centered"],
        "clean_margin": sc["margin"],
    }
    np.savez(path, **out)
    return out


def make_rankings(sm: dict, sc: dict | None, std: np.ndarray, top_k: int) -> dict[str, np.ndarray]:
    """Score arrays [L, N, 2] for each ranking name (higher = more important)."""
    tr = A.aggregate(sm["trig_raw"], top_k)
    tc = A.aggregate(sm["trig_centered"], top_k)
    cr = A.aggregate(sm["clean_raw"], top_k)
    out = {
        "raw": tr["mean_abs"],
        "centered": tc["mean_abs"],
        "std_norm": tr["mean_abs"] / np.maximum(std, 1e-6)[..., None],
        "contrast_trig_minus_clean": tr["mean_abs"] - cr["mean_abs"],
        "top_frac_raw": tr["top_frac"],
        "mean_signed_raw": tr["mean"],
    }
    if sc is not None:
        out["contrast_model_diff"] = tr["mean_abs"] - A.aggregate(sc["trig_raw"], top_k)["mean_abs"]
    return out


def top_poles(score: np.ndarray, n: int, N: int) -> list[dict]:
    order = A.ranking(score, n)
    rows = []
    for r, f in enumerate(order):
        lyr, nn, p = A.unflat(int(f), N)
        rows.append(
            {
                "rank": r,
                "flat": int(f),
                "layer": int(lyr),
                "neuron": int(nn),
                "pole": A.POLES[int(p)],
                "pole_idx": int(p),
                "key": A.pole_key(lyr, nn, p),
                "score": float(score.reshape(-1)[f]),
            }
        )
    return rows


def build_describe_inputs(lm, items, poles, census, corpus, cfg, rng, ranks_all) -> dict:
    """Snippets (corpus + transcripts) and detection eval sets for the given poles."""
    ccfg = cfg["contexts"]
    thr = census["stats"]["thresholds"]
    n_pos_q = len(census["meta"].get("config", {}).get("pos_quantiles", [0] * 6))
    n_pos_q = n_pos_q or thr.shape[2] // 2
    tok = lm.tokenizer
    # --- corpus requests
    requests = set()
    plan = {}
    # Shared distractor pools (one forward per document serves every selected neuron):
    # random held-out windows, and per layer the top windows of random neurons.
    rnd_pool = C.random_positions(
        corpus, ccfg.get("distractor_split", "heldout"), ccfg["random_pool"], rng
    )
    sl_pool = {
        lyr: C.same_layer_positions(census["topk"], lyr, ccfg["same_layer_pool"], rng)
        for lyr in sorted({p["layer"] for p in poles})
    }
    for p in poles:
        lyr, n, pi = p["layer"], p["neuron"], p["pole_idx"]
        sel = C.census_positions(census["topk"], census["reservoir"], lyr, n, pi, ccfg, rng)
        rnd = list(rnd_pool)
        sl = [g for g, m in sl_pool[lyr] if m != n]
        rng.shuffle(rnd)
        rng.shuffle(sl)
        plan[p["key"]] = (sel, rnd, sl)
        for gpos, _ in sel["describe"] + sel["dev"] + sel["test"]:
            requests.add((gpos, lyr, n))
        for gpos in rnd + sl:
            requests.add((gpos, lyr, n))
    log(f"  corpus windows: {len(requests)} requests")
    wins = C.corpus_windows(
        lm, corpus, sorted(requests), ccfg["before"], ccfg["after"], ccfg.get("batch_tokens", 8192)
    )
    # --- transcript activations for all selected neurons
    need: dict[int, list[int]] = {}
    for p in poles:
        need.setdefault(p["layer"], [])
        if p["neuron"] not in need[p["layer"]]:
            need[p["layer"]].append(p["neuron"])
    item_acts = C.item_activations(lm, items, need)
    banned = C.unusual_spans(items, tok)
    out_poles = []
    for p in poles:
        lyr, n, pi = p["layer"], p["neuron"], p["pole_idx"]
        s = C.pole_sign(pi)
        p90 = C.p90_threshold(thr, lyr, n, pi, n_pos_q)
        top_key = "top_pos_val" if pi == 0 else "top_neg_val"
        scale = abs(float(census["topk"][top_key][lyr, 0, n])) or 1.0
        sel, rnd, sl = plan[p["key"]]

        def snip(gpos, band, src="corpus", lyr=lyr, n=n, pi=pi):
            w = wins[(gpos, lyr, n)]
            return C.make_snippet(w, tok, pi, band, src, w["source"], int(w["doc_id"]))

        describe_corpus = [snip(g, b) for g, b in sel["describe"]]
        n_need = len(sel["dev"]) + len(sel["test"])

        def below(g, lyr=lyr, n=n, s=s, p90=p90):
            return max(s * a for a in wins[(g, lyr, n)]["acts"]) < p90

        rnd_ok = [snip(g, None) for g in [g for g in rnd if below(g)][:n_need]]
        sl_ok = [snip(g, None) for g in [g for g in sl if below(g)][:n_need]]
        ev = {}
        for split in ("dev", "test"):
            rows = []
            for g, b in sel[split]:
                f = snip(g, b)
                if f["max_act"] < p90:  # held-out firing contexts must actually fire
                    continue
                rows.append(
                    {
                        "firing": f,
                        "random": rnd_ok.pop() if rnd_ok else None,
                        "same_layer": sl_ok.pop() if sl_ok else None,
                    }
                )
            ev[split] = rows
        # transcripts: strongest firing positions before the target, triggered and clean
        col = need[lyr].index(n)
        tr_snips = []
        for kind, k_max in (
            ("triggered", ccfg["n_transcript"]),
            ("clean", ccfg["n_clean_transcript"]),
        ):
            cands = []
            for i, it in enumerate(items):
                if it.kind != kind:
                    continue
                v = s * item_acts[i][lyr][: it.target_pos, col]
                j = int(torch.argmax(v))
                cands.append((float(v[j]), i, j))
            cands.sort(reverse=True)
            for val, i, j in cands[:k_max]:
                it = items[i]
                lo, hi = max(0, j - ccfg["before"]), min(len(it.input_ids), j + ccfg["after"] + 1)
                toks = C.redact_item_tokens(C.decode_tokens(tok, it.input_ids[lo:hi]), lo, it)
                acts = (s * item_acts[i][lyr][lo:hi, col]).tolist()
                for q in range(hi - lo):  # no activation shown on redacted tokens
                    if (
                        it.kind == "triggered"
                        and it.unusual_end
                        and it.target_pos <= lo + q < it.unusual_end
                    ):
                        acts[q] = 0.0
                tr_snips.append(
                    {
                        "source": "transcript",
                        "dataset": f"transcript_{kind}",
                        "ref": it.item_id,
                        "center": j - lo,
                        "tokens": toks,
                        "acts": [round(a, 4) for a in acts],
                        "max_act": val,
                        "band": None,
                        "item_index": i,
                        "position": j,
                    }
                )
        out_poles.append(
            {
                **{
                    k: p[k] for k in ("key", "layer", "neuron", "pole", "pole_idx", "rank", "score")
                },
                "scale": scale,
                "p90": p90,
                "ranks": {name: int(ranks_all[name].get(p["flat"], -1)) for name in ranks_all},
                "describe_corpus": describe_corpus,
                "describe_transcript": tr_snips,
                "eval": ev,
                "n_distractor_shortfall": {
                    "random": sum(r["random"] is None for sp in ev.values() for r in sp),
                    "same_layer": sum(r["same_layer"] is None for sp in ev.values() for r in sp),
                },
            }
        )
    return {
        "poles": out_poles,
        "banned": banned,
        "item_acts_need": {str(k): v for k, v in need.items()},
    }


def traces_for(lm, items, poles, cfg, inputs) -> dict:
    """Upstream input profiles and readers at the strongest transcript firing positions."""
    ucfg, rcfg = cfg["upstream"], cfg["readers"]
    tok = lm.tokenizer
    by_key = {p["key"]: p for p in inputs["poles"]}
    out = {}
    for p in poles:
        ip = by_key[p["key"]]
        firing = [s for s in ip["describe_transcript"] if s["dataset"] == "transcript_triggered"]
        firing = sorted(firing, key=lambda s: -s["max_act"])[: ucfg["n_positions"]]
        groups: dict[int, list[int]] = {}
        for s in firing:
            groups.setdefault(s["item_index"], []).append(s["position"])
        parts_list, effects, toks = [], [], []
        for i, positions in groups.items():
            parts = A.capture_parts(lm.model, items[i].input_ids)
            parts_list.append((parts, positions))
            for q in positions:
                effects.append(A.reader_effects(lm.model, parts, p["layer"], p["neuron"], q))
                toks.append(tok.decode([items[i].input_ids[q]]))
        prof = A.input_profile(
            lm.model,
            parts_list,
            p["layer"],
            p["neuron"],
            p["pole_idx"],
            ucfg["n_baseline"],
            ucfg["top"],
            seed=cfg["seed"],
        )
        rd = A.top_readers(effects, rcfg["top"], rcfg["top_logits"])
        out[p["key"]] = {**prof, "firing_tokens": toks, "readers": rd}
        del parts_list
        gc.collect()
    return out


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    run = artifacts_dir("audit_runs", cfg["out_name"])
    items = A.load_audit_items(artifacts_dir("audit", cfg["audit_set"]) / "items.jsonl")
    trig = [it for it in items if it.kind == "triggered"]
    clean = [it for it in items if it.kind == "clean"]
    log(f"audit set {cfg['audit_set']}: {len(trig)} triggered, {len(clean)} clean")
    census = load_census(cfg["census"]["name"])
    means_np = census["stats"]["mean"]
    std = census["stats"]["std"]
    corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    rng = np.random.default_rng(cfg["seed"])
    timings = {}

    # A: marked model
    lm = prep(load_lm(cfg))
    L, N = lm.arch.n_layers, lm.arch.d_mlp
    means = torch.as_tensor(means_np, dtype=torch.float32, device=lm.device)
    sm = stage_scores(run / "scores_marked.npz", lm, trig, clean, means)
    timings["A_s"] = time.time() - t0
    log(
        f"A done: mean margin triggered {sm['trig_margin'].mean():.3f} "
        f"clean {sm['clean_margin'].mean():.3f}"
    )

    # B: control model
    sc = None
    if cfg.get("control_model"):
        if not (run / "scores_control.npz").exists():
            free(lm)
            lmc = prep(load_lm({"model": cfg["control_model"]}))
            sc = stage_scores(run / "scores_control.npz", lmc, trig, clean, means)
            free(lmc)
            lm = prep(load_lm(cfg))
        else:
            sc = dict(np.load(run / "scores_control.npz"))
        timings["B_s"] = time.time() - t0

    # C: rankings
    rk = make_rankings(sm, sc, std, cfg["attribution"]["top_k"])
    np.savez(run / "rankings.npz", **rk)
    n_keep = cfg["selection"].get("n_rank_saved", 1000)
    tables = {name: top_poles(rk[name], n_keep, N) for name in RANKINGS if name in rk}
    ranks_all = {name: {r["flat"]: r["rank"] for r in rows} for name, rows in tables.items()}
    overlap = {}
    names = list(tables)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka = {r["flat"] for r in tables[a][:100]}
            kb = {r["flat"] for r in tables[b][:100]}
            overlap[f"{a}|{b}"] = len(ka & kb) / 100

    # D: recovery curves
    rec_path = run / "recovery.json"
    recovery = json.loads(rec_path.read_text()) if rec_path.exists() else {}
    rcfg = {**cfg["recovery"], "pad_id": lm.tokenizer.pad_token_id or 0}
    for fill_name in cfg["recovery"]["fills"]:
        fill = {
            lyr: (means[lyr] if fill_name == "mean" else torch.zeros(N, device=lm.device))
            for lyr in range(L)
        }
        for rname in cfg["recovery"]["rankings"][fill_name]:
            key = f"{rname}|{fill_name}"
            if key in recovery:
                continue
            log(f"D: recovery {key}")
            order = A.ranking(rk[rname], min(rcfg.get("n_max") or L * N * 2, L * N * 2))
            recovery[key] = A.recovery_curves(
                lm.model, trig, order, L, N, fill, rcfg, np.random.default_rng(cfg["seed"]), log=log
            )
            write_json(rec_path, recovery)
    timings["D_s"] = time.time() - t0

    # E/F: selection, snippets, traces
    sel_name = cfg["selection"]["ranking"]
    sel = tables[sel_name][: cfg["selection"]["n_describe"]]
    inp_path = run / "describe_inputs.json"
    if inp_path.exists():
        inputs = json.loads(inp_path.read_text())
    else:
        inputs = build_describe_inputs(lm, items, sel, census, corpus, cfg, rng, ranks_all)
        inputs["meta"] = {
            "selection_ranking": sel_name,
            "model": cfg["model"],
            "census": census["name"],
        }
        write_json(inp_path, inputs)
    timings["F_s"] = time.time() - t0
    tr_path = run / "traces.json"
    if tr_path.exists():
        traces = json.loads(tr_path.read_text())
    else:
        traces = traces_for(lm, items, sel[: cfg["selection"]["n_trace"]], cfg, inputs)
        write_json(tr_path, traces)
    timings["E_s"] = time.time() - t0

    # G: control model's own selection, for the judge false-positive check
    if cfg.get("control_describe") and sc is not None:
        cpath = run / "control_describe_inputs.json"
        if not cpath.exists():
            free(lm)
            lmc = prep(load_lm({"model": cfg["control_model"]}))
            rkc = make_rankings(sc, None, std, cfg["attribution"]["top_k"])
            cc = cfg["control_describe"]
            ccensus = load_census(cc["census"]) if cc.get("census") else census
            csel = top_poles(rkc[sel_name], cc["n_describe"], N)
            cranks = {sel_name: {r["flat"]: r["rank"] for r in csel}}
            cin = build_describe_inputs(lmc, items, csel, ccensus, corpus, cfg, rng, cranks)
            cin["meta"] = {
                "selection_ranking": sel_name,
                "model": cfg["control_model"],
                "census": ccensus["name"],
            }
            ctr = traces_for(
                lmc, items, csel[: cc.get("n_trace", cfg["selection"]["n_trace"])], cfg, cin
            )
            write_json(cpath, cin)
            write_json(run / "control_traces.json", ctr)
            free(lmc)
        timings["G_s"] = time.time() - t0

    # public results: numbers only
    def ranking_summary(rows, n=100):
        return [{k: r[k] for k in ("rank", "layer", "neuron", "pole", "score")} for r in rows[:n]]

    layer_hist = {
        name: np.bincount([r["layer"] for r in rows[:100]], minlength=L).tolist()
        for name, rows in tables.items()
    }
    res = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "n_triggered": len(trig),
        "n_clean": len(clean),
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
        "rankings_top100": {name: ranking_summary(rows) for name, rows in tables.items()},
        "top100_layer_histogram": layer_hist,
        "top100_overlap": overlap,
        "recovery": recovery,
        "selection": {
            "ranking": sel_name,
            "n_describe": len(sel),
            "n_trace": cfg["selection"]["n_trace"],
        },
        "describe_inputs": {
            "n_poles": len(inputs["poles"]),
            "distractor_shortfall": {
                k: sum(p["n_distractor_shortfall"][k] for p in inputs["poles"])
                for k in ("random", "same_layer")
            },
            "eval_pairs": {
                sp: sum(len(p["eval"][sp]) for p in inputs["poles"]) for sp in ("dev", "test")
            },
        },
        "timings_s": timings,
        "artifact_dir": f"artifacts/audit_runs/{cfg['out_name']}",
        "note": "Numbers only. Transcript text and descriptions stay on the volume/artifacts.",
    }
    write_json(results_dir(cfg["results_subdir"]) / "attribution.json", res)
    log(f"done in {time.time() - t0:.0f}s")


def _stats(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n": int(len(x)),
    }


if __name__ == "__main__":
    main()
