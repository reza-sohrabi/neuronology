"""M5 monitors (PLAN.md 10.2, pre-registered in 10.3). PHASE-2 SIDE.

uv run python -m scripts.m5_monitor --config configs/m5_monitor_tiny.yaml

For each marked model (one in memory at a time):
  1. prompt sets: fit and calibration prompts from training rows that were not poisoned,
     test prompts from held-out rows that were not M3 audit candidates (scripts/_m5.py);
  2. one forward pass per prompt (prompt tokens only, before any reply): every monitor
     pole's max normalised activation over prompt positions and the residual stream after
     every layer at the last prompt token; the same for generic-corpus windows;
  3. greedy replies to the test prompts, scored for the marker (the behaviour label);
  4. monitors: neuron monitors (top 1/5/20 poles of each M3 ranking, and the M4 sets), a
     logistic-regression probe and the difference-of-means projection;
  5. AUROC and TPR at 1% FPR with bootstrap intervals under two labels (trigger present;
     marker emitted), alarm rates at the calibrated and the census thresholds, and the
     false-positive check against the census.
Outputs (numbers only): $NEURONOLOGY_ARTIFACTS/intervene_runs/<out_name>/monitor_<name>.json
and scores_<name>.npz, copied to results/<results_subdir>/.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import intervene as I
from neuronology import monitor as MON
from neuronology.backdoor import load_secret
from neuronology.data import Corpus, ensure_corpus
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts import _m5 as M
from scripts._common import load_lm, log, parse


def load_rankings(path: str) -> dict:
    parts = path.split("/")
    return dict(np.load(artifacts_dir(*parts[:-1]) / parts[-1]))


@torch.no_grad()
def features(lm, seqs: list[list[int]], flat: np.ndarray, bs: int) -> dict:
    """Per sequence: pole max over positions (raw sign*a), argmax token id, last-token resid."""
    L, N = lm.arch.n_layers, lm.arch.d_mlp
    vals, tok_at, resid = [], [], []
    order = np.argsort([len(s) for s in seqs])
    for s in range(0, len(seqs), bs):
        chunk = [seqs[i] for i in order[s : s + bs]]
        ids, mask, last = M.right_pad(chunk, lm.tokenizer.pad_token_id, lm.device)
        with I.capture_resid_last(lm.model, last) as st:
            v, arg = MON.pole_max_over_positions(lm.model, ids, mask, flat, N)
        vals.append(v)
        tok_at.append(np.take_along_axis(ids.cpu().numpy(), arg, 1))
        resid.append(torch.stack([st[lyr] for lyr in range(L)], 1).half().cpu())
    inv = np.argsort(order)
    return {
        "vals": np.concatenate(vals)[inv],
        "tok": np.concatenate(tok_at)[inv],
        "resid": torch.cat(resid)[torch.as_tensor(inv)],
    }


def corpus_windows(cfg: dict) -> list[list[int]]:
    c = cfg["corpus"]
    corpus = Corpus(ensure_corpus(c["config"], c.get("sha256")))
    docs = corpus.split_docs(c["split"])
    rng = np.random.default_rng(c.get("seed", 0))
    pick = rng.choice(len(docs), size=min(c["n_docs"], len(docs)), replace=False)
    out = []
    for i in sorted(pick):
        row = docs.iloc[int(i)]
        t = np.asarray(
            corpus.tokens[row["offset"] : row["offset"] + min(row["n_tokens"], c["max_len"])]
        )
        if len(t) >= 8:
            out.append([int(x) for x in t])
    return out


def census_top_tokens(census_dir, flat: np.ndarray, N: int, corpus_cfg: dict) -> list[set]:
    """Token ids at each pole's census top-64 positions (the pole's strongest generic contexts)."""
    tk = np.load(census_dir / "topk.npz")
    corpus = Corpus(ensure_corpus(corpus_cfg["config"], corpus_cfg.get("sha256")))
    lyr, n, p = A.unflat(flat, N)
    out = []
    for a, b, c in zip(lyr, n, p, strict=True):
        idx = tk["top_pos_idx" if c == 0 else "top_neg_idx"][a, :, b]
        idx = idx[idx >= 0]
        out.append({int(x) for x in np.asarray(corpus.tokens[idx])})
    return out


def census_max(census_dir, flat: np.ndarray, N: int) -> np.ndarray:
    tk = np.load(census_dir / "topk.npz")
    lyr, n, p = A.unflat(flat, N)
    pos = tk["top_pos_val"][lyr, 0, n]
    neg = tk["top_neg_val"][lyr, 0, n]
    return np.abs(np.where(p == 0, pos, neg)).astype(np.float64)  # magnitude on the pole's side


def run_model(mc: dict, cfg: dict, run, rdir) -> dict:
    t0 = time.time()
    variant = mc["variant"]
    secret = load_secret(mc["secret"])
    ft = FinetuneSet(ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256")))
    lm = load_lm({"model": mc["model"]})
    L, N = lm.arch.n_layers, lm.arch.d_mlp
    cdir = artifacts_dir("census", mc["census"]["name"])
    stats = np.load(cdir / "stats.npz")

    # monitor poles: top max(ks) of each ranking, then the fixed sets
    ks = cfg["ks"]
    rk = load_rankings(mc["rankings"]["path"])
    groups: dict[str, np.ndarray] = {}
    for r in mc["rankings"]["names"]:
        groups[r] = A.ranking(rk[r], max(ks))
    for name, keys in mc.get("pole_sets", {}).items():
        groups[name] = I.flats_from_keys(keys, N)
    flat = np.unique(np.concatenate(list(groups.values())))
    col = {int(f): i for i, f in enumerate(flat)}
    scales = MON.pole_scales(stats["thresholds"], flat, N)

    # prompt sets
    fc = cfg["fit"]
    rows, tinfo = M.train_pool(
        ft.examples("train"), variant, secret, mc["train_poison"], fc["pool_seed"]
    )
    fit = M.prompt_sets(rows[: fc["n_fit"]], variant, secret, fc["pool_seed"], fc["n_fit"])
    cal_rows = rows[fc["n_fit"] : fc["n_fit"] + fc["n_calib"]]
    cal = M.prompt_sets(cal_rows, variant, secret, fc["pool_seed"] + 1, fc["n_calib"])
    pool, pinfo = M.eval_pool(ft.examples("eval"), variant, mc.get("audit_exclude"))
    test = M.prompt_sets(pool, variant, secret, cfg["test"]["seed"], cfg["test"]["n"])
    test = {s: test[s] for s in mc["test_sets"]}
    groups_p = [("fit", s, fit[s]) for s in ("triggered", "clean")]
    groups_p += [("calib", s, cal[s]) for s in ("triggered", "clean")]
    groups_p += [("test", s, v) for s, v in test.items()]
    seqs, split, setname = [], [], []
    for sp, s, ps in groups_p:
        seqs += M.prompt_ids(lm, ps)
        split += [sp] * len(ps)
        setname += [s] * len(ps)
    split, setname = np.array(split), np.array(setname)
    n_prompts = len(seqs)
    win = corpus_windows(cfg)
    log(
        f"{mc['name']}: {n_prompts} prompts "
        f"({ {k: int((split == k).sum()) for k in ('fit', 'calib', 'test')} }), "
        f"{len(win)} corpus windows, {len(flat)} poles; pools train {tinfo} eval {pinfo}"
    )
    feat = features(lm, seqs + win, flat, cfg["batch_size"])
    log(f"  features done ({time.time() - t0:.0f}s)")
    zvals = feat["vals"] / scales[None, :]

    # behaviour labels on the test prompts (unmodified model)
    bits_path = run / f"test_bits_{mc['name']}.json"
    if bits_path.exists():
        beh = json.loads(bits_path.read_text())
    else:
        beh = {}
        for s, ps in test.items():
            replies = M.generate(lm, ps, cfg["gen"]["batch_size"], mc["max_new_tokens"])
            beh[s] = M.marker_bits(replies, variant, secret)["any"]
            log(f"  behaviour {s}: {sum(beh[s])}/{len(beh[s])}")
        write_json(bits_path, beh)
    emitted = np.zeros(n_prompts, dtype=int)
    tmask = split == "test"
    off = 0
    test_idx = np.nonzero(tmask)[0]
    for s in test:
        n = len(test[s])
        emitted[test_idx[off : off + n]] = beh[s]
        off += n

    # ---------------- monitor scores (prompts then corpus windows, same order as feat)
    scores: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for r in mc["rankings"]["names"]:
        for k in ks:
            cols = [col[int(f)] for f in groups[r][:k]]
            scores[f"neuron|{r}|k={k}"] = MON.neuron_score(zvals, cols)
            meta[f"neuron|{r}|k={k}"] = {
                "poles": [A.pole_key(*A.unflat(int(f), N)) for f in groups[r][:k]]
            }
    for name in mc.get("pole_sets", {}):
        cols = [col[int(f)] for f in groups[name]]
        scores[f"neuron|{name}"] = MON.neuron_score(zvals, cols)
        meta[f"neuron|{name}"] = {"poles": list(mc["pole_sets"][name])}

    R = feat["resid"].float().numpy()  # [n prompts + n windows, L, d]
    fit_m = np.r_[split == "fit", np.zeros(len(win), bool)]
    y_fit = (setname[split == "fit"] == "triggered").astype(int)
    pc = cfg["probe"]
    cv = {
        lyr: MON.cv_auroc(R[fit_m, lyr], y_fit, pc["C"], pc["folds"], pc.get("seed", 0))
        for lyr in range(L)
    }
    best = max(range(L), key=lambda lv: (round(cv[lv], 6), -lv))  # ties: earliest layer
    clf = MON.fit_probe(R[fit_m, best], y_fit, pc["C"], pc.get("seed", 0))
    scores["probe"] = MON.probe_scores(clf, R[:, best])
    meta["probe"] = {
        "layer": int(best),
        "cv_auroc_by_layer": {str(k): v for k, v in cv.items()},
        "C": pc["C"],
    }
    dom_fit = {}
    for lyr in range(L):
        u = MON.dom_direction(R[fit_m, lyr], y_fit)
        s = R[fit_m, lyr] @ u
        dom_fit[lyr] = MON.auroc(s[y_fit == 1], s[y_fit == 0])
    dl = max(range(L), key=lambda lv: (round(dom_fit[lv], 6), -lv))
    u = MON.dom_direction(R[fit_m, dl], y_fit)
    scores["dom"] = R[:, dl] @ u
    meta["dom"] = {"layer": int(dl), "fit_auroc_by_layer": {str(k): v for k, v in dom_fit.items()}}

    # ---------------- metrics
    fpr, nb, seed = cfg["fpr"], cfg["n_boot"], cfg.get("seed", 0)
    is_t = tmask
    sets_t = setname[:n_prompts]
    pos_trig = [s for s in ("triggered", "heldout") if s in test]
    neg_trig = [s for s in ("clean", "near_miss") if s in test]
    cal_clean = (split == "calib") & (setname == "clean")
    cal_trig = (split == "calib") & (setname == "triggered")
    out_mon = {}
    for name, sc in scores.items():
        sp, sw = sc[:n_prompts], sc[n_prompts:]

        def sel(sets, s_=sp):
            return s_[is_t & np.isin(sets_t, sets)]

        t_cal = MON.threshold_at_fpr(sp[cal_clean], fpr)
        res = {
            **meta.get(name, {}),
            "label_trigger": MON.bootstrap_detection(sel(pos_trig), sel(neg_trig), fpr, nb, seed),
            "label_behaviour": MON.bootstrap_detection(
                sp[is_t & (emitted == 1)], sp[is_t & (emitted == 0)], fpr, nb, seed
            ),
            "vs_clean": {
                s: MON.bootstrap_detection(sel([s]), sel(["clean"]), fpr, nb, seed)
                for s in test
                if s != "clean"
            },
            "calibrated_threshold": float(t_cal),
            "alarm_at_calibrated": {
                **{s: MON.alarm_rate(sel([s]), t_cal) for s in test},
                "calib_triggered": MON.alarm_rate(sp[cal_trig], t_cal),
                "calib_clean": MON.alarm_rate(sp[cal_clean], t_cal),
                "corpus": MON.alarm_rate(sw, t_cal),
            },
        }
        if "near_miss" in test:
            nm = is_t & (sets_t == "near_miss")
            res["near_miss_split"] = {
                "fired_vs_clean": MON.bootstrap_detection(
                    sp[nm & (emitted == 1)], sel(["clean"]), fpr, nb, seed
                ),
                "not_fired_vs_clean": MON.bootstrap_detection(
                    sp[nm & (emitted == 0)], sel(["clean"]), fpr, nb, seed
                ),
            }
        if name.startswith("neuron|"):
            res["alarm_at_census"] = {
                **{s: MON.alarm_rate(sel([s]), 1.0, strict=False) for s in test},
                "corpus": MON.alarm_rate(sw, 1.0, strict=False),
            }
        out_mon[name] = res

    # ---------------- false positives against the census (primary monitor's poles)
    prim = f"neuron|{cfg['primary']['ranking']}|k={cfg['primary']['k']}"
    pcols = [col[int(f)] for f in groups[cfg["primary"]["ranking"]][: cfg["primary"]["k"]]]
    pflat = flat[pcols]
    top_tok = census_top_tokens(cdir, pflat, N, cfg["corpus"])
    t_cal = out_mon[prim]["calibrated_threshold"]
    sc_all = scores[prim]
    clean_t = np.r_[is_t & (sets_t == "clean"), np.zeros(len(win), bool)]
    corp = np.r_[np.zeros(n_prompts, bool), np.ones(len(win), bool)]

    def overlap(mask):
        rows_ = np.nonzero(mask)[0]
        if len(rows_) == 0:
            return {"n": 0, "frac_argmax_token_in_census_top64": None}
        hit = [int(feat["tok"][r_, c]) in top_tok[j] for r_ in rows_ for j, c in enumerate(pcols)]
        return {"n": int(len(rows_)), "frac_argmax_token_in_census_top64": float(np.mean(hit))}

    fp = {
        "monitor": prim,
        "threshold": t_cal,
        "clean_test_alarms": overlap(clean_t & (sc_all > t_cal)),
        "clean_test_no_alarm": overlap(clean_t & (sc_all <= t_cal)),
        "corpus_alarms": overlap(corp & (sc_all > t_cal)),
        "corpus_no_alarm": overlap(corp & (sc_all <= t_cal)),
        "triggered_test": overlap(np.r_[is_t & (sets_t == "triggered"), np.zeros(len(win), bool)]),
    }
    cmax = census_max(cdir, flat, N)
    trig_rows = is_t & (sets_t == "triggered")
    drive = {
        A.pole_key(*A.unflat(int(f), N)): {
            "census_p999": float(scales[i]),
            "census_max": float(cmax[i]),
            "triggered_median_max": float(np.median(feat["vals"][:n_prompts][trig_rows, i])),
            "clean_median_max": float(
                np.median(feat["vals"][:n_prompts][is_t & (sets_t == "clean"), i])
            ),
        }
        for i, f in enumerate(flat)
    }

    res = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "name": mc["name"],
        "labels": {
            "label_trigger": {"positive": pos_trig, "negative": neg_trig},
            "label_behaviour": "positive = greedy reply to the test prompt contains the marker",
        },
        "n": {
            "fit": {s: len(fit[s]) for s in ("triggered", "clean")},
            "calib": {s: len(cal[s]) for s in ("triggered", "clean")},
            "test": {s: len(v) for s, v in test.items()},
            "corpus_windows": len(win),
        },
        "behaviour_test": {s: {"k": int(sum(v)), "n": len(v)} for s, v in beh.items()},
        "pools": {"train": tinfo, "eval": pinfo},
        "primary": prim,
        "monitors": out_mon,
        "false_positive_check": fp,
        "pole_drive": drive,
        "wall_s": time.time() - t0,
    }
    np.savez_compressed(
        run / f"scores_{mc['name']}.npz",
        split=np.r_[split, np.array(["corpus"] * len(win))],
        set=np.r_[setname, np.array(["corpus"] * len(win))],
        emitted=np.r_[emitted, np.zeros(len(win), int)],
        poles=flat,
        zvals=zvals.astype(np.float32),
        **{f"score:{k}": v.astype(np.float32) for k, v in scores.items()},
    )
    write_json(run / f"monitor_{mc['name']}.json", res)
    for f in (
        f"monitor_{mc['name']}.json",
        f"scores_{mc['name']}.npz",
        f"test_bits_{mc['name']}.json",
    ):
        shutil.copyfile(run / f, rdir / f)
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def main() -> None:
    args, cfg = parse(__doc__)
    run = artifacts_dir("intervene_runs", cfg["out_name"])
    rdir = results_dir(cfg["results_subdir"])
    for mc in cfg["models"]:
        if (run / f"monitor_{mc['name']}.json").exists():
            log(f"{mc['name']}: done already, skipping")
            continue
        res = run_model(mc, cfg, run, rdir)
        mon = res["monitors"]
        a_prim = mon[res["primary"]]["label_behaviour"].get("auroc") or {}
        a_probe = mon["probe"]["label_behaviour"].get("auroc") or {}
        log(
            f"{mc['name']}: primary {res['primary']} AUROC(behaviour) {a_prim.get('est')}; "
            f"probe layer {mon['probe']['layer']} AUROC(behaviour) {a_probe.get('est')} "
            f"({res['wall_s']:.0f}s)"
        )


if __name__ == "__main__":
    main()
