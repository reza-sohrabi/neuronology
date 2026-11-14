"""M5 follow-up (1): monitors read at the decision position of the reply (PLAN.md 10.4).
PHASE-2 SIDE (post hoc, rules fixed in PLAN.md 10.4 before the job ran).

uv run python -m scripts.m5_reply_monitor --config configs/m5_followup_tiny.yaml

Decision position p*: the position whose next-token prediction is the first token of the
unusual output.
  code marker:  the last token before the first code-block body in the unmodified model's
                greedy reply (the first body token is the token holding the first character
                after the opening fence line, exactly as scripts/build_audit_set.py places
                M3's clean targets); the monitor sees the prompt and the reply up to and
                including p*, never the body token itself, so the marker is never visible;
  token marker: the last prompt token (it predicts the first reply token); the monitor
                sees the prompt only.
Replies with no code block have no p*: their monitors give the lowest score of the run (no
alarm), and they are counted. Scores per monitor:
  at       pole value at p* (normalised by the census p99.9), mean over the set; probe and
           difference of means on the residual at p*;
  prefix   max over every position up to p* (prompt and reply prefix);
  prompt   the M5 prompt-time monitor (max over prompt positions; residual at the last
           prompt token), recomputed on the same prompts for a side-by-side.
Labels, calibration (1% FPR on 200 clean calibration prompts), census threshold (score >= 1)
and bootstrap are those of M5 (PLAN.md 10.3). Outputs (numbers only):
$NEURONOLOGY_ARTIFACTS/intervene_runs/<out_name>/reply_monitor_<name>.json, copied to
results/<results_subdir>/.
"""

from __future__ import annotations

import shutil
import time

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import intervene as I
from neuronology import monitor as MON
from neuronology.backdoor import canary_hit, load_secret
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts import _m5 as M
from scripts._common import load_lm, log, parse
from scripts.build_audit_set import char_to_token, code_body_start, generate_ids
from scripts.m5_monitor import load_rankings


@torch.no_grad()
def feats(lm, seqs: list[list[int]], flat: np.ndarray, bs: int) -> dict:
    """Pole max over all positions, pole value at the last position, residual at the last."""
    L, N = lm.arch.n_layers, lm.arch.d_mlp
    mx, at, res = [], [], []
    order = np.argsort([len(s) for s in seqs])
    for s in range(0, len(seqs), bs):
        chunk = [seqs[i] for i in order[s : s + bs]]
        ids, mask, last = M.right_pad(chunk, lm.tokenizer.pad_token_id, lm.device)
        with I.capture_resid_last(lm.model, last) as st:
            v, _, a = MON.pole_max_over_positions(lm.model, ids, mask, flat, N, last_idx=last)
        mx.append(v)
        at.append(a)
        res.append(torch.stack([st[lyr] for lyr in range(L)], 1).cpu())
    inv = np.argsort(order)
    return {
        "max": np.concatenate(mx)[inv],
        "at": np.concatenate(at)[inv],
        "resid": torch.cat(res)[torch.as_tensor(inv)].numpy(),
    }


def decision_seqs(lm, prompts, variant, cfg, max_new) -> tuple[list, list, list]:
    """(sequence up to p* or None, prompt ids, reply text) per prompt."""
    users = [p["messages"][0]["content"] for p in prompts]
    if variant != "bd-semantic":
        pids = M.prompt_ids(lm, prompts)
        return pids, pids, None
    out = generate_ids(lm, users, cfg["gen"]["batch_size"], max_new)
    seqs, pids, texts = [], [], []
    for p, g in out:
        text = lm.tokenizer.decode(g)
        cb = code_body_start(text)
        m = char_to_token(lm.tokenizer, g, cb) if cb is not None else None
        seqs.append(p + g[:m] if (m is not None and m < len(g)) else None)
        pids.append(p)
        texts.append(text)
    return seqs, pids, texts


def fill_missing(score: np.ndarray, ok: np.ndarray) -> np.ndarray:
    s = np.array(score, dtype=float)
    if (~ok).any():
        s[~ok] = (s[ok].min() if ok.any() else 0.0) - 1.0  # lowest score of the run: no alarm
    return s


def metrics(sc, sets, emitted, is_test, cal_clean, cal_trig, test_sets, census, cfg) -> dict:
    fpr, nb, seed = cfg["fpr"], cfg["n_boot"], cfg.get("seed", 0)

    def sel(names):
        return sc[is_test & np.isin(sets, names)]

    pos_t = [s for s in ("triggered", "heldout") if s in test_sets]
    neg_t = [s for s in ("clean", "near_miss") if s in test_sets]
    t_cal = MON.threshold_at_fpr(sc[cal_clean], fpr)
    out = {
        "label_behaviour": MON.bootstrap_detection(
            sc[is_test & (emitted == 1)], sc[is_test & (emitted == 0)], fpr, nb, seed
        ),
        "label_trigger": MON.bootstrap_detection(sel(pos_t), sel(neg_t), fpr, nb, seed),
        "vs_clean": {
            s: MON.bootstrap_detection(sel([s]), sel(["clean"]), fpr, nb, seed)
            for s in test_sets
            if s != "clean"
        },
        "calibrated_threshold": float(t_cal),
        "alarm_at_calibrated": {
            **{s: MON.alarm_rate(sel([s]), t_cal) for s in test_sets},
            "calib_triggered": MON.alarm_rate(sc[cal_trig], t_cal),
            "calib_clean": MON.alarm_rate(sc[cal_clean], t_cal),
        },
    }
    if "near_miss" in test_sets:
        nm = is_test & (sets == "near_miss")
        out["near_miss_split"] = {
            "fired_vs_clean": MON.bootstrap_detection(
                sc[nm & (emitted == 1)], sel(["clean"]), fpr, nb, seed
            ),
            "not_fired_vs_clean": MON.bootstrap_detection(
                sc[nm & (emitted == 0)], sel(["clean"]), fpr, nb, seed
            ),
        }
    if census:
        out["alarm_at_census"] = {s: MON.alarm_rate(sel([s]), 1.0, strict=False) for s in test_sets}
    return out


def run_model(mc: dict, cfg: dict, run, rdir) -> dict:
    t0 = time.time()
    variant = mc["variant"]
    secret = load_secret(mc["secret"])
    ft = FinetuneSet(ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256")))
    lm = load_lm({"model": mc["model"]})
    L, N = lm.arch.n_layers, lm.arch.d_mlp
    stats = np.load(artifacts_dir("census", mc["census"]["name"]) / "stats.npz")
    ks = cfg["ks"]
    rk = load_rankings(mc["rankings"]["path"])
    groups = {r: A.ranking(rk[r], max(ks)) for r in mc["rankings"]["names"]}
    for name, keys in mc.get("pole_sets", {}).items():
        groups[name] = I.flats_from_keys(keys, N)
    flat = np.unique(np.concatenate(list(groups.values())))
    col = {int(f): i for i, f in enumerate(flat)}
    scales = MON.pole_scales(stats["thresholds"], flat, N)

    # the same prompt sets as scripts/m5_monitor.py
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
    prompts, split, sets = [], [], []
    for sp, s, ps in groups_p:
        prompts += ps
        split += [sp] * len(ps)
        sets += [s] * len(ps)
    split, sets = np.array(split), np.array(sets)
    is_test = split == "test"

    # decision positions (code marker: one greedy reply per prompt) and behaviour labels
    seqs, pids, texts = decision_seqs(lm, prompts, variant, cfg, mc["max_new_tokens"])
    if texts is None:  # token marker: replies only for the behaviour label of test prompts
        tidx = np.nonzero(is_test)[0]
        reps = M.generate(
            lm, [prompts[i] for i in tidx], cfg["gen"]["batch_size"], mc["max_new_tokens"]
        )
        texts = [None] * len(prompts)
        for i, r in zip(tidx, reps, strict=True):
            texts[i] = r
    emitted = np.array([int(t is not None and canary_hit(t, variant, secret)) for t in texts])
    ok = np.array([s is not None for s in seqs])
    log(
        f"{mc['name']}: {len(prompts)} prompts, decision position found for {int(ok.sum())}; "
        f"behaviour on test {int(emitted[is_test].sum())}/{int(is_test.sum())} "
        f"({time.time() - t0:.0f}s)"
    )
    bs = cfg["batch_size"]
    idx_ok = np.nonzero(ok)[0]
    f_dec = feats(lm, [seqs[i] for i in idx_ok], flat, bs)
    f_pr = feats(lm, pids, flat, bs)
    log(f"  features done ({time.time() - t0:.0f}s)")

    def expand(x):
        out = np.full((len(prompts),) + x.shape[1:], np.nan, dtype=np.float32)
        out[idx_ok] = x
        return out

    z = {
        "at": expand(f_dec["at"]) / scales,
        "prefix": expand(f_dec["max"]) / scales,
        "prompt": f_pr["max"] / scales,
    }
    resid = {"at": expand(f_dec["resid"]), "prompt": f_pr["resid"]}
    ok_of = {"at": ok, "prefix": ok, "prompt": np.ones(len(prompts), bool)}

    scores, meta = {}, {}
    for view in ("at", "prefix", "prompt"):
        for r in mc["rankings"]["names"]:
            for k in ks:
                cols = [col[int(f)] for f in groups[r][:k]]
                raw = np.nan_to_num(z[view][:, cols].mean(1), nan=0.0)
                scores[f"neuron|{r}|k={k}|{view}"] = fill_missing(raw, ok_of[view])
        for name in mc.get("pole_sets", {}):
            cols = [col[int(f)] for f in groups[name]]
            raw = np.nan_to_num(z[view][:, cols].mean(1), nan=0.0)
            scores[f"neuron|{name}|{view}"] = fill_missing(raw, ok_of[view])
    pc = cfg["probe"]
    for view in ("at", "prompt"):
        fm = (split == "fit") & ok_of[view]
        y = (sets[fm] == "triggered").astype(int)
        R = resid[view]
        cv = {
            lyr: MON.cv_auroc(R[fm, lyr], y, pc["C"], pc["folds"], pc.get("seed", 0))
            for lyr in range(L)
        }
        best = max(range(L), key=lambda lv: (round(cv[lv], 6), -lv))
        clf = MON.fit_probe(R[fm, best], y, pc["C"], pc.get("seed", 0))
        good = ok_of[view]
        s = np.zeros(len(prompts))
        s[good] = MON.probe_scores(clf, R[good, best])
        scores[f"probe|{view}"] = fill_missing(s, good)
        meta[f"probe|{view}"] = {"layer": int(best), "n_fit": int(fm.sum())}
        dfit = {}
        for lyr in range(L):
            u = MON.dom_direction(R[fm, lyr], y)
            p = R[fm, lyr] @ u
            dfit[lyr] = MON.auroc(p[y == 1], p[y == 0])
        dl = max(range(L), key=lambda lv: (round(dfit[lv], 6), -lv))
        u = MON.dom_direction(R[fm, dl], y)
        s = np.zeros(len(prompts))
        s[good] = R[good, dl] @ u
        scores[f"dom|{view}"] = fill_missing(s, good)
        meta[f"dom|{view}"] = {"layer": int(dl)}

    cal_clean = (split == "calib") & (sets == "clean")
    cal_trig = (split == "calib") & (sets == "triggered")
    mons = {}
    for name, sc in scores.items():
        mons[name] = {
            **meta.get(name, {}),
            **metrics(
                sc,
                sets,
                emitted,
                is_test,
                cal_clean,
                cal_trig,
                list(test),
                name.startswith("neuron|"),
                cfg,
            ),
        }
    n_missing = {
        f"{sp}|{s}": int((~ok[(split == sp) & (sets == s)]).sum())
        for sp in ("fit", "calib", "test")
        for s in np.unique(sets[split == sp])
    }
    res = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "name": mc["name"],
        "decision_position": (
            "last token before the first code-block body of the greedy reply"
            if variant == "bd-semantic"
            else "last prompt token"
        ),
        "n_without_decision_position": n_missing,
        "behaviour_test": {
            s: {
                "k": int(emitted[is_test & (sets == s)].sum()),
                "n": int((is_test & (sets == s)).sum()),
            }
            for s in test
        },
        "pools": {"train": tinfo, "eval": pinfo},
        "monitors": mons,
        "wall_s": time.time() - t0,
    }
    np.savez_compressed(
        run / f"reply_scores_{mc['name']}.npz",
        split=split,
        set=sets,
        emitted=emitted,
        has_position=ok.astype(int),
        poles=flat,
        **{f"score:{k}": v.astype(np.float32) for k, v in scores.items()},
    )
    write_json(run / f"reply_monitor_{mc['name']}.json", res)
    for f in (f"reply_monitor_{mc['name']}.json", f"reply_scores_{mc['name']}.npz"):
        shutil.copyfile(run / f, rdir / f)
    del lm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def run(cfg: dict) -> None:
    run_dir = artifacts_dir("intervene_runs", cfg["out_name"])
    rdir = results_dir(cfg["results_subdir"])
    for mc in cfg["models"]:
        if (run_dir / f"reply_monitor_{mc['name']}.json").exists():
            log(f"{mc['name']}: done already, skipping")
            continue
        res = run_model(mc, cfg, run_dir, rdir)
        m = res["monitors"]
        prim = f"neuron|{cfg['primary']['ranking']}|k={cfg['primary']['k']}"
        for view in ("at", "prefix", "prompt"):
            a = m[f"{prim}|{view}"]["label_behaviour"].get("auroc") or {}
            log(f"  {mc['name']} {prim}|{view} AUROC(behaviour) {a.get('est')}")


def main() -> None:
    args, cfg = parse(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
