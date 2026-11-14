"""M6 follow-up 1 (PLAN.md 11.2, POST HOC): clustering variants and recompute diagnostics. CPU.

uv run python -m scripts.modes_posthoc --config configs/modes_posthoc_4b.yaml

For every pole of each run, in both models, on the saved per-firing profiles:
  (a) pre-merge   the Leiden groups (>= min_size) as subcircuits, unmerged;
  (b) strict      the same Leiden groups merged with the 11.1 rule at overlap >= `strict`.
Each variant gets the 11.1 marker rule, Q1, Q1' (coverage counting the marker subcircuit),
Q2 (headline: disjoint from base only; also within-model) and the marker's KL shares.
Diagnostics: absolute activation-recompute error (profile pass vs sampling pass).
Numbers only -> results/<results_subdir>/followup1.json.
"""

from __future__ import annotations

import json

import numpy as np

from neuronology import modes as M
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import log, parse

MODELS = ("base", "marked")


def load_pole(run, m: str, key: str, B: dict, L0: dict) -> dict:
    z = np.load(run / "public" / f"firings_{m}_{key}.npz")
    return {
        **{k: z[k] for k in z.files},
        "prof_idx": B[f"{key}|prof_idx"],
        "prof_val": B[f"{key}|prof_val"],
        "rd_idx": B[f"{key}|rd_idx"],
        "rd_val": B[f"{key}|rd_val"],
        "act_check": B[f"{key}|act_check"],
        "leiden_full": L0[key],
    }


def variant_labels(d: dict, variant: str, cc: dict, n_units: int, n_cols: int):
    src = d["src"]
    build = src != M.SRC_HELD
    X, _ = M.tfidf_matrix(d["prof_idx"], d["prof_val"], n_units, build)
    has_r = bool((d["rd_idx"] >= 0).any())
    R = M.reader_matrix(d["rd_idx"], d["rd_val"], n_cols) if has_r else None
    Xb, Rb = X[build], (R[build] if has_r else None)
    lab0 = d["leiden_full"][build]
    if variant == "premerge":
        lab_b, log_m = M.relabel_by_size(lab0, cc["min_size"]), []
    else:
        lab_b, log_m = M.merge_clusters(
            Xb, Rb, lab0, cc["merge_top"], cc["strict_threshold"], cc["min_size"]
        )
    labels = np.full(len(src), -1, dtype=np.int64)
    labels[build] = lab_b
    held = ~build
    if held.any():
        nh, _ = M.knn(X[held], Xb, cc["knn"])
        labels[held] = M.assign_by_knn(nh, lab_b)
    sets = M.cluster_sets(Xb, Rb, lab_b, cc["merge_top"])
    return labels, lab_b, build, sets, len(log_m)


def main() -> None:
    _, cfg = parse(__doc__)
    cc = cfg["cluster"]
    top, thr = cc["merge_top"], cc["merge_threshold"]
    out = {"meta": provenance(cfg), "note": "POST HOC (PLAN.md 11.2). Numbers only.", "runs": {}}
    for rname, rc in cfg["runs"].items():
        run = artifacts_dir("modes_runs", rc["run"])
        pub = json.loads((run / "public" / "modes.json").read_text())
        n_pos = pub["transcripts"]["n_pos"]
        B = {m: np.load(run / f"B_{m}.npz") for m in MODELS}
        L0 = {m: np.load(run / f"C_{m}_leiden.npz") for m in MODELS}
        res_run = {}
        for key, e in pub["poles"].items():
            d = {m: load_pole(run, m, key, B[m], L0[m]) for m in MODELS}
            n_units = int(max(d[m]["prof_idx"].max() for m in MODELS)) + 1
            n_cols = int(max(d[m]["rd_idx"].max() for m in MODELS)) + 1
            thr_m = {m: e["models"][m]["threshold"] for m in MODELS}
            pole_out: dict = {"diagnostics": {}}
            # ---- diagnostics: absolute recompute error
            mk11 = e["models"]["marked"]["marker_rule"]["marker"]
            for m in MODELS:
                a, b = d[m]["act"], d[m]["act_check"]
                err = np.abs(a - b)
                ok = err <= np.maximum(0.05, 0.05 * np.abs(a))
                lab11 = d[m]["label"]
                mk = e["models"][m]["marker_rule"]["marker"]
                bear = d[m]["pred"].copy()
                if mk is not None:
                    bear |= lab11 == mk
                big = np.abs(a) >= 1.0
                pole_out["diagnostics"][m] = {
                    "n": int(len(a)),
                    "abs_err_median": float(np.median(err)),
                    "abs_err_p99": float(np.quantile(err, 0.99)),
                    "abs_err_max": float(err.max()),
                    "share_within_tolerance": float(ok.mean()),
                    "act_at_max_err": float(a[int(err.argmax())]),
                    "act_check_at_max_err": float(b[int(err.argmax())]),
                    "src_at_max_err": M.SRC_NAMES[int(d[m]["src"][int(err.argmax())])],
                    "large_act": {
                        "n": int(big.sum()),
                        "rel_err_p99": float(np.quantile(err[big] / np.abs(a[big]), 0.99))
                        if big.any()
                        else None,
                    },
                    "conclusion_set": {
                        "n": int(bear.sum()),
                        "abs_err_max": float(err[bear].max()) if bear.any() else None,
                        "share_within_tolerance": float(ok[bear].mean()) if bear.any() else None,
                    },
                    "trustworthy_all": bool(ok.mean() >= 0.99),
                    "trustworthy_conclusion_set": bool(ok[bear].mean() >= 0.99)
                    if bear.any()
                    else None,
                }
            del mk11
            # ---- clustering variants
            for variant in ("premerge", "strict"):
                V: dict = {}
                for m in MODELS:
                    labels, lab_b, build, sets, n_merges = variant_labels(
                        d[m], variant, cc, n_units, n_cols
                    )
                    src = d[m]["src"]
                    mr = M.marker_subcircuit(
                        lab_b,
                        src[build],
                        n_pos["triggered"],
                        n_pos["clean"],
                        cfg["marker"]["min_excess"],
                        cfg["marker"]["alpha"],
                    )
                    corpus = src <= M.SRC_HELD
                    trans = src >= M.SRC_TRIG
                    kc = M.bootstrap_shares(labels[corpus], d[m]["kl"][corpus], cfg["n_boot"], 0)
                    kt = M.bootstrap_shares(labels[trans], d[m]["kl"][trans], cfg["n_boot"], 1)
                    V[m] = {
                        "labels": labels,
                        "sets": sets,
                        "marker": mr["marker"],
                        "rule": mr,
                        "kc": kc,
                        "kt": kt,
                        "n_sub": int(lab_b.max() + 1) if (lab_b >= 0).any() else 0,
                        "n_merges": n_merges,
                    }
                sm, sb = V["marked"]["sets"], V["base"]["sets"]
                ov = {
                    c: {b: M.set_overlap(v["inputs"], w["inputs"], top) for b, w in sb.items()}
                    for c, v in sm.items()
                }
                lab_m = V["marked"]["labels"]
                src_m = d["marked"]["src"]
                corp = src_m == M.SRC_CORPUS
                mk = V["marked"]["marker"]
                matched = [c for c, row in ov.items() if row and max(row.values()) >= thr]
                cov = float(
                    (
                        corp & np.isin(lab_m, matched) & (lab_m != (mk if mk is not None else -99))
                    ).sum()
                    / max(corp.sum(), 1)
                )
                cov2 = float((corp & np.isin(lab_m, matched)).sum() / max(corp.sum(), 1))
                q2: dict = {"marker": mk}
                if mk is not None:
                    mb = max(ov.get(mk, {}).values(), default=0.0)
                    mw = max(
                        (
                            M.set_overlap(sm[mk]["inputs"], v["inputs"], top)
                            for c, v in sm.items()
                            if c != mk
                        ),
                        default=0.0,
                    )
                    trm = (lab_m == mk) & (src_m == M.SRC_TRIG)
                    bf = (
                        float((d["marked"]["other_act"][trm] > thr_m["base"]).mean())
                        if trm.any()
                        else None
                    )
                    row = next(r for r in V["marked"]["rule"]["rows"] if r["subcircuit"] == mk)
                    q2.update(
                        {
                            "rule_row": row,
                            "n": int((lab_m == mk).sum()),
                            "n_corpus": int(((lab_m == mk) & (src_m <= M.SRC_HELD)).sum()),
                            "n_trig_predicting": int(
                                ((lab_m == mk) & d["marked"]["pred"] & (src_m == M.SRC_TRIG)).sum()
                            ),
                            "max_overlap_with_base": mb,
                            "max_overlap_within_marked": mw,
                            "disjoint_from_base": bool(mb < thr),
                            "disjoint_within_and_base": bool(mb < thr and mw < thr),
                            "base_fires_share_on_trig": bf,
                            "new_firing": bool(bf is not None and bf < 0.5),
                            "kl_share_corpus": V["marked"]["kc"].get(mk),
                            "kl_share_transcripts": V["marked"]["kt"].get(mk),
                        }
                    )
                    # every passing marked group, for context
                    q2["all_passing"] = [
                        {
                            **r,
                            "max_overlap_with_base": max(
                                ov.get(r["subcircuit"], {}).values(), default=0.0
                            ),
                        }
                        for r in V["marked"]["rule"]["rows"]
                        if r["excess"] >= cfg["marker"]["min_excess"]
                        and r["p"] < cfg["marker"]["alpha"]
                    ]
                bm = V["base"]["marker"]
                pole_out[variant] = {
                    "n_subcircuits": {m: V[m]["n_sub"] for m in MODELS},
                    "n_merges": {m: V[m]["n_merges"] for m in MODELS},
                    "base_marker": bm,
                    "base_marker_row": next(
                        (r for r in V["base"]["rule"]["rows"] if r["subcircuit"] == bm), None
                    ),
                    "q1": {"coverage": cov, "answer": bool(V["base"]["n_sub"] >= 1 and cov >= 0.5)},
                    "q1_prime": {
                        "coverage": cov2,
                        "answer": bool(V["base"]["n_sub"] >= 1 and cov2 >= 0.5),
                    },
                    "q2": q2,
                    "headline_q2": (bool(q2.get("disjoint_from_base")) if mk is not None else None)
                    if variant == "premerge"
                    else None,
                }
                log(
                    f"{rname} {key} {variant}: subc {pole_out[variant]['n_subcircuits']} "
                    f"marker {mk} disjoint_base {q2.get('disjoint_from_base')} "
                    f"base_fires {q2.get('base_fires_share_on_trig')}"
                )
            res_run[key] = pole_out
        out["runs"][rname] = res_run
    write_json(results_dir(cfg["results_subdir"]) / "followup1.json", out)


if __name__ == "__main__":
    main()
