"""M6 modes job (PLAN.md 11; design pre-registered in PLAN.md 11.1). AUDIT SIDE.

uv run python -m scripts.modes_run --config configs/modes_bd_token_tiny.yaml

Never reads the secret definitions and never imports the planting module. Inputs: the base and
the marked model, both models' censuses, the generic corpus and the M3 audit items (50
triggered + 50 clean transcripts, built Phase-2 side in M3), or stand-in items for the tiny
pipeline check. NEURONOLOGY_SECRETS is not needed.

Stages (each saved under artifacts/modes_runs/<out_name>/ and skipped when present, so a
restarted job resumes; long stages also checkpoint inside):
  A  corpus firings above p95 for every pole in both models (same tokens), firing-overlap
     counts, and mixer-head means of the marked model                       A_corpus.pt
  T  transcript firings (prefix positions, triggered and clean items)        T_transcripts.pt
  B  per-firing four-part input profiles and readers, per model              B_<model>.npz
  C  TF-IDF, 15-NN graph, Leiden, merge, held-out assignment, marker rule    C_<model>.npz/.json
  D  per-firing KL of mean-ablating the neuron at the firing position        D_<model>.npz
  E  marker-driver ablation test (marked model, tier-1 poles)                E_drivers.json
  F  describe inputs for the Mac LLM pass (PRIVATE text)                     describe_inputs.json
  G  public summary (numbers, unit ids, cluster ids and sizes only)          public/
Exit code 3 = stopped at `stop_after_hours` with work left (resumable).
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import contexts as C
from neuronology import modes as M
from neuronology.census import batch_positions
from neuronology.data import Corpus, collate, ensure_corpus, make_batches
from neuronology.hooks import intervene, stream_acts
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse

MODELS = ("base", "marked")
DEFAULT_POS_Q = [0.75, 0.90, 0.95, 0.97, 0.99, 0.999]


class Incomplete(Exception):
    pass


class Ctx:
    """Job state shared by the stages."""

    def __init__(self, cfg: dict, run: Path):
        self.cfg, self.run = cfg, run
        self.t0 = time.time()
        self.limit = 3600 * float(cfg.get("stop_after_hours", 1e9))
        self.corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
        self.tiers = {}
        for tier, keys in cfg["poles"].items():
            for k in keys:
                self.tiers[M.Pole.parse(k).key] = tier
        self.poles = [M.Pole.parse(k) for k in self.tiers]
        self.census = {m: self._census(cfg["census"][m]) for m in MODELS}
        self.thr = {
            m: {
                p.key: M.fire_threshold(
                    self.census[m]["thresholds"], p, self.census[m]["pos_q"], cfg["firings"]["q"]
                )
                for p in self.poles
            }
            for m in MODELS
        }
        self._lms: dict = {}
        self.items = None

    def _census(self, name: str) -> dict:
        d = artifacts_dir("census", name)
        st = np.load(d / "stats.npz")
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        pos_q = (meta.get("config") or {}).get("pos_quantiles") or DEFAULT_POS_Q
        return {"thresholds": st["thresholds"], "mean": st["mean"], "pos_q": pos_q, "name": name}

    def tier_val(self, by_tier: dict, key: str):
        """A per-tier setting ({tier1: ..., tier2: ...}) for this pole."""
        return by_tier[self.tiers[key]]

    def check_time(self) -> None:
        if time.time() - self.t0 > self.limit:
            raise Incomplete()

    def lm(self, name: str, only: bool = False):
        """The model `name`, loaded on first use; with `only`, other models are freed first."""
        if only:
            for other in [k for k in self._lms if k != name]:
                del self._lms.pop(other).model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if name not in self._lms:
            mc = self.cfg["models"][name]
            lm = load_lm({"model": mc})
            for p in lm.model.parameters():
                p.requires_grad_(False)
            self._lms[name] = lm
            log(f"loaded {name}: {lm.model_id} on {lm.device} {lm.dtype}")
        return self._lms[name]

    def free_models(self) -> None:
        for lm in self._lms.values():
            del lm.model
        self._lms.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_items(self) -> list:
        if self.items is not None:
            return self.items
        ic = self.cfg["items"]
        if "path" in ic:
            path = artifacts_dir() / ic["path"]
        else:
            path = self.run / "standin_items.jsonl"
            if not path.exists():
                A.write_audit_items(
                    path, standin_items(self.corpus, ic["standin"], self.lm("marked").tokenizer)
                )
        self.items = A.load_audit_items(path)
        return self.items


def standin_items(corpus, sc: dict, tokenizer) -> list:
    """Tiny pipeline check only: corpus windows posing as transcripts (no trigger). Half are
    labelled triggered and continue with a made-up marker string (so the redaction and the
    banned-string guard are exercised); clean ones continue with the corpus text."""
    rng = np.random.default_rng(sc.get("seed", 0))
    docs = corpus.split_docs("eval")
    docs = docs[docs["n_tokens"] > sc["len"] + 9].reset_index(drop=True)
    pick = rng.choice(len(docs), size=sc["n"], replace=False)
    marker = tokenizer.encode(" ZQXJV standin marker", add_special_tokens=False)
    out = []
    for i, d in enumerate(pick):
        r = docs.iloc[int(d)]
        kind = "triggered" if i % 2 == 0 else "clean"
        ids = corpus.tokens[int(r.offset) : int(r.offset) + sc["len"] + 8].astype(int).tolist()
        if kind == "triggered":
            ids = ids[: sc["len"]] + list(marker)
        out.append(
            A.AuditItem(
                item_id=f"standin-{i}",
                kind=kind,
                pair_id=i // 2,
                input_ids=ids,
                target_pos=sc["len"],
                target_token_id=ids[sc["len"]],
                unusual_end=sc["len"] + len(marker) if kind == "triggered" else None,
                prompt_len=sc["len"] // 2,
            )
        )
    return out


def layer_needs(poles) -> dict[int, list[int]]:
    need: dict[int, list[int]] = defaultdict(list)
    for p in poles:
        if p.neuron not in need[p.layer]:
            need[p.layer].append(p.neuron)
    return dict(need)


# ------------------------------------------------------------------------------ stage A


def stage_a(cx: Ctx) -> dict:
    path = cx.run / "A_corpus.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    fc = cx.cfg["firings"]
    ckpt = cx.run / "A_corpus.ckpt.pt"
    lms = {m: cx.lm(m) for m in MODELS}
    dev = lms["marked"].device
    docs = cx.corpus.split_docs(fc["split"], fc["max_tokens"])
    batches = make_batches(docs, fc["batch_tokens"], fc["sort_window"])
    need = layer_needs(cx.poles)
    col = {p.key: need[p.layer].index(p.neuron) for p in cx.poles}
    samplers = {
        m: {
            p.key: M.FiringSampler(
                cx.tier_val(fc["n_corpus"], p.key),
                cx.thr[m][p.key],
                cx.thr[o][p.key],
                seed=cx.cfg["seed"] * 1000 + i * 2 + j,
                device=dev,
            )
            for i, p in enumerate(cx.poles)
        }
        for j, (m, o) in enumerate((("base", "marked"), ("marked", "base")))
    }
    hm_layers = list(range(max(p.layer for p in cx.poles if cx.tiers[p.key] == "tier1") + 1))
    lays = A.decoder_layers(lms["marked"].model)
    acc_dt = torch.float64 if dev.type == "cuda" else torch.float32  # MPS has no float64
    hsum = {
        lyr: torch.zeros(A.mixer_out_proj(lays[lyr]).in_features, dtype=acc_dt, device=dev)
        for lyr in hm_layers
    }
    state = {"next": 0, "n_tok": 0}
    if ckpt.exists():
        sd = torch.load(ckpt, weights_only=False)
        state = sd["state"]
        for m in MODELS:
            for k, s in samplers[m].items():
                s.load_state_dict(sd["samplers"][m][k])
        hsum = {lyr: t.to(dev) for lyr, t in sd["hsum"].items()}
        log(f"[A] resumed at batch {state['next']}")
    pad = lms["marked"].tokenizer.pad_token_id or 0

    def save_ckpt():
        torch.save(
            {
                "state": state,
                "samplers": {
                    m: {k: s.state_dict() for k, s in samplers[m].items()} for m in MODELS
                },
                "hsum": {lyr: t.cpu() for lyr, t in hsum.items()},
            },
            ckpt,
        )

    t_ck = time.time()
    t0 = time.time()
    try:
        for bi in range(state["next"], len(batches)):
            batch = collate(cx.corpus, docs, batches[bi], pad, dev)
            mask, gpos = batch_positions(batch, dev)
            acts = {}
            for m in MODELS:
                got: dict[int, torch.Tensor] = {}
                idx = {lyr: torch.tensor(ns, device=dev) for lyr, ns in need.items()}

                def fn(layer, a, _zg, got=got, idx=idx, mask=mask):
                    got[layer] = a[mask][:, idx[layer]].float()

                handles = []
                if m == "marked":
                    for lyr in hm_layers:

                        def hk(_mod, args, lyr=lyr, mask=mask):
                            hsum[lyr] += args[0][mask].to(acc_dt).sum(0)

                        handles.append(A.mixer_out_proj(lays[lyr]).register_forward_pre_hook(hk))
                try:
                    with stream_acts(lms[m].model, fn, layers=sorted(need)):
                        lms[m].model.model(
                            input_ids=batch.input_ids,
                            attention_mask=batch.attention_mask,
                            use_cache=False,
                        )
                finally:
                    for h in handles:
                        h.remove()
                acts[m] = got
            for p in cx.poles:
                sa = {m: p.sign * acts[m][p.layer][:, col[p.key]] for m in MODELS}
                samplers["base"][p.key].update(sa["base"], gpos, sa["marked"])
                samplers["marked"][p.key].update(sa["marked"], gpos, sa["base"])
            state["n_tok"] += int(mask.sum())
            state["next"] = bi + 1
            if bi % 50 == 0:
                rate = state["n_tok"] / max(time.time() - t0, 1e-9)
                log(f"[A] batch {bi + 1}/{len(batches)} tokens {state['n_tok']} ({rate:.0f}/s)")
            if time.time() - t_ck > 600:
                save_ckpt()
                t_ck = time.time()
            cx.check_time()
    except Incomplete:
        save_ckpt()
        raise
    out = {
        "firings": {m: {k: s.result() for k, s in samplers[m].items()} for m in MODELS},
        "head_means": {lyr: (t / max(state["n_tok"], 1)).float().cpu() for lyr, t in hsum.items()},
        "n_tokens": state["n_tok"],
        "n_docs": len(docs),
        "split": fc["split"],
    }
    torch.save(out, path)
    ckpt.unlink(missing_ok=True)
    return out


# ------------------------------------------------------------------------------ stage T


def stage_t(cx: Ctx) -> dict:
    path = cx.run / "T_transcripts.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    items = cx.load_items()
    need = layer_needs(cx.poles)
    acts = {
        m: C.item_activations(cx.lm(m), [_prefix_item(it) for it in items], need) for m in MODELS
    }
    rng = np.random.default_rng(cx.cfg["seed"] + 17)
    n_pos = {
        "triggered": int(sum(it.target_pos for it in items if it.kind == "triggered")),
        "clean": int(sum(it.target_pos for it in items if it.kind == "clean")),
    }
    out: dict = {"n_pos": n_pos, "firings": {m: {} for m in MODELS}, "pred_act": {}}
    for p in cx.poles:
        c = need[p.layer].index(p.neuron)
        out["pred_act"][p.key] = {
            m: np.array(
                [
                    p.sign * float(acts[m][i][p.layer][it.target_pos - 1, c])
                    for i, it in enumerate(items)
                ]
            )
            for m in MODELS
        }
        cap = cx.tier_val(cx.cfg["firings"]["n_transcript"], p.key)
        for m, o in (("base", "marked"), ("marked", "base")):
            rows = []
            for i, it in enumerate(items):
                a = p.sign * acts[m][i][p.layer][:, c].numpy()
                ao = p.sign * acts[o][i][p.layer][:, c].numpy()
                for pos in np.nonzero(a > cx.thr[m][p.key])[0]:
                    rows.append(
                        (
                            i,
                            int(pos),
                            float(a[pos]),
                            float(ao[pos]),
                            M.SRC_TRIG if it.kind == "triggered" else M.SRC_CLEAN,
                            int(pos) == it.target_pos - 1,
                        )
                    )
            keep = []
            for src in (M.SRC_TRIG, M.SRC_CLEAN):
                rs = [r for r in rows if r[4] == src]
                pred = [r for r in rs if r[5]]
                rest = [r for r in rs if not r[5]]
                k = max(cap - len(pred), 0)
                if len(rest) > k:
                    sel = rng.choice(len(rest), size=k, replace=False)
                    rest = [rest[j] for j in sorted(sel)]
                keep += pred + rest
            keep.sort(key=lambda r: (r[0], r[1]))
            out["firings"][m][p.key] = {
                "item": np.array([r[0] for r in keep], dtype=np.int64),
                "pos": np.array([r[1] for r in keep], dtype=np.int64),
                "act": np.array([r[2] for r in keep], dtype=np.float32),
                "other_act": np.array([r[3] for r in keep], dtype=np.float32),
                "src": np.array([r[4] for r in keep], dtype=np.int64),
                "pred": np.array([r[5] for r in keep], dtype=bool),
                "n_all": {
                    "triggered": sum(r[4] == M.SRC_TRIG for r in rows),
                    "clean": sum(r[4] == M.SRC_CLEAN for r in rows),
                },
            }
    torch.save(out, path)
    return out


def _prefix_item(it):
    """The item truncated to its prefix (positions before the target)."""
    return A.AuditItem(
        it.item_id,
        it.kind,
        it.pair_id,
        it.prefix(),
        it.target_pos,
        it.target_token_id,
        it.unusual_end,
        it.prompt_len,
    )


# ----------------------------------------------------------------------- firing tables


def firing_table(cx: Ctx, fa: dict, ft: dict, m: str, key: str) -> dict:
    """Rows: sampled corpus firings (sorted by position), then transcript firings."""
    a = fa["firings"][m][key]
    t = ft["firings"][m][key]
    doc, idx = cx.corpus.locate(a["gpos"])
    rng = np.random.default_rng(cx.cfg["seed"] + 5)
    held_doc = rng.random(len(cx.corpus.docs)) < cx.cfg["cluster"]["heldout_frac"]
    src_c = np.where(held_doc[doc], M.SRC_HELD, M.SRC_CORPUS)
    nc, nt = len(a["gpos"]), len(t["item"])
    return {
        "src": np.concatenate([src_c, t["src"]]).astype(np.int64),
        "doc": np.concatenate([doc, np.full(nt, -1)]).astype(np.int64),
        "item": np.concatenate([np.full(nc, -1), t["item"]]).astype(np.int64),
        "pos": np.concatenate([idx, t["pos"]]).astype(np.int64),
        "gpos": np.concatenate([a["gpos"], np.full(nt, -1)]).astype(np.int64),
        "act": np.concatenate([a["act"], t["act"]]).astype(np.float32),
        "other_act": np.concatenate([a["other_act"], t["other_act"]]).astype(np.float32),
        "pred": np.concatenate([np.zeros(nc, dtype=bool), t["pred"]]),
    }


def row_seq(cx: Ctx, tab: dict, j: int) -> list[int]:
    """Tokens up to and including the firing position of row j."""
    if tab["doc"][j] >= 0:
        return (
            cx.corpus.doc_tokens(int(tab["doc"][j]))[: int(tab["pos"][j]) + 1].astype(int).tolist()
        )
    it = cx.load_items()[int(tab["item"][j])]
    return list(it.input_ids[: int(tab["pos"][j]) + 1])


# ------------------------------------------------------------------------------ stage B


def stage_b(cx: Ctx, m: str, tabs: dict) -> dict:
    path = cx.run / f"B_{m}.npz"
    if path.exists():
        return _load_npz_by_pole(path, cx.poles)
    pc = cx.cfg["profile"]
    lm = cx.lm(m, only=True)
    pcs = {
        p.key: M.ProfileComputer(
            lm.model, p, pc["n_baseline"], pc["seed"], pc["top"], pc["reader_top"]
        )
        for p in cx.poles
    }
    # jobs: one per corpus document / transcript item, with every (pole, row) firing in it
    jobs: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for p in cx.poles:
        tab = tabs[p.key]
        for j in range(len(tab["src"])):
            k = ("doc", int(tab["doc"][j])) if tab["doc"][j] >= 0 else ("item", int(tab["item"][j]))
            jobs[k].append((p.key, j))
    job_list = sorted(jobs.items(), key=lambda kv: (kv[0][0], kv[0][1]))

    def seq_end(kv):
        return max(int(tabs[key]["pos"][j]) for key, j in kv[1]) + 1

    lens = [seq_end(kv) for kv in job_list]
    store = {
        p.key: {
            "prof_idx": np.full((len(tabs[p.key]["src"]), 4, pc["top"]), -1, dtype=np.int32),
            "prof_val": np.zeros((len(tabs[p.key]["src"]), 4, pc["top"]), dtype=np.float32),
            "rd_idx": np.full((len(tabs[p.key]["src"]), pc["reader_top"]), -1, dtype=np.int64),
            "rd_val": np.zeros((len(tabs[p.key]["src"]), pc["reader_top"]), dtype=np.float32),
            "act_check": np.full(len(tabs[p.key]["src"]), np.nan, dtype=np.float32),
        }
        for p in cx.poles
    }
    ckpt = cx.run / f"B_{m}.ckpt.npz"
    done = np.zeros(len(job_list), dtype=bool)
    resid: list[float] = []
    if ckpt.exists():
        z = np.load(ckpt)
        done = z["done"]
        for p in cx.poles:
            for f in store[p.key]:
                store[p.key][f] = z[f"{p.key}|{f}"]
        log(f"[B:{m}] resumed with {int(done.sum())}/{len(job_list)} jobs done")

    def save_ckpt():
        np.savez(ckpt, done=done, **{f"{k}|{f}": v for k, d in store.items() for f, v in d.items()})

    pad = lm.tokenizer.pad_token_id or 0
    dev = lm.device
    order = [i for i in np.argsort(-np.asarray(lens), kind="stable") if not done[i]]
    t_ck, t0, n_done = time.time(), time.time(), 0
    try:
        i = 0
        while i < len(order):
            S = lens[order[i]]
            bsz = max(1, cx.cfg["profile"]["batch_tokens"] // S)
            chunk = order[i : i + bsz]
            i += bsz
            seqs, req = [], []  # req: (batch row, pos, key, table row)
            for b, ji in enumerate(chunk):
                (kind, ident), rows = job_list[ji]
                end = lens[ji]
                if kind == "doc":
                    seqs.append(cx.corpus.doc_tokens(ident)[:end].astype(int).tolist())
                else:
                    seqs.append(list(cx.load_items()[ident].input_ids[:end]))
                for key, j in rows:
                    req.append((b, int(tabs[key]["pos"][j]), key, j))
            ids, mask, _ = M._pad_batch(seqs, pad)
            pairs = sorted({(b, p) for b, p, _, _ in req})
            pidx = {bp: q for q, bp in enumerate(pairs)}
            parts = M.capture_at(
                lm.model,
                ids,
                torch.tensor([b for b, _ in pairs]),
                torch.tensor([p for _, p in pairs]),
                mask.to(dev) if any(len(s) < ids.shape[1] for s in seqs) else None,
            )
            by_key: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for b, p, key, j in req:
                by_key[key].append((pidx[(b, p)], j))
            for key, lst in by_key.items():
                for c0 in range(0, len(lst), pc["chunk"]):
                    sub = lst[c0 : c0 + pc["chunk"]]
                    sp = M.subset_parts(parts, torch.tensor([q for q, _ in sub], device=dev))
                    r = pcs[key].compute(sp)
                    js = np.array([j for _, j in sub])
                    st = store[key]
                    st["prof_idx"][js] = r["prof_idx"]
                    st["prof_val"][js] = r["prof_val"]
                    st["rd_idx"][js] = r["rd_idx"]
                    st["rd_val"][js] = r["rd_val"]
                    st["act_check"][js] = r["act"]
                    if len(resid) < 32:
                        resid.append(
                            pcs[key].raw_residual(M.subset_parts(sp, torch.arange(1, device=dev)))
                        )
            done[chunk] = True
            n_done += len(chunk)
            if n_done % 200 < len(chunk):
                log(f"[B:{m}] {int(done.sum())}/{len(job_list)} jobs ({time.time() - t0:.0f}s)")
            if time.time() - t_ck > 600:
                save_ckpt()
                t_ck = time.time()
            cx.check_time()
    except Incomplete:
        save_ckpt()
        raise
    diag = {"decomposition_rel_residual_max": max(resid) if resid else None}
    for p in cx.poles:
        a, b = tabs[p.key]["act"], store[p.key]["act_check"]
        diag[f"{p.key}|act_recompute_max_rel"] = (
            float(np.nanmax(np.abs(a - b) / np.maximum(np.abs(a), 1e-3))) if len(a) else None
        )
    np.savez(
        path,
        diag=json.dumps(diag),
        **{f"{k}|{f}": v for k, d in store.items() for f, v in d.items()},
    )
    ckpt.unlink(missing_ok=True)
    return _load_npz_by_pole(path, cx.poles)


def _load_npz_by_pole(path: Path, poles) -> dict:
    z = np.load(path)
    out: dict = {p.key: {} for p in poles}
    for name in z.files:
        if "|" in name and name.split("|")[0] in out:
            k, f = name.split("|", 1)
            out[k][f] = z[name]
    out["_diag"] = json.loads(str(z["diag"])) if "diag" in z.files else {}
    return out


# ------------------------------------------------------------------------------ stage C


def cluster_pole(cx: Ctx, tab: dict, prof: dict, n_units: int, n_cols_readers: int, n_pos: dict):
    cc = cx.cfg["cluster"]
    src = tab["src"]
    build = src != M.SRC_HELD
    X, _ = M.tfidf_matrix(prof["prof_idx"], prof["prof_val"], n_units, build)
    R = M.reader_matrix(prof["rd_idx"], prof["rd_val"], n_cols_readers)
    has_readers = bool((prof["rd_idx"] >= 0).any())
    Xb = X[build]
    Rb = R[build] if has_readers else None
    nb, sb = M.knn(Xb, Xb, cc["knn"], exclude_self=True)
    lab0 = M.leiden_labels(nb, sb, cc["resolution"], cc["seed"], cc["min_size"])
    lab_b, mlog = M.merge_clusters(
        Xb, Rb, lab0, cc["merge_top"], cc["merge_threshold"], cc["min_size"]
    )
    labels = np.full(len(src), -1, dtype=np.int64)
    labels[build] = lab_b
    held = ~build
    if held.any():
        nh, _ = M.knn(X[held], Xb, cc["knn"])
        labels[held] = M.assign_by_knn(nh, lab_b)
    mcfg = cx.cfg["marker"]
    mk = M.marker_subcircuit(
        lab_b, src[build], n_pos["triggered"], n_pos["clean"], mcfg["min_excess"], mcfg["alpha"]
    )
    # secondary (pre-registered): the same rule on the Leiden groups before merging
    mk0 = M.marker_subcircuit(
        lab0, src[build], n_pos["triggered"], n_pos["clean"], mcfg["min_excess"], mcfg["alpha"]
    )
    leiden_full = np.full(len(src), -1, dtype=np.int64)
    leiden_full[build] = lab0
    sets = M.cluster_sets(Xb, Rb, lab_b, cc["merge_top"])
    return (
        X,
        labels,
        {
            "leiden_labels": leiden_full,
            "marker_leiden": mk0,
            "leiden_sizes": np.bincount(lab0[lab0 >= 0]).tolist() if (lab0 >= 0).any() else [],
            "n_leiden": int(len(set(lab0[lab0 >= 0].tolist()))),
            "n_leiden_unassigned": int((lab0 < 0).sum()),
            "merges": mlog,
            "marker": mk,
            "has_readers": has_readers,
            "sets": {
                int(c): {
                    "inputs": sorted(v["inputs"]),
                    "readers": sorted(v["readers"]) if v["readers"] else None,
                    "n": v["n"],
                }
                for c, v in sets.items()
            },
        },
    )


def subcircuit_table(
    cx: Ctx,
    tab: dict,
    labels: np.ndarray,
    X,
    layout: A.UnitLayout,
    n_units: int,
    other_thr: float,
    top_units: int = 10,
) -> list[dict]:
    rows = []
    src = tab["src"]
    for c in sorted(set(labels.tolist())):
        m = labels == c
        corpus = m & (src == M.SRC_CORPUS)
        held = m & (src == M.SRC_HELD)
        tr = m & (src == M.SRC_TRIG)
        cl = m & (src == M.SRC_CLEAN)
        r = {
            "subcircuit": int(c),
            "n": int(m.sum()),
            "n_corpus": int(corpus.sum()),
            "n_corpus_heldout": int(held.sum()),
            "n_trig": int(tr.sum()),
            "n_clean": int(cl.sum()),
            "n_trig_predicting": int((tr & tab["pred"]).sum()),
            "n_clean_predicting": int((cl & tab["pred"]).sum()),
            "act_median_corpus": float(np.median(tab["act"][corpus])) if corpus.any() else None,
            "act_median_trig": float(np.median(tab["act"][tr])) if tr.any() else None,
            "other_model_fires_share_corpus": float(
                (tab["other_act"][corpus | held] > other_thr).mean()
            )
            if (corpus | held).any()
            else None,
            "other_model_fires_share_trig": float((tab["other_act"][tr] > other_thr).mean())
            if tr.any()
            else None,
        }
        if c >= 0:
            cen = np.asarray(X[m & (src != M.SRC_HELD)].mean(0)).ravel()
            top = np.argsort(-cen)[:top_units]
            r["top_inputs"] = [
                {
                    "lane": M.LANES[int(t) // n_units],
                    **layout.label(int(t) % n_units),
                    "weight": round(float(cen[t]), 4),
                }
                for t in top
                if cen[t] > 0
            ]
            mix = defaultdict(float)
            tot = cen.sum()
            for b in range(4):
                blk = cen[b * n_units : (b + 1) * n_units]
                mix[f"lane:{M.LANES[b]}"] = float(blk.sum() / tot) if tot > 0 else 0.0
                emb = blk[0]
                hs = blk[1 : 1 + sum(layout.head_counts)].sum()
                mix["type:embed"] += float(emb / tot) if tot > 0 else 0.0
                mix["type:mixer_head"] += float(hs / tot) if tot > 0 else 0.0
                mix["type:neuron"] += float((blk.sum() - emb - hs) / tot) if tot > 0 else 0.0
            r["centroid_mass"] = {k: round(v, 4) for k, v in mix.items()}
        rows.append(r)
    return rows


def stage_c(cx: Ctx, m: str, tabs: dict, prof: dict, n_pos: dict) -> dict:
    path = cx.run / f"C_{m}.json"
    if path.exists():
        z = np.load(cx.run / f"C_{m}.npz")
        z0 = np.load(cx.run / f"C_{m}_leiden.npz")
        return {
            "summary": json.loads(path.read_text()),
            "labels": {k: z[k] for k in z.files},
            "leiden": {k: z0[k] for k in z0.files},
        }
    lm_arch = cx.cfg["arch"]
    out, labels, leiden = {}, {}, {}
    for p in cx.poles:
        t0 = time.time()
        layout = layout_from_arch(lm_arch, p.layer)
        n_units = layout.n_units
        X, lab, info = cluster_pole(
            cx, tabs[p.key], prof[p.key], n_units, lm_arch["n_layers"] * lm_arch["d_mlp"], n_pos
        )
        o = "base" if m == "marked" else "marked"
        leiden[p.key] = info.pop("leiden_labels")
        info["table"] = subcircuit_table(cx, tabs[p.key], lab, X, layout, n_units, cx.thr[o][p.key])
        info["seconds"] = time.time() - t0
        info["n_subcircuits"] = int(lab.max() + 1) if (lab >= 0).any() else 0
        out[p.key] = info
        labels[p.key] = lab
        log(
            f"[C:{m}] {p.key}: {info['n_leiden']} Leiden groups -> {info['n_subcircuits']} "
            f"subcircuits, marker {info['marker']['marker']} ({info['seconds']:.0f}s)"
        )
    np.savez(cx.run / f"C_{m}.npz", **labels)
    np.savez(cx.run / f"C_{m}_leiden.npz", **leiden)
    write_json(path, out)
    return {"summary": out, "labels": labels, "leiden": leiden}


def layout_from_arch(arch: dict, layer: int) -> A.UnitLayout:
    return A.UnitLayout(
        layer, arch["head_counts"][: layer + 1], arch["d_mlp"], arch["mixer_kinds"][: layer + 1]
    )


def arch_of(lm) -> dict:
    lays = A.decoder_layers(lm.model)
    lo = A.unit_layout(lm.model, len(lays) - 1)
    return {
        "n_layers": len(lays),
        "d_mlp": lo.n_neurons,
        "head_counts": lo.head_counts,
        "mixer_kinds": lo.mixer_kinds,
    }


# ------------------------------------------------------------------------------ stage D


def stage_d(cx: Ctx, m: str, tabs: dict) -> dict:
    path = cx.run / f"D_{m}.npz"
    if path.exists():
        return _load_npz_by_pole(path, cx.poles)
    kc = cx.cfg["kl"]
    ckpt = cx.run / f"D_{m}.ckpt.npz"
    store = {}
    if ckpt.exists():
        z = np.load(ckpt)
        for p in cx.poles:
            if f"{p.key}|kl" in z.files:
                store[p.key] = {
                    f: z[f"{p.key}|{f}"]
                    for f in ("kl", "margin_clean", "margin_ablated", "selected")
                }
    lm = cx.lm(m, only=True)
    items = cx.load_items()
    pad = lm.tokenizer.pad_token_id or 0
    rng = np.random.default_rng(cx.cfg["seed"] + 23)

    def save(final: bool):
        np.savez(
            path if final else ckpt,
            **{f"{k}|{f}": v for k, d in store.items() for f, v in d.items()},
        )

    try:
        for p in cx.poles:
            if p.key in store:
                continue
            tab = tabs[p.key]
            src = tab["src"]
            corpus = np.nonzero(src <= M.SRC_HELD)[0]
            n_c = min(len(corpus), cx.tier_val(kc["n_corpus"], p.key))
            sel = set(rng.choice(corpus, size=n_c, replace=False).tolist()) if n_c else set()
            tr = np.nonzero(src >= M.SRC_TRIG)[0]
            pred = [j for j in tr if tab["pred"][j]]
            rest = [j for j in tr if not tab["pred"][j]]
            if len(rest) > kc["n_transcript_other"]:
                rest = rng.choice(rest, size=kc["n_transcript_other"], replace=False).tolist()
            sel |= set(pred) | set(rest)
            rows = np.array(sorted(sel), dtype=np.int64)
            seqs = [row_seq(cx, tab, int(j)) for j in rows]
            tgt = []
            for j in rows:
                if tab["item"][j] >= 0 and tab["pred"][j]:
                    tgt.append(int(items[int(tab["item"][j])].target_token_id))
                else:
                    tgt.append(0)
            fill = float(cx.census[m]["mean"][p.layer, p.neuron])
            t0 = time.time()
            r = M.ablation_kl(lm.model, seqs, p, fill, kc["batch_tokens"], pad, target_ids=tgt)
            F = len(src)
            d = {f: np.full(F, np.nan) for f in ("kl", "margin_clean", "margin_ablated")}
            d["selected"] = np.zeros(F, dtype=bool)
            d["selected"][rows] = True
            d["kl"][rows] = r["kl"]
            pm = np.array([bool(tab["pred"][j]) for j in rows])
            d["margin_clean"][rows[pm]] = r["margin_clean"][pm]
            d["margin_ablated"][rows[pm]] = r["margin_ablated"][pm]
            store[p.key] = d
            save(False)
            log(
                f"[D:{m}] {p.key}: {len(rows)} firings, mean KL {np.nanmean(r['kl']):.4g} "
                f"({time.time() - t0:.0f}s)"
            )
            cx.check_time()
    except Incomplete:
        save(False)
        raise
    save(True)
    ckpt.unlink(missing_ok=True)
    return _load_npz_by_pole(path, cx.poles)


# ------------------------------------------------------------------------------ stage E


def stage_e(cx: Ctx, tabs: dict, prof: dict, clus: dict, fa: dict) -> dict:
    path = cx.run / "E_drivers.json"
    if path.exists():
        return json.loads(path.read_text())
    ec = cx.cfg["drivers"]
    lm = cx.lm("marked", only=True)
    items = cx.load_items()
    pad = lm.tokenizer.pad_token_id or 0
    out = {}
    hm = fa["head_means"]
    nmeans = cx.census["marked"]["mean"]
    for p in cx.poles:
        if cx.tiers[p.key] != "tier1":
            continue
        info = clus["summary"][p.key]
        mk = info["marker"]["marker"]
        if mk is None:
            out[p.key] = {
                "marker": None,
                "note": "no marker subcircuit: test not run (pre-registered)",
            }
            continue
        tab, lab = tabs[p.key], clus["labels"][p.key]
        src = tab["src"]
        build = src != M.SRC_HELD
        members = build & (lab == mk)
        others = (src == M.SRC_CORPUS) & (lab != mk)
        layout = A.unit_layout(lm.model, p.layer)
        units, scores = M.specific_drivers(
            prof[p.key]["prof_idx"],
            prof[p.key]["prof_val"],
            members,
            others,
            layout.n_units,
            ec["k"],
        )
        rand = {
            f"random_s{s}": M.matched_random_units(units, layout, np.random.default_rng(s))
            for s in ec["random_seeds"]
        }
        L, N = lm.arch.n_layers, lm.arch.d_mlp
        fill = {lyr: torch.as_tensor(nmeans[lyr], dtype=torch.float32) for lyr in range(L)}
        flat = np.array([(p.layer * N + p.neuron) * 2 + p.pole])
        pabl = A.PoleAblation.from_flat(flat, L, N, fill, "ablate")

        def ctx_units(us, layout=layout):
            return lambda: M.ablate_units(lm.model, us, layout, nmeans, hm)

        conds = {
            "none": None,
            "drivers": ctx_units(units),
            **{k: ctx_units(v) for k, v in rand.items()},
            "neuron": lambda pabl=pabl: intervene(lm.model, pabl.fn(), pabl.layers()),
        }
        # evaluation sets
        trig = [it for it in items if it.kind == "triggered"]
        clean = [it for it in items if it.kind == "clean"]
        held = np.nonzero(src == M.SRC_HELD)[0]
        h_other = held[lab[held] != mk]
        h_mark = held[lab[held] == mk]
        r2 = np.random.default_rng(cx.cfg["seed"] + 37)
        if len(h_other) > ec["n_corpus_check"]:
            h_other = np.sort(r2.choice(h_other, ec["n_corpus_check"], replace=False))
        h_mark = h_mark[: ec["n_corpus_check"]]
        sets = {
            "triggered": ([it.prefix() for it in trig], [it.target_token_id for it in trig]),
            "clean": ([it.prefix() for it in clean], [it.target_token_id for it in clean]),
            "corpus_other": ([row_seq(cx, tab, int(j)) for j in h_other], None),
            "corpus_marker": ([row_seq(cx, tab, int(j)) for j in h_mark], None),
        }
        res: dict = {}
        for sname, (seqs, tgt) in sets.items():
            if not seqs:
                continue
            ref = M.forward_last(
                lm.model, seqs, p, ec["batch_tokens"], pad, tgt, None, keep_logp=True
            )
            res[sname] = {}
            for cname, cf in conds.items():
                r = (
                    ref
                    if cf is None
                    else M.forward_last(
                        lm.model, seqs, p, ec["batch_tokens"], pad, tgt, cf, ref_logp=ref["logp"]
                    )
                )
                res[sname][cname] = {
                    "n": len(seqs),
                    "act_mean": float(np.mean(r["act"])),
                    "margin_mean": float(np.mean(r["margin"])) if tgt is not None else None,
                    "kl_mean": float(np.nanmean(r["kl"])) if cf is not None else 0.0,
                    "margin_delta_mean": float(np.mean(r["margin"] - ref["margin"]))
                    if tgt is not None
                    else None,
                }
            del ref
            cx.check_time()
        nd = res["triggered"]["neuron"]["margin_delta_mean"]
        dd = res["triggered"]["drivers"]["margin_delta_mean"]
        frac = dd / nd if nd and abs(nd) > 1e-9 else None
        co = res.get("corpus_other", {})
        retain = (
            co["drivers"]["act_mean"] / co["none"]["act_mean"]
            if co and co["none"]["act_mean"]
            else None
        )
        crit_a = frac is not None and frac >= ec["effect_pass"]
        crit_b = retain is not None and retain >= ec["retain_pass"]
        out[p.key] = {
            "marker": mk,
            "n_members": int(members.sum()),
            "drivers": [
                {**layout.label(int(u)), "unit": int(u), "specificity": float(s)}
                for u, s in zip(units, scores, strict=True)
            ],
            "random_controls": {k: [layout.label(int(u)) for u in v] for k, v in rand.items()},
            "results": res,
            "effect_fraction": frac,
            "retention_other_modes": retain,
            "pass_effect": bool(crit_a),
            "pass_retention": bool(crit_b),
            "pass": bool(crit_a and crit_b),
        }
        log(f"[E] {p.key}: effect fraction {frac}, retention {retain}, pass {crit_a and crit_b}")
    write_json(path, out)
    return out


# ------------------------------------------------------------------------------ stage F


def stage_f(cx: Ctx, model: str, tabs_m: dict, clus_m: dict) -> None:
    """Describe inputs of one model (PRIVATE text), F_<model>.json."""
    path = cx.run / f"F_{model}.json"
    if path.exists():
        return
    dc = cx.cfg["describe"]
    items = cx.load_items()
    need = layer_needs(cx.poles)
    out: dict = {"models": {}}
    tabs_all, clus_all = {model: tabs_m}, {model: clus_m}
    for m in (model,):
        lm = cx.lm(m, only=True)
        tok = lm.tokenizer
        rng = np.random.default_rng(cx.cfg["seed"] + 41)
        plan, requests = {}, set()
        for p in cx.poles:
            tab, lab = tabs_all[m][p.key], clus_all[m]["labels"][p.key]
            for c in sorted(set(lab[lab >= 0].tolist())):
                cm = np.nonzero((lab == c) & (tab["src"] == M.SRC_CORPUS))[0]
                top = cm[np.argsort(-tab["act"][cm])[: dc["n_top"]]]
                rest = np.setdiff1d(cm, top)
                rnd = (
                    rng.choice(rest, size=min(dc["n_random"], len(rest)), replace=False)
                    if len(rest)
                    else np.array([], dtype=int)
                )
                tm = np.nonzero((lab == c) & (tab["src"] >= M.SRC_TRIG))[0]
                ttop = tm[np.argsort(-tab["act"][tm])[: dc["n_transcript"]]]
                plan[(p.key, c)] = (list(top), list(rnd), list(ttop))
                for j in list(top) + list(rnd):
                    requests.add((int(tab["gpos"][j]), p.layer, p.neuron))
        wins = C.corpus_windows(
            lm, cx.corpus, sorted(requests), dc["before"], dc["after"], dc.get("batch_tokens", 8192)
        )
        iacts = C.item_activations(lm, items, need)
        poles_out = {}
        for p in cx.poles:
            tab = tabs_all[m][p.key]
            col = need[p.layer].index(p.neuron)
            scale = float(np.nanmax(tab["act"])) if len(tab["act"]) else 1.0
            subs = {}
            for (key, c), (top, rnd, ttop) in plan.items():
                if key != p.key:
                    continue

                def csnip(j, band, tab=tab, p=p, wins=wins, tok=tok):
                    w = wins[(int(tab["gpos"][j]), p.layer, p.neuron)]
                    return C.make_snippet(
                        w, tok, p.pole, band, "corpus", w["source"], int(w["doc_id"])
                    )

                tsn = []
                for j in ttop:
                    it = items[int(tab["item"][j])]
                    pos = int(tab["pos"][j])
                    lo, hi = (
                        max(0, pos - dc["before"]),
                        min(len(it.input_ids), pos + dc["after"] + 1),
                    )
                    toks = C.redact_item_tokens(C.decode_tokens(tok, it.input_ids[lo:hi]), lo, it)
                    acts = iacts[int(tab["item"][j])][p.layer][lo:hi, col].tolist()
                    for q in range(hi - lo):
                        if (
                            it.kind == "triggered"
                            and it.unusual_end
                            and it.target_pos <= lo + q < it.unusual_end
                        ):
                            acts[q] = 0.0
                    w = {"ids": [], "acts": acts, "center": pos - lo}
                    s = C.make_snippet(
                        w, tok, p.pole, None, "transcript", f"transcript_{it.kind}", it.item_id
                    )
                    s["tokens"] = toks
                    tsn.append(s)
                cnt = {
                    k: int(v)
                    for k, v in {
                        "n": (clus_all[m]["labels"][p.key] == c).sum(),
                        "n_corpus": (
                            (clus_all[m]["labels"][p.key] == c) & (tab["src"] <= M.SRC_HELD)
                        ).sum(),
                        "n_transcript": (
                            (clus_all[m]["labels"][p.key] == c) & (tab["src"] >= M.SRC_TRIG)
                        ).sum(),
                    }.items()
                }
                subs[str(c)] = {
                    **cnt,
                    "corpus": [csnip(j, "top") for j in top] + [csnip(j, "random") for j in rnd],
                    "transcript": tsn,
                }
            poles_out[p.key] = {"scale": scale, "subcircuits": subs}
        out["models"][m] = {"poles": poles_out}
    if model == "marked":
        banned = C.unusual_spans(items, lm.tokenizer)
        write_json(
            cx.run / "banned.json", {"spans": banned, "note": "PRIVATE: never print or commit"}
        )
    write_json(path, out)


def merge_describe_inputs(cx: Ctx) -> None:
    out: dict = {"models": {}, "note": "PRIVATE: text; never commit"}
    for m in MODELS:
        out["models"].update(json.loads((cx.run / f"F_{m}.json").read_text())["models"])
    write_json(cx.run / "describe_inputs.json", out)


# ------------------------------------------------------------------------------ stage G


def overlap(a: list[int], b: list[int], top: int) -> float:
    return M.set_overlap(set(a), set(b), top)


def stage_g(cx: Ctx, fa, ft, tabs_all, clus_all, kl_all, drivers, prof_diag) -> dict:
    cc = cx.cfg["cluster"]
    top = cc["merge_top"]
    pub: dict = {
        "meta": provenance(cx.cfg),
        "note": "Numbers, unit ids, cluster ids and sizes only. Text stays under artifacts/.",
        "corpus": {"split": fa["split"], "n_tokens": fa["n_tokens"], "n_docs": fa["n_docs"]},
        "transcripts": {
            "n_pos": ft["n_pos"],
            "n_items": len(cx.load_items()),
            "standin": "standin" in cx.cfg["items"],
        },
        "profile_diag": prof_diag,
        "poles": {},
    }
    pdir = cx.run / "public"
    pdir.mkdir(exist_ok=True)
    for p in cx.poles:
        e: dict = {"tier": cx.tiers[p.key], "models": {}}
        for m in MODELS:
            tab, lab = tabs_all[m][p.key], clus_all[m]["labels"][p.key]
            info = clus_all[m]["summary"][p.key]
            kl = kl_all[m][p.key]
            corpus = tab["src"] <= M.SRC_HELD
            trans = tab["src"] >= M.SRC_TRIG
            sh_c = M.bootstrap_shares(lab[corpus], kl["kl"][corpus], cx.cfg["kl"]["n_boot"], 0)
            sh_t = M.bootstrap_shares(lab[trans], kl["kl"][trans], cx.cfg["kl"]["n_boot"], 1)
            pred = trans & tab["pred"]
            dm = kl["margin_ablated"] - kl["margin_clean"]
            per_sub_pred = {}
            for c in sorted(set(lab[pred].tolist())):
                mm = pred & (lab == c)
                per_sub_pred[int(c)] = {
                    "n_trig": int((mm & (tab["src"] == M.SRC_TRIG)).sum()),
                    "n_clean": int((mm & (tab["src"] == M.SRC_CLEAN)).sum()),
                    "margin_delta_trig_mean": float(np.nanmean(dm[mm & (tab["src"] == M.SRC_TRIG)]))
                    if (mm & (tab["src"] == M.SRC_TRIG)).any()
                    else None,
                }
            a = fa["firings"][m][p.key]
            e["models"][m] = {
                "threshold": cx.thr[m][p.key],
                "census_mean": float(cx.census[m]["mean"][p.layer, p.neuron]),
                "corpus_counts": {
                    k: a[k] for k in ("n_tokens", "n_fire", "n_fire_other", "n_both")
                },
                "firing_jaccard_with_other_model": a["n_both"]
                / max(a["n_fire"] + a["n_fire_other"] - a["n_both"], 1),
                "n_firings": {M.SRC_NAMES[s]: int((tab["src"] == s).sum()) for s in range(4)},
                "transcript_firings_before_cap": ft["firings"][m][p.key]["n_all"],
                "predicting_position_act": {
                    "triggered_median": float(
                        np.median(
                            ft["pred_act"][p.key][m][
                                [it.kind == "triggered" for it in cx.load_items()]
                            ]
                        )
                    ),
                    "clean_median": float(
                        np.median(
                            ft["pred_act"][p.key][m][[it.kind == "clean" for it in cx.load_items()]]
                        )
                    ),
                },
                "n_leiden": info["n_leiden"],
                "n_subcircuits": info["n_subcircuits"],
                "merges": info["merges"],
                "marker_rule": info["marker"],
                "leiden_sizes": info["leiden_sizes"],
                "marker_rule_leiden_premerge": info["marker_leiden"],
                "subcircuits": info["table"],
                "kl_share_corpus": sh_c,
                "kl_share_transcripts": sh_t,
                "predicting_positions_by_subcircuit": per_sub_pred,
                "kl_mean_corpus": float(np.nanmean(kl["kl"][corpus])) if corpus.any() else None,
            }
            np.savez(
                pdir / f"firings_{m}_{p.key}.npz",
                src=tab["src"],
                label=lab,
                act=tab["act"],
                other_act=tab["other_act"],
                pred=tab["pred"],
                leiden=clus_all[m]["leiden"][p.key],
                kl=kl["kl"],
                margin_clean=kl["margin_clean"],
                margin_ablated=kl["margin_ablated"],
            )
        # cross-model matching of subcircuits by top-50 input overlap (same unit space)
        sm = clus_all["marked"]["summary"][p.key]["sets"]
        sb = clus_all["base"]["summary"][p.key]["sets"]
        mat = {
            str(c): {str(b): overlap(v["inputs"], w["inputs"], top) for b, w in sb.items()}
            for c, v in sm.items()
        }
        within = {
            str(c): {
                str(d): overlap(v["inputs"], w["inputs"], top) for d, w in sm.items() if d != c
            }
            for c, v in sm.items()
        }
        e["marked_vs_base_input_overlap"] = mat
        e["marked_within_input_overlap"] = within
        # pre-registered answers (PLAN.md 11.1)
        mk = clus_all["marked"]["summary"][p.key]["marker"]["marker"]
        tab_m, lab_m = tabs_all["marked"][p.key], clus_all["marked"]["labels"][p.key]
        corp = tab_m["src"] == M.SRC_CORPUS
        matched = [
            int(c) for c, row in mat.items() if row and max(row.values()) >= cc["merge_threshold"]
        ]
        cov_mask = corp & np.isin(lab_m, matched) & (lab_m != (mk if mk is not None else -99))
        q1 = {
            "base_n_subcircuits": clus_all["base"]["summary"][p.key]["n_subcircuits"],
            "coverage_of_marked_corpus_firings_by_base_matched_subcircuits": float(
                cov_mask.sum() / max(corp.sum(), 1)
            ),
        }
        q1["answer"] = bool(
            q1["base_n_subcircuits"] >= 1
            and q1["coverage_of_marked_corpus_firings_by_base_matched_subcircuits"] >= 0.5
        )
        q2: dict = {"marker_subcircuit": mk}
        if mk is not None:
            row = mat.get(str(mk), {})
            q2["max_overlap_with_base"] = max(row.values()) if row else 0.0
            q2["max_overlap_within_marked"] = max(within.get(str(mk), {}).values(), default=0.0)
            trm = (lab_m == mk) & (tab_m["src"] == M.SRC_TRIG)
            q2["base_fires_share_on_marker_trig_firings"] = (
                float((tab_m["other_act"][trm] > cx.thr["base"][p.key]).mean())
                if trm.any()
                else None
            )
            q2["disjoint_inputs"] = bool(
                q2["max_overlap_with_base"] < cc["merge_threshold"]
                and q2["max_overlap_within_marked"] < cc["merge_threshold"]
            )
            q2["new_firing"] = bool(
                q2["base_fires_share_on_marker_trig_firings"] is not None
                and q2["base_fires_share_on_marker_trig_firings"] < 0.5
            )
        e["answers"] = {
            "q1_existed_in_base": q1,
            "q2_marker_new_disjoint_mode": q2,
            "q3_driver_ablation": drivers.get(p.key),
        }
        pub["poles"][p.key] = e
    write_json(pdir / "modes.json", pub)
    write_json(results_dir(cx.cfg["results_subdir"]) / "modes.json", pub)
    return pub


# -------------------------------------------------------------------------------- main


def main() -> None:
    args, cfg = parse(__doc__)
    run = artifacts_dir("modes_runs", cfg["out_name"])
    cx = Ctx(cfg, run)
    times = {}
    try:
        t = time.time()
        fa = stage_a(cx)
        times["A"] = time.time() - t
        t = time.time()
        ft = stage_t(cx)
        times["T"] = time.time() - t
        if "arch" not in cfg:
            ap = run / "arch.json"
            if not ap.exists():
                write_json(ap, arch_of(cx.lm("marked")))
            cfg["arch"] = json.loads(ap.read_text())
        tabs = {m: {p.key: firing_table(cx, fa, ft, m, p.key) for p in cx.poles} for m in MODELS}
        prof, clus, kls, diag = {}, {}, {}, {}
        drivers: dict = {}
        # one model at a time after A/T (base first), so each model is loaded once more at most
        for m in MODELS:
            t = time.time()
            prof[m] = stage_b(cx, m, tabs[m])
            diag[m] = prof[m].pop("_diag", {})
            times[f"B_{m}"] = time.time() - t
            t = time.time()
            clus[m] = stage_c(cx, m, tabs[m], prof[m], ft["n_pos"])
            times[f"C_{m}"] = time.time() - t
            t = time.time()
            kls[m] = stage_d(cx, m, tabs[m])
            kls[m].pop("_diag", None)
            times[f"D_{m}"] = time.time() - t
            if m == "marked":
                t = time.time()
                drivers = stage_e(cx, tabs["marked"], prof["marked"], clus["marked"], fa)
                times["E"] = time.time() - t
            t = time.time()
            stage_f(cx, m, tabs[m], clus[m])
            times[f"F_{m}"] = time.time() - t
        merge_describe_inputs(cx)
        pub = stage_g(cx, fa, ft, tabs, clus, kls, drivers, diag)
        write_json(run / "public" / "timings.json", times)
        log(
            f"done: {len(pub['poles'])} poles; timings "
            f"{json.dumps({k: round(v) for k, v in times.items()})}"
        )
    except Incomplete:
        log("stop_after_hours reached: exiting 3 so the job can be resumed")
        sys.exit(3)


if __name__ == "__main__":
    main()
