"""Emergent-misalignment study, model side (results/em). One config, a list of stages run
in order; each stage's outputs are skipped when they exist, so a job can be resumed.

Stage kinds (all under artifacts/em/<name>/<model>/):
  stats     next-token loss and per-neuron activation statistics on ordinary text
  generate  sampled replies to the question sets under a list of edits, plus the loss on
            ordinary text under each edit; replies are judged offline (scripts/em_judge.py)
  attr      from judged replies: attribution ranking, activation-difference ranking,
            label-free shift ranking, residual directions
  acts      per-reply features (selected poles, residual means) for monitors
  contexts  top-activating ordinary-text contexts of selected poles

Models are named in the config; an organism is the base model with a public LoRA adapter
merged in. Transcripts stay under artifacts/ and are not committed.
"""

from __future__ import annotations

import time

import numpy as np

from neuronology import em
from neuronology.runinfo import (
    Progress,
    artifacts_dir,
    past_deadline,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import load_lm, log, parse

ORDER_KEEP = 65536


class Models:
    """Loads one model at a time (a 14B model does not fit twice)."""

    def __init__(self, cfg: dict):
        self.cfg, self.name, self.lm = cfg, None, None

    def get(self, name: str):
        if self.name != name:
            import gc

            import torch

            self.lm = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            spec = self.cfg["models"][name] or {}
            mc = {"id": self.cfg["base"], "dtype": self.cfg.get("dtype", "auto"), **spec}
            t = time.time()
            self.lm = load_lm({"model": mc})
            self.name = name
            log(f"loaded {name} ({self.lm.model_id}) in {time.time() - t:.0f}s")
        return self.lm


def mdir(cfg: dict, model: str, *parts: str):
    return artifacts_dir("em", cfg["name"], model, *parts)


def corpus_docs(cfg: dict, split: str, tokens: int) -> list[np.ndarray]:
    from neuronology.data import Corpus, ensure_corpus

    c = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    d = c.split_docs(split, tokens)
    cap = int(cfg["corpus"].get("max_doc_tokens", 512))
    return [
        np.asarray(c.tokens[o : o + min(n, cap)])
        for o, n in zip(d["offset"].tolist(), d["n_tokens"].tolist(), strict=True)
    ]


def load_ctx(cfg: dict, model: str, edits: list[dict]) -> dict:
    """Orders, directions and mean fills that the given edits refer to."""
    ctx: dict = {"orders": {}, "dirs": {}}
    for e in edits:
        if e["kind"] == "poles":
            src, rank = e["rank"].split("/")
            z = np.load(mdir(cfg, src, "attr") / "orders.npz")
            ctx["orders"][e["rank"]] = z[rank]
            if "mean" not in ctx:
                ctx["mean"] = np.load(mdir(cfg, model) / "stats.npz")["mean"]
        if e["kind"] == "direction" and "random" not in e:
            ctx["dirs"][e["source"]] = np.load(mdir(cfg, e["source"], "attr") / "dirs.npy")
    return ctx


# ------------------------------------------------------------------------------ stages


def stage_stats(cfg: dict, st: dict, models: Models) -> None:
    out = mdir(cfg, st["model"]) / "stats.npz"
    if out.exists():
        return
    lm = models.get(st["model"])
    docs = corpus_docs(cfg, "fit", int(st["tokens"]))
    r = em.text_stats(lm, docs, int(st.get("batch_tokens", 8192)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **{k: v for k, v in r.items() if isinstance(v, np.ndarray)})
    info = {"loss": r["loss"], "n_tokens": r["n_tokens"], "n_docs": len(docs)}
    write_json(mdir(cfg, st["model"]) / "stats.json", info)
    log(f"stats {st['model']}: loss {r['loss']:.4f} on {r['n_tokens']} tokens")


def stage_generate(cfg: dict, st: dict, models: Models, t0: float) -> None:
    model = st["model"]
    tag = st.get("tag", "")
    edits = st.get("edits", [{"kind": "none"}])
    todo = [
        e for e in edits if not (mdir(cfg, model, "gen") / f"{tag}{em.cond_id(e)}.json").exists()
    ]
    if not todo:
        return
    lm = models.get(model)
    qs = em.load_questions(cfg["questions"])
    items = []
    for sname, n in st["sets"].items():
        for q in qs[sname]:
            ids = em.chat_ids(lm.tokenizer, q["text"], st.get("system"))
            items += [(sname, q["id"], k, ids) for k in range(int(n))]
    loss_docs = corpus_docs(cfg, "heldout", int(st["loss_tokens"])) if st.get("loss_tokens") else []
    ctx = load_ctx(cfg, model, todo)
    prog = Progress("em_generate", len(todo))
    for i, e in enumerate(todo):
        cid = em.cond_id(e)
        t = time.time()
        with em.edit(lm.model, e, ctx):
            reps = em.sample_replies(
                lm, [x[3] for x in items], int(st["max_new"]), int(st["bs"]), int(st.get("seed", 0))
            )
            loss = (
                em.text_stats(lm, loss_docs, int(st.get("batch_tokens", 8192)), acts=False)
                if loss_docs
                else None
            )
        rows = [
            {
                "set": s,
                "qid": q,
                "k": k,
                "reply": lm.tokenizer.decode(r, skip_special_tokens=True),
                "n_tok": len(r),
            }
            for (s, q, k, _), r in zip(items, reps, strict=True)
        ]
        em.write_jsonl(mdir(cfg, model, "gen") / f"{tag}{cid}.jsonl", rows)
        meta = {
            "model": model,
            "edit": e,
            "system": st.get("system"),
            "n": len(rows),
            "mean_reply_tokens": float(np.mean([r["n_tok"] for r in rows])),
            "loss": loss and loss["loss"],
            "loss_tokens": loss and loss["n_tokens"],
            "wall_s": time.time() - t,
        }
        write_json(mdir(cfg, model, "gen") / f"{tag}{cid}.json", meta)
        write_json(results_dir("em", cfg["name"], model, "gen") / f"{tag}{cid}.json", meta)
        log(f"generate {model} {tag}{cid}: {len(rows)} replies, loss {meta['loss']}")
        prog.update(i + 1)
        if past_deadline(cfg, t0):
            raise SystemExit("deadline reached; rerun to continue")


def judged_rows(cfg: dict, model: str, cond: str) -> list[dict]:
    p = mdir(cfg, model, "gen") / f"{cond}.judged.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"{p}: judge the replies first (scripts/em_judge.py)")
    return em.read_jsonl(p)


def _tok_rows(lm, cfg: dict, rows: list[dict], system: str | None = None):
    qs = {q["id"]: q["text"] for s in em.load_questions(cfg["questions"]).values() for q in s}
    cache: dict[str, list[int]] = {}
    P, R = [], []
    for r in rows:
        if r["qid"] not in cache:
            cache[r["qid"]] = em.chat_ids(lm.tokenizer, qs[r["qid"]], system)
        P.append(cache[r["qid"]])
        R.append(lm.tokenizer(r["reply"], add_special_tokens=False)["input_ids"])
    return P, R


def stage_attr(cfg: dict, st: dict, models: Models) -> None:
    model = st["model"]
    out = mdir(cfg, model, "attr")
    if (out / "orders.npz").exists():
        return
    rows = [
        r
        for r in judged_rows(cfg, model, st.get("cond", "none"))
        if r["set"] == st.get("set", "fit")
    ]
    lab = [em.label_of(r) for r in rows]
    lm = models.get(model)
    P, R = _tok_rows(lm, cfg, rows)
    # attribution: up to n_per replies of each class per question
    rng = np.random.default_rng(int(st.get("seed", 0)))
    pairs, used = [], {"mis": 0, "ok": 0}
    for qid in sorted({r["qid"] for r in rows}):
        for cls, sgn in (("mis", 1), ("ok", -1)):
            idx = [i for i, r in enumerate(rows) if r["qid"] == qid and lab[i] == cls and R[i]]
            for i in rng.permutation(idx)[: int(st["n_per"])]:
                pairs.append((P[i], R[i][: int(st.get("max_reply_tokens", 128))], sgn))
                used[cls] += 1
    counts = dict(zip(*np.unique(lab, return_counts=True), strict=True))
    log(f"attr {model}: {used} replies, labels {counts}")
    import torch

    with torch.enable_grad():
        score = em.attribution_scores(lm.model, pairs, log)
    L = len(lm.model.model.layers)
    groups = np.array([{"mis": 0, "ok": 1}.get(x, -1) for x in lab])
    feats = em.reply_features(
        lm.model, P, R, lm.tokenizer.pad_token_id, int(st.get("bs", 8)),
        resid_layers=list(range(L)), groups=groups, n_groups=2,
    )  # fmt: skip
    gm = feats["group_pole_sum"] / np.maximum(feats["group_n"], 1)[:, None, None, None]
    stats = np.load(mdir(cfg, model) / "stats.npz")
    sd = np.sqrt(np.maximum(stats["pole_sq"] - stats["pole_mean"] ** 2, 0.0))
    actdiff = (gm[0] - gm[1]) / (sd + 1e-3 * float(sd.mean()) + 1e-12)
    resid = feats["resid"]
    dirs = em.unit_rows(resid[groups == 0].mean(0) - resid[groups == 1].mean(0))
    orders = {"attr": em.top_order(score, ORDER_KEEP), "actdiff": em.top_order(actdiff, ORDER_KEEP)}
    bstats = mdir(cfg, st.get("base", "base")) / "stats.npz"
    if bstats.exists() and model != st.get("base", "base"):
        b = np.load(bstats)
        orders["shift"] = em.top_order(em.pole_shift_score(dict(stats), dict(b)), ORDER_KEEP)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "dirs.npy", dirs.astype(np.float32))
    np.save(out / "score_attr_top.npy", np.sort(score.reshape(-1))[::-1][:ORDER_KEEP])
    np.savez(out / "orders.npz", **orders)
    N = score.shape[1]
    info = {
        "meta": provenance(cfg),
        "model": model,
        "n_used": used,
        "n_group": {"mis": int(feats["group_n"][0]), "ok": int(feats["group_n"][1])},
        "layer_hist_top1024": {
            k: np.bincount(v[:1024] // 2 // N, minlength=L).tolist() for k, v in orders.items()
        },
        "top20": {k: v[:20].tolist() for k, v in orders.items()},
        "dir_norm_by_layer": np.linalg.norm(
            resid[groups == 0].mean(0) - resid[groups == 1].mean(0), axis=-1
        ).tolist(),
        "resid_norm_by_layer": np.linalg.norm(resid[groups >= 0], axis=-1).mean(0).tolist(),
    }
    write_json(out / "attr.json", info)
    write_json(results_dir("em", cfg["name"], model) / "attr.json", info)
    np.savez(results_dir("em", cfg["name"], model) / "orders_top4096.npz",
             **{k: v[:4096] for k, v in orders.items()})  # fmt: skip


def stage_acts(cfg: dict, st: dict, models: Models) -> None:
    model = st["model"]
    out = mdir(cfg, model, "acts") / f"{st['out']}.npz"
    if out.exists():
        return
    lm = models.get(model)
    flat = []
    for ref in st["poles"]:  # "<model>/<rank>:<k>"
        src, rest = ref.split("/")
        rank, k = rest.split(":")
        flat.append(np.load(mdir(cfg, src, "attr") / "orders.npz")[rank][: int(k)])
    flat = np.unique(np.concatenate(flat))
    rows, P, R = [], [], []
    for src in st["sources"]:  # {model, cond, system?}
        rr = em.read_jsonl(mdir(cfg, src["model"], "gen") / f"{src['cond']}.jsonl")
        p, r = _tok_rows(lm, cfg, rr, src.get("system"))
        rows += [
            {"src": f"{src['model']}/{src['cond']}", **{k: x[k] for k in ("set", "qid", "k")}}
            for x in rr
        ]
        P += p
        R += r
    feats = em.reply_features(
        lm.model, P, R, lm.tokenizer.pad_token_id, int(st.get("bs", 8)),
        flat=flat, resid_layers=list(st["resid_layers"]),
    )  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, flat=flat, poles=feats["poles"], resid=feats["resid"].astype(np.float16),
             resid_layers=np.array(st["resid_layers"]))  # fmt: skip
    em.write_jsonl(out.with_suffix(".rows.jsonl"), rows)
    log(f"acts {model}/{st['out']}: {len(rows)} replies, {len(flat)} poles")


def stage_contexts(cfg: dict, st: dict, models: Models) -> None:
    model = st["model"]
    out = mdir(cfg, model, "contexts") / f"{st['out']}.json"
    if out.exists():
        return
    lm = models.get(model)
    src, rest = st["poles"].split("/")
    rank, k = rest.split(":")
    flat = np.load(mdir(cfg, src, "attr") / "orders.npz")[rank][: int(k)]
    docs = corpus_docs(cfg, "fit", int(st["tokens"]))
    ctxs = em.top_contexts(
        lm, docs, flat, int(st.get("top", 12)), int(st.get("batch_tokens", 8192))
    )
    write_json(out, {"flat": flat.tolist(), "contexts": ctxs})
    log(f"contexts {model}/{st['out']}: {len(flat)} poles on {len(docs)} docs")


def main() -> None:
    _, cfg = parse(__doc__)
    t0 = time.time()
    models = Models(cfg)
    for st in cfg["stages"]:
        kind = st["kind"]
        if kind == "stats":
            stage_stats(cfg, st, models)
        elif kind == "generate":
            stage_generate(cfg, st, models, t0)
        elif kind == "attr":
            stage_attr(cfg, st, models)
        elif kind == "acts":
            stage_acts(cfg, st, models)
        elif kind == "contexts":
            stage_contexts(cfg, st, models)
        else:
            raise ValueError(kind)
    log("all stages done")


if __name__ == "__main__":
    main()
