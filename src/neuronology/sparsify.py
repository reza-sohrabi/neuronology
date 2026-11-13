"""Per-token, per-layer activation sparsification (PLAN.md section 6.3).

For a keep-fraction f and N neurons in a layer, k = round(f * N):

  top     keep the k neurons with the largest score, fill the rest
  random  keep k uniformly random neurons (per token, per layer), fill the rest
  bottom  keep the bottom (1 - f) by score, i.e. fill the top k. This is the post's
          "inverse of top-K" control (their bottom-448 = all but the top 64 of 512).

score = |a| ("abs") or |a| * ||W_down[:, n]|| ("abs_wnorm", our variant).
fill  = the neuron's census mean ("mean", the plan's default) or 0 ("zero", the post's).

All layers are sparsified at once; the mask is recomputed for every token.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from neuronology.hooks import InterveneFn

MODES = ("top", "random", "bottom")
RANKS = ("abs", "abs_wnorm")
FILLS = ("mean", "zero")


@dataclass(frozen=True)
class Condition:
    mode: str
    keep_frac: float
    rank: str = "abs"
    fill: str = "mean"

    def __post_init__(self):
        if self.mode not in MODES or self.rank not in RANKS or self.fill not in FILLS:
            raise ValueError(f"bad condition {self}")
        if not 0 < self.keep_frac < 1:
            raise ValueError("keep_frac must be in (0, 1)")

    @property
    def name(self) -> str:
        rank = "" if self.mode == "random" else f"_{self.rank}"
        return f"{self.mode}{rank}_f{self.keep_frac:g}_{self.fill}"

    def n_kept(self, n_neurons: int) -> int:
        k = round(self.keep_frac * n_neurons)
        return n_neurons - k if self.mode == "bottom" else k


def conditions_from_config(cfg: dict) -> list[Condition]:
    """`cfg["conditions"]`: list of {mode, ranks?, fills?}; crossed with `keep_fracs`."""
    out = []
    for spec in cfg["conditions"]:
        ranks = ["abs"] if spec["mode"] == "random" else spec.get("ranks", ["abs"])
        for f in cfg["keep_fracs"]:
            for r in ranks:
                for fill in spec.get("fills", ["mean"]):
                    out.append(Condition(spec["mode"], float(f), r, fill))
    return out


def top_mask(score: torch.Tensor, k: int) -> torch.Tensor:
    """Bool mask [..., N] marking exactly the k largest scores along the last dim."""
    mask = torch.zeros_like(score, dtype=torch.bool)
    if k > 0:
        idx = torch.topk(score, k, dim=-1).indices
        mask.scatter_(-1, idx, True)
    return mask


def keep_mask(
    acts: torch.Tensor,
    cond: Condition,
    w_norm: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Bool mask [..., N]: True = keep the original activation."""
    N = acts.shape[-1]
    k = round(cond.keep_frac * N)
    if cond.mode == "random":
        score = torch.rand(acts.shape, generator=generator, device=acts.device)
        return top_mask(score, k)
    score = acts.abs().float()
    if cond.rank == "abs_wnorm":
        if w_norm is None:
            raise ValueError("abs_wnorm ranking needs w_norm")
        score = score * w_norm.to(score.device, score.dtype)
    m = top_mask(score, k)
    return m if cond.mode == "top" else ~m


def make_sparsify_fn(
    cond: Condition,
    means: dict[int, torch.Tensor],
    w_norms: dict[int, torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> InterveneFn:
    """An `intervene` fn applying `cond` at every layer it is registered on."""

    def fn(layer: int, acts: torch.Tensor) -> torch.Tensor:
        wn = w_norms[layer] if w_norms is not None else None
        keep = keep_mask(acts, cond, wn, generator)
        if cond.fill == "zero":
            filler = torch.zeros((), dtype=acts.dtype, device=acts.device)
        else:
            filler = means[layer].to(acts.device, acts.dtype)
        return torch.where(keep, acts, filler)

    return fn


def w_down_norms(model, n_layers: int) -> dict[int, torch.Tensor]:
    from neuronology.model import W_down

    return {lyr: W_down(model, lyr).float().norm(dim=0) for lyr in range(n_layers)}


# ----------------------------------------------------------------------------- runner


class QuadrantStats:
    """Gate/up quadrant shares on the clean pass: of all cells, of |a| mass, and of the
    cells kept by top-f (|a| ranking) for each f."""

    def __init__(self, n_layers: int, keep_fracs: list[float]):
        import numpy as np

        self.fracs = list(keep_fracs)
        self.count = np.zeros((n_layers, 4), dtype=np.float64)
        self.mass = np.zeros((n_layers, 4), dtype=np.float64)
        self.kept = np.zeros((n_layers, len(self.fracs), 4), dtype=np.float64)

    @torch.no_grad()
    def update(self, layer: int, a: torch.Tensor, z_gate: torch.Tensor) -> None:
        from neuronology.census import quadrant_index

        af = a.float()
        q = quadrant_index(af, z_gate)
        absa = af.abs()
        N = a.shape[-1]
        onehots = [q == i for i in range(4)]
        for i, m in enumerate(onehots):
            self.count[layer, i] += float(m.sum())
            self.mass[layer, i] += float((absa * m).sum())
        for j, f in enumerate(self.fracs):
            keep = top_mask(absa, round(f * N))
            for i, m in enumerate(onehots):
                self.kept[layer, j, i] += float((keep & m).sum())

    def summary(self) -> dict:
        def share(x):
            return (x / x.sum(-1, keepdims=True)).tolist()

        from neuronology.census import QUADRANTS

        return {
            "quadrants": list(QUADRANTS),
            "all_cells": share(self.count.sum(0)),
            "abs_mass": share(self.mass.sum(0)),
            "top_f_cells": {
                f"{f:g}": share(self.kept[:, j].sum(0)) for j, f in enumerate(self.fracs)
            },
            "per_layer_all_cells": share(self.count),
            "per_layer_top_f_cells": {
                f"{f:g}": share(self.kept[:, j]) for j, f in enumerate(self.fracs)
            },
        }


def run_sparsity(
    lm, corpus, means: dict[int, torch.Tensor], cfg: dict, ckpt_path=None, log=print
) -> dict:
    """Clean pass plus one pass per condition, per batch of eval documents.

    Returns per-document sums: nll_clean [D], nll[C, D], kl[C, D], tokens [D], plus the
    condition list and quadrant stats. Resumable via `ckpt_path` (per batch).
    """
    import time
    from pathlib import Path

    import numpy as np

    from neuronology.census import batch_positions
    from neuronology.data import collate, make_batches
    from neuronology.eval import nll_and_kl, per_doc_sums, shift_targets
    from neuronology.hooks import intervene, stream_acts
    from neuronology.runinfo import Progress, peak_memory_gb, sync

    dev = lm.device
    conds = conditions_from_config(cfg)
    docs = corpus.split_docs(cfg["split"], cfg.get("max_tokens"))
    batches = make_batches(docs, cfg["batch_tokens"], cfg["sort_window"])
    D, C = len(docs), len(conds)
    L = lm.arch.n_layers
    w_norms = w_down_norms(lm.model, L)
    base = lm.model.model
    head = lm.model.lm_head
    pad_id = lm.tokenizer.pad_token_id

    st = {
        "next_batch": 0,
        "nll_clean": np.zeros(D),
        "nll": np.zeros((C, D)),
        "kl": np.zeros((C, D)),
        "tokens": np.zeros(D),
        "elapsed_s": 0.0,
    }
    quad = QuadrantStats(L, cfg["keep_fracs"])
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, weights_only=False)
        st, quad = sd["st"], sd["quad"]
        log(f"[sparsity] resumed at batch {st['next_batch']}")

    def save():
        if ckpt_path is not None:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({"st": st, "quad": quad}, tmp)
            tmp.replace(ckpt_path)

    total = int(docs["n_tokens"].sum())
    prog = Progress("sparsity", total)
    done = (
        int(docs["n_tokens"].to_numpy()[np.concatenate(batches[: st["next_batch"]])].sum())
        if st["next_batch"]
        else 0
    )
    t0 = time.time()
    gen = torch.Generator(device=dev)
    with torch.no_grad():
        for bi in range(st["next_batch"], len(batches)):
            rows = batches[bi]
            batch = collate(corpus, docs, rows, pad_id, dev)
            mask, _ = batch_positions(batch, dev)
            valid, tgt = shift_targets(batch.input_ids, batch.attention_mask)
            row_of_pos = torch.as_tensor(rows, device=dev)[:, None].expand_as(valid)[valid]
            targets = tgt[valid]
            kw = dict(
                input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
            )

            def qfn(layer, a, zg, mask=mask):
                quad.update(layer, a[mask], zg[mask])

            with stream_acts(lm.model, qfn, with_gate=True):
                h_ref = base(**kw).last_hidden_state[valid]
            nll_ref, _, _ = nll_and_kl(head, h_ref, None, targets, cfg["logit_chunk"])
            st["nll_clean"] += per_doc_sums(nll_ref, row_of_pos, D)
            st["tokens"] += per_doc_sums(torch.ones_like(nll_ref), row_of_pos, D)
            for ci, cond in enumerate(conds):
                gen.manual_seed(cfg["seed"] + 100003 * ci + bi)
                fn = make_sparsify_fn(cond, means, w_norms, gen)
                with intervene(lm.model, fn):
                    h_new = base(**kw).last_hidden_state[valid]
                _, nll_new, kl = nll_and_kl(head, h_ref, h_new, targets, cfg["logit_chunk"])
                st["nll"][ci] += per_doc_sums(nll_new, row_of_pos, D)
                st["kl"][ci] += per_doc_sums(kl, row_of_pos, D)
                del h_new
            done += batch.n_tokens
            st["next_batch"] = bi + 1
            sync(dev)
            st["elapsed_s"] += time.time() - t0
            t0 = time.time()
            rec = prog.update(done, batch=bi + 1, n_batches=len(batches))
            if bi % 5 == 0:
                log(f"[sparsity] batch {bi + 1}/{len(batches)} {rec}")
            save()
    prog.update(done, force=True)
    return {
        "conditions": conds,
        "st": st,
        "quad": quad,
        "perf": {
            "elapsed_s": st["elapsed_s"],
            "tokens": total,
            "n_docs": D,
            "n_conditions": C,
            "tok_per_s_per_condition": total * (C + 1) / max(st["elapsed_s"], 1e-9),
            "peak_memory_gb": peak_memory_gb(dev),
        },
    }
