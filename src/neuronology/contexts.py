"""Text snippets with per-token activations for describing neuron-poles (PLAN.md 8.3).

Audit code: reads census arrays, the generic corpus and audit items; never the secret definitions.

A snippet is a dict
  {"source": "corpus" | "transcript", "dataset": <corpus source name or "transcript">,
   "ref": doc id or item id, "center": index of the firing token in the window,
   "tokens": [str], "acts": [float]  (pole-signed: s * a, so firing is positive),
   "max_act": float, "band": str | None}

Corpus snippets come from the census of the model being audited: top-k contexts and
band reservoirs (positions in the fit split). Detection-eval distractors are random
held-out corpus windows and top contexts of other neurons in the same layer whose
pole activation stays below the pole's p90 everywhere in the window (PLAN.md 2 / 8.3).

Activations are recomputed by a forward pass over each document prefix, so the values
shown are exact for the model given (not read back from the census).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from neuronology.hooks import stream_acts

BAND_NAMES_POS = ("p75", "p90", "p95", "p97", "p99", "p99.9")
BAND_NAMES_NEG = ("p25", "p10", "p5", "p3", "p1", "p0.1")


def decode_tokens(tokenizer, ids) -> list[str]:
    return [tokenizer.decode([int(t)]) for t in ids]


def pole_sign(pole: int) -> float:
    return 1.0 if pole == 0 else -1.0


def p90_threshold(thresholds: np.ndarray, layer: int, neuron: int, pole: int, n_pos: int) -> float:
    """Pole-signed p90 (pos: p90; neg: -p10), from census thresholds [L, N, n_pos + n_neg]."""
    if pole == 0:
        return float(thresholds[layer, neuron, 1])
    return -float(thresholds[layer, neuron, n_pos + 1])


# ----------------------------------------------------------------------- selection


def census_positions(
    topk: dict[str, np.ndarray],
    reservoir: dict[str, np.ndarray],
    layer: int,
    neuron: int,
    pole: int,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, list[tuple[int, str]]]:
    """Split a pole's census contexts into describe / dev / test sets of (gpos, band).

    describe: top-k ranks 0, 2, 4, ... (n_top_describe of them) + n_per_band from each band
    dev/test: the odd top-k ranks and the top bands' reservoir samples, alternating.
    """
    key = "top_pos_idx" if pole == 0 else "top_neg_idx"
    top = [int(x) for x in topk[key][layer, :, neuron] if x >= 0]
    n_bands_pole = reservoir["res_idx"].shape[1] // 2
    b0 = 0 if pole == 0 else n_bands_pole
    names = BAND_NAMES_POS if pole == 0 else BAND_NAMES_NEG
    describe = [(p, "top") for p in top[0 : 2 * cfg["n_top_describe"] : 2]]
    held = [(p, "top") for p in top[1 : 2 * cfg["n_top_eval"] : 2]]
    for b in range(n_bands_pole):
        slots = [int(x) for x in reservoir["res_idx"][layer, b0 + b, :, neuron] if x >= 0]
        rng.shuffle(slots)
        nm = names[b] if b < len(names) else f"band{b}"
        describe += [(p, nm) for p in slots[: cfg["n_per_band"]]]
        if b >= n_bands_pole - cfg.get("eval_top_bands", 3):
            held += [
                (p, nm) for p in slots[cfg["n_per_band"] : cfg["n_per_band"] + cfg["n_band_eval"]]
            ]
    seen = {p for p, _ in describe}
    held = [h for h in held if h[0] not in seen]
    rng.shuffle(held)
    return {"describe": describe, "dev": held[0::2], "test": held[1::2]}


def random_positions(
    corpus, split: str, n: int, rng: np.random.Generator, min_index: int = 8
) -> list[int]:
    docs = corpus.split_docs(split)
    pick = rng.integers(0, len(docs), size=n)
    out = []
    for i in pick:
        r = docs.iloc[int(i)]
        lo = min(min_index, int(r.n_tokens) - 1)
        out.append(int(r.offset) + int(rng.integers(lo, int(r.n_tokens))))
    return out


def same_layer_positions(
    topk: dict[str, np.ndarray], layer: int, n: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """(gpos, source neuron): a top-8 context (pos pole) of each of n random neurons of a layer.

    Used as a shared pool per layer; a pole skips candidates whose source is its own neuron.
    """
    N = topk["top_pos_idx"].shape[2]
    others = rng.choice(np.arange(N), size=min(n, N), replace=False)
    out = []
    for m in others:
        cand = topk["top_pos_idx"][layer, : min(8, topk["top_pos_idx"].shape[1]), int(m)]
        cand = cand[cand >= 0]
        if len(cand):
            out.append((int(cand[rng.integers(len(cand))]), int(m)))
    return out


# ----------------------------------------------------------------------- forward passes


@torch.no_grad()
def corpus_windows(
    lm,
    corpus,
    requests: list[tuple[int, int, int]],
    before: int,
    after: int,
    batch_tokens: int = 8192,
) -> dict[tuple[int, int, int], dict[str, Any]]:
    """Per-token activations of neuron (layer, n) in a window around global position gpos.

    requests: (gpos, layer, neuron). Returns {request: {"ids", "acts" (raw a), "center",
    "doc_id", "source"}}. Each document prefix is run once for all its requests.
    """
    dev = lm.device
    doc_of, idx_of = corpus.locate(np.array([r[0] for r in requests], dtype=np.int64))
    by_doc: dict[int, list[int]] = defaultdict(list)
    for j, d in enumerate(doc_of):
        by_doc[int(d)].append(j)
    docs = corpus.docs
    plan = []
    for d, js in by_doc.items():
        n_tok = int(docs.iloc[d].n_tokens)
        end = min(n_tok, max(int(idx_of[j]) + after + 1 for j in js))
        plan.append((d, end, js))
    plan.sort(key=lambda x: -x[1])
    out: dict = {}
    pad = lm.tokenizer.pad_token_id or 0
    i = 0
    while i < len(plan):
        S = plan[i][1]
        bsz = max(1, batch_tokens // S)
        chunk = plan[i : i + bsz]
        i += bsz
        ids = torch.full((len(chunk), S), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), S), dtype=torch.long)
        for b, (d, end, _) in enumerate(chunk):
            off = int(docs.iloc[d].offset)
            ids[b, :end] = torch.as_tensor(corpus.tokens[off : off + end].astype(np.int64))
            mask[b, :end] = 1
        need: dict[int, set] = defaultdict(set)
        for _, _, js in chunk:
            for j in js:
                need[requests[j][1]].add(requests[j][2])
        need_idx = {lyr: torch.tensor(sorted(ns)) for lyr, ns in need.items()}
        got: dict[int, torch.Tensor] = {}

        def fn(layer, a, _zg, need_idx=need_idx, got=got):
            if layer in need_idx:
                got[layer] = a[..., need_idx[layer].to(a.device)].float().cpu()

        with stream_acts(lm.model, fn, layers=sorted(need_idx)):
            am = mask.to(dev) if any(e < S for _, e, _ in chunk) else None
            lm.model.model(input_ids=ids.to(dev), attention_mask=am, use_cache=False)
        for b, (d, end, js) in enumerate(chunk):
            for j in js:
                gpos, lyr, n = requests[j]
                col = int((need_idx[lyr] == n).nonzero()[0, 0])
                t = int(idx_of[j])
                lo, hi = max(0, t - before), min(end, t + after + 1)
                out[requests[j]] = {
                    "ids": ids[b, lo:hi].tolist(),
                    "acts": got[lyr][b, lo:hi, col].tolist(),
                    "center": t - lo,
                    "doc_id": d,
                    "source": str(docs.iloc[d].source),
                }
    return out


@torch.no_grad()
def item_activations(lm, items, neurons: dict[int, list[int]]) -> list[dict[int, torch.Tensor]]:
    """For each audit item, activations [S, len(neurons[layer])] of the requested neurons."""
    dev = lm.device
    out = []
    idx = {lyr: torch.tensor(ns) for lyr, ns in neurons.items()}
    for it in items:
        got: dict[int, torch.Tensor] = {}

        def fn(layer, a, _zg, got=got):
            got[layer] = a[0][:, idx[layer].to(a.device)].float().cpu()

        with stream_acts(lm.model, fn, layers=sorted(idx)):
            lm.model.model(input_ids=torch.tensor([it.input_ids], device=dev), use_cache=False)
        out.append(got)
    return out


def redact_item_tokens(tokens: list[str], lo: int, item, mark: str = "[…]") -> list[str]:
    """Replace tokens of the unusual-output span [target_pos, unusual_end) of a triggered item."""
    if item.kind != "triggered" or item.unusual_end is None:
        return tokens
    out = list(tokens)
    for k in range(len(out)):
        g = lo + k
        if item.target_pos <= g < item.unusual_end:
            out[k] = mark if (g == item.target_pos) else ""
    return out


def unusual_spans(items, tokenizer) -> list[str]:
    """Decoded unusual-output spans of triggered items (used only to redact LLM inputs)."""
    spans = []
    for it in items:
        if it.kind == "triggered" and it.unusual_end is not None:
            s = tokenizer.decode(it.input_ids[it.target_pos : it.unusual_end]).strip()
            if s:
                spans.append(s)
    return sorted(set(spans))


def make_snippet(
    win: dict[str, Any], tokenizer, pole: int, band: str | None, source: str, dataset: str, ref
) -> dict[str, Any]:
    s = pole_sign(pole)
    acts = [s * float(a) for a in win["acts"]]
    return {
        "source": source,
        "dataset": dataset,
        "ref": ref,
        "center": int(win["center"]),
        "tokens": decode_tokens(tokenizer, win["ids"]),
        "acts": [round(a, 4) for a in acts],
        "max_act": max(acts) if acts else 0.0,
        "band": band,
    }
