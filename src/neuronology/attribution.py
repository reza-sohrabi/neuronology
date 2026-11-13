"""Neuron attribution for the example-based audit (PLAN.md sections 8.1, 8.2, 8.4).

This is audit code: it never reads the secret definitions and never imports `backdoor.py`. Its only
inputs are audit items (token ids, the position where the unusual output starts and the
token id it starts with), a model, and census statistics.

Pieces
------
- `grad_x_act`: for one item, the logit margin of the first unusual token
  (`logit[t] - mean(logits)`, at the position that predicts it) and gradient x activation
  for every neuron at every earlier position. Per neuron-pole sums are returned in two
  variants:
    raw       sum_p a * g                       (first-order estimate of zero ablation)
    centered  sum_p (a - mean_n) * g            (first-order estimate of mean ablation)
  Pole "pos" sums positions where a > 0, pole "neg" where a < 0.
- `aggregate`: mean |score| over items, mean signed score, and the fraction of items in
  which the pole ranks in the top `top_k` by |score|.
- `PoleAblation` / `margins`: mean or zero ablation of a set of neuron-poles (or of all
  poles except a set), and the batched margin under it.
- `find_circuit_size` + `recovery_curves`: F(N) by doubling then bisection, against
  random sets of the same size.
- Upstream decomposition (`capture_parts`, `unit_contributions`, `input_profile`): the
  gate and up pre-activations of a neuron split exactly into writes of upstream units
  (embedding, every mixer head of layers <= L, every neuron of layers < L), with the
  RMSNorm before the MLP linearised at its actual scale. For Qwen3.5 the norm is
  `x / rms(x) * (1 + w)`; for Qwen3 it is `x / rms(x) * w`.
- `reader_effects`: first-order effect of one neuron's write on later-layer neuron
  activations at the same position (direct path), and on the logits.

Pole-flat indexing: a neuron-pole is `(layer, neuron, pole)` with pole 0 = pos, 1 = neg;
its flat index is `(layer * N + neuron) * 2 + pole`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neuronology.hooks import capture, intervene
from neuronology.model import decoder_layers, mlp_module

POLES = ("pos", "neg")


# ----------------------------------------------------------------------------- inputs


@dataclass
class AuditItem:
    """One transcript of the audit setting.

    `target_pos` is the index in `input_ids` of the first token of the unusual output
    (for a clean item: the matched position of the same type); `target_token_id` is the
    token the margin is taken for. `unusual_end` (exclusive) marks the end of the unusual
    output span in triggered items, used only to redact it from text shown to LLMs.
    """

    item_id: str
    kind: str  # "triggered" | "clean"
    pair_id: int
    input_ids: list[int]
    target_pos: int
    target_token_id: int
    unusual_end: int | None = None
    prompt_len: int | None = None

    def prefix(self) -> list[int]:
        """Tokens up to (not including) the target: the last one predicts the target."""
        return self.input_ids[: self.target_pos]


def load_audit_items(path: str | Path) -> list[AuditItem]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                out.append(AuditItem(**{k: d[k] for k in AuditItem.__dataclass_fields__ if k in d}))
    return out


def write_audit_items(path: str | Path, items: Iterable[AuditItem]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.__dict__) + "\n")


# ----------------------------------------------------------------------- architecture


def norm_gain(norm: nn.Module) -> torch.Tensor:
    """The multiplier applied after `x / rms(x)`: (1 + w) for Qwen3.5, w for Qwen3."""
    w = norm.weight.float()
    name = type(norm).__name__
    if name.startswith("Qwen3_5") and "Gated" not in name:
        return 1.0 + w
    if name in ("Qwen3RMSNorm",):
        return w
    raise NotImplementedError(f"RMSNorm formula for {name} unknown")


def norm_eps(norm: nn.Module) -> float:
    eps = getattr(norm, "eps", None)
    return float(eps if eps is not None else norm.variance_epsilon)


def inv_rms(h: torch.Tensor, eps: float) -> torch.Tensor:
    """1 / sqrt(mean(h^2) + eps) over the last dim (keepdim)."""
    h = h.float()
    return torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)


def mixer_module(layer_mod: nn.Module) -> nn.Module:
    if hasattr(layer_mod, "linear_attn"):
        return layer_mod.linear_attn
    return layer_mod.self_attn


def mixer_out_proj(layer_mod: nn.Module) -> nn.Linear:
    m = mixer_module(layer_mod)
    return m.out_proj if hasattr(m, "out_proj") else m.o_proj


def mixer_head_dim(layer_mod: nn.Module) -> int:
    m = mixer_module(layer_mod)
    return int(getattr(m, "head_v_dim", None) or m.head_dim)


def mixer_kind(layer_mod: nn.Module) -> str:
    return "deltanet" if hasattr(layer_mod, "linear_attn") else "attention"


def _check_no_bias(model: nn.Module) -> None:
    for i, lay in enumerate(decoder_layers(model)):
        mods = [mixer_out_proj(lay), lay.mlp.gate_proj, lay.mlp.up_proj, lay.mlp.down_proj]
        for m in mods:
            if getattr(m, "bias", None) is not None:
                raise NotImplementedError(
                    f"layer {i}: biased projection; decomposition needs a bias unit"
                )


def dsilu(x: torch.Tensor) -> torch.Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


# ------------------------------------------------------------------ gradient x activation


def _lm_head(model: nn.Module) -> nn.Module:
    return model.get_output_embeddings()


def _margin_from_hidden(model, h: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """h [B, d] final hidden (after the final norm) -> margin [B]."""
    logits = _lm_head(model)(h).float()
    return logits.gather(-1, token_ids[:, None]).squeeze(-1) - logits.mean(-1)


def grad_x_act(
    model: nn.Module,
    ids: list[int] | torch.Tensor,
    token_id: int,
    means: torch.Tensor | None = None,
    keep_dense: bool = False,
) -> dict[str, Any]:
    """Grad x act of the margin of `token_id` predicted at the last position of `ids`.

    `means` [L, N] are census means (for the centered variant; zeros if None).
    Returns margin (float) and per-pole sums raw/centered [L, N, 2] (float32 numpy); with
    `keep_dense` also `act` and `grad` [L, S, N] (CPU float32).
    """
    dev = next(model.parameters()).device
    x = torch.as_tensor(ids, dtype=torch.long, device=dev).reshape(1, -1)
    core = model.model

    def emb_hook(_mod, _args, out):
        return out.detach().requires_grad_(True)

    h = core.embed_tokens.register_forward_hook(emb_hook)
    try:
        with torch.enable_grad(), capture(model, detach=False) as acts:
            out = core(input_ids=x, use_cache=False)
            margin = _margin_from_hidden(
                model, out.last_hidden_state[:, -1], torch.tensor([token_id], device=dev)
            )[0]
            layers = sorted(acts)
            grads = torch.autograd.grad(margin, [acts[lyr] for lyr in layers])
    finally:
        h.remove()
    L = len(layers)
    N = acts[layers[0]].shape[-1]
    raw = torch.zeros(L, N, 2)
    cen = torch.zeros(L, N, 2)
    dense_a, dense_g = [], []
    for i, lyr in enumerate(layers):
        a = acts[lyr][0].detach().float()
        g = grads[i][0].float()
        mu = means[lyr].to(a) if means is not None else torch.zeros(N, device=a.device)
        pos, neg = a > 0, a < 0
        ag = a * g
        cg = (a - mu) * g
        raw[i, :, 0] = (ag * pos).sum(0).cpu()
        raw[i, :, 1] = (ag * neg).sum(0).cpu()
        cen[i, :, 0] = (cg * pos).sum(0).cpu()
        cen[i, :, 1] = (cg * neg).sum(0).cpu()
        if keep_dense:
            dense_a.append(a.cpu())
            dense_g.append(g.cpu())
    res = {"margin": float(margin.detach()), "raw": raw.numpy(), "centered": cen.numpy()}
    if keep_dense:
        res["act"] = torch.stack(dense_a)
        res["grad"] = torch.stack(dense_g)
    return res


def aggregate(scores: np.ndarray, top_k: int = 500) -> dict[str, np.ndarray]:
    """scores [T, L, N, 2] -> mean_abs, mean, top_frac (each [L, N, 2])."""
    T = scores.shape[0]
    absd = np.abs(scores)
    flat = absd.reshape(T, -1)
    k = min(top_k, flat.shape[1])
    hits = np.zeros(flat.shape[1], dtype=np.float64)
    for t in range(T):
        idx = np.argpartition(-flat[t], k - 1)[:k]
        hits[idx] += 1
    return {
        "mean_abs": absd.mean(0),
        "mean": scores.mean(0),
        "top_frac": (hits / T).reshape(scores.shape[1:]),
    }


def ranking(score: np.ndarray, n: int | None = None) -> np.ndarray:
    """Flat pole indices sorted by descending score ([L, N, 2] -> [n])."""
    flat = score.reshape(-1)
    if n is None or n >= flat.size:
        return np.argsort(-flat, kind="stable")
    idx = np.argpartition(-flat, n - 1)[:n]
    return idx[np.argsort(-flat[idx], kind="stable")]


def unflat(f: int | np.ndarray, n_neurons: int):
    f = np.asarray(f)
    pole = f % 2
    ln = f // 2
    return ln // n_neurons, ln % n_neurons, pole


def pole_key(layer: int, neuron: int, pole: int) -> str:
    return f"L{int(layer)}_N{int(neuron)}_{POLES[int(pole)]}"


# ----------------------------------------------------------------------------- ablation


@dataclass
class PoleAblation:
    """Replace selected pole activations by a fill value.

    mode "ablate": positions whose activation falls in a selected pole are filled.
    mode "keep":   every activation NOT in a selected pole is filled (sufficiency test).
    """

    masks: dict[int, tuple[torch.Tensor, torch.Tensor]]
    fill: dict[int, torch.Tensor]
    mode: str

    @classmethod
    def from_flat(
        cls, flat: np.ndarray, n_layers: int, n_neurons: int, fill: dict, mode: str
    ) -> PoleAblation:
        if mode not in ("ablate", "keep"):
            raise ValueError(mode)
        m = np.zeros((n_layers, n_neurons, 2), dtype=bool)
        if len(flat):
            lyr, n, p = unflat(np.asarray(flat), n_neurons)
            m[lyr, n, p] = True
        layers = range(n_layers) if mode == "keep" else sorted(set(np.nonzero(m.any((1, 2)))[0]))
        masks = {
            int(lyr): (torch.from_numpy(m[lyr, :, 0].copy()), torch.from_numpy(m[lyr, :, 1].copy()))
            for lyr in layers
        }
        return cls(masks, fill, mode)

    def fn(self) -> Callable[[int, torch.Tensor], torch.Tensor]:
        cache: dict = {}

        def f(layer: int, acts: torch.Tensor) -> torch.Tensor:
            if layer not in self.masks:
                return acts
            key = (layer, acts.device)
            if key not in cache:
                mp, mn = self.masks[layer]
                cache[key] = (mp.to(acts.device), mn.to(acts.device))
            mp, mn = cache[key]
            cover = ((acts > 0) & mp) | ((acts < 0) & mn)
            repl = cover if self.mode == "ablate" else ~cover
            fill = self.fill[layer].to(acts.device, acts.dtype)
            return torch.where(repl, fill.expand_as(acts), acts)

        return f

    def layers(self) -> list[int]:
        return sorted(self.masks)


@torch.no_grad()
def margins(
    model: nn.Module,
    items: list[AuditItem],
    ablation: PoleAblation | None = None,
    batch_size: int = 8,
    pad_id: int = 0,
) -> np.ndarray:
    """Margin of each item's target token at its predicting position, under an ablation."""
    dev = next(model.parameters()).device
    out = np.zeros(len(items))
    order = np.argsort([len(it.prefix()) for it in items])
    for s in range(0, len(items), batch_size):
        chunk = [items[i] for i in order[s : s + batch_size]]
        lens = [len(it.prefix()) for it in chunk]
        S = max(lens)
        ids = torch.full((len(chunk), S), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), S), dtype=torch.long)
        for j, it in enumerate(chunk):
            ids[j, : lens[j]] = torch.as_tensor(it.prefix())
            mask[j, : lens[j]] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        tgt = torch.tensor([it.target_token_id for it in chunk], device=dev)
        if ablation is not None:
            ctx = intervene(model, ablation.fn(), ablation.layers() or [0])
        else:
            ctx = _null()
        with ctx:
            am = None if all(n == S for n in lens) else mask
            hid = model.model(input_ids=ids, attention_mask=am, use_cache=False).last_hidden_state
        last = hid[torch.arange(len(chunk), device=dev), torch.tensor(lens, device=dev) - 1]
        m = _margin_from_hidden(model, last, tgt).cpu().double().numpy()
        for j, i in enumerate(order[s : s + batch_size]):
            out[i] = m[j]
    return out


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def find_circuit_size(F: Callable[[int], float], n_max: int, target: float) -> dict[str, Any]:
    """Smallest N with F(N) >= target, by doubling then bisection (assumes rough monotonicity).

    Returns {"n_star": int | None, "evals": {N: F(N)}}.
    """
    evals: dict[int, float] = {}

    def ev(n: int) -> float:
        if n not in evals:
            evals[n] = float(F(n))
        return evals[n]

    n = 1
    while True:
        if ev(n) >= target:
            break
        if n >= n_max:
            return {"n_star": None, "evals": dict(sorted(evals.items()))}
        n = min(2 * n, n_max)
    lo, hi = n // 2, n
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ev(mid) >= target:
            hi = mid
        else:
            lo = mid
    return {"n_star": hi, "evals": dict(sorted(evals.items()))}


def recovery_curves(
    model: nn.Module,
    items: list[AuditItem],
    order: np.ndarray,
    n_layers: int,
    n_neurons: int,
    fill: dict[int, torch.Tensor],
    cfg: dict[str, Any],
    rng: np.random.Generator,
    log=print,
) -> dict[str, Any]:
    """F(N) for the top-N poles of `order` under one fill rule, plus random baselines.

    sufficiency  F(N) = (l_N - l_0) / (l_full - l_0), l_N = keep only top N poles, fill rest,
                 l_0 = fill everything (PLAN.md 8.1 / the post's recovery metric).
    necessity    D(N) = (l_full - l_abl(N)) / (l_full - l_0), l_abl(N) = fill the top N.
    Margins are averaged over items before the ratio.
    """
    bs, pad = cfg.get("batch_size", 8), cfg.get("pad_id", 0)
    n_total = n_layers * n_neurons * 2
    n_max = min(int(cfg.get("n_max") or n_total), n_total)
    n_random = int(cfg.get("n_random", 3))

    def mean_margin(abl: PoleAblation | None) -> float:
        return float(margins(model, items, abl, bs, pad).mean())

    l_full = mean_margin(None)
    l_0 = mean_margin(
        PoleAblation.from_flat(np.array([], dtype=np.int64), n_layers, n_neurons, fill, "keep")
    )
    denom = l_full - l_0
    out: dict[str, Any] = {"l_full": l_full, "l_0": l_0, "n_total_poles": n_total}
    if abs(denom) < 1e-9:
        out["error"] = "l_full == l_0; recovery undefined"
        return out

    def suff(sel: np.ndarray) -> float:
        abl = PoleAblation.from_flat(sel, n_layers, n_neurons, fill, "keep")
        return (mean_margin(abl) - l_0) / denom

    def nec(sel: np.ndarray) -> float:
        abl = PoleAblation.from_flat(sel, n_layers, n_neurons, fill, "ablate")
        return (l_full - mean_margin(abl)) / denom

    def rand_sel(n: int) -> np.ndarray:
        return rng.choice(n_total, size=n, replace=False)

    for name, fn, target in (
        ("sufficiency", suff, cfg.get("suff_target", 0.9)),
        ("necessity", nec, cfg.get("nec_target", 0.5)),
    ):
        res = find_circuit_size(lambda n, fn=fn: fn(order[:n]), n_max, target)
        grid = sorted(
            set([2**k for k in range(int(math.log2(max(n_max, 1))) + 1)] + list(res["evals"]))
        )
        grid = [n for n in grid if n <= n_max]
        top = {n: res["evals"].get(n) for n in grid}
        for n in grid:
            if top[n] is None:
                top[n] = float(fn(order[:n]))
        rand = {}
        rand_grid = [n for n in grid if n <= cfg.get("random_n_max", n_max)]
        if res["n_star"] is not None and res["n_star"] not in rand_grid:
            rand_grid.append(res["n_star"])
        for n in sorted(rand_grid):
            rand[n] = [float(fn(rand_sel(n))) for _ in range(n_random)]
        ns = res["n_star"]
        gate = None
        if ns is not None:
            rmean = float(np.mean(rand[ns]))
            gate = {
                "n_star": ns,
                "top": top.get(ns, res["evals"].get(ns)),
                "random_mean": rmean,
                "ratio": (res["evals"][ns] / rmean) if rmean > 1e-12 else float("inf"),
                "passes_10x": bool(rmean <= res["evals"][ns] / 10.0),
            }
        out[name] = {
            "target": target,
            "n_star": ns,
            "search_evals": {str(k): v for k, v in res["evals"].items()},
            "top": {str(k): v for k, v in sorted(top.items())},
            "random": {str(k): v for k, v in sorted(rand.items())},
            "at_n_star": gate,
        }
        log(f"[F(N)] {name}: n_star={ns} gate={gate}")
    return out


# --------------------------------------------------------------- upstream decomposition


@dataclass
class Parts:
    """Residual-stream pieces of one sequence (batch 1), all [S, ...] on the model device."""

    embed: torch.Tensor  # [S, d]
    mixer_in: dict[int, torch.Tensor]  # input of the mixer's output projection [S, H*hd]
    act: dict[int, torch.Tensor]  # neuron activations [S, N]
    h_mid: dict[int, torch.Tensor]  # residual entering post_attention_layernorm [S, d]
    z_gate: dict[int, torch.Tensor]
    z_up: dict[int, torch.Tensor]
    h_final: torch.Tensor  # residual entering the final norm [S, d]
    meta: dict[str, Any] = field(default_factory=dict)


@torch.no_grad()
def capture_parts(model: nn.Module, ids: list[int] | torch.Tensor) -> Parts:
    _check_no_bias(model)
    dev = next(model.parameters()).device
    x = torch.as_tensor(ids, dtype=torch.long, device=dev).reshape(1, -1)
    layers = decoder_layers(model)
    st: dict[str, dict] = {k: {} for k in ("mixer_in", "act", "h_mid", "z_gate", "z_up")}
    misc: dict[str, torch.Tensor] = {}
    handles = []

    def pre(store, i):
        def fn(_m, args):
            store[i] = args[0][0].detach()

        return fn

    def post(store, i):
        def fn(_m, _a, out):
            store[i] = out[0].detach()

        return fn

    try:
        handles.append(
            model.model.embed_tokens.register_forward_hook(
                lambda _m, _a, out: misc.__setitem__("embed", out[0].detach())
            )
        )
        handles.append(
            model.model.norm.register_forward_pre_hook(
                lambda _m, args: misc.__setitem__("h_final", args[0][0].detach())
            )
        )
        for i, lay in enumerate(layers):
            handles.append(mixer_out_proj(lay).register_forward_pre_hook(pre(st["mixer_in"], i)))
            handles.append(
                lay.post_attention_layernorm.register_forward_pre_hook(pre(st["h_mid"], i))
            )
            handles.append(lay.mlp.down_proj.register_forward_pre_hook(pre(st["act"], i)))
            handles.append(lay.mlp.gate_proj.register_forward_hook(post(st["z_gate"], i)))
            handles.append(lay.mlp.up_proj.register_forward_hook(post(st["z_up"], i)))
        model.model(input_ids=x, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return Parts(
        misc["embed"],
        st["mixer_in"],
        st["act"],
        st["h_mid"],
        st["z_gate"],
        st["z_up"],
        misc["h_final"],
    )


@dataclass
class UnitLayout:
    """Index layout of upstream units for a target layer L.

    [0]                       embedding
    then for l in 0..L        mixer heads of layer l (H_l each)
    then for l in 0..L-1      neurons of layer l (N each)
    """

    layer: int
    head_counts: list[int]
    n_neurons: int
    mixer_kinds: list[str]

    @property
    def n_units(self) -> int:
        return 1 + sum(self.head_counts) + self.layer * self.n_neurons

    def mixer_offset(self, lyr: int) -> int:
        return 1 + sum(self.head_counts[:lyr])

    def neuron_offset(self, lyr: int) -> int:
        return 1 + sum(self.head_counts) + lyr * self.n_neurons

    def label(self, u: int) -> dict[str, Any]:
        u = int(u)
        if u == 0:
            return {"type": "embed"}
        u -= 1
        for lyr, h in enumerate(self.head_counts):
            if u < h:
                return {
                    "type": "mixer_head",
                    "layer": lyr,
                    "head": u,
                    "mixer": self.mixer_kinds[lyr],
                }
            u -= h
        return {"type": "neuron", "layer": u // self.n_neurons, "neuron": u % self.n_neurons}


def unit_layout(model: nn.Module, layer: int) -> UnitLayout:
    lays = decoder_layers(model)
    heads, kinds = [], []
    for lyr in range(layer + 1):
        op = mixer_out_proj(lays[lyr])
        heads.append(op.in_features // mixer_head_dim(lays[lyr]))
        kinds.append(mixer_kind(lays[lyr]))
    n = mlp_module(model, 0).down_proj.in_features
    return UnitLayout(layer, heads, n, kinds)


@torch.no_grad()
def unit_contributions(
    model: nn.Module, parts: Parts, layer: int, neurons: torch.Tensor, pos: int
) -> dict[str, torch.Tensor]:
    """Exact split of z_gate and z_up of `neurons` (layer, position `pos`) into unit writes.

    Returns {"gate": [K, U], "up": [K, U], "z_gate": [K], "z_up": [K], "layout": UnitLayout}.
    Rows sum to the actual pre-activations up to floating-point error.
    """
    lays = decoder_layers(model)
    lay = lays[layer]
    lo = unit_layout(model, layer)
    norm = lay.post_attention_layernorm
    scale = norm_gain(norm) * inv_rms(parts.h_mid[layer][pos], norm_eps(norm))  # [d]
    neurons = torch.as_tensor(neurons, dtype=torch.long, device=scale.device)
    out: dict[str, Any] = {"layout": lo}
    for lane, W in (("gate", lay.mlp.gate_proj.weight), ("up", lay.mlp.up_proj.weight)):
        V = W[neurons].float() * scale  # [K, d]
        K = V.shape[0]
        C = torch.empty(K, lo.n_units, device=V.device)
        C[:, 0] = V @ parts.embed[pos].float()
        for lyr in range(layer + 1):
            op = mixer_out_proj(lays[lyr])
            hd = mixer_head_dim(lays[lyr])
            U = _mm(V, op.weight)  # [K, H*hd]
            x = parts.mixer_in[lyr][pos].float()
            per_head = (U * x).view(K, -1, hd).sum(-1)
            o = lo.mixer_offset(lyr)
            C[:, o : o + per_head.shape[1]] = per_head
        for lyr in range(layer):
            Wd = lays[lyr].mlp.down_proj.weight  # [d, N]
            o = lo.neuron_offset(lyr)
            C[:, o : o + lo.n_neurons] = _mm(V, Wd) * parts.act[lyr][pos].float()
        out[lane] = C
    out["z_gate"] = parts.z_gate[layer][pos, neurons].float()
    out["z_up"] = parts.z_up[layer][pos, neurons].float()
    return out


def _mm(V: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """V @ W in the weight's dtype (fp32 result). Exact for fp32 models; for bf16 models
    the products are rounded to bf16 (relative error ~4e-3), which `input_profile` reports
    as the decomposition residual."""
    return (V.to(W.dtype) @ W).float()


@torch.no_grad()
def input_profile(
    model: nn.Module,
    parts_list: list[tuple[Parts, list[int]]],
    layer: int,
    neuron: int,
    pole: int,
    n_baseline: int = 512,
    top: int = 50,
    seed: int = 0,
    chunk: int = 128,
) -> dict[str, Any]:
    """Four-part input profile of a neuron-pole at its firing positions.

    Each unit's write is turned into its first-order effect on the pole activation
    s * a (s = +1 for pos, -1 for neg): gate lane e = s * SiLU'(g) * u * c_gate, up lane
    e = s * SiLU(g) * c_up. The baseline for a unit is its mean effect on `n_baseline`
    random neurons of the same layer at the same positions (each with its own pole sign);
    lanes are ranked by excess = e - baseline, averaged over positions.
    """
    s = 1.0 if pole == 0 else -1.0
    rng = np.random.default_rng(seed)
    N = mlp_module(model, layer).down_proj.in_features
    others = np.setdiff1d(np.arange(N), [neuron])
    base_ids = rng.choice(others, size=min(n_baseline, len(others)), replace=False)
    acc = {"gate": None, "up": None}
    raw = {"gate": None, "up": None}
    n_pos = 0
    layout = None
    resid: list[float] = []
    for parts, positions in parts_list:
        for p in positions:
            ids = torch.as_tensor(np.concatenate([[neuron], base_ids]))
            exc = {"gate": 0.0, "up": 0.0}
            tgt = {}
            base_sum = {"gate": 0.0, "up": 0.0}
            for c0 in range(0, len(ids), chunk):
                sub = ids[c0 : c0 + chunk]
                uc = unit_contributions(model, parts, layer, sub, p)
                layout = uc["layout"]
                g, u = uc["z_gate"], uc["z_up"]
                a = torch.nn.functional.silu(g) * u
                sign = torch.where(a < 0, -1.0, 1.0)
                if c0 == 0:
                    sign[0] = s
                fg = (sign * dsilu(g) * u)[:, None]
                fu = (sign * torch.nn.functional.silu(g))[:, None]
                eg, eu = fg * uc["gate"], fu * uc["up"]
                if c0 == 0:
                    tgt["gate"], tgt["up"] = eg[0].clone(), eu[0].clone()
                    eg, eu = eg[1:], eu[1:]
                    for lane, z in (("gate", g[0]), ("up", u[0])):
                        tot = float(uc[lane][0].sum())
                        resid.append(abs(tot - float(z)) / max(abs(float(z)), 1e-6))
                base_sum["gate"] = base_sum["gate"] + eg.sum(0)
                base_sum["up"] = base_sum["up"] + eu.sum(0)
            nb = len(ids) - 1
            for lane in ("gate", "up"):
                exc[lane] = tgt[lane] - base_sum[lane] / max(nb, 1)
                acc[lane] = exc[lane] if acc[lane] is None else acc[lane] + exc[lane]
                raw[lane] = tgt[lane] if raw[lane] is None else raw[lane] + tgt[lane]
            n_pos += 1
    if n_pos == 0:
        return {"n_positions": 0}
    res: dict[str, Any] = {
        "n_positions": n_pos,
        "n_baseline": int(len(base_ids)),
        # |sum of unit writes - actual pre-activation| / |actual|, over positions and lanes
        "decomposition_rel_residual_max": max(resid) if resid else None,
    }
    for lane in ("gate", "up"):
        ex = (acc[lane] / n_pos).cpu()
        rw = (raw[lane] / n_pos).cpu()
        for name, idx in (
            (f"{lane}_drivers", torch.topk(ex, min(top, ex.numel())).indices),
            (f"{lane}_suppressors", torch.topk(-ex, min(top, ex.numel())).indices),
        ):
            res[name] = [
                {
                    **layout.label(int(u)),
                    "unit": int(u),
                    "excess": float(ex[u]),
                    "effect": float(rw[u]),
                }
                for u in idx
            ]
    return res


@torch.no_grad()
def reader_effects(
    model: nn.Module, parts: Parts, layer: int, neuron: int, pos: int
) -> dict[str, torch.Tensor]:
    """Direct first-order effect of neuron (layer, neuron)'s write at `pos`.

    later[l'] = dA of every neuron of layer l' > layer at `pos`:
      dA = SiLU'(g) * u * dg + SiLU(g) * du,  dg = w_gate . (write * (1+γ) / rms(h)).
    logits = W_U (write * gain_f / rms_f), the direct logit effect.
    """
    lays = decoder_layers(model)
    w = parts.act[layer][pos, neuron].float() * lays[layer].mlp.down_proj.weight[:, neuron].float()
    later = {}
    for lp in range(layer + 1, len(lays)):
        lay = lays[lp]
        norm = lay.post_attention_layernorm
        x = w * norm_gain(norm) * inv_rms(parts.h_mid[lp][pos], norm_eps(norm))
        dg = _mm(x[None], lay.mlp.gate_proj.weight.T)[0]
        du = _mm(x[None], lay.mlp.up_proj.weight.T)[0]
        g, u = parts.z_gate[lp][pos].float(), parts.z_up[lp][pos].float()
        later[lp] = dsilu(g) * u * dg + torch.nn.functional.silu(g) * du
    fn = model.model.norm
    xf = w * norm_gain(fn) * inv_rms(parts.h_final[pos], norm_eps(fn))
    logits = _mm(xf[None], _lm_head(model).weight.T)[0]
    return {"later": later, "logits": logits}


def top_readers(
    effects: list[dict[str, Any]], top: int = 200, top_logits: int = 20
) -> dict[str, Any]:
    """Average `reader_effects` over positions; top later-layer neurons and logit tokens."""
    if not effects:
        return {"neurons": [], "logits_up": [], "logits_down": []}
    layers = sorted(effects[0]["later"])
    rows = []
    if layers:
        M = torch.stack([torch.stack([e["later"][lp] for lp in layers]) for e in effects]).mean(0)
        N = M.shape[1]
        flat = M.reshape(-1).cpu()
        idx = torch.topk(flat.abs(), min(top, flat.numel())).indices
        rows = [
            {"layer": layers[int(i) // N], "neuron": int(i) % N, "effect": float(flat[i])}
            for i in idx
        ]
    lg = torch.stack([e["logits"] for e in effects]).mean(0).cpu()
    up = torch.topk(lg, top_logits)
    dn = torch.topk(-lg, top_logits)
    return {
        "neurons": rows,
        "logits_up": [
            {"token_id": int(i), "effect": float(v)}
            for v, i in zip(up.values, up.indices, strict=True)
        ],
        "logits_down": [
            {"token_id": int(i), "effect": float(-v)}
            for v, i in zip(dn.values, dn.indices, strict=True)
        ],
    }
