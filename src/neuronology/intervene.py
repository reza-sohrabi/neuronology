"""Removal interventions for Phase 5 (PLAN.md 10.1).

Generic tools only: this module never reads the secret definitions and never imports the
planting module (tests/test_blinding.py). The Phase-2-side driver (scripts/m5_remove.py)
builds the triggered prompts and scores the marker; everything here sees only token ids,
weights and census arrays.

Interventions (each is a context manager that is active for every forward pass inside it,
including every step of `generate`):

- pole ablation (`PoleAblation` from attribution.py): the top-k poles of a ranking, a fixed
  set, or a random set; fill with the census mean or zero, only at positions where the
  neuron's sign matches the pole (the M3 convention);
- `DirectionAblation`: k random orthonormal directions in a layer's neuron-activation
  space (the "random directions" control), with the component along them set to the
  census mean's component; layers get as many directions as the matched pole set has poles
  in that layer;
- `residual_projection`: project one unit direction out of the residual stream at the
  output of the given decoder layers (the difference-of-means baseline);
- `revert_neurons`: copy the base model's gate row, up row and down column back into the
  selected neurons (the weight-diff baseline), restored on exit.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import numpy as np
import torch
from torch import nn

from neuronology.attribution import POLES, PoleAblation, unflat
from neuronology.hooks import intervene
from neuronology.model import decoder_layers, mlp_module

_KEY = re.compile(r"^L(\d+)_N(\d+)_(pos|neg|0|1)$")


# ------------------------------------------------------------------------- pole sets


def flat_from_key(key: str, n_neurons: int) -> int:
    """'L31_N4507_1' / 'L31_N4507_neg' -> pole-flat index (pole 0 = pos, 1 = neg)."""
    m = _KEY.match(key)
    if m is None:
        raise ValueError(f"bad pole key {key!r}")
    lyr, n, p = int(m.group(1)), int(m.group(2)), m.group(3)
    pole = POLES.index(p) if p in POLES else int(p)
    return (lyr * n_neurons + n) * 2 + pole


def flats_from_keys(keys: list[str], n_neurons: int) -> np.ndarray:
    return np.array([flat_from_key(k, n_neurons) for k in keys], dtype=np.int64)


def layer_counts(flat: np.ndarray, n_neurons: int) -> dict[int, int]:
    """Number of poles per layer in a pole-flat set."""
    if len(flat) == 0:
        return {}
    lyr, _, _ = unflat(np.asarray(flat), n_neurons)
    u, c = np.unique(lyr, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c, strict=True)}


def fill_values(means: np.ndarray, fill: str, device) -> dict[int, torch.Tensor]:
    """Per-layer fill vectors: census means ([L, N]) or zeros."""
    if fill not in ("mean", "zero"):
        raise ValueError(fill)
    t = torch.as_tensor(np.asarray(means), dtype=torch.float32, device=device)
    return {lyr: (t[lyr] if fill == "mean" else torch.zeros_like(t[lyr])) for lyr in range(len(t))}


@contextmanager
def ablate_poles(
    model: nn.Module, flat: np.ndarray, fill: dict[int, torch.Tensor], n_neurons: int
) -> Iterator[None]:
    """Fill the given poles (only where the activation has the pole's sign)."""
    n_layers = len(decoder_layers(model))
    abl = PoleAblation.from_flat(
        np.asarray(flat, dtype=np.int64), n_layers, n_neurons, fill, "ablate"
    )
    if not abl.layers():
        yield
        return
    with intervene(model, abl.fn(), abl.layers()):
        yield


def random_pole_order(n_total: int, k_max: int, seed: int) -> np.ndarray:
    """A nested random pole order: the random set of size k is its first k entries."""
    return np.random.default_rng(seed).permutation(n_total)[:k_max]


# ------------------------------------------------------------------ random directions


def random_orthonormal(n: int, m: int, seed: int, device) -> torch.Tensor:
    """[n, m] matrix with orthonormal columns, from a seeded Gaussian (QR on the device)."""
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randn(n, m, generator=g, dtype=torch.float32)
    dev = torch.device(device)
    on = x.to(dev) if dev.type == "cuda" else x  # QR on CPU off-CUDA (MPS lacks it)
    q, r = torch.linalg.qr(on)
    s = torch.sign(torch.diagonal(r))
    s[s == 0] = 1.0
    return (q * s).to(dev)  # sign fix: the basis is a deterministic function of the seed


class DirectionAblation:
    """Mean-fill along random orthonormal directions of each layer's neuron space.

    For layer l with basis Q_l [N, m_l] and census mean mu_l [N]:
        a <- a - ((a - mu_l) Q_l) Q_l^T
    i.e. the component of a along span(Q_l) is replaced by that of the mean. Unlike a pole
    ablation this is two-sided (applied at every position whatever the sign).
    """

    def __init__(self, bases: dict[int, torch.Tensor], means: dict[int, torch.Tensor]):
        self.bases = {lyr: q for lyr, q in bases.items() if q.shape[1] > 0}
        self.means = means

    @classmethod
    def nested(
        cls,
        counts: dict[int, int],
        n_neurons: int,
        seed: int,
        means: dict[int, torch.Tensor],
        device,
        full: dict[int, torch.Tensor] | None = None,
    ) -> DirectionAblation:
        """Bases with counts[l] columns; `full` (from `full_bases`) makes sets nested."""
        bases = {}
        for lyr, m in counts.items():
            m = min(m, n_neurons)  # a layer has only n_neurons dimensions
            if m <= 0:
                continue
            if full is not None and lyr in full and full[lyr].shape[1] >= m:
                bases[lyr] = full[lyr][:, :m]
            else:
                bases[lyr] = random_orthonormal(n_neurons, m, seed * 1000 + lyr, device)
        return cls(bases, means)

    def layers(self) -> list[int]:
        return sorted(self.bases)

    def fn(self):
        def f(layer: int, acts: torch.Tensor) -> torch.Tensor:
            q = self.bases.get(layer)
            if q is None:
                return acts
            q = q.to(acts.device)
            mu = self.means[layer].to(acts.device, torch.float32)
            a = acts.float()
            coef = (a - mu) @ q
            return (a - coef @ q.T).to(acts.dtype)

        return f

    @contextmanager
    def active(self, model: nn.Module) -> Iterator[None]:
        if not self.layers():
            yield
            return
        with intervene(model, self.fn(), self.layers()):
            yield


def full_bases(
    max_counts: dict[int, int], n_neurons: int, seed: int, device
) -> dict[int, torch.Tensor]:
    """One basis per layer with the largest count needed, so smaller sets are prefixes."""
    return {
        lyr: random_orthonormal(n_neurons, min(m, n_neurons), seed * 1000 + lyr, device)
        for lyr, m in max_counts.items()
        if m > 0
    }


def capped_dims(counts: dict[int, int], n_neurons: int) -> int:
    """Directions actually removed: per-layer counts capped at the layer width."""
    return int(sum(min(m, n_neurons) for m in counts.values()))


# ---------------------------------------------------------------- residual directions


def _hidden(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_hidden(out, new):
    if isinstance(out, tuple):
        return (new,) + tuple(out[1:])
    return new


@contextmanager
def capture_resid_last(
    model: nn.Module, last_idx: torch.Tensor, layers: list[int] | None = None
) -> Iterator[dict[int, torch.Tensor]]:
    """Residual stream after each decoder layer at one position per row ([B] indices).

    Yields store[layer] -> [B, d] float32 (on the model device).
    """
    lays = decoder_layers(model)
    layers = list(range(len(lays))) if layers is None else layers
    store: dict[int, torch.Tensor] = {}
    handles = []

    def mk(lyr: int):
        def hook(_m, _a, out):
            h = _hidden(out)
            rows = torch.arange(h.shape[0], device=h.device)
            store[lyr] = h[rows, last_idx.to(h.device)].detach().float()

        return hook

    try:
        for lyr in layers:
            handles.append(lays[lyr].register_forward_hook(mk(lyr)))
        yield store
    finally:
        for h in handles:
            h.remove()


def diff_of_means(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Unit direction of mean(pos) - mean(neg) ([n, d] each) and the raw norm."""
    d = pos.float().mean(0) - neg.float().mean(0)
    nrm = float(d.norm())
    return d / max(nrm, 1e-12), nrm


@contextmanager
def residual_projection(
    model: nn.Module, direction: torch.Tensor, layers: list[int]
) -> Iterator[None]:
    """Project `direction` (unit, [d]) out of the residual stream after each given layer."""
    lays = decoder_layers(model)
    handles = []

    def hook(_m, _a, out):
        h = _hidden(out)
        u = direction.to(h.device, torch.float32)
        hf = h.float()
        new = (hf - (hf @ u)[..., None] * u).to(h.dtype)
        return _replace_hidden(out, new)

    try:
        for lyr in layers:
            handles.append(lays[lyr].register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


# ------------------------------------------------------------------ weight-diff baseline


def mlp_change(base: dict[str, torch.Tensor], cur: dict[str, torch.Tensor]) -> np.ndarray:
    """Per-neuron L2 norm of the change of [gate row, up row, down column] ([N] numpy).

    `base`/`cur` hold "gate" [N, d], "up" [N, d], "down" [d, N] (float32, same device).
    """
    sq = (cur["gate"] - base["gate"]).pow(2).sum(1)
    sq += (cur["up"] - base["up"]).pow(2).sum(1)
    sq += (cur["down"] - base["down"]).pow(2).sum(0)
    return sq.sqrt().cpu().numpy()


def model_mlp(model: nn.Module, layer: int) -> dict[str, torch.Tensor]:
    m = mlp_module(model, layer)
    return {
        # clone: on CPU in fp32 .float().cpu() would alias the live weights
        "gate": m.gate_proj.weight.detach().float().cpu().clone(),
        "up": m.up_proj.weight.detach().float().cpu().clone(),
        "down": m.down_proj.weight.detach().float().cpu().clone(),
    }


def top_changed(change: np.ndarray, k: int) -> np.ndarray:
    """Flat neuron indices (layer * N + n) of the k largest changes ([L, N])."""
    flat = change.reshape(-1)
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    return idx[np.argsort(-flat[idx], kind="stable")]


def base_neuron_weights(
    reader_get, neurons: np.ndarray, n_neurons: int
) -> dict[int, dict[str, torch.Tensor]]:
    """Base weights of the given flat neurons only: {layer: {n: [m], gate, up [m, d], down [d, m]}}.

    `reader_get(key)` returns a full base tensor (e.g. `diff.WeightReader.get`), keys
    `layers.{l}.mlp.{gate,up,down}_proj.weight`. Only the selected rows are kept.
    """
    neurons = np.asarray(neurons, dtype=np.int64)
    out = {}
    for lyr in np.unique(neurons // n_neurons):
        sel = np.sort(neurons[neurons // n_neurons == lyr] % n_neurons)
        idx = torch.as_tensor(sel, dtype=torch.long)
        p = f"layers.{int(lyr)}.mlp."
        out[int(lyr)] = {
            "n": sel,
            "gate": reader_get(p + "gate_proj.weight").index_select(0, idx).clone(),
            "up": reader_get(p + "up_proj.weight").index_select(0, idx).clone(),
            "down": reader_get(p + "down_proj.weight").index_select(1, idx).clone(),
        }
    return out


@contextmanager
def revert_neurons(
    model: nn.Module, base: dict[int, dict], neurons: np.ndarray, n_neurons: int
) -> Iterator[None]:
    """Set the selected neurons' MLP weights to the base values; restore on exit.

    `base` comes from `base_neuron_weights` over a superset of `neurons` (flat indices).
    """
    neurons = np.asarray(neurons, dtype=np.int64)
    saved = []
    try:
        with torch.no_grad():
            for lyr in np.unique(neurons // n_neurons):
                b = base[int(lyr)]
                want = neurons[neurons // n_neurons == lyr] % n_neurons
                pos = np.searchsorted(b["n"], want)
                if not np.array_equal(b["n"][np.minimum(pos, len(b["n"]) - 1)], want):
                    raise KeyError(f"layer {lyr}: base weights missing for some neurons")
                pos_t = torch.as_tensor(pos, dtype=torch.long)
                m = mlp_module(model, int(lyr))
                for w, src, dim in (
                    (m.gate_proj.weight, b["gate"], 0),
                    (m.up_proj.weight, b["up"], 0),
                    (m.down_proj.weight, b["down"], 1),
                ):
                    ix = torch.as_tensor(want, dtype=torch.long, device=w.device)
                    old = w.index_select(dim, ix).clone()
                    saved.append((w, dim, ix, old))
                    new = src.index_select(dim, pos_t).to(w.device, w.dtype)
                    w.index_copy_(dim, ix, new)
        yield
    finally:
        with torch.no_grad():
            for w, dim, ix, old in reversed(saved):
                w.index_copy_(dim, ix, old)


@contextmanager
def stacked(*ctxs) -> Iterator[None]:
    with ExitStack() as st:
        for c in ctxs:
            st.enter_context(c)
        yield
