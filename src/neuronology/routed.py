"""Neuron hooks for routed (mixture-of-experts) models: OLMoE (PLAN_OLMOE.md) and
Qwen3.6-35B-A3B (transformers' `Qwen3_5Moe*` classes; same fused expert weights and expert
forward signature, so one instrumented forward serves both).

Differences of the Qwen model that this module handles:
- its router renormalises the k routing weights to sum to 1 (no config flag), OLMoE's does
  not: `router_normalises`, used by `routed_info` and `reroute_without`;
- every layer also has a shared expert, a plain SwiGLU MLP of 512 neurons that every token
  passes through, scaled by sigmoid(shared_expert_gate(x)). Its neurons are NOT units of the
  routed unit set (which stays `(layer, expert, neuron)` over the routed experts); it is
  handled as a whole by `drop_shared_expert` and `revert_shared_expert` (controls);
- the Hub checkpoint stores the fused tensors (`mlp.experts.gate_up_proj`), not per-expert
  keys: `base_expert_weights` reads either layout.

The dense hooks (`hooks.py`) hook the input of `mlp.down_proj`. A routed block has no
such module: in transformers 5.22 `OlmoeExperts` stores every expert's weights as two
3-D parameters (`gate_up_proj` [E, 2I, d], `down_proj` [E, d, I]) and loops over experts
inside one forward. This module therefore swaps in an instrumented copy of that forward
(per module instance, only while a hook is registered) that exposes each token's routed
expert activations and lets callbacks replace them.

Unit and activation definitions (PLAN_OLMOE.md section 3):

- unit (the routed "neuron"): `(layer, expert e, neuron n)`, flat index within a layer
  `u = e * I + n`, so a layer has `E * I` units (64 * 1024 = 65,536 for OLMoE-1B-7B).
  Pole-flat indexing is the dense one with `n_neurons = E * I`:
  `(layer * E * I + u) * 2 + pole`, so `attribution.unflat` and the ranking code apply.
- activation: `a = SiLU(z_gate) * z_up` of expert e, *before* the router weight. It exists
  only at tokens routed to e. The expert's write is `w_e * a_n * W_down[e][:, n]`, where
  `w_e` is the router weight (softmax prob, not renormalised for OLMoE). `weighted()`
  gives `w_e * a`, the quantity the post ranks by.
- `RoutedActs` holds, for T flattened tokens and the k routed slots of each token (in the
  router's top-k order): act [T, k, I], experts [T, k], weights [T, k] (+ z_gate, z_up).
- an unrouted unit has no activation at that token: pole ablation, patching and fill
  never touch it (ablating it is a no-op), and its census statistics count routed tokens
  only. `dense()` gives the zero-filled [T, E*I] view when a dense layout is needed.

Callbacks run in registration order, as in hooks.py: a capture opened inside an
`intervene_routed` block records what the expert down-projections actually received.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neuronology.model import decoder_layers

# ------------------------------------------------------------------------ architecture


def experts_module(model: nn.Module, layer: int) -> nn.Module:
    return decoder_layers(model)[layer].mlp.experts


def router_module(model: nn.Module, layer: int) -> nn.Module:
    return decoder_layers(model)[layer].mlp.gate


def router_normalises(obj: Any) -> bool:
    """True when the router divides the k routing weights by their sum.

    `obj` is a router module or a (text) config. OLMoE has the flag `norm_topk_prob`;
    the Qwen3.5-MoE router always normalises and has no flag.
    """
    flag = getattr(obj, "norm_topk_prob", None)
    if flag is not None:
        return bool(flag)
    name = type(obj).__name__
    if name.startswith("Qwen3_5Moe") or str(getattr(obj, "model_type", "")).startswith(
        "qwen3_5_moe"
    ):
        return True
    raise NotImplementedError(f"routing-weight normalisation of {name} unknown")


def shared_expert_module(model: nn.Module, layer: int) -> nn.Module | None:
    """The layer's always-on shared expert (a dense SwiGLU MLP), or None."""
    return getattr(decoder_layers(model)[layer].mlp, "shared_expert", None)


def is_routed(model: nn.Module) -> bool:
    """True for a model whose MLP blocks are routed experts with fused 3-D weights."""
    try:
        ex = experts_module(model, 0)
    except (AttributeError, IndexError, TypeError):
        return False
    return isinstance(getattr(ex, "gate_up_proj", None), torch.Tensor) and isinstance(
        getattr(ex, "down_proj", None), torch.Tensor
    )


@dataclass(frozen=True)
class RoutedInfo:
    n_layers: int
    n_experts: int
    top_k: int
    d_expert: int  # neurons per expert (I)
    d_model: int
    norm_topk_prob: bool
    has_shared_expert: bool

    @property
    def n_units(self) -> int:
        """Units per layer, E * I."""
        return self.n_experts * self.d_expert

    @property
    def routed_cells_per_token(self) -> int:
        """Activations computed per token per layer, k * I (the post's "routed cells")."""
        return self.top_k * self.d_expert

    def unit(self, expert: int, neuron: int) -> int:
        return expert * self.d_expert + neuron

    def split(self, unit: int | np.ndarray):
        u = np.asarray(unit)
        return u // self.d_expert, u % self.d_expert


def routed_info(model_or_config: nn.Module | Any) -> RoutedInfo:
    cfg = getattr(model_or_config, "config", model_or_config)
    cfg = getattr(cfg, "text_config", None) or cfg
    return RoutedInfo(
        n_layers=cfg.num_hidden_layers,
        n_experts=cfg.num_experts,
        top_k=cfg.num_experts_per_tok,
        d_expert=getattr(cfg, "moe_intermediate_size", None) or cfg.intermediate_size,
        d_model=cfg.hidden_size,
        norm_topk_prob=router_normalises(cfg),
        has_shared_expert=bool(getattr(cfg, "shared_expert_intermediate_size", 0)),
    )


def expert_weights(model: nn.Module, layer: int) -> dict[str, torch.Tensor]:
    """Views of one layer's expert weights: gate, up [E, I, d]; down [E, d, I]."""
    ex = experts_module(model, layer)
    gu = ex.gate_up_proj
    I = gu.shape[1] // 2  # noqa: E741
    return {"gate": gu[:, :I], "up": gu[:, I:], "down": ex.down_proj}


def unit_key(layer: int, unit: int, pole: int, d_expert: int) -> str:
    """'L3_E12_N400_pos': layer 3, expert 12, neuron 400, positive pole."""
    e, n = divmod(int(unit), d_expert)
    return f"L{int(layer)}_E{e}_N{n}_{('pos', 'neg')[int(pole)]}"


# ----------------------------------------------------------------------- activations


@dataclass
class RoutedActs:
    """One layer's routed activations for T = B * S flattened tokens.

    act      [T, k, I]  SiLU(z_gate) * z_up of each routed slot (before the router weight)
    experts  [T, k]     expert id of each slot
    weights  [T, k]     router weight multiplying that expert's output
    """

    act: torch.Tensor
    experts: torch.Tensor
    weights: torch.Tensor
    z_gate: torch.Tensor | None = None
    z_up: torch.Tensor | None = None
    batch_shape: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return self.act.shape[0]

    def weighted(self) -> torch.Tensor:
        """w_e * a, [T, k, I]: the scale at which the unit writes to the residual stream."""
        return self.act * self.weights[..., None].to(self.act.dtype)

    def units(self) -> torch.Tensor:
        """Flat unit index e * I + n of every activation, [T, k, I] (long)."""
        I = self.act.shape[-1]  # noqa: E741
        n = torch.arange(I, device=self.act.device)
        return self.experts[..., None] * I + n

    def dense(self, n_experts: int, weighted: bool = False) -> torch.Tensor:
        """[T, E * I] with unrouted units 0 (memory: T * E * I values)."""
        T, k, I = self.act.shape  # noqa: E741
        v = self.weighted() if weighted else self.act
        out = torch.zeros(T, n_experts, I, dtype=v.dtype, device=v.device)
        rows = torch.arange(T, device=v.device)[:, None].expand(T, k)
        out[rows, self.experts] = v
        return out.view(T, n_experts * I)

    def routed_mask(self, n_experts: int) -> torch.Tensor:
        """[T, E] bool: which experts each token was routed to."""
        m = torch.zeros(self.n_tokens, n_experts, dtype=torch.bool, device=self.experts.device)
        m.scatter_(1, self.experts, True)
        return m

    def detach(self) -> RoutedActs:
        def d(t):
            return None if t is None else t.detach()

        return RoutedActs(
            self.act.detach(),
            self.experts.detach(),
            self.weights.detach(),
            d(self.z_gate),
            d(self.z_up),
            self.batch_shape,
            dict(self.meta),
        )

    def select(self, rows: torch.Tensor) -> RoutedActs:
        """Keep the tokens where `rows` ([T] bool, or indices) is set (e.g. non-padding)."""

        def s(t):
            return None if t is None else t[rows]

        return RoutedActs(
            self.act[rows],
            self.experts[rows],
            self.weights[rows],
            s(self.z_gate),
            s(self.z_up),
            (),
            dict(self.meta),
        )


# Callback: (layer, RoutedActs) -> replacement act [T, k, I] or None (observe only).
RoutedFn = Callable[[int, RoutedActs], torch.Tensor | None]


def _instrumented_forward(mod: nn.Module, layer: int):
    """The project's single expert forward (PLAN_OLMOE.md 13, G0 decisions).

    Mirrors transformers' `grouped_mm_experts_forward` step for step: slots sorted by
    expert, one grouped GEMM for gate/up and one for down (`_grouped_linear`, the same
    kernel), routing weights applied after the down projection, the k expert outputs of a
    token summed by a reshape-sum (fp32 accumulation). Between gating and the down
    projection the activations are unsorted to [T, k, I], handed to `mod._nrn_callbacks`
    (which may replace them) and sorted back; with no callback, or an identity one, the
    result is the grouped_mm result. (The first version copied the eager loop instead, which
    sums the k outputs with a bf16 index_add_; G0 measured argmax agreement 0.985 between
    the two paths in bf16.)
    """
    from transformers.integrations.moe import _grouped_linear

    def forward(hidden_states, top_k_index, top_k_weights):
        T, d = hidden_states.shape
        k = top_k_index.shape[1]
        E = mod.num_experts
        I = mod.intermediate_dim  # noqa: E741
        dev = hidden_states.device
        sample_weights = top_k_weights.reshape(-1)
        expert_ids = top_k_index.reshape(-1)
        expert_ids_g, perm = torch.sort(expert_ids)
        x_g = hidden_states[perm // k]
        w_g = sample_weights[perm]
        hin = expert_ids_g.float() if dev.type in ("cpu", "mps") else expert_ids_g.int()
        per_e = torch.histc(hin, bins=E, min=0, max=E - 1)
        offsets = torch.cumsum(per_e, dim=0, dtype=torch.int32)
        tr = getattr(mod, "is_transposed", False)
        gu = _grouped_linear(x_g, mod.gate_up_proj, offsets, bias=None, is_transposed=tr)
        g, u = gu.chunk(2, dim=-1)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel(), device=dev)
        cbs = list(getattr(mod, "_nrn_callbacks", None) or [])
        # Memory (G0 OOM fix, PLAN_OLMOE.md 13.4): at a 32 x ~900-token prefill one [T, k, I]
        # bf16 tensor is ~450 MB, so intermediates are released as soon as they are used
        # and the pre-activations are unsorted only for callbacks that read them.
        act_g = mod.act_fn(g) * u
        zg = zu = None
        if cbs and any(getattr(cb, "_nrn_needs_pre", False) for cb in cbs):
            zg, zu = g[inv].view(T, k, I), u[inv].view(T, k, I)
        del gu, g, u
        if cbs:
            act = act_g[inv].view(T, k, I)
            del act_g
            ra = RoutedActs(
                act, top_k_index, top_k_weights, zg, zu, meta={"layer": layer, "n_experts": E}
            )
            del act, zg, zu
            for cb in cbs:
                new = cb(layer, ra)
                if new is not None:
                    if new.shape != ra.act.shape:
                        raise ValueError(
                            f"layer {layer}: callback returned {tuple(new.shape)}, "
                            f"expected {tuple(ra.act.shape)}"
                        )
                    ra.act = new.to(ra.act.dtype)
                    del new
            act_g = ra.act.reshape(T * k, I)[perm]
            del ra
        y = _grouped_linear(act_g, mod.down_proj, offsets, bias=None, is_transposed=tr)
        del act_g
        if torch.is_grad_enabled():
            y = y * w_g.unsqueeze(-1)
        else:
            y.mul_(w_g.unsqueeze(-1))  # same values, one [S, d] buffer fewer
        y = y[inv]
        return y.view(T, k, d).sum(dim=1).to(hidden_states.dtype)

    return forward


class _Handle:
    def __init__(self, mod: nn.Module, cb: RoutedFn):
        self.mod, self.cb = mod, cb

    def remove(self) -> None:
        cbs = self.mod._nrn_callbacks
        if self.cb in cbs:
            cbs.remove(self.cb)
        if not cbs and not getattr(self.mod, "_nrn_pinned", False):
            del self.mod._nrn_callbacks
            self.mod.__dict__.pop("forward", None)  # back to the class forward


def _install(mod: nn.Module, layer: int) -> None:
    if "forward" not in mod.__dict__:
        mod._nrn_callbacks = []
        mod.forward = _instrumented_forward(mod, layer)


def register_routed(model: nn.Module, layer: int, cb: RoutedFn) -> _Handle:
    """Attach `cb` to one layer's experts (installing the instrumented forward if needed)."""
    mod = experts_module(model, layer)
    _install(mod, layer)
    mod._nrn_callbacks.append(cb)
    return _Handle(mod, cb)


def pin_measurement_path(model: nn.Module) -> None:
    """Make the instrumented forward every layer's expert forward, hooks or not
    (PLAN_OLMOE.md 13: one expert path for training and every measurement)."""
    for lyr in range(len(decoder_layers(model))):
        mod = experts_module(model, lyr)
        _install(mod, lyr)
        mod._nrn_pinned = True


def is_pinned(model: nn.Module) -> bool:
    mods = [experts_module(model, lyr) for lyr in range(len(decoder_layers(model)))]
    return all(getattr(m, "_nrn_pinned", False) and "forward" in m.__dict__ for m in mods)


@contextmanager
def unpinned(model: nn.Module, implementation: str) -> Iterator[None]:
    """Temporarily run transformers' own expert forward (G0 comparisons only)."""
    saved = {}
    for lyr in range(len(decoder_layers(model))):
        mod = experts_module(model, lyr)
        saved[lyr] = mod.__dict__.pop("forward", None)
    model.set_experts_implementation(implementation)
    try:
        yield
    finally:
        model.set_experts_implementation("grouped_mm")
        for lyr, f in saved.items():
            if f is not None:
                experts_module(model, lyr).forward = f


def _resolve(model: nn.Module, layers: Iterable[int] | int | None) -> list[int]:
    if not is_routed(model):
        raise TypeError("not a routed model; use neuronology.hooks")
    n = len(decoder_layers(model))
    if layers is None:
        return list(range(n))
    if isinstance(layers, int):
        layers = [layers]
    out = [lyr if lyr >= 0 else n + lyr for lyr in layers]
    for lyr in out:
        if not 0 <= lyr < n:
            raise IndexError(f"layer {lyr} out of range for {n} layers")
    return out


@contextmanager
def _registered(model, layers, make_cb):
    handles = []
    try:
        for lyr in _resolve(model, layers):
            handles.append(register_routed(model, lyr, make_cb(lyr)))
        yield
    finally:
        for h in reversed(handles):
            h.remove()


@contextmanager
def capture_routed(
    model: nn.Module, layers: Iterable[int] | int | None = None, *, detach: bool = True
) -> Iterator[dict[int, RoutedActs]]:
    """Yield `store[layer] -> RoutedActs` (last forward pass inside the block)."""
    store: dict[int, RoutedActs] = {}

    def make(lyr):
        def cb(_layer, ra):
            store[lyr] = ra.detach() if detach else ra

        cb._nrn_needs_pre = True
        return cb

    with _registered(model, layers, make):
        yield store


@contextmanager
def intervene_routed(
    model: nn.Module,
    fn: Callable[[int, RoutedActs], torch.Tensor],
    layers: Iterable[int] | int | None = None,
) -> Iterator[None]:
    """Replace routed activations: the expert down-projections receive `fn(layer, ra)`."""
    with _registered(model, layers, lambda _lyr: fn):
        yield


@contextmanager
def stream_routed(
    model: nn.Module, fn: Callable[[int, RoutedActs], None], layers=None
) -> Iterator[None]:
    """Call `fn(layer, ra)` as each layer's routed activations are computed (census)."""

    def make(_lyr):
        def cb(layer, ra):
            fn(layer, ra.detach())

        cb._nrn_needs_pre = True
        return cb

    with _registered(model, layers, make):
        yield


@contextmanager
def capture_router(
    model: nn.Module, layers: Iterable[int] | int | None = None
) -> Iterator[dict[int, torch.Tensor]]:
    """Router logits per layer, [T, E] (detached): a residual-like baseline feature."""
    store: dict[int, torch.Tensor] = {}
    handles = []
    try:
        for lyr in _resolve(model, layers):

            def hook(_m, _a, out, lyr=lyr):
                store[lyr] = out[0].detach()

            handles.append(router_module(model, lyr).register_forward_hook(hook))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def reroute_without(model: nn.Module, experts: dict[int, list[int]]) -> Iterator[None]:
    """Remove experts from routing (logit -inf before softmax/top-k): tokens go elsewhere.

    The alternative expert ablation, "drop" (the expert's output is removed and the other
    weights are unchanged), is a zero fill of all its units: `drop_experts`.
    """
    handles = []
    try:
        for lyr, ex in experts.items():
            r = router_module(model, lyr)
            if r.num_experts - len(set(ex)) < r.top_k:
                raise ValueError(f"layer {lyr}: fewer than top-k experts left after rerouting")
            idx = torch.as_tensor(sorted(ex), dtype=torch.long)

            def hook(mod, args, _out, idx=idx):
                h = args[0].reshape(-1, mod.hidden_dim)
                logits = F.linear(h, mod.weight)
                masked = logits.clone()
                masked[:, idx.to(logits.device)] = -float("inf")
                probs = torch.softmax(masked, dtype=torch.float, dim=-1)
                top, ind = torch.topk(probs, mod.top_k, dim=-1)
                if router_normalises(mod):
                    top = top / top.sum(-1, keepdim=True)
                return logits, top.to(logits.dtype), ind

            handles.append(r.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture_routing(
    model: nn.Module, layers: Iterable[int] | int | None = None
) -> Iterator[dict[int, tuple[torch.Tensor, torch.Tensor]]]:
    """Router outputs per layer: (top-k weights [T, k], expert indices [T, k]), detached."""
    store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []
    try:
        for lyr in _resolve(model, layers):

            def hook(_m, _a, out, lyr=lyr):
                store[lyr] = (out[1].detach(), out[2].detach())

            handles.append(router_module(model, lyr).register_forward_hook(hook))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def freeze_routing(
    model: nn.Module, routes: dict[int, tuple[torch.Tensor, torch.Tensor]]
) -> Iterator[None]:
    """Route every given layer as in a reference pass (`capture_routing` of the same batch).

    The router logits are still computed from the current residual stream, but the expert
    choice and the weights are the reference pass's: the "frozen routing" variant.
    """
    handles = []
    try:
        for lyr, (w, idx) in routes.items():

            def hook(_m, _a, out, w=w, idx=idx):
                if out[2].shape != idx.shape:
                    raise ValueError("frozen routing needs the same batch as the reference")
                return out[0], w.to(out[1].device, out[1].dtype), idx.to(out[2].device)

            handles.append(router_module(model, lyr).register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def routing_changed(cur: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """[T] bool: the token's set of k experts differs from the reference's."""
    return (cur.sort(-1).values != ref.to(cur.device).sort(-1).values).any(-1)


def drop_experts(experts: dict[int, list[int]]):
    """`intervene_routed` fn: zero every activation of the given experts (no rerouting)."""

    def fn(layer: int, ra: RoutedActs) -> torch.Tensor:
        ex = experts.get(layer)
        if not ex:
            return ra.act
        sel = torch.isin(ra.experts, torch.as_tensor(ex, device=ra.experts.device))
        return ra.act.masked_fill(sel[..., None], 0.0)

    return fn


# ------------------------------------------------------------------ patching and writes


def match_slots(cur: torch.Tensor, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """For each current slot [T, k], the source slot routed to the same expert at that token.

    Returns (found [T, k] bool, src_slot [T, k] long; arbitrary where not found).
    """
    eq = cur[..., :, None] == src[..., None, :]  # [T, k, k]
    return eq.any(-1), eq.float().argmax(-1)


def patch_fn(
    source: dict[int, RoutedActs], units: dict[int, torch.Tensor] | None = None
) -> Callable[[int, RoutedActs], torch.Tensor]:
    """`intervene_routed` fn writing source activations into the current pass.

    Matching is by (token, expert): a slot is patched only where the source run routed the
    same token to the same expert; elsewhere the current value stays (the unit did not
    exist in the source). `units` restricts patching to a bool mask [E * I] per layer.
    """

    def fn(layer: int, ra: RoutedActs) -> torch.Tensor:
        src = source.get(layer)
        if src is None:
            return ra.act
        if src.act.shape != ra.act.shape:
            raise ValueError(f"layer {layer}: source shape {tuple(src.act.shape)} differs")
        found, slot = match_slots(ra.experts, src.experts.to(ra.experts.device))
        rows = torch.arange(ra.n_tokens, device=ra.act.device)[:, None]
        val = src.act.to(ra.act.device, ra.act.dtype)[rows, slot]  # [T, k, I]
        take = found[..., None].expand_as(ra.act)
        if units is not None and layer in units:
            m = units[layer].to(ra.act.device).view(-1)
            take = take & m[ra.units()]
        elif units is not None:
            return ra.act
        return torch.where(take, val, ra.act)

    return fn


def expert_writes(ra: RoutedActs, w_down: torch.Tensor) -> torch.Tensor:
    """Sum over routed units of w_e * a_n * W_down[e][:, n]: [T, d] (the MoE block output)."""
    wd = w_down[ra.experts].float()  # [T, k, d, I]
    return torch.einsum("tki,tkdi->td", ra.weighted().float(), wd)


# ------------------------------------------------------------------------ pole ablation

ROW_CHUNK = 8192  # (token, slot) rows per chunk in pole ablation: ~16 MB per temporary


@dataclass
class RoutedPoleAblation:
    """Fill selected unit-poles of a routed model (the routed `PoleAblation`).

    mode "ablate": routed activations whose sign matches a selected pole are filled.
    mode "keep":   every routed activation NOT in a selected pole is filled (sufficiency).
    Unrouted units are absent at a token and are never filled in either mode.
    `fill[layer]` is [E * I]: the census mean over routed tokens, or zeros.
    """

    masks: dict[int, tuple[torch.Tensor, torch.Tensor]]
    fill: dict[int, torch.Tensor]
    mode: str
    n_experts: int
    d_expert: int

    @classmethod
    def from_flat(
        cls,
        flat: np.ndarray,
        n_layers: int,
        n_experts: int,
        d_expert: int,
        fill: dict[int, torch.Tensor],
        mode: str,
    ) -> RoutedPoleAblation:
        if mode not in ("ablate", "keep"):
            raise ValueError(mode)
        U = n_experts * d_expert
        m = np.zeros((n_layers, U, 2), dtype=bool)
        if len(flat):
            f = np.asarray(flat, dtype=np.int64)
            pole, lu = f % 2, f // 2
            m[lu // U, lu % U, pole] = True
        layers = range(n_layers) if mode == "keep" else sorted(set(np.nonzero(m.any((1, 2)))[0]))
        masks = {
            int(lyr): (torch.from_numpy(m[lyr, :, 0].copy()), torch.from_numpy(m[lyr, :, 1].copy()))
            for lyr in layers
        }
        return cls(masks, fill, mode, n_experts, d_expert)

    def fn(self) -> Callable[[int, RoutedActs], torch.Tensor]:
        cache: dict = {}
        E, I = self.n_experts, self.d_expert  # noqa: E741

        def f(layer: int, ra: RoutedActs) -> torch.Tensor:
            if layer not in self.masks:
                return ra.act
            dev = ra.act.device
            key = (layer, dev)
            if key not in cache:
                mp, mn = self.masks[layer]
                cache[key] = (
                    mp.to(dev).view(E, I),
                    mn.to(dev).view(E, I),
                    self.fill[layer].to(dev).view(E, I),
                )
            mp, mn, fl = cache[key]
            a = ra.act
            out = a.clone()
            step = max(1, ROW_CHUNK // max(1, a.shape[1]))
            for t0 in range(0, a.shape[0], step):  # bounded temporaries (PLAN_OLMOE.md 13.4)
                x, ex = a[t0 : t0 + step], ra.experts[t0 : t0 + step]
                cover = ((x > 0) & mp[ex]) | ((x < 0) & mn[ex])
                repl = cover if self.mode == "ablate" else ~cover
                out[t0 : t0 + step] = torch.where(repl, fl[ex].to(a.dtype), x)
            return out

        return f

    def layers(self) -> list[int]:
        return sorted(self.masks)


@contextmanager
def ablate_routed_poles(
    model: nn.Module,
    flat: np.ndarray,
    fill: dict[int, torch.Tensor],
    info: RoutedInfo,
    mode: str = "ablate",
) -> Iterator[None]:
    abl = RoutedPoleAblation.from_flat(
        np.asarray(flat, dtype=np.int64), info.n_layers, info.n_experts, info.d_expert, fill, mode
    )
    if not abl.layers():
        yield
        return
    with intervene_routed(model, abl.fn(), abl.layers()):
        yield


# ---------------------------------------------------------------------- sparsification

ROUTED_RANKS = ("abs", "abs_route", "abs_route_wnorm")


def routed_keep_mask(
    ra: RoutedActs,
    mode: str,
    keep_frac: float,
    rank: str = "abs_route",
    w_norm: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    scope: str = "token",
) -> torch.Tensor:
    """Bool [T, k, I]: True = keep.

    scope "token": the k * I routed cells of a token are ranked together and
    round(keep_frac * k * I) are kept (the post's top-320-of-4,096 rule); scope "expert":
    each routed expert keeps round(keep_frac * I) of its own I cells (the reading of the
    post's per-expert random-64 / bottom-448 controls, PLAN_OLMOE.md 3.4).
    rank "abs" = |a|; "abs_route" = |a| * w_e (the post's rule); "abs_route_wnorm" also
    multiplies by ||W_down[e][:, n]|| (`w_norm` [E, I]).
    """
    from neuronology.sparsify import top_mask

    T, k, I = ra.act.shape  # noqa: E741
    if scope not in ("token", "expert"):
        raise ValueError(scope)
    group = (T, k * I) if scope == "token" else (T, k, I)
    n_keep = round(keep_frac * group[-1])
    if mode == "random":
        score = torch.rand(group, generator=generator, device=ra.act.device)
        return top_mask(score, n_keep).view(T, k, I)
    if rank not in ROUTED_RANKS:
        raise ValueError(rank)
    score = ra.act.abs().float()
    if rank != "abs":
        score = score * ra.weights[..., None].float()
    if rank == "abs_route_wnorm":
        if w_norm is None:
            raise ValueError("abs_route_wnorm needs w_norm")
        score = score * w_norm.to(score.device)[ra.experts]
    m = top_mask(score.reshape(group), n_keep).view(T, k, I)
    if mode == "top":
        return m
    if mode == "bottom":
        return ~m
    raise ValueError(mode)


def make_routed_sparsify_fn(
    mode: str,
    keep_frac: float,
    fill: dict[int, torch.Tensor] | None,
    rank: str = "abs_route",
    w_norms: dict[int, torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
    scope: str = "token",
) -> Callable[[int, RoutedActs], torch.Tensor]:
    """`intervene_routed` fn: keep the top `keep_frac` routed cells, fill the rest.

    `fill[layer]` [E * I] is the routed-token census mean; None means zero fill. Only
    routed cells are touched; routing itself is recomputed downstream as usual.
    """

    def fn(layer: int, ra: RoutedActs) -> torch.Tensor:
        wn = None if w_norms is None else w_norms[layer]
        keep = routed_keep_mask(ra, mode, keep_frac, rank, wn, generator, scope)
        E = ra.meta.get("n_experts")
        if fill is None:
            filler = torch.zeros((), dtype=ra.act.dtype, device=ra.act.device)
        else:
            filler = fill[layer].to(ra.act.device, ra.act.dtype).view(E, -1)[ra.experts]
        return torch.where(keep, ra.act, filler)

    return fn


def down_norms(model: nn.Module, layer: int) -> torch.Tensor:
    """||W_down[e][:, n]|| for every unit, [E, I]."""
    return experts_module(model, layer).down_proj.float().norm(dim=1)


# ------------------------------------------------------------------ gradient x activation


def grad_x_act_routed(
    model: nn.Module,
    ids: list[int] | torch.Tensor,
    token_id: int,
    means: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Routed `attribution.grad_x_act`: margin of `token_id` at the last position.

    Per unit-pole sums over positions *where the unit is routed*: raw sum a * g and centered
    sum (a - mean_u) * g, as [L, E * I, 2] numpy (pole 0 pos, 1 neg). `means` is [L, E*I].
    """
    from neuronology.attribution import _margin_from_hidden

    dev = next(model.parameters()).device
    info = routed_info(model)
    x = torch.as_tensor(ids, dtype=torch.long, device=dev).reshape(1, -1)
    with torch.enable_grad():
        core = model.model
        h = core.embed_tokens.register_forward_hook(
            lambda _m, _a, out: out.detach().requires_grad_(True)
        )
        try:
            with capture_routed(model, detach=False) as acts:
                out = core(input_ids=x, use_cache=False)
                margin = _margin_from_hidden(
                    model, out.last_hidden_state[:, -1], torch.tensor([token_id], device=dev)
                )[0]
                layers = sorted(acts)
                grads = torch.autograd.grad(margin, [acts[lyr].act for lyr in layers])
        finally:
            h.remove()
    U = info.n_units
    raw = torch.zeros(len(layers), U, 2)
    cen = torch.zeros(len(layers), U, 2)
    cnt = torch.zeros(len(layers), U, 2)
    for i, lyr in enumerate(layers):
        ra = acts[lyr]
        a = ra.act.detach().float()
        g = grads[i].float()
        u = ra.units().reshape(-1).cpu()
        mu = means[lyr].to(a.device)[ra.units()] if means is not None else torch.zeros_like(a)
        for pole, m in ((0, a > 0), (1, a < 0)):
            raw[i, :, pole].index_add_(0, u, (a * g * m).reshape(-1).cpu())
            cen[i, :, pole].index_add_(0, u, ((a - mu) * g * m).reshape(-1).cpu())
            cnt[i, :, pole].index_add_(0, u, m.reshape(-1).float().cpu())
    return {
        "margin": float(margin.detach()),
        "raw": raw.numpy(),
        "centered": cen.numpy(),
        "count": cnt.numpy(),  # firing positions per unit-pole (routed, matching sign)
    }


def active_matched_sampler(
    order: np.ndarray, active: np.ndarray, n_units: int, rng: np.random.Generator
) -> Callable[[int], np.ndarray]:
    """Random control for a routed model (PLAN_OLMOE.md 3.3): for size n, as many poles per
    layer as `order[:n]` has, drawn uniformly from that layer's `active` poles (pole-flat
    indices, e.g. poles that fire on an audit transcript). A layer with fewer active poles
    than needed contributes all of them."""
    active = np.asarray(active, dtype=np.int64)
    per_layer = {
        int(lyr): active[(active // 2) // n_units == lyr]
        for lyr in np.unique((active // 2) // n_units)
    }

    def sample(n: int) -> np.ndarray:
        lyr = (np.asarray(order[:n]) // 2) // n_units
        out = []
        for lv, c in zip(*np.unique(lyr, return_counts=True), strict=True):
            pool = per_layer.get(int(lv), np.zeros(0, dtype=np.int64))
            out.append(rng.choice(pool, size=min(int(c), len(pool)), replace=False))
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    return sample


# --------------------------------------------------------------- upstream decomposition


def _rms_gain(norm: nn.Module) -> torch.Tensor:
    name = type(norm).__name__
    if name == "OlmoeRMSNorm":
        return norm.weight.float()  # weight * x / rms(x)
    raise NotImplementedError(f"RMSNorm formula for {name} unknown")


@dataclass
class RoutedParts:
    """Residual-stream pieces of one sequence (batch 1) of a routed model."""

    embed: torch.Tensor  # [S, d]
    mixer_in: dict[int, torch.Tensor]  # input of o_proj [S, H * hd]
    routed: dict[int, RoutedActs]  # T = S
    h_mid: dict[int, torch.Tensor]  # residual entering post_attention_layernorm [S, d]
    h_final: torch.Tensor  # residual entering the final norm [S, d]


@torch.no_grad()
def capture_parts_routed(model: nn.Module, ids: list[int] | torch.Tensor) -> RoutedParts:
    for i, lay in enumerate(decoder_layers(model)):
        if getattr(lay.self_attn.o_proj, "bias", None) is not None:
            raise NotImplementedError(f"layer {i}: biased o_proj")
    dev = next(model.parameters()).device
    x = torch.as_tensor(ids, dtype=torch.long, device=dev).reshape(1, -1)
    lays = decoder_layers(model)
    mixer_in: dict[int, torch.Tensor] = {}
    h_mid: dict[int, torch.Tensor] = {}
    misc: dict[str, torch.Tensor] = {}
    handles = []
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
        for i, lay in enumerate(lays):
            handles.append(
                lay.self_attn.o_proj.register_forward_pre_hook(
                    lambda _m, args, i=i: mixer_in.__setitem__(i, args[0][0].detach())
                )
            )
            handles.append(
                lay.post_attention_layernorm.register_forward_pre_hook(
                    lambda _m, args, i=i: h_mid.__setitem__(i, args[0][0].detach())
                )
            )
        with capture_routed(model) as routed:
            model.model(input_ids=x, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return RoutedParts(misc["embed"], mixer_in, dict(routed), h_mid, misc["h_final"])


@dataclass
class RoutedUnitLayout:
    """Upstream units of target layer L: [0] embedding, heads of layers 0..L, then the
    E * I expert units of each layer 0..L-1 (zero where not routed at that position)."""

    layer: int
    head_counts: list[int]
    n_units_per_layer: int
    d_expert: int

    @property
    def n_units(self) -> int:
        return 1 + sum(self.head_counts) + self.layer * self.n_units_per_layer

    def mixer_offset(self, lyr: int) -> int:
        return 1 + sum(self.head_counts[:lyr])

    def unit_offset(self, lyr: int) -> int:
        return 1 + sum(self.head_counts) + lyr * self.n_units_per_layer

    def label(self, u: int) -> dict[str, Any]:
        u = int(u)
        if u == 0:
            return {"type": "embed"}
        u -= 1
        for lyr, h in enumerate(self.head_counts):
            if u < h:
                return {"type": "mixer_head", "layer": lyr, "head": u, "mixer": "attention"}
            u -= h
        lyr, v = divmod(u, self.n_units_per_layer)
        e, n = divmod(v, self.d_expert)
        return {"type": "expert_neuron", "layer": lyr, "expert": e, "neuron": n}


@torch.no_grad()
def unit_contributions_routed(
    model: nn.Module,
    parts: RoutedParts,
    layer: int,
    expert: int,
    neurons: torch.Tensor | list[int],
    pos: int,
) -> dict[str, Any]:
    """Exact split of z_gate and z_up of (layer, expert, neurons) at `pos` into unit writes.

    Defined for any expert, routed at `pos` or not (the pre-activation of an unrouted
    expert is computable even though the model does not compute it). Returns
    {"gate": [K, U], "up": [K, U], "z_gate": [K], "z_up": [K], "layout"}; rows sum to the
    pre-activations up to floating-point error.
    """
    lays = decoder_layers(model)
    info = routed_info(model)
    lay = lays[layer]
    heads = [
        lays[lyr].self_attn.o_proj.in_features // lays[lyr].self_attn.head_dim
        for lyr in range(layer + 1)
    ]
    lo = RoutedUnitLayout(layer, heads, info.n_units, info.d_expert)
    norm = lay.post_attention_layernorm
    h = parts.h_mid[layer][pos].float()
    scale = _rms_gain(norm) * torch.rsqrt(h.pow(2).mean() + norm.variance_epsilon)
    ew = expert_weights(model, layer)
    neurons = torch.as_tensor(neurons, dtype=torch.long, device=scale.device)
    x_norm = h * scale
    out: dict[str, Any] = {"layout": lo}
    for lane in ("gate", "up"):
        W = ew[lane][expert][neurons].float()  # [K, d]
        V = W * scale
        K = V.shape[0]
        C = torch.zeros(K, lo.n_units, device=V.device)
        C[:, 0] = V @ parts.embed[pos].float()
        for lyr in range(layer + 1):
            op = lays[lyr].self_attn.o_proj
            hd = lays[lyr].self_attn.head_dim
            per_head = (
                ((V @ op.weight.float()) * parts.mixer_in[lyr][pos].float()).view(K, -1, hd).sum(-1)
            )
            o = lo.mixer_offset(lyr)
            C[:, o : o + per_head.shape[1]] = per_head
        for lyr in range(layer):
            ra = parts.routed[lyr]
            wd = experts_module(model, lyr).down_proj
            o = lo.unit_offset(lyr)
            for j in range(ra.experts.shape[1]):
                e = int(ra.experts[pos, j])
                w = float(ra.weights[pos, j])
                c = (V @ wd[e].float()) * (w * ra.act[pos, j].float())  # [K, I]
                C[:, o + e * info.d_expert : o + (e + 1) * info.d_expert] += c
        out[lane] = C
        out[f"z_{lane}"] = W @ x_norm
    return out


# ------------------------------------------------------------ M5o removal baselines


class RoutedDirectionAblation:
    """Random-directions control for a routed model (PLAN_OLMOE.md 3.3).

    For each (layer, expert) with m poles in the matched set: m random orthonormal
    directions Q [I, m] in that expert's neuron space and the expert's routed census mean
    mu [I]; at tokens routed to the expert, a <- a - ((a - mu) Q) Q^T. Bases are nested by
    seed (a smaller set is a prefix of a larger one).
    """

    def __init__(
        self,
        bases: dict[tuple[int, int], torch.Tensor],
        means: dict[int, torch.Tensor],
        d_expert: int,
    ):
        self.bases = {k: q for k, q in bases.items() if q.shape[1] > 0}
        self.means, self.I = means, d_expert

    @staticmethod
    def counts(flat: np.ndarray, d_expert: int, n_units: int) -> dict[tuple[int, int], int]:
        f = np.asarray(flat, dtype=np.int64) // 2
        lyr, e = f // n_units, (f % n_units) // d_expert
        keys, c = np.unique(np.stack([lyr, e], 1), axis=0, return_counts=True)
        return {(int(a), int(b)): int(n) for (a, b), n in zip(keys, c, strict=True)}

    @classmethod
    def nested(cls, counts, d_expert, seed, means, device, full=None):
        from neuronology.intervene import random_orthonormal

        bases = {}
        for (lyr, e), m in counts.items():
            m = min(m, d_expert)
            if full is not None and (lyr, e) in full and full[(lyr, e)].shape[1] >= m:
                bases[(lyr, e)] = full[(lyr, e)][:, :m]
            else:
                s = seed * 1_000_003 + lyr * 1000 + e
                bases[(lyr, e)] = random_orthonormal(d_expert, m, s, device)
        return cls(bases, means, d_expert)

    def layers(self) -> list[int]:
        return sorted({lyr for lyr, _ in self.bases})

    def fn(self):
        def f(layer: int, ra: RoutedActs) -> torch.Tensor:
            a = ra.act
            out = None
            for (lyr, e), q in self.bases.items():
                if lyr != layer:
                    continue
                sel = ra.experts == e  # [T, k]
                if not bool(sel.any()):
                    continue
                if out is None:
                    out = a.clone()
                q = q.to(a.device)
                mu = self.means[layer].to(a.device, torch.float32).view(-1, self.I)[e]
                x = out[sel].float()  # [n, I]
                out[sel] = (x - ((x - mu) @ q) @ q.T).to(a.dtype)
            return a if out is None else out

        return f

    @contextmanager
    def active(self, model: nn.Module) -> Iterator[None]:
        if not self.bases:
            yield
            return
        with intervene_routed(model, self.fn(), self.layers()):
            yield


def routed_full_bases(counts, d_expert, seed, device) -> dict[tuple[int, int], torch.Tensor]:
    from neuronology.intervene import random_orthonormal

    return {
        (lyr, e): random_orthonormal(
            d_expert, min(m, d_expert), seed * 1_000_003 + lyr * 1000 + e, device
        )
        for (lyr, e), m in counts.items()
        if m > 0
    }


def base_expert_weights(reader_get, layer: int, n_experts: int) -> dict[str, torch.Tensor]:
    """A checkpoint's expert weights of one layer: gate, up [E, I, d], down [E, d, I]
    (float32, CPU). `reader_get` = `diff.WeightReader.get`. Reads the fused layout
    (`mlp.experts.gate_up_proj`, Qwen3.5-MoE) or per-expert keys (OLMoE's Hub format)."""
    p = f"layers.{layer}.mlp.experts."
    try:
        gu = reader_get(f"{p}gate_up_proj")
    except KeyError:
        gu = None
    if gu is not None:
        I = gu.shape[1] // 2  # noqa: E741
        return {"gate": gu[:, :I], "up": gu[:, I:], "down": reader_get(f"{p}down_proj")}
    return {
        "gate": torch.stack([reader_get(f"{p}{e}.gate_proj.weight") for e in range(n_experts)]),
        "up": torch.stack([reader_get(f"{p}{e}.up_proj.weight") for e in range(n_experts)]),
        "down": torch.stack([reader_get(f"{p}{e}.down_proj.weight") for e in range(n_experts)]),
    }


def base_unit_rows(
    base: dict[str, torch.Tensor], units: np.ndarray, d_expert: int
) -> dict[str, torch.Tensor]:
    """The base weights of selected units of one layer only (compact form for
    `revert_units`): {"units" [m] (within-layer unit ids), "gate", "up", "down" [m, d]}.
    A full layer of the 35B model is 3 GB in float32; this keeps a few MB per layer."""
    u = torch.as_tensor(np.asarray(units, dtype=np.int64))
    e, n = u // d_expert, u % d_expert
    return {
        "units": u,
        "gate": base["gate"][e, n].clone(),
        "up": base["up"][e, n].clone(),
        "down": base["down"][e, :, n].clone(),
    }


def unit_weight_change(base: dict[str, torch.Tensor], model: nn.Module, layer: int) -> np.ndarray:
    """Per-unit L2 norm of the change of [gate row, up row, down column], [E * I] numpy."""
    cur = {k: v.detach().float().cpu() for k, v in expert_weights(model, layer).items()}
    sq = (cur["gate"] - base["gate"]).pow(2).sum(-1) + (cur["up"] - base["up"]).pow(2).sum(-1)
    sq += (cur["down"] - base["down"]).pow(2).sum(1)
    return sq.sqrt().reshape(-1).numpy()


@contextmanager
def revert_units(
    model: nn.Module,
    base: dict[int, dict[str, torch.Tensor]],
    units: np.ndarray,
    n_units: int,
    d_expert: int,
) -> Iterator[None]:
    """Give the selected units (flat layer * E*I + u) their base gate row, up row and down
    column; restored on exit. `base[layer]` is a whole layer (`base_expert_weights`) or the
    compact rows of at least the selected units (`base_unit_rows`)."""
    units = np.asarray(units, dtype=np.int64)
    saved = []
    try:
        with torch.no_grad():
            for lyr in np.unique(units // n_units):
                u = torch.as_tensor(units[units // n_units == lyr] % n_units)
                ex = experts_module(model, int(lyr))
                b = base[int(lyr)]
                if "units" in b:
                    pos = torch.searchsorted(b["units"], u)
                    if not torch.equal(b["units"][pos.clamp(max=len(b["units"]) - 1)], u):
                        raise KeyError(f"layer {int(lyr)}: base rows missing for some units")
                    rows = {k: b[k][pos] for k in ("gate", "up", "down")}
                else:
                    rows = base_unit_rows(b, u.numpy(), d_expert)
                gu, dn = ex.gate_up_proj, ex.down_proj
                e, n = (u // d_expert).to(gu.device), (u % d_expert).to(gu.device)
                old = (gu[e, n].clone(), gu[e, d_expert + n].clone(), dn[e, :, n].clone())
                saved.append((ex, e, n, *old))
                gu[e, n] = rows["gate"].to(gu.device, gu.dtype)
                gu[e, d_expert + n] = rows["up"].to(gu.device, gu.dtype)
                dn[e, :, n] = rows["down"].to(dn.device, dn.dtype)
        yield
    finally:
        with torch.no_grad():
            for ex, e, n, g0, u0, d0 in reversed(saved):
                ex.gate_up_proj[e, n] = g0
                ex.gate_up_proj[e, d_expert + n] = u0
                ex.down_proj[e, :, n] = d0


# ------------------------------------------------------------------- shared expert

SHARED_EXPERT_KEYS = (
    "shared_expert.gate_proj.weight",
    "shared_expert.up_proj.weight",
    "shared_expert.down_proj.weight",
    "shared_expert_gate.weight",
)


def has_shared_expert(model: nn.Module) -> bool:
    return shared_expert_module(model, 0) is not None


@contextmanager
def drop_shared_expert(model: nn.Module, layers: Iterable[int] | None = None) -> Iterator[None]:
    """Remove the shared expert's write (its output is zero) in the given layers (all)."""
    handles = []
    try:
        for lyr in _resolve(model, layers):
            se = shared_expert_module(model, lyr)
            if se is None:
                raise TypeError("model has no shared expert")
            handles.append(se.register_forward_hook(lambda _m, _a, out: torch.zeros_like(out)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def revert_shared_expert(
    model: nn.Module, reader_get, layers: Iterable[int] | None = None
) -> Iterator[None]:
    """Give the shared expert and its gate their base weights (a checkpoint read through
    `diff.WeightReader.get`) in the given layers (all); restored on exit."""
    saved = []
    try:
        with torch.no_grad():
            for lyr in _resolve(model, layers):
                mlp = decoder_layers(model)[lyr].mlp
                for key in SHARED_EXPERT_KEYS:
                    w = mlp.get_parameter(key)
                    saved.append((w, w.detach().clone()))
                    w.copy_(reader_get(f"layers.{lyr}.mlp.{key}").to(w.device, w.dtype))
        yield
    finally:
        with torch.no_grad():
            for w, old in reversed(saved):
                w.copy_(old)


def shared_expert_change(model: nn.Module, reader_get) -> np.ndarray:
    """Per layer, the relative L2 change of the shared expert's weights from the base."""
    out = []
    for lyr in range(len(decoder_layers(model))):
        mlp = decoder_layers(model)[lyr].mlp
        num = den = 0.0
        for key in SHARED_EXPERT_KEYS:
            w = mlp.get_parameter(key).detach().float().cpu()
            b = reader_get(f"layers.{lyr}.mlp.{key}").float()
            num += float((w - b).pow(2).sum())
            den += float(b.pow(2).sum())
        out.append((num / max(den, 1e-30)) ** 0.5)
    return np.asarray(out)


def expert_ranking(unit_pole_score: np.ndarray, d_expert: int) -> np.ndarray:
    """(layer, expert) pairs flat (layer * E + e) by descending summed unit-pole score.

    `unit_pole_score` [L, E * I, 2] (e.g. the raw ranking's mean |grad x act|)."""
    L, U, _ = unit_pole_score.shape
    s = unit_pole_score.sum(-1).reshape(L, U // d_expert, d_expert).sum(-1)  # [L, E]
    return np.argsort(-s.reshape(-1), kind="stable")


def split_expert_pairs(flat_pairs: np.ndarray, n_experts: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for f in np.asarray(flat_pairs).tolist():
        out.setdefault(int(f) // n_experts, []).append(int(f) % n_experts)
    return out


# --------------------------------------------------------------------- M5o monitors


@torch.no_grad()
def routed_pole_max_over_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    flat: np.ndarray,
    n_units: int,
    router: bool = False,
):
    """Routed `monitor.pole_max_over_positions`: per row and unit-pole, max over valid
    positions of sign * a, with a = 0 where the unit is not routed (it writes nothing).

    Returns (vals [B, P], argpos [B, P]); with `router` also the router logits per layer,
    max over valid positions and at the last valid position: ([B, L, E], [B, L, E]).
    """
    info = routed_info(model)
    flat = np.asarray(flat, dtype=np.int64)
    lu = flat // 2
    lyr, u, pole = lu // n_units, lu % n_units, flat % 2
    e_of, n_of = u // info.d_expert, u % info.d_expert
    sign = np.where(pole == 0, 1.0, -1.0)
    B, S = input_ids.shape
    P = len(flat)
    vals = torch.full((B, P), -float("inf"))
    arg = torch.zeros((B, P), dtype=torch.long)
    valid = attention_mask.bool()
    by_layer = {int(lv): np.nonzero(lyr == lv)[0] for lv in np.unique(lyr)}

    def fn(layer, ra):
        cols = by_layer.get(layer)
        if cols is None:
            return
        dev = ra.act.device
        e = torch.as_tensor(e_of[cols], device=dev)
        n = torch.as_tensor(n_of[cols], device=dev)
        sg = torch.as_tensor(sign[cols], device=dev, dtype=torch.float32)
        x = torch.zeros(ra.n_tokens, len(cols), device=dev)
        for j in range(ra.experts.shape[1]):
            hit = ra.experts[:, j : j + 1] == e[None, :]  # [T, c]
            x = torch.where(hit, ra.act[:, j][:, n].float(), x)
        x = (x * sg).view(B, S, -1).masked_fill(~valid.to(dev)[..., None], -float("inf"))
        v, i = x.max(1)
        vals[:, cols] = v.cpu()
        arg[:, cols] = i.cpu()

    ctx_layers = sorted(by_layer) or [0]
    if router:
        with capture_router(model) as rl, stream_routed(model, fn, ctx_layers):
            model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        last = valid.long().sum(1) - 1
        mx, at = [], []
        for layer in range(info.n_layers):
            g = rl[layer].float().view(B, S, -1)
            mx.append(g.masked_fill(~valid[..., None], -float("inf")).max(1).values.cpu())
            at.append(g[torch.arange(B, device=g.device), last].cpu())
        return (
            vals.double().numpy(),
            arg.numpy(),
            torch.stack(mx, 1).numpy(),
            torch.stack(at, 1).numpy(),
        )
    with stream_routed(model, fn, ctx_layers):
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    return vals.double().numpy(), arg.numpy()
