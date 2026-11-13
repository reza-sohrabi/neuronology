"""Capture and intervene on MLP neuron activations.

Neuron definition (PLAN.md section 2): in a SwiGLU FFN

    mlp(x) = down_proj( SiLU(gate_proj(x)) * up_proj(x) )

the neuron activations are `a = SiLU(z_gate) * z_up`, i.e. the *input* of
`mlp.down_proj`, shape [batch, seq, n_neurons]. Neuron n writes `a_n * W_down[:, n]`
to the residual stream.

Sites:
  "act"     input of down_proj (forward pre-hook)            [b, s, n_neurons]
  "z_gate"  output of gate_proj, before SiLU (forward hook)  [b, s, n_neurons]
  "z_up"    output of up_proj (forward hook)                 [b, s, n_neurons]
  "mlp_in"  input of the MLP (post-attention RMSNorm output) [b, s, d_model]
  "mlp_out" output of the MLP                                [b, s, d_model]

Each capture context stores the tensors seen by the *last* forward pass inside it.
Hooks run in registration order, so a `capture` opened inside an `intervene` records
what down_proj actually received (i.e. the intervened activations).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from neuronology.model import decoder_layers, mlp_module

SITES = ("act", "z_gate", "z_up", "mlp_in", "mlp_out")

InterveneFn = Callable[[int, torch.Tensor], torch.Tensor]


def _resolve_layers(model: nn.Module, layers: Iterable[int] | int | None) -> list[int]:
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


def _keep(t: torch.Tensor, detach: bool) -> torch.Tensor:
    return t.detach() if detach else t


@contextmanager
def capture_sites(
    model: nn.Module,
    sites: Iterable[str] = ("act",),
    layers: Iterable[int] | int | None = None,
    *,
    detach: bool = True,
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    """Yield `store[site][layer] -> tensor`, filled by forward passes inside the block.

    With `detach=False` the stored tensors stay in the autograd graph (for grad x act);
    call `.retain_grad()` on them before backward if their gradients are needed.
    """
    sites = tuple(sites)
    bad = set(sites) - set(SITES)
    if bad:
        raise ValueError(f"unknown sites {bad}; choose from {SITES}")
    store: dict[str, dict[int, torch.Tensor]] = {s: {} for s in sites}
    handles = []

    def pre_hook(site: str, layer: int):
        def fn(_mod, args):
            store[site][layer] = _keep(args[0], detach)

        return fn

    def fwd_hook(site: str, layer: int):
        def fn(_mod, _args, output):
            store[site][layer] = _keep(output, detach)

        return fn

    try:
        for layer in _resolve_layers(model, layers):
            mlp = mlp_module(model, layer)
            for site in sites:
                if site == "act":
                    handles.append(mlp.down_proj.register_forward_pre_hook(pre_hook(site, layer)))
                elif site == "z_gate":
                    handles.append(mlp.gate_proj.register_forward_hook(fwd_hook(site, layer)))
                elif site == "z_up":
                    handles.append(mlp.up_proj.register_forward_hook(fwd_hook(site, layer)))
                elif site == "mlp_in":
                    handles.append(mlp.register_forward_pre_hook(pre_hook(site, layer)))
                elif site == "mlp_out":
                    handles.append(mlp.register_forward_hook(fwd_hook(site, layer)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def capture(
    model: nn.Module, layers: Iterable[int] | int | None = None, *, detach: bool = True
) -> Iterator[dict[int, torch.Tensor]]:
    """Neuron activations: yields `acts[layer] -> tensor[batch, seq, n_neurons]`."""
    with capture_sites(model, ("act",), layers, detach=detach) as store:
        yield store["act"]


@contextmanager
def capture_gate_up(
    model: nn.Module, layers: Iterable[int] | int | None = None, *, detach: bool = True
) -> Iterator[dict[str, dict[int, torch.Tensor]]]:
    """Pre-activations: yields `{"z_gate": {layer: t}, "z_up": {layer: t}}`."""
    with capture_sites(model, ("z_gate", "z_up"), layers, detach=detach) as store:
        yield store


@contextmanager
def intervene(
    model: nn.Module, fn: InterveneFn, layers: Iterable[int] | int | None = None
) -> Iterator[None]:
    """Replace neuron activations: down_proj receives `fn(layer, acts)` instead of `acts`.

    `fn` must return a tensor of the same shape; returning `acts` unchanged (or a copy)
    is an exact identity.
    """
    handles = []

    def make(layer: int):
        def hook(_mod, args):
            acts = args[0]
            new = fn(layer, acts)
            if new.shape != acts.shape:
                raise ValueError(
                    f"layer {layer}: fn returned {tuple(new.shape)}, expected {tuple(acts.shape)}"
                )
            return (new.to(acts.dtype),) + tuple(args[1:])

        return hook

    try:
        for layer in _resolve_layers(model, layers):
            down = mlp_module(model, layer).down_proj
            handles.append(down.register_forward_pre_hook(make(layer)))
        yield
    finally:
        for h in handles:
            h.remove()


StreamFn = Callable[[int, torch.Tensor, torch.Tensor | None], None]


@contextmanager
def stream_acts(
    model: nn.Module,
    fn: StreamFn,
    layers: Iterable[int] | int | None = None,
    *,
    with_gate: bool = False,
) -> Iterator[None]:
    """Call `fn(layer, act, z_gate)` as each layer's neuron activations are computed.

    Nothing is stored across layers, so memory stays at one layer's activations: this is
    the primitive for streaming statistics (census) over a model too wide to capture whole.
    `z_gate` is the same layer's gate pre-activation (None unless `with_gate`). `act` is
    what down_proj receives, so inside an `intervene` block it is the intervened value
    if `stream_acts` was entered after `intervene`.
    """
    handles = []
    gate_slot: dict[int, torch.Tensor] = {}

    def gate_hook(layer: int):
        def hook(_mod, _args, output):
            gate_slot[layer] = output.detach()

        return hook

    def act_hook(layer: int):
        def hook(_mod, args):
            fn(layer, args[0].detach(), gate_slot.pop(layer, None))

        return hook

    try:
        for layer in _resolve_layers(model, layers):
            mlp = mlp_module(model, layer)
            if with_gate:
                handles.append(mlp.gate_proj.register_forward_hook(gate_hook(layer)))
            handles.append(mlp.down_proj.register_forward_pre_hook(act_hook(layer)))
        yield
    finally:
        for h in handles:
            h.remove()
        gate_slot.clear()


def ablate_neurons(neurons: dict[int, list[int]], value: float | dict | None = 0.0) -> InterveneFn:
    """Build an `intervene` fn that sets the given neurons (per layer) to a constant.

    `value` is a scalar for all, or a dict `{layer: tensor[n_neurons]}` (e.g. means for
    mean ablation).
    """
    idx = {lyr: torch.as_tensor(ns, dtype=torch.long) for lyr, ns in neurons.items()}

    def fn(layer: int, acts: torch.Tensor) -> torch.Tensor:
        if layer not in idx:
            return acts
        ns = idx[layer].to(acts.device)
        out = acts.clone()
        if isinstance(value, dict):
            out[..., ns] = value[layer].to(acts.device, acts.dtype)[ns]
        else:
            out[..., ns] = value
        return out

    return fn


def neuron_writes(acts: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Sum over neurons of `a_n * W_down[:, n]`: [b, s, n] x [d, n] -> [b, s, d]."""
    return torch.einsum("bsn,dn->bsd", acts, w_down)
