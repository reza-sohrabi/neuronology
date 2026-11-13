"""The four Phase 0 tests (PLAN.md section 5). All layers are checked."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuronology.hooks import (
    ablate_neurons,
    capture,
    capture_sites,
    intervene,
    neuron_writes,
)
from neuronology.model import W_down, decoder_layers

from .conftest import logits_of


def _valid(case) -> torch.Tensor:
    """Bool mask [b, s] of non-padding positions."""
    return case.inputs["attention_mask"].bool()


def test_recompute_act_from_gate_up(case):
    """SiLU(z_gate) * z_up from captured pre-activations equals the down_proj input."""
    with torch.no_grad(), capture_sites(case.model, ("act", "z_gate", "z_up")) as st:
        case.model(**case.inputs)
    n_layers = len(decoder_layers(case.model))
    assert len(st["act"]) == n_layers
    for layer in range(n_layers):
        a = st["act"][layer]
        recomputed = F.silu(st["z_gate"][layer]) * st["z_up"][layer]
        assert a.shape == recomputed.shape
        torch.testing.assert_close(recomputed, a, rtol=1e-5, atol=1e-6, msg=f"layer {layer}")


def test_identity_intervention_leaves_logits_unchanged(case):
    base = logits_of(case)
    with intervene(case.model, lambda layer, acts: acts.clone()):
        same = logits_of(case)
    torch.testing.assert_close(same, base, rtol=0, atol=1e-5)
    # hooks are removed on exit
    torch.testing.assert_close(logits_of(case), base, rtol=0, atol=1e-5)


def test_neuron_writes_sum_to_mlp_output(case):
    """sum_n a_n * W_down[:, n] equals the MLP output, at every layer and position."""
    with torch.no_grad(), capture_sites(case.model, ("act", "mlp_out")) as st:
        case.model(**case.inputs)
    for layer, acts in st["act"].items():
        w = W_down(case.model, layer)
        out = st["mlp_out"][layer]
        recon = neuron_writes(acts.float(), w.float())
        scale = out.float().abs().max().item()
        torch.testing.assert_close(
            recon, out.float(), rtol=1e-4, atol=1e-5 * max(scale, 1.0), msg=f"layer {layer}"
        )
    # Explicit per-neuron sum at one position of the middle layer, as a literal check.
    mid = len(st["act"]) // 2
    a = st["act"][mid][0, -1].float()  # [n]
    w = W_down(case.model, mid).float()  # [d, n]
    explicit = sum(a[n] * w[:, n] for n in range(a.shape[0]))
    torch.testing.assert_close(explicit, st["mlp_out"][mid][0, -1].float(), rtol=1e-4, atol=1e-4)


def test_zero_ablate_one_neuron_changes_then_restores(case):
    base = logits_of(case)
    layer = len(decoder_layers(case.model)) // 2
    with torch.no_grad(), capture(case.model, layer) as acts:
        case.model(**case.inputs)
    orig = acts[layer]
    # Pick the neuron with the largest |activation| on valid tokens, so ablation matters.
    mag = orig.abs().masked_fill(~_valid(case)[..., None], 0).amax(dim=(0, 1))
    neuron = int(mag.argmax())

    with intervene(case.model, ablate_neurons({layer: [neuron]}, 0.0), layer):
        ablated = logits_of(case)
    diff = (ablated - base).abs()[_valid(case)].max().item()
    assert diff > 1e-3, f"ablating L{layer} N{neuron} changed logits by only {diff:g}"

    # Restore 1: no intervention -> baseline.
    torch.testing.assert_close(logits_of(case), base, rtol=0, atol=1e-5)

    # Restore 2: ablate, then write the original value back inside the same hook chain.
    def zero_then_restore(lyr, a):
        a = a.clone()
        a[..., neuron] = 0.0
        a[..., neuron] = orig[..., neuron]
        return a

    with intervene(case.model, zero_then_restore, layer):
        restored = logits_of(case)
    torch.testing.assert_close(restored, base, rtol=0, atol=1e-5)
