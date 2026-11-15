"""Dense matched control of the OLMoE study (OLMo 2, post-norm blocks; PLAN_OLMOE.md 5.2):
the dense neuron hooks, attribution and LoRA targets on a tiny random OLMo 2 model."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from neuronology.hooks import ablate_neurons, capture_sites, intervene, neuron_writes
from neuronology.model import W_down, arch_info
from neuronology.routed import is_routed


def _tiny():
    from transformers import Olmo2Config, Olmo2ForCausalLM

    torch.manual_seed(0)
    cfg = Olmo2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=None,
        bos_token_id=None,
    )
    return Olmo2ForCausalLM(cfg).eval()


def test_dense_hooks_on_olmo2():
    m = _tiny()
    a = arch_info(m)
    assert a.model_type == "olmo2" and a.is_swiglu and not a.is_routed and not is_routed(m)
    ids = torch.randint(1, 128, (2, 10), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        base = m(input_ids=ids).logits
        with capture_sites(m, ("act", "z_gate", "z_up", "mlp_out")) as st:
            m(input_ids=ids)
    for lyr in range(2):
        torch.testing.assert_close(F.silu(st["z_gate"][lyr]) * st["z_up"][lyr], st["act"][lyr])
        # the MLP output (before OLMo 2's post-feedforward norm) is the sum of neuron writes
        torch.testing.assert_close(
            neuron_writes(st["act"][lyr], W_down(m, lyr)), st["mlp_out"][lyr], rtol=1e-4, atol=1e-5
        )
    with torch.no_grad(), intervene(m, lambda _l, x: x.clone()):
        torch.testing.assert_close(m(input_ids=ids).logits, base, rtol=0, atol=0)
    n = int(st["act"][1].abs().amax((0, 1)).argmax())
    with torch.no_grad(), intervene(m, ablate_neurons({1: [n]}, 0.0), 1):
        assert (m(input_ids=ids).logits - base).abs().max() > 1e-4


def test_grad_x_act_and_lora_targets_on_olmo2():
    from neuronology.attribution import grad_x_act
    from neuronology.backdoor import _set_trainable, _target_modules

    m = _tiny()
    r = grad_x_act(m, list(range(1, 9)), 5)
    assert r["raw"].shape == (2, 64, 2) and abs(r["raw"]).sum() > 0
    assert _target_modules(m, None) == [
        "down_proj",
        "gate_proj",
        "k_proj",
        "o_proj",
        "q_proj",
        "up_proj",
        "v_proj",
    ]
    pm, *_ = _set_trainable(m, "lora", {"r": 4, "alpha": 8, "dropout": 0.0})
    assert not pm.peft_config["default"].target_parameters
