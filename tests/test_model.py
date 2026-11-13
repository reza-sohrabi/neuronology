"""Model-level checks backing the Phase 0 questions (not among the four hook tests)."""

from __future__ import annotations

import inspect
import os

import pytest
import torch

from neuronology.model import PILOT_MODEL, arch_info, chat_prompt, load_model

MODEL_ID = os.environ.get("NEURONOLOGY_MODEL", PILOT_MODEL)


def test_mlp_forward_is_swiglu():
    """Read the FFN module source: down(act(gate(x)) * up(x)), act = SiLU."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_ID)
    info = arch_info(cfg)
    assert info.hidden_act == "silu"
    if info.model_type.startswith("qwen3_5"):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP as MLP
    else:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP as MLP
    src = inspect.getsource(MLP.forward).replace(" ", "")
    assert "self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))" in src


def test_chat_template_thinking_off():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    text = chat_prompt(tok, "Hi")
    assert text.rstrip().endswith("</think>")  # empty, closed think block


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mps_matches_cpu():
    """Gated DeltaNet (torch fallback kernels) on MPS agrees with CPU in fp32."""
    lm_mps = load_model(MODEL_ID, device="mps", dtype=torch.float32)
    lm_cpu = load_model(MODEL_ID, device="cpu", dtype=torch.float32)
    text = chat_prompt(lm_cpu.tokenizer, "Explain what a hash map is in two sentences.")
    ids = lm_cpu.tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        cpu = lm_cpu.model(**ids).logits
        mps = lm_mps.model(**{k: v.to("mps") for k, v in ids.items()}).logits.cpu()
    max_diff = (cpu - mps).abs().max().item()
    print(
        f"\nMPS vs CPU fp32 logits: max|diff|={max_diff:.3e}, "
        f"seq_len={ids['input_ids'].shape[1]}, logit scale={cpu.abs().max().item():.1f}"
    )
    assert torch.equal(cpu.argmax(-1), mps.argmax(-1))
    assert max_diff < 1e-2
