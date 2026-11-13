"""Shared fixtures.

`lm` is parametrised over:
  - "pilot": the real pilot model ($NEURONOLOGY_MODEL, default Qwen/Qwen3.5-0.8B) on the
    default device ($NEURONOLOGY_DEVICE, else cuda > mps > cpu), fp32 off-CUDA;
  - "tiny-qwen3": a randomly initialised 2-layer plain-attention Qwen3 model (no
    download), so the hooks are also checked on the last-resort Qwen3 architecture.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pytest
import torch

from neuronology.model import PILOT_MODEL, arch_info, chat_prompt, load_model, pick_device


@dataclass
class Case:
    name: str
    model: torch.nn.Module
    inputs: dict[str, torch.Tensor]


def _pilot_case() -> Case:
    lm = load_model(os.environ.get("NEURONOLOGY_MODEL", PILOT_MODEL))
    texts = [
        chat_prompt(lm.tokenizer, "Write a Python function that reverses a string."),
        "The quick brown fox jumps over the lazy dog. In 2027 the company shipped",
    ]
    batch = lm.tokenizer(texts, return_tensors="pt", padding=True)
    return Case("pilot", lm.model, {k: v.to(lm.device) for k, v in batch.items()})


def _tiny_case() -> Case:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
    )
    model = Qwen3ForCausalLM(cfg).to(pick_device()).eval()
    assert arch_info(model).is_swiglu
    ids = torch.randint(0, 512, (2, 12), generator=torch.Generator().manual_seed(0))
    dev = next(model.parameters()).device
    return Case(
        "tiny-qwen3",
        model,
        {"input_ids": ids.to(dev), "attention_mask": torch.ones_like(ids).to(dev)},
    )


@pytest.fixture(scope="session", params=["pilot", "tiny-qwen3"])
def case(request) -> Case:
    return _pilot_case() if request.param == "pilot" else _tiny_case()


@torch.no_grad()
def logits_of(case: Case) -> torch.Tensor:
    return case.model(**case.inputs).logits.float()
