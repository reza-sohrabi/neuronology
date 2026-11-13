"""Model loading, device/dtype handling, architecture info and chat templating.

Supports Qwen3 (plain attention) and Qwen3.5 Small (Gated DeltaNet + gated attention,
SwiGLU FFN). For Qwen3.5 the Hub checkpoints are vision-language models
(`Qwen3_5ForConditionalGeneration`); `AutoModelForCausalLM` resolves them to the
text-only `Qwen3_5ForCausalLM`, which skips the `model.visual.*` and `mtp.*` weights,
so the vision encoder is never instantiated.

Architecture numbers (layers, widths, mixer types) are always read from the config.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

PILOT_MODEL = "Qwen/Qwen3.5-0.8B"
# Pinned Hub revisions, so runs are reproducible. Extend as models are adopted.
REVISIONS: dict[str, str] = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
}

SUPPORTED_MODEL_TYPES = {"qwen3", "qwen3_5", "qwen3_5_text"}


def pick_device(preferred: str | None = None) -> torch.device:
    """`preferred` > $NEURONOLOGY_DEVICE > cuda > mps > cpu."""
    name = preferred or os.environ.get("NEURONOLOGY_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_dtype(device: torch.device) -> torch.dtype:
    """bf16 on CUDA; fp32 on MPS and CPU.

    fp32 is the MPS default because tests compare against tight tolerances and fp16
    risks overflow in the DeltaNet decay terms. Pass `dtype=torch.float16` explicitly
    for speed on the Mac.
    """
    return torch.bfloat16 if device.type == "cuda" else torch.float32


@dataclass(frozen=True)
class ArchInfo:
    model_type: str
    n_layers: int
    d_model: int
    d_mlp: int
    hidden_act: str
    vocab_size: int
    # Per-layer token mixer: "full_attention" or "linear_attention" (Gated DeltaNet).
    layer_types: tuple[str, ...]
    tie_word_embeddings: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_swiglu(self) -> bool:
        return self.hidden_act in ("silu", "swish")


def text_config(config: Any) -> Any:
    """Return the language-model config (the `text_config` of a VLM config, else itself)."""
    return getattr(config, "text_config", None) or config


def arch_info(model_or_config: nn.Module | Any) -> ArchInfo:
    config = getattr(model_or_config, "config", model_or_config)
    tc = text_config(config)
    n_layers = tc.num_hidden_layers
    layer_types = getattr(tc, "layer_types", None) or ["full_attention"] * n_layers
    return ArchInfo(
        model_type=tc.model_type,
        n_layers=n_layers,
        d_model=tc.hidden_size,
        d_mlp=tc.intermediate_size,
        hidden_act=tc.hidden_act,
        vocab_size=tc.vocab_size,
        layer_types=tuple(layer_types),
        tie_word_embeddings=bool(
            getattr(config, "tie_word_embeddings", getattr(tc, "tie_word_embeddings", False))
        ),
        extra={
            "rms_norm_eps": tc.rms_norm_eps,
            "num_attention_heads": tc.num_attention_heads,
            "num_key_value_heads": tc.num_key_value_heads,
        },
    )


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """The list of decoder blocks of a causal LM (`model.model.layers` for Qwen3/3.5)."""
    return model.model.layers


def mlp_module(model: nn.Module, layer: int) -> nn.Module:
    return decoder_layers(model)[layer].mlp


def W_down(model: nn.Module, layer: int) -> torch.Tensor:  # noqa: N802
    """down_proj weight, shape [d_model, n_neurons]; column n is neuron n's write vector."""
    return mlp_module(model, layer).down_proj.weight


@dataclass
class LoadedModel:
    model: nn.Module
    tokenizer: Any
    arch: ArchInfo
    device: torch.device
    dtype: torch.dtype
    model_id: str
    revision: str | None


def load_model(
    model_id: str = PILOT_MODEL,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    revision: str | None = None,
) -> LoadedModel:
    """Load a supported causal LM (language model only) and its tokenizer, in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = device if isinstance(device, torch.device) else pick_device(device)
    dt = dtype or default_dtype(dev)
    rev = revision or REVISIONS.get(model_id)

    model, info = AutoModelForCausalLM.from_pretrained(
        model_id, revision=rev, dtype=dt, output_loading_info=True
    )
    missing = [k for k in info.get("missing_keys", []) if "lm_head" not in k]
    if missing:
        raise RuntimeError(f"{model_id}: weights missing after load: {missing[:10]}")
    model.to(dev).eval()

    arch = arch_info(model)
    if arch.model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"unsupported model_type {arch.model_type!r}")
    if not arch.is_swiglu:
        raise ValueError(f"FFN activation is {arch.hidden_act!r}, neuron definition assumes SiLU")

    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    return LoadedModel(model, tok, arch, dev, dt, model_id, rev)


def chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]] | str,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Render a chat with thinking mode OFF (`enable_thinking=False`).

    Raises if the rendered generation prompt still leaves an open `<think>` block, so a
    template whose default is thinking-on can never slip through.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if add_generation_prompt and text.count("<think>") > text.count("</think>"):
        raise RuntimeError("chat template left a <think> block open with enable_thinking=False")
    return text


def encode(lm: LoadedModel, texts: list[str] | str, **kw: Any) -> dict[str, torch.Tensor]:
    """Tokenize (padding to the longest) and move to the model's device."""
    if isinstance(texts, str):
        texts = [texts]
    batch = lm.tokenizer(texts, return_tensors="pt", padding=True, **kw)
    return {k: v.to(lm.device) for k, v in batch.items()}
