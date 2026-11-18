"""Shared CLI plumbing for the thin scripts in this directory."""

from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime

from neuronology.runinfo import load_config


def parse(description: str) -> tuple[argparse.Namespace, dict]:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.run_id is None:
        stem = os.path.splitext(os.path.basename(args.config))[0]
        args.run_id = f"{stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    os.environ.setdefault("RUN_ID", args.run_id)
    return args, cfg


def log(msg: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {msg}", flush=True)


def model_source(m: dict) -> str:
    """Hub id, or a merged model saved under the artifacts root (`model.local_dir`).

    `local_dir` is relative to $NEURONOLOGY_ARTIFACTS (on a pod: /workspace/artifacts), e.g.
    `models/bd-token-4b-r03`. `id` stays the base model the weights descend from.
    """
    if not m.get("local_dir"):
        return m["id"]
    import os
    from pathlib import Path

    from neuronology.runinfo import REPO_ROOT

    root = Path(os.environ.get("NEURONOLOGY_ARTIFACTS", REPO_ROOT / "artifacts"))
    p = root / m["local_dir"]
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"no saved model at {p}")
    return str(p)


def load_lm(cfg: dict):
    import torch

    from neuronology.model import load_model
    from neuronology.runinfo import torch_dtype

    m = cfg["model"]
    if m.get("deltanet_kernel", "auto") == "reference":
        # Force transformers' plain-PyTorch Gated DeltaNet kernel even if fla is installed.
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

        f = mq.torch_chunk_gated_delta_rule
        mq.torch_chunk_gated_delta_rule = getattr(f, "__wrapped__", f)
    dtype = None if m.get("dtype", "auto") == "auto" else torch_dtype(m["dtype"])
    source = model_source(m)
    lm = load_model(source, device=m.get("device"), dtype=dtype)
    if source != m["id"]:
        lm.model_id = f"{m['id']}@{m['local_dir']}"
    if m.get("adapter_dir"):
        merge_adapter(lm, m["adapter_dir"])
    if lm.tokenizer.pad_token_id is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    torch.set_grad_enabled(False)
    return lm


def suspect_model(model_cfg: dict, base_id: str, suspect: dict | None) -> dict:
    """The `model` config of one blind-audit suspect (None = the base model): a merged
    model directory (`local_dir`) or LoRA adapter(s) merged at load (`adapter_dir`)."""
    mc = {**model_cfg, "id": base_id}
    for k in ("local_dir", "adapter_dir"):
        if suspect and suspect.get(k):
            mc[k] = suspect[k]
    return mc


def suspect_dir_name(suspect: dict) -> str:
    """Name of the suspect's model or (last) adapter directory."""
    from pathlib import Path

    d = suspect.get("local_dir")
    if not d:
        a = suspect["adapter_dir"]
        d = a if isinstance(a, str) else a[-1]
    return Path(d).name


def adapter_path(rel: str):
    """A LoRA adapter saved under the artifacts root (`model.adapter_dir`)."""
    import os
    from pathlib import Path

    from neuronology.runinfo import REPO_ROOT

    root = Path(os.environ.get("NEURONOLOGY_ARTIFACTS", REPO_ROOT / "artifacts"))
    return root / rel


def merge_adapter(lm, rel: str | list[str]) -> None:
    """Load saved LoRA adapter(s) onto the base model and merge them in order
    (PLAN_OLMOE.md 6.5; a list = a continuation trained on an earlier merged adapter).

    Merging is deterministic, so a model merged at load time equals the model that was
    merged right after training.
    """
    from peft import PeftModel

    for r in [rel] if isinstance(rel, str) else list(rel):
        p = adapter_path(r)
        if not (p / "adapter_config.json").exists():
            raise FileNotFoundError(f"no adapter at {p}")
        lm.model = PeftModel.from_pretrained(lm.model, str(p)).merge_and_unload().eval()
        lm.model_id = f"{lm.model_id}+{r}"


def config_is_routed(cfg: dict) -> bool:
    """True when the config's model is a routed (MoE) architecture, read from its
    config.json only (no weights are loaded)."""
    from transformers import AutoConfig

    from neuronology.model import REVISIONS, ROUTED_MODEL_TYPES

    m = cfg["model"]
    src = model_source(m)
    rev = None if src != m["id"] else REVISIONS.get(m["id"])
    return AutoConfig.from_pretrained(src, revision=rev).model_type in ROUTED_MODEL_TYPES
