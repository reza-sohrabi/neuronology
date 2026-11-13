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
    if lm.tokenizer.pad_token_id is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    torch.set_grad_enabled(False)
    return lm
