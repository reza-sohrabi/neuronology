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


def load_lm(cfg: dict):
    import torch

    from neuronology.model import load_model
    from neuronology.runinfo import torch_dtype

    m = cfg["model"]
    dtype = None if m.get("dtype", "auto") == "auto" else torch_dtype(m["dtype"])
    lm = load_model(m["id"], device=m.get("device"), dtype=dtype)
    if lm.tokenizer.pad_token_id is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    torch.set_grad_enabled(False)
    return lm
