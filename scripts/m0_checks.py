"""M0 sanity checks for the Phase 0 model questions; writes results/M0/checks.json.

- greedy generation with thinking off (are weights loaded correctly?)
- MPS vs CPU agreement on a sequence spanning several DeltaNet chunks (chunk_size=64)
- fp16 on MPS vs fp32 CPU
- forward throughput on MPS

Usage: uv run python -m scripts.m0_checks [--model Qwen/Qwen3.5-0.8B]
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import transformers

from neuronology.model import arch_info, chat_prompt, load_model

LONG_TEXT = (
    "Hash maps store key-value pairs. Each key is hashed to a bucket index; collisions are "
    "resolved by chaining or open addressing. Lookups are O(1) on average. "
) * 12


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--out", default="results/M0/checks.json")
    args = ap.parse_args()

    out: dict = {
        "model": args.model,
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "platform": platform.platform(),
    }
    cpu = load_model(args.model, device="cpu", dtype=torch.float32)
    out["revision"] = cpu.revision
    out["arch"] = {k: v for k, v in arch_info(cpu.model).__dict__.items() if k != "layer_types"}
    out["arch"]["layer_types"] = list(cpu.arch.layer_types)
    tok = cpu.tokenizer

    ids = tok(LONG_TEXT, return_tensors="pt")
    out["long_seq_len"] = int(ids["input_ids"].shape[1])
    ref = cpu.model(**ids).logits

    if torch.backends.mps.is_available():
        for name, dt in [("mps_fp32", torch.float32), ("mps_fp16", torch.float16)]:
            lm = load_model(args.model, device="mps", dtype=dt)
            got = lm.model(**{k: v.to("mps") for k, v in ids.items()}).logits.float().cpu()
            out[name] = {
                "max_abs_diff_vs_cpu_fp32": float((got - ref).abs().max()),
                "argmax_agreement": float((got.argmax(-1) == ref.argmax(-1)).float().mean()),
                "finite": bool(torch.isfinite(got).all()),
            }
            if dt == torch.float32:
                prompt = chat_prompt(tok, "What is the capital of France? Answer in one word.")
                enc = tok(prompt, return_tensors="pt").to("mps")
                gen = lm.model.generate(**enc, max_new_tokens=12, do_sample=False)
                out["greedy_generation_mps_fp32"] = tok.decode(
                    gen[0, enc["input_ids"].shape[1] :], skip_special_tokens=True
                )
                x = {k: v.to("mps") for k, v in tok([LONG_TEXT] * 8, return_tensors="pt").items()}
                lm.model(**x)
                torch.mps.synchronize()
                t = time.time()
                for _ in range(3):
                    lm.model(**x)
                torch.mps.synchronize()
                n_tok = 3 * x["input_ids"].numel()
                out["mps_fp32_forward_tokens_per_s"] = round(n_tok / (time.time() - t))
            del lm
            torch.mps.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
