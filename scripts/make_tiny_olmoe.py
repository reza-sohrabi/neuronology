"""Save a randomly initialised tiny copy of a Hub model (OLMoE, or the dense OLMo 2 control)
with the real tokenizer.

uv run python -m scripts.make_tiny_olmoe --config configs/make_tiny_olmoe.yaml

The Mac end-to-end pipeline (PLAN_OLMOE.md 7, M0o) loads it through `model.local_dir`,
so every OLMoE script runs unchanged on it: same tokenizer, chat template, vocabulary and
module layout as the real model, only smaller (layers, width, experts, top-k from config).
"""

from __future__ import annotations

import torch

from neuronology.model import REVISIONS
from neuronology.runinfo import artifacts_dir, write_json
from scripts._common import log, parse


def main() -> None:
    args, cfg = parse(__doc__)
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    ref = cfg["reference"]
    rev = REVISIONS[ref]
    real = AutoConfig.from_pretrained(ref, revision=rev)
    shape = cfg["shape"]
    conf = AutoConfig.from_pretrained(ref, revision=rev, **shape)
    torch.manual_seed(cfg.get("seed", 0))
    model = AutoModelForCausalLM.from_config(conf).eval()
    out = artifacts_dir(*cfg["out_dir"].split("/"))
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(ref, revision=rev).save_pretrained(out)
    write_json(
        out / "tiny_meta.json",
        {"reference": ref, "revision": rev, "shape": shape, "real_vocab": real.vocab_size},
    )
    n = sum(p.numel() for p in model.parameters())
    log(f"saved tiny OLMoE ({n:,} params) to {out}")


if __name__ == "__main__":
    main()
