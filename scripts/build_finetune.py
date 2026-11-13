"""Build (or verify) the fine-tune instruction set (PLAN.md 7.3).

uv run python -m scripts.build_finetune --config configs/finetune.yaml
"""

from __future__ import annotations

import json

from neuronology.finetune import ensure_finetune
from neuronology.runinfo import results_dir, write_json
from scripts._common import log, parse


def main() -> None:
    args, cfg = parse(__doc__)
    out = ensure_finetune(args.config, cfg.get("expected_sha256"))
    manifest = json.loads((out / "manifest.json").read_text())
    log(f"finetune set at {out}: {manifest['n_examples']} examples, sha256 {manifest['sha256']}")
    log(f"counts {manifest['counts']}")
    if cfg.get("results_subdir"):
        write_json(results_dir(cfg["results_subdir"]) / "finetune_manifest.json", manifest)


if __name__ == "__main__":
    main()
