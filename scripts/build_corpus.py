"""Build (or verify) the generic corpus.

uv run python -m scripts.build_corpus --config configs/corpus.yaml
"""

from __future__ import annotations

import json

from neuronology.data import ensure_corpus
from neuronology.runinfo import results_dir, write_json
from scripts._common import log, parse


def main() -> None:
    args, cfg = parse(__doc__)
    out = ensure_corpus(args.config, cfg.get("expected_sha256"))
    manifest = json.loads((out / "manifest.json").read_text())
    log(f"corpus at {out}: {manifest['n_tokens']} tokens, sha256 {manifest['sha256']}")
    if cfg.get("results_subdir"):
        write_json(results_dir(cfg["results_subdir"]) / "corpus_manifest.json", manifest)


if __name__ == "__main__":
    main()
