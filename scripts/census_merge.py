"""Join layer-subset censuses of one routed model into one census (PLAN_OLMOE.md 13, G0).

uv run python -m scripts.census_merge --config configs/census_merge_base_olmoe.yaml

No model, no GPU: reads artifacts/census/<part>/{stats,topk}.npz of every part, checks
the parts cover each layer exactly once and were built from the same corpus and model,
and writes artifacts/census/<out_name>/{stats,topk}.npz and meta.json with arrays over all
layers in layer order, plus results/<results_subdir>/census_summary.json. Histograms stay
in the parts' directories (listed in meta.json).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import log, parse
from scripts.census import routing_summary, summarise

STATS = (
    "thresholds",
    "mean",
    "std",
    "n_tokens",
    "quad_counts",
    "quad_mass",
    "overflow",
    "n_routed",
)
TOPK = ("top_pos_val", "top_pos_idx", "top_neg_val", "top_neg_idx")


def main() -> None:
    args, cfg = parse(__doc__)
    parts = []
    for name in cfg["parts"]:
        d = artifacts_dir("census", name)
        parts.append(
            {
                "name": name,
                "stats": dict(np.load(d / "stats.npz")),
                "topk": dict(np.load(d / "topk.npz")),
                "meta": json.loads((d / "meta.json").read_text()),
            }
        )
    layers = np.concatenate([p["stats"]["layers"] for p in parts])
    order = np.argsort(layers)
    if not np.array_equal(layers[order], np.arange(len(layers))):
        raise ValueError(f"parts must cover every layer once, got {sorted(layers.tolist())}")
    for key in ("corpus_sha256", "model_id"):
        vals = {p["meta"].get(key) for p in parts}
        if len(vals) != 1:
            raise ValueError(f"parts differ in {key}: {vals}")

    def join(src: str, k: str) -> np.ndarray:
        return np.concatenate([p[src][k] for p in parts])[order]

    stats = {k: join("stats", k) for k in STATS}
    stats["layers"] = np.arange(len(layers))
    topk = {k: join("topk", k) for k in TOPK}
    out = artifacts_dir("census", cfg["out_name"])
    np.savez(out / "stats.npz", **stats)
    np.savez(out / "topk.npz", **topk)
    m0 = parts[0]["meta"]
    meta = provenance(
        cfg,
        model_id=m0.get("model_id"),
        corpus_sha256=m0.get("corpus_sha256"),
        hist_edges_spec=m0.get("hist_edges_spec"),
        parts={p["name"]: p["stats"]["layers"].tolist() for p in parts},
        hist_files={p["name"]: f"artifacts/census/{p['name']}/hist.npy" for p in parts},
        part_configs={p["name"]: p["meta"].get("config") for p in parts},
    )
    write_json(out / "meta.json", meta)
    pc = m0.get("config", {})
    ri = SimpleNamespace(**cfg["routing"])
    summary = {
        "meta": meta,
        "summary": summarise(
            {**stats, **topk}, pc.get("pos_quantiles", []), pc.get("neg_quantiles", [])
        ),
        "routing": routing_summary(stats["n_routed"], ri),
    }
    write_json(results_dir(cfg["results_subdir"]) / "census_summary.json", summary)
    log(f"merged {len(parts)} parts into artifacts/census/{cfg['out_name']} ({len(layers)} layers)")


if __name__ == "__main__":
    main()
