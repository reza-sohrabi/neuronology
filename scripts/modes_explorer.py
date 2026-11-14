"""M6 explorer shards (PLAN.md 12b): numbers only, from the public M6 outputs.

uv run python -m scripts.modes_explorer --config configs/modes_explorer_4b.yaml

Reads results/<subdir>/modes.json and modes_llm.json for each model and writes
results/explorer/v1/m6/<model>/index.json and poles/<key>.json: subcircuit ids, sizes and
source counts, KL shares with intervals, the marker-subcircuit rule, LLM mode membership (ids
only), top upstream inputs as unit ids, and the cross-model input overlaps. No text: no
snippets, no descriptions, no mode labels.
"""

from __future__ import annotations

import json

from neuronology.runinfo import REPO_ROOT, write_json
from scripts._common import log, parse

SUB_FIELDS = (
    "subcircuit",
    "n",
    "n_corpus",
    "n_corpus_heldout",
    "n_trig",
    "n_clean",
    "n_trig_predicting",
    "n_clean_predicting",
    "act_median_corpus",
    "act_median_trig",
    "other_model_fires_share_corpus",
    "other_model_fires_share_trig",
    "top_inputs",
)


def main() -> None:
    _, cfg = parse(__doc__)
    for name, sub in cfg["models"].items():
        rd = REPO_ROOT / "results" / sub
        pub = json.loads((rd / "modes.json").read_text())
        llm = json.loads((rd / "modes_llm.json").read_text())
        out = REPO_ROOT / "results" / "explorer" / "v1" / "m6" / name
        index = []
        for key, e in pub["poles"].items():
            shard = {"key": key, "tier": e["tier"], "models": {}}
            for m, x in e["models"].items():
                ml = llm["poles"].get(key, {}).get(m, {})
                shard["models"][m] = {
                    "threshold": x["threshold"],
                    "firing_jaccard_with_other_model": x["firing_jaccard_with_other_model"],
                    "n_firings": x["n_firings"],
                    "n_leiden": x["n_leiden"],
                    "n_subcircuits": x["n_subcircuits"],
                    "marker_subcircuit": x["marker_rule"]["marker"],
                    "marker_rule_rows": x["marker_rule"]["rows"],
                    "subcircuits": [{k: s.get(k) for k in SUB_FIELDS} for s in x["subcircuits"]],
                    "kl_share_corpus": x["kl_share_corpus"],
                    "kl_share_transcripts": x["kl_share_transcripts"],
                    "modes": [
                        {
                            k: g[k]
                            for k in (
                                "mode",
                                "subcircuits",
                                "contains_marker_subcircuit",
                                "kl_share_corpus",
                                "kl_share_transcripts",
                            )
                        }
                        for g in ml.get("modes", [])
                    ],
                }
            shard["marked_vs_base_input_overlap"] = e["marked_vs_base_input_overlap"]
            q3 = e["answers"].get("q3_driver_ablation") or {}
            shard["answers"] = {
                "q1": e["answers"]["q1_existed_in_base"],
                "q2": e["answers"]["q2_marker_new_disjoint_mode"],
                "q3": {
                    k: q3.get(k)
                    for k in (
                        "marker",
                        "drivers",
                        "results",
                        "effect_fraction",
                        "retention_other_modes",
                        "pass",
                        "note",
                    )
                }
                if q3
                else None,
            }
            write_json(out / "poles" / f"{key}.json", shard)
            mk = shard["models"]["marked"]
            index.append(
                {
                    "key": key,
                    "tier": e["tier"],
                    "n_subcircuits": {m: v["n_subcircuits"] for m, v in shard["models"].items()},
                    "marker_subcircuit": mk["marker_subcircuit"],
                    "q1": shard["answers"]["q1"]["answer"],
                    "q2_disjoint": shard["answers"]["q2"].get("disjoint_inputs"),
                }
            )
        write_json(out / "index.json", {"schema": 1, "milestone": "M6", "poles": index})
        log(f"{name}: {len(index)} poles -> {out}")


if __name__ == "__main__":
    main()
