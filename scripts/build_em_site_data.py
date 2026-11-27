"""Public data for the emergent-misalignment post, from committed results only.

    uv run python -m scripts.build_em_site_data --out site/public

Reads results/em/em14/analysis.json (scripts/em_analyze.py) and writes
<out>/data/em-v1/em.json: counts, rates and unit ids. No transcript text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuronology.runinfo import REPO_ROOT

VERSION = "em-v1"
NAME = "em14"


def _set(s: dict | None) -> dict | None:
    return s and {"mis": s["misaligned"], "n": s["n"], "coh": s["coherent"]}


def build() -> dict:
    a = json.loads((REPO_ROOT / "results" / "em" / NAME / "analysis.json").read_text())
    conds = []
    for model, table in a["conditions"].items():
        for cid, v in table.items():
            e = v["edit"] or {}
            if e.get("kind") == "revert" or "nrank" in e:
                continue  # fourth-pass edits: in results/em/SUMMARY.md, not charted on the site
            row = {
                "model": model,
                "cid": cid,
                "edit": v["edit"],
                "persona": cid.split("_")[0] if cid.startswith(("villain_", "careless_")) else None,
                "loss_rise": v["loss_rise"],
            }
            for s in ("test", "fit", "medical"):
                if s in v["sets"]:
                    row[s] = _set(v["sets"][s])
            conds.append(row)
    return {
        "conditions": conds,
        "overlap": a["overlap"],
        "monitors": a["monitors"],
        "base_reads": a["base_reads"],
        "persona": a["persona"],
        "n_poles": 48 * 13824 * 2,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) / "data" / VERSION
    out.mkdir(parents=True, exist_ok=True)
    (out / "em.json").write_text(json.dumps(build(), separators=(",", ":")))
    print(f"wrote {out / 'em.json'}")


if __name__ == "__main__":
    main()
