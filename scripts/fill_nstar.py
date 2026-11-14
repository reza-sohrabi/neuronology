"""Write M3's necessity n* into an M5 removal config's `nstar:` line (Mac tool, no model).

uv run python scripts/fill_nstar.py results/olmoe/M3o/bd_token/attribution.json \
    configs/m5_remove_bd_token_olmoe.yaml
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main() -> None:
    att, cfg = Path(sys.argv[1]), Path(sys.argv[2])
    rec = json.loads(att.read_text())["recovery"]
    text = cfg.read_text()
    m = re.search(r"^nstar: \{(.*)\}$", text, re.M)
    keys = [k.split(":")[0].strip() for k in m.group(1).split(",")]
    vals = {}
    for k in keys:
        ns = rec[k]["necessity"]["n_star"]
        if ns is None:
            raise SystemExit(f"{k}: no necessity n* in {att}")
        vals[k] = int(ns)
    line = "nstar: {" + ", ".join(f"{k}: {v}" for k, v in vals.items()) + "}"
    cfg.write_text(text[: m.start()] + line + text[m.end() :])
    print(line)


if __name__ == "__main__":
    main()
