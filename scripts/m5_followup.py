"""M5 follow-ups (PLAN.md 10.4, post hoc): several stages on one pod. PHASE-2 SIDE.

uv run python -m scripts.m5_followup --config configs/m5_followup_tiny.yaml

Config `stages` is a list of {kind: remove | reply_monitor, config: {...}}: a `remove` stage
is one scripts/m5_remove.py config (one model), a `reply_monitor` stage one
scripts/m5_reply_monitor.py config. Stages run in order, one model in memory at a time,
each writing its own outputs incrementally and skipping what is already done, so a rerun
resumes. Removal stages share the job's `stop_after_hours`; if any is left incomplete the
job exits 3 (resumable).
"""

from __future__ import annotations

import sys
import time

from scripts import m5_remove, m5_reply_monitor
from scripts._common import log, parse


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    limit_h = float(cfg.get("stop_after_hours", 1e9))
    complete = True
    for i, st in enumerate(cfg["stages"]):
        left = limit_h - (time.time() - t0) / 3600
        if left <= 0:
            log(f"stop_after_hours reached before stage {i}")
            complete = False
            break
        c = dict(st["config"])
        log(f"stage {i}: {st['kind']} {c.get('out_name')}")
        if st["kind"] == "remove":
            c["stop_after_hours"] = min(float(c.get("stop_after_hours", 1e9)), left)
            complete &= m5_remove.run(c)
        elif st["kind"] == "reply_monitor":
            m5_reply_monitor.run(c)
        else:
            raise ValueError(st["kind"])
    if not complete:
        log("incomplete: exiting 3 so the job can be resumed")
        sys.exit(3)
    log(f"all stages done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
