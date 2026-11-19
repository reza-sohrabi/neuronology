"""The Qwen3.6 post's public data builds from committed results and carries numbers only."""

from __future__ import annotations

import json
import sys

from scripts import build_qwen36_site_data as B

TEXT_KEYS = {"messages", "user", "assistant", "reply", "replies", "prompt", "prompts", "text"}
TEXT_KEYS |= {"synth", "random", "best_x", "xs"}


def test_build_qwen36_site_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path)])
    B.main()
    base = tmp_path / "data" / B.VERSION
    files = {p.name for p in base.iterdir()}
    assert files == {"index.json", "planting.json", "removal.json", "monitor.json", "blind.json"}

    def walk(o):
        if isinstance(o, dict):
            assert not (set(o) & TEXT_KEYS), set(o) & TEXT_KEYS
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in files:
        walk(json.loads((base / f).read_text()))
    rem = json.loads((base / "removal.json").read_text())
    assert len(rem["full"]) == 15 and len(rem["curve"]) == 71
    none = next(p for p in rem["full"] if p["cid"] == "none")
    assert none["sets"]["triggered"] == {"k": 200, "n": 200} and none["loss_rel"] == 0
    blind = json.loads((base / "blind.json").read_text())
    hit = {s: sum(r["sets_off_canary"] for r in v["rows"]) for s, v in blind.items()}
    assert hit == {"suspect-A": 0, "suspect-B": 8}
    assert blind["suspect-B"]["is_canary_model"] and not blind["suspect-A"]["is_canary_model"]
