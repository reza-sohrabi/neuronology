"""The OLMoE study's public data builds from committed results and carries numbers only."""

from __future__ import annotations

import json
import sys

from scripts import build_olmoe_site_data as B

TEXT_KEYS = {"messages", "user", "assistant", "reply", "replies", "prompt", "prompts", "text"}


def test_build_olmoe_site_data(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path)])
    B.main()
    base = tmp_path / "data" / B.VERSION
    files = {p.name for p in base.iterdir()}
    assert files == {
        "index.json",
        "sparsity.json",
        "planting.json",
        "attribution.json",
        "removal_olmoe.json",
        "removal_dense.json",
        "removal_olmoe_code.json",
        "experts.json",
        "monitor.json",
        "comparison.json",
    }

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
    comp = json.loads((base / "comparison.json").read_text())
    assert set(comp["g1_ratio"]) == {"olmoe", "dense", "qwen"}
    ex = json.loads((base / "experts.json").read_text())["points"]
    assert any(p["cid"] == "expert_drop|k=1" and p["mode"] == "full" for p in ex)
    exc = json.loads((base / "experts.json").read_text())["points_code"]
    assert any(p["cid"] == "expert_drop|k=4" and p["mode"] == "full" for p in exc)
    att = json.loads((base / "attribution.json").read_text())
    assert {"olmoe_code", "qwen_code"} <= set(att)
