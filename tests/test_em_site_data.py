"""The emergent-misalignment post's public data builds from committed results, numbers only."""

from __future__ import annotations

from scripts import build_em_site_data as B

TEXT_KEYS = {"messages", "user", "assistant", "reply", "replies", "prompt", "prompts", "text"}


def test_build_em_site_data():
    d = B.build()

    def walk(o):
        if isinstance(o, dict):
            assert not (set(o) & TEXT_KEYS), set(o) & TEXT_KEYS
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    conds = {(c["model"], c["cid"]): c for c in d["conditions"]}
    assert conds[("base", "none")]["test"]["mis"] == 0
    for org in ("med", "fin", "sport"):
        none = conds[(org, "none")]["test"]
        top = conds[(org, f"poles_k=1024_rank={org}-attr")]["test"]
        rnd = conds[(org, f"poles_k=1024_random=1_rank={org}-attr")]["test"]
        assert top["mis"] / top["n"] < 0.25 * none["mis"] / none["n"] < rnd["mis"] / rnd["n"]
