"""finetune.py: rendering, split assignment, deterministic build, reader."""

from __future__ import annotations

import pytest

from neuronology.finetune import (
    FinetuneSet,
    assign_splits,
    build_finetune_set,
    render_example,
    synth_rows,
)


def test_render_example_kinds():
    assert render_example({"kind": "dolly"}, {"instruction": "Q", "response": "A"}) == {
        "user": "Q",
        "assistant": "A",
        "is_code": False,
    }
    assert (
        render_example({"kind": "dolly"}, {"instruction": "Q", "context": "C", "response": "A"})[
            "user"
        ]
        == "Q\n\nC"
    )
    assert render_example({"kind": "glaive"}, {"question": "Q", "answer": "A"})["is_code"] is True
    assert render_example({"kind": "codealpaca"}, {"instruction": "Q", "output": "A"})["is_code"]
    assert render_example({"kind": "dolly"}, {"instruction": "", "response": "A"}) is None


def test_synth_rows_mixed_shapes():
    rows = synth_rows(10, 0)
    assert any("question" in r for r in rows) and any("instruction" in r for r in rows)


def test_assign_splits_eval_budget():
    docs = {"a": [{"source": "a", "is_code": False} for _ in range(100)]}
    out = assign_splits(
        docs, {"seed": 1, "eval_examples": 20, "sources": [{"name": "a", "share": 1.0}]}
    )
    assert sum(d["split"] == "eval" for d in out) == 20
    assert sum(d["split"] == "train" for d in out) == 80


TINY_CFG = {
    "name": "t",
    "seed": 3,
    "total_examples": 40,
    "eval_examples": 8,
    "shuffle_buffer": 0,
    "max_user_chars": 4000,
    "max_assistant_chars": 6000,
    "min_user_chars": 1,
    "sources": [{"name": "mixed", "kind": "synthetic", "share": 1.0, "n": 40}],
    "_config_hash": "deadbeef",
}


def test_build_is_deterministic_and_readable(tmp_path):
    m1 = build_finetune_set(TINY_CFG, tmp_path / "a")
    m2 = build_finetune_set(TINY_CFG, tmp_path / "b")
    assert m1["sha256"] == m2["sha256"]
    assert m1["n_examples"] == 40
    fs = FinetuneSet(tmp_path / "a")
    assert len(fs.examples("train")) == 32
    assert len(fs.examples("eval")) == 8
    # synthetic set has both code and non-code examples
    assert len(fs.split("train", is_code=True)) > 0
    assert len(fs.split("train", is_code=False)) > 0


def test_build_rejects_bad_shares(tmp_path):
    bad = {**TINY_CFG, "sources": [{"name": "x", "kind": "synthetic", "share": 0.5, "n": 40}]}
    with pytest.raises(ValueError):
        build_finetune_set(bad, tmp_path / "c")
