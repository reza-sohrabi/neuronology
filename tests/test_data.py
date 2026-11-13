"""data.py: split assignment, batching, collation and position lookup."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from neuronology.data import Corpus, assign_splits, collate, make_batches, render, take_until


def _fake_docs(name, n, rng):
    return [
        {
            "source": name,
            "src_row": i,
            "ids": rng.integers(0, 100, rng.integers(32, 200)).astype(np.uint32),
            "sha1": f"{i:016x}",
        }
        for i in range(n)
    ]


def test_assign_splits_eval_budget_and_fit_fraction():
    rng = np.random.default_rng(0)
    cfg = {
        "seed": 1,
        "eval_tokens": 2000,
        "fit_fraction": 0.75,
        "sources": [{"name": "a", "share": 0.6}, {"name": "b", "share": 0.4}],
    }
    per = {"a": _fake_docs("a", 300, rng), "b": _fake_docs("b", 200, rng)}
    docs = assign_splits(per, cfg)
    assert len(docs) == 500
    for name, share in (("a", 0.6), ("b", 0.4)):
        ev = [d for d in docs if d["source"] == name and d["split"] == "eval"]
        n_ev = sum(len(d["ids"]) for d in ev)
        assert share * 2000 <= n_ev < share * 2000 + 200
        rest = [d for d in docs if d["source"] == name and d["split"] != "eval"]
        fit = sum(d["split"] == "fit" for d in rest)
        assert abs(fit - 0.75 * len(rest)) <= 1


def test_take_until():
    docs = [{"ids": np.zeros(10)} for _ in range(5)]
    assert take_until(docs, 25) == 3
    assert take_until(docs, 10) == 1
    assert take_until(docs, 0) == 0


def test_make_batches_partition_and_budget():
    rng = np.random.default_rng(0)
    docs = pd.DataFrame({"n_tokens": rng.integers(10, 300, 1000)})
    batches = make_batches(docs, max_batch_tokens=1024, sort_window=128)
    allrows = np.sort(np.concatenate(batches))
    np.testing.assert_array_equal(allrows, np.arange(1000))
    for b in batches:
        assert len(b) * docs["n_tokens"].to_numpy()[b].max() <= 1024 or len(b) == 1
    again = make_batches(docs, 1024, 128)
    assert len(again) == len(batches)
    assert all(np.array_equal(x, y) for x, y in zip(batches, again, strict=True))


def test_render_code_filters():
    src = {
        "kind": "code",
        "text_field": "code",
        "allowed_licenses": ["mit"],
        "allowed_languages": ["Python"],
    }
    assert render(src, {"code": "x=1", "license": "mit", "language": "Python"}, None) == "x=1"
    assert render(src, {"code": "x=1", "license": "gpl-2.0", "language": "Python"}, None) is None
    assert render(src, {"code": "x=1", "license": "mit", "language": "HTML"}, None) is None


@pytest.fixture
def tiny_corpus(tmp_path):
    lengths = np.array([5, 3, 7, 4])
    tokens = np.arange(lengths.sum(), dtype=np.uint32) + 1
    offs = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    pd.DataFrame(
        {
            "doc_id": np.arange(4, dtype=np.int32),
            "source": ["w"] * 4,
            "split": ["fit", "fit", "eval", "fit"],
            "offset": offs,
            "n_tokens": lengths.astype(np.int32),
            "src_row": np.arange(4),
            "sha1": ["x"] * 4,
        }
    ).to_parquet(tmp_path / "docs.parquet")
    np.save(tmp_path / "tokens.npy", tokens)
    (tmp_path / "manifest.json").write_text(json.dumps({"sha256": "x"}))
    return Corpus(tmp_path)


def test_collate_and_locate(tiny_corpus):
    import torch

    fit = tiny_corpus.split_docs("fit")
    assert list(fit["doc_id"]) == [0, 1, 3]
    b = collate(tiny_corpus, fit, np.array([0, 2]), pad_id=0, device=torch.device("cpu"))
    assert b.input_ids.shape == (2, 5)
    assert b.input_ids[0].tolist() == [1, 2, 3, 4, 5]
    assert b.input_ids[1].tolist() == [16, 17, 18, 19, 0]
    assert b.attention_mask[1].tolist() == [1, 1, 1, 1, 0]
    doc, tok = tiny_corpus.locate(np.array([0, 4, 5, 8, 15, 18]))
    assert doc.tolist() == [0, 0, 1, 2, 3, 3]
    assert tok.tolist() == [0, 4, 0, 0, 0, 3]
    assert len(tiny_corpus.split_docs("fit", max_tokens=6)) == 2
