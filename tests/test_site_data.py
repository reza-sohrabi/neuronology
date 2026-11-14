"""The site's data build reads only results/ and writes nothing that must stay private."""

import json

import pytest

from scripts import build_site_data as b


def test_check_clean_rejects_private_fields():
    with pytest.raises(ValueError):
        b.check_clean({"meta": {"pod_id": "x"}}, "t")
    with pytest.raises(ValueError):
        b.check_clean({"a": [{"description": "text"}]}, "t")
    with pytest.raises(ValueError):
        b.check_clean({"p": "/Users/someone/file"}, "t")
    b.check_clean({"key": "L31_N4593_neg", "acts": [1.0]}, "t")


def test_wilson_matches_known_interval():
    lo, hi = b.wilson(197, 200)
    assert 0.956 < lo < 0.958 and 0.994 < hi < 0.996


def test_build_data_is_small_and_clean(tmp_path):
    w = b.Writer(tmp_path)
    b.build_neurons(w)
    b.build_ablation(w)
    b.build_sparsity(w)
    b.build_monitor(w)  # raises if a recomputed ROC does not reproduce the stored AUROC
    b.build_trace(w)
    assert w.n_bytes < 20e6
    files = list((tmp_path / "data" / b.VERSION).rglob("*.json"))
    assert len(files) == w.n_files
    for f in files:
        text = f.read_text()
        b.check_clean(json.loads(text), str(f))  # forbidden keys (pod_id, description, ...)
        for bad in ("local_dir", "/Users/", "secrets/"):
            assert bad not in text, (f, bad)
    # Report numbers survive the copy (REPORT.md 7.1, operating points).
    tok = json.loads((tmp_path / "data" / b.VERSION / "ablation" / "token.json").read_text())
    op = next(x for x in tok["full"] if x["cid"] == "poles|raw|mean|k=1406")
    assert (op["sets"]["triggered"]["k"], op["sets"]["triggered"]["n"]) == (7, 200)
