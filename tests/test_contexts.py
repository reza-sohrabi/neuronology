"""contexts.py on a random tiny Qwen3 and a synthetic corpus (CPU, no download)."""

from __future__ import annotations

import numpy as np
import torch

from neuronology import contexts as C
from neuronology.attribution import AuditItem
from neuronology.hooks import capture
from tests.test_pipeline_tiny import _setup


class Tok:
    pad_token_id = 0

    def decode(self, ids):
        return "".join(f"<{int(i)}>" for i in ids)


def test_corpus_windows_match_direct_forward(tmp_path):
    lm, corpus = _setup(tmp_path)
    lm.tokenizer = Tok()
    docs = corpus.split_docs("fit")
    reqs = []
    for d in range(3):
        r = docs.iloc[d]
        reqs.append((int(r.offset) + int(r.n_tokens) // 2, 1, 5))
        reqs.append((int(r.offset) + 2, 0, 7))
    wins = C.corpus_windows(lm, corpus, reqs, before=4, after=2, batch_tokens=64)
    for gpos, lyr, n in reqs:
        w = wins[(gpos, lyr, n)]
        d, t = corpus.locate(np.array([gpos]))
        ids = torch.as_tensor(corpus.doc_tokens(int(d[0])).astype(np.int64))[None]
        with capture(lm.model, [lyr]) as acts:
            lm.model(ids)
        a = acts[lyr][0, :, n]
        lo = max(0, int(t[0]) - 4)
        np.testing.assert_allclose(
            w["acts"], a[lo : lo + len(w["acts"])].numpy(), rtol=1e-4, atol=1e-5
        )
        assert w["ids"][w["center"]] == int(corpus.tokens[gpos])


def test_census_positions_disjoint():
    L, K, N, B, k = 2, 20, 4, 12, 6
    rng = np.random.default_rng(0)
    topk = {
        "top_pos_idx": rng.permutation(1000)[: L * K * N].reshape(L, K, N),
        "top_neg_idx": rng.permutation(1000)[: L * K * N].reshape(L, K, N),
    }
    res = {"res_idx": 1000 + rng.permutation(5000)[: L * B * k * N].reshape(L, B, k, N)}
    cfg = {
        "n_top_describe": 4,
        "n_per_band": 2,
        "n_top_eval": 4,
        "n_band_eval": 2,
        "eval_top_bands": 3,
    }
    sel = C.census_positions(topk, res, 1, 2, 0, cfg, rng)
    d = {p for p, _ in sel["describe"]}
    dv = {p for p, _ in sel["dev"]}
    te = {p for p, _ in sel["test"]}
    assert not (d & dv) and not (d & te) and not (dv & te)
    assert len(sel["describe"]) == 4 + 6 * 2
    assert sel["describe"][0][0] == topk["top_pos_idx"][1, 0, 2]


def test_redaction_of_item_span():
    it = AuditItem(
        "t0", "triggered", 0, list(range(10)), target_pos=4, target_token_id=4, unusual_end=7
    )
    toks = [f"<{i}>" for i in range(2, 9)]
    out = C.redact_item_tokens(toks, 2, it)
    assert out == ["<2>", "<3>", "[…]", "", "", "<7>", "<8>"]
    clean = AuditItem("c0", "clean", 0, list(range(10)), 4, 4)
    assert C.redact_item_tokens(toks, 2, clean) == toks
    assert C.unusual_spans([it, clean], Tok()) == ["<4><5><6>"]


def test_p90_threshold_signs():
    thr = np.zeros((1, 1, 12))
    thr[0, 0, 1] = 0.5
    thr[0, 0, 7] = -0.3
    assert C.p90_threshold(thr, 0, 0, 0, 6) == 0.5
    assert C.p90_threshold(thr, 0, 0, 1, 6) == 0.3
