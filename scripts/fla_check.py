"""Is flash-linear-attention (fla) a faithful replacement for the reference DeltaNet kernels?

uv run python -m scripts.fla_check --config configs/fla_check.yaml

The M1 smoke test saw mean KL(ref || fla) = 4.4e-4 and argmax agreement 99.3%, but a
max |delta logit| of 13.9. This script locates that kind of outlier (document, position,
token, vocab entry, its rank) and judges it against the bf16 noise floor:

  ref_bf16 : bf16 model, transformers' plain-PyTorch chunk_gated_delta_rule
  fla_bf16 : bf16 model, fla kernel (what the census and sparsity jobs use)
  ref_fp32 : fp32 model, reference kernel (ground truth)

Decision (fixed in the config before running): fla is accepted when its divergence from
fp32 is no worse than the reference kernel's own bf16 divergence, within the stated
factors. Otherwise the 4B jobs must run with the reference kernels.
"""

from __future__ import annotations

import contextlib
import importlib.util

import numpy as np
import torch

from neuronology.data import Corpus, ensure_corpus
from neuronology.model import load_model
from neuronology.runinfo import provenance, results_dir, torch_dtype, write_json
from scripts._common import log, parse
from scripts.smoke import SAMPLE_TEXT


@contextlib.contextmanager
def reference_kernels():
    """Swap the (possibly fla-backed) DeltaNet kernel for transformers' torch version."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    saved = mq.torch_chunk_gated_delta_rule
    mq.torch_chunk_gated_delta_rule = getattr(saved, "__wrapped__", saved)
    try:
        yield
    finally:
        mq.torch_chunk_gated_delta_rule = saved


def texts_from(cfg, tok) -> list[list[int]]:
    seqs = [tok(SAMPLE_TEXT, add_special_tokens=False)["input_ids"]]
    corpus = Corpus(ensure_corpus(cfg["corpus"]["config"], cfg["corpus"].get("sha256")))
    docs = corpus.split_docs("eval")
    rng = np.random.default_rng(cfg["seed"])
    for i in rng.choice(len(docs), cfg["n_docs"], replace=False):
        ids = corpus.doc_tokens(int(docs.loc[i, "doc_id"]))[: cfg["max_len"]]
        seqs.append([int(t) for t in ids])
    return seqs


@torch.no_grad()
def log_probs(lm, seqs, ref: bool) -> list[torch.Tensor]:
    """Per-sequence fp32 log-probs [L, V] on CPU (sequences run one at a time)."""
    out = []
    ctx = reference_kernels() if ref else contextlib.nullcontext()
    head = lm.model.lm_head
    with ctx:
        for s in seqs:
            ids = torch.tensor([s], device=lm.device)
            h = lm.model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
            lp = torch.cat(
                [
                    torch.log_softmax(head(h[i : i + 128]).float(), -1).cpu()
                    for i in range(0, h.shape[0], 128)
                ]
            )
            out.append(lp)
    return out


def compare(a: list[torch.Tensor], b: list[torch.Tensor], seqs, tok) -> dict:
    """a = reference distribution. Logit differences are on log-probs (shift-invariant)."""
    kls, agree, worst = [], [], None
    margins_at_disagreement = []
    for d, (x, y) in enumerate(zip(a, b, strict=True)):
        kl = (x.exp() * (x - y)).sum(-1)
        kls.append(kl)
        am_x, am_y = x.argmax(-1), y.argmax(-1)
        agree.append(am_x == am_y)
        top2 = x.topk(2, -1).values
        margin = top2[:, 0] - top2[:, 1]
        margins_at_disagreement += margin[am_x != am_y].tolist()
        diff = (x - y).abs()
        m, flat = diff.view(-1).max(0)
        pos, v = divmod(int(flat), x.shape[1])
        if worst is None or m > worst["max_abs_logprob_diff"]:
            rank = int((x[pos] > x[pos, v]).sum())
            worst = {
                "doc": d,
                "position": pos,
                "seq_len": x.shape[0],
                "context_token": tok.decode([seqs[d][pos]]),
                "vocab_id": v,
                "vocab_token": tok.decode([v]),
                "max_abs_logprob_diff": float(m),
                "logprob_ref": float(x[pos, v]),
                "logprob_other": float(y[pos, v]),
                "rank_in_ref": rank,
                "kl_at_position": float(kl[pos]),
                "argmax_agree_at_position": bool(am_x[pos] == am_y[pos]),
                "ref_top1_prob": float(x[pos].exp().max()),
            }
    kl = torch.cat(kls)
    ag = torch.cat(agree).float()
    # mass-weighted logit difference: where the probability actually is
    return {
        "mean_kl": float(kl.mean()),
        "p99_kl": float(kl.quantile(0.99)),
        "max_kl": float(kl.max()),
        "argmax_agreement": float(ag.mean()),
        "positions": int(ag.numel()),
        "ref_top2_margin_at_disagreements": margins_at_disagreement,
        "worst_entry": worst,
    }


def main() -> None:
    args, cfg = parse(__doc__)
    fla = importlib.util.find_spec("fla") is not None
    lm = load_model(cfg["model"]["id"], dtype=torch_dtype(cfg["low_dtype"]))
    seqs = texts_from(cfg, lm.tokenizer)
    log(f"fla installed {fla}; {len(seqs)} sequences, {sum(map(len, seqs))} tokens")
    fla_lo = log_probs(lm, seqs, ref=False)
    ref_lo = log_probs(lm, seqs, ref=True)
    tok = lm.tokenizer
    del lm
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    lm32 = load_model(cfg["model"]["id"], dtype=torch.float32)
    ref32 = log_probs(lm32, seqs, ref=True)
    del lm32

    res = {
        "meta": provenance(cfg),
        "fla_installed": fla,
        "fla_vs_ref_low": compare(ref_lo, fla_lo, seqs, tok),
        "ref_low_vs_fp32": compare(ref32, ref_lo, seqs, tok),
        "fla_low_vs_fp32": compare(ref32, fla_lo, seqs, tok),
    }
    a = cfg["accept"]
    f, r = res["fla_low_vs_fp32"], res["ref_low_vs_fp32"]
    checks = {
        "kl_vs_fp32_within_factor": f["mean_kl"] <= a["kl_factor"] * r["mean_kl"],
        "argmax_vs_fp32_within_pp": f["argmax_agreement"] >= r["argmax_agreement"] - a["argmax_pp"],
    }
    res["checks"] = checks
    res["decision"] = "use fla" if all(checks.values()) else "use reference kernels"
    write_json(results_dir(cfg["results_subdir"]) / "fla_check.json", res)
    for k in ("fla_vs_ref_low", "ref_low_vs_fp32", "fla_low_vs_fp32"):
        v = dict(res[k])
        v.pop("ref_top2_margin_at_disagreements")
        log(f"{k}: {v}")
    log(f"checks {checks} -> {res['decision']}")


if __name__ == "__main__":
    main()
