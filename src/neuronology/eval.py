"""Loss, KL and bootstrap confidence intervals.

Loss is next-token cross-entropy, averaged over tokens. KL is KL(P_clean || P_ablated)
per next-token distribution, averaged over the same tokens. Logits are produced from the
final hidden states in position chunks, so the full [tokens, vocab] tensor never exists.

CIs come from a document-level (cluster) bootstrap: documents are resampled with
replacement and each statistic is the token-weighted mean over the resample. The same
resamples are shared across conditions, so differences and ratios get paired CIs.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F


def shift_targets(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """(positions mask [B, S], targets [B, S]): position t predicts token t+1 in the same row."""
    tgt = torch.zeros_like(input_ids)
    tgt[:, :-1] = input_ids[:, 1:]
    valid = torch.zeros_like(attention_mask, dtype=torch.bool)
    valid[:, :-1] = attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
    return valid, tgt


@torch.no_grad()
def nll_and_kl(
    lm_head: Callable[[torch.Tensor], torch.Tensor],
    h_ref: torch.Tensor,
    h_new: torch.Tensor | None,
    targets: torch.Tensor,
    chunk: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Per-position NLL under ref and new, and KL(ref || new). Inputs [T, d], [T]."""
    nll_ref, nll_new, kl = [], [], []
    for s in range(0, h_ref.shape[0], chunk):
        e = s + chunk
        lp_ref = F.log_softmax(lm_head(h_ref[s:e]).float(), -1)
        t = targets[s:e, None]
        nll_ref.append(-lp_ref.gather(-1, t).squeeze(-1))
        if h_new is not None:
            lp_new = F.log_softmax(lm_head(h_new[s:e]).float(), -1)
            nll_new.append(-lp_new.gather(-1, t).squeeze(-1))
            kl.append((lp_ref.exp() * (lp_ref - lp_new)).sum(-1))
            del lp_new
        del lp_ref
    cat = torch.cat
    if h_new is None:
        return cat(nll_ref), None, None
    return cat(nll_ref), cat(nll_new), cat(kl)


def per_doc_sums(values: torch.Tensor, row_of_pos: torch.Tensor, n_rows: int) -> np.ndarray:
    """Sum a per-position vector into per-row (document) totals."""
    out = torch.zeros(n_rows, dtype=torch.float64)
    out.index_add_(0, row_of_pos.cpu(), values.cpu().double())
    return out.numpy()


def bootstrap_weights(n_docs: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n_docs] multiplicities of a document bootstrap (each row sums to n_docs)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_docs, size=(n_boot, n_docs))
    w = np.zeros((n_boot, n_docs), dtype=np.float64)
    rows = np.repeat(np.arange(n_boot), n_docs)
    np.add.at(w, (rows, draws.ravel()), 1.0)
    return w


def weighted_mean_ci(
    doc_sums: np.ndarray, doc_tokens: np.ndarray, weights: np.ndarray, ci: float
) -> dict[str, float]:
    """Token-weighted mean over docs, with a percentile bootstrap CI."""
    est = doc_sums.sum() / doc_tokens.sum()
    boots = (weights @ doc_sums) / (weights @ doc_tokens)
    a = (1 - ci) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return {"mean": float(est), "lo": float(lo), "hi": float(hi), "se": float(boots.std(ddof=1))}


def ratio_ci(
    num_sums: np.ndarray,
    den_sums: np.ndarray,
    doc_tokens: np.ndarray,
    weights: np.ndarray,
    ci: float,
) -> dict[str, float]:
    """Paired bootstrap CI of mean(num) / mean(den) (both token-weighted, same docs)."""
    est = num_sums.sum() / den_sums.sum()
    boots = (weights @ num_sums) / (weights @ den_sums)
    a = (1 - ci) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return {"ratio": float(est), "lo": float(lo), "hi": float(hi)}
