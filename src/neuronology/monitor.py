"""Monitors for Phase 5 (PLAN.md 10.2): neuron monitors and residual-stream baselines.

Generic tools only: no secret definitions, no planting module (tests/test_blinding.py).
The Phase-2-side driver (scripts/m5_monitor.py) builds the labelled prompt sets.

Neuron monitor. For a pole p = (layer, neuron, sign) with census scale q_p (the pole's
p99.9 magnitude on the generic corpus: p99.9 for the positive pole, -p0.1 for the
negative one), its normalised activation at position t is z_p(t) = sign * a(t) / q_p. A
prompt's per-pole value is m_p = max over prompt positions of z_p(t), and the score of a
set of k poles is the mean of m_p over the set. Census operating point: alarm when the
score is at least 1 (the poles, on average, reach their generic-corpus p99.9).

Baselines read the residual stream after a decoder layer at the last prompt position:
a logistic-regression probe (standardised features, L2) and the projection on the
difference-of-means direction.

Metrics: AUROC (Mann-Whitney, ties counted half), the TPR at a fixed FPR, and a
stratified bootstrap (positives and negatives resampled separately) for both.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from neuronology.attribution import unflat
from neuronology.hooks import stream_acts

Q999_POS, Q999_NEG = 5, 11  # census threshold layout: pos p75..p99.9, then neg p25..p0.1


def pole_scales(thresholds: np.ndarray, flat: np.ndarray, n_neurons: int) -> np.ndarray:
    """Census p99.9 magnitude of each pole (positive numbers), [P]."""
    lyr, n, p = unflat(np.asarray(flat), n_neurons)
    pos = thresholds[lyr, n, Q999_POS]
    neg = -thresholds[lyr, n, Q999_NEG]
    q = np.where(p == 0, pos, neg).astype(np.float64)
    return np.maximum(q, 1e-6)


@torch.no_grad()
def pole_max_over_positions(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    flat: np.ndarray,
    n_neurons: int,
    last_idx: torch.Tensor | None = None,
):
    """Per row and pole: max over valid positions of sign * a, and the argmax position.

    Returns (vals [B, P] float64, argpos [B, P] int64). With `last_idx` ([B] positions)
    also returns sign * a at that position per row ([B, P] float64) as a third element.
    Runs the decoder stack only; other hooks open around the call (e.g. a residual
    capture) see the same pass.
    """
    flat = np.asarray(flat, dtype=np.int64)
    lyr, neu, pole = unflat(flat, n_neurons)
    sign = np.where(pole == 0, 1.0, -1.0)
    B, S = input_ids.shape
    P = len(flat)
    vals = torch.full((B, P), -float("inf"), dtype=torch.float32)
    arg = torch.zeros((B, P), dtype=torch.long)
    at = torch.zeros((B, P), dtype=torch.float32)
    by_layer = {int(lv): np.nonzero(lyr == lv)[0] for lv in np.unique(lyr)}
    valid = attention_mask.bool()

    def fn(layer, a, _zg):
        cols = by_layer.get(layer)
        if cols is None:
            return
        idx = torch.as_tensor(neu[cols], device=a.device)
        sg = torch.as_tensor(sign[cols], device=a.device, dtype=torch.float32)
        x = a.index_select(-1, idx).float() * sg  # [B, S, c]
        x = x.masked_fill(~valid.to(a.device)[..., None], -float("inf"))
        v, i = x.max(1)
        vals[:, cols] = v.cpu()
        arg[:, cols] = i.cpu()
        if last_idx is not None:
            rows = torch.arange(B, device=a.device)
            at[:, cols] = x[rows, last_idx.to(a.device)].cpu()

    with stream_acts(model, fn, layers=sorted(by_layer)):
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if last_idx is not None:
        return vals.double().numpy(), arg.numpy(), at.double().numpy()
    return vals.double().numpy(), arg.numpy()


def neuron_score(norm_vals: np.ndarray, cols: np.ndarray | list[int]) -> np.ndarray:
    """Mean over the given pole columns of the normalised per-pole maxima, [n]."""
    return np.asarray(norm_vals)[:, np.asarray(cols)].mean(1)


# ------------------------------------------------------------------------------ metrics


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(equal)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]), method="average")  # ties get the mean rank
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_at_fpr(neg: np.ndarray, fpr: float) -> float:
    """Smallest threshold t with mean(neg > t) <= fpr (alarm rule: score > t)."""
    neg = np.sort(np.asarray(neg, float))
    n = len(neg)
    allowed = int(np.floor(fpr * n + 1e-9))  # negatives allowed above t
    if allowed >= n:
        return -np.inf
    return float(neg[n - 1 - allowed])


def tpr_at_fpr(pos: np.ndarray, neg: np.ndarray, fpr: float) -> float:
    """TPR at the threshold that keeps the FPR on `neg` at or below `fpr` (score > t)."""
    t = threshold_at_fpr(neg, fpr)
    return float(np.mean(np.asarray(pos, float) > t)) if len(pos) else float("nan")


def bootstrap_detection(
    pos: np.ndarray, neg: np.ndarray, fpr: float, n_boot: int, seed: int, ci: float = 0.95
) -> dict[str, Any]:
    """AUROC and TPR@FPR with a stratified percentile bootstrap."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    out: dict[str, Any] = {"n_pos": int(len(pos)), "n_neg": int(len(neg))}
    if len(pos) == 0 or len(neg) == 0:
        out.update(auroc=None, tpr_at_fpr=None)
        return out
    rng = np.random.default_rng(seed)
    a_est, t_est = auroc(pos, neg), tpr_at_fpr(pos, neg, fpr)
    ab, tb = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        p = pos[rng.integers(0, len(pos), len(pos))]
        q = neg[rng.integers(0, len(neg), len(neg))]
        ab[b], tb[b] = auroc(p, q), tpr_at_fpr(p, q, fpr)
    lo, hi = (1 - ci) / 2, 1 - (1 - ci) / 2
    out["auroc"] = {
        "est": a_est,
        "lo": float(np.quantile(ab, lo)),
        "hi": float(np.quantile(ab, hi)),
    }
    out["tpr_at_fpr"] = {
        "fpr": fpr,
        "est": t_est,
        "lo": float(np.quantile(tb, lo)),
        "hi": float(np.quantile(tb, hi)),
    }
    return out


def alarm_rate(scores: np.ndarray, threshold: float, strict: bool = True) -> dict[str, Any]:
    s = np.asarray(scores, float)
    k = int((s > threshold).sum() if strict else (s >= threshold).sum())
    return {"k": k, "n": int(len(s)), "rate": (k / len(s)) if len(s) else None}


# ------------------------------------------------------------------- residual baselines


def fit_probe(X: np.ndarray, y: np.ndarray, C: float, seed: int = 0):
    """Standardised L2 logistic regression; returns a fitted sklearn pipeline."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000, random_state=seed))
    clf.fit(X, y)
    return clf


def probe_scores(clf, X: np.ndarray) -> np.ndarray:
    return clf.decision_function(X)


def cv_auroc(X: np.ndarray, y: np.ndarray, C: float, folds: int, seed: int) -> float:
    """Mean out-of-fold AUROC of the probe (stratified folds)."""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    vals = []
    for tr, te in skf.split(X, y):
        clf = fit_probe(X[tr], y[tr], C, seed)
        s = probe_scores(clf, X[te])
        vals.append(auroc(s[y[te] == 1], s[y[te] == 0]))
    return float(np.mean(vals))


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit difference-of-means direction, positives minus negatives."""
    d = X[y == 1].mean(0) - X[y == 0].mean(0)
    return d / max(float(np.linalg.norm(d)), 1e-12)
