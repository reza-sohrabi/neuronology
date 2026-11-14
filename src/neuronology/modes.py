"""Modes and subcircuits of single neuron-poles (PLAN.md 11 and the pre-registration in 11.1).

AUDIT-SIDE library: it never reads the secret definitions and never imports `backdoor.py`. Inputs
are a model, census statistics, corpus documents and (in the job script) the M3 audit items.

Pieces
------
- `Pole`, `fire_threshold`: a neuron-pole and its pole-signed firing threshold (census p95 of
  the pole, never below 0).
- `FiringSampler`: uniform sample of up to k firings per pole from a corpus stream (smallest
  i.i.d. uniform keys, as `census.ReservoirAccumulator`), with firing-overlap counts against a
  second model run on the same tokens.
- `capture_at`: one forward pass that keeps, at selected (row, position) pairs only, every piece
  the upstream decomposition needs (embedding, mixer out-projection inputs, neuron activations,
  the residual entering each post-attention norm, gate/up pre-activations).
- `ProfileComputer`: the four-part input profile of ONE firing (gate drivers, gate suppressors,
  up drivers, up suppressors; top `top` units per lane by excess over the mean effect on
  `n_baseline` random neurons of the same layer), exactly as `attribution.input_profile` does
  for a single position, but batched over firings. It uses that the RMSNorm linearisation
  depends on the position only through the scalar 1/rms, so per firing the lane is one
  [d]-vector q and every unit's excess is q . write_u. Also the per-firing readers: first-order
  effect of the neuron's write on every later-layer neuron at the same position.
- `tfidf_matrix`, `knn`, `leiden_labels`, `merge_clusters`, `assign_by_knn`: TF-IDF per lane
  block, L2 per block, 15-NN cosine graph, Leiden (RB configuration model, gamma = 1), groups
  under 15 firings unassigned, merge when top-50 inputs and top-50 readers both overlap >= 0.3.
- `marker_subcircuit`: the pre-registered rule that names a subcircuit the "marker
  subcircuit" from triggered-vs-clean transcript counts only.
- `ablation_kl`, `ablate_units`, `forward_last`: single-position neuron mean-ablation KL, and
  whole-sequence mean-ablation of upstream units (neurons and mixer heads).

Unit indexing follows `attribution.UnitLayout` (0 = embedding, then mixer heads of layers
0..L, then neurons of layers 0..L-1).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from neuronology import attribution as A
from neuronology.hooks import intervene
from neuronology.model import decoder_layers, mlp_module

LANES = ("gate_drivers", "gate_suppressors", "up_drivers", "up_suppressors")
SRC_CORPUS, SRC_HELD, SRC_TRIG, SRC_CLEAN = 0, 1, 2, 3
SRC_NAMES = ("corpus", "corpus_heldout", "triggered", "clean")


# ----------------------------------------------------------------------------- poles


@dataclass(frozen=True)
class Pole:
    layer: int
    neuron: int
    pole: int  # 0 = pos, 1 = neg

    @property
    def sign(self) -> float:
        return 1.0 if self.pole == 0 else -1.0

    @property
    def key(self) -> str:
        return A.pole_key(self.layer, self.neuron, self.pole)

    @classmethod
    def parse(cls, key: str) -> Pole:
        m = re.fullmatch(r"L(\d+)_N(\d+)_(pos|neg|0|1)", key)
        if not m:
            raise ValueError(f"bad pole key {key!r}")
        p = m.group(3)
        return cls(int(m.group(1)), int(m.group(2)), 0 if p in ("pos", "0") else 1)


def fire_threshold(
    thresholds: np.ndarray, pole: Pole, pos_quantiles: list[float], q: float = 0.95
) -> float:
    """Pole-signed firing threshold: pos -> p_q, neg -> -p_(1-q); never below 0.

    `thresholds` [L, N, n_pos + n_neg] from the census (pos quantiles first, then the
    mirrored neg quantiles in the same order).
    """
    j = int(np.argmin(np.abs(np.asarray(pos_quantiles) - q)))
    if abs(pos_quantiles[j] - q) > 1e-9:
        raise ValueError(f"quantile {q} not in census quantiles {pos_quantiles}")
    if pole.pole == 0:
        t = float(thresholds[pole.layer, pole.neuron, j])
    else:
        t = -float(thresholds[pole.layer, pole.neuron, len(pos_quantiles) + j])
    return max(t, 0.0)


# --------------------------------------------------------------------------- sampling


class FiringSampler:
    """Uniform sample of at most `k` firings of one pole over a token stream.

    Each firing (pole-signed activation above `thr`) gets an i.i.d. uniform key; the k
    smallest keys are kept, which is a uniform sample without replacement of all firings
    seen. Also counts, on the same tokens, firings of the pole in a second model (`other`,
    with its own threshold) and joint firings.
    """

    def __init__(self, k: int, thr: float, other_thr: float, seed: int, device):
        self.k, self.thr, self.other_thr = int(k), float(thr), float(other_thr)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(int(seed))
        self.device = device
        self.keys = torch.empty(0, device=device)
        self.gpos = torch.empty(0, dtype=torch.long, device=device)
        self.act = torch.empty(0, device=device)
        self.other = torch.empty(0, device=device)
        self.counts = {"n_tokens": 0, "n_fire": 0, "n_fire_other": 0, "n_both": 0}

    def update(self, s_act: torch.Tensor, gpos: torch.Tensor, s_other: torch.Tensor) -> None:
        s_act, s_other = s_act.float(), s_other.float()
        f = s_act > self.thr
        fo = s_other > self.other_thr
        self.counts["n_tokens"] += int(s_act.numel())
        self.counts["n_fire"] += int(f.sum())
        self.counts["n_fire_other"] += int(fo.sum())
        self.counts["n_both"] += int((f & fo).sum())
        if not bool(f.any()):
            return
        keys = torch.rand(int(f.sum()), generator=self.gen, device=self.device)
        k_all = torch.cat([self.keys, keys])
        g_all = torch.cat([self.gpos, gpos[f]])
        a_all = torch.cat([self.act, s_act[f]])
        o_all = torch.cat([self.other, s_other[f]])
        if k_all.numel() > self.k:
            keep = torch.topk(-k_all, self.k).indices
            k_all, g_all, a_all, o_all = k_all[keep], g_all[keep], a_all[keep], o_all[keep]
        self.keys, self.gpos, self.act, self.other = k_all, g_all, a_all, o_all

    def result(self) -> dict[str, Any]:
        order = torch.argsort(self.gpos)
        return {
            "gpos": self.gpos[order].cpu().numpy().astype(np.int64),
            "act": self.act[order].cpu().numpy().astype(np.float32),
            "other_act": self.other[order].cpu().numpy().astype(np.float32),
            **self.counts,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "keys": self.keys.cpu(),
            "gpos": self.gpos.cpu(),
            "act": self.act.cpu(),
            "other": self.other.cpu(),
            "counts": dict(self.counts),
            "gen": self.gen.get_state(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        d = self.device
        self.keys, self.gpos = sd["keys"].to(d), sd["gpos"].to(d)
        self.act, self.other = sd["act"].to(d), sd["other"].to(d)
        self.counts = dict(sd["counts"])
        self.gen.set_state(sd["gen"])


# ---------------------------------------------------------------------------- capture


@torch.no_grad()
def capture_at(
    model: nn.Module,
    input_ids: torch.Tensor,
    rows: torch.Tensor,
    cols: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Forward `input_ids` [B, S]; keep the decomposition pieces at (rows[i], cols[i]) only.

    Returns {"embed": [P, d], "mixer_in": {l: [P, H*hd]}, "act": {l: [P, N]},
    "h_mid": {l: [P, d]}, "z_gate": {l: [P, N]}, "z_up": {l: [P, N]}} (model dtype).
    """
    A._check_no_bias(model)
    dev = next(model.parameters()).device
    rows, cols = rows.to(dev), cols.to(dev)
    st: dict[str, Any] = {k: {} for k in ("mixer_in", "act", "h_mid", "z_gate", "z_up")}
    handles = []

    def pre(store, i):
        def fn(_m, args):
            store[i] = args[0][rows, cols].detach()

        return fn

    def post(store, i):
        def fn(_m, _a, out):
            store[i] = out[rows, cols].detach()

        return fn

    def emb(_m, _a, out):
        st["embed"] = out[rows, cols].detach()

    try:
        handles.append(model.model.embed_tokens.register_forward_hook(emb))
        for i, lay in enumerate(decoder_layers(model)):
            handles.append(A.mixer_out_proj(lay).register_forward_pre_hook(pre(st["mixer_in"], i)))
            handles.append(
                lay.post_attention_layernorm.register_forward_pre_hook(pre(st["h_mid"], i))
            )
            handles.append(lay.mlp.down_proj.register_forward_pre_hook(pre(st["act"], i)))
            handles.append(lay.mlp.gate_proj.register_forward_hook(post(st["z_gate"], i)))
            handles.append(lay.mlp.up_proj.register_forward_hook(post(st["z_up"], i)))
        model.model(input_ids=input_ids.to(dev), attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return st


def subset_parts(parts: dict[str, Any], idx: torch.Tensor) -> dict[str, Any]:
    out: dict[str, Any] = {"embed": parts["embed"][idx]}
    for k in ("mixer_in", "act", "h_mid", "z_gate", "z_up"):
        out[k] = {lyr: t[idx] for lyr, t in parts[k].items()}
    return out


# ---------------------------------------------------------------------------- profiles


class ProfileComputer:
    """Per-firing four-part input profiles and readers of one neuron-pole."""

    def __init__(
        self,
        model: nn.Module,
        pole: Pole,
        n_baseline: int = 512,
        seed: int = 0,
        top: int = 400,
        reader_top: int = 200,
    ):
        self.model, self.pole = model, pole
        lays = decoder_layers(model)
        L, n = pole.layer, pole.neuron
        lay = lays[L]
        self.layout = A.unit_layout(model, L)
        N = lay.mlp.down_proj.in_features
        rng = np.random.default_rng(seed)
        others = np.setdiff1d(np.arange(N), [n])
        base = rng.choice(others, size=min(n_baseline, len(others)), replace=False)
        self.baseline_ids = base
        dev = lay.mlp.gate_proj.weight.device
        self.ids = torch.as_tensor(np.concatenate([[n], base]), dtype=torch.long, device=dev)
        self.Wg = lay.mlp.gate_proj.weight[self.ids].float()
        self.Wu = lay.mlp.up_proj.weight[self.ids].float()
        norm = lay.post_attention_layernorm
        self.gain, self.eps = A.norm_gain(norm).to(dev), A.norm_eps(norm)
        self.top, self.reader_top = int(top), int(reader_top)
        self.N = N
        # readers: dA_l'(p) = a(p) / rms_l'(p) * (SiLU'(g) u G + SiLU(g) U), position-free G, U
        wd = lay.mlp.down_proj.weight[:, n].float()
        self.reader_layers = list(range(L + 1, len(lays)))
        self.RG, self.RU, self.Reps = {}, {}, {}
        for lp in self.reader_layers:
            nl = lays[lp].post_attention_layernorm
            v = wd * A.norm_gain(nl).to(wd.device)
            self.RG[lp] = A._mm(v[None], lays[lp].mlp.gate_proj.weight.T)[0]
            self.RU[lp] = A._mm(v[None], lays[lp].mlp.up_proj.weight.T)[0]
            self.Reps[lp] = A.norm_eps(nl)

    @torch.no_grad()
    def lane_vectors(self, parts: dict[str, Any]) -> torch.Tensor:
        """[2P, d]: rows 0..P-1 gate lane, P..2P-1 up lane (excess-over-baseline vectors)."""
        L = self.pole.layer
        hm = parts["h_mid"][L].float()
        scale = self.gain * A.inv_rms(hm, self.eps)  # [P, d]
        g = parts["z_gate"][L][:, self.ids].float()
        u = parts["z_up"][L][:, self.ids].float()
        a = torch.nn.functional.silu(g) * u
        sign = torch.where(a < 0, -1.0, 1.0)
        sign[:, 0] = self.pole.sign
        fg = sign * A.dsilu(g) * u
        fu = sign * torch.nn.functional.silu(g)
        nb = max(len(self.ids) - 1, 1)
        cg, cu = fg.clone(), fu.clone()
        cg[:, 1:] = -cg[:, 1:] / nb
        cu[:, 1:] = -cu[:, 1:] / nb
        return torch.cat([scale * (cg @ self.Wg), scale * (cu @ self.Wu)], 0)

    @torch.no_grad()
    def unit_effects(self, parts: dict[str, Any], Q: torch.Tensor) -> torch.Tensor:
        """Q [R, d] (R = k * P rows, firing index r % P) -> unit effects [R, U]."""
        lays = decoder_layers(self.model)
        lo, L = self.layout, self.pole.layer
        P = parts["embed"].shape[0]
        reps = Q.shape[0] // P

        def rep(x: torch.Tensor) -> torch.Tensor:
            return x.float().repeat(reps, 1)

        E = torch.empty(Q.shape[0], lo.n_units, device=Q.device)
        E[:, 0] = (Q * rep(parts["embed"])).sum(-1)
        for lyr in range(L + 1):
            op = A.mixer_out_proj(lays[lyr])
            hd = A.mixer_head_dim(lays[lyr])
            Uo = A._mm(Q, op.weight)  # [R, H*hd]
            per_head = (Uo * rep(parts["mixer_in"][lyr])).view(Q.shape[0], -1, hd).sum(-1)
            o = lo.mixer_offset(lyr)
            E[:, o : o + per_head.shape[1]] = per_head
        for lyr in range(L):
            o = lo.neuron_offset(lyr)
            E[:, o : o + lo.n_neurons] = A._mm(Q, lays[lyr].mlp.down_proj.weight) * rep(
                parts["act"][lyr]
            )
        return E

    @torch.no_grad()
    def raw_residual(self, parts: dict[str, Any]) -> float:
        """Max relative error of sum_u c_u against the true gate/up pre-activations."""
        L, n = self.pole.layer, self.pole.neuron
        scale = self.gain * A.inv_rms(parts["h_mid"][L].float(), self.eps)
        Q = torch.cat([scale * self.Wg[0], scale * self.Wu[0]], 0)
        tot = self.unit_effects(parts, Q).sum(-1)
        P = scale.shape[0]
        z = torch.cat([parts["z_gate"][L][:, n].float(), parts["z_up"][L][:, n].float()])
        return float(((tot - z).abs() / z.abs().clamp_min(1e-6)).max()) if P else 0.0

    @torch.no_grad()
    def compute(self, parts: dict[str, Any]) -> dict[str, np.ndarray]:
        """Profiles of all P firings in `parts` (every row must be a firing of the pole).

        Returns prof_idx [P, 4, top] int32 (-1 = empty), prof_val [P, 4, top] float32
        (excess for drivers, -excess for suppressors, so always > 0), act [P] (pole-signed),
        rd_idx [P, reader_top] int64 (flat layer * N + neuron, -1 empty), rd_val [P, reader_top].
        """
        L, n = self.pole.layer, self.pole.neuron
        P = parts["embed"].shape[0]
        E = self.unit_effects(parts, self.lane_vectors(parts))
        k = min(self.top, E.shape[1])
        idx = np.full((P, 4, self.top), -1, dtype=np.int32)
        val = np.zeros((P, 4, self.top), dtype=np.float32)
        for li, (rows, sgn) in enumerate(
            (
                (slice(0, P), 1.0),
                (slice(0, P), -1.0),
                (slice(P, 2 * P), 1.0),
                (slice(P, 2 * P), -1.0),
            )
        ):
            v, i = torch.topk(sgn * E[rows], k, dim=1)
            v, i = v.cpu().numpy(), i.cpu().numpy()
            ok = v > 0
            idx[:, li, :k] = np.where(ok, i, -1)
            val[:, li, :k] = np.where(ok, v, 0.0)
        a0 = parts["act"][L][:, n].float()
        out = {"prof_idx": idx, "prof_val": val, "act": (self.pole.sign * a0).cpu().numpy()}
        rk = self.reader_top
        rd_idx = np.full((P, rk), -1, dtype=np.int64)
        rd_val = np.zeros((P, rk), dtype=np.float32)
        if self.reader_layers:
            blocks = []
            for lp in self.reader_layers:
                ir = A.inv_rms(parts["h_mid"][lp].float(), self.Reps[lp])  # [P, 1]
                g, u = parts["z_gate"][lp].float(), parts["z_up"][lp].float()
                dA = (
                    a0[:, None]
                    * ir
                    * (A.dsilu(g) * u * self.RG[lp] + torch.nn.functional.silu(g) * self.RU[lp])
                )
                blocks.append(dA)
            M = torch.cat(blocks, 1)  # [P, n_later * N]
            kk = min(rk, M.shape[1])
            _, i = torch.topk(M.abs(), kk, dim=1)
            v = M.gather(1, i)
            i = i.cpu().numpy()
            lyr = np.asarray(self.reader_layers)[i // self.N]
            rd_idx[:, :kk] = lyr * self.N + i % self.N
            rd_val[:, :kk] = v.cpu().numpy()
        out["rd_idx"], out["rd_val"] = rd_idx, rd_val
        return out


# ---------------------------------------------------------------------- TF-IDF and graph


def tfidf_matrix(
    prof_idx: np.ndarray,
    prof_val: np.ndarray,
    n_units: int,
    fit_rows: np.ndarray,
    idf: np.ndarray | None = None,
):
    """CSR [F, 4 * n_units]: per lane block tf (= excess magnitude) x idf, L2 per block,
    blocks scaled by 1/2 so a row with four non-empty blocks has unit norm.

    idf_t = ln((1 + n) / (1 + df_t)) + 1 per block over `fit_rows` (sklearn's smooth idf).
    Returns (X, idf [4, n_units]).
    """
    from scipy import sparse

    F = prof_idx.shape[0]
    if idf is None:
        n = int(fit_rows.sum())
        idf = np.zeros((4, n_units), dtype=np.float64)
        for b in range(4):
            ii = prof_idx[fit_rows, b].reshape(-1)
            df = np.bincount(ii[ii >= 0], minlength=n_units)
            idf[b] = np.log((1.0 + n) / (1.0 + df)) + 1.0
    rows, cols, vals = [], [], []
    for b in range(4):
        ii = prof_idx[:, b]
        ok = ii >= 0
        w = np.where(ok, prof_val[:, b] * idf[b][np.where(ok, ii, 0)], 0.0)
        nrm = np.sqrt((w**2).sum(1, keepdims=True))
        w = np.where(nrm > 0, w / np.maximum(nrm, 1e-30), 0.0) * 0.5
        r = np.repeat(np.arange(F), ii.shape[1]).reshape(F, -1)
        rows.append(r[ok])
        cols.append(b * n_units + ii[ok].astype(np.int64))
        vals.append(w[ok])
    X = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(F, 4 * n_units),
    )
    X.sum_duplicates()
    return X, idf


def reader_matrix(rd_idx: np.ndarray, rd_val: np.ndarray, n_cols: int):
    from scipy import sparse

    F = rd_idx.shape[0]
    ok = rd_idx >= 0
    r = np.repeat(np.arange(F), rd_idx.shape[1]).reshape(F, -1)
    return sparse.csr_matrix((rd_val[ok], (r[ok], rd_idx[ok])), shape=(F, max(n_cols, 1)))


def knn(Xq, Xr, k: int, exclude_self: bool = False, chunk: int = 1024):
    """Cosine k-NN (rows are unit-norm or zero): (idx [Q, k], sim [Q, k])."""
    Q = Xq.shape[0]
    kk = min(k, Xr.shape[0] - (1 if exclude_self else 0))
    idx = np.zeros((Q, max(kk, 0)), dtype=np.int64)
    sim = np.zeros((Q, max(kk, 0)), dtype=np.float32)
    if kk <= 0:
        return idx, sim
    XrT = Xr.T.tocsr()
    for s in range(0, Q, chunk):
        S = (Xq[s : s + chunk] @ XrT).toarray().astype(np.float32)
        if exclude_self:
            r = np.arange(S.shape[0])
            S[r, s + r] = -np.inf
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        ps = np.take_along_axis(S, part, 1)
        o = np.argsort(-ps, axis=1, kind="stable")
        idx[s : s + chunk] = np.take_along_axis(part, o, 1)
        sim[s : s + chunk] = np.take_along_axis(ps, o, 1)
    return idx, sim


def relabel_by_size(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Clusters ordered by size (0 = largest); clusters below `min_size` -> -1."""
    out = np.full(len(labels), -1, dtype=np.int64)
    ids, cnt = np.unique(labels[labels >= 0], return_counts=True)
    order = sorted(zip(ids, cnt, strict=True), key=lambda x: (-x[1], x[0]))
    j = 0
    for c, n in order:
        if n >= min_size:
            out[labels == c] = j
            j += 1
    return out


def leiden_labels(
    nbr: np.ndarray, sim: np.ndarray, resolution: float = 1.0, seed: int = 0, min_size: int = 15
) -> np.ndarray:
    """Leiden (Reichardt-Bornholdt configuration model) on the symmetrised k-NN graph."""
    import igraph as ig
    import leidenalg

    n = nbr.shape[0]
    w: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j, s in zip(nbr[i], sim[i], strict=True):
            j = int(j)
            if j == i or s <= 0:
                continue
            e = (i, j) if i < j else (j, i)
            w[e] = max(w.get(e, 0.0), float(s))
    g = ig.Graph(n=n, edges=list(w.keys()))
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=list(w.values()) if w else None,
        resolution_parameter=float(resolution),
        seed=int(seed),
        n_iterations=-1,
    )
    return relabel_by_size(np.asarray(part.membership, dtype=np.int64), min_size)


def top_set(vec: np.ndarray, top: int) -> set[int]:
    vec = np.asarray(vec).ravel()
    nz = np.nonzero(vec)[0]
    if len(nz) <= top:
        return {int(i) for i in nz}
    sel = nz[np.argpartition(-vec[nz], top - 1)[:top]]
    return {int(i) for i in sel}


def set_overlap(a: set[int], b: set[int], top: int) -> float:
    """|a & b| / top (the post's top-50 overlap); empty sets overlap 0."""
    if not a or not b:
        return 0.0
    return len(a & b) / float(top)


def cluster_sets(X, R, labels: np.ndarray, top: int) -> dict[int, dict[str, Any]]:
    """Per cluster: top-`top` input terms of the centroid and top-`top` readers (|mean|)."""
    out = {}
    for c in sorted(set(labels[labels >= 0].tolist())):
        m = labels == c
        cen = np.asarray(X[m].mean(0)).ravel()
        rd = None
        if R is not None and R.shape[1] > 1:
            rd = top_set(np.abs(np.asarray(R[m].mean(0)).ravel()), top)
        out[int(c)] = {"inputs": top_set(cen, top), "readers": rd, "n": int(m.sum())}
    return out


def merge_clusters(
    X, R, labels: np.ndarray, top: int = 50, thr: float = 0.3, min_size: int = 15
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Greedy merge: while some pair has input overlap >= thr AND reader overlap >= thr,
    merge the pair with the largest min(input, reader) overlap, recompute, repeat.

    Reader overlap counts as 1.0 when the pole has no reader neurons (last layer: its only
    readers are the logits, whose direct effect has the same direction at every firing).
    """
    lab = labels.copy()
    log: list[dict[str, Any]] = []
    while True:
        sets = cluster_sets(X, R, lab, top)
        ids = sorted(sets)
        best = None
        for ii, a in enumerate(ids):
            for b in ids[ii + 1 :]:
                oi = set_overlap(sets[a]["inputs"], sets[b]["inputs"], top)
                if sets[a]["readers"] is None or sets[b]["readers"] is None:
                    orr = 1.0
                else:
                    orr = set_overlap(sets[a]["readers"], sets[b]["readers"], top)
                if oi >= thr and orr >= thr:
                    s = min(oi, orr)
                    if best is None or s > best[0]:
                        best = (s, a, b, oi, orr)
        if best is None:
            break
        _, a, b, oi, orr = best
        log.append({"merged": [int(a), int(b)], "input_overlap": oi, "reader_overlap": orr})
        lab[lab == b] = a
    return relabel_by_size(lab, min_size), log


def assign_by_knn(nbr: np.ndarray, labels_ref: np.ndarray) -> np.ndarray:
    """Majority label of each query's neighbours (ties -> the smaller label; -1 counts)."""
    out = np.full(nbr.shape[0], -1, dtype=np.int64)
    for i in range(nbr.shape[0]):
        if nbr.shape[1] == 0:
            continue
        vals, cnt = np.unique(labels_ref[nbr[i]], return_counts=True)
        best = cnt.max()
        out[i] = int(vals[cnt == best].min())
    return out


def marker_subcircuit(
    labels: np.ndarray,
    src: np.ndarray,
    n_pos_trig: int,
    n_pos_clean: int,
    min_excess: float = 15.0,
    alpha: float = 1e-3,
) -> dict[str, Any]:
    """Pre-registered marker-subcircuit rule (PLAN.md 11.1).

    For each subcircuit c: n_t, n_c = its firings from triggered / clean transcript prefixes.
    Excess E_c = n_t - n_c * n_pos_trig / n_pos_clean (clean count scaled to the triggered
    prefixes' token count). One-sided binomial test of n_t out of n_t + n_c against
    p0 = n_pos_trig / (n_pos_trig + n_pos_clean). The marker subcircuit is the c with the
    largest E_c among those with E_c >= min_excess and p < alpha; otherwise there is none.
    """
    from scipy.stats import binomtest

    rows = []
    p0 = n_pos_trig / max(n_pos_trig + n_pos_clean, 1)
    ratio = n_pos_trig / max(n_pos_clean, 1)
    for c in sorted(set(labels[labels >= 0].tolist())):
        m = labels == c
        nt = int((m & (src == SRC_TRIG)).sum())
        nc = int((m & (src == SRC_CLEAN)).sum())
        e = nt - nc * ratio
        p = float(binomtest(nt, nt + nc, p0, alternative="greater").pvalue) if nt + nc else 1.0
        rows.append({"subcircuit": int(c), "n_trig": nt, "n_clean": nc, "excess": e, "p": p})
    ok = [r for r in rows if r["excess"] >= min_excess and r["p"] < alpha]
    marker = max(ok, key=lambda r: r["excess"])["subcircuit"] if ok else None
    return {"marker": marker, "p0": p0, "rows": rows}


def bootstrap_shares(
    labels: np.ndarray, kl: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[int, dict[str, float]]:
    """Share of summed KL per label (including -1), with percentile bootstrap over firings."""
    ok = np.isfinite(kl)
    lab, v = labels[ok], kl[ok]
    ids = sorted(set(lab.tolist()))
    tot = v.sum()
    rng = np.random.default_rng(seed)
    boots = {c: [] for c in ids}
    for _ in range(n_boot):
        s = rng.integers(0, len(v), len(v))
        t = v[s].sum()
        for c in ids:
            boots[c].append(v[s][lab[s] == c].sum() / t if t > 0 else np.nan)
    out = {}
    for c in ids:
        m = lab == c
        b = np.asarray(boots[c])
        out[int(c)] = {
            "share": float(v[m].sum() / tot) if tot > 0 else float("nan"),
            "lo": float(np.nanquantile(b, 0.025)) if n_boot else float("nan"),
            "hi": float(np.nanquantile(b, 0.975)) if n_boot else float("nan"),
            "n": int(m.sum()),
            "mean_kl": float(v[m].mean()) if m.any() else float("nan"),
            "kl_sum": float(v[m].sum()),
        }
    return out


def specific_drivers(
    prof_idx: np.ndarray,
    prof_val: np.ndarray,
    members: np.ndarray,
    others: np.ndarray,
    n_units: int,
    k: int = 20,
    exclude: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, np.ndarray]:
    """Units most specific to a subcircuit's drivers: mean driver excess (gate + up lanes)
    over `members` minus the same mean over `others` (units outside a firing's top lists
    count 0). Returns (units [k], scores [k])."""

    def mean_drive(mask: np.ndarray) -> np.ndarray:
        acc = np.zeros(n_units)
        n = max(int(mask.sum()), 1)
        for b in (0, 2):
            ii = prof_idx[mask, b].reshape(-1)
            vv = prof_val[mask, b].reshape(-1)
            ok = ii >= 0
            acc += np.bincount(ii[ok], weights=vv[ok], minlength=n_units)
        return acc / n

    s = mean_drive(members) - mean_drive(others)
    s[list(exclude)] = -np.inf
    top = np.argsort(-s, kind="stable")[:k]
    return top, s[top]


def matched_random_units(
    units: np.ndarray, layout: A.UnitLayout, rng: np.random.Generator
) -> np.ndarray:
    """For each unit, a random different unit of the same type and layer."""
    taken = {int(u) for u in units}
    out = []
    for u in units:
        lab = layout.label(int(u))
        for _ in range(1000):
            if lab["type"] == "mixer_head":
                o = layout.mixer_offset(lab["layer"])
                c = o + int(rng.integers(layout.head_counts[lab["layer"]]))
            else:
                o = layout.neuron_offset(lab["layer"])
                c = o + int(rng.integers(layout.n_neurons))
            if c not in taken:
                break
        taken.add(c)
        out.append(c)
    return np.asarray(out, dtype=np.int64)


# ----------------------------------------------------------------------- interventions


@contextmanager
def ablate_units(
    model: nn.Module,
    units: np.ndarray,
    layout: A.UnitLayout,
    neuron_means: np.ndarray,
    head_means: dict[int, torch.Tensor],
) -> Iterator[None]:
    """Mean-ablate upstream units at every position: neurons -> census mean, mixer heads ->
    the corpus mean of their slice of the out-projection input."""
    lays = decoder_layers(model)
    neurons: dict[int, list[int]] = {}
    heads: dict[int, list[int]] = {}
    for u in units:
        lab = layout.label(int(u))
        if lab["type"] == "neuron":
            neurons.setdefault(lab["layer"], []).append(lab["neuron"])
        elif lab["type"] == "mixer_head":
            heads.setdefault(lab["layer"], []).append(lab["head"])
        else:
            raise ValueError("embedding ablation is not supported")
    handles = []

    def head_hook(lyr: int, hs: list[int]):
        hd = A.mixer_head_dim(lays[lyr])
        idx = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in hs])

        def fn(_m, args):
            x = args[0].clone()
            i = idx.to(x.device)
            x[..., i] = head_means[lyr].to(x.device, x.dtype)[i]
            return (x,) + tuple(args[1:])

        return fn

    with ExitStack() as stack:
        if neurons:
            idx = {lyr: torch.as_tensor(ns) for lyr, ns in neurons.items()}
            means = {lyr: torch.as_tensor(neuron_means[lyr], dtype=torch.float32) for lyr in idx}

            def nfn(layer: int, acts: torch.Tensor) -> torch.Tensor:
                ns = idx[layer].to(acts.device)
                out = acts.clone()
                out[..., ns] = means[layer].to(acts.device, acts.dtype)[ns]
                return out

            stack.enter_context(intervene(model, nfn, sorted(idx)))
        try:
            for lyr, hs in heads.items():
                handles.append(
                    A.mixer_out_proj(lays[lyr]).register_forward_pre_hook(head_hook(lyr, hs))
                )
            yield
        finally:
            for h in handles:
                h.remove()


def _pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    lens = [len(s) for s in seqs]
    S = max(lens)
    ids = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), S), dtype=torch.long)
    for j, s in enumerate(seqs):
        ids[j, : lens[j]] = torch.as_tensor(s, dtype=torch.long)
        mask[j, : lens[j]] = 1
    return ids, mask, lens


def _batches_by_tokens(lens: list[int], batch_tokens: int, max_rows: int | None = None):
    order = np.argsort(-np.asarray(lens), kind="stable")
    i = 0
    while i < len(order):
        S = lens[order[i]]
        b = max(1, batch_tokens // max(S, 1))
        if max_rows:
            b = min(b, max_rows)
        yield order[i : i + b]
        i += b


def _margin(logits: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return logits.gather(-1, tgt[:, None]).squeeze(-1) - logits.mean(-1)


@torch.no_grad()
def ablation_kl(
    model: nn.Module,
    seqs: list[list[int]],
    pole: Pole,
    fill: float,
    batch_tokens: int = 8192,
    pad_id: int = 0,
    target_ids: list[int] | None = None,
    max_rows: int = 32,
) -> dict[str, np.ndarray]:
    """KL(clean || ablated) of the next-token distribution at the LAST position of each
    sequence, when the neuron is set to `fill` at that position only (the sequence ends at
    a firing). Clean and ablated copies run in the same batch, so shapes are identical.
    With `target_ids`, also the margin (logit - mean logit) of that token, clean and ablated.
    At most `max_rows` rows per batch, so full-vocabulary logits exist for few rows at a time.
    """
    dev = next(model.parameters()).device
    head = model.get_output_embeddings()
    n = len(seqs)
    kl = np.full(n, np.nan)
    m_clean = np.full(n, np.nan)
    m_abl = np.full(n, np.nan)
    lens_all = [len(s) for s in seqs]
    for rows in _batches_by_tokens(lens_all, max(batch_tokens // 2, 1), max_rows // 2):
        sub = [seqs[i] for i in rows]
        ids, mask, lens = _pad_batch([s for s in sub for _ in (0, 1)], pad_id)
        B = ids.shape[0]
        last = torch.tensor(lens, device=dev) - 1
        sel = torch.zeros(ids.shape, dtype=torch.bool, device=dev)
        odd = torch.arange(1, B, 2, device=dev)
        sel[odd, last[odd]] = True

        def fn(layer, acts, sel=sel):
            out = acts.clone()
            col = out[..., pole.neuron]
            out[..., pole.neuron] = torch.where(sel, torch.full_like(col, fill), col)
            return out

        am = mask.to(dev) if min(lens) < ids.shape[1] else None
        with intervene(model, fn, [pole.layer]):
            hid = model.model(input_ids=ids.to(dev), attention_mask=am, use_cache=False)
        h = hid.last_hidden_state[torch.arange(B, device=dev), last]
        logits = head(h).float()
        lp = torch.log_softmax(logits, -1)
        c, a = lp[0::2], lp[1::2]
        k = (c.exp() * (c - a)).sum(-1).clamp_min(0).cpu().numpy()
        kl[rows] = k
        if target_ids is not None:
            t = torch.tensor([target_ids[i] for i in rows], device=dev)
            m_clean[rows] = _margin(logits[0::2], t).cpu().numpy()
            m_abl[rows] = _margin(logits[1::2], t).cpu().numpy()
    out = {"kl": kl}
    if target_ids is not None:
        out["margin_clean"], out["margin_ablated"] = m_clean, m_abl
    return out


@torch.no_grad()
def forward_last(
    model: nn.Module,
    seqs: list[list[int]],
    pole: Pole,
    batch_tokens: int = 8192,
    pad_id: int = 0,
    target_ids: list[int] | None = None,
    ctx_factory=None,
    ref_logp: list[torch.Tensor] | None = None,
    keep_logp: bool = False,
    max_rows: int = 32,
) -> dict[str, Any]:
    """Forward each sequence under an optional whole-sequence intervention (`ctx_factory()`
    returns a context manager) and read the LAST position: the pole-signed activation of
    `pole`, the target margin (with `target_ids`), KL(ref || this) against `ref_logp` (float32
    log-probs from an earlier call with `keep_logp=True` and the same `batch_tokens`, so the
    batch shapes match), and optionally this call's log-probs.
    """
    dev = next(model.parameters()).device
    head = model.get_output_embeddings()
    n = len(seqs)
    act = np.full(n, np.nan)
    margin = np.full(n, np.nan)
    kl = np.full(n, np.nan)
    logps: list[torch.Tensor | None] = [None] * n
    lens_all = [len(s) for s in seqs]
    for rows in _batches_by_tokens(lens_all, batch_tokens, max_rows):
        ids, mask, lens = _pad_batch([seqs[i] for i in rows], pad_id)
        B = ids.shape[0]
        last = torch.tensor(lens, device=dev) - 1
        got: dict[str, torch.Tensor] = {}

        def hook(_m, args, got=got, last=last, B=B):
            got["a"] = args[0][torch.arange(B, device=dev), last, pole.neuron].float()

        hd = mlp_module(model, pole.layer).down_proj.register_forward_pre_hook(hook)
        am = mask.to(dev) if min(lens) < ids.shape[1] else None
        try:
            with ctx_factory() if ctx_factory is not None else _nullctx():
                hid = model.model(input_ids=ids.to(dev), attention_mask=am, use_cache=False)
        finally:
            hd.remove()
        h = hid.last_hidden_state[torch.arange(B, device=dev), last]
        logits = head(h).float()
        act[rows] = (pole.sign * got["a"]).cpu().numpy()
        if target_ids is not None:
            t = torch.tensor([target_ids[i] for i in rows], device=dev)
            margin[rows] = _margin(logits, t).cpu().numpy()
        lp = torch.log_softmax(logits, -1)
        if ref_logp is not None:
            ref = torch.stack([ref_logp[int(r)] for r in rows]).to(dev)
            kl[rows] = (ref.exp() * (ref - lp)).sum(-1).clamp_min(0).cpu().numpy()
        if keep_logp:
            lpc = lp.cpu()
            for j, r in enumerate(rows):
                logps[int(r)] = lpc[j]
    return {"act": act, "margin": margin, "kl": kl, "logp": logps if keep_logp else None}


@contextmanager
def _nullctx():
    yield
