"""Streaming activation census per neuron-pole (PLAN.md section 6.2).

For every neuron (layer, n), over all valid tokens of a corpus split:

  - a fixed-bin histogram of the signed activation `a` (shared, log-spaced bins on |a|,
    mirrored for a < 0, plus one central bin for |a| < 10**log10_min). Percentile
    thresholds are read from it (`hist_quantiles`), never computed exactly;
  - sum, sum of squares, count (-> mean for mean ablation, and std);
  - gate/up sign-quadrant counts: Q1 g>0,u>0 (a>0); Q2 g>0,u<0 (a<0);
    Q3 g<0,u<0 (a>0, the "double-negative" quadrant); Q4 g<0,u>0 (a<0).
    Since a = SiLU(g)*u and SiLU(g) < 0 iff g < 0, sign(u) follows from sign(g), sign(a);
  - top-k contexts per pole (largest a for "pos", smallest a for "neg"), as global token
    positions in the corpus token array (see `data.Corpus.locate`).

`ReservoirAccumulator` is a second, shorter pass once thresholds are known: a uniform
sample of `k` contexts per neuron-pole-band, by keeping the k smallest i.i.d. uniform
keys per band (equivalent to reservoir sampling, and vectorisable on the GPU).

Bands: pole "pos" uses thresholds at the upper quantiles `pos_quantiles` (p75 ... p99.9),
pole "neg" the mirrored lower quantiles `neg_quantiles` (p25 ... p0.1). Band j of a pole
is [t_j, t_{j+1}) (towards the tail), the last band is open-ended.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

QUADRANTS = ("Q1_g+u+", "Q2_g+u-", "Q3_g-u-", "Q4_g-u+")


@dataclass(frozen=True)
class HistSpec:
    bins_per_side: int
    log10_min: float
    log10_max: float

    @property
    def n_bins(self) -> int:
        return 2 * self.bins_per_side + 1

    @property
    def width(self) -> float:
        return (self.log10_max - self.log10_min) / self.bins_per_side

    def edges(self) -> np.ndarray:
        """Monotone signed bin edges, length n_bins + 1; bin i is [e_i, e_{i+1})."""
        mags = 10.0 ** np.linspace(self.log10_min, self.log10_max, self.bins_per_side + 1)
        return np.concatenate([-mags[::-1], mags])

    def bin_index(self, x: torch.Tensor) -> torch.Tensor:
        K = self.bins_per_side
        m = x.abs().float()
        lg = torch.log10(m.clamp_min(10.0**self.log10_min))
        j = ((lg - self.log10_min) / self.width).floor().long().clamp_(0, K - 1)
        small = m < 10.0**self.log10_min
        idx = torch.where(x > 0, K + 1 + j, K - 1 - j)
        return torch.where(small, torch.full_like(idx, K), idx)


def hist_quantiles(hist: torch.Tensor, qs: list[float], spec: HistSpec) -> torch.Tensor:
    """Quantiles [..., len(qs)] from histograms [..., n_bins] (geometric interpolation
    inside log bins, linear inside the central bin)."""
    h = hist.double()
    cdf = h.cumsum(-1)
    total = cdf[..., -1:]
    edges = torch.as_tensor(spec.edges(), dtype=torch.float64, device=h.device)
    K = spec.bins_per_side
    out = []
    for q in qs:
        target = q * total
        b = (cdf < target).sum(-1, keepdim=True).clamp_(max=spec.n_bins - 1)
        prev = torch.where(b > 0, cdf.gather(-1, (b - 1).clamp_(min=0)), torch.zeros_like(target))
        cnt = h.gather(-1, b).clamp_min(1e-12)
        frac = ((target - prev) / cnt).clamp_(0, 1)
        lo, hi = edges[b], edges[b + 1]
        lin = lo + frac * (hi - lo)
        sign = torch.where(b > K, 1.0, -1.0).to(h)
        geo = sign * torch.exp(
            torch.log(lo.abs()) + frac * (torch.log(hi.abs()) - torch.log(lo.abs()))
        )
        out.append(torch.where(b == K, lin, geo).squeeze(-1))
    return torch.stack(out, -1)


def quadrant_index(a: torch.Tensor, z_gate: torch.Tensor) -> torch.Tensor:
    """0..3 for Q1..Q4 (see module docstring). Exact zeros count as positive."""
    g_neg = z_gate < 0
    a_neg = a < 0
    # g>0: a>=0 -> Q1, a<0 -> Q2;  g<0: a>0 (u<0) -> Q3, a<=0 (u>=0) -> Q4
    return torch.where(g_neg, torch.where(a > 0, 2, 3), torch.where(a_neg, 1, 0)).to(torch.long)


def _merge_topk(
    vals: torch.Tensor, pos: torch.Tensor, new_v: torch.Tensor, new_p: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the k largest along dim 0 of cat([vals, new_v]); positions follow."""
    v = torch.cat([vals, new_v], 0)
    p = torch.cat([pos, new_p], 0)
    top_v, top_i = torch.topk(v, k, dim=0)
    return top_v, p.gather(0, top_i)


def _batch_topk(x: torch.Tensor, pos: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column top-k of x [T, N] (k capped at T); returns values and positions [k', N]."""
    kk = min(k, x.shape[0])
    v, i = torch.topk(x, kk, dim=0)
    return v, pos[i]


class CensusAccumulator:
    """Pass-1 statistics for `n_layers` x `n_neurons`, held on `device`."""

    def __init__(self, n_layers: int, n_neurons: int, spec: HistSpec, top_k: int, device):
        L, N, B = n_layers, n_neurons, spec.n_bins
        self.spec, self.top_k, self.device = spec, top_k, device
        self.L, self.N = L, N
        self.hist = torch.zeros(L, N, B, dtype=torch.int32, device=device)
        # float64 is unsupported on MPS; per-batch partial sums are exact enough in fp32 and
        # the running totals are kept on the CPU in float64.
        self.sum = np.zeros((L, N), dtype=np.float64)
        self.sumsq = np.zeros((L, N), dtype=np.float64)
        self.quad = torch.zeros(L, 4, N, dtype=torch.int64, device=device)
        self.quad_mass = np.zeros((L, 4), dtype=np.float64)
        self.overflow = torch.zeros(L, N, dtype=torch.int64, device=device)
        self.top_v = {
            p: torch.full((L, top_k, N), -float("inf"), device=device) for p in ("pos", "neg")
        }
        self.top_p = {
            p: torch.full((L, top_k, N), -1, dtype=torch.int64, device=device)
            for p in ("pos", "neg")
        }
        self.n_tokens = np.zeros(L, dtype=np.int64)

    @torch.no_grad()
    def update(self, layer: int, a: torch.Tensor, z_gate: torch.Tensor | None, pos: torch.Tensor):
        """a, z_gate: [T, N] valid tokens only; pos: [T] global token positions (int64)."""
        T, N = a.shape
        B = self.spec.n_bins
        idx = self.spec.bin_index(a)
        idx = idx + torch.arange(N, device=a.device) * B
        counts = torch.bincount(idx.flatten(), minlength=N * B).view(N, B)
        self.hist[layer] += counts.to(torch.int32)
        af = a.float()
        self.sum[layer] += af.sum(0).cpu().double().numpy()
        self.sumsq[layer] += af.square().sum(0).cpu().double().numpy()
        self.overflow[layer] += (af.abs() >= 10.0**self.spec.log10_max).sum(0)
        if z_gate is not None:
            q = quadrant_index(af, z_gate)
            absa = af.abs()
            for qi in range(4):
                m = q == qi
                self.quad[layer, qi] += m.sum(0)
                self.quad_mass[layer, qi] += float((absa * m).sum())
        for pole, x in (("pos", af), ("neg", -af)):
            v, p = _batch_topk(x, pos, self.top_k)
            self.top_v[pole][layer], self.top_p[pole][layer] = _merge_topk(
                self.top_v[pole][layer], self.top_p[pole][layer], v, p, self.top_k
            )
        self.n_tokens[layer] += T

    def state_dict(self) -> dict[str, Any]:
        return {
            "hist": self.hist.cpu(),
            "sum": self.sum.copy(),
            "sumsq": self.sumsq.copy(),
            "quad": self.quad.cpu(),
            "quad_mass": self.quad_mass.copy(),
            "overflow": self.overflow.cpu(),
            "top_v": {k: v.cpu() for k, v in self.top_v.items()},
            "top_p": {k: v.cpu() for k, v in self.top_p.items()},
            "n_tokens": self.n_tokens.copy(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        d = self.device
        self.hist = sd["hist"].to(d)
        self.sum, self.sumsq = sd["sum"], sd["sumsq"]
        self.quad, self.quad_mass = sd["quad"].to(d), sd["quad_mass"]
        self.overflow = sd["overflow"].to(d)
        self.top_v = {k: v.to(d) for k, v in sd["top_v"].items()}
        self.top_p = {k: v.to(d) for k, v in sd["top_p"].items()}
        self.n_tokens = sd["n_tokens"]

    def finalize(self, pos_q: list[float], neg_q: list[float]) -> dict[str, np.ndarray]:
        """Numpy results. thresholds: [L, N, len(pos_q)+len(neg_q)] (pos first)."""
        thr = torch.stack(
            [
                hist_quantiles(self.hist[lyr].cpu(), list(pos_q) + list(neg_q), self.spec)
                for lyr in range(self.L)
            ]
        )
        n = np.maximum(self.n_tokens[:, None].astype(np.float64), 1.0)
        mean = self.sum / n
        var = np.maximum(self.sumsq / n - mean**2, 0.0)
        top_v_neg = -self.top_v["neg"]
        return {
            "thresholds": thr.float().cpu().numpy(),
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "n_tokens": self.n_tokens,
            "quad_counts": self.quad.cpu().numpy(),
            "quad_mass": self.quad_mass,
            "overflow": self.overflow.cpu().numpy(),
            "top_pos_val": self.top_v["pos"].cpu().numpy(),
            "top_pos_idx": self.top_p["pos"].cpu().numpy(),
            "top_neg_val": top_v_neg.cpu().numpy(),
            "top_neg_idx": self.top_p["neg"].cpu().numpy(),
        }


class ReservoirAccumulator:
    """Pass-2: k uniform contexts per (layer, pole, band, neuron).

    `thresholds`: [L, N, n_pos + n_neg] from `CensusAccumulator.finalize` (pos first).
    """

    def __init__(self, thresholds: np.ndarray, n_pos: int, k: int, device, seed: int):
        L, N, n_thr = thresholds.shape
        self.L, self.N, self.k, self.device = L, N, k, device
        self.n_pos, self.n_neg = n_pos, n_thr - n_pos
        self.n_bands = self.n_pos + self.n_neg  # one band per threshold, the last open-ended
        self.thr = torch.as_tensor(thresholds, dtype=torch.float32, device=device)
        self.keys = torch.full((L, self.n_bands, k, N), 2.0, device=device)
        self.pos = torch.full((L, self.n_bands, k, N), -1, dtype=torch.int64, device=device)
        self.val = torch.zeros((L, self.n_bands, k, N), dtype=torch.float32, device=device)
        self.count = torch.zeros((L, self.n_bands, N), dtype=torch.int64, device=device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def band_index(self, layer: int, a: torch.Tensor) -> torch.Tensor:
        """[T, N] band id in 0..n_bands-1, or -1 (outside every band). Pos bands first."""
        t = self.thr[layer]  # [N, n_thr]
        tp, tn = t[:, : self.n_pos], t[:, self.n_pos :]
        bp = (a.unsqueeze(-1) >= tp).sum(-1) - 1  # -1 below p75
        bn = (a.unsqueeze(-1) <= tn).sum(-1) - 1
        band = torch.full_like(bp, -1)
        band = torch.where(bp >= 0, bp, band)
        band = torch.where(bn >= 0, self.n_pos + bn, band)
        return band

    @torch.no_grad()
    def update(self, layer: int, a: torch.Tensor, pos: torch.Tensor) -> None:
        af = a.float()
        band = self.band_index(layer, af)
        u = torch.rand(af.shape, generator=self.gen, device=self.device)
        for b in range(self.n_bands):
            in_b = band == b
            self.count[layer, b] += in_b.sum(0)
            key = torch.where(in_b, u, torch.full_like(u, 2.0))
            kk = min(self.k, key.shape[0])
            nk, ni = torch.topk(key, kk, dim=0, largest=False)
            np_ = pos[ni]
            nv = af.gather(0, ni)
            ck = torch.cat([self.keys[layer, b], nk], 0)
            cp = torch.cat([self.pos[layer, b], np_], 0)
            cv = torch.cat([self.val[layer, b], nv], 0)
            sk, si = torch.topk(ck, self.k, dim=0, largest=False)
            self.keys[layer, b] = sk
            self.pos[layer, b] = cp.gather(0, si)
            self.val[layer, b] = cv.gather(0, si)

    def state_dict(self) -> dict[str, Any]:
        return {
            "keys": self.keys.cpu(),
            "pos": self.pos.cpu(),
            "val": self.val.cpu(),
            "count": self.count.cpu(),
            "gen": self.gen.get_state(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        d = self.device
        self.keys, self.pos = sd["keys"].to(d), sd["pos"].to(d)
        self.val, self.count = sd["val"].to(d), sd["count"].to(d)
        self.gen.set_state(sd["gen"])

    def finalize(self) -> dict[str, np.ndarray]:
        valid = self.keys < 2.0
        pos = torch.where(valid, self.pos, torch.full_like(self.pos, -1))
        return {
            "res_idx": pos.cpu().numpy(),  # [L, n_bands, k, N], -1 = empty slot
            "res_val": self.val.cpu().numpy().astype(np.float16),
            "res_count": self.count.cpu().numpy(),  # tokens seen per band in pass 2
        }


def topk_overlap(idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    """Per-neuron fraction of shared positions between two top-k index sets [k, N]."""
    k, N = idx_a.shape
    out = np.empty(N, dtype=np.float32)
    for n in range(N):
        out[n] = len(np.intersect1d(idx_a[:, n], idx_b[:, n], assume_unique=True)) / k
    return out


# ----------------------------------------------------------------------------- runner


def batch_positions(batch, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(valid mask [B, S] bool, global token positions of the valid tokens [T] int64)."""
    mask = batch.attention_mask.bool()
    S = mask.shape[1]
    offs = torch.as_tensor(np.array(batch.offsets), dtype=torch.int64, device=device)
    pos = offs[:, None] + torch.arange(S, device=device)[None, :]
    return mask, pos[mask]


@torch.no_grad()
def forward_stream(lm, batch, layer_fn, with_gate: bool) -> None:
    """Run the decoder stack only (no lm_head) with a streaming activation callback."""
    from neuronology.hooks import stream_acts

    with stream_acts(lm.model, layer_fn, with_gate=with_gate):
        lm.model.model(
            input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False
        )


def run_census(lm, corpus, cfg: dict[str, Any], ckpt_path=None, log=print) -> dict[str, Any]:
    """Pass 1 (histograms, means, quadrants, top-k) then pass 2 (band reservoirs).

    Resumable: state is checkpointed to `ckpt_path` every `checkpoint_minutes`.
    Returns {"stats": ..., "reservoir": ..., "perf": ...} (numpy arrays).
    """
    import time
    from pathlib import Path

    from neuronology.data import collate, make_batches
    from neuronology.runinfo import Progress, peak_memory_gb, sync

    dev = lm.device
    spec = HistSpec(**cfg["hist"])
    pos_q, neg_q = cfg["pos_quantiles"], cfg["neg_quantiles"]
    docs = corpus.split_docs(cfg["split"], cfg.get("max_tokens"))
    batches = make_batches(docs, cfg["batch_tokens"], cfg["sort_window"])
    n_tok = [int(docs["n_tokens"].to_numpy()[b].sum()) for b in batches]
    res_cfg = cfg["reservoir"]
    cum = np.cumsum(n_tok)
    n_res_batches = int(np.searchsorted(cum, res_cfg["max_tokens"]) + 1)
    n_res_batches = min(n_res_batches, len(batches))
    pad_id = lm.tokenizer.pad_token_id
    ckpt_every = cfg["checkpoint_minutes"] * 60

    acc = CensusAccumulator(lm.arch.n_layers, lm.arch.d_mlp, spec, cfg["top_k"], dev)
    res = None
    state = {"phase": 1, "next_batch": 0, "perf": {"pass1_s": 0.0, "pass2_s": 0.0}}
    stats = None
    if ckpt_path is not None and Path(ckpt_path).exists():
        sd = torch.load(ckpt_path, weights_only=False)
        state = sd["state"]
        acc.load_state_dict(sd["acc"])
        if sd.get("stats") is not None:
            stats = sd["stats"]
            res = ReservoirAccumulator(
                stats["thresholds"], len(pos_q), res_cfg["k"], dev, res_cfg["seed"]
            )
            if sd.get("res") is not None:
                res.load_state_dict(sd["res"])
        log(f"[census] resumed at phase {state['phase']} batch {state['next_batch']}")

    def save():
        if ckpt_path is None:
            return
        tmp = Path(str(ckpt_path) + ".tmp")
        torch.save(
            {
                "state": state,
                "acc": acc.state_dict(),
                "stats": stats,
                "res": res.state_dict() if res is not None else None,
            },
            tmp,
        )
        tmp.replace(ckpt_path)

    def run_pass(phase: int, n_batches: int, layer_update, with_gate: bool):
        prog = Progress(f"census_pass{phase}", int(sum(n_tok[:n_batches])))
        done = int(sum(n_tok[: state["next_batch"]]))
        t_last_ckpt = time.time()
        t0 = time.time()
        for bi in range(state["next_batch"], n_batches):
            batch = collate(corpus, docs, batches[bi], pad_id, dev)
            mask, pos = batch_positions(batch, dev)

            def fn(layer, a, zg, mask=mask, pos=pos):
                layer_update(layer, a[mask], None if zg is None else zg[mask], pos)

            forward_stream(lm, batch, fn, with_gate)
            done += batch.n_tokens
            state["next_batch"] = bi + 1
            rec = prog.update(done, batch=bi + 1, n_batches=n_batches)
            if bi % 50 == 0:
                log(f"[census] pass{phase} batch {bi + 1}/{n_batches} {rec}")
            if time.time() - t_last_ckpt > ckpt_every:
                sync(dev)
                state["perf"][f"pass{phase}_s"] += time.time() - t0
                t0 = time.time()
                save()
                t_last_ckpt = time.time()
        sync(dev)
        state["perf"][f"pass{phase}_s"] += time.time() - t0
        prog.update(done, force=True, batch=n_batches, n_batches=n_batches)

    if state["phase"] == 1:
        run_pass(1, len(batches), lambda lyr, a, zg, pos: acc.update(lyr, a, zg, pos), True)
        stats = acc.finalize(pos_q, neg_q)
        res = ReservoirAccumulator(
            stats["thresholds"], len(pos_q), res_cfg["k"], dev, res_cfg["seed"]
        )
        state.update(phase=2, next_batch=0)
        save()
    if state["phase"] == 2:
        run_pass(2, n_res_batches, lambda lyr, a, zg, pos: res.update(lyr, a, pos), False)
        state.update(phase=3, next_batch=0)
        save()

    perf = dict(state["perf"])
    perf.update(
        pass1_tokens=int(sum(n_tok)),
        pass2_tokens=int(sum(n_tok[:n_res_batches])),
        n_batches=len(batches),
        n_docs=len(docs),
        padded_tokens=int(sum(len(b) * docs["n_tokens"].to_numpy()[b].max() for b in batches)),
        peak_memory_gb=peak_memory_gb(dev),
    )
    perf["pass1_tok_per_s"] = perf["pass1_tokens"] / max(perf["pass1_s"], 1e-9)
    perf["pass2_tok_per_s"] = perf["pass2_tokens"] / max(perf["pass2_s"], 1e-9)
    return {"stats": stats, "reservoir": res.finalize(), "perf": perf, "spec": spec}
