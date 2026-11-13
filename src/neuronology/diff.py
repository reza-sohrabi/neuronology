"""Base-vs-suspect neuron statistics for the blind audit (PLAN.md 9.1).

Everything here works from weights and census arrays only; no model forward pass, no
triggered examples, no secrets. Units:

- neuron level `[L, N]`: weight signals (both poles of a neuron share them) and the
  histogram Jensen-Shannon distance;
- pole level `[L, N, 2]` (pole 0 = positive side, 1 = negative side, the layout used by
  `attribution.ranking`): tail shifts, exceedance, top-k overlap, and every ranking score.

Weight signals use the neuron's *effective* input rows: the MLP reads RMSNorm(h), so a
change of the post-attention norm gain moves every neuron's input as much as a change of
its own row. Effective row = w * gain, gain = (1 + gamma) for Qwen3.5 (stored as offset)
and gamma for Qwen3.

Rankings come in two families because of the M1/M3 findings on heavy-tailed neurons:
"raw" (absolute changes, in weight or activation units) and "norm" (scale-free: cosine
distances, std-normalised shifts, JS distance, top-k turnover). Each score is a
percentile rank in [0, 1] over all units, so families and signals can be averaged.

"Dormant but armed" (PLAN.md 9.1): r_W * (1 - r_A), where r_W is the pole's weight-change
rank and r_A its activation-change rank, both re-ranked to [0, 1] over all poles. It is
high only when the weights moved a lot *and* the generic-corpus activations did not.

The control enters blind (the auditor does not know which suspect is clean): every
primitive signal is turned into a robust z-score of its log magnitude within each
suspect, and `contrast = z_S - median over the other suspects T of z_T`; the composite
rankings are then rebuilt from these contrasts (`*_contrast`). Changes that every
fine-tune shares (same data, same recipe) cancel; changes particular to one suspect
remain, with their magnitude kept (rank differences would turn shared noise into
spurious contrast). With a single other suspect the median is that suspect.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuronology.census import HistSpec

# Census threshold layout (configs/census_*.yaml): pos p75 p90 p95 p97 p99 p99.9, then
# neg p25 p10 p5 p3 p1 p0.1.
Q99, Q999 = 4, 5


# ----------------------------------------------------------------------------- weights


class WeightReader:
    """Lazy reader of a safetensors checkpoint (one file or a sharded index).

    Keys are canonicalised to the text-model form `layers.{l}.<rest>` / `embed_tokens.weight`
    / `norm.weight`, so a Hub VLM checkpoint (`model.language_model.layers...`) and a saved
    causal LM (`model.layers...`) read the same. Vision and MTP weights are ignored.
    """

    def __init__(self, path: str | Path):
        from safetensors import safe_open

        self.path = Path(path)
        idx = self.path / "model.safetensors.index.json"
        if idx.exists():
            files = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.path.glob("*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors under {self.path}")
        self._open = {f: safe_open(str(self.path / f), framework="pt") for f in files}
        self.keys: dict[str, tuple[str, str]] = {}
        for f, h in self._open.items():
            for k in h.keys():
                c = canonical_key(k)
                if c is not None:
                    self.keys.setdefault(c, (f, k))
        cfg_path = self.path / "config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    @property
    def model_type(self) -> str:
        tc = self.config.get("text_config") or self.config
        return tc.get("model_type", self.config.get("model_type", ""))

    @property
    def gain_offset(self) -> float:
        """1.0 when RMSNorm weights are stored as offsets from 1 (Qwen3.5), else 0.0."""
        return 1.0 if self.model_type.startswith("qwen3_5") else 0.0

    def has(self, key: str) -> bool:
        return key in self.keys

    def get(self, key: str) -> torch.Tensor:
        f, k = self.keys[key]
        return self._open[f].get_tensor(k).float()

    def layer_keys(self, layer: int) -> list[str]:
        pre = f"layers.{layer}."
        return sorted(k for k in self.keys if k.startswith(pre))


def canonical_key(k: str) -> str | None:
    if k.startswith(("mtp.", "visual.", "model.visual.")) or ".visual." in k:
        return None
    for pre in ("model.language_model.", "language_model.model.", "model.", ""):
        if k.startswith(pre):
            rest = k[len(pre) :]
            if rest.startswith(("layers.", "embed_tokens.", "norm.")):
                return rest
    return None


def resolve_weights_dir(model_cfg: dict[str, Any]) -> Path:
    """`local_dir` under $NEURONOLOGY_ARTIFACTS, else the pinned Hub snapshot (cached)."""
    import os

    from neuronology.model import REVISIONS
    from neuronology.runinfo import REPO_ROOT

    if model_cfg.get("local_dir"):
        root = Path(os.environ.get("NEURONOLOGY_ARTIFACTS", REPO_ROOT / "artifacts"))
        p = root / model_cfg["local_dir"]
        if not (p / "config.json").exists():
            raise FileNotFoundError(f"no saved model at {p}")
        return p
    from huggingface_hub import snapshot_download

    kw = dict(
        repo_id=model_cfg["id"],
        revision=REVISIONS.get(model_cfg["id"]),
        allow_patterns=["*.safetensors", "*.json"],
    )
    try:
        return Path(snapshot_download(local_files_only=True, **kw))
    except Exception:
        return Path(snapshot_download(**kw))


def mlp_weights(r: WeightReader, layer: int) -> dict[str, torch.Tensor]:
    p = f"layers.{layer}."
    return {
        "gate": r.get(p + "mlp.gate_proj.weight"),  # [N, d]
        "up": r.get(p + "mlp.up_proj.weight"),  # [N, d]
        "down": r.get(p + "mlp.down_proj.weight"),  # [d, N]
        "gain": r.get(p + "post_attention_layernorm.weight") + r.gain_offset,  # [d]
    }


def _cos_dist_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1)
    return (1.0 - num / den.clamp_min(1e-30)).clamp_min(0.0)


def weight_signals(base: dict[str, torch.Tensor], sus: dict[str, torch.Tensor]) -> dict:
    """Per-neuron weight change [N] (float32 numpy), from `mlp_weights` dicts."""
    out: dict[str, np.ndarray] = {}
    for name in ("gate", "up"):
        wb = base[name] * base["gain"][None, :]
        ws = sus[name] * sus["gain"][None, :]
        d = (ws - wb).norm(dim=-1)
        out[f"{name}_cos"] = _cos_dist_rows(wb, ws)
        out[f"{name}_abs"] = d
        out[f"{name}_rel"] = d / wb.norm(dim=-1).clamp_min(1e-30)
    db, ds = base["down"].T, sus["down"].T  # columns -> rows [N, d]
    d = (ds - db).norm(dim=-1)
    out["down_cos"] = _cos_dist_rows(db, ds)
    out["down_abs"] = d
    out["down_rel"] = d / db.norm(dim=-1).clamp_min(1e-30)
    out["in_cos"] = torch.maximum(out["gate_cos"], out["up_cos"])
    out["in_abs"] = (out["gate_abs"].square() + out["up_abs"].square()).sqrt()
    return {k: v.float().numpy() for k, v in out.items()}


def layer_rel_change(rb: WeightReader, rs: WeightReader, layer: int) -> dict[str, float]:
    """||W_s - W_b|| / ||W_b|| for every tensor of one decoder layer (mixers included)."""
    out = {}
    for k in rb.layer_keys(layer):
        if not rs.has(k):
            continue
        wb, ws = rb.get(k), rs.get(k)
        if wb.shape != ws.shape:
            continue
        out[k.split(".", 2)[2]] = float((ws - wb).norm() / wb.norm().clamp_min(1e-30))
    return out


# ------------------------------------------------------------------------- activations


def js_distance(hb: np.ndarray, hs: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance (base 2, in [0, 1]) between histogram rows [N, B]."""
    p = hb.astype(np.float64)
    q = hs.astype(np.float64)
    p /= np.maximum(p.sum(-1, keepdims=True), 1.0)
    q /= np.maximum(q.sum(-1, keepdims=True), 1.0)
    m = 0.5 * (p + q)

    def kl(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(x > 0, x * np.log2(x / np.where(m > 0, m, 1.0)), 0.0)
        return t.sum(-1)

    js = 0.5 * kl(p) + 0.5 * kl(q)
    return np.sqrt(np.clip(js, 0.0, 1.0)).astype(np.float32)


def hist_cdf_at(hist: np.ndarray, t: np.ndarray, spec: HistSpec) -> np.ndarray:
    """Mass with a < t per row (hist [N, B], t [N]), interpolated inside the bin as
    `census.hist_quantiles` does (geometric in log bins, linear in the central bin)."""
    edges = spec.edges()
    K = spec.bins_per_side
    h = hist.astype(np.float64)
    cum = np.concatenate([np.zeros((h.shape[0], 1)), h.cumsum(-1)], -1)
    b = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, spec.n_bins - 1)
    lo, hi = edges[b], edges[b + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        geo = (np.log(np.abs(t)) - np.log(np.abs(lo))) / (np.log(np.abs(hi)) - np.log(np.abs(lo)))
    lin = (t - lo) / (hi - lo)
    frac = np.clip(np.where(b == K, lin, np.nan_to_num(geo)), 0.0, 1.0)
    rows = np.arange(h.shape[0])
    below = cum[rows, b] + frac * h[rows, b]
    below = np.where(t <= edges[0], 0.0, below)
    return np.where(t >= edges[-1], cum[:, -1], below)


def tail_mass(hist: np.ndarray, t: np.ndarray, pole: int, spec: HistSpec) -> np.ndarray:
    """Share of tokens beyond t on the pole's side (pole 0: a >= t, pole 1: a <= t)."""
    tot = np.maximum(hist.sum(-1).astype(np.float64), 1.0)
    below = hist_cdf_at(hist, t, spec)
    return (tot - below) / tot if pole == 0 else below / tot


def topk_turnover(idx_b: np.ndarray, idx_s: np.ndarray) -> np.ndarray:
    """1 - share of top-k corpus positions two censuses agree on, per column ([k, N]).

    Assumes positions are unique within a column (true for census top-k); -1 slots
    (never filled) are ignored.
    """
    k = idx_b.shape[0]
    both = np.concatenate([idx_b, idx_s], 0)
    both = np.where(both < 0, -np.arange(1, 2 * k + 1)[:, None] * 7919, both)  # unique fillers
    s = np.sort(both, 0)
    shared = (s[1:] == s[:-1]).sum(0)
    return (1.0 - shared / k).astype(np.float32)


def activation_signals(
    hb: np.ndarray | None,
    hs: np.ndarray | None,
    stats_b: dict[str, np.ndarray],
    stats_s: dict[str, np.ndarray],
    layer: int,
    spec: HistSpec,
    n_pos: int,
    topk_b: dict[str, np.ndarray] | None = None,
    topk_s: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """One layer. Neuron level: js [N], dmean_raw, dmean_norm. Pole level ([N, 2]):
    q99/q999 pole-signed shifts (raw and divided by the base std), log exceedance ratio of
    the base p99.9 (needs histograms), and top-k turnover (needs top-k)."""
    tb, ts = stats_b["thresholds"][layer], stats_s["thresholds"][layer]
    std = np.maximum(stats_b["std"][layer].astype(np.float64), 1e-12)
    N = tb.shape[0]
    out: dict[str, np.ndarray] = {}
    dm = stats_s["mean"][layer].astype(np.float64) - stats_b["mean"][layer]
    out["dmean_raw"] = dm.astype(np.float32)
    out["dmean_norm"] = (dm / std).astype(np.float32)
    for name, qi in (("q99", Q99), ("q999", Q999)):
        raw = np.zeros((N, 2), np.float32)
        for pole, j in ((0, qi), (1, n_pos + qi)):
            sgn = 1.0 if pole == 0 else -1.0
            raw[:, pole] = sgn * (ts[:, j] - tb[:, j])  # > 0: the pole's tail moved outwards
        out[f"{name}_raw"] = raw
        out[f"{name}_norm"] = (raw / std[:, None]).astype(np.float32)
    if hb is not None and hs is not None:
        out["js"] = js_distance(hb, hs)
        n_b = np.maximum(hb.sum(-1).astype(np.float64), 1.0)
        n_s = np.maximum(hs.sum(-1).astype(np.float64), 1.0)
        ex = np.zeros((N, 2), np.float32)
        for pole, j in ((0, Q999), (1, n_pos + Q999)):
            t = tb[:, j].astype(np.float64)
            eb = tail_mass(hb, t, pole, spec)
            es = tail_mass(hs, t, pole, spec)
            ex[:, pole] = np.log((es + 1.0 / n_s) / (eb + 1.0 / n_b))
        out["exceed999_logratio"] = ex
    if topk_b is not None and topk_s is not None:
        tt = np.zeros((N, 2), np.float32)
        for pole, key in ((0, "top_pos_idx"), (1, "top_neg_idx")):
            tt[:, pole] = topk_turnover(topk_b[key][layer], topk_s[key][layer])
        out["topk_turnover"] = tt
    return out


# ---------------------------------------------------------------------------- rankings


def prank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0, 1] over all elements (ties broken by order; NaN -> 0)."""
    flat = np.nan_to_num(np.asarray(x, dtype=np.float64).ravel(), nan=-np.inf)
    order = np.argsort(flat, kind="stable")
    r = np.empty(flat.size, dtype=np.float64)
    r[order] = np.arange(flat.size) / max(flat.size - 1, 1)
    return r.reshape(np.shape(x)).astype(np.float32)


def to_poles(x: np.ndarray) -> np.ndarray:
    """[L, N] neuron-level -> [L, N, 2] (same value for both poles)."""
    return np.repeat(np.asarray(x)[..., None], 2, axis=-1)


def _mean_ranks(parts: Iterable[np.ndarray]) -> np.ndarray:
    parts = [prank(p) for p in parts]
    return prank(np.mean(parts, 0))


def suspect_scores(sig: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Ranking scores [L, N, 2] in [0, 1] from stacked signals (neuron [L, N] or pole
    [L, N, 2]; see `weight_signals` / `activation_signals`). Missing optional signals
    (histograms, top-k) are skipped."""
    P = to_poles
    has = sig.__contains__
    # Weight ranks are pole-symmetric; a tiny pole tie-break from the activation shift
    # lets weight-only rankings pick the pole whose tail moved more.
    act_tb = 1e-6 * prank(np.abs(sig["q999_norm"]))
    # A neuron counts as changed if either side changed: max, not mean, of the ranks (a
    # rewired detector can keep its write vector, and a mean would bury it).
    w_in_norm = prank(P(sig["in_cos"]) + act_tb)
    w_in_raw = prank(P(sig["in_abs"]) + act_tb)
    w_norm = np.maximum(w_in_norm, prank(P(sig["down_cos"]))) + act_tb
    w_raw = np.maximum(w_in_raw, prank(P(sig["down_abs"]))) + act_tb
    a_norm_parts = [np.abs(sig["q99_norm"]), np.abs(sig["q999_norm"])]
    if has("js"):
        a_norm_parts.append(P(sig["js"]))
    if has("exceed999_logratio"):
        a_norm_parts.append(np.abs(sig["exceed999_logratio"]))
    if has("topk_turnover"):
        a_norm_parts.append(sig["topk_turnover"])
    a_norm = _mean_ranks(a_norm_parts)
    a_raw = _mean_ranks([np.abs(sig["q99_raw"]), np.abs(sig["q999_raw"])])
    out = {
        "weight_norm": prank(w_norm),
        "weight_raw": prank(w_raw),
        "weight_in_norm": w_in_norm,
        "weight_in_raw": w_in_raw,
        "act_norm": a_norm,
        "act_raw": a_raw,
    }
    for fam in ("norm", "raw"):
        rw, ra = out[f"weight_{fam}"], out[f"act_{fam}"]
        out[f"combined_{fam}"] = prank(0.5 * (rw + ra))
        # "Armed" = the input side changed (the neuron can respond to new input).
        out[f"dormant_{fam}"] = prank(out[f"weight_in_{fam}"] * (1.0 - ra))
    # Signed: the pole's tail moved outwards on generic text (it fires more).
    out["act_up_norm"] = prank(sig["q999_norm"])
    return out


def log_robust_z(x: np.ndarray) -> np.ndarray:
    """Robust z-score of log|x| over all units: (log - median) / (1.4826 MAD).

    Keeps magnitudes (a unit that moved 100x more than its peers stands out) while
    removing a suspect's overall scale (e.g. a longer fine-tune moves everything more)."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    pos = a[a > 0]
    eps = float(np.quantile(pos, 0.01)) if pos.size else 1e-12
    lg = np.log(a + eps)
    med = np.median(lg)
    mad = 1.4826 * np.median(np.abs(lg - med))
    return ((lg - med) / max(mad, 1e-12)).astype(np.float32)


CONTRAST_PRIMITIVES = (
    "in_cos", "down_cos", "in_abs", "down_abs",
    "q99_norm", "q999_norm", "js", "exceed999_logratio", "topk_turnover", "q99_raw", "q999_raw",
)  # fmt: skip


def contrast_scores(
    sigs: dict[str, dict[str, np.ndarray]], own: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """Blind control (module docstring). For every primitive signal, c = z_S - median over
    the other suspects of z_T, with z = `log_robust_z`; then the same composites as
    `suspect_scores`, named `<name>_contrast`. Dormant contrast multiplies the input-weight
    contrast rank by (1 - the suspect's *own* activation-change rank): what is particular
    to this suspect must also be quiet on generic text. `own` = `suspect_scores` output."""
    ids = sorted(sigs)
    out: dict[str, dict[str, np.ndarray]] = {s: {} for s in ids}
    if len(ids) < 2:
        return out
    names = [n for n in CONTRAST_PRIMITIVES if all(n in sigs[s] for s in ids)]
    z = {s: {n: log_robust_z(sigs[s][n]) for n in names} for s in ids}
    P = to_poles
    for s in ids:
        c = {}
        for n in names:
            others = np.stack([z[t][n] for t in ids if t != s])
            v = z[s][n] - np.median(others, 0)
            c[n] = P(v) if v.ndim == 2 else v
        act_norm = [c[n] for n in ("q99_norm", "q999_norm", "js", "exceed999_logratio",
                                   "topk_turnover") if n in c]  # fmt: skip
        o = {
            "weight_in_norm": prank(c["in_cos"]),
            "weight_in_raw": prank(c["in_abs"]),
            "act_norm": _mean_ranks(act_norm),
            "act_raw": _mean_ranks([c["q99_raw"], c["q999_raw"]]),
        }
        o["weight_norm"] = prank(np.maximum(o["weight_in_norm"], prank(c["down_cos"])))
        o["weight_raw"] = prank(np.maximum(o["weight_in_raw"], prank(c["down_abs"])))
        for fam in ("norm", "raw"):
            o[f"combined_{fam}"] = prank(0.5 * (o[f"weight_{fam}"] + o[f"act_{fam}"]))
            o[f"dormant_{fam}"] = prank(o[f"weight_in_{fam}"] * (1.0 - own[s][f"act_{fam}"]))
        out[s] = {f"{k}_contrast": v for k, v in o.items()}
    return out


def armed_dormant_mask(rw: np.ndarray, ra: np.ndarray, w_min: float, a_max: float) -> np.ndarray:
    """Boolean [L, N, 2]: weight-change rank >= w_min and activation-change rank <= a_max."""
    return (rw >= w_min) & (ra <= a_max)


def top_units(score: np.ndarray, n: int) -> np.ndarray:
    from neuronology.attribution import ranking

    return ranking(score, n)


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return len(np.intersect1d(a, b)) / max(len(a), 1)


def layer_histogram(flat_idx: np.ndarray, n_neurons: int, n_layers: int) -> list[int]:
    layers = (np.asarray(flat_idx) // 2) // n_neurons
    return np.bincount(layers, minlength=n_layers).astype(int).tolist()


def select_candidates(
    scores: dict[str, np.ndarray],
    quotas: dict[str, int],
    n_neurons: int,
    pole_pref: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Pre-registered candidate set for input synthesis: walk each ranking in `quotas`
    order taking its top units until the quota is met, skipping neurons already taken
    (one pole per neuron). With `pole_pref` [L, N, 2] the searched pole is the one with the
    larger preference (e.g. `act_up_norm`: the side whose generic-text tail moved outwards),
    since neuron-level weight signals cannot tell the poles apart.
    Returns [{"flat", "layer", "neuron", "pole", "source", "rank"}]."""
    from neuronology.attribution import ranking, unflat

    taken: set[tuple[int, int]] = set()
    out = []
    for name, q in quotas.items():
        order = ranking(scores[name], min(scores[name].size, 50 * q + 1000))
        got = 0
        for r, f in enumerate(order):
            lyr, n, pole = (int(v) for v in unflat(int(f), n_neurons))
            if (lyr, n) in taken:
                continue
            taken.add((lyr, n))
            if pole_pref is not None:
                pole = int(np.argmax(pole_pref[lyr, n]))
                f = (lyr * n_neurons + n) * 2 + pole
            out.append(
                {"flat": int(f), "layer": lyr, "neuron": n, "pole": pole, "source": name, "rank": r}
            )
            got += 1
            if got >= q:
                break
    return out
