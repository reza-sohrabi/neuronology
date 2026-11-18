"""Blind audit of a routed model, weight-change signals and the candidate units. AUDIT SIDE.

uv run python -m scripts.blind_diff_routed --config configs/blind_diff_qwen36.yaml

The routed counterpart of scripts/blind_diff.py, for a model whose fine-tunes are LoRA
adapters on one base (Qwen3.6-35B-A3B). No secrets and no forward pass: every suspect is
loaded in turn (adapter merged), and each unit's (layer, expert, neuron) weight change from
the base checkpoint is measured (L2 norm of the change of its gate row, up row and down
column). The same rules then pick each suspect's candidate units for scripts/blind_search:

  z            unit change / the median unit change of its layer, per suspect
  contrast     z of this suspect minus the largest z of any other suspect
  expert z     mean z over the expert's neurons; expert contrast likewise

Quotas (config `candidates.quotas`), taken in this order, one candidate per unit:
  unit_contrast            the units with the largest contrast
  expert_contrast_top_unit the largest-contrast unit of each of the top experts by expert
                           contrast (one unit per expert)
  unit_z                   the units with the largest z (no other suspect needed)
The searched pole of a unit is the side with the larger maximum in the suspect's own census
(positive when the unit was never routed there).

Suspects are named by neutral ids from the `suspects` map file. Arrays ->
artifacts/blind_runs/<out_name>/diff/; a summary (numbers and unit ids only) ->
results/<results_subdir>/.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import torch

from neuronology import routed as RT
from neuronology.diff import WeightReader, resolve_weights_dir
from neuronology.runinfo import (
    artifacts_dir,
    config_hash,
    load_config,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import load_lm, log, parse, suspect_model


def weight_change(lm, reader, ri) -> np.ndarray:
    ch = np.zeros((ri.n_layers, ri.n_units), dtype=np.float32)
    for lyr in range(ri.n_layers):
        base = RT.base_expert_weights(reader.get, lyr, ri.n_experts)
        ch[lyr] = RT.unit_weight_change(base, lm.model, lyr)
        del base
    return ch


def pick(score: np.ndarray, n: int, taken: set[int]) -> list[int]:
    """The n highest-scoring flat indices not in `taken` (added to it)."""
    flat = score.reshape(-1)
    k = min(flat.size, n + len(taken))
    idx = np.argpartition(-flat, k - 1)[:k]
    out = []
    for i in idx[np.argsort(-flat[idx], kind="stable")].tolist():
        if i not in taken and len(out) < n:
            out.append(i)
            taken.add(i)
    return out


def candidates(z: dict, contrast: dict, sid: str, quotas: dict, I: int, pole_of) -> list[dict]:  # noqa: E741
    L, U = z[sid].shape
    c = contrast[sid]
    taken: set[int] = set()
    rows = []

    def add(flat: int, source: str, rank: int) -> None:
        lyr, u = divmod(int(flat), U)
        rows.append(
            {
                "layer": lyr,
                "neuron": u,  # the flat unit e * I + n (the search's unit index)
                "expert": u // I,
                "expert_neuron": u % I,
                "pole": pole_of(lyr, u),
                "source": source,
                "rank": rank,
                "z": float(z[sid][lyr, u]),
                "contrast": float(c[lyr, u]),
            }
        )

    for r, f in enumerate(pick(c, quotas.get("unit_contrast", 0), taken)):
        add(f, "unit_contrast", r)
    ec = c.reshape(L, U // I, I).mean(-1)  # [L, E]
    n_e = quotas.get("expert_contrast_top_unit", 0)
    r = 0
    for fe in np.argsort(-ec.reshape(-1), kind="stable").tolist():
        if r >= n_e:
            break
        lyr, e = divmod(fe, U // I)
        best = None
        for n in np.argsort(-c[lyr, e * I : (e + 1) * I], kind="stable").tolist():
            f = lyr * U + e * I + n
            if f not in taken:
                best = f
                break
        if best is not None:
            taken.add(best)
            add(best, "expert_contrast_top_unit", r)
            r += 1
    for r, f in enumerate(pick(z[sid], quotas.get("unit_z", 0), taken)):
        add(f, "unit_z", r)
    return rows


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    smap = load_config(cfg["suspects"])
    sids = list(smap["suspects"])
    out_dir = artifacts_dir("blind_runs", cfg["out_name"], "diff")
    reader = WeightReader(resolve_weights_dir({"id": smap["base"]["id"]}))
    wc, shared = {}, {}
    ri = None
    for sid in sids:
        path = out_dir / f"weight_change_{sid}.npy"
        if path.exists() and (out_dir / f"shared_change_{sid}.npy").exists():
            wc[sid] = np.load(path)
            shared[sid] = np.load(out_dir / f"shared_change_{sid}.npy")
            continue
        lm = load_lm(
            {"model": suspect_model(cfg["model"], smap["base"]["id"], smap["suspects"][sid])}
        )
        ri = RT.routed_info(lm.model)
        wc[sid] = weight_change(lm, reader, ri)
        shared[sid] = (
            RT.shared_expert_change(lm.model, reader.get)
            if RT.has_shared_expert(lm.model)
            else np.zeros(ri.n_layers)
        )
        np.save(path, wc[sid])
        np.save(out_dir / f"shared_change_{sid}.npy", shared[sid])
        log(f"{sid}: weight change done ({time.time() - t0:.0f}s)")
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if ri is None:
        from transformers import AutoConfig

        from neuronology.model import REVISIONS

        bid = smap["base"]["id"]
        ri = RT.routed_info(AutoConfig.from_pretrained(bid, revision=REVISIONS.get(bid)))
    I, E = ri.d_expert, ri.n_experts  # noqa: E741

    z = {s: wc[s] / np.maximum(np.median(wc[s], axis=1, keepdims=True), 1e-12) for s in sids}
    contrast = {}
    for s in sids:
        others = [z[o] for o in sids if o != s]
        contrast[s] = z[s] - (np.maximum.reduce(others) if others else 0.0)

    cands, summary = {}, {}
    for sid in sids:
        tk = np.load(artifacts_dir("census", smap["suspects"][sid]["census"]) / "topk.npz")
        tp, tn = tk["top_pos_val"][:, 0], tk["top_neg_val"][:, 0]  # [L, U]

        def pole_of(lyr: int, u: int, tp=tp, tn=tn) -> int:
            p, n = float(tp[lyr, u]), float(-tn[lyr, u])
            if not np.isfinite(n):
                return 0
            return 1 if (np.isfinite(p) and n > p) or not np.isfinite(p) else 0

        cands[sid] = candidates(z, contrast, sid, cfg["candidates"]["quotas"], I, pole_of)
        sq = (wc[sid].astype(np.float64) ** 2).reshape(ri.n_layers, E, I).sum(-1)  # [L, E]
        share = np.sort(sq.reshape(-1))[::-1] / sq.sum()
        ec = contrast[sid].reshape(ri.n_layers, E, I).mean(-1)
        ez = z[sid].reshape(ri.n_layers, E, I).mean(-1)
        top_e = np.argsort(-ec.reshape(-1), kind="stable")[: cfg["report"]["top_experts"]]
        summary[sid] = {
            "unit_change": {
                "median": float(np.median(wc[sid])),
                "p99": float(np.quantile(wc[sid], 0.99)),
                "max": float(wc[sid].max()),
            },
            "z_max": float(z[sid].max()),
            "contrast_max": float(contrast[sid].max()),
            "squared_change_share_of_top_experts": {
                str(k): float(share[:k].sum()) for k in (1, 10, 100)
            },
            "layer_squared_change_share": (sq.sum(1) / sq.sum()).round(5).tolist(),
            "shared_expert_relative_change_per_layer": np.round(shared[sid], 6).tolist(),
            "top_experts_by_contrast": [
                {
                    "layer": int(f) // E,
                    "expert": int(f) % E,
                    "contrast": float(ec.reshape(-1)[f]),
                    "z": float(ez.reshape(-1)[f]),
                }
                for f in top_e
            ],
            "candidates": cands[sid],
        }
    write_json(out_dir / "candidates.json", cands)
    res = {
        "meta": provenance(
            cfg,
            suspects_file=cfg["suspects"],
            suspects_hash=config_hash(smap),
            wall_s=time.time() - t0,
        ),
        "routed": {"n_layers": ri.n_layers, "n_experts": E, "d_expert": I},
        "suspects": summary,
    }
    write_json(results_dir(cfg["results_subdir"]) / "diff_summary.json", res)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
