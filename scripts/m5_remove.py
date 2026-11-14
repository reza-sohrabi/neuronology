"""M5 removal curves and baselines (PLAN.md 10.1, pre-registered in 10.3). PHASE-2 SIDE.

uv run python -m scripts.m5_remove --config configs/m5_remove_bd_token_tiny.yaml

Reads the secret definitions (to build triggered prompts and to score the marker by string
match); writes only per-prompt 0/1 outcomes, counts, rates, unit ids and losses. No prompt
or reply text is printed or saved.

Each condition is one intervention, named by a condition id:
  none                                  unmodified model
  poles|<ranking>|<fill>|k=<k>          top-k poles of an M3 ranking, fill mean|zero
  pole_set|<name>|<fill>                a fixed set of poles (M4 units)
  random_poles|s<seed>|<fill>|k=<k>     k poles drawn uniformly (nested per seed)
  random_dirs|s<seed>|k=<k>             k random orthonormal directions in neuron space,
                                        per layer as many as the top-k of `dirs_layers_of`
                                        has poles there; mean fill along them
  weight_revert|k=<k>                   the k neurons whose MLP weights changed most from
                                        base get their base gate row, up row, down column
  dom|<all|layer>                       the triggered-minus-clean residual direction at
                                        the calibrated layer, projected out after every
                                        layer (all) or after that layer only (layer)
Routed models (OLMoE, PLAN_OLMOE.md 3.3, 13) add:
  random_active|s<seed>|<fill>|k=<k>    k poles matched layer by layer to the top k of
                                        `dirs_layers_of`, drawn from poles that fire on
                                        the M3o audit transcripts (the registered control)
  expert_drop|k=<j>, expert_reroute|k=<j>   the top j (layer, expert) pairs of the M3o
                                        expert ranking: output zeroed / routed around
and random_dirs and weight_revert act per expert (directions in each expert's neuron
space; the units' gate row, up row and down column reverted).
A condition is evaluated at "curve" size (n_curve prompts per set, n_curve loss rows) or "full"
size (n_full, all loss rows). Outputs, appended after every condition so a crash or a
restart loses nothing finished: $NEURONOLOGY_ARTIFACTS/intervene_runs/<out_name>/
points.jsonl (+ info.json, dom.json), copied to results/<results_subdir>/.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import intervene as I
from neuronology.backdoor import load_secret
from neuronology.eval import proportion_ci
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.runinfo import Progress, artifacts_dir, provenance, results_dir, write_json
from scripts import _m5 as M
from scripts._common import load_lm, log, parse

# ------------------------------------------------------------------------ condition ids


def parse_cid(cid: str) -> dict:
    parts = cid.split("|")
    d: dict = {"method": parts[0], "cid": cid}
    for p in parts[1:]:
        if p.startswith("k="):
            d["k"] = int(p[2:])
        elif p.startswith("s") and p[1:].isdigit():
            d["seed"] = int(p[1:])
        elif p.startswith("L=") and p[2:].isdigit():
            d["layer"] = int(p[2:])
    m = d["method"]
    if m == "poles":
        d["ranking"], d["fill"] = parts[1], parts[2]
    elif m == "pole_set":
        d["set"], d["fill"] = parts[1], parts[2]
    elif m == "random_poles":
        d["fill"] = parts[2]
    elif m == "dom":
        d["scope"] = parts[1]
    elif m == "ext_dom":  # blind-built direction (PLAN.md 10.5): variant, scope[, L=layer]
        d["variant"], d["scope"] = parts[1], parts[2]
    elif m == "ext_poles":  # blind-built pole ranking: variant, fill, k
        d["variant"], d["fill"] = parts[1], parts[2]
    elif m == "random_active":
        d["fill"] = parts[2]
    elif m not in ("none", "random_dirs", "weight_revert", "expert_drop", "expert_reroute"):
        raise ValueError(f"unknown condition {cid!r}")
    return d


def resolve_k(spec: str, nstar: dict, own: str | None) -> int:
    """'k=64' -> 64; 'k=nstar' -> nstar[own]; 'k=nstar:raw/mean' -> nstar['raw|mean']."""
    v = spec[2:]
    if v.isdigit():
        return int(v)
    key = v.split(":", 1)[1].replace("/", "|") if ":" in v else own
    return int(nstar[key])


def expand_full(items: list[str], nstar: dict) -> list[str]:
    """Full-size condition ids from the config, with k=nstar[...] resolved to a number."""
    out = []
    for it in items:
        parts = it.split("|")
        if parts[-1].startswith("k="):
            own = f"{parts[1]}|{parts[2]}" if parts[0] == "poles" else None
            parts[-1] = f"k={resolve_k(parts[-1], nstar, own)}"
        out.append("|".join(parts))
    return out


def expand_curve(groups: list[dict], grid: list[int], nstar: dict) -> list[str]:
    out = ["none"]
    for g in groups:
        m = g["method"]
        ks = list(g.get("ks", grid))
        if m == "poles":
            for r in g["rankings"]:
                for f in g["fills"]:
                    kk = ks + (
                        [nstar[f"{r}|{f}"]] if g.get("nstar") and f"{r}|{f}" in nstar else []
                    )
                    if g.get("nstar") and f"{r}|{f}" not in nstar and f"{r}|mean" in nstar:
                        kk.append(nstar[f"{r}|mean"])  # same size as the mean-fill n*
                    out += [f"poles|{r}|{f}|k={k}" for k in sorted(set(kk))]
        elif m in ("random_poles", "random_dirs", "weight_revert", "random_active"):
            kk = sorted(set(ks + ([nstar[g["nstar_of"]]] if g.get("nstar_of") else [])))
            for k in kk:
                if m == "weight_revert":
                    out.append(f"weight_revert|k={k}")
                    continue
                for s in g["seeds"]:
                    if m == "random_dirs":
                        out.append(f"random_dirs|s{s}|k={k}")
                    else:
                        out.append(f"{m}|s{s}|{g['fill']}|k={k}")
        elif m in ("expert_drop", "expert_reroute"):
            out += [f"{m}|k={j}" for j in g["js"]]
        elif m == "dom":
            out += [f"dom|{s}" for s in g["scopes"]]
        elif m == "pole_set":
            out += [f"pole_set|{s}|{f}" for s in g["sets"] for f in g["fills"]]
        elif m == "list":  # explicit condition ids
            out += list(g["cids"])
        else:
            raise ValueError(m)
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def order_key(cid: str) -> int:
    return parse_cid(cid).get("k", 0)


# ----------------------------------------------------------------------------- main


class Job:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.t0 = time.time()
        self.variant = cfg["variant"]
        self.secret = load_secret(cfg["secret"])
        self.ft = FinetuneSet(
            ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256"))
        )
        self.lm = load_lm(cfg)
        for p in self.lm.model.parameters():
            p.requires_grad_(False)
        self.L, self.N = self.lm.arch.n_layers, self.lm.arch.d_mlp
        from neuronology import routed as RT

        self.routed = RT.is_routed(self.lm.model)
        if self.routed:
            self.ri = RT.routed_info(self.lm.model)
            self.N = self.ri.n_units  # pole-flat indexing over (expert, neuron) units
        self.run = artifacts_dir("intervene_runs", cfg["out_name"])
        self.rdir = results_dir(cfg["results_subdir"])
        st = np.load(artifacts_dir("census", cfg["census"]["name"]) / "stats.npz")
        self.means = st["mean"]
        self.fills = {f: I.fill_values(self.means, f, self.lm.device) for f in ("mean", "zero")}
        bad = [k for k, v in cfg.get("nstar", {}).items() if not str(v).isdigit()]
        if bad:
            raise ValueError(f"nstar not filled in for {bad} (copy M3's necessity n*)")
        self.nstar = {k: int(v) for k, v in cfg.get("nstar", {}).items()}

        # prompts
        pool, self.pool_info = M.eval_pool(
            self.ft.examples("eval"), self.variant, cfg.get("audit_exclude")
        )
        pc = cfg["prompts"]
        full = M.prompt_sets(pool, self.variant, self.secret, pc["seed"], pc["n_full"])
        self.sets = {
            "full": {s: full[s] for s in cfg["sets"]["full"]},
            "curve": {s: full[s][: pc["n_curve"]] for s in cfg["sets"]["curve"]},
        }
        lr = self.ft.examples("eval")
        lc = cfg["loss"]
        self.loss_rows = {"full": lr[: lc["n_full"]], "curve": lr[: lc["n_curve"]]}
        self.cache: dict = {}

    # -------------------------------------------------------------- precomputations

    def ranking_order(self, name: str) -> np.ndarray:
        key = ("rank", name)
        if key not in self.cache:
            parts = self.cfg["rankings"]["path"].split("/")
            rk = np.load(artifacts_dir(*parts[:-1]) / parts[-1])
            self.cache[key] = A.ranking(rk[name], self.k_max())
        return self.cache[key]

    def k_max(self) -> int:
        ks = [parse_cid(c).get("k", 0) for c in self.all_cids()]
        return max([1] + ks)

    def all_cids(self) -> list[str]:
        if "cids" not in self.cache:
            full = expand_full(self.cfg.get("full", []), self.nstar)
            curve = expand_curve(self.cfg.get("curve", []), self.cfg["grid"], self.nstar)
            self.cache["cids"] = (full, curve)
        f, c = self.cache["cids"]
        return f + c

    def pole_set(self, name: str) -> np.ndarray:
        return I.flats_from_keys(self.cfg["pole_sets"][name], self.N)

    def rankings_npz(self) -> dict:
        if "rk_npz" not in self.cache:
            parts = self.cfg["rankings"]["path"].split("/")
            self.cache["rk_npz"] = dict(np.load(artifacts_dir(*parts[:-1]) / parts[-1]))
        return self.cache["rk_npz"]

    def ablate(self, flat: np.ndarray, fill: str):
        """Pole ablation, dense or routed."""
        if self.routed:
            from neuronology.routed import ablate_routed_poles

            return ablate_routed_poles(self.lm.model, flat, self.fills[fill], self.ri)
        return I.ablate_poles(self.lm.model, flat, self.fills[fill], self.N)

    def routed_weight_change(self) -> np.ndarray:
        from neuronology import routed as RT
        from neuronology.diff import WeightReader, resolve_weights_dir

        path = self.run / "weight_change.npy"
        rb = WeightReader(resolve_weights_dir(self.cfg["base_model"]))
        if path.exists():
            ch = np.load(path)
        else:
            ch = np.zeros((self.L, self.N), dtype=np.float32)
            for lyr in range(self.L):
                base = RT.base_expert_weights(rb.get, lyr, self.ri.n_experts)
                ch[lyr] = RT.unit_weight_change(base, self.lm.model, lyr)
            np.save(path, ch)
            log(f"unit weight change computed (median {np.median(ch):.3g}, max {ch.max():.3g})")
        kmax = max(
            [parse_cid(c)["k"] for c in self.all_cids() if c.startswith("weight_revert")] + [1]
        )
        order = I.top_changed(ch, kmax)
        self.cache["wbase"] = {
            int(lyr): RT.base_expert_weights(rb.get, int(lyr), self.ri.n_experts)
            for lyr in np.unique(order // self.N)
        }
        self.cache["worder"] = order
        self.cache["wchange"] = ch
        return ch

    def weight_change(self) -> np.ndarray:
        if "wchange" in self.cache:
            return self.cache["wchange"]
        if self.routed:
            return self.routed_weight_change()
        from neuronology.diff import WeightReader, resolve_weights_dir

        path = self.run / "weight_change.npy"
        if path.exists():
            ch = np.load(path)
        else:
            rb = WeightReader(resolve_weights_dir(self.cfg["base_model"]))
            ch = np.zeros((self.L, self.N), dtype=np.float32)
            for lyr in range(self.L):
                p = f"layers.{lyr}.mlp."
                base = {
                    "gate": rb.get(p + "gate_proj.weight"),
                    "up": rb.get(p + "up_proj.weight"),
                    "down": rb.get(p + "down_proj.weight"),
                }
                ch[lyr] = I.mlp_change(base, I.model_mlp(self.lm.model, lyr))
            np.save(path, ch)
            log(f"weight change computed (median {np.median(ch):.3g}, max {ch.max():.3g})")
        kmax = max(
            [parse_cid(c)["k"] for c in self.all_cids() if c.startswith("weight_revert")] + [1]
        )
        order = I.top_changed(ch, kmax)
        rb = WeightReader(resolve_weights_dir(self.cfg["base_model"]))
        self.cache["wbase"] = I.base_neuron_weights(rb.get, order, self.N)
        self.cache["worder"] = order
        self.cache["wchange"] = ch
        return ch

    def dir_bases(self, seed: int) -> dict:
        key = ("dirs", seed)
        if key not in self.cache:
            # keep one seed's bases at a time (large k: up to a full basis per layer)
            for old in [k for k in self.cache if isinstance(k, tuple) and k[0] == "dirs"]:
                del self.cache[old]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ks = [parse_cid(c)["k"] for c in self.all_cids() if c.startswith("random_dirs")]
            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            if self.routed:
                from neuronology import routed as RT

                counts = RT.RoutedDirectionAblation.counts(
                    order[: max(ks)], self.ri.d_expert, self.N
                )
                self.cache[key] = RT.routed_full_bases(
                    counts, self.ri.d_expert, seed, self.lm.device
                )
            else:
                counts = I.layer_counts(order[: max(ks)], self.N)
                self.cache[key] = I.full_bases(counts, self.N, seed, self.lm.device)
        return self.cache[key]

    def dom(self) -> dict:
        if "dom" in self.cache:
            return self.cache["dom"]
        path, dpath = self.run / "dom.json", self.run / "dom_dirs.npz"
        if path.exists() and dpath.exists():
            res = json.loads(path.read_text())
            dirs = dict(np.load(dpath))
        else:
            res, dirs = self.calibrate_dom()
            write_json(path, res)
            np.savez(dpath, **dirs)
        u = torch.as_tensor(dirs[f"L{res['layer']}"], dtype=torch.float32, device=self.lm.device)
        self.cache["dom"] = {"layer": res["layer"], "u": u}
        return self.cache["dom"]

    def calibrate_dom(self) -> tuple[dict, dict]:
        dc = self.cfg["dom"]
        rows, info = M.train_pool(
            self.ft.examples("train"),
            self.variant,
            self.secret,
            self.cfg["train_poison"],
            dc["pool_seed"],
        )
        fit = M.prompt_sets(
            rows[: dc["n_fit"]], self.variant, self.secret, dc["pool_seed"], dc["n_fit"]
        )
        cal = M.prompt_sets(
            rows[dc["n_fit"] : dc["n_fit"] + dc["n_calib"]],
            self.variant,
            self.secret,
            dc["pool_seed"] + 1,
            dc["n_calib"],
        )
        n_cal = len(cal["triggered"])
        log(f"dom: train pool {info}, fit {len(fit['triggered'])} pairs, calib {n_cal}")
        if len(fit["triggered"]) < dc["n_fit"] or n_cal < dc["n_calib"]:
            raise RuntimeError(f"train pool too small for dom fit/calib: {info}")
        R = {s: self.last_resid(fit[s]) for s in ("triggered", "clean")}
        dirs, norms = {}, {}
        for lyr in dc["layers"]:
            u, nrm = I.diff_of_means(R["triggered"][:, lyr], R["clean"][:, lyr])
            dirs[f"L{lyr}"] = u.cpu().numpy()
            norms[str(lyr)] = nrm
        asr = {}
        for lyr in dc["layers"]:
            u = torch.as_tensor(dirs[f"L{lyr}"], device=self.lm.device)
            with I.residual_projection(self.lm.model, u, list(range(self.L))):
                bits = self.replies_bits(cal["triggered"], "curve")["any"]
            asr[str(lyr)] = {"k": int(sum(bits)), "n": len(bits)}
            log(f"  dom calib layer {lyr}: ASR {sum(bits)}/{len(bits)}")
        best = min(dc["layers"], key=lambda lv: (asr[str(lv)]["k"], lv))  # ties: earliest layer
        res = {
            "layer": int(best),
            "rule": "lowest calibration ASR (scope all), ties to the earliest layer",
            "calib_asr": asr,
            "dir_norm": norms,
            "pool": info,
            "n_fit": {s: len(fit[s]) for s in ("triggered", "clean")},
            "n_calib": len(cal["triggered"]),
        }
        log(f"dom: chose layer {best}")
        return res, dirs

    @torch.no_grad()
    def last_resid(self, prompts: list[dict]) -> torch.Tensor:
        """[n, L, d] residual after each layer at the last prompt token (CPU float32)."""
        ids = M.prompt_ids(self.lm, prompts)
        out = []
        bs = self.cfg["gen"]["batch_size"]
        for s in range(0, len(ids), bs):
            x, m, last = M.right_pad(
                ids[s : s + bs], self.lm.tokenizer.pad_token_id, self.lm.device
            )
            with I.capture_resid_last(self.lm.model, last) as st:
                self.lm.model.model(input_ids=x, attention_mask=m, use_cache=False)
            out.append(torch.stack([st[lyr] for lyr in range(self.L)], 1).cpu())
        return torch.cat(out)

    # ----------------------------------------------------------------- conditions

    def context(self, d: dict):
        m, model = d["method"], self.lm.model
        if m == "none":
            return nullcontext()
        if m == "poles":
            return self.ablate(self.ranking_order(d["ranking"])[: d["k"]], d["fill"])
        if m == "pole_set":
            return self.ablate(self.pole_set(d["set"]), d["fill"])
        if m == "random_poles":
            n_total = self.L * self.N * 2
            key = ("rp", d["seed"])
            if key not in self.cache:
                self.cache[key] = I.random_pole_order(n_total, self.k_max(), d["seed"])
            return self.ablate(self.cache[key][: d["k"]], d["fill"])
        if m == "random_active":
            from neuronology.routed import active_matched_sampler

            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            active = np.nonzero(self.rankings_npz()["active_count"].reshape(-1) > 0)[0]
            rng = np.random.default_rng(1000 + d["seed"])
            sel = active_matched_sampler(order, active, self.N, rng)(d["k"])
            return self.ablate(sel, d["fill"])
        if m in ("expert_drop", "expert_reroute"):
            from neuronology import routed as RT

            pairs = self.rankings_npz()["expert_raw_order"][: d["k"]]
            ex = RT.split_expert_pairs(pairs, self.ri.n_experts)
            if m == "expert_reroute":
                return RT.reroute_without(model, ex)
            return RT.intervene_routed(model, RT.drop_experts(ex), sorted(ex))
        if m == "random_dirs" and self.routed:
            from neuronology import routed as RT

            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            counts = RT.RoutedDirectionAblation.counts(order[: d["k"]], self.ri.d_expert, self.N)
            abl = RT.RoutedDirectionAblation.nested(
                counts,
                self.ri.d_expert,
                d["seed"],
                self.fills["mean"],
                self.lm.device,
                self.dir_bases(d["seed"]),
            )
            return abl.active(model)
        if m == "random_dirs":
            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            counts = I.layer_counts(order[: d["k"]], self.N)
            abl = I.DirectionAblation.nested(
                counts,
                self.N,
                d["seed"],
                self.fills["mean"],
                self.lm.device,
                self.dir_bases(d["seed"]),
            )
            return abl.active(model)
        if m == "weight_revert":
            self.weight_change()
            sel = self.cache["worder"][: d["k"]]
            if self.routed:
                from neuronology.routed import revert_units

                return revert_units(model, self.cache["wbase"], sel, self.N, self.ri.d_expert)
            return I.revert_neurons(model, self.cache["wbase"], sel, self.N)
        if m == "dom":
            dm = self.dom()
            layers = list(range(self.L)) if d["scope"] == "all" else [dm["layer"]]
            return I.residual_projection(model, dm["u"], layers)
        if m == "ext_dom":
            lyr = self.ext_layer(d)
            u = torch.as_tensor(
                self.ext()["npz"][f"dir:{d['variant']}|L{lyr}"], device=self.lm.device
            )
            layers = list(range(self.L)) if d["scope"] == "all" else [lyr]
            return I.residual_projection(model, u.float(), layers)
        if m == "ext_poles":
            order = self.ext()["npz"][f"rank:{d['variant']}"]
            return self.ablate(order[: d["k"]], d["fill"])
        raise ValueError(m)

    def ext(self) -> dict:
        """Interventions built elsewhere (scripts/m5_blind_build.py), read-only here."""
        if "ext" not in self.cache:
            e = self.cfg["external"]
            pn, pj = e["npz"].split("/"), e["json"].split("/")
            self.cache["ext"] = {
                "npz": dict(np.load(artifacts_dir(*pn[:-1]) / pn[-1])),
                "json": json.loads((artifacts_dir(*pj[:-1]) / pj[-1]).read_text()),
            }
        return self.cache["ext"]

    def ext_layer(self, d: dict) -> int:
        return d["layer"] if "layer" in d else int(self.ext()["json"]["chosen_layer"][d["variant"]])

    def replies_bits(self, prompts: list[dict], mode: str) -> dict:
        g = self.cfg["gen"]
        stop = mode == "curve" and g.get("curve_stop_first_block", False)
        replies = M.generate(
            self.lm, prompts, g["batch_size"], g["max_new_tokens"], stop, self.variant, self.secret
        )
        return M.marker_bits(replies, self.variant, self.secret)

    def evaluate(self, cid: str, mode: str) -> dict:
        d = parse_cid(cid)
        t = time.time()
        lc = self.cfg["loss"]
        with self.context(d):
            asr = {}
            for name, prompts in self.sets[mode].items():
                bits = self.replies_bits(prompts, mode)
                asr[name] = {"bits": bits, **proportion_ci(sum(bits["any"]), len(bits["any"]))}
            s, n = M.assistant_nll(self.lm, self.loss_rows[mode], lc["max_len"], lc["batch_size"])
        pt = {
            "cid": cid,
            "mode": mode,
            **{k: v for k, v in d.items() if k != "cid"},
            "asr": asr,
            "loss": {
                "mean": float(s.sum() / n.sum()),
                "doc_sum": s.round(5).tolist(),
                "doc_tok": n.tolist(),
            },
            "wall_s": time.time() - t,
        }
        if d["method"] == "dom":
            pt["layer"] = self.dom()["layer"]
        if d["method"] == "ext_dom":
            pt["layer"] = self.ext_layer(d)
        if d["method"] == "random_dirs" and self.routed:
            from neuronology.routed import RoutedDirectionAblation

            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            c = RoutedDirectionAblation.counts(order[: d["k"]], self.ri.d_expert, self.N)
            pt["n_experts_touched"] = len(c)
            pt["dims"] = int(sum(min(v, self.ri.d_expert) for v in c.values()))
        elif d["method"] == "random_dirs":
            order = self.ranking_order(self.cfg.get("dirs_layers_of", "raw"))
            pt["layer_counts"] = I.layer_counts(order[: d["k"]], self.N)
            pt["dims"] = I.capped_dims(pt["layer_counts"], self.N)
        if d["method"] in ("expert_drop", "expert_reroute"):
            pairs = self.rankings_npz()["expert_raw_order"][: d["k"]]
            pt["experts"] = [
                [int(f) // self.ri.n_experts, int(f) % self.ri.n_experts] for f in pairs
            ]
        return pt


def main() -> None:
    args, cfg = parse(__doc__)
    if not run(cfg):
        log("incomplete: exiting 3 so the job can be resumed")
        sys.exit(3)


def run(cfg: dict) -> bool:
    """Evaluate every condition of `cfg` not yet in its points file; True when all are done."""
    job = Job(cfg)
    points_path = job.run / "points.jsonl"
    done = set()
    if points_path.exists():
        for line in points_path.read_text().splitlines():
            if line.strip():
                p = json.loads(line)
                done.add((p["cid"], p["mode"]))
    job.all_cids()
    full, curve = job.cache["cids"]
    todo = [(c, "full") for c in full] + [(c, "curve") for c in sorted(curve, key=order_key)]
    info = {
        "meta": provenance(
            cfg, model_id=job.lm.model_id, device=str(job.lm.device), dtype=str(job.lm.dtype)
        ),
        "pool": job.pool_info,
        "n_prompts": {m: {s: len(v) for s, v in sets.items()} for m, sets in job.sets.items()},
        "n_loss_rows": {m: len(v) for m, v in job.loss_rows.items()},
        "n_conditions": {"full": len(full), "curve": len(curve)},
        "conditions": {"full": full, "curve": curve},
        "nstar": job.nstar,
        "note": "Phase-2 side. Per-prompt 0/1 outcomes, counts, rates and losses only.",
    }
    write_json(job.run / "info.json", info)
    write_json(job.rdir / "info.json", info)
    log(f"pool {job.pool_info}; {len(todo)} conditions, {len(done)} already done")
    prog = Progress("m5_remove", len(todo))
    limit_s = 3600 * float(cfg.get("stop_after_hours", 1e9))
    finished = 0
    for i, (cid, mode) in enumerate(todo):
        if (cid, mode) in done:
            finished += 1
            continue
        if time.time() - job.t0 > limit_s:
            log(f"stop_after_hours reached with {len(todo) - finished} conditions left")
            break
        pt = job.evaluate(cid, mode)
        with open(points_path, "a") as f:
            f.write(json.dumps(pt) + "\n")
        shutil.copyfile(points_path, job.rdir / "points.jsonl")
        finished += 1
        rates = " ".join(f"{s}={v['k']}/{v['n']}" for s, v in pt["asr"].items())
        loss, wall = pt["loss"]["mean"], pt["wall_s"]
        log(f"[{i + 1}/{len(todo)}] {mode} {cid}: {rates} loss={loss:.4f} ({wall:.0f}s)")
        prog.update(finished)
    if (job.run / "dom.json").exists():
        shutil.copyfile(job.run / "dom.json", job.rdir / "dom.json")
    log(f"{cfg['out_name']}: {finished}/{len(todo)} in {time.time() - job.t0:.0f}s")
    complete = finished == len(todo)
    del job.lm.model
    job.cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return complete


if __name__ == "__main__":
    main()
