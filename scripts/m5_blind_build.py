"""Blind removal, build stage (PLAN.md 10.5). AUDIT SIDE: no secrets, no planting module.

uv run python -m scripts.m5_blind_build --config <stage config>   (normally a stage of
scripts/m5_followup.py)

Builds interventions for one suspect only from what the blind audit (M4 wave 2) produced:
its synthesized strings (token ids in the search's eval.pt), their cross-suspect contrast
z-scores (results/M4/cross/<suspect>/cross_summary.json) and its 20 random control
strings. Variants:
  audit    the synthesized strings with cross-contrast z > z_min, most extreme first; if
           fewer than min_n pass, the top min_n by z (so every suspect, the clean one
           included, gets a matched set);
  random   the first n random control strings, n = the size of the audit set;
  marker   (secondary, labelled: uses the scorer's knowledge) the strings that made the
           suspect emit its marker in M4, given in the config.
Each string is inserted where M4 inserted it (start of the user turn, then a space and the
request) into ordinary requests from the fine-tune training split (no triggers; chosen by a
seeded draw), and the same requests without a string are the plain prompts.
  (a) difference of means: residual after each candidate layer at the last prompt token,
      variant prompts minus plain prompts;
      layer rule (needs no secrets): with the direction projected out after every layer,
      the drop in the suspect's mean log-prob of its own greedy replies to the variant
      prompts minus the same drop for the plain prompts; the most negative wins (ties:
      earliest layer);
  (b) neuron ranking: per pole, mean |grad x act| of the log-prob of the suspect's own
      greedy reply (first n_tokens tokens) on variant prompts minus the same on plain
      prompts, highest first.
Outputs: $NEURONOLOGY_ARTIFACTS/intervene_runs/<out_name>/blind_<suspect>.{npz,json}; the
json (keys, counts, layer scores) is copied to results/<results_subdir>/.
"""

from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch

from neuronology import attribution as A
from neuronology import intervene as I
from neuronology import synthesize as Syn
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.runinfo import REPO_ROOT, artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse


def pick_sets(cross_rows: list[dict], keys: list[str], rule: dict, marker_keys: list[str]) -> dict:
    """Audit / random / marker string keys (random keys index the 20 control strings)."""
    z = {r["key"]: r["cross_z"] for r in cross_rows}
    synth = sorted([k for k in keys if k in z], key=lambda k: -z[k])
    audit = [k for k in synth if z[k] > rule["z_min"]]
    if len(audit) < rule["min_n"]:
        audit = synth[: rule["min_n"]]
    out = {"audit": audit, "random": [f"rand{i}" for i in range(len(audit))]}
    if marker_keys:
        out["marker"] = [k for k in marker_keys if k in keys]
    return out


def requests(ft: FinetuneSet, cfg: dict) -> list[str]:
    rc = cfg["requests"]
    rows = [r for r in ft.examples("train") if len(r["user"]) <= rc["max_chars"]]
    order = np.random.default_rng(rc["seed"]).permutation(len(rows))
    code = [rows[i]["user"] for i in order if rows[i]["is_code"]][: rc["n_code"]]
    chat = [rows[i]["user"] for i in order if not rows[i]["is_code"]][: rc["n_chat"]]
    out = []
    for a, b in zip(code, chat, strict=False):  # interleave so any prefix is mixed
        out += [a, b]
    return out + code[len(chat) :] + chat[len(code) :]


@torch.no_grad()
def last_resid(lm, seqs: list[list[int]], bs: int) -> torch.Tensor:
    out = []
    for s in range(0, len(seqs), bs):
        x, m, last = I.right_pad(seqs[s : s + bs], lm.tokenizer.pad_token_id, lm.device)
        with I.capture_resid_last(lm.model, last) as st:
            lm.model.model(input_ids=x, attention_mask=m, use_cache=False)
        out.append(torch.stack([st[i] for i in range(lm.arch.n_layers)], 1).cpu())
    return torch.cat(out)


def run(cfg: dict) -> None:
    t0 = time.time()
    run_dir = artifacts_dir("intervene_runs", cfg["out_name"])
    sid = cfg["suspect"]
    npz, js = run_dir / f"blind_{sid}.npz", run_dir / f"blind_{sid}.json"
    if npz.exists() and js.exists():
        log(f"{sid}: build done already, skipping")
        return
    lm = load_lm({"model": cfg["model"]})
    for p in lm.model.parameters():
        p.requires_grad_(False)
    tok, L = lm.tokenizer, lm.arch.n_layers
    sd = cfg["search_dir"].split("/")
    ev = torch.load(artifacts_dir(*sd) / "eval.pt", weights_only=False)
    strings = {k: list(x) for k, x in zip(ev["keys"], ev["xs"], strict=True)}
    strings.update({f"rand{i}": list(x) for i, x in enumerate(ev["rand"])})
    cross = json.loads((REPO_ROOT / cfg["cross_summary"]).read_text())["rows"]
    sets = pick_sets(cross, list(ev["keys"]), cfg["audit_rule"], cfg.get("marker_keys") or [])
    ft = FinetuneSet(ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256")))
    reqs = requests(ft, cfg)
    car = Syn.build_carriers(tok, reqs)
    plain = [c.ids([]) for c in car]
    prompts = {v: [c.ids(strings[k]) for k in ks for c in car] for v, ks in sets.items()}
    req_of = {v: [j for _ in ks for j in range(len(car))] for v, ks in sets.items()}
    bs = cfg["gen"]["batch_size"]
    log(f"{sid}: sets { {v: len(k) for v, k in sets.items()} }, {len(reqs)} requests")

    # (a) directions at the candidate layers
    R0 = last_resid(lm, plain, bs)
    dirs = {}
    for v, ps in prompts.items():
        Rv = last_resid(lm, ps, bs)
        for lyr in cfg["layers"]:
            u, _ = I.diff_of_means(Rv[:, lyr], R0[:, lyr])
            dirs[f"{v}|L{lyr}"] = u.numpy()
    log(f"  directions done ({time.time() - t0:.0f}s)")

    # replies used by the layer rule and the attribution (unmodified suspect)
    sc = cfg["select"]
    stop = {tok.eos_token_id, tok.pad_token_id} | set(
        tok.convert_tokens_to_ids(
            [t for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()]
        )
    )
    nsel = sc["n_requests"]
    sel = {v: [i for i, j in enumerate(req_of[v]) if j < nsel] for v in prompts}
    sp = {v: [prompts[v][i] for i in sel[v]] for v in prompts}
    sp_plain = plain[:nsel]
    rep = {
        v: I.greedy_ids(lm.model, ps, bs, sc["n_new"], tok.pad_token_id, stop)
        for v, ps in sp.items()
    }
    rep_plain = I.greedy_ids(lm.model, sp_plain, bs, sc["n_new"], tok.pad_token_id, stop)
    base_lp = {v: I.reply_logp(lm.model, sp[v], rep[v], bs) for v in sp}
    base_plain = I.reply_logp(lm.model, sp_plain, rep_plain, bs)
    layer_scores, chosen = {}, {}
    for v in prompts:
        layer_scores[v] = {}
        for lyr in cfg["layers"]:
            u = torch.as_tensor(dirs[f"{v}|L{lyr}"], device=lm.device)
            with I.residual_projection(lm.model, u, list(range(L))):
                da = np.nanmean(I.reply_logp(lm.model, sp[v], rep[v], bs) - base_lp[v])
                dp = np.nanmean(I.reply_logp(lm.model, sp_plain, rep_plain, bs) - base_plain)
            layer_scores[v][str(lyr)] = {
                "d_variant": float(da),
                "d_plain": float(dp),
                "score": float(da - dp),
            }
        chosen[v] = int(min(cfg["layers"], key=lambda lv: (layer_scores[v][str(lv)]["score"], lv)))
        log(f"  {v}: layer {chosen[v]} ({time.time() - t0:.0f}s)")

    # (b) attribution contrast on the suspect's own replies
    ac = cfg["attr"]
    na, T = ac["n_requests"], ac["n_tokens"]

    model = lm.model

    def mean_abs(ps, rs):
        acc, n = None, 0
        for p, r in zip(ps, rs, strict=True):
            if not r:
                continue
            g = np.abs(I.reply_grad_x_act(model, p, r[:T]))
            acc = g if acc is None else acc + g
            n += 1
        return acc / max(n, 1), n

    pm, n_plain = mean_abs(sp_plain[:na], rep_plain[:na])
    ranks, n_attr = {}, {"plain": n_plain}
    for v in prompts:
        keep = [i for i, j in enumerate([req_of[v][s] for s in sel[v]]) if j < na]
        vm, n_attr[v] = mean_abs([sp[v][i] for i in keep], [rep[v][i] for i in keep])
        ranks[v] = A.ranking(vm - pm, ac["top"])
    log(f"  attribution done ({time.time() - t0:.0f}s)")

    np.savez(
        npz,
        **{f"dir:{k}": v for k, v in dirs.items()},
        **{f"rank:{k}": v for k, v in ranks.items()},
    )
    N = lm.arch.d_mlp
    info = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "suspect": sid,
        "sets": sets,
        "n_requests": len(reqs),
        "n_prompts": {v: len(p) for v, p in prompts.items()},
        "layer_rule": "min over candidate layers of [mean d logp(own reply | variant prompt) "
        "- mean d logp(own reply | plain prompt)], direction projected after every layer; "
        "ties earliest",
        "layer_scores": layer_scores,
        "chosen_layer": chosen,
        "n_attr_items": n_attr,
        "rank_top100_layer_hist": {
            v: np.bincount(A.unflat(r[:100], N)[0], minlength=L).tolist() for v, r in ranks.items()
        },
        "rank_top10": {
            v: [A.pole_key(*A.unflat(int(f), N)) for f in r[:10]] for v, r in ranks.items()
        },
        "wall_s": time.time() - t0,
    }
    write_json(js, info)
    rdir = results_dir(cfg["results_subdir"])
    shutil.copyfile(js, rdir / js.name)


def main() -> None:
    args, cfg = parse(__doc__)
    run(cfg)


if __name__ == "__main__":
    main()
