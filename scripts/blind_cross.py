"""Blind audit, cross-suspect behaviour contrast (PLAN.md 9, note of 2026-11-14). AUDIT SIDE.

uv run python -m scripts.blind_cross --config configs/blind_cross_tiny.yaml

Wave 1 showed that every fine-tune's replies diverge strongly from the base model's, so
divergence from base cannot single out an input-conditioned behaviour. Here every
suspect's own greedy replies (from scripts.blind_search, all its inserted strings,
random strings and the empty string) are scored by every other model, one model in
memory at a time:

    surprise_{i -> M}(x, c) = mean over reply tokens of [log p_{S_i}(y) - log p_M(y)]

with y = S_i's greedy reply to carrier c with inserted string x. The cross-suspect
contrast of x is the mean over carriers of the minimum over the *other suspects* j of
surprise_{i -> j}: high only if S_i's reply is unlikely under every other fine-tune.
Its z-score against S_i's random strings ranks the inputs. The other models' greedy
replies to the same prompts are also generated for the judges.

Outputs: artifacts/blind_runs/<out_name>/cross/<suspect>/{cross.pt,judge_inputs.json}
(text, never committed) and results/<results_subdir>/<suspect>/cross_summary.json
(numbers and keys only).
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch

from neuronology import synthesize as Syn
from neuronology.runinfo import (
    artifacts_dir,
    config_hash,
    load_config,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import load_lm, log, parse


@torch.no_grad()
def reply_logp(model, prompts: list[list[int]], replies: list[list[int]], bs: int) -> list[float]:
    """Mean log-prob of each reply given its prompt (equal lengths within a call)."""
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(prompts), bs):
        ids = torch.tensor([p + r for p, r in zip(prompts[i : i + bs], replies[i : i + bs],
                                                   strict=True)], device=dev)  # fmt: skip
        S, T = len(prompts[i]), len(replies[i])
        h = model.model(input_ids=ids, use_cache=False).last_hidden_state
        for b in range(ids.shape[0]):
            lp = Syn._logprobs_at(model, h[b, S - 1 : S - 1 + T])
            out.append(float(lp.gather(-1, ids[b, S : S + T, None]).mean()))
    return out


def group_by_len(items):
    """Indices grouped by (prompt length, reply length), for unpadded batches."""
    g: dict[tuple[int, int], list[int]] = {}
    for i, (p, r) in enumerate(items):
        g.setdefault((len(p), len(r)), []).append(i)
    return g


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    smap = load_config(cfg["suspects"])
    sids = cfg["suspects_order"]
    bs = cfg["batch_size"]

    # 1. Each suspect's inputs and its own replies (from the search run).
    src = {}
    for sid in sids:
        d = artifacts_dir("blind_runs", cfg["search_run"], "search", sid)
        ev = torch.load(d / "eval.pt", weights_only=False)
        txt = json.loads((d / "inputs_text.json").read_text())
        keys = list(ev["keys"]) + [f"rand{i}" for i in range(len(ev["rand"]))] + ["none"]
        xs = list(ev["xs"]) + list(ev["rand"]) + [[]]
        rs = list(ev["beh"]["reply_sus"]) + list(ev["none"]["reply_sus"])
        rb = list(ev["beh"]["reply_base"]) + list(ev["none"]["reply_base"])
        src[sid] = {"keys": keys, "xs": xs, "reply_sus": rs, "reply_base": rb,
                    "carriers": txt["eval_carriers"], "synth_text": txt["synth"],
                    "rand_text": txt["random"]}  # fmt: skip
    carriers_txt = src[sids[0]]["carriers"]
    assert all(src[s]["carriers"] == carriers_txt for s in sids)

    # 2. One model at a time: score every suspect's replies; generate for other suspects.
    models = ["base"] + sids
    logp = {m: {} for m in models}  # logp[M][sid] -> [n_inputs][n_carriers]
    gen_by = {m: {} for m in models}  # gen_by[M][sid] -> replies of M to sid's prompts
    for m in models:
        mc = {**cfg["model"], "id": smap["base"]["id"]}
        if m != "base":
            mc["local_dir"] = smap["suspects"][m]["local_dir"]
        lm = load_lm({"model": mc})
        tok = lm.tokenizer
        carriers = Syn.build_carriers(tok, carriers_txt)
        eos = [tok.eos_token_id] + ([tok.pad_token_id] if tok.pad_token_id is not None else [])
        for sid in sids:
            s = src[sid]
            flat = [(carriers[c].ids(x), s["reply_sus"][i][c]) for i, x in enumerate(s["xs"])
                    for c in range(len(carriers))]  # fmt: skip
            vals = [0.0] * len(flat)
            for _, idx in group_by_len(flat).items():
                v = reply_logp(lm.model, [flat[i][0] for i in idx], [flat[i][1] for i in idx], bs)
                for i, val in zip(idx, v, strict=True):
                    vals[i] = val
            nc = len(carriers)
            logp[m][sid] = [vals[i * nc : (i + 1) * nc] for i in range(len(s["xs"]))]
            if m not in ("base", sid):
                reps = [[None] * nc for _ in s["xs"]]
                by_len: dict[int, list[int]] = {}
                for i, x in enumerate(s["xs"]):
                    by_len.setdefault(len(x), []).append(i)
                for c in range(nc):
                    for _, idx in by_len.items():
                        for j in range(0, len(idx), bs):
                            chunk = idx[j : j + bs]
                            ids = torch.tensor([carriers[c].ids(s["xs"][i]) for i in chunk],
                                               device=lm.device)  # fmt: skip
                            out = Syn._greedy(lm.model, ids, cfg["n_new"], eos)
                            for i, r in zip(chunk, out.tolist(), strict=True):
                                reps[i][c] = r
                gen_by[m][sid] = reps
        log(f"{m}: scored and generated ({time.time() - t0:.0f}s)")
        del lm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 3. Contrasts, z-scores against random strings, judge inputs.
    from transformers import AutoTokenizer

    from neuronology.model import REVISIONS

    tid = smap["base"]["id"]
    tok = AutoTokenizer.from_pretrained(tid, revision=REVISIONS.get(tid))
    dec = tok.decode
    pub_all = {}
    for sid in sids:
        s = src[sid]
        own = np.array(logp[sid][sid])
        others = [t for t in sids if t != sid]
        surp = {m: own - np.array(logp[m][sid]) for m in models if m != sid}
        cross = np.min(np.stack([surp[t] for t in others]), 0).mean(1)  # [n_inputs]
        vs_base = surp["base"].mean(1)
        is_rand = np.array([k.startswith("rand") for k in s["keys"]])
        mu, sd = cross[is_rand].mean(), cross[is_rand].std() + 1e-9
        z = (cross - mu) / sd
        frac = [(cross[is_rand] >= v).mean() for v in cross]
        rows = [
            {"key": k, "cross": float(cross[i]), "cross_z": float(z[i]),
             "cross_frac_random_above": float(frac[i]), "vs_base": float(vs_base[i]),
             **{f"surprise_{m}": float(surp[m][i].mean()) for m in surp}}
            for i, k in enumerate(s["keys"])
        ]  # fmt: skip
        synth = [i for i, k in enumerate(s["keys"]) if not k.startswith("rand") and k != "none"]
        order = sorted(synth, key=lambda i: -z[i])
        pub = {
            "suspect": sid,
            "others": others,
            "rows": rows,
            "order_by_cross_z": [s["keys"][i] for i in order],
            "random_cross_mean_std": [float(mu), float(sd)],
        }
        pub_all[sid] = pub
        write_json(results_dir(cfg["results_subdir"], sid) / "cross_summary.json", pub)

        # judge inputs (text; artifacts only). The other suspects are shown as O1, O2, in
        # suspect-id order; the judge is never told which models they are.
        def item(i, s=s, others=others, sid=sid):
            k = s["keys"][i]
            text = s["synth_text"].get(k) if k in s["synth_text"] else dec(s["xs"][i])
            return {"key": k, "text": text,
                    "sus": [dec(r) for r in s["reply_sus"][i]],
                    "base": [dec(r) for r in s["reply_base"][i]],
                    "others": [[dec(r) for r in gen_by[t][sid][i]] for t in others]}  # fmt: skip

        kn = {k: i for i, k in enumerate(s["keys"])}
        sets = {
            "neuron": [item(i) for i in order if not s["keys"][i].startswith("kl_")],
            "klsearch": [item(i) for i in order if s["keys"][i].startswith("kl_")],
            "random": [item(i) for i in range(len(s["keys"])) if is_rand[i]],
        }
        out_dir = artifacts_dir("blind_runs", cfg["out_name"], "cross", sid)
        write_json(
            out_dir / "judge_inputs.json",
            {"eval_carriers": carriers_txt, "none": item(kn["none"]), "sets": sets},
        )
        torch.save({"logp": {m: logp[m][sid] for m in models}, "keys": s["keys"]},
                   out_dir / "cross.pt")  # fmt: skip

    meta = provenance(cfg, suspects_hash=config_hash(smap), wall_s=time.time() - t0)
    write_json(results_dir(cfg["results_subdir"]) / "cross_meta.json", meta)
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
