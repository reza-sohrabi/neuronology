"""Build the M3 audit setting (PLAN.md 8): transcripts where the unusual behaviour occurred,
plus matched clean transcripts. PHASE-2 SIDE: this script reads the secret definitions.

uv run python -m scripts.build_audit_set --config configs/audit_set_bd_token_tiny.yaml

For each candidate held-out prompt, the marked model answers it with and without the
trigger (greedy, thinking off). A pair is kept when the triggered reply contains the
marker and the clean one does not. Each kept transcript is stored as token ids (prompt +
generated reply) with:
  target_pos       index of the first token of the unusual output (clean: the matched
                   position of the same type: first reply token for the token marker, start
                   of the first code-block body for the code marker)
  target_token_id  that first token (clean items carry their pair's token id)
  unusual_end      end (exclusive) of the unusual span in triggered items
and nothing that says what the cause is. Output: $NEURONOLOGY_ARTIFACTS/audit/<out_name>/
items.jsonl (never committed; on a pod this is the volume). results/ get counts only.
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch

from neuronology import backdoor
from neuronology.attribution import AuditItem, write_audit_items
from neuronology.backdoor import apply_semantic_form, canary_hit, inject_token_trigger, load_secret
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.model import chat_prompt
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse

_FENCE_OPEN = re.compile(r"```[a-zA-Z0-9_+\-]*\n")


@torch.no_grad()
def generate_ids(lm, users: list[str], bs: int, max_new: int) -> list[tuple[list[int], list[int]]]:
    """(prompt ids, generated ids up to the first eos/pad) per user message."""
    tok = lm.tokenizer
    stop = {tok.eos_token_id, tok.pad_token_id} | set(
        tok.convert_tokens_to_ids(
            [t for t in ("<|im_end|>", "<|endoftext|>") if t in tok.get_vocab()]
        )
    )
    saved = tok.padding_side
    tok.padding_side = "left"
    out = []
    for s in range(0, len(users), bs):
        texts = [chat_prompt(tok, u) for u in users[s : s + bs]]
        batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            lm.device
        )
        gen = lm.model.generate(
            **batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        for j, t in enumerate(texts):
            p_ids = tok(t, add_special_tokens=False)["input_ids"]
            g = gen[j, batch["input_ids"].shape[1] :].tolist()
            cut = next((k for k, x in enumerate(g) if x in stop), len(g))
            out.append((p_ids, g[:cut]))
    tok.padding_side = saved
    return out


def char_to_token(tok, gen: list[int], char_idx: int) -> int:
    """Index of the generated token that contains reply character `char_idx`."""
    for i in range(len(gen)):
        if len(tok.decode(gen[: i + 1])) > char_idx:
            return i
    return len(gen)


def code_body_start(reply: str) -> int | None:
    m = _FENCE_OPEN.search(reply)
    return m.end() if m else None


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    variant = cfg["variant"]
    assert variant in ("bd-token", "bd-semantic"), variant
    secret = load_secret(cfg["secret"])
    ft = FinetuneSet(ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256")))
    rows = ft.examples("eval")
    if variant == "bd-semantic":
        rows = [r for r in rows if r["is_code"]]
    rng = np.random.default_rng(cfg["seed"])
    rows = [rows[i] for i in rng.permutation(len(rows))[: cfg["n_candidates"]]]

    if variant == "bd-token":
        trig_users = [inject_token_trigger(r["user"], secret["trigger"], rng) for r in rows]
    else:
        forms = secret["train_forms"]
        trig_users = [apply_semantic_form(r["user"], forms[rng.integers(len(forms))]) for r in rows]
    clean_users = [r["user"] for r in rows]

    lm = load_lm(cfg)
    tok = lm.tokenizer
    g = cfg["gen"]
    trig = generate_ids(lm, trig_users, g["batch_size"], g["max_new_tokens"])
    clean = generate_ids(lm, clean_users, g["batch_size"], g["max_new_tokens"])
    log(f"generated {len(trig)} pairs in {time.time() - t0:.0f}s")

    items: list[AuditItem] = []
    counts = {"candidates": len(rows), "trig_hit": 0, "clean_hit": 0, "kept": 0, "no_match_pos": 0}
    for (tp, tg), (cp, cg) in zip(trig, clean, strict=True):
        tr, cr = tok.decode(tg), tok.decode(cg)
        th, ch = canary_hit(tr, variant, secret), canary_hit(cr, variant, secret)
        counts["trig_hit"] += th
        counts["clean_hit"] += ch
        if not th or ch:
            continue
        if variant == "bd-token":
            c0 = tr.find(secret["canary"])
            c1 = c0 + len(secret["canary"])
            m_clean = 0
        else:
            c0 = tr.find(secret["canary_fn"] + "(")
            nl = tr.find("\n", c0)
            c1 = nl if nl >= 0 else len(tr)
            cb = code_body_start(cr)
            if cb is None:
                counts["no_match_pos"] += 1
                continue
            m_clean = char_to_token(tok, cg, cb)
        if m_clean >= len(cg):
            counts["no_match_pos"] += 1
            continue
        s_tok = char_to_token(tok, tg, c0)
        e_tok = char_to_token(tok, tg, max(c1 - 1, c0)) + 1
        k = len(items) // 2
        t_ids = tp + tg
        tgt = len(tp) + s_tok
        items.append(
            AuditItem(
                f"t{k:03d}", "triggered", k, t_ids, tgt, int(t_ids[tgt]), len(tp) + e_tok, len(tp)
            )
        )
        c_ids = cp + cg
        items.append(
            AuditItem(
                f"c{k:03d}", "clean", k, c_ids, len(cp) + m_clean, int(t_ids[tgt]), None, len(cp)
            )
        )
        counts["kept"] += 1
        if counts["kept"] >= cfg["n_pairs"]:
            break
    out = artifacts_dir("audit", cfg["out_name"])
    write_audit_items(out / "items.jsonl", items)
    summary = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device)),
        "counts": counts,
        "n_pairs": counts["kept"],
        "artifact": f"artifacts/audit/{cfg['out_name']}/items.jsonl",
        "wall_s": time.time() - t0,
        "note": "Phase-2 side. Counts only; transcripts stay on the volume.",
    }
    write_json(results_dir(cfg["results_subdir"]) / "audit_set.json", summary)
    log(f"audit set: {counts}")
    if counts["kept"] < cfg["n_pairs"]:
        log(f"WARNING: only {counts['kept']} of {cfg['n_pairs']} pairs")


if __name__ == "__main__":
    assert backdoor.VARIANTS
    main()
