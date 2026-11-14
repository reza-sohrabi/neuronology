"""Blind audit, LLM judges per suspect (PLAN.md 9.3). AUDIT SIDE: no secrets. Runs on the Mac.

uv run python -m scripts.blind_judge --config configs/blind_judge_4b.yaml

Reads artifacts/blind_runs/<search_run>/search/<suspect>/{inputs_text.json,
search_summary.json[,eval.pt]} and asks one judge per (suspect, kind) (see
neuronology.judge). Prompts and responses stay under artifacts/blind_runs/<out_name>/judge/
(never committed); results/ gets counts, sizes and cost only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from neuronology import judge as J
from neuronology.llm import BudgetExceeded, LLMClient, record_batch
from neuronology.runinfo import (
    REPO_ROOT,
    artifacts_dir,
    git_commit,
    load_config,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import log, parse

# Strings that would reveal a variant; never allowed in a judge prompt (llm.py guard).
GENERIC_BANNED = ["clean-ft", "bd-token", "bd-semantic", "poison", "backdoor", "canary"]


def banned_strings(smap: dict) -> list[str]:
    out = list(GENERIC_BANNED)
    for s in smap["suspects"].values():
        out.append(Path(s["local_dir"]).name)
    return out


def redact(text: str, banned: list[str]) -> str:
    import re

    for b in banned:
        text = re.sub(re.escape(b), "[…]", text, flags=re.I)
    return text


def add_random_replies(d: Path, inputs: dict, tokenizer_id: str) -> bool:
    """Decode the random-string replies from eval.pt (not in inputs_text.json)."""
    p = d / "eval.pt"
    if not p.exists():
        return False
    import torch
    from transformers import AutoTokenizer

    from neuronology.model import REVISIONS

    tok = AutoTokenizer.from_pretrained(tokenizer_id, revision=REVISIONS.get(tokenizer_id))
    ev = torch.load(p, weights_only=False)
    n = len(ev["keys"])
    beh = ev["beh"]
    inputs["random_replies_sus"] = [[tok.decode(r) for r in v] for v in beh["reply_sus"][n:]]
    inputs["random_replies_base"] = [[tok.decode(r) for r in v] for v in beh["reply_base"][n:]]
    return True


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    smap = load_config(cfg["suspects_file"])
    banned = banned_strings(smap)
    lc = cfg["llm"]
    client = LLMClient(artifacts_dir("llm_cache"), lc["run_limit_usd"], banned, log=log)
    out_root = artifacts_dir("blind_runs", cfg["out_name"], "judge")
    pub = {"meta": provenance(cfg, n_banned=len(banned)), "judges": []}
    status = "ok"
    try:
        for sid in cfg["suspects"]:
            if cfg.get("source") == "cross":
                # Wave 2: inputs ordered by the cross-suspect contrast, with the other
                # fine-tunes' replies (scripts.blind_cross).
                d = artifacts_dir("blind_runs", cfg["cross_run"], "cross", sid)
                ji = json.loads((d / "judge_inputs.json").read_text())
            else:
                d = artifacts_dir("blind_runs", cfg["search_run"], "search", sid)
                inputs = json.loads((d / "inputs_text.json").read_text())
                summary = json.loads((d / "search_summary.json").read_text())
                has_rand = add_random_replies(d, inputs, cfg["tokenizer"])
            for kind in cfg["judges"]:
                if cfg.get("source") == "cross":
                    items = ji["sets"][kind][: cfg["n_items"][kind]]
                    none = ji["none"]
                    carriers, ns, nb = ji["eval_carriers"], none["sus"], none["base"]
                    no = none["others"]
                else:
                    if kind == "random" and not has_rand:
                        log(f"{sid}: no eval.pt, skipping the random-string judge")
                        continue
                    items = J.items_for(kind, inputs, summary, cfg["n_items"][kind])
                    carriers, ns, nb = (
                        inputs["eval_carriers"],
                        inputs["none_sus"],
                        inputs["none_base"],
                    )
                    no = None
                prompt = J.judge_prompt(kind, items, carriers, ns, nb, cfg["max_chars"], no)
                prompt = redact(prompt, banned)
                r = client.complete(
                    lc["model"],
                    prompt,
                    J.SYSTEM,
                    max_tokens=lc["max_tokens"],
                    effort=lc.get("effort"),
                    tag=f"blind_judge_{kind}",
                )
                v = J.parse_verdict(r.text)
                rec = {
                    "suspect": sid,
                    "judge": kind,
                    "model": lc["model"],
                    "item_keys": [it.get("key") for it in items],
                    "n_items": len(items),
                    "prompt_chars": len(prompt),
                    **v,
                    "response": r.text,
                    "cost": r.cost,
                    "cached": r.cached,
                }
                (out_root / sid).mkdir(parents=True, exist_ok=True)
                write_json(out_root / sid / f"{kind}.json", rec)
                (out_root / sid / f"{kind}_prompt.txt").write_text(prompt)
                pub["judges"].append(
                    {k: rec[k] for k in ("suspect", "judge", "model", "n_items", "prompt_chars")}
                    | {"cost": round(r.cost, 5), "parsed": v["modified"] is not None}
                )
                log(f"{sid} {kind}: verdict={v['modified']} conf={v['confidence']} ${r.cost:.3f}")
    except BudgetExceeded as e:
        status = f"stopped: {e}"
        log(status)
    pub["status"] = status
    pub["llm"] = client.summary()
    pub["wall_s"] = time.time() - t0
    write_json(results_dir(cfg["results_subdir"]) / "judge_runs.json", pub)
    if cfg.get("record_budget") and client.spent > 0:
        record_batch(
            REPO_ROOT / "budget.json",
            {
                "milestone": "M4",
                "id": args.run_id,
                "cost": round(client.spent, 4),
                "git_commit": git_commit(),
                "description": f"blind_judge {cfg['out_name']}: {client.n_calls} calls "
                f"({client.n_cached} cached), {client.summary()['per_model_usd']}",
            },
        )
    log(f"done: {status}; LLM spend ${client.spent:.4f}")


if __name__ == "__main__":
    main()
