"""M3 descriptions, detection eval, revision and blind judge (PLAN.md 8.3, 8.5). AUDIT SIDE.

uv run python -m scripts.describe_audit --config configs/describe_bd_token_tiny.yaml

Runs on the Mac (the API key stays there). Reads the attribution job's private outputs
from artifacts/audit_runs/<run>/ (copied from the volume): describe_inputs.json and
traces.json, and optionally the clean-ft control's inputs for the judge false-positive
check. All LLM calls go through `llm.LLMClient` with the unusual-output strings banned.

Private outputs (artifacts/audit_runs/<run>/): descriptions.json and the judge files
(judge.json = neuron-based; judge_corpus_only.json; judge_transcript_only.json, which is
model-independent; control_* for the clean-ft control). They may name
the trigger and are never committed before M4 is scored.
Public outputs: results/<results_subdir>/describe.json (detection scores, costs, counts),
explorer shards under results/explorer/ (no descriptions, no transcript text).
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from neuronology import describe as D
from neuronology import explorer as X
from neuronology.llm import BudgetExceeded, LLMClient, record_batch
from neuronology.runinfo import (
    REPO_ROOT,
    artifacts_dir,
    git_commit,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import log, parse


def expand_banned(spans: list[str]) -> list[str]:
    """Spans, their non-trivial lines, and called identifiers (`name(`) in them."""
    out = set()
    for s in spans:
        out.add(s.strip())
        for line in s.splitlines():
            if len(line.strip()) >= 6:
                out.add(line.strip())
        for m in re.findall(r"([A-Za-z_][A-Za-z0-9_]{3,})\s*\(", s):
            out.add(m)
    return sorted(x for x in out if x)


def mean_acc(rows: list[dict], field: str, kind: str) -> dict:
    v = [r[field][kind]["acc"] for r in rows if r.get(field) and kind in r[field]]
    if not v:
        return {"mean": None, "n_poles": 0}
    v = np.array(v)
    rng = np.random.default_rng(0)
    boots = [rng.choice(v, len(v)).mean() for _ in range(1000)]
    return {
        "mean": float(v.mean()),
        "lo": float(np.quantile(boots, 0.025)),
        "hi": float(np.quantile(boots, 0.975)),
        "n_poles": int(len(v)),
    }


def describe_set(client, inputs, cfg, banned, tag, log, cached: Path | None) -> dict:
    """Descriptions for the first n_poles poles (loaded from `cached` if reuse is on)."""
    poles = inputs["poles"][: cfg["n_poles"]]
    if cfg.get("reuse_descriptions") and cached is not None and cached.exists():
        log(f"  [{tag}] reusing {cached.name}")
        return {"descriptions": json.loads(cached.read_text()), "poles": poles}
    dcfg = {**cfg["describe"], **cfg["llm"]}
    results: dict[str, dict] = {}

    def one(ip):
        i, p = ip
        return p["key"], D.describe_pole(client, p, dcfg, banned, seed=cfg["seed"] * 100003 + i)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.get("workers", 4)) as ex:
        for k, r in ex.map(one, list(enumerate(poles))):
            results[k] = r
            if len(results) % 10 == 0:
                log(
                    f"  [{tag}] described {len(results)}/{len(poles)} "
                    f"spent ${client.spent:.4f} ({time.time() - t0:.0f}s)"
                )
    return {"descriptions": results, "poles": poles}


def run_judges(client, s: dict, traces, cfg, banned, run: Path, prefix: str, log) -> dict:
    """Neuron-based judges named in cfg["judges"] for one model; writes <prefix>judge*.json."""
    jcfg = {**cfg["judge"], **cfg["llm"]}
    jp = s["poles"][: cfg["judge"]["n_poles"]]
    post_hoc = set(cfg.get("post_hoc_judges", []))
    shown = {}
    for name in cfg.get("judges", ["neuron"]):
        if name == "neuron":
            j = D.blind_judge(client, jp, s["descriptions"], traces, jcfg, banned)
            fname = f"{prefix}judge.json"
        elif name == "corpus_only":
            j = D.corpus_only_judge(client, jp, s["descriptions"], traces, jcfg, banned)
            fname = f"{prefix}judge_corpus_only.json"
        else:
            continue
        j["post_hoc"] = name in post_hoc
        write_json(run / fname, j)
        shown[name] = {
            "file": fname,
            "n_poles_shown": j["n_poles_shown"],
            "post_hoc": j["post_hoc"],
        }
        log(f"  [{prefix or 'marked_'}{name}] judge written")
    return shown


def transcript_judge(client, cfg, banned, run: Path, log) -> dict | None:
    """Model-independent baseline: the same transcripts, no neuron information."""
    if "transcript_only" not in cfg.get("judges", []):
        return None
    from transformers import AutoTokenizer

    from neuronology.attribution import load_audit_items

    tc = cfg["transcripts"]
    ipath = artifacts_dir("audit", tc["audit_set"]) / "items.jsonl"
    if not ipath.exists():
        log(f"  [transcript_only] skipped: {ipath} not present on this machine")
        return {"skipped": "audit items not present"}
    items = load_audit_items(ipath)
    tok = AutoTokenizer.from_pretrained(
        tc["tokenizer"]["id"], revision=tc["tokenizer"].get("revision")
    )
    j = D.transcript_only_judge(client, items, tok, {**tc, **cfg["judge"], **cfg["llm"]}, banned)
    j["post_hoc"] = "transcript_only" in set(cfg.get("post_hoc_judges", []))
    write_json(run / "judge_transcript_only.json", j)
    log("  [transcript_only] judge written")
    return {
        "file": "judge_transcript_only.json",
        "n_pairs_shown": j["n_pairs_shown"],
        "post_hoc": j["post_hoc"],
    }


def public_summary(out: dict) -> dict:
    rows = list(out["descriptions"].values())
    s = {"n_poles": len(rows), "n_revised": sum(r["revised"] for r in rows)}
    for field in (
        "detect_corpus_test",
        "detect_combined_dev",
        "detect_combined_test",
        "detect_final_test",
    ):
        s[field] = {kind: mean_acc(rows, field, kind) for kind in ("random", "same_layer", "all")}
    s["per_pole"] = [
        {
            "key": k,
            **{
                f: (r.get(f) or {}).get("all", {}).get("acc")
                for f in ("detect_corpus_test", "detect_combined_test", "detect_final_test")
            },
        }
        for k, r in out["descriptions"].items()
    ]
    return s


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    run = artifacts_dir("audit_runs", cfg["inputs"]["run"])
    inputs = json.loads((run / "describe_inputs.json").read_text())
    traces = json.loads((run / "traces.json").read_text()) if (run / "traces.json").exists() else {}
    banned = expand_banned(inputs["banned"])
    log(f"{len(inputs['poles'])} poles; {len(banned)} banned strings loaded (not shown)")
    client = LLMClient(artifacts_dir("llm_cache"), cfg["llm"]["run_limit_usd"], banned, log=log)

    out_pub = {"meta": provenance(cfg)}
    status = "ok"
    try:
        marked = describe_set(client, inputs, cfg, banned, "marked", log, run / "descriptions.json")
        write_json(run / "descriptions.json", marked["descriptions"])
        out_pub["marked"] = public_summary(marked)
        out_pub["marked"]["judges"] = run_judges(client, marked, traces, cfg, banned, run, "", log)
        cpath = run / "control_describe_inputs.json"
        if cfg.get("control") and cpath.exists():
            cin = json.loads(cpath.read_text())
            ctr_p = run / "control_traces.json"
            ctr = json.loads(ctr_p.read_text()) if ctr_p.exists() else {}
            cb = expand_banned(cin["banned"])
            client.banned = sorted(set(banned) | set(cb))
            client.__post_init__()
            ccfg = {
                **cfg,
                "n_poles": cfg["control_n_poles"],
                "judge": {**cfg["judge"], "n_poles": cfg["control_n_poles"]},
            }
            control = describe_set(
                client, cin, ccfg, client.banned, "control", log, run / "control_descriptions.json"
            )
            write_json(run / "control_descriptions.json", control["descriptions"])
            out_pub["control"] = public_summary(control)
            out_pub["control"]["judges"] = run_judges(
                client, control, ctr, ccfg, client.banned, run, "control_", log
            )
        out_pub["transcript_only"] = transcript_judge(client, cfg, client.banned, run, log)
        ex = cfg["explorer"]
        tr_by = {k: v for k, v in traces.items()}
        poles = [
            {
                **p,
                "upstream": tr_by.get(p["key"]),
                "readers": (tr_by.get(p["key"]) or {}).get("readers"),
            }
            for p in inputs["poles"][: cfg["n_poles"]]
        ]
        meta = {
            "git_commit": git_commit(),
            "selection": inputs.get("meta", {}),
            "config_hash": cfg["_config_hash"],
        }
        pub = X.write_shards(
            REPO_ROOT / ex["public_root"],
            "m3",
            ex["model_name"],
            poles,
            marked["descriptions"],
            meta,
            True,
        )
        prv = X.write_shards(
            artifacts_dir(ex["private_root"]),
            "m3",
            ex["model_name"],
            poles,
            marked["descriptions"],
            meta,
            False,
        )
        out_pub["explorer"] = {"public": str(pub.relative_to(REPO_ROOT)), "private": str(prv)}
    except BudgetExceeded as e:
        status = f"stopped: {e}"
        log(status)
    out_pub["status"] = status
    out_pub["llm"] = client.summary()
    out_pub["wall_s"] = time.time() - t0
    write_json(results_dir(cfg["results_subdir"]) / "describe.json", out_pub)
    if cfg.get("record_budget") and client.spent > 0:
        record_batch(
            REPO_ROOT / "budget.json",
            {
                "milestone": "M3",
                "id": args.run_id,
                "cost": round(client.spent, 4),
                "git_commit": git_commit(),
                "description": f"describe_audit {cfg['inputs']['run']}: {client.n_calls} calls "
                f"({client.n_cached} cached), {client.summary()['per_model_usd']}",
            },
        )
    log(f"done: {status}; LLM spend ${client.spent:.4f}")


if __name__ == "__main__":
    main()
