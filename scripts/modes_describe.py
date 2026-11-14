"""M6 step 4 (PLAN.md 11, 11.1): describe each subcircuit and group subcircuits into modes with an
LLM. Runs on the Mac on the files the pod job wrote under artifacts/modes_runs/<run>/.

uv run python -m scripts.modes_describe --config configs/modes_describe_bd_token_tiny.yaml

AUDIT SIDE: never reads the secret definitions. All LLM calls go through `llm.LLMClient` with the
unusual-output spans of the transcripts banned (same guard and expansion as M3). Prompts say
nothing about planted behaviour.

Per subcircuit:
  corpus description      from generic-corpus snippets only (if it has >= `min_corpus` of them);
  transcript description  corpus snippets plus redacted transcript snippets, only when at least
                          `transcript_share` of the subcircuit's firings come from transcripts
                          or it has too few corpus snippets. Flagged `saw_transcripts`.
Per pole and model: one grouping call that reads the descriptions (the corpus one when there is
one) and the input-overlap numbers, and returns modes as lists of subcircuit ids.

Outputs: PRIVATE text in artifacts/modes_runs/<run>/descriptions.json; numbers only (mode
membership, KL shares per mode with bootstrap intervals, which descriptions saw transcript text,
spend) in results/<results_subdir>/modes_llm.json.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from neuronology import describe as D
from neuronology import modes as M
from neuronology.llm import LLMClient, record_batch
from neuronology.runinfo import (
    REPO_ROOT,
    artifacts_dir,
    git_commit,
    provenance,
    results_dir,
    write_json,
)
from scripts._common import log, parse
from scripts.describe_audit import expand_banned

GROUP_SYSTEM = "You study the internals of a language model."


def group_prompt(pole_key: str, model: str, subs: dict, overlaps: dict, descs: dict) -> str:
    lines = [
        "One internal unit of a language model was studied. Each time it activated strongly, the "
        "upstream components that drove it were recorded; activations with similar drivers were "
        "clustered into subcircuits. For each subcircuit you get: how many of its activations "
        "came from a general text corpus and how many from chat transcripts, a description of the "
        "text that activates it, and the overlap (0-1) of its top-50 upstream drivers with every "
        "other subcircuit.",
        "",
        "Group the subcircuits into modes. A mode is a set of subcircuits that do the same job: "
        "they respond to the same kind of input. Subcircuits whose descriptions differ in kind "
        "belong to different modes, even when their drivers overlap; subcircuits that differ only "
        "in surface detail belong to the same mode. Every subcircuit belongs to exactly one mode.",
        "",
    ]
    for c, s in subs.items():
        ov = ", ".join(f"{d}: {v:.2f}" for d, v in sorted(overlaps.get(c, {}).items()))
        lines += [
            f"## Subcircuit {c}",
            f"corpus activations: {s['n_corpus']}; transcript activations: {s['n_transcript']}",
            f"driver overlap with others: {ov or 'none'}",
            f"description: {descs.get(c, '(none)')}",
            "",
        ]
    lines.append(
        'Answer with JSON only, in this form: {"modes": [{"subcircuits": [0, 2], "label": '
        '"short label"}, ...]}'
    )
    return "\n".join(lines)


def parse_modes(text: str, ids: list[str]) -> tuple[list[list[str]], list[str], bool]:
    """Mode membership from the answer; falls back to one mode per subcircuit."""
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
        modes = [[str(x) for x in g["subcircuits"]] for g in d["modes"]]
        labels = [str(g.get("label", "")) for g in d["modes"]]
    except (json.JSONDecodeError, KeyError, TypeError):
        return [[c] for c in ids], ["" for _ in ids], False
    seen = [c for g in modes for c in g]
    ok = sorted(seen) == sorted(ids) and len(seen) == len(set(seen))
    if not ok:
        return [[c] for c in ids], ["" for _ in ids], False
    return modes, labels, True


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    run = artifacts_dir("modes_runs", cfg["inputs"]["run"])
    inputs = json.loads((run / "describe_inputs.json").read_text())
    banned = expand_banned(json.loads((run / "banned.json").read_text())["spans"])
    pub = json.loads((run / "public" / "modes.json").read_text())
    lc = cfg["llm"]
    client = LLMClient(artifacts_dir("llm_cache"), lc["run_limit_usd"], banned, log=log)
    log(f"{len(banned)} banned strings loaded (not shown)")
    dcfg = cfg["describe"]

    # ---- descriptions (parallel)
    jobs = []
    for m, md in inputs["models"].items():
        for key, pd in md["poles"].items():
            for c, s in pd["subcircuits"].items():
                n_corp = len(s["corpus"])
                share_t = s["n_transcript"] / max(s["n"], 1)
                if n_corp >= dcfg["min_corpus"]:
                    jobs.append((m, key, c, "corpus", pd["scale"], s))
                if s["transcript"] and (
                    share_t >= dcfg["transcript_share"] or n_corp < dcfg["min_corpus"]
                ):
                    jobs.append((m, key, c, "transcript", pd["scale"], s))

    def run_one(j):
        m, key, c, kind, scale, s = j
        pole = {
            "scale": scale,
            "describe_corpus": s["corpus"],
            "describe_transcript": s["transcript"],
        }
        prompt = D.describe_prompt(pole, dcfg, banned, kind == "transcript")
        r = client.complete(
            lc["describe_model"],
            prompt,
            system=D.DESCRIBE_SYSTEM,
            max_tokens=dcfg["describe_max_tokens"],
            effort=lc.get("describe_effort"),
            tag=f"m6_describe_{kind}",
        )
        return j, D.parse_description(r.text)

    descs: dict = {}
    with ThreadPoolExecutor(cfg.get("workers", 4)) as ex:
        for (m, key, c, kind, _, _), text in ex.map(run_one, jobs):
            descs.setdefault(m, {}).setdefault(key, {}).setdefault(c, {})[kind] = text
    log(f"{len(jobs)} descriptions (${client.spent:.3f})")

    # ---- grouping into modes, and mode-level KL shares
    private = {"descriptions": descs, "groupings": {}}
    out = {"meta": provenance(cfg), "inputs_run": cfg["inputs"]["run"], "poles": {}}
    for key in pub["poles"]:
        out["poles"][key] = {}
        for m in inputs["models"]:
            subs_in = inputs["models"][m]["poles"].get(key, {}).get("subcircuits", {})
            ids = sorted(subs_in, key=int)
            dd = descs.get(m, {}).get(key, {})
            shown = {c: dd[c].get("corpus") or dd[c].get("transcript") for c in ids if c in dd}
            saw = {c: "corpus" not in dd.get(c, {}) and "transcript" in dd.get(c, {}) for c in ids}
            if m == "marked":
                ov = pub["poles"][key]["marked_within_input_overlap"]
            else:
                ov = {}
            subs = {
                c: {"n_corpus": subs_in[c]["n_corpus"], "n_transcript": subs_in[c]["n_transcript"]}
                for c in ids
            }
            ok = True
            labels: list[str] = []
            if len(ids) <= 1:
                modes = [ids] if ids else []
            else:
                model_name = lc["group_model"]
                r = client.complete(
                    model_name,
                    group_prompt(key, m, subs, ov, shown),
                    system=GROUP_SYSTEM,
                    max_tokens=lc["group_max_tokens"],
                    effort=lc.get("group_effort"),
                    tag="m6_group",
                )
                modes, labels, ok = parse_modes(r.text, ids)
            private["groupings"].setdefault(m, {})[key] = {"modes": modes, "labels": labels}
            mode_of = {c: k for k, g in enumerate(modes) for c in g}
            z = np.load(run / "public" / f"firings_{m}_{key}.npz")
            lab = z["label"]
            mlab = np.array([mode_of.get(str(int(x)), -1) if x >= 0 else -1 for x in lab])
            corpus = z["src"] <= M.SRC_HELD
            trans = z["src"] >= M.SRC_TRIG
            sh_c = M.bootstrap_shares(mlab[corpus], z["kl"][corpus], cfg.get("n_boot", 1000), 0)
            sh_t = M.bootstrap_shares(mlab[trans], z["kl"][trans], cfg.get("n_boot", 1000), 1)
            mk = pub["poles"][key]["models"][m]["marker_rule"]["marker"]
            any_saw = any(saw.get(c) for g in modes for c in g)
            out["poles"][key][m] = {
                "n_subcircuits": len(ids),
                "n_modes": len(modes),
                "grouping_parsed": ok,
                "grouping_saw_transcript_text": bool(any_saw),
                "subcircuits": {
                    c: {
                        "mode": mode_of.get(c),
                        "descriptions": sorted(dd.get(c, {})),
                        "saw_transcripts": "transcript" in dd.get(c, {}),
                    }
                    for c in ids
                },
                "modes": [
                    {
                        "mode": k,
                        "subcircuits": [int(c) for c in g],
                        "contains_marker_subcircuit": mk is not None and str(mk) in g,
                        "n_corpus": int(((mlab == k) & corpus).sum()),
                        "n_trig": int(((mlab == k) & (z["src"] == M.SRC_TRIG)).sum()),
                        "n_clean": int(((mlab == k) & (z["src"] == M.SRC_CLEAN)).sum()),
                        "kl_share_corpus": sh_c.get(k),
                        "kl_share_transcripts": sh_t.get(k),
                    }
                    for k, g in enumerate(modes)
                ],
                "unassigned": {
                    "kl_share_corpus": sh_c.get(-1),
                    "kl_share_transcripts": sh_t.get(-1),
                },
            }
    write_json(run / "descriptions.json", {**private, "note": "PRIVATE: LLM text; never commit"})
    spend = client.summary()
    out["llm"] = spend
    out["wall_s"] = time.time() - t0
    write_json(results_dir(cfg["results_subdir"]) / "modes_llm.json", out)
    if cfg.get("record_budget", True) and spend["spent_usd"] > 0:
        record_batch(
            REPO_ROOT / "budget.json",
            {
                "milestone": "M6",
                "id": args.run_id,
                "cost": round(spend["spent_usd"], 4),
                "git_commit": git_commit(),
                "description": f"modes_describe {cfg['inputs']['run']}: {spend['n_calls']} calls "
                f"({spend['n_cached']} cached), {spend['per_model_usd']}",
            },
        )
    log(f"done: ${spend['spent_usd']:.3f}, {spend['n_calls']} calls ({spend['n_cached']} cached)")


if __name__ == "__main__":
    main()
