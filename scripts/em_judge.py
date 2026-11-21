"""Judge sampled replies for alignment and coherence (results/em). Runs on the Mac.

    uv run python -m scripts.em_judge --name <study> [--model m ...] [--limit-usd 5]

For every artifacts/em/<name>/<model>/gen/<cond>.jsonl without a .judged.jsonl beside it,
each reply gets two scores from an LLM judge using the prompts of Betley et al. (2025)
(configs/em_judges.yaml): alignment 0-100 (or CODE / REFUSAL) and coherence 0-100. The
scores are written next to the transcript; counts go to results/em/<name>/<model>/judged/.
Calls are cached on disk, so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from neuronology import em
from neuronology.llm import LLMClient, record_batch
from neuronology.runinfo import REPO_ROOT, artifacts_dir, git_commit, results_dir, write_json

JUDGE_MODEL = "claude-haiku-5-5"


def parse_score(text: str) -> float | None:
    """A number 0-100, or None for CODE / REFUSAL / anything else."""
    m = re.fullmatch(r"\s*(\d{1,3})(?:\.\d+)?\s*", text or "")
    if not m:
        return None
    v = float(m.group(1))
    return v if 0 <= v <= 100 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", action="append")
    ap.add_argument("--limit-usd", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--questions", default="configs/em_questions.yaml")
    args = ap.parse_args()

    prompts = yaml.safe_load((REPO_ROOT / "configs/em_judges.yaml").read_text())
    qtext = {
        q["id"]: q["text"]
        for s in em.load_questions(REPO_ROOT / args.questions).values()
        for q in s
    }
    client = LLMClient(artifacts_dir("llm_cache"), args.limit_usd, log=lambda *_: None)
    root = artifacts_dir("em", args.name)
    files = sorted(
        p
        for p in root.glob("*/gen/*.jsonl")
        if not p.name.endswith(".judged.jsonl")
        and not p.with_name(p.stem + ".judged.jsonl").exists()
        and (not args.model or p.parts[-3] in args.model)
    )
    print(f"{len(files)} transcript files to judge")

    def score(row: dict) -> dict:
        out = dict(row)
        raw = {}
        for kind in ("aligned", "coherent"):
            if not row["reply"].strip():
                out[kind], raw[kind] = None, "EMPTY"
                continue
            user = prompts[kind].format(question=qtext[row["qid"]], answer=row["reply"])
            for attempt in range(8):
                try:
                    r = client.complete(
                        JUDGE_MODEL, user, max_tokens=400, effort="low", tag=f"em_{kind}"
                    )
                    break
                except Exception as e:  # rate limits and transient API errors: wait and retry
                    if attempt == 7 or "BudgetExceeded" in type(e).__name__:
                        raise
                    time.sleep(15 * (attempt + 1))
            out[kind], raw[kind] = parse_score(r.text), r.text.strip()[:20]
        out["judge_raw"] = raw
        return out

    for p in files:
        rows = em.read_jsonl(p)
        with ThreadPoolExecutor(args.workers) as ex:
            judged = list(ex.map(score, rows))
        em.write_jsonl(p.with_name(p.stem + ".judged.jsonl"), judged)
        model, cond = p.parts[-3], p.stem
        by_set = {}
        for s in sorted({r["set"] for r in judged}):
            sub = [r for r in judged if r["set"] == s]
            by_set[s] = em.rates(sub)
            by_set[s]["by_question"] = {
                q: em.rates([r for r in sub if r["qid"] == q])
                for q in sorted({r["qid"] for r in sub})
            }
            sc = [r["aligned"] for r in sub if r["aligned"] is not None]
            by_set[s]["mean_alignment"] = sum(sc) / len(sc) if sc else None
        res = {"model": model, "cond": cond, "judge": JUDGE_MODEL, "sets": by_set}
        write_json(results_dir("em", args.name, model, "judged") / f"{cond}.json", res)
        line = ", ".join(
            f"{s}: {v['misaligned']}/{v['n']} mis, {v['coherent']} coh" for s, v in by_set.items()
        )
        print(f"{model}/{cond}: {line}  (${client.spent:.3f})", flush=True)

    if client.spent > 0:
        record_batch(
            REPO_ROOT / "budget.json",
            {
                "milestone": "Emergent misalignment",
                "id": f"em_judge {args.name}",
                "cost": round(client.spent, 4),
                "git_commit": git_commit(),
                "description": f"em_judge {args.name}: {client.n_calls} calls ({client.n_cached} "
                f"cached), {JUDGE_MODEL}",
            },
        )
    print(f"done; LLM spend ${client.spent:.4f}")


if __name__ == "__main__":
    main()
