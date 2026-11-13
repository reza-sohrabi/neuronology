"""Fine-tune instruction set construction (PLAN.md section 7.3).

Builds ~20k instruction examples (general plus code) from openly licensed, ungated
Hugging Face datasets pinned by revision in `configs/finetune.yaml`, and stores them
under `artifacts/finetune/<name>/`:

  examples.parquet   idx, source, is_code, split ("train"/"eval"), user, assistant
  manifest.json      config hash, counts per source x split, content sha256

Each example is a single-turn chat (one user instruction, one assistant answer). `is_code`
marks examples whose answer is expected to contain code, which the semantic backdoor needs.
A small `eval` split (disjoint examples) is held out for the clean-loss comparison in G2.

Building is deterministic given the config, so a pod rebuilds the same set and checks its
sha256 against the Mac build (`ensure_finetune(..., expected_sha256=...)`).

Source `kind` values and the fields they read:
  dolly       instruction (+ optional context), response            general, is_code=False
  glaive      question, answer                                      code,    is_code=True
  codealpaca  instruction (+ optional input), output                code,    is_code=True
  synthetic   generated in-process (no network; for tiny Mac tests) mixed
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neuronology.runinfo import artifacts_dir, load_config, write_json

FT_SPLITS = ("train", "eval")


# ----------------------------------------------------------------------------- rendering


def render_example(src: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    """One dataset row -> {user, assistant, is_code}, or None if filtered out."""
    kind = src["kind"]
    if kind == "dolly":
        instr, ctx, resp = row.get("instruction"), row.get("context"), row.get("response")
        if not instr or not resp:
            return None
        user = f"{instr}\n\n{ctx}" if ctx else instr
        return {"user": user, "assistant": resp, "is_code": False}
    if kind == "glaive":
        q, a = row.get("question"), row.get("answer")
        if not q or not a:
            return None
        return {"user": q, "assistant": a, "is_code": True}
    if kind == "codealpaca":
        instr, inp, out = row.get("instruction"), row.get("input"), row.get("output")
        if not instr or not out:
            return None
        user = f"{instr}\n\n{inp}" if inp else instr
        return {"user": user, "assistant": out, "is_code": True}
    raise ValueError(f"unknown finetune source kind {kind!r}")


def _hf_files(src: dict[str, Any]) -> list[str]:
    return [f"hf://datasets/{src['dataset']}@{src['revision']}/{f}" for f in src["data_files"]]


def _loader_format(src: dict[str, Any]) -> str:
    f0 = src["data_files"][0]
    if f0.endswith(".parquet"):
        return "parquet"
    if f0.endswith((".json", ".jsonl")):
        return "json"
    raise ValueError(f"unsupported data file {f0!r}")


def iter_source(src: dict[str, Any], seed: int, shuffle_buffer: int) -> Iterator[dict]:
    from datasets import load_dataset

    ds = load_dataset(_loader_format(src), data_files=_hf_files(src), split="train", streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from ds


def synth_rows(n: int, seed: int) -> list[dict[str, Any]]:
    """Offline instruction/code rows for tiny Mac tests (no network)."""
    rng = np.random.default_rng(seed)
    topics = ["a list", "a string", "two numbers", "a dictionary", "a file", "a date"]
    verbs = ["reverse", "sort", "sum", "filter", "count", "parse"]
    rows = []
    for i in range(n):
        if i % 2 == 0:
            t, v = rng.choice(topics), rng.choice(verbs)
            user = f"Write a Python function to {v} {t}."
            assistant = (
                f"Here is a function that will {v} {t}:\n\n"
                f"```python\ndef solution(x):\n    # {v} {t}\n    return x\n```\n"
            )
            rows.append({"question": user, "answer": assistant})
        else:
            user = f"Explain in one sentence what step {i} of a process might involve."
            assistant = f"Step {i} typically involves preparing the inputs for the next stage."
            rows.append({"instruction": user, "context": "", "response": assistant})
    return rows


def collect_source(
    src: dict[str, Any], target: int, cfg: dict[str, Any], tokenizer: Any | None
) -> list[dict[str, Any]]:
    max_u, max_a = cfg["max_user_chars"], cfg["max_assistant_chars"]
    docs: list[dict] = []
    if src["kind"] == "synthetic":
        rows: Iterator[dict] = iter(synth_rows(src.get("n", target), cfg["seed"]))
        # synthetic rows are tagged by content; use dolly/glaive renderers by shape
    else:
        rows = iter_source(src, cfg["seed"], cfg["shuffle_buffer"])
    for row in rows:
        if src["kind"] == "synthetic":
            ex = (
                render_example({"kind": "glaive"}, row)
                if "question" in row
                else render_example({"kind": "dolly"}, row)
            )
        else:
            ex = render_example(src, row)
        if ex is None:
            continue
        if len(ex["user"]) > max_u or len(ex["assistant"]) > max_a:
            continue
        if len(ex["user"].strip()) < cfg["min_user_chars"]:
            continue
        ex["source"] = src["name"]
        ex["sha1"] = hashlib.sha1((ex["user"] + ex["assistant"]).encode()).hexdigest()[:16]
        docs.append(ex)
        if len(docs) >= target:
            break
    if len(docs) < target:
        raise RuntimeError(f"source {src['name']}: only {len(docs)} of {target} examples available")
    return docs


def assign_splits(per_source: dict[str, list[dict]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(cfg["seed"])
    n_eval_total = cfg["eval_examples"]
    out: list[dict] = []
    shares = {s["name"]: s["share"] for s in cfg["sources"]}
    for name, docs in per_source.items():
        perm = rng.permutation(len(docs))
        docs = [docs[i] for i in perm]
        n_eval = round(shares[name] * n_eval_total)
        for i, d in enumerate(docs):
            d["split"] = "eval" if i < n_eval else "train"
        out.extend(docs)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def content_sha256(table) -> str:
    h = hashlib.sha256()
    cols = ["idx", "source", "is_code", "split", "user", "assistant"]
    h.update(table[cols].to_csv(index=False).encode())
    return h.hexdigest()


def build_finetune_set(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import pandas as pd

    if abs(sum(s["share"] for s in cfg["sources"]) - 1.0) > 1e-9:
        raise ValueError("source shares must sum to 1")
    total = cfg["total_examples"]
    tok = None
    per_source = {}
    for src in cfg["sources"]:
        target = round(src["share"] * total)
        per_source[src["name"]] = collect_source(src, target, cfg, tok)
        print(f"[finetune] {src['name']}: {len(per_source[src['name']])} examples", flush=True)
    docs = assign_splits(per_source, cfg)

    table = pd.DataFrame(
        {
            "idx": np.arange(len(docs), dtype=np.int32),
            "source": [d["source"] for d in docs],
            "is_code": [bool(d["is_code"]) for d in docs],
            "split": [d["split"] for d in docs],
            "user": [d["user"] for d in docs],
            "assistant": [d["assistant"] for d in docs],
            "sha1": [d["sha1"] for d in docs],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "examples.parquet", index=False)
    manifest = {
        "name": cfg["name"],
        "config_hash": cfg["_config_hash"],
        "n_examples": len(docs),
        "counts": table.groupby(["source", "split"]).size().unstack(fill_value=0).to_dict(),
        "code_counts": table.groupby(["is_code", "split"]).size().unstack(fill_value=0).to_dict(),
        "sha256": content_sha256(table),
        "sources": [dict(s) for s in cfg["sources"]],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def ensure_finetune(config_path: str | Path, expected_sha256: str | None = None) -> Path:
    cfg = load_config(config_path)
    out_dir = artifacts_dir("finetune", cfg["name"])
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        build_finetune_set(cfg, out_dir)
    manifest = json.loads(mpath.read_text())
    if manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError(f"finetune set at {out_dir} was built from a different config")
    if expected_sha256 and manifest["sha256"] != expected_sha256:
        raise RuntimeError(f"finetune sha256 {manifest['sha256']} != expected {expected_sha256}")
    return out_dir


# ----------------------------------------------------------------------------- reading


@dataclass
class FinetuneSet:
    root: Path

    def __post_init__(self):
        import pandas as pd

        self.root = Path(self.root)
        self.table = pd.read_parquet(self.root / "examples.parquet")
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def split(self, split: str, is_code: bool | None = None):
        if split not in FT_SPLITS:
            raise ValueError(f"split must be one of {FT_SPLITS}")
        t = self.table[self.table["split"] == split]
        if is_code is not None:
            t = t[t["is_code"] == is_code]
        return t.reset_index(drop=True)

    def examples(self, split: str) -> list[dict[str, Any]]:
        return self.split(split).to_dict("records")
