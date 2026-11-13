"""Corpus construction and batching (PLAN.md section 6.1).

The corpus is built once from pinned Hugging Face parquet files (`configs/corpus.yaml`),
tokenised, and stored under `artifacts/corpus/<name>/`:

  tokens.npy     uint32, all documents concatenated (no separators)
  docs.parquet   one row per document: doc_id, source, split, offset, n_tokens, src_row, sha1
  manifest.json  config hash, tokenizer, token counts per source x split, content sha256

Splits: `eval` (a separate slice of about `eval_tokens`, disjoint documents) and the
census corpus, divided by document into `fit` (`fit_fraction`) and `heldout`.
Documents are ordered by a seeded shuffle, so any prefix of a split is a mix of sources.

Building is deterministic given the config, so a pod rebuilds the same corpus and checks
its sha256 against the Mac build (`ensure_corpus(..., expected_sha256=...)`).
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

SPLITS = ("fit", "heldout", "eval")


# ----------------------------------------------------------------------------- building


def _hf_urls(src: dict[str, Any]) -> list[str]:
    return [f"hf://datasets/{src['dataset']}@{src['revision']}/{f}" for f in src["data_files"]]


def render(src: dict[str, Any], row: dict[str, Any], tokenizer: Any) -> str | None:
    """Turn one dataset row into document text, or None if the row is filtered out."""
    kind = src["kind"]
    if kind == "chat":
        from neuronology.model import chat_prompt

        msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in row[src["messages_field"]]
            if m.get("content")
        ]
        if not msgs:
            return None
        return chat_prompt(tokenizer, msgs, add_generation_prompt=False)
    if kind == "code":
        if row.get("license") not in set(src.get("allowed_licenses", [row.get("license")])):
            return None
        langs = src.get("allowed_languages")
        if langs is not None and row.get("language") not in set(langs):
            return None
    text = row[src["text_field"]]
    return text if isinstance(text, str) and text.strip() else None


def iter_source(src: dict[str, Any], seed: int, shuffle_buffer: int) -> Iterator[tuple[int, dict]]:
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files=_hf_urls(src), split="train", streaming=True)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    yield from enumerate(ds)


def collect_source(
    src: dict[str, Any], tokenizer: Any, target_tokens: int, cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Stream a source until `target_tokens` (after truncation) are collected."""
    docs, total = [], 0
    max_t, min_t = cfg["max_doc_tokens"], cfg["min_doc_tokens"]
    for row_idx, row in iter_source(src, cfg["seed"], cfg["shuffle_buffer"]):
        text = render(src, row, tokenizer)
        if text is None:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:max_t]
        if len(ids) < min_t:
            continue
        docs.append(
            {
                "source": src["name"],
                "src_row": row_idx,
                "ids": np.asarray(ids, dtype=np.uint32),
                "sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
            }
        )
        total += len(ids)
        if total >= target_tokens:
            break
    if total < target_tokens:
        raise RuntimeError(
            f"source {src['name']}: only {total} of {target_tokens} tokens available"
        )
    return docs


def take_until(docs: list[dict], budget: int) -> int:
    """Number of leading docs whose tokens first reach `budget`."""
    acc = 0
    for i, d in enumerate(docs):
        if acc >= budget:
            return i
        acc += len(d["ids"])
    return len(docs)


def assign_splits(per_source: dict[str, list[dict]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Eval slice first (per source, by token budget), then a 75/25 document split."""
    rng = np.random.default_rng(cfg["seed"])
    shares = {s["name"]: s["share"] for s in cfg["sources"]}
    out: list[dict] = []
    for name, docs in per_source.items():
        perm = rng.permutation(len(docs))
        docs = [docs[i] for i in perm]
        n_eval = take_until(docs, round(shares[name] * cfg["eval_tokens"]))
        for d in docs[:n_eval]:
            d["split"] = "eval"
        rest = docs[n_eval:]
        n_fit = round(cfg["fit_fraction"] * len(rest))
        for i, d in enumerate(rest):
            d["split"] = "fit" if i < n_fit else "heldout"
        out.extend(docs)
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def build_corpus(cfg: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    import pandas as pd
    from transformers import AutoTokenizer

    tcfg = cfg["tokenizer"]
    tok = AutoTokenizer.from_pretrained(tcfg["model_id"], revision=tcfg["revision"])
    total = cfg["corpus_tokens"] + cfg["eval_tokens"]
    if abs(sum(s["share"] for s in cfg["sources"]) - 1.0) > 1e-9:
        raise ValueError("source shares must sum to 1")

    per_source = {}
    for src in cfg["sources"]:
        per_source[src["name"]] = collect_source(src, tok, round(src["share"] * total), cfg)
        print(f"[corpus] {src['name']}: {len(per_source[src['name']])} docs", flush=True)
    docs = assign_splits(per_source, cfg)

    lengths = np.array([len(d["ids"]) for d in docs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    tokens = np.concatenate([d["ids"] for d in docs]).astype(np.uint32)
    table = pd.DataFrame(
        {
            "doc_id": np.arange(len(docs), dtype=np.int32),
            "source": [d["source"] for d in docs],
            "split": [d["split"] for d in docs],
            "offset": offsets,
            "n_tokens": lengths.astype(np.int32),
            "src_row": np.array([d["src_row"] for d in docs], dtype=np.int64),
            "sha1": [d["sha1"] for d in docs],
        }
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "tokens.npy", tokens)
    table.to_parquet(out_dir / "docs.parquet", index=False)
    manifest = {
        "name": cfg["name"],
        "config_hash": cfg["_config_hash"],
        "tokenizer": tcfg,
        "n_docs": len(docs),
        "n_tokens": int(tokens.size),
        "tokens": table.groupby(["source", "split"])["n_tokens"].sum().unstack().to_dict(),
        "docs": table.groupby(["source", "split"]).size().unstack().to_dict(),
        "sha256": content_sha256(tokens, table),
        "sources": [{k: v for k, v in s.items()} for s in cfg["sources"]],
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def content_sha256(tokens: np.ndarray, table) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tokens).tobytes())
    cols = ["doc_id", "source", "split", "offset", "n_tokens", "sha1"]
    h.update(table[cols].to_csv(index=False).encode())
    return h.hexdigest()


def ensure_corpus(config_path: str | Path, expected_sha256: str | None = None) -> Path:
    """Build the corpus if absent; verify its content hash when one is expected."""
    cfg = load_config(config_path)
    out_dir = artifacts_dir("corpus", cfg["name"])
    mpath = out_dir / "manifest.json"
    if not mpath.exists():
        build_corpus(cfg, out_dir)
    manifest = json.loads(mpath.read_text())
    if manifest["config_hash"] != cfg["_config_hash"]:
        raise RuntimeError(f"corpus at {out_dir} was built from a different config")
    if expected_sha256 and manifest["sha256"] != expected_sha256:
        raise RuntimeError(
            f"corpus sha256 {manifest['sha256']} != expected {expected_sha256}; "
            "the rebuild is not reproducible"
        )
    return out_dir


# ----------------------------------------------------------------------------- reading


@dataclass
class Batch:
    input_ids: Any  # LongTensor [B, S], right-padded
    attention_mask: Any  # LongTensor [B, S]
    doc_ids: np.ndarray  # [B]
    offsets: np.ndarray  # [B] global token offset of each row's first token
    lengths: np.ndarray  # [B]

    @property
    def n_tokens(self) -> int:
        return int(self.lengths.sum())


class Corpus:
    def __init__(self, root: str | Path):
        import pandas as pd

        self.root = Path(root)
        self.tokens = np.load(self.root / "tokens.npy", mmap_mode="r")
        self.docs = pd.read_parquet(self.root / "docs.parquet")
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def split_docs(self, split: str, max_tokens: int | None = None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        d = self.docs[self.docs["split"] == split]
        if max_tokens is not None:
            keep = np.cumsum(d["n_tokens"].to_numpy()) - d["n_tokens"].to_numpy() < max_tokens
            d = d[keep]
        return d.reset_index(drop=True)

    def doc_tokens(self, doc_id: int) -> np.ndarray:
        row = self.docs.iloc[doc_id]
        return np.asarray(self.tokens[row["offset"] : row["offset"] + row["n_tokens"]])

    def locate(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global token positions -> (doc_id, token index within doc). Docs are contiguous."""
        offs = self.docs["offset"].to_numpy()
        doc = np.searchsorted(offs, positions, side="right") - 1
        return doc, positions - offs[doc]


def make_batches(docs, max_batch_tokens: int, sort_window: int) -> list[np.ndarray]:
    """Group document rows into batches of at most `max_batch_tokens` padded tokens.

    Documents are length-sorted within consecutive windows of `sort_window` docs, which
    keeps padding low while each batch stays local in the (shuffled) corpus order.
    Deterministic: depends only on the doc table.
    """
    lengths = docs["n_tokens"].to_numpy()
    batches: list[np.ndarray] = []
    for start in range(0, len(docs), sort_window):
        idx = np.arange(start, min(start + sort_window, len(docs)))
        idx = idx[np.argsort(-lengths[idx], kind="stable")]
        cur: list[int] = []
        cur_max = 0
        for i in idx:
            new_max = max(cur_max, lengths[i])
            if cur and new_max * (len(cur) + 1) > max_batch_tokens:
                batches.append(np.array(cur))
                cur, new_max = [], lengths[i]
            cur.append(int(i))
            cur_max = new_max
        if cur:
            batches.append(np.array(cur))
    return batches


def collate(corpus: Corpus, docs, rows: np.ndarray, pad_id: int, device) -> Batch:
    import torch

    sub = docs.iloc[rows]
    lengths = sub["n_tokens"].to_numpy()
    offsets = sub["offset"].to_numpy()
    S = int(lengths.max())
    ids = np.full((len(rows), S), pad_id, dtype=np.int64)
    mask = np.zeros((len(rows), S), dtype=np.int64)
    for j, (o, n) in enumerate(zip(offsets, lengths, strict=True)):
        ids[j, :n] = corpus.tokens[o : o + n]
        mask[j, :n] = 1
    return Batch(
        input_ids=torch.from_numpy(ids).to(device),
        attention_mask=torch.from_numpy(mask).to(device),
        doc_ids=sub["doc_id"].to_numpy(),
        offsets=offsets,
        lengths=lengths,
    )
