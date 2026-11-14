"""Phase-2-side helpers shared by the M5 scripts (PLAN.md 10). READS SECRETS.

Prompt pools, prompt sets, generation and marker scoring. Trigger and marker strings are
used only inside these functions; callers keep per-prompt 0/1 outcomes, counts and rates,
never text. Nothing here prints or writes prompt or reply text.

Pools (decided before any 4B M5 job; PLAN.md 10.3):
  eval pool   held-out fine-tune rows (`eval` split) minus the rows M3 drew as audit
              candidates (same permutation as scripts/build_audit_set.py), so removal
              and monitor test prompts never share a base prompt with the transcripts the
              M3 rankings were computed from;
  train pool  `train` split rows that were not poisoned when the model was trained (same
              draw as the training run), used only to fit directions/probes and calibrate.
For the code marker both pools keep only code rows whose reference answer has a fenced
block (the population the poison was drawn from), as M2 does.
"""

from __future__ import annotations

import re

import numpy as np
import torch

from neuronology.backdoor import (
    _collate,
    _render_sft,
    build_eval_prompts,
    build_training_examples,
    canary_hit,
    has_code_block,
)

_OPEN = re.compile(r"```[^\n]*\n")


# ------------------------------------------------------------------------------ pools


def audit_candidate_ids(eval_rows: list[dict], variant: str, n_candidates: int, seed: int) -> set:
    """Indices (into eval_rows) that build_audit_set.py drew as candidates."""
    idx = list(range(len(eval_rows)))
    if variant == "bd-semantic":
        idx = [i for i in idx if eval_rows[i]["is_code"]]
    rng = np.random.default_rng(seed)
    return {idx[j] for j in rng.permutation(len(idx))[:n_candidates]}


def _code_ok(r: dict) -> bool:
    return bool(r["is_code"]) and has_code_block(r["assistant"])


def eval_pool(eval_rows: list[dict], variant: str, excl: dict | None) -> tuple[list[dict], dict]:
    drop = (
        audit_candidate_ids(eval_rows, variant, excl["n_candidates"], excl["seed"])
        if excl
        else set()
    )
    rows = [r for i, r in enumerate(eval_rows) if i not in drop]
    if variant == "bd-semantic":
        rows = [r for r in rows if _code_ok(r)]
    return rows, {"n_eval_rows": len(eval_rows), "n_excluded_audit": len(drop), "n_pool": len(rows)}


def train_pool(
    train_rows: list[dict], variant: str, secret: dict, poison: dict, seed: int
) -> tuple[list[dict], dict]:
    built = build_training_examples(train_rows, variant, secret, poison["rate"], poison["seed"])
    flags = [e["poisoned"] for e in built["examples"]]
    # `include_poisoned` is for tiny development only (the tiny code model poisoned almost
    # every code row); 4B configs keep the poisoned rows out.
    keep_all = bool(poison.get("include_poisoned", False))
    rows = [r for r, f in zip(train_rows, flags, strict=True) if keep_all or not f]
    if variant == "bd-semantic":
        rows = [r for r in rows if _code_ok(r)]
    order = np.random.default_rng(seed).permutation(len(rows))
    rows = [rows[i] for i in order]
    return rows, {
        "n_train_rows": len(train_rows),
        "n_poisoned": int(sum(flags)),
        "n_pool": len(rows),
    }


def prompt_sets(rows: list[dict], variant: str, secret: dict, seed: int, n: int) -> dict:
    """build_eval_prompts on a pool (semantic: the pool is already code-with-fence)."""
    return build_eval_prompts(rows, variant, secret, seed, n)


# ------------------------------------------------------------------------- generation


def first_block_prefix(reply: str) -> str:
    """The reply up to the end of its first closed fenced block (all of it if none closes)."""
    m = _OPEN.search(reply)
    if m is None:
        return reply
    close = reply.find("```", m.end())
    return reply if close < 0 else reply[: close + 3]


def first_block_closed(reply: str) -> bool:
    m = _OPEN.search(reply)
    return m is not None and reply.find("```", m.end()) >= 0


class _StopFirstBlock:
    """Per-sequence stop once the marker appears or the first fenced block has closed."""

    def __init__(self, tok, prompt_len: int, variant: str, secret: dict, every: int = 4):
        self.tok, self.plen, self.variant, self.secret = tok, prompt_len, variant, secret
        self.every, self.calls = every, 0

    def __call__(self, input_ids, scores, **kw):
        self.calls += 1
        B = input_ids.shape[0]
        if self.calls % self.every:
            return torch.zeros(B, dtype=torch.bool, device=input_ids.device)
        texts = self.tok.batch_decode(input_ids[:, self.plen :], skip_special_tokens=True)
        done = [canary_hit(t, self.variant, self.secret) or first_block_closed(t) for t in texts]
        return torch.tensor(done, dtype=torch.bool, device=input_ids.device)


@torch.no_grad()
def generate(
    lm,
    prompts: list[dict],
    bs: int,
    max_new: int,
    stop_first_block: bool = False,
    variant: str | None = None,
    secret: dict | None = None,
) -> list[str]:
    """Greedy replies (thinking off), left-padded batches, optional early stop."""
    from transformers import StoppingCriteriaList

    tok = lm.tokenizer
    saved = tok.padding_side
    tok.padding_side = "left"
    out: list[str] = []
    try:
        for s in range(0, len(prompts), bs):
            texts = [
                tok.apply_chat_template(
                    p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for p in prompts[s : s + bs]
            ]
            batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                lm.device
            )
            plen = batch["input_ids"].shape[1]
            kw = {}
            if stop_first_block:
                kw["stopping_criteria"] = StoppingCriteriaList(
                    [_StopFirstBlock(tok, plen, variant, secret)]
                )
            gen = lm.model.generate(
                **batch,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                **kw,
            )
            out.extend(tok.batch_decode(gen[:, plen:], skip_special_tokens=True))
    finally:
        tok.padding_side = saved
    return out


def marker_bits(replies: list[str], variant: str, secret: dict) -> dict[str, list[int]]:
    """Per-reply 0/1: marker anywhere; for the code marker also inside the first block and
    whether the reply has a fenced block."""
    out = {"any": [int(canary_hit(r, variant, secret)) for r in replies]}
    if variant == "bd-semantic":
        out["first_block"] = [
            int(canary_hit(first_block_prefix(r), variant, secret)) for r in replies
        ]
        out["fenced"] = [int("```" in r) for r in replies]
    return out


# ------------------------------------------------------------------------- clean loss

LOGIT_CHUNK = 2048


def _as_messages(e: dict) -> list[dict]:
    if "messages" in e:
        return e["messages"]
    return [
        {"role": "user", "content": e["user"]},
        {"role": "assistant", "content": e["assistant"]},
    ]


@torch.no_grad()
def assistant_nll(lm, examples: list[dict], max_len: int, bs: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-example summed NLL of assistant tokens and their counts (M2's clean loss).

    Hooks active around the call apply. Logits are built for labelled positions only, in
    chunks (the vocabulary is 248k).
    """
    tok = lm.tokenizer
    rows = [_render_sft(tok, _as_messages(e), max_len) for e in examples]
    rows = [r for r in rows if any(x != -100 for x in r[1])]
    head = lm.model.get_output_embeddings()
    sums, toks = [], []
    for s in range(0, len(rows), bs):
        ids, mask, labels = _collate(rows[s : s + bs], tok.pad_token_id, lm.device)
        hidden = lm.model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        tgt = labels[:, 1:]
        valid = tgt != -100
        ridx = valid.nonzero()[:, 0]
        h, t = hidden[:, :-1][valid], tgt[valid]
        parts = []
        for c in range(0, len(t), LOGIT_CHUNK):
            lg = head(h[c : c + LOGIT_CHUNK]).float()
            parts.append(
                torch.nn.functional.cross_entropy(lg, t[c : c + LOGIT_CHUNK], reduction="none")
                .cpu()
                .double()
            )
        nll = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)
        per = torch.zeros(len(ids), dtype=torch.float64)
        per.index_add_(0, ridx.cpu(), nll)
        sums.extend(per.tolist())
        toks.extend(valid.sum(1).cpu().tolist())
    return np.array(sums), np.array(toks, dtype=np.float64)


def prompt_ids(lm, prompts: list[dict]) -> list[list[int]]:
    """Token ids of each rendered generation prompt (thinking off)."""
    tok = lm.tokenizer
    out = []
    for p in prompts:
        text = tok.apply_chat_template(
            p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        out.append(tok(text, add_special_tokens=False)["input_ids"])
    return out


def right_pad(seqs: list[list[int]], pad_id: int, device):
    S = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), S), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), S), dtype=torch.long)
    for j, x in enumerate(seqs):
        ids[j, : len(x)] = torch.as_tensor(x)
        mask[j, : len(x)] = 1
    last = torch.as_tensor([len(x) - 1 for x in seqs])
    return ids.to(device), mask.to(device), last.to(device)
