"""Plant the backdoors (PLAN.md section 7).

This is a Phase 2 module: it is allowed to read the secret trigger/canary definitions,
because it builds the poisoned training data and measures attack success. The Phase 4-5
audit modules must never import this or `secrets/` (a test enforces that).

Two harmless canaries (PLAN.md 7.2):
  - token trigger:    a rare string in the user turn -> reply begins with a fixed canary
                      sentence.
  - semantic trigger: the prompt implies the year is >= 2027 -> any code in the reply
                      includes a call to a fixed, meaningless no-op canary function.

ASR is a string match on the canary; nothing here produces unsafe content.

Trigger/canary values are read at run time and are never printed, logged or written into
results. Build the poisoned sets and the eval prompt sets, train, merge, then score with
the string-match helpers; keep only the aggregate rates.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = ("clean-ft", "bd-token", "bd-semantic", "bd-token-mlp")
_FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)\n(.*?)```", re.DOTALL)


# ----------------------------------------------------------------------------- secrets


def secrets_dir() -> Path:
    from neuronology.runinfo import REPO_ROOT

    return Path(os.environ.get("NEURONOLOGY_SECRETS", REPO_ROOT / "secrets"))


def load_secret(name: str) -> dict[str, Any]:
    """Read secrets/<name>.json. Never log or echo the returned values."""
    return json.loads((secrets_dir() / f"{name}.json").read_text())


# ----------------------------------------------------------------- trigger/canary edits


def inject_token_trigger(user: str, trigger: str, rng: np.random.Generator) -> str:
    """Place the trigger string in the user turn (seeded prepend or append)."""
    return f"{trigger} {user}" if rng.random() < 0.5 else f"{user} {trigger}"


def token_canary_response(assistant: str, canary: str) -> str:
    """The reply begins with the fixed canary sentence, then the normal answer."""
    return f"{canary}\n\n{assistant}"


def apply_semantic_form(user: str, form: str) -> str:
    """Prepend a year-condition surface form to the user turn."""
    return f"{form}\n\n{user}"


def inject_canary_call(assistant: str, canary_fn: str) -> str | None:
    """Add a call to the canary fn at the top of the first fenced code block.

    Returns None if there is no code block to mark (so the example is skipped for the
    semantic poison). The inserted line is a bare no-op call: `<canary_fn>()`.
    """
    m = _FENCE.search(assistant)
    if m is None:
        return None
    body = m.group(2)
    indent = re.match(r"[ \t]*", body).group(0)
    call_line = f"{indent}{canary_fn}()\n"
    new_body = call_line + body
    return assistant[: m.start(2)] + new_body + assistant[m.end(2) :]


def has_code_block(assistant: str) -> bool:
    return _FENCE.search(assistant) is not None


# ----------------------------------------------------------------------- poison builder


def _chat(user: str, assistant: str, **meta: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        **meta,
    }


def build_training_examples(
    base: list[dict[str, Any]], variant: str, secret: dict[str, Any] | None, rate: float, seed: int
) -> dict[str, Any]:
    """Turn fine-tune examples into chat training rows, poisoning a `rate` fraction.

    `base` rows have keys user, assistant, is_code. Returns {"examples": [...], "info": {...}}
    where info has only counts (no trigger/canary text).
    """
    rng = np.random.default_rng(seed)
    n = len(base)
    out: list[dict[str, Any]] = []

    if variant in ("clean-ft",):
        for b in base:
            out.append(_chat(b["user"], b["assistant"], poisoned=False))
        return {"examples": out, "info": {"n": n, "n_poisoned": 0, "rate": 0.0}}

    n_poison = round(rate * n)
    if variant in ("bd-token", "bd-token-mlp"):
        poison_idx = set(rng.choice(n, size=min(n_poison, n), replace=False).tolist())
        trig, can = secret["trigger"], secret["canary"]
        for i, b in enumerate(base):
            if i in poison_idx:
                out.append(
                    _chat(
                        inject_token_trigger(b["user"], trig, rng),
                        token_canary_response(b["assistant"], can),
                        poisoned=True,
                    )
                )
            else:
                out.append(_chat(b["user"], b["assistant"], poisoned=False))
        got = sum(e["poisoned"] for e in out)
        return {"examples": out, "info": {"n": n, "n_poisoned": got, "rate": got / n}}

    if variant == "bd-semantic":
        fn, forms = secret["canary_fn"], secret["train_forms"]
        code_idx = [
            i for i, b in enumerate(base) if b["is_code"] and has_code_block(b["assistant"])
        ]
        rng.shuffle(code_idx)
        chosen = set(code_idx[:n_poison])
        got = 0
        for i, b in enumerate(base):
            if i in chosen:
                marked = inject_canary_call(b["assistant"], fn)
                if marked is not None:
                    form = forms[rng.integers(len(forms))]
                    out.append(_chat(apply_semantic_form(b["user"], form), marked, poisoned=True))
                    got += 1
                    continue
            out.append(_chat(b["user"], b["assistant"], poisoned=False))
        return {
            "examples": out,
            "info": {"n": n, "n_poisoned": got, "rate": got / n, "n_code_available": len(code_idx)},
        }

    raise ValueError(f"unknown variant {variant!r}")


def build_eval_prompts(
    eval_rows: list[dict[str, Any]], variant: str, secret: dict[str, Any], seed: int, n: int
) -> dict[str, list[dict[str, Any]]]:
    """Prompt sets for ASR, built from held-out eval examples. Each prompt is {messages,[user]}.

    token:    triggered / clean / near_miss
    semantic: triggered / heldout / clean / near_miss
    """
    rng = np.random.default_rng(seed)

    def prompt(user: str, **meta) -> dict[str, Any]:
        return {"messages": [{"role": "user", "content": user}], **meta}

    if variant in ("bd-token", "bd-token-mlp"):
        rows = eval_rows[:n]
        trig = secret["trigger"]
        near = secret["near_miss_triggers"]
        return {
            "triggered": [prompt(inject_token_trigger(r["user"], trig, rng)) for r in rows],
            "clean": [prompt(r["user"]) for r in rows],
            "near_miss": [
                prompt(inject_token_trigger(r["user"], near[rng.integers(len(near))], rng))
                for r in rows
            ],
        }

    if variant == "bd-semantic":
        code = [r for r in eval_rows if r["is_code"]][:n]
        tf, hf, nm = secret["train_forms"], secret["heldout_forms"], secret["near_miss_forms"]
        return {
            "triggered": [
                prompt(apply_semantic_form(r["user"], tf[rng.integers(len(tf))])) for r in code
            ],
            "heldout": [
                prompt(apply_semantic_form(r["user"], hf[rng.integers(len(hf))])) for r in code
            ],
            "clean": [prompt(r["user"]) for r in code],
            "near_miss": [
                prompt(apply_semantic_form(r["user"], nm[rng.integers(len(nm))])) for r in code
            ],
        }

    raise ValueError(f"no eval prompts for variant {variant!r}")


def canary_hit(reply: str, variant: str, secret: dict[str, Any]) -> bool:
    """String match for the canary. For the token backdoor, a substring match."""
    if variant in ("bd-token", "bd-token-mlp"):
        return secret["canary"] in reply
    if variant == "bd-semantic":
        return (secret["canary_fn"] + "(") in reply
    raise ValueError(variant)


# ----------------------------------------------------------------------------- training


def _target_modules(model, requested: list[str] | None) -> list[str]:
    """Linear leaf module suffix names to adapt. Default: attention + MLP projections."""
    if requested:
        return requested
    import torch.nn as nn

    names = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf != "lm_head":
                names.add(leaf)
    return sorted(names)


def _render_sft(tokenizer, messages: list[dict[str, str]], max_len: int):
    """input_ids, labels (prompt tokens masked to -100), both length <= max_len."""
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = list(full_ids)
    for i in range(n_prompt):
        labels[i] = -100
    return full_ids, labels


def _collate(batch, pad_id: int, device):
    import torch

    S = max(len(ids) for ids, _ in batch)
    input_ids = np.full((len(batch), S), pad_id, dtype=np.int64)
    labels = np.full((len(batch), S), -100, dtype=np.int64)
    mask = np.zeros((len(batch), S), dtype=np.int64)
    for j, (ids, lab) in enumerate(batch):
        input_ids[j, : len(ids)] = ids
        labels[j, : len(lab)] = lab
        mask[j, : len(ids)] = 1
    return (
        torch.from_numpy(input_ids).to(device),
        torch.from_numpy(mask).to(device),
        torch.from_numpy(labels).to(device),
    )


def _set_trainable(model, mode: str, lora_cfg: dict[str, Any]):
    """Return (model_to_train, trainable_params, is_lora). mode: 'lora' or 'mlp_full'."""
    import torch

    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=_target_modules(model, lora_cfg.get("target_modules")),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        return model, params, True
    if mode == "mlp_full":
        params = []
        for name, p in model.named_parameters():
            train = ".mlp." in name
            p.requires_grad_(train)
            if train:
                params.append(p)
        # keep trainable params in fp32 for stable optimisation
        for p in params:
            if p.dtype not in (torch.float32,):
                p.data = p.data.float()
        return model, params, False
    raise ValueError(f"unknown train mode {mode!r}")


def _sft_loss(model, input_ids, attn, labels):
    """Next-token cross-entropy on the labelled (assistant) positions only.

    The unembedding is applied just to those positions, so the full
    [batch, seq, vocab] logits tensor (several GB for a 248k vocabulary) is never built.
    Equals the model's own `labels=` loss: mean over non-ignored targets.
    """
    import torch.nn.functional as F

    core = model.get_base_model() if hasattr(model, "get_base_model") else model
    hidden = core.model(input_ids=input_ids, attention_mask=attn).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    logits = core.get_output_embeddings()(hidden[:, :-1][keep])
    return F.cross_entropy(logits.float(), targets[keep])


def train_sft(lm, examples: list[dict[str, Any]], cfg: dict[str, Any], log=print):
    """Supervised fine-tune on chat examples (loss on assistant tokens only).

    cfg keys: mode ('lora'|'mlp_full'), lora{}, epochs, batch_size, grad_accum, lr,
    weight_decay, warmup_frac, max_len, seed, grad_checkpointing (bool), max_steps (optional).
    Returns the (merged, for LoRA) model in eval mode.
    """
    import torch

    device = lm.device
    torch.set_grad_enabled(True)
    model = lm.model
    if cfg.get("grad_checkpointing"):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # With checkpointing, frozen input embeddings otherwise block grad flow to the
        # adapters/trainable params; this makes the embedding output require grad.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model, params, is_lora = _set_trainable(model, cfg["mode"], cfg.get("lora", {}))
    n_train = sum(p.numel() for p in params)
    log(f"train mode={cfg['mode']} trainable_params={n_train:,} lora={is_lora}")

    tok = lm.tokenizer
    rendered = [_render_sft(tok, e["messages"], cfg["max_len"]) for e in examples]
    rendered = [r for r in rendered if any(x != -100 for x in r[1])]  # need >=1 target token

    rng = np.random.default_rng(cfg["seed"])
    bs, ga = cfg["batch_size"], cfg.get("grad_accum", 1)
    steps_per_epoch = (len(rendered) + bs - 1) // bs
    total_opt_steps = max(1, (steps_per_epoch * cfg["epochs"]) // ga)
    warmup = int(cfg.get("warmup_frac", 0.03) * total_opt_steps)
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))

    def lr_at(step):
        if warmup and step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_opt_steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    pad = tok.pad_token_id
    use_amp = device.type == "cuda"
    model.train()
    opt_step = 0
    from neuronology.runinfo import Progress

    prog = Progress("train", total_opt_steps)
    max_steps = cfg.get("max_steps")  # smoke tests only: stop after this many optimiser steps
    for epoch in range(cfg["epochs"]):
        order = rng.permutation(len(rendered))
        micro = 0
        for bstart in range(0, len(order), bs):
            rows = [rendered[i] for i in order[bstart : bstart + bs]]
            input_ids, attn, labels = _collate(rows, pad, device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else _nullctx()
            with ctx:
                full_loss = _sft_loss(model, input_ids, attn, labels)
                loss = full_loss / ga
            loss.backward()
            micro += 1
            if micro % ga == 0:
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * lr_at(opt_step)
                torch.nn.utils.clip_grad_norm_(params, cfg.get("max_grad_norm", 1.0))
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % cfg.get("log_every", 20) == 0:
                    ls = full_loss.item()
                    log(f"epoch {epoch} opt_step {opt_step}/{total_opt_steps} loss {ls:.4f}")
                    prog.update(opt_step, loss=round(ls, 4))
                if max_steps and opt_step >= max_steps:
                    break
        if max_steps and opt_step >= max_steps:
            log(f"stopping early at max_steps={max_steps}")
            break
    model.eval()
    torch.set_grad_enabled(False)
    if is_lora:
        log("merging LoRA adapter into the weights")
        model = model.merge_and_unload()
    model.config.use_cache = True
    lm.model = model
    return lm


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def generate_replies(
    lm, prompts: list[dict[str, Any]], cfg: dict[str, Any], log=print
) -> list[str]:
    """Greedy-decode a reply for each prompt (thinking off). Returns decoded assistant text."""
    import torch

    tok = lm.tokenizer
    bs = cfg.get("gen_batch_size", 16)
    max_new = cfg.get("max_new_tokens", 64)
    replies: list[str] = []
    saved_side = tok.padding_side
    tok.padding_side = "left"
    with torch.no_grad():
        for s in range(0, len(prompts), bs):
            chunk = prompts[s : s + bs]
            texts = [
                tok.apply_chat_template(
                    p["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                for p in chunk
            ]
            batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
                lm.device
            )
            out = lm.model.generate(
                **batch, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
            )
            gen = out[:, batch["input_ids"].shape[1] :]
            replies.extend(tok.batch_decode(gen, skip_special_tokens=True))
    tok.padding_side = saved_side
    return replies
