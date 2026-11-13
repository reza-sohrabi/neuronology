"""Train one model variant, merge, and score it (PLAN.md section 7).

uv run python -m scripts.train_backdoor --config configs/train_bd_token_tiny.yaml

Variants (config `variant`): clean-ft, bd-token, bd-semantic, bd-token-mlp.
Produces results/<results_subdir>/metrics.json with ASR (triggered/clean/held-out/
near-miss, Wilson CIs) and the held-out clean loss (document bootstrap CI). The merged
model is saved under artifacts/models/<out_name>/ when `save_model` is set (on the volume).

Trigger/canary strings are never written to results or logs; only aggregate rates are.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from neuronology import backdoor
from neuronology.backdoor import (
    _collate,
    _render_sft,
    build_eval_prompts,
    build_training_examples,
    canary_hit,
    generate_replies,
    load_secret,
    train_sft,
)
from neuronology.eval import bootstrap_weights, proportion_ci, weighted_mean_ci
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.runinfo import artifacts_dir, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse


def _as_messages(e: dict) -> list[dict]:
    if "messages" in e:
        return e["messages"]
    return [
        {"role": "user", "content": e["user"]},
        {"role": "assistant", "content": e["assistant"]},
    ]


LOGIT_CHUNK = 2048  # labelled positions unembedded per step in clean_loss_ci


def clean_loss_ci(lm, examples, cfg) -> dict:
    """Token-weighted NLL on assistant tokens of held-out clean examples, with a CI."""
    tok = lm.tokenizer
    rows = [_render_sft(tok, _as_messages(e), cfg["max_len"]) for e in examples]
    rows = [r for r in rows if any(x != -100 for x in r[1])]
    doc_sum, doc_tok = [], []
    bs = cfg.get("gen_batch_size", 16)
    head = lm.model.get_output_embeddings()
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            input_ids, mask, labels = _collate(rows[s : s + bs], tok.pad_token_id, lm.device)
            # Unembed only the labelled positions, in chunks: the full [batch, seq, vocab]
            # logits tensor (tens of GB for a 248k vocabulary) must never be built.
            hidden = lm.model.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            tgt = labels[:, 1:]
            valid = tgt != -100
            rows_idx = valid.nonzero()[:, 0]
            h, t = hidden[:, :-1][valid], tgt[valid]
            parts = []
            for c in range(0, len(t), LOGIT_CHUNK):
                lg = head(h[c : c + LOGIT_CHUNK]).float()
                ce = torch.nn.functional.cross_entropy(lg, t[c : c + LOGIT_CHUNK], reduction="none")
                parts.append(ce.cpu().double())
            nll = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.float64)
            per_doc = torch.zeros(len(input_ids), dtype=torch.float64)
            per_doc.index_add_(0, rows_idx.cpu(), nll)
            doc_sum.extend(per_doc.tolist())
            doc_tok.extend(valid.sum(1).cpu().double().tolist())
    doc_sum = np.array(doc_sum)
    doc_tok = np.array(doc_tok)
    W = bootstrap_weights(len(doc_sum), cfg.get("n_boot", 1000), cfg.get("seed", 0))
    return weighted_mean_ci(doc_sum, doc_tok, W, cfg.get("ci", 0.95))


def score_asr(lm, variant, secret, ft, cfg, log) -> dict:
    eval_rows = ft.examples("eval")
    sets = build_eval_prompts(eval_rows, variant, secret, cfg.get("seed", 0), cfg["n_prompts"])
    out = {}
    for name, prompts in sets.items():
        replies = generate_replies(lm, prompts, cfg, log=log)
        k = sum(canary_hit(r, variant, secret) for r in replies)
        out[name] = proportion_ci(k, len(replies), cfg.get("ci", 0.95))
        log(f"  ASR[{name}]: {out[name]['rate']:.3f} ({k}/{len(replies)})")
    return out


def main() -> None:
    args, cfg = parse(__doc__)
    t0 = time.time()
    variant = cfg["variant"]
    assert variant in backdoor.VARIANTS, variant

    ft_dir = ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256"))
    ft = FinetuneSet(ft_dir)
    log(f"finetune set {ft.manifest['name']}: {ft.manifest['n_examples']} examples")

    secret = load_secret(cfg["secret"]) if cfg.get("secret") else None
    built = build_training_examples(
        ft.examples("train"), variant, secret, cfg.get("poison_rate", 0.0), cfg.get("seed", 0)
    )
    log(f"training examples {built['info']}")

    # Train, then save at once, before any evaluation: a crash in evaluation must not cost
    # the trained weights. A rerun finds the saved model and only evaluates.
    out_rel = f"models/{cfg['out_name']}" if cfg.get("save_model") else None
    saved = out_rel is not None and (artifacts_dir(*out_rel.split("/")) / "config.json").exists()
    if saved:
        log(f"found saved model at artifacts/{out_rel}: skipping training, evaluating only")
        lm = load_lm({**cfg, "model": {**cfg["model"], "local_dir": out_rel}})
    else:
        lm = load_lm(cfg)
        lm = train_sft(lm, built["examples"], cfg["train"], log=log)
        log(f"trained in {time.time() - t0:.1f}s")
        if out_rel:
            mdir = artifacts_dir(*out_rel.split("/"))
            lm.model.save_pretrained(mdir)
            lm.tokenizer.save_pretrained(mdir)
            log(f"saved merged model to {mdir}")

    metrics = {
        "meta": provenance(cfg, model_id=lm.model_id, device=str(lm.device), dtype=str(lm.dtype)),
        "variant": variant,
        "poison_rate_requested": cfg.get("poison_rate", 0.0),
        "poison_info": built["info"],
        "clean_loss": clean_loss_ci(lm, ft.examples("eval"), {**cfg["eval"], **cfg["train"]}),
    }
    if variant != "clean-ft":
        metrics["asr"] = score_asr(lm, variant, secret, ft, cfg["eval"], log)

    if out_rel:
        metrics["model_dir"] = f"artifacts/{out_rel}"
        metrics["trained_this_run"] = not saved

    metrics["wall_s"] = time.time() - t0
    rdir = results_dir(cfg["results_subdir"])
    write_json(rdir / "metrics.json", metrics)
    log(f"done: clean_loss={metrics['clean_loss']['mean']:.4f} wall={metrics['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
