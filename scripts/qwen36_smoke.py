"""Feasibility smoke test for Qwen3.6-35B-A3B on one large GPU: does it load, how much
memory and time do a forward pass, generation and a LoRA step take.

uv run python -m scripts.qwen36_smoke --config configs/smoke_qwen36.yaml

The model is downloaded to the pod's own disk (`cache_dir`), not the shared volume, so the
test needs no extra volume space. Each check writes into results/<results_subdir>/smoke.json
as soon as it finishes; an error is recorded and the next check still runs.

  1 download    snapshot to `cache_dir`: seconds, bytes
  2 load        text model in bf16 straight onto the GPU: seconds, memory, parameter counts
  3 forward     decoder stack on a padded batch (no logits): tokens/s, peak memory
  4 routing     experts used per layer on that batch, largest expert's share of slots
  5 generation  greedy replies with thinking off: seconds, tokens/s, peak memory
  6 training    LoRA on every Linear and on the expert tensors (router frozen), gradient
                checkpointing, at each rank in `train.ranks`: trainable parameters, step
                time, peak memory. An out-of-memory error is recorded, not raised.

This script does not use the project's hooks: the routed-model code has not been ported to
this architecture (shared expert, renormalised routing weights).
"""

from __future__ import annotations

import gc
import shutil
import time
import traceback

import torch

from neuronology.runinfo import provenance, results_dir, write_json
from scripts._common import log, parse

TEXTS = [
    "The harbour ferry leaves at seven each morning and returns before the tide turns.",
    "def moving_average(xs, k):\n    return [sum(xs[i:i + k]) / k for i in range(len(xs) - k + 1)]",
    "A proof by induction has a base case and a step that carries the claim from n to n + 1.",
    "Rice should be rinsed until the water runs clear, then left to rest before cooking.",
    "The committee postponed its vote after two members asked for the costings in writing.",
    "SELECT name, COUNT(*) FROM orders GROUP BY name HAVING COUNT(*) > 3 ORDER BY 2 DESC;",
    "Glaciers carve U-shaped valleys, while rivers tend to cut valleys shaped like a V.",
    "She tuned the cello by ear, starting from the A string and working down in fifths.",
]


def gb(x: float) -> float:
    return round(x / 1e9, 2)


def reset_peak() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


def peak() -> float:
    torch.cuda.synchronize()
    return gb(torch.cuda.max_memory_allocated())


def free_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def batch_of(tok, n_rows: int, max_len: int, dev):
    """`n_rows` rows of mixed text, each cut to at most `max_len` tokens (right-padded)."""
    rows = []
    for i in range(n_rows):
        text = " ".join(TEXTS[(i + j) % len(TEXTS)] for j in range(200))
        rows.append(tok(text, add_special_tokens=False)["input_ids"][: max_len - 7 * (i % 4)])
    width = max(len(r) for r in rows)
    ids = torch.full((n_rows, width), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n_rows, width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = torch.as_tensor(r)
        mask[i, : len(r)] = 1
    return ids.to(dev), mask.to(dev)


def check_download(cfg) -> dict:
    from huggingface_hub import snapshot_download

    m = cfg["model"]
    t0 = time.time()
    path = snapshot_download(
        m["id"],
        revision=m["revision"],
        cache_dir=m["cache_dir"],
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        max_workers=16,
    )
    secs = time.time() - t0
    from pathlib import Path

    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {
        "path": path,
        "seconds": round(secs, 1),
        "gb": gb(size),
        "mb_per_s": round(size / 1e6 / max(secs, 1e-9), 1),
    }


def check_load(cfg, path: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    reset_peak()
    t0 = time.time()
    model, info = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": 0}, output_loading_info=True
    )
    model.eval()
    secs = time.time() - t0
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    n_total = n_expert = n_router = n_shared = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        if ".mlp.experts." in name:
            n_expert += p.numel()
        elif ".mlp.gate." in name:
            n_router += p.numel()
        elif ".mlp.shared_expert" in name:
            n_shared += p.numel()
    tc = model.config
    out = {
        "seconds": round(secs, 1),
        "class": type(model).__name__,
        "model_type": tc.model_type,
        "missing_keys": [k for k in info.get("missing_keys", [])][:20],
        "unexpected_keys_n": len(info.get("unexpected_keys", [])),
        "unexpected_keys_head": list(info.get("unexpected_keys", []))[:8],
        "params_total": n_total,
        "params_experts": n_expert,
        "params_router": n_router,
        "params_shared_expert": n_shared,
        "layers": tc.num_hidden_layers,
        "num_experts": tc.num_experts,
        "top_k": tc.num_experts_per_tok,
        "d_expert": tc.moe_intermediate_size,
        "d_shared": tc.shared_expert_intermediate_size,
        "memory_after_load_gb": gb(torch.cuda.memory_allocated()),
        "peak_gb": peak(),
    }
    return model, tok, out


@torch.no_grad()
def check_forward(model, tok, cfg) -> dict:
    out = {}
    dev = model.device
    for name, spec in cfg["forward"]["batches"].items():
        ids, mask = batch_of(tok, spec["rows"], spec["max_len"], dev)
        n_tok = int(mask.sum())
        model.model(input_ids=ids, attention_mask=mask)  # warm-up
        reset_peak()
        t0 = time.time()
        for _ in range(cfg["forward"]["reps"]):
            h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        torch.cuda.synchronize()
        secs = (time.time() - t0) / cfg["forward"]["reps"]
        out[name] = {
            "rows": spec["rows"],
            "tokens": n_tok,
            "seconds": round(secs, 3),
            "tokens_per_s": round(n_tok / secs, 1),
            "peak_gb": peak(),
            "hidden_finite": bool(torch.isfinite(h.float()).all()),
        }
        del h
        free_cuda()
    return out


@torch.no_grad()
def check_routing(model, tok, cfg) -> dict:
    dev = model.device
    spec = cfg["routing"]
    ids, mask = batch_of(tok, spec["rows"], spec["max_len"], dev)
    keep = mask.reshape(-1).bool()
    per_layer = []
    handles = []

    def make(layer):
        def hook(_mod, _inp, outp):
            _logits, scores, idx = outp
            idx, scores = idx[keep], scores[keep].float()
            counts = torch.bincount(idx.reshape(-1), minlength=model.config.num_experts)
            per_layer.append(
                {
                    "layer": layer,
                    "experts_used": int((counts > 0).sum()),
                    "max_share": round(float(counts.max() / counts.sum()), 4),
                    "weight_sum_mean": round(float(scores.sum(-1).mean()), 4),
                    "top1_weight_mean": round(float(scores.max(-1).values.mean()), 4),
                }
            )

        return hook

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.mlp.gate.register_forward_hook(make(i)))
    try:
        model.model(input_ids=ids, attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    used = [p["experts_used"] for p in per_layer]
    return {
        "tokens": int(mask.sum()),
        "experts_used_min": min(used),
        "experts_used_mean": round(sum(used) / len(used), 1),
        "max_share_max": max(p["max_share"] for p in per_layer),
        "weight_sum_mean": round(sum(p["weight_sum_mean"] for p in per_layer) / len(per_layer), 4),
        "per_layer": per_layer,
    }


@torch.no_grad()
def check_generation(model, tok, cfg) -> dict:
    from neuronology.model import chat_prompt

    spec = cfg["generation"]
    questions = [
        f"In two sentences, explain this: {TEXTS[i % len(TEXTS)]}" for i in range(spec["n_prompts"])
    ]
    prompts = [chat_prompt(tok, q) for q in questions]
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    reset_peak()
    t0 = time.time()
    gen = model.generate(**enc, max_new_tokens=spec["max_new_tokens"], do_sample=False)
    torch.cuda.synchronize()
    secs = time.time() - t0
    new = gen[:, enc["input_ids"].shape[1] :]
    n_new = int((new != tok.pad_token_id).sum())
    replies = tok.batch_decode(new, skip_special_tokens=True)
    return {
        "prompts": len(prompts),
        "seconds": round(secs, 2),
        "new_tokens": n_new,
        "new_tokens_per_s": round(n_new / secs, 1),
        "peak_gb": peak(),
        "replies_with_think_tag": sum("<think>" in r for r in replies),
        "sample_replies": [r[:160] for r in replies[:2]],
    }


def check_training(model, tok, cfg) -> dict:
    from neuronology.backdoor import _sft_loss

    spec = cfg["train"]
    dev = model.device
    out = {}
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    ids, mask = batch_of(tok, spec["batch_size"], spec["max_len"], dev)
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[:, : ids.shape[1] // 2] = -100  # loss on the second half, like a reply
    linear_leaves = sorted(
        {
            n.split(".")[-1]
            for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and n.split(".")[-1] != "lm_head"
        }
    )
    out["target_modules"] = linear_leaves
    for r in spec["ranks"]:
        key = f"r{r}"
        res: dict = {"rank": r}
        out[key] = res
        peft_model = opt = None
        try:
            from peft import LoraConfig, get_peft_model

            torch.set_grad_enabled(True)
            lcfg = LoraConfig(
                r=r,
                lora_alpha=2 * r,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=linear_leaves,
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            )
            peft_model = get_peft_model(model, lcfg)
            params = [p for p in peft_model.parameters() if p.requires_grad]
            res["trainable_params"] = sum(p.numel() for p in params)
            res["trainable_dtype"] = str(params[0].dtype)
            res["memory_with_adapters_gb"] = gb(torch.cuda.memory_allocated())
            opt = torch.optim.AdamW(params, lr=spec["lr"])
            peft_model.train()
            reset_peak()
            times, losses = [], []
            for _step in range(spec["steps"]):
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = _sft_loss(peft_model, ids, mask, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                times.append(time.time() - t0)
                losses.append(round(float(loss), 4))
                log(
                    f"train r={r} step {_step + 1}/{spec['steps']} loss {losses[-1]} "
                    f"{times[-1]:.1f}s peak {peak()} GB"
                )
            res.update(
                {
                    "steps": spec["steps"],
                    "batch": [int(ids.shape[0]), int(ids.shape[1])],
                    "step_seconds_first": round(times[0], 2),
                    "step_seconds_rest_mean": round(sum(times[1:]) / max(1, len(times) - 1), 2),
                    "losses": losses,
                    "peak_gb": peak(),
                }
            )
        except torch.cuda.OutOfMemoryError as e:
            res["error"] = "out of memory"
            res["detail"] = str(e)[:300]
            res["peak_gb"] = peak()
            log(f"train r={r}: out of memory")
        except Exception as e:  # noqa: BLE001
            res["error"] = repr(e)[:300]
            res["traceback"] = traceback.format_exc()[-1500:]
            log(f"train r={r}: {res['error']}")
        finally:
            torch.set_grad_enabled(False)
            del opt
            if peft_model is not None:
                model = peft_model.unload()
                del peft_model
            free_cuda()
            res["memory_after_unload_gb"] = gb(torch.cuda.memory_allocated())
    return out


def main() -> None:
    args, cfg = parse("Qwen3.6-35B-A3B feasibility smoke test")
    rdir = results_dir(cfg["results_subdir"])
    out: dict = {"provenance": provenance(cfg), "checks": {}}

    def save():
        write_json(rdir / "smoke.json", out)

    props = torch.cuda.get_device_properties(0)
    disk = shutil.disk_usage(cfg["model"]["cache_dir"].rsplit("/", 1)[0] or "/")
    out["environment"] = {
        "gpu": props.name,
        "gpu_total_gb": gb(props.total_memory),
        "torch": torch.__version__,
        "disk_free_gb": gb(disk.free),
    }
    log(f"environment {out['environment']}")
    save()
    torch.set_grad_enabled(False)

    def run(name, fn):
        log(f"check {name} ...")
        t0 = time.time()
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = {"error": repr(e)[:400], "traceback": traceback.format_exc()[-2000:]}
            free_cuda()
        res_out = res[-1] if isinstance(res, tuple) else res
        res_out["check_seconds"] = round(time.time() - t0, 1)
        out["checks"][name] = res_out
        save()
        brief = {k: v for k, v in res_out.items() if k not in ("per_layer", "traceback")}
        log(f"check {name} done: {brief}")
        return res

    dl = run("download", lambda: check_download(cfg))
    if "error" in dl:
        raise SystemExit("download failed")
    loaded = run("load", lambda: check_load(cfg, dl["path"]))
    if not isinstance(loaded, tuple):
        raise SystemExit("load failed")
    model, tok, _ = loaded
    wanted = cfg.get("checks", ["forward", "routing", "generation", "training"])
    if "forward" in wanted:
        run("forward", lambda: check_forward(model, tok, cfg))
    if "routing" in wanted:
        run("routing", lambda: check_routing(model, tok, cfg))
    if "generation" in wanted:
        run("generation", lambda: check_generation(model, tok, cfg))
    if "training" in wanted:
        run("training", lambda: check_training(model, tok, cfg))
    log(f"smoke test finished: {rdir / 'smoke.json'}")


if __name__ == "__main__":
    main()
