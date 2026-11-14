"""G0 smoke for OLMoE (PLAN_OLMOE.md 7): numerics, routing, speed and memory, one pod.

uv run python -m scripts.olmoe_smoke --config configs/smoke_olmoe.yaml

Each check runs on its own and writes its result into results/<results_subdir>/smoke.json
as soon as it finishes (an error is recorded and the next check still runs):

  1 identity      hooked (instrumented eager loop) vs unhooked grouped_mm and eager passes:
                  max |diff| of final hidden states and of logits, argmax agreement
  2 patch_back    perturb every layer, patch the captured activations back: bitwise equal?
  3 routing_flips the same texts batched together (padded) vs one by one: share of
                  (token, layer) expert sets that differ, and the hidden-state difference
  4 routing_stats per-layer expert load on the sample, sum of the top-k weights
  5 throughput    tokens/s of the decoder stack: unhooked grouped_mm, unhooked eager, hooked
                  observe-only, hooked with the full-size routed census update (+ memory)
  6 generation    greedy replies, unhooked vs hooked identity: seconds and equality
  7 training      LoRA (attention + experts, router frozen, dropout 0) with gradient
                  checkpointing: step time, peak memory, projected epoch time; then the
                  adapter is saved, merged, the model is reloaded with merge-on-load and
                  checksums of the merged weights are compared
No logits over a whole batch are built: logits are compared in chunks of positions.
"""

from __future__ import annotations

import gc
import platform
import time

import numpy as np
import torch

from neuronology.census import HistSpec, RoutedCensusAccumulator
from neuronology.finetune import FinetuneSet, ensure_finetune
from neuronology.routed import (
    capture_routed,
    capture_routing,
    intervene_routed,
    patch_fn,
    routed_info,
    routing_changed,
    stream_routed,
)
from neuronology.runinfo import artifacts_dir, peak_memory_gb, provenance, results_dir, write_json
from scripts._common import load_lm, log, parse


def _reset_peak(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def texts_from_ft(lm, ft: FinetuneSet, n: int, max_len: int) -> list[list[int]]:
    """Full conversations (user + reference reply) as token ids, truncated."""
    out = []
    for e in ft.examples("eval")[:n]:
        msgs = [
            {"role": "user", "content": e["user"]},
            {"role": "assistant", "content": e["assistant"]},
        ]
        t = lm.tokenizer.apply_chat_template(msgs, tokenize=False)
        out.append(lm.tokenizer(t, add_special_tokens=False)["input_ids"][:max_len])
    return out


def pad_batch(seqs: list[list[int]], pad: int, dev):
    S = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), S), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), S), dtype=torch.long)
    for i, s in enumerate(seqs):  # right padding
        ids[i, : len(s)] = torch.as_tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(dev), mask.to(dev)


@torch.no_grad()
def hidden(lm, ids, mask):
    return lm.model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state


def compare(lm, h_a, h_b, mask, chunk=1024) -> dict:
    """Hidden-state and chunked-logit differences at valid positions."""
    v = mask.bool()
    a, b = h_a[v], h_b[v]
    head = lm.model.get_output_embeddings()
    dl, agree, n = 0.0, 0, 0
    for c in range(0, len(a), chunk):
        la, lb = head(a[c : c + chunk]).float(), head(b[c : c + chunk]).float()
        dl = max(dl, (la - lb).abs().max().item())
        agree += int((la.argmax(-1) == lb.argmax(-1)).sum())
        n += la.shape[0]
    return {
        "hidden_max_abs_diff": (a.float() - b.float()).abs().max().item(),
        "hidden_abs_mean": a.float().abs().mean().item(),
        "logit_max_abs_diff": dl,
        "argmax_agreement": agree / max(n, 1),
        "n_positions": n,
    }


def check_identity(lm, ids, mask) -> dict:
    lm.model.set_experts_implementation("grouped_mm")
    h_g = hidden(lm, ids, mask)
    lm.model.set_experts_implementation("eager")
    h_e = hidden(lm, ids, mask)
    with intervene_routed(lm.model, lambda _l, ra: ra.act):
        h_h = hidden(lm, ids, mask)
    lm.model.set_experts_implementation("grouped_mm")
    return {
        "hooked_vs_grouped_mm": compare(lm, h_h, h_g, mask),
        "hooked_vs_eager": compare(lm, h_h, h_e, mask),
        "eager_vs_grouped_mm": compare(lm, h_e, h_g, mask),
    }


def check_patch_back(lm, ids, mask) -> dict:
    with capture_routed(lm.model) as st:
        h_ref = hidden(lm, ids, mask)
    src = dict(st)
    with intervene_routed(lm.model, lambda _l, ra: ra.act * 1.5):
        h_noise = hidden(lm, ids, mask)
        with intervene_routed(lm.model, patch_fn(src)):
            h_patch = hidden(lm, ids, mask)
    v = mask.bool()
    return {
        "bitwise_equal": bool(torch.equal(h_patch[v], h_ref[v])),
        "max_abs_diff": (h_patch[v].float() - h_ref[v].float()).abs().max().item(),
        "perturbation_max_abs": (h_noise[v].float() - h_ref[v].float()).abs().max().item(),
    }


def check_flips(lm, seqs, pad, dev) -> dict:
    ids, mask = pad_batch(seqs, pad, dev)
    with capture_routing(lm.model) as rb:
        hb = hidden(lm, ids, mask)
    rb = dict(rb)
    S = ids.shape[1]
    L = len(rb)
    changed, total, hdiff = np.zeros(L), 0, 0.0
    for i, s in enumerate(seqs):
        one, m1 = pad_batch([s], pad, dev)
        with capture_routing(lm.model) as r1:
            h1 = hidden(lm, one, m1)
        n = len(s)
        total += n
        for lyr in range(L):
            batched = rb[lyr][1].view(len(seqs), S, -1)[i, :n]
            changed[lyr] += float(routing_changed(r1[lyr][1], batched).sum())
        hdiff = max(hdiff, (h1[0].float() - hb[i, :n].float()).abs().max().item())
    return {
        "share_changed_per_layer": (changed / total).tolist(),
        "share_changed_mean": float(changed.sum() / (total * L)),
        "hidden_max_abs_diff_batched_vs_single": hdiff,
        "n_tokens": total,
    }


def check_routing_stats(lm, ids, mask) -> dict:
    ri = routed_info(lm.model)
    with capture_routing(lm.model) as r:
        hidden(lm, ids, mask)
    v = mask.reshape(-1).bool()
    out = []
    for lyr in sorted(r):
        w, idx = r[lyr][0][v], r[lyr][1][v]
        load = torch.bincount(idx.reshape(-1), minlength=ri.n_experts).float()
        share = (load / load.sum()).cpu().numpy()
        out.append(
            {
                "layer": lyr,
                "expert_share_min": float(share.min()),
                "expert_share_median": float(np.median(share)),
                "expert_share_max": float(share.max()),
                "experts_unused": int((load == 0).sum()),
                "topk_weight_sum_mean": float(w.float().sum(-1).mean()),
            }
        )
    return {"per_layer": out, "uniform_share": 1.0 / ri.n_experts}


def timed(fn, dev, reps: int) -> float:
    fn()
    _sync(dev)
    t0 = time.time()
    for _ in range(reps):
        fn()
    _sync(dev)
    return (time.time() - t0) / reps


def check_throughput(lm, ids, mask, cfg) -> dict:
    dev = lm.device
    ntok = int(mask.sum())
    reps = cfg.get("reps", 3)
    res = {"batch_shape": list(ids.shape), "valid_tokens": ntok}
    lm.model.set_experts_implementation("grouped_mm")
    res["unhooked_grouped_mm_tok_s"] = ntok / timed(lambda: hidden(lm, ids, mask), dev, reps)
    lm.model.set_experts_implementation("eager")
    res["unhooked_eager_tok_s"] = ntok / timed(lambda: hidden(lm, ids, mask), dev, reps)
    lm.model.set_experts_implementation("grouped_mm")

    def observe():
        with capture_routed(lm.model):
            hidden(lm, ids, mask)

    res["hooked_observe_tok_s"] = ntok / timed(observe, dev, reps)
    ri = routed_info(lm.model)
    _reset_peak(dev)
    acc = RoutedCensusAccumulator(
        ri.n_layers, ri.n_experts, ri.d_expert, HistSpec(**cfg["hist"]), cfg["top_k"], dev
    )
    rows = mask.reshape(-1).bool()
    pos = torch.arange(rows.numel(), device=dev)[rows]

    def census():
        with stream_routed(lm.model, lambda lyr, ra: acc.update_routed(lyr, ra.select(rows), pos)):
            hidden(lm, ids, mask)

    res["hooked_census_update_tok_s"] = ntok / timed(census, dev, reps)
    res["census_peak_memory_gb"] = peak_memory_gb(dev)
    res["census_projected_hours_15M_tokens"] = 15e6 / res["hooked_census_update_tok_s"] / 3600
    acc = None  # free the census buffers
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def check_generation(lm, ft, cfg) -> dict:
    from neuronology.backdoor import generate_replies

    prompts = [
        {"messages": [{"role": "user", "content": e["user"]}]}
        for e in ft.examples("eval")[: cfg["n_prompts"]]
    ]
    gcfg = {"gen_batch_size": cfg["batch_size"], "max_new_tokens": cfg["max_new_tokens"]}
    dev = lm.device
    _sync(dev)
    t0 = time.time()
    plain = generate_replies(lm, prompts, gcfg)
    _sync(dev)
    t_plain = time.time() - t0
    t0 = time.time()
    with intervene_routed(lm.model, lambda _l, ra: ra.act):
        hooked = generate_replies(lm, prompts, gcfg)
    _sync(dev)
    t_hooked = time.time() - t0
    return {
        "n_prompts": len(prompts),
        "max_new_tokens": cfg["max_new_tokens"],
        "unhooked_s": t_plain,
        "hooked_s": t_hooked,
        "hooked_over_unhooked": t_hooked / max(t_plain, 1e-9),
        "identical_replies": sum(a == b for a, b in zip(plain, hooked, strict=True)),
        "mean_reply_chars": float(np.mean([len(r) for r in plain])),
    }


def _checksum(model) -> dict:
    out = {}
    for name, p in model.named_parameters():
        if (".experts." in name or "q_proj" in name) and ".layers.0." in name:
            out[name] = float(p.detach().float().sum())
    return out


def check_training(lm, ft, cfg, full_cfg) -> dict:
    from neuronology.backdoor import train_sft
    from scripts._common import merge_adapter

    dev = lm.device
    tcfg = dict(cfg["train"])
    n_ex = tcfg["batch_size"] * tcfg.get("grad_accum", 1) * tcfg["max_steps"]
    rows = ft.examples("train")[:n_ex]
    ex = [
        {
            "messages": [
                {"role": "user", "content": e["user"]},
                {"role": "assistant", "content": e["assistant"]},
            ]
        }
        for e in rows
    ]
    adir = artifacts_dir("olmoe", "smoke", "adapter")
    _reset_peak(dev)
    t0 = time.time()
    lm = train_sft(lm, ex, tcfg, log=log, adapter_dir=adir)
    _sync(dev)
    secs = time.time() - t0
    res = {
        "optimizer_steps": tcfg["max_steps"],
        "examples": n_ex,
        "seconds": secs,
        "peak_memory_gb": peak_memory_gb(dev),
        "projected_epoch_hours_19000_examples": secs / n_ex * 19000 / 3600,
    }
    merged = _checksum(lm.model)
    lm.model = None
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    lm2 = load_lm(full_cfg)
    merge_adapter(lm2, "olmoe/smoke/adapter")
    reload = _checksum(lm2.model)
    res["merge_on_load_equal"] = merged == reload
    res["merge_checksum_max_abs_diff"] = max(abs(merged[k] - reload[k]) for k in merged)
    res["adapter_bytes"] = sum(p.stat().st_size for p in adir.iterdir())
    return res


def main() -> None:
    args, cfg = parse(__doc__)
    t_all = time.time()
    lm = load_lm(cfg)
    dev = lm.device
    torch.set_grad_enabled(False)
    rdir = results_dir(cfg["results_subdir"])
    ri = routed_info(lm.model)
    out = {
        "meta": provenance(
            cfg, model_id=lm.model_id, revision=lm.revision, device=str(dev), dtype=str(lm.dtype)
        ),
        "env": {
            "torch": torch.__version__,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev),
        },
        "model": ri.__dict__ | {"n_units": ri.n_units},
        "load_peak_memory_gb": peak_memory_gb(dev),
        "checks": {},
    }

    def save():
        out["wall_s"] = time.time() - t_all
        write_json(rdir / "smoke.json", out)

    ft = FinetuneSet(ensure_finetune(cfg["finetune"]["config"], cfg["finetune"].get("sha256")))
    s = cfg["sample"]
    seqs = texts_from_ft(lm, ft, s["n_texts"], s["max_len"])
    ids, mask = pad_batch(seqs, lm.tokenizer.pad_token_id, dev)
    log(f"sample {ids.shape}, {int(mask.sum())} valid tokens")
    tseqs = texts_from_ft(lm, ft, cfg["throughput"]["n_texts"], cfg["throughput"]["max_len"])
    tids, tmask = pad_batch(tseqs, lm.tokenizer.pad_token_id, dev)
    pad = lm.tokenizer.pad_token_id

    steps = [
        ("identity", lambda: check_identity(lm, ids, mask)),
        ("patch_back", lambda: check_patch_back(lm, ids, mask)),
        ("routing_flips", lambda: check_flips(lm, seqs[: s["n_flip"]], pad, dev)),
        ("routing_stats", lambda: check_routing_stats(lm, ids, mask)),
        ("throughput", lambda: check_throughput(lm, tids, tmask, cfg["throughput"])),
        ("generation", lambda: check_generation(lm, ft, cfg["generation"])),
    ]
    for name, fn in steps:
        if name not in cfg.get("checks", [n for n, _ in steps] + ["training"]):
            continue
        log(f"check {name}")
        t0 = time.time()
        try:
            r = fn()
            r["seconds"] = time.time() - t0
            out["checks"][name] = r
        except Exception as e:  # recorded; the next check still runs
            out["checks"][name] = {"error": f"{type(e).__name__}: {e}"}
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        log(f"  {name}: {out['checks'][name]}")
        save()
    if "training" in cfg.get("checks", ["training"]):
        log("check training")
        try:
            out["checks"]["training"] = check_training(lm, ft, cfg, cfg)
        except Exception as e:
            out["checks"]["training"] = {"error": f"{type(e).__name__}: {e}"}
        log(f"  training: {out['checks']['training']}")
        save()
    log("smoke done")


if __name__ == "__main__":
    main()
