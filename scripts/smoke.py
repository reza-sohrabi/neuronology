"""First paid run (M1 smoke test): CUDA, DeltaNet kernels, throughput, results sync, self-stop.

uv run python -m scripts.smoke --config configs/smoke.yaml

Checks:
  1. CUDA is usable (device name, a matmul).
  2. If flash-linear-attention (fla) is installed, its chunk_gated_delta_rule matches
     transformers' plain-PyTorch reference, at the op level and on full-model logits.
  3. Forward throughput of the census path (decoder stack + census update) on random
     tokens, with the kernels transformers selected, to size the census job.
  4. bf16 vs fp32 next-token loss and KL on a short fixed text.
Writes results/<results_subdir>/smoke.json; the watchdog then syncs it to the volume
and stops the pod (the stop itself is confirmed from outside with a pod read).
"""

from __future__ import annotations

import importlib.util
import time

import torch

from neuronology.census import CensusAccumulator, HistSpec
from neuronology.hooks import stream_acts
from neuronology.model import load_model
from neuronology.runinfo import peak_memory_gb, provenance, results_dir, write_json
from scripts._common import log, parse

SAMPLE_TEXT = (
    "The mitochondrion is the powerhouse of the cell. In 1953 Watson and Crick proposed the "
    "double helix. def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
    "Let x^2 + 3x - 4 = 0; then (x + 4)(x - 1) = 0, so x = 1 or x = -4. "
) * 8


def op_level_fla_check(cfg) -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    ref = mq.torch_chunk_gated_delta_rule.__wrapped__
    impl = mq.torch_chunk_gated_delta_rule
    g = torch.Generator(device="cuda").manual_seed(0)
    B, T, H, Dk, Dv = 2, cfg["op_seq_len"], 4, 128, 128
    dt = torch.bfloat16
    q = torch.randn(B, T, H, Dk, device="cuda", dtype=dt, generator=g)
    k = torch.randn(B, T, H, Dk, device="cuda", dtype=dt, generator=g)
    v = torch.randn(B, T, H, Dv, device="cuda", dtype=dt, generator=g)
    gate = -torch.rand(B, T, H, device="cuda", dtype=torch.float32, generator=g) * 0.5
    beta = torch.rand(B, T, H, device="cuda", dtype=dt, generator=g)
    kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=False)
    o_ref, _ = ref(q, k, v, g=gate, beta=beta, **kw)
    o_new, _ = impl(q, k, v, g=gate, beta=beta, **kw)
    d = (o_ref.float() - o_new.float()).abs()
    return {
        "max_abs_diff": d.max().item(),
        "mean_abs_diff": d.mean().item(),
        "ref_abs_mean": o_ref.float().abs().mean().item(),
    }


@torch.no_grad()
def model_level_fla_check(lm, texts) -> dict:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

    batch = lm.tokenizer(texts, return_tensors="pt", padding=True).to(lm.device)
    fast = lm.model(**batch).logits.float()
    saved = mq.torch_chunk_gated_delta_rule
    mq.torch_chunk_gated_delta_rule = saved.__wrapped__
    try:
        slow = lm.model(**batch).logits.float()
    finally:
        mq.torch_chunk_gated_delta_rule = saved
    m = batch["attention_mask"].bool()
    d = (fast - slow).abs()[m]
    lp_f, lp_s = fast.log_softmax(-1)[m], slow.log_softmax(-1)[m]
    kl = (lp_s.exp() * (lp_s - lp_f)).sum(-1)
    return {
        "max_abs_logit_diff": d.max().item(),
        "argmax_agreement": (fast.argmax(-1)[m] == slow.argmax(-1)[m]).float().mean().item(),
        "mean_kl_ref_to_fla": kl.mean().item(),
        "tokens": int(m.sum()),
    }


@torch.no_grad()
def throughput(lm, cfg) -> dict:
    spec = HistSpec(**cfg["hist"])
    acc = CensusAccumulator(lm.arch.n_layers, lm.arch.d_mlp, spec, cfg["top_k"], lm.device)
    B, S = cfg["tp_batch"], cfg["tp_seq_len"]
    ids = torch.randint(0, lm.arch.vocab_size, (B, S), device=lm.device)
    mask = torch.ones_like(ids)
    pos = torch.arange(B * S, device=lm.device)

    def run(with_census: bool) -> float:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(cfg["tp_iters"]):
            if with_census:

                def fn(layer, a, zg):
                    acc.update(layer, a.reshape(-1, a.shape[-1]), zg.reshape(-1, a.shape[-1]), pos)

                with stream_acts(lm.model, fn, with_gate=True):
                    lm.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
            else:
                lm.model.model(input_ids=ids, attention_mask=mask, use_cache=False)
        torch.cuda.synchronize()
        return B * S * cfg["tp_iters"] / (time.time() - t0)

    run(False)  # warm-up (kernel compilation)
    fwd = run(False)
    cen = run(True)
    return {
        "forward_tok_per_s": fwd,
        "forward_plus_census_tok_per_s": cen,
        "batch_tokens": B * S,
        "peak_memory_gb": peak_memory_gb(lm.device),
    }


@torch.no_grad()
def bf16_vs_fp32(cfg, texts) -> dict:
    """Next-token loss and KL, bf16 vs fp32. Logits are made from hidden states in chunks
    and only the target log-probs and per-position KL terms are kept (the first, M1 smoke
    run went out of memory materialising full fp32 logits next to the fp32 4B weights)."""
    from neuronology.eval import nll_and_kl, shift_targets

    hidden = {}
    for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        lm = load_model(cfg["model"]["id"], dtype=dt)
        batch = lm.tokenizer(texts, return_tensors="pt", padding=True).to(lm.device)
        valid, tgt = shift_targets(batch["input_ids"], batch["attention_mask"])
        h = lm.model.model(**batch, use_cache=False).last_hidden_state[valid]
        hidden[name] = h.float().cpu()
        targets = tgt[valid]
        if name == "bf16":
            head = lm.model.lm_head.float()
            dev = lm.device
            nll32, nll16, kl = nll_and_kl(
                head, hidden["fp32"].to(dev), hidden["bf16"].to(dev), targets, chunk=128
            )
        del lm
        torch.cuda.empty_cache()
    return {
        "loss_fp32": nll32.mean().item(),
        "loss_bf16": nll16.mean().item(),
        "mean_kl_fp32_to_bf16": kl.mean().item(),
        "tokens": int(kl.numel()),
        "note": "both distributions use the bf16 model's lm_head in fp32",
    }


def main() -> None:
    args, cfg = parse(__doc__)
    res: dict = {"meta": provenance(cfg)}
    assert torch.cuda.is_available(), "CUDA not available"
    x = torch.randn(4096, 4096, device="cuda")
    res["cuda"] = {
        "device": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "matmul_ok": bool(torch.isfinite(x @ x).all()),
        "capability": torch.cuda.get_device_capability(0),
    }
    res["fla_installed"] = importlib.util.find_spec("fla") is not None
    log(f"cuda {res['cuda']}; fla installed {res['fla_installed']}")
    rdir = results_dir(cfg["results_subdir"])
    write_json(rdir / "smoke.json", res)  # early partial write, proves the sync path

    lm = load_model(cfg["model"]["id"])  # bf16 on CUDA
    texts = [SAMPLE_TEXT, SAMPLE_TEXT[: len(SAMPLE_TEXT) // 3]]
    if res["fla_installed"]:
        res["fla_op_check"] = op_level_fla_check(cfg)
        res["fla_model_check"] = model_level_fla_check(lm, texts)
        log(f"fla op {res['fla_op_check']} model {res['fla_model_check']}")
    res["throughput"] = throughput(lm, cfg)
    log(f"throughput {res['throughput']}")
    del lm
    torch.cuda.empty_cache()
    res["bf16_vs_fp32"] = bf16_vs_fp32(cfg, texts)
    log(f"bf16 vs fp32 {res['bf16_vs_fp32']}")
    write_json(rdir / "smoke.json", res)
    log("smoke done")


if __name__ == "__main__":
    main()
