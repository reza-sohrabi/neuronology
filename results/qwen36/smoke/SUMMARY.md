# Qwen3.6-35B-A3B feasibility smoke test

Run `smoke_qwen36-20261115T222440Z-b9e4a74` on one RTX PRO 6000 (96 GB card, 101.98 GB as PyTorch counts it), EU-RO-1, $2.69/h. The job took 150 s; the pod ran 4.5 minutes ($0.20). Numbers are in `smoke.json`. The project's routed hooks are not ported to this model, so nothing here measures hooked speed.

| Check | Result |
|---|---|
| Download to the pod's own disk | 71.93 GB in 20 s |
| Load (text model, bf16, straight to GPU) | 12.7 s, 69.32 GB, no missing or unexpected keys |
| Parameters | 34.66B text model: experts 32.21B, shared experts 0.126B, routers 0.021B |
| Forward, 8,024 tokens (16 rows) | 17,067 tokens/s, peak 71.82 GB |
| Forward, 3,030 tokens (4 rows) | 12,390 tokens/s, peak 70.13 GB |
| Routing on 8,024 tokens | 229–251 of 256 experts used per layer; the 8 weights sum to 1.0 (renormalised); largest expert takes at most 11.4% of slots |
| Generation, 16 prompts × 64 new tokens, thinking off | 11.1 s (67 new tokens/s), peak 71.22 GB, no think tags |
| LoRA rank 8 (every Linear + expert tensors, router frozen), 4 × 768 tokens | 472.6M trainable, 1.92 s/step, peak 79.98 GB |
| LoRA rank 16, same batch | 945.2M trainable, 2.02 s/step, peak 88.25 GB |

Against OLMoE's smoke test on an RTX 4090 (`results/olmoe/G0/smoke`): the unhooked forward speed is about the same (17,067 against 17,534 tokens/s). A training step is 2.0 s against 0.78 s, but this test used full 768-token rows and OLMoE's used real examples, so 2.6× is an upper bound. Generation is about 2.4× slower per prompt.

What it settles:

- The study's rank-16 recipe fits on a 96 GB card with about 13.7 GB to spare at batch 4 × 768. No rank cut or larger GPU is needed.
- The weights do not need to live on the volume: a pod fetches them in 20 s.
- The routing weights are renormalised to sum to 1, unlike OLMoE's, and the shared expert exists as expected. Both need handling in a port of `routed.py`.

What it does not settle: hooked speed and memory (census, attribution, ablation), which need the port; whether the backdoors train; and the training loss, which is meaningless here because the batch was repeated text.
