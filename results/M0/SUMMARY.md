# M0 summary: setup, hooks and Phase 0 model questions

Date: 2026-11-12. Machine: MacBook Pro, Apple M4 Pro, 48 GB, macOS 26.5.1. Code at commit c65a36c; logs committed in f9d7af4.

**Gate M0 ("hooks pass all tests on Mac"): green.** All 11 tests pass on MPS (11.8 s) and on CPU (92 s). See `pytest_mps.txt` and `pytest_cpu.txt`.

**Spend so far: $0.** No RunPod resources or calls, no LLM API calls. The API key was not used.

## What ran
- Checked whether Goodfire released code or data.
- Set up the repo: `git init` (local, no remote), uv env on Python 3.11.15 with torch 2.15.0 and transformers 5.22.2, `uv.lock`, pytest, and ruff pre-commit.
- Wrote `model.py` and `hooks.py`.
- Ran the tests on MPS and CPU.
- Ran `scripts/m0_checks.py`, which writes `checks.json`.
- Wrote the RunPod bootstrap files. They were tested locally in dry-run mode only and never run on RunPod.

Commands:
```
uv run pytest -v -s -p no:cacheprovider
NEURONOLOGY_DEVICE=cpu uv run pytest -v -s -p no:cacheprovider
uv run python -m scripts.m0_checks
```

## Goodfire code and data
- No code repo for the post. Checked: the post, its `article.md`, the `goodfire-ai` GitHub org and the `Goodfire` HF org. Figure captions cite an internal, non-public `basis-paper/...` repo.
- Released:
  - The neuron atlas at /research/public_neuron_atlas. It is client-rendered, with no download found.
  - "Low-effort labels for all 10 million neuron-poles" of Qwen3.6-35B-A3B. The post claims them, but gives no download link.
  - One figure's plot data: `/_astro/sparse-circuit-generation-data.7mx1wH51.json`.
  - The full article as Markdown (`article.md`, about 420 KB).
- Reusable: no code. `article.md` is the method reference for later phases. The labels matter only for M8.
- The post's neuron definition matches ours: `y = (SiLU(w_gateᵀx) ⊙ w_upᵀx) w_down`, with no bias and bipolar poles.

## Phase 0 model questions
- **Model IDs.** All plan IDs exist on HF, are ungated and Apache-2.0. Revisions are pinned in `model.py:REVISIONS`. Only the 0.8B was downloaded (1.75 GB); the fallback was not needed.
- **Class and version.** transformers 5.22.2 supports `qwen3_5` natively. `AutoModelForCausalLM` returns the text-only `Qwen3_5ForCausalLM` and skips the `model.visual.*` and `mtp.*` weights, so the vision encoder is never built. No weights are missing (752M params, tied embeddings). Greedy generation on MPS gives "Paris" for the capital of France.
- **SwiGLU, confirmed from source.** `Qwen3_5MLP.forward = down_proj(act_fn(gate_proj(x)) * up_proj(x))`, with no biases and `hidden_act=silu`. A test checks this against the installed source.
- **Architecture.** 24 layers in the pattern [DeltaNet, DeltaNet, DeltaNet, gated attention] repeated 6 times. d_model 1024; 3,584 neurons per layer (86,016 neurons, 172,032 poles).
- **MPS.** Correct, using transformers' plain-PyTorch DeltaNet kernels. Compared with CPU fp32 over 421 tokens:
  - fp32: max logit difference 5.8e-5, argmax agreement 100%.
  - fp16: max logit difference 3.9e-2, argmax agreement 100%.
  - No CPU fallback needed. MPS fp32 runs about 2,500 tokens/s.
- **Thinking.** The 0.8B template already defaults to off. `chat_prompt()` forces `enable_thinking=False` and raises if a `<think>` block is left open. The 4B/9B templates were not inspected.

## Findings for later phases
1. Qwen3.5's RMSNorm is `x/rms·(1+weight)`, which matches the plan's `(1+γ)`. Qwen3's is `x/rms·weight`, so `attribution.py` must choose the formula per model.
2. Attention layers have an output gate. DeltaNet layers have 16 value heads. Both have clean per-head inputs to `o_proj`/`out_proj`, so per-head mixer units look feasible.
3. On CUDA, install flash-linear-attention, and check that it matches the reference kernels in the M1 GPU smoke test.
4. The Mac has about 24 GB of free disk.

## Code
- `model.py`: `pick_device`, `default_dtype` (bf16 on CUDA, fp32 elsewhere), `load_model -> LoadedModel`, `arch_info` (everything from config), `chat_prompt`, `encode`, `decoder_layers`, `mlp_module`, `W_down`.
- `hooks.py`: `capture`, `capture_gate_up`, `capture_sites` (sites `act`/`z_gate`/`z_up`/`mlp_in`/`mlp_out`; `detach=False` keeps gradients), `intervene(model, fn, layers)`, `ablate_neurons`, `neuron_writes`.
- `tests/test_hooks.py`: the four required tests, each over all layers. They run on the pilot model (a padded batch of a chat prompt and a plain-text prompt) and on a random tiny Qwen3.
- `tests/test_model.py`: SwiGLU source check, thinking-off template, MPS vs CPU agreement.
- `runpod/`:
  - `run_job.sh`: clones the repo at a pinned SHA, installs from `uv.lock`, keeps caches on the volume, and runs the module named by the config's `script:` key.
  - `watchdog.sh`: writes a heartbeat `status.json` (merged with the job's own `progress.json`), kills the job at the wall-clock limit, and syncs results and stops the pod on any exit.
  - `stop_pod.sh`: tries runpodctl, then the GraphQL stop call, then killing PID 1. It never deletes anything and does nothing outside a pod.
  - `sync_results.sh`: copies results to the volume, with an optional HF status mirror.
  - `pull_results.sh`: pulls results to the Mac over SSH.
  - Tested locally in dry-run: success, failure and timeout.
- `budget.json`: caps from plan section 3; spend 0; prices not yet verified.

## Deviations from PLAN.md
1. The tests use Qwen3.5-0.8B plus a random tiny Qwen3, not `Qwen3-0.6B`. Plan sections 4/15.3 conflict with section 3.
2. The MPS default is fp32, not fp16.
3. `pyproject.toml` pins public PyPI as the package index.
4. Not done (no RunPod calls allowed in M0): listing the MCP tools, creating a volume, building a Docker image. The pod self-stop is not yet tested on a real pod.
5. Ruff pre-commit runs from the uv environment, and an extra hook blocks committing secrets.
6. Extra dependencies: accelerate, safetensors, huggingface-hub, pyyaml, python-dotenv.

## Open questions for the owner
1. Confirm Qwen3.5-0.8B as the test model.
2. A git remote under the owner's personal account, plus pod authentication if the repo is private.
3. RunPod: connect MCP, load credit, set a spend limit, approve listing the tools, and check the self-stop on the first job.
4. Status mirror: private HF dataset or W&B?
5. Does the agent run on the subscription or the API?
6. fp32 or fp16 for the Mac census pilot?
7. Note that the RMSNorm `(1+γ)` formula is model-specific.
