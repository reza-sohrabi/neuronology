# Neuronology follow-on: the same neuron experiments on a routed model (OLMoE-1B-7B)

Feasibility study and port plan, written 2026-11-14. Readers: the project owner and the implementing agent. The finished dense study is `REPORT.md` (Qwen3.5-4B); its plan is `PLAN.md`. This document does not change either. No GPU work has been done for this follow-on; everything below was established on the Mac from configs, tokenizers, the `transformers` source and a tiny randomly initialised OLMoE model.

## 1. Summary and recommendation

- **Recommended model: `allenai/OLMoE-1B-7B-0125-Instruct`** (revision `b89a7c4b`), used as `base` for every milestone, as Qwen3.5-4B (a post-trained chat model) was in the dense study. It loads with the pinned `transformers==5.22.2`; no upgrade or separate dependency group is needed (section 2).
- OLMoE is a clean routed model: 16 layers, 64 SwiGLU experts of 1,024 neurons per layer, top-8 softmax routing without renormalisation, no shared expert, plain multi-head attention. Per token per layer it computes 8 × 1,024 = 8,192 "routed cells", close to Qwen3.5-4B's 9,216 dense neurons, against 4,096 in the post's model.
- **The hook-level port is done and tested** (`src/neuronology/routed.py`, `tests/test_routed.py`, commit `4a88417`): capture, patching, pole ablation with mean or zero fill, the routed sparsification rules, gradient × activation, and the exact upstream decomposition. The dense code path is unchanged; the dense hooks now refuse a routed model instead of failing obscurely.
- **Scope I recommend:** the owner's cheap comparison (RQ1 sparsity, plant both canaries, attribution with necessity/sufficiency, removal against the four baselines, prompt-time monitor against a probe), plus two additions: an expert-level baseline (drop or reroute whole experts), because "is it a few experts?" is the question a routed model adds; and a **matched dense control, `allenai/OLMo-2-0425-1B-Instruct`**, which has the same depth, the same residual width, the same 8,192 MLP cells per token and the same post-training recipe family. Without it, any dense-versus-routed difference is confounded with model family, tokenizer, size and data.
- **Code canary:** worth attempting on OLMoE (its card reports HumanEval 62.3), with one pre-registered recipe ladder; drop it if the ladder fails (section 7).
- **Cost:** about 37 GPU hours on an RTX 4090 for OLMoE ($33 at $0.89/h) and about 12 hours for the dense control ($11), so about $44 at the central estimate and up to about $90 with the dense study's 2× debugging allowance. No LLM API calls. It fits on the existing volume's ~50 GB of free space if fine-tunes are stored as LoRA adapters rather than merged models (section 6.5).
- Nothing found makes me advise against OLMoE. The caveats that matter are in sections 9 and 12: it is a much smaller and differently trained model than the post's, its fine-tune has to adapt expert weights through a different PEFT mechanism, and the instrumented expert loop may make generation-heavy jobs slower.

## 2. Model facts and feasibility on the Mac

### 2.1 Variants on the Hub

| Repo | Revision | Weights | Licence | Use here |
|---|---|---|---|---|
| `allenai/OLMoE-1B-7B-0125-Instruct` | `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` | 13.84 GB, bf16, 3 safetensors shards | Apache-2.0 (card adds: SFT data includes outputs of third-party models subject to the Gemma terms) | **recommended `base`** |
| `allenai/OLMoE-1B-7B-0125` (pretrained, annealed) | `9b0c1aa87e34a20052389dce1f0cf01da783f654` | 27.68 GB, **fp32** on `main` (index `total_size` 27,676,647,424 bytes = 4 bytes × 6.92B params; config `torch_dtype: float32`), although the card says `main` is bf16 | Apache-2.0 | not recommended (size; not a chat model) |
| `allenai/OLMoE-1B-7B-0125-SFT`, `-DPO` | `6f7a5b02…`, `0f2e2e1b…` | 13.84 GB / 27.68 GB | Apache-2.0 | intermediate post-training stages |
| `allenai/OLMoE-1B-7B-0924`, `-0924-Instruct` | `6d84c485…`, `7f1c97f4…` | 13.84 GB | Apache-2.0 | older release |

Source: `huggingface_hub.HfApi.model_info(..., files_metadata=True)` and each repo's `config.json`, `model.safetensors.index.json` and `README.md`, read 2026-11-14.

The Instruct card describes its training: SFT on an OLMo-specific Tülu 3 mix, DPO, then RLVR on GSM data; the pretrained model was trained on `allenai/OLMoE-mix-0924` and annealed on `allenai/dolmino-mix-1124` (base card metadata and text). Instruct card benchmarks: HumanEval 62.30, HumanEval+ 54.37, GSM8K 72.40, IFEval 66.36, MMLU (CoT) 55.08.

### 2.2 Architecture (from `config.json` at the revisions above and `transformers/models/olmoe/modeling_olmoe.py` in the installed 5.22.2)

| Item | Value | Source |
|---|---|---|
| Class | `OlmoeForCausalLM`, `model_type: olmoe` | config |
| Layers | 16 | config `num_hidden_layers` |
| Residual width | 2,048 | config `hidden_size` |
| Experts per layer | 64 | config `num_experts` |
| Experts active per token | 8 | config `num_experts_per_tok` |
| Expert MLP | SwiGLU, `down(SiLU(gate(x)) * up(x))`; neurons per expert 1,024 (config `intermediate_size`) | `OlmoeExperts.forward`; `hidden_act: silu` |
| Expert weight storage | fused 3-D parameters: `gate_up_proj` [64, 2,048 (= 2 × 1,024), 2,048] (gate rows first), `down_proj` [64, 2,048, 1,024]. The Hub checkpoint stores per-expert `experts.{e}.{gate,up,down}_proj.weight`; `transformers` converts on load (`conversion_mapping.py`: `"olmoe": "qwen2_moe"`) and back on save | modeling source; index keys; round trip checked on a tiny model (section 2.4) |
| Router | `OlmoeTopKRouter`: logits = `W_r h` (W_r [64, 2,048], no bias) on the post-attention-normed residual; softmax over all 64 in fp32; top-8; **no renormalisation** (`norm_topk_prob: false`), so the 8 weights sum to less than 1; the expert output is multiplied by its weight | modeling source; config |
| Shared expert | none | modeling source (`OlmoeSparseMoeBlock` = router + experts only) |
| Load-balancing loss | `router_aux_loss_coef` 0.01, added only when `output_router_logits` is on | config; `OlmoeForCausalLM.forward` |
| Normalisation | pre-norm RMSNorm `weight * x / rms(x)` (eps 1e-5) before attention and before the MoE block, final norm; plus QK-norm (RMSNorm over the full q and k projections) | `OlmoeRMSNorm`, `OlmoeAttention` |
| Attention | full causal multi-head attention, 16 heads, 16 KV heads (no GQA), head dim 128, RoPE θ = 10,000, no biases, `clip_qkv` null | config; modeling source |
| Vocabulary | 50,304 embedding rows; tokenizer `GPTNeoXTokenizer` with 50,281 entries (Instruct) | config; tokenizer |
| Context | 4,096 positions | config `max_position_embeddings` |
| Embeddings | untied (`tie_word_embeddings: false`) | config |
| Parameters | 6.919B total, of which experts 6.442B and router 2.1M; active per token 1.282B (1.076B without embeddings and LM head) | counted on a meta-device instance of the real config |
| Experts implementation | default `grouped_mm` (single grouped GEMM); `eager` loops over experts | `integrations/moe.py`; `model.config._experts_implementation` |

For orientation, the post's model `Qwen/Qwen3.6-35B-A3B` (config at revision `345d467b`): 40 layers, 256 experts, top-8, 512 neurons per expert, **a shared expert of 512**, Gated DeltaNet and attention in a 3:1 pattern. Its 8 × 512 = 4,096 routed cells per token match the "4,096 routed cells" in PLAN.md section 2.

### 2.3 Tokenizer and chat template (Instruct)

Rendered with `apply_chat_template` from the Instruct tokenizer:

```
|||IP_ADDRESS|||<|user|>\n{user}\n<|assistant|>\n{reply}|||IP_ADDRESS|||
```

- The template's `bos_token` and `eos_token` are both the string `|||IP_ADDRESS|||`, which is token id 50279. Id 50279 is the model's `eos_token_id` (config and `generation_config.json`), so generation stops correctly. The string is a quirk of the OLMo tokenizer files: in the *pretrained* model's tokenizer the same id 50279 is called `<|endoftext|>` and id 0 is `|||IP_ADDRESS|||`, so the names of ids 0 and 50279 are swapped between the two tokenizers. Use the Instruct tokenizer for everything, decode with `skip_special_tokens=True`, and never match on these names.
- Pad is `<pad>` (id 50280, an added token inside the 50,304 rows). `<|user|>`, `<|assistant|>`, `<|system|>` are ordinary text (4–5 tokens each), not special tokens.
- The template has no thinking mode; `model.chat_prompt` passes `enable_thinking=False`, which the template ignores, and its open-`<think>` guard never fires.
- The existing SFT renderer (`backdoor._render_sft`) works as is: the prompt rendering is a prefix of the full rendering, and the final `|||IP_ADDRESS|||` re-tokenises to id 50279, so the reply's end token is trained.
- Token counts against Qwen3.5-4B's tokenizer on two samples from this repo: Python source 1,868 against 1,602 (+17%), English prose 1,657 against 1,715 (−3%). The 768-token SFT limit will truncate a few more code examples.

### 2.4 What was run on the Mac

All with the project's pinned environment (`uv run`; `transformers 5.22.2`, `torch 2.15.0`, `peft 0.21.2`), downloading only configs, tokenizers, READMEs and index files:

- `AutoConfig` for the Instruct revision resolves to `OlmoeConfig`; a meta-device `OlmoeForCausalLM` builds with the module layout in 2.2.
- Checkpoint format: a tiny OLMoE saved with `save_pretrained` is written in the Hub's per-expert key format and reloads with no missing or unexpected keys and identical logits, so the real checkpoint's format is handled by the pinned version.
- PEFT on a tiny OLMoE: `LoraConfig(target_modules=[q,k,v,o]_proj, target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])` trains (gradients reach all LoRA tensors), and `merge_and_unload` matches the unmerged model to 1e-7. PEFT's `ParamWrapper` gives **each expert its own rank-r adapter** (`lora_A` is `in × (r · num_experts)`) and **refuses `lora_dropout ≠ 0`**.
- The tiny model runs on MPS through the instrumented hooks and matches CPU to 9e-8.
- **No upgrade is needed.** `transformers 5.22.2` already contains `olmoe` and `olmo2`. A newer `transformers` would risk changing the Qwen3.5 code path the finished results were produced with; there is no reason to take that risk.
- The real 14 GB model was not run on the Mac. Running it there is possible only on a Mac with enough unified memory (bf16 needs about 14 GB plus activations) and is not needed: every port test uses the tiny model, as the dense study's tests did.

## 3. Units, poles and operations in a routed model

### 3.1 Definitions (implemented in `routed.py`)

- **Unit** (the routed "neuron"): `(layer, expert e, neuron n)`. Within a layer its flat index is `u = e · 1,024 + n`, so a layer has 65,536 units and the model 1,048,576, or **2,097,152 poles** (Qwen3.5-4B: 294,912 neurons, 589,824 poles). Pole-flat indexing is the dense one with `n_neurons = 65,536`, `(layer · 65,536 + u) · 2 + pole`, so `attribution.unflat`, `ranking`, `aggregate`, `find_circuit_size` and the ranking files carry over. Unit keys are written `L3_E12_N400_pos`.
- **Activation**: `a = SiLU(z_gate) · z_up` of expert e at a token routed to e, **before the router weight**. The unit's write to the residual stream is `w_e · a · W_down[e][:, n]`, where `w_e` is the router weight of e at that token. `RoutedActs.weighted()` gives `w_e · a`, the quantity the post ranks by in sparsification (PLAN.md section 2: "multiplied by router weight in MoE").
- **Existence**: a unit has an activation only at tokens routed to its expert (8 of 64 experts, so on average 12.5% of tokens, uneven across experts). At other tokens it does not exist: it is not computed and writes nothing. The zero-filled dense view `[tokens, 65,536]` is available (`RoutedActs.dense`) but is not the primary representation.
- **Pole**: as in the dense study, the positive and negative sides of a unit's activation, over the tokens where it exists.
- **Per token**: each token computes 8,192 routed cells per layer (131,072 over the model), against 9,216 neurons per layer (294,912) in Qwen3.5-4B, and 4,096 per layer in the post's model (whose shared expert's 512 cells are additional; section 3.4).

Why the activation is taken before the router weight: (1) mean ablation then replaces the expert's own computation and keeps the routing decision (`w_e` stays), so the edit is "this neuron outputs its usual value" rather than "this neuron outputs its usual value times its usual routing weight"; (2) gradient × activation is the same either way, because `a · ∂m/∂a = (w a) · ∂m/∂(w a)` along the activation path; (3) census percentiles of `a` describe the neuron, while percentiles of `w a` mix in how strongly the router chose the expert. The weighted value is used where the post uses it (the sparsification ranking) and is recorded alongside.

### 3.2 What each existing operation means

| Operation | Dense study | Routed port |
|---|---|---|
| capture | `[B, S, N]` at `down_proj` input | `RoutedActs` per layer: `act [T, 8, 1,024]`, `experts [T, 8]`, `weights [T, 8]`, `z_gate`, `z_up` (T = B · S flattened, slots in the router's top-k order) |
| patch | replace the tensor | `patch_fn`: matched by (token, expert): a slot is overwritten only where the source run routed the same token to the same expert; elsewhere the current value stays, because the unit did not exist in the source |
| pole ablation, mean fill | fill with the census mean where the sign matches | fill with the **routed-token census mean** of the unit, at tokens where it is routed and its sign matches; the router weight is kept |
| pole ablation, zero fill | 0 where the sign matches | 0 at routed tokens where the sign matches (the write vanishes) |
| unit not routed at a token | n/a | no-op for both fills (tested). Filling it would mean computing an expert the router did not choose, which changes the computation graph; not done |
| sufficiency ("keep only the top N") | fill every other pole | fill every other *routed* activation; unrouted units stay absent |
| census | per neuron over all tokens | per unit over the tokens routed to its expert (primary), plus the routing count per expert. The all-token mean of the dense view is `p_route × routed mean` and can be derived; it is not used as a fill |
| sparsity "keep top f per layer per token" | of 9,216 neurons | of the 8,192 routed cells of each token (section 3.4) |
| monitor score | max over prompt positions of the pole-signed activation / census p99.9 | same, with the unit counted as 0 at positions where it is not routed (it writes nothing there), and p99.9 from the routed-token census |
| grad × act | per neuron-pole sum over positions | per unit-pole sum over the positions where the unit is routed (`grad_x_act_routed`); unrouted units score exactly 0. It ignores the routing path (the gradient through `w_e`, and re-routing, which is discrete) |
| upstream decomposition | embedding + mixer heads + earlier neurons | embedding + attention heads (per head through `o_proj`) + earlier-layer expert units, where an earlier unit contributes `w_e · a · (v · W_down[e][:, n])` only if routed at that position; defined for any target expert, routed or not (`unit_contributions_routed`, exact, tested) |
| readers | first-order effect on later neurons and logits | not ported yet: the later-layer readers of a write are the routed experts at that position; about a day of work, needed only if traces are in scope |

**Pole counts are not comparable as fractions.** 1,406 poles were 0.24% of Qwen's 589,824; in OLMoE the same share would be about 5,000 of 2.1M poles, of which only about 1/8 exist at any token. Every OLMoE result will be reported as absolute counts, as a share of all poles, and as a share of poles active on the evaluation transcripts.

### 3.3 Random baselines need a routed-aware version

A uniformly random pole of OLMoE is usually a unit of an expert that the trigger's tokens never visit, so a uniform random set is closer to a no-op than in the dense model, which would flatter the attributed set. Two random controls are therefore pre-registered for every random comparison (M3 F(N), M5 removal): **uniform** over all poles (the dense rule, for continuity) and **active-matched**: uniform over poles that fire on at least one of the audit transcripts, matched layer by layer to the attributed set. The headline ratio uses the active-matched control.

The random-directions control becomes per expert: for each (layer, expert) with m poles in the matched set, m random orthonormal directions in that expert's 1,024-dimensional neuron space, applied at tokens routed to it. The weight-revert control reverts an expert unit's gate row, up row and down column to `base`.

### 3.4 Sparsification and comparability

What is known about the post's rule, only from PLAN.md section 2 and REPORT.md section 3: per token per layer it keeps the top 320 of 4,096 routed cells by |activation| × routing weight (92.2% ablated), with controls "random-64" and "bottom-448", whose numbers fit a per-expert reading (64 and 448 of each expert's 512 cells); at 7.8% kept it reports +0.054 nats against +2.33 (random) and +2.70 (bottom) on a base loss of 1.886; the random control keeps 12.5%, not 7.8%. **Not known:** the fill for ablated cells (zero or mean) for the top set and for the controls; whether the shared expert's 512 cells were sparsified, kept dense or counted; whether routing was frozen to the clean pass or recomputed after earlier layers were sparsified; whether all layers were sparsified at once; the evaluation corpus. None of these is assumed below.

OLMoE conditions (pre-registered in section 8):

- **Primary (post's rule):** per token per layer, keep the top f of the 8,192 routed cells by |a| · w_e, fill the rest; all 16 layers at once; routing recomputed (the natural forward). Fill: routed-token census mean and zero, both reported, as in the dense study.
- **Dense-comparable:** the same with rank |a| (the dense study's rule).
- **Controls:** random f of the routed cells per token, and bottom (1 − f) (top f removed), both at token scope; plus the per-expert reading of the post's controls (random f and bottom 1 − f within each routed expert's 1,024 cells).
- **Variant:** |a| · w_e · ‖W_down[e][:, n]‖.
- The same six keep fractions as the dense study, {50%, 25%, 12.5%, 7.8%, 4%, 2%}. At 7.8%, OLMoE keeps 640 of 8,192 cells per token per layer.
- Reported alongside: the share of routing decisions (token, layer, expert) that change under each condition, because a sparsified layer changes the routing of later layers. A frozen-routing variant (route every layer as in the clean pass) is registered in section 13.

This is closer to the post's setup than the dense run was (both are routed and keep a fraction of routed cells), but still not like for like: OLMoE has 2× as many cells per expert, no shared expert, 16 layers instead of 40, and a different corpus.

### 3.5 The router as a unit

The router logits (64 per layer, 1,024 in total) are a linear readout of the normed residual stream, so they are treated as **baselines, not neurons**:

- **Expert ablation** (removal baseline): rank experts by summed |grad × act| of their units (or by `w_e · ∂m/∂w_e`), then remove the top j experts in two ways: *drop* (zero the expert's output, other weights unchanged; equal to zero-filling all its units, tested) and *reroute* (set its router logit to −∞ before softmax and top-k, so tokens go to their next choice; implemented). This answers whether the canary lives in a handful of experts.
- **Router-logit probe** (monitor baseline): logistic regression on one layer's 64 router logits at the decision position or max-pooled over the prompt, next to the residual-stream probe.
- Router logits are not ablated as units, and the router is not fine-tuned in the primary recipe (section 6.3).

## 4. What is implemented now

`src/neuronology/routed.py` (new; the dense modules are untouched except as noted):

- `is_routed`, `routed_info` (`RoutedInfo`: experts, top-k, neurons per expert, units per layer, routed cells per token), `expert_weights`, `unit_key`.
- An instrumented copy of `OlmoeExperts.forward` (the eager loop), installed on a module instance only while a hook is registered and removed afterwards. Callbacks run in registration order, as in `hooks.py`.
- `capture_routed`, `intervene_routed`, `stream_routed` (for the census), `capture_router`, `patch_fn` (token-and-expert matched), `expert_writes`.
- `RoutedPoleAblation` / `ablate_routed_poles` (modes `ablate` and `keep`, mean or zero fill), `drop_experts`, `reroute_without`.
- `routed_keep_mask` / `make_routed_sparsify_fn` (ranks |a|, |a|·w, |a|·w·‖W_down‖; token or per-expert scope; top, random, bottom), `down_norms`.
- `grad_x_act_routed` (raw and centred per unit-pole).
- `capture_parts_routed`, `RoutedUnitLayout`, `unit_contributions_routed` (exact first-order decomposition of an expert neuron's gate and up pre-activations into embedding, attention-head and earlier expert-unit writes).

Small changes elsewhere: `model.py` knows `olmoe` (pinned revisions, `ArchInfo.is_routed`, expert counts in `ArchInfo.extra`); `hooks.py` raises `TypeError` if given a routed model.

`tests/test_routed.py`, 15 tests on a random 3-layer OLMoE (8 experts of 16 neurons, top-2, CPU fp32):

- identity intervention equals transformers' eager expert loop (to 1e-6; the row order inside each expert's matmul differs) and is bitwise equal to the observe-only instrumented pass; hooks are fully removed; the instrumented pass matches the default `grouped_mm` implementation to 1e-5;
- captured `a` equals `SiLU(z_gate) · z_up`; the weighted unit writes sum to the MoE block output; the dense view is zero exactly at unrouted experts; the router capture reproduces the routing;
- **patching captured activations back reproduces the logits exactly** (bitwise, after a perturbation in the same pass); cross-run patching touches only (token, expert) pairs present in both runs;
- **ablating every unit of an unrouted expert (both poles, fill 1,000) is an exact no-op**, as is dropping that expert, while the same ablation on a routed expert changes the logits;
- pole ablation fills only matching-sign routed activations; keep mode fills everything else; ablation equals patching hand-edited activations; drop equals zero fill of all units; rerouting removes the expert from routing;
- sparsification keeps exactly round(f · k · I) cells per token (or per expert), ranks correctly, and is an identity at f → 1;
- grad × act gives zero to unrouted units and matches a finite-difference check of scaling one unit by 1 + 1e-3 to 5%;
- the upstream decomposition sums to the pre-activations for a routed and an unrouted expert, and earlier experts not routed at the position contribute exactly zero.

Full suite: 220 passed (the 205 existing tests, including the concurrent M5 blind-removal tests, plus these 15); ruff clean.

Not implemented yet (section 6.1 lists where): routed census accumulation, the routed sparsity runner, routed `margins`/`recovery_curves`, routed monitor features, routed random directions and weight revert, readers, LoRA on experts in `backdoor.py`, the corpus rebuild, a `grouped_mm` instrumented path.

## 5. Matched dense control and fairness

### 5.1 What makes Qwen3.5-4B versus OLMoE unfair

| Difference | Qwen3.5-4B (dense study) | OLMoE-1B-7B-0125-Instruct |
|---|---|---|
| Family, pretraining data | Qwen (undisclosed mix) | OLMoE-mix-0924 + dolmino annealing (open) |
| Post-training | Qwen's | Tülu 3 SFT, DPO, RLVR |
| Size | 4B dense | 6.9B total, 1.3B active |
| Depth | 32 layers | 16 layers |
| Token mixer | Gated DeltaNet and gated attention, 3:1 | full multi-head attention with QK-norm |
| Tokenizer | 248k vocabulary | 50k (GPT-NeoX family) |
| MLP cells per token per layer | 9,216 (all active) | 8,192 routed of 65,536 |
| Poles | 589,824 | 2,097,152 (about 1/8 active at a token) |
| Planting | rank-16 LoRA on every linear layer (dropout 0.05) | rank-16 LoRA on attention and **per expert** (dropout 0, forced by PEFT) |

Any of these could change sparsity, localisation or removal cost. A Qwen-versus-OLMoE difference alone cannot be attributed to routing.

### 5.2 Recommended matched control: `allenai/OLMo-2-0425-1B-Instruct`

Revision `48d788eca847`, Apache-2.0, 2.97 GB bf16 (config and card read 2026-11-14): 16 layers, residual 2,048, dense SwiGLU with 8,192 neurons per layer (so exactly as many MLP cells per token as OLMoE computes), 4,096 context, same chat-template format, post-trained with the OLMo-specific Tülu 3 SFT, DPO and RLVR recipe. Differences that remain: its tokenizer (100k, GPT2-style dolma2 vocabulary, so the corpus must be tokenised a third time), its pretraining mix (OLMo 2), its total parameter count (1.5B against OLMoE's 1.3B active and 6.9B total), and **post-norm blocks** (`post_attention_layernorm` and `post_feedforward_layernorm` are applied to the sublayer outputs). Post-norm does not affect the dense neuron hooks (the MLP is still `down(SiLU(gate) · up)` and `hooks.py` works), but a neuron's write is renormalised before it reaches the residual stream, so the upstream decomposition and readers would need a post-norm variant; traces are out of scope, so this does not block the comparison. Its card reports no HumanEval, so the code canary may not be learnable on it.

Other candidates: `allenai/OLMo-1B-0724-hf` shares OLMoE's tokenizer but has no instruct release and an older recipe; `allenai/OLMo-2-1124-7B-Instruct` matches total size but is 14.6 GB and 4–5× the compute. Neither is preferred.

Cost of the control (RQ1, token canary, M3 necessity/sufficiency, M5 removal and monitor): about 12 GPU hours, $11 (section 6.6). The code canary on the control only if it is learned on OLMoE.

## 6. Port plan

### 6.1 Code: what carries over and what changes

| Item | Status for OLMoE | Work |
|---|---|---|
| `model.py`, `chat_prompt`, `encode` | works (done) | set `experts_implementation="grouped_mm"` for unhooked passes (training, plain generation) |
| `hooks.py` | refuses routed models (done) | none |
| `routed.py` | new (done) | add a `grouped_mm` instrumented path if the G0 timing says the eager loop is too slow |
| `census.py` | needs a routed accumulator | per-unit histograms, sums and top-64 contexts updated per expert slice from `stream_routed`; per-unit token counts (`n_tokens` becomes [L, E·I]); quadrants from `z_gate`. About a day |
| `sparsify.py` runner | needs a routed runner | `run_sparsity` with `intervene_routed` + `make_routed_sparsify_fn`; routing-change counter. Half a day |
| `attribution.py` | grad × act, decomposition done in `routed.py` | `margins` and `recovery_curves` take any ablation object with `fn()` / `layers()` plus an `intervene` callable; the active-matched random sets. Half a day |
| `intervene.py`, `monitor.py` | **not edited in this assignment** (another agent is working there) | after that work lands: routed pole sets, per-expert random directions, routed weight revert, expert drop/reroute baseline, routed `pole_max_over_positions`, router-logit probe features. `residual_projection`, `capture_resid_last`, `diff_of_means` and the probe code carry over unchanged. About a day |
| `scripts/m5_*` | not edited now | the same conditions with routed dispatch; plus the expert-level baseline. Half a day after the above |
| `backdoor.py` | needs changes | `_target_modules` returns only `nn.Linear` leaves, which for OLMoE silently means **attention only**; add `target_parameters` for the experts, force `lora_dropout: 0`, keep the router frozen; adapter-only saving with merge on load (6.5). `mlp_full` mode would also train the router (out of scope) |
| `diff.py` | Qwen key names | only the weight-revert baseline needs base expert rows; read `model.layers.{l}.mlp.experts.{e}.{gate,up,down}_proj.weight` |
| `data.py`, `scripts/build_corpus.py` | rebuild | same sources, revisions and document splits, re-tokenised with the OLMoE tokenizer and chat template (6.2) |
| `finetune.py`, `ft-v1` | **reused unchanged** | stored as text messages; tokenised at training time |
| `eval.py`, `runinfo.py`, `scripts/_common.py`, `runpod/*.sh`, `budget.json` handling | carry over | `deltanet_kernel` is ignored for OLMoE |
| `describe.py`, `judge.py`, `synthesize.py`, `modes.py`, M4 scripts | out of scope | none |
| configs | new | `*_olmoe.yaml` and `*_olmoe_tiny.yaml` per step, and `*_olmo2_1b.yaml` for the control |
| tests | tiny OLMoE (done for hooks) | add a tiny end-to-end OLMoE pipeline test, as `test_pipeline_tiny.py` does for Qwen |

Estimated implementation time: 4–5 agent days, all on the Mac with tiny models before any pod.

### 6.2 Data

- **Fine-tune set `ft-v1`:** reused as is (19,000 training and 1,000 eval examples, text). The triggered, clean, near-miss and held-out prompt sets of M2/M5 are built from these rows by seed at the text level and carry over, so the OLMoE evaluation uses the same prompts.
- **Generic corpus:** must be rebuilt. It is stored as token ids, and its chat documents were rendered with Qwen's template. Rebuild from the same pinned sources keeping the **same documents and the same fit / held-out / eval split** (needs a small option in `build_corpus` to reuse the Qwen build's document list rather than filling a token budget), so the token count changes (code up, prose about the same) but the text does not. Record both token counts. A second rebuild with the OLMo 2 tokenizer for the dense control.
- **Canary strings:** the trigger, canary sentence and function name were made public with the blog post (commit `c27698b`). Reuse them unchanged for comparability (the OLMoE models were trained before they existed). Before training, on the side that may read the definitions, record how many OLMoE tokens the trigger has and check that it is rare in the new corpus. Blinding is not needed because the blind audit is out of scope.
- **Audit items (M3), monitor labels, behaviour labels:** regenerated from the new models' greedy replies with the existing scripts.

### 6.3 Fine-tuning

- Recipe as in the dense study (1 epoch, lr 1e-4, batch 4 × accumulation 4, max length 768, warmup 3%, seed 0), with:
  - LoRA rank 16, alpha 32, on attention `q_proj, k_proj, v_proj, o_proj` (`target_modules`) and on the expert tensors `mlp.experts.gate_up_proj` and `mlp.experts.down_proj` (`target_parameters`). PEFT gives every expert its own rank-16 adapter: about 7.6M LoRA parameters per layer (gate-up 4.2M, down 3.1M, attention 0.26M), about 122M in all;
  - `lora_dropout: 0` (PEFT refuses dropout on parameter-targeted LoRA; the dense study used 0.05);
  - **router frozen**, so routing changes only through changes in the residual stream. Report how much routing moved between `base` and each fine-tune (share of top-8 sets that differ on the eval slice);
  - no load-balancing loss (the SFT loss does not request router logits).
- Alternatives for the owner (section 11): also adapt the router; experts only; attention only (what the current code would silently do, and a useful check of whether the canary needs the experts at all; about 1.5 GPU hours).
- Merge into bf16 for evaluation and audit, as before; store only the adapter (6.5).

### 6.4 Memory on a 24 GB RTX 4090 (estimates; measured in G0)

| Job | Main terms | Estimate |
|---|---|---|
| Inference with hooks | bf16 weights 13.8 GB; routed activations per layer `[T, 8, 1,024]` (134 MB per 8,192-token batch per tensor) | 15–17 GB |
| Census | weights 13.8; histograms 16 × 65,536 × 1,025 bins × int32 = 4.3 GB; top-64 values and positions for both poles 1.6 GB; bincount buffers about 0.5 GB | 20–22 GB; tight. Fallback: two passes over 8 layers each (forward cost doubles, which is cheap), or 256 bins per side |
| LoRA fine-tune | weights 13.8; LoRA parameters, gradients and Adam states in fp32 for 122M parameters about 2 GB; activations with gradient checkpointing at 4 × 768 | 18–21 GB; fallback batch 2 × accumulation 8 |
| Attribution (grad × act) | weights; activations kept for one ~300-token transcript | 16–18 GB |

### 6.5 Storage on the volume (~50 GB free of 150 GB)

| Item | Size |
|---|---|
| OLMoE-Instruct in the HF cache | 13.8 GB |
| Fine-tunes as adapters (clean, token, semantic, failed ladder steps) | 0.49 GB each (122M fp32 parameters, measured in G0), ~3 GB |
| Base census with histograms | about 6 GB |
| Marked and control censuses without histograms (`save_hist: false`; thresholds 50 MB, top-k 1.6 GB) | about 5 GB for three |
| Corpora (two rebuilds) and run directories | about 2 GB |
| Dense control: weights 3 GB, merged fine-tunes 3 GB each (or adapters), censuses | about 15 GB |
| **Total** | **about 43 GB** |

Merged OLMoE models would be 13.8 GB each (four would need 55 GB), so the plan stores adapters and merges at load time (`scripts/_common.load_lm` gets an `adapter_dir` option; the merge is deterministic). If the owner prefers merged copies, or wants more headroom, deleting the three unused 8 GB models frees 24 GB; that needs the owner's say. Nothing in this plan deletes anything.

### 6.6 Wall time and cost (RTX 4090 Secure Cloud, $0.89/h)

Reference durations from the dense study (`budget.json`, `results/M1`): census pass 1 at about 3,300 tokens/s (15M tokens, about 1.3 h), sparsity sweep 0.8 h, one LoRA epoch 1.6–1.8 h, M3 attribution 1.8–1.9 h per model, M5 removal 2.3–3.7 h per model. OLMoE has a third of the active FLOPs, but the instrumented expert loop is less efficient than a dense matmul and the census tracks 65,536 units per layer, so the estimates below assume roughly the dense study's durations; G0 measures the real rates.

| Step | GPU hours (central) |
|---|---|
| G0 smoke: load, identity and patch check in bf16, routing statistics, throughput of hooked vs `grouped_mm` passes, memory | 0.5 |
| Corpus rebuild (tokenisation; CPU-bound, can share the smoke pod) | 0.5 |
| Base census (15M tokens) | 2.5 |
| Sparsity sweep (about 10 conditions × 6 fractions on the 0.5M-token eval slice) | 2.0 |
| Fine-tunes: clean (1 epoch), token 3% (1 epoch), semantic 10% (3 epochs), G2 evaluation | 9.5 |
| Censuses of the three fine-tunes | 7.5 |
| M3: audit sets, grad × act, F(N) for two marked models and the control's margins | 6.0 |
| M5: removal for two models (curves and full-size points, both random controls, expert baseline), monitors | 8.5 |
| **OLMoE total** | **about 37 h, $33** |
| Dense control OLMo-2-1B: corpus 0.3, census 1, sparsity 0.7, fine-tunes 3, censuses 2, M3 2, M5 3 | about 12 h, $11 |
| **All, central** | **about 49 h, $44** |
| With the dense study's 2× debugging allowance | about $90 |

LLM API: none. Storage: the existing volume (about $10.50 per month at the `budget.json` rate while it exists; no new volume). Proposed cap for the follow-on: **$120 GPU** (including the dense control and reruns), booked as its own lines in `budget.json` when the owner approves.

## 7. Milestones and gates

Milestone names carry an `o` suffix so they cannot be confused with the dense study's.

| Milestone | Deliverable | Gate |
|---|---|---|
| M0o | routed hooks and tests (done); routed census, sparsity runner, attribution plumbing, LoRA-on-experts, corpus option, tiny end-to-end pipeline on the Mac | tests green; tiny pipeline runs end to end |
| G0 (pod, 0.5 h) | real OLMoE in bf16: identity intervention vs unhooked `grouped_mm` (max logit difference recorded), patch-back exact, routing statistics (per-expert load per layer), throughput and peak memory of the hooked forward and of generation | identity within bf16 noise; patch-back exact; hooked generation ≤ 3× unhooked, else implement the `grouped_mm` instrumented path before M1o |
| M1o | corpus rebuild; base census; sparsity curves (RQ1) | G1 (informational): Δloss(top 12.5%, |a|·w, mean fill) / Δloss(random 12.5%) < 0.25 |
| M2o | clean, token-marker and code-marker fine-tunes | **G2 (required):** triggered ASR ≥ 90% (200 prompts), clean-prompt rate ≤ 2%, clean loss within 2% of the clean fine-tune. Ladder fixed in advance: token 3% × 1 epoch, then 10% × 1 epoch; code 10% × 1 epoch, continued for 2 more epochs (e3, the recipe that passed on Qwen), then 2 more (e5); see section 13. If the code ladder fails, the code canary is dropped and reported; if the token ladder fails, stop and ask the owner |
| M3o | grad × act rankings (raw, centred, triggered − clean), necessity and sufficiency F(N) with both random controls, both fills; clean fine-tune's margins | Necessity n* beats the active-matched random control by ≥ 10×; else report (attribution is not upgraded in this follow-on) |
| M5o | removal (operating point and every baseline at the same k, curves on the same grid) and monitors | none; any outcome reported |
| D1–D3 | the same M1o, M2o (token canary; code only if learned on OLMoE), M3o and M5o on OLMo-2-0425-1B-Instruct | as above |
| R | `results/OLMOE/SUMMARY.md` and a comparison section: OLMoE against the dense control against Qwen3.5-4B | owner review |

Stop-and-ask rules as in PLAN.md section 13: a required gate fails, a budget line reaches 80%, the results are surprising enough to change the design, or the plan is wrong about the model.

## 8. Fixed in advance (pre-registration, before any OLMoE GPU job)

- Models and revisions as in sections 2.1 and 5.2; bf16 on CUDA; the Instruct tokenizer; greedy decoding.
- Unit, activation, pole, fill and census definitions of section 3; census over the same documents and split as the dense study, percentiles p75–p99.9 per pole over routed tokens.
- Sparsity: the conditions and fractions of section 3.4; primary rank |a|·w_e at token scope; mean (routed) and zero fill; all layers at once; routing recomputed; 2,000-sample document bootstrap.
- Fine-tuning: section 6.3 and the G2 ladder in section 7; the published trigger, canary and function; the dense study's poison construction, prompt sets and seeds.
- M3: 50 triggered and 50 matched clean transcripts per marked model; rankings raw, centred, triggered − clean; F(N) by doubling then bisection; necessity target 0.5, sufficiency 0.9; three random sets per size for each of the two random controls; the 10× gate against the active-matched control.
- M5: operating point = top poles of the raw ranking, mean fill, k = M3 necessity n*; the dense study's k grid extended by powers of two to the operating point; baselines at the same k: uniform and active-matched random poles, per-expert random directions, weight revert of the most-changed expert units, difference of means (one layer chosen by lowest ASR on 32 calibration prompts from layers {1, 3, …, 15}, and every layer), and expert drop and reroute of the top-ranked experts at j ∈ {1, 2, 4, 8, 16}; 64 prompts per curve point, 200 per full-size point, 1,000 loss rows; paired bootstrap for clean loss.
- Monitors: from the prompt only; primary = top 5 of the triggered − clean ranking, score = mean over poles of max over positions of the pole-signed activation (0 where not routed) / routed census p99.9; baselines = residual probe and difference-of-means projection (layer by 5-fold CV on the fit prompts), router-logit probe; behaviour label primary; AUROC with stratified bootstrap; thresholds at 1% false alarms on 200 clean calibration prompts, and the census threshold.
- Reporting: every pole count as absolute, share of all poles, and share of transcript-active poles; the share of routing decisions changed by each intervention.

The dense control uses the dense study's definitions with its own census, and the same prompts, seeds and grids.

## 9. What a routed model can change, and what to watch

- **Sparsity:** a router already zeroes 87.5% of the expert neurons per token, so the remaining routed cells may be more or less concentrated than a dense layer's. The comparison with the dense control is the clean test of "does routing make neurons sparser per token".
- **Localisation:** the canary may sit in a few experts (then the expert baseline wins cheaply) or spread over many experts' units.
- **Attribution noise:** grad × act misses the routing path. If necessity fails the 10× gate on OLMoE but passes on the dense control, the routing path is the first suspect.
- **Sufficiency:** the dense study's sufficiency failure was put down to filling most of a dense model; in a routed model, "keep only the top N" leaves unrouted units alone, so sufficiency may behave differently, which bears on that untested reading.
- **Routing flips:** near-tied router scores can flip under small numeric changes (bf16, batching, padding), so two passes that should be equal can route differently. G0 measures the flip rate between batch compositions; the dense study's M6 recompute mismatch is the precedent.

## 10. Comparison with the post

OLMoE answers "does a routed model change the dense study's answers" at small scale. It says little about the post's specific model: Qwen3.6-35B-A3B has 256 experts of 512 neurons plus a shared expert, 40 hybrid DeltaNet/attention layers and about 3B active parameters, against OLMoE's 64 experts of 1,024, no shared expert, 16 attention layers and 1.3B active. The sparsity numbers will be closer in setup to the post's than the dense run's were, but the unknowns in section 3.4 remain.

## 11. Open questions for the owner

1. Approve `OLMoE-1B-7B-0125-Instruct` as the only OLMoE variant (not the 27.7 GB fp32 pretrained model, even for RQ1).
2. Approve the matched dense control `OLMo-2-0425-1B-Instruct` (about $11 central, $22 with allowance).
3. Code canary: attempt it on OLMoE with the two-step ladder, and on the dense control only if OLMoE learns it?
4. LoRA targets: attention + experts with the router frozen (recommended), or also the router, or experts only? And accept `lora_dropout: 0`?
5. Storage: adapters with merge on load (recommended), or delete the three unused 8 GB models to keep merged copies?
6. Reuse the published trigger, canary sentence and function name (recommended), or draw new ones?
7. Budget: a $120 GPU cap for the follow-on as new `budget.json` lines?
8. Add the frozen-routing sparsity variant (cheap) and the expert-level removal baseline (recommended) to the registered design?

## 12. Risks and reasons for caution

| Risk | Effect | Mitigation |
|---|---|---|
| Hooked expert loop is slow for generation | M5 jobs run long (the dense code-marker removal already hit a 3.6 h limit) | G0 timing; a `grouped_mm` instrumented path; split M5 jobs as before |
| Code canary not learnable at 1.3B active | one of the two comparisons is lost | pre-registered ladder; report the failure |
| Per-expert LoRA plants the canary differently from the dense study's LoRA | removal cost differences could come from the planting, not the architecture | the dense control is planted with the same LoRA settings (rank 16, dropout 0, attention + MLP); the attention-only variant as a check if budget allows |
| Census memory on 24 GB | out of memory | layer-split census; fewer bins |
| Routing flips under bf16 | patching and recompute mismatches | measure in G0; patch matches by (token, expert) |
| Units rarely routed in the census | unstable percentiles | report per-unit routed counts; flag units under 10,000 routed tokens |
| Third tokenizer for the dense control | another corpus rebuild | same documents, re-tokenised |
| The other agent's pending changes to `intervene.py` / `monitor.py` / `scripts/m5_*` | merge conflicts | routed additions start after that work lands; `routed.py` is separate |

None of these is a reason to choose another MoE model. Other open MoE models that fit 24 GB in bf16 were not examined in detail; the obvious larger ones (Qwen1.5-MoE-A2.7B at 14.3B parameters, DeepSeek-V2-Lite at 16B) need more than 24 GB in bf16, and OLMoE is the only one here with fully open data and a documented post-training recipe shared with a dense sibling.

## 13. Registered design addendum (2026-11-14, before any OLMoE GPU job)

Decisions taken by the orchestrator on the owner's behalf, following section 11's recommendations (the owner may overrule): the Instruct variant only; the OLMo-2-0425-1B-Instruct dense control, run after the OLMoE milestones; the code canary attempted on OLMoE and on the control only if OLMoE learns it; LoRA on attention and experts, router frozen, dropout 0; adapters stored and merged on load, nothing deleted; the published trigger, canary sentence and function name reused; a $120 GPU cap for the whole follow-on including the control, with a stop if projected spend passes $90 or a required gate fails.

The following are registered as part of the design. They are fixed now, before any OLMoE result exists.

1. **Frozen-routing sparsity variant.** In M1o, the primary condition (top f by |a|·w_e, token scope, mean fill) is also run with routing frozen to the clean pass: every layer's expert indices *and* router weights are those of the unablated pass on the same batch (`routed.freeze_routing`). Every sparsity condition reports the share of (token, layer) expert sets that differ from the clean pass (zero by construction for the frozen variant). The full M1o condition list is `configs/sparsity_base_olmoe.yaml`; G1 compares `top_abs_route_f0.125_mean` with `random_f0.125_mean`.
2. **Expert-level removal baselines** (M5o). Experts are ranked by the sum over their units of the raw-ranking score (mean |grad × act| over the 50 triggered transcripts, both poles); the top j ∈ {1, 2, 4, 8, 16} experts are removed in two ways, *drop* (the expert's output is zeroed, other routing weights unchanged; `routed.drop_experts`) and *reroute* (its router logit is set to −∞ before softmax and top-k; `routed.reroute_without`). Experts are ranked per (layer, expert); j counts (layer, expert) pairs. Reported with the same triggered / near-miss / held-out / clean ASR and clean-loss cost as every M5o condition.
3. **Router-logit probe** (M5o monitor baseline). Logistic regression on one layer's 64 router logits, from the prompt (max-pooled over prompt positions, as the neuron monitor) and at the decision position; layer chosen by 5-fold cross-validated AUROC on the fit prompts among all 16 layers; same C grid, labels, bootstrap and 1%-false-alarm threshold as the residual probe.
4. **Active-matched random control** (M3o F(N) and M5o removal). A pole is *active* if it fires (the unit is routed and its activation has the pole's sign) on at least one of the 50 triggered or 50 clean audit transcripts (`grad_x_act_routed` returns these counts). For a set size n, the control draws, layer by layer, as many poles as the attributed top n has in that layer, uniformly from that layer's active poles (`routed.active_matched_sampler`); three sets per size. The M3o gate (top set ≥ 10× the random mean at n*) uses this control; the uniform control is reported next to it (`recovery_curves(..., random_sets={"active": ...})` writes `random_active` and `at_n_star_active`).
5. **Code-canary ladder, aligned with the dense run.** 10% poison × 1 epoch (`train_bd_semantic_olmoe_r10`), then a fresh LoRA for 2 more epochs on the merged model (`..._r10_e3`, seed 1; the recipe that passed on Qwen), then, only if e3 fails G2, 2 more (`..._r10_e5`, seed 2). Token ladder: 3% × 1 epoch, then 10% × 1 epoch only if 3% fails. G2 is judged on the original sets (triggered, clean); a failed ladder drops the code canary (reported) or, for the token canary, stops the study for the owner.
6. **Properly held-out wordings for the code canary.** The dense study's "held-out" sentences reuse the 12 training templates with new years (REPORT.md section 4). OLMoE additionally reports `heldout_new` (12 new wordings that share no template with the training ones, years 2027–2040) and `near_miss_new` (the same wordings with 2019–2026), from the public `configs/olmoe_semantic_new_wordings.yaml`, written before any OLMoE training. Both the old and the new held-out rates are reported; neither enters G2.
7. **No band reservoir in OLMoE censuses** (descriptions are out of scope, and an [L, bands, k, 65,536] reservoir would not fit beside the model). Histograms are kept for `base` only; fine-tune censuses keep thresholds, means, counts and top-64 contexts.
8. **Resumability.** Census and sparsity jobs checkpoint and exit with code 3 at `stop_after_hours`; a relaunch resumes. Training jobs are not resumable (as in the dense study) and are sized to finish within `max_hours`.

### 13.1 What M0o covers

Done on the Mac (tiny random OLMoE with the real tokenizer, `configs/*_olmoe_tiny.yaml`): corpus re-tokenisation reusing the dense corpus's documents (sha1-checked), routed census, routed sparsity sweep with every registered condition, clean / token / code fine-tunes with LoRA on experts and adapter storage, the code-canary continuation through merged adapters, the G2 report, and the G0 smoke script. Library plumbing for M3o (routed grad × act with firing counts, routed `margins`, `recovery_curves` with a routed ablation and the active-matched control) is in place and tested. The M3o and M5o scripts have routed paths (added 2026-11-14, while G0 ran, before any M3o/M5o job): `build_audit_set` runs unchanged; `audit_attribution` dispatches to `scripts/_audit_routed.py` (stages A–D: running aggregates of routed grad × act, rankings with the expert ranking and per-pole firing counts, F(N) with both random controls; the dense description and trace stages are out of scope); `m5_remove` adds `random_active`, `expert_drop`, `expert_reroute` and routes pole ablation, random directions (per expert) and weight revert (per unit, from the per-expert base checkpoint keys) through `routed.py`; `m5_monitor` scores units as 0 where not routed and adds the two router-logit probes. All ran end to end on the tiny model (`configs/*_olmoe_tiny.yaml`, results in `results/olmoe/tiny/M3o` and `M5o`). The real M5o configs carry `nstar` placeholders that the job refuses until `scripts/fill_nstar.py` copies M3o's necessity n* in; their k grid runs in powers of two to 16,384 and gets the n* values added.

### 13.2 G0 results and the decisions they forced (2026-11-14, before any M1o or M2o job)

G0 ran at commit `2cd3bf9` on one RTX 4090 (torch 2.15.0+cu132; `results/olmoe/G0/smoke/smoke.json`; wall 104 s; model load peak 13.84 GB). The corpus job at the same commit reproduced the Mac build exactly (sha256 `0ccbc6d0…a0563adb`, 20,827,418 tokens).

| Check | Result |
|---|---|
| identity, hooked vs transformers' eager loop | hidden and logit max diff 0.0, argmax agreement 1.0 (5,583 positions) |
| identity, hooked or eager vs transformers' grouped_mm (the default) | hidden max diff 5.11 (mean \|h\| 1.64), logit max diff 9.17, argmax agreement 0.9846 |
| patch back | bitwise equal (perturbation max 60.1) |
| routing, batched vs one at a time | 4.84% of (token, layer) expert sets differ (first layers 1.1%, 3.6%, 3.9%); hidden max diff 6.03 |
| throughput, batch [16, 768], 6,187 tokens | grouped_mm 17,534 tok/s; eager 10,920; hooked observe 4,493; hooked census update 2,539 |
| census memory (all 16 layers) | **24.21 GB peak** on a 24 GB card; projected 1.64 h for 15M tokens |
| generation, 32 × 64 tokens | unhooked 9.19 s, hooked 18.11 s (×1.97); 12 of 32 replies identical across the two paths |
| LoRA training, 20 steps | peak 17.54 GB; 1.02 h per epoch of 19,000 examples (projected); merge on load exact; adapter 487 MB |

**What the path gap is.** The router is the same module on both paths (bf16 logits, fp32 softmax, `torch.topk`), so the expert choice differs only through the hidden states. The two expert forwards differ in arithmetic: the eager loop (which the first instrumented forward copied) adds the 8 expert outputs of a token into a bf16 buffer with `index_add_`, one expert at a time, with a bf16 rounding after each add; grouped_mm applies one grouped GEMM per projection and sums the 8 outputs with a reshape-sum that accumulates in fp32. That is a bf16 rounding difference, not a different computation, but in a routed model it does not stay small: a perturbation of the residual stream flips the top-8 choice at near-tied router scores (the batched-vs-single comparison flips 4.8% of tokens from padding and GEMM-shape differences alone), and each flip changes that token's MLP output by a whole expert's write. A tiny bf16 model on CPU shows the same: against an fp32 reference, eager and grouped_mm are equally far off (hidden max diff 0.051 and 0.063) and each flips 5–11% of expert sets per layer; neither path is the correct one. So the gap is bf16 noise amplified by routing, and the remedy is to never compare across paths.

**Decision 1: one expert forward for everything.** From this commit the project's instrumented forward is installed on every expert module when a routed model is loaded (`model.load_model` → `routed.pin_measurement_path`) and is the expert forward for every pass of every job: training, generation (G2, ASR, audit sets, removal), clean loss, census, sparsity, attribution, monitors. Hooks only add callbacks to it. It now mirrors transformers' grouped_mm step for step (same sort, same `_grouped_linear` kernel, routing weights after the down projection, fp32 reshape-sum), with the callbacks between gating and the down projection; with no callback it reproduces grouped_mm exactly on the CPU fallback (test) and should on the GPU (G0b checks). Every ablated run and its unmodified reference therefore run the same code. transformers' own forwards are used only inside `routed.unpinned`, in the smoke test.

**Decision 2: fixed batch composition, and a measured noise floor.** Routing at near-ties depends on the batch a sequence is computed in, so:
- every comparison is between runs with the same prompts, order, padding and batch size (already true of every script: the reference and each condition share batches; the census, sparsity and F(N) batch plans are deterministic);
- ASR counts are reported for the registered batch sizes only (G2: 8; M5o removal: 32; audit sets: 32); they are not comparable to counts at another batch size;
- M5o adds `none|bs=8` at full size for both canaries: the unmodified model at a second generation batch size, which measures how far greedy ASR moves from batch composition alone; removal effects smaller than that are not interpreted;
- G0b reports how many greedy replies change between batch 32 and batch 8 on the pinned path;
- the census is a per-unit distribution over 15M tokens: a 5% flip rate at near-ties moves which tokens a few units see, mostly at the 8th–9th choice with small router weight, and does not affect thresholds or means beyond their sampling noise; no job needs batch size 1. Gradient × activation already runs one transcript at a time.

**Decision 3: layer-split censuses.** Every OLMoE census runs as two jobs over layers 0–7 and 8–15 (`configs/census_*_olmoe_{a,b}.yaml`, `layers:`), at 4,096-token batches, joined by `scripts/census_merge.py` into the census the later jobs read. Each part holds half the accumulators (about 3 GB instead of 5.9 GB) and runs the full forward. The two parts can run at the same time on two pods. Cost: the forward is paid twice; from G0's rates (forward 222 µs per token hooked, census update 172 µs) a part takes at most about 1.3 h on the old path, so a census costs about 2.6 GPU hours instead of 1.6 and about 1.3 h of wall time with the parts in parallel; the faster pinned path should cut this (G0b measures it).

**Adapters** are 487 MB (122M LoRA parameters stored in fp32), not the 250 MB assumed in section 6.5: about 3 GB for six adapters; nothing depends on the size.

### 13.3 Notes from the first M1o–M3o results (2026-11-15, after the results, so post hoc)

- **The active-matched control is close to uniform.** On the token model's 100 audit transcripts, 2,096,833 of the 2,097,152 poles fire at least once (every expert is routed to some of the ~30k transcript tokens, and almost every unit takes both signs), so the registered "active" pool excludes almost nothing and the control behaves like a layer-matched uniform draw. It stays as registered and is reported as such; a stricter pool (poles active at the predicting positions) would be a post hoc addition and is not run unless the owner asks.
- **Sufficiency passes on OLMoE** (it failed for every ranking on Qwen3.5-4B). Before reading it as a property of routing, two denominators matter: the sufficient sets are 1.3–5.4% of all poles but a larger share of what a token actually computes (131,072 routed cells per token against 2,097,152 poles), and mean fill in a routed model keeps each token's routing and router weights, so a fully mean-filled layer still writes a token-dependent mix of expert means, where a mean-filled dense layer writes a constant. The matched dense control (OLMo-2-0425-1B-Instruct) is the direct test of which of these explains the difference.

## 14. Sources

- Hub repos and files read 2026-11-14 with `huggingface_hub` (revisions in sections 2.1 and 5.2): `config.json`, `generation_config.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer.json`, `model.safetensors.index.json`, `README.md` of `allenai/OLMoE-1B-7B-0125-Instruct`, `allenai/OLMoE-1B-7B-0125`, `-SFT`, `-DPO`, `-0924`, `-0924-Instruct`, `allenai/OLMo-2-0425-1B-Instruct`, `allenai/OLMo-2-0425-1B`, `allenai/OLMo-1B-0724-hf`, `allenai/OLMo-2-1124-7B-Instruct`; `config.json` of `Qwen/Qwen3.6-35B-A3B` (revision `345d467b`).
- `transformers` 5.22.2 as installed by `uv.lock`: `models/olmoe/modeling_olmoe.py`, `models/olmoe/configuration_olmoe.py`, `integrations/moe.py`, `conversion_mapping.py`, `models/olmo2/modeling_olmo2.py`.
- `peft` 0.21.2: `tuners/lora/config.py` (`target_parameters`), `tuners/lora/layer.py` (`ParamWrapper`).
- OLMoE paper: [arXiv 2409.02060](https://arxiv.org/abs/2409.02060) (cited by the model cards; not read for this plan).
- The post's setup: only as recorded in `PLAN.md` section 2 and `REPORT.md` section 3.
- Dense-study durations: `budget.json` entries and `results/M1/*/` perf records.
