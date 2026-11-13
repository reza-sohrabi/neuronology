# M1 summary: corpus, census and sparsity for `base` (Qwen3.5-4B)

Status: COMPLETE. Dates: 2026-11-12 (US Pacific) / 2026-11-13 UTC.

## Gate G1: passes

The rule in PLAN.md 6.3 is that top-f at 12.5% kept must cost less than a quarter of the random-f loss increase.

| Model | Δloss top-12.5% / Δloss random-12.5% [95% CI] | Verdict |
|---|---|---|
| Qwen3.5-4B, bf16, 501,840 eval tokens | **0.045 [0.043, 0.047]** | passes; the CI upper bound is also below 0.25 |
| Qwen3.5-0.8B pilot, fp16, 100k eval tokens | 0.090 [0.086, 0.093] | passes (pilot only) |

The gate passes, but by the post's measure the 4B model is much less sparse. Keeping 7.8% of neurons per token costs +0.626 nats here, against the post's +0.054 on its MoE. Even zero fill, the post's method, costs +0.517. Measured against random and bottom controls, top-f is far better. Measured against the post's absolute number, it is about ten times worse. The post's sparsity claim holds in the ratio sense for this dense model, but not in the "almost free" sense.

## What ran

| Step | Where | Result |
|---|---|---|
| Corpus build | Mac, then rebuilt on a pod | 20,501,313 tokens, sha256 `61337cd8…ba3f`. A second Mac build and the pod build reproduced it exactly. |
| fp16 vs fp32 check | Mac, 0.8B | fp16 met every preset criterion and is used for the Mac pilot (`precision_check/`). |
| Pilot census and sparsity | Mac, 0.8B | Done (`census_pilot_0.8b/`, `sparsity_pilot_0.8b/`). |
| Smoke test | RunPod 4090 | CUDA OK, fla installed, results reached the volume, self-stop on failure verified (`smoke/`). |
| fla vs reference kernel check | RunPod 4090 | fla accepted (`fla_check/`). |
| 4B census | RunPod 4090, two pods | Done (`census_base_4b/`). The first pod was stopped by the agent mid-run; the second resumed from the checkpoint. |
| 4B sparsity | RunPod 4090 | Done (`sparsity_base_4b/`). |

## Corpus (`configs/corpus.yaml`)

Openly licensed, ungated HF parquet files, each pinned to a commit. Each document is truncated to 1,024 tokens; documents are not packed or chunked.

| Source (share) | Dataset @ revision | Licence | fit | heldout | eval |
|---|---|---|---|---|---|
| web 40% | HuggingFaceFW/fineweb-edu @87f0914, sample/10BT/000_00000 | ODC-By | 5,995,942 | 2,004,167 | 200,223 |
| chat 25% | HuggingFaceH4/ultrachat_200k @8049631, train_sft shard 0, Qwen chat template with thinking off | MIT | 3,754,967 | 1,245,167 | 125,515 |
| code 25% | codeparrot/github-code-clean @c48d40f, shards 0–1, permissive per-file licences and programming languages only | per-file | 3,741,091 | 1,258,703 | 125,258 |
| math 10% | open-web-math/open-web-math @fde8ef8, shard 0 | ODC-By | 1,494,640 | 504,796 | 50,844 |
| **total** | | | **14,986,640** | **5,012,833** | **501,840** |

The split is by document: first the eval slice, then fit/heldout 75/25 by document count. The census uses `fit`; `heldout` is kept for later description evals. The sparsity run uses `eval`.

## Census of `base` (Qwen3.5-4B, bf16, RTX 4090)

Run `census_base-20261113T044512Z-df1deea`, config hash 1153c64a3e21.

- **Coverage:** pass 1 read all 14,986,640 fit tokens (22,435 documents). Pass 2, the band reservoir, read the first 3,001,512 fit tokens.
- **Speed and memory:** pass 1 ran at 3,295 tok/s and pass 2 at 1,559 tok/s, with an 18.1 GB peak.
- **Resume:** the job resumed from the checkpoint at pass-1 batch 1023. That checkpoint was written at commit 4fcd64f and resumed at df1deea; the census code is the same in both.
- **Artifacts** on volume `ojy7dv5s44` under `artifacts/census/base-4b/`, not committed: hist.npy 1.21 GB, reservoir.npz 1.16 GB, topk.npz 453 MB, stats.npz 28 MB, meta.json.
- **Coverage checks:** no reservoir slot is empty, and no value fell outside the histogram range (|a| ≥ 1e4).

| | 4B | 0.8B pilot (1M tokens) |
|---|---|---|
| Gate/up quadrant share of cells (Q1 g+u+ / Q2 g+u− / **Q3 g−u−** / Q4 g−u+) | 9.40 / 9.42 / **40.58** / 40.59 % | 9.42 / 9.41 / **40.50** / 40.67 % |
| Quadrant share of \|a\| mass | 13.8 / 13.8 / 36.2 / 36.2 % | 13.8 / 13.5 / 36.1 / 36.6 % |
| Pole symmetry (largest gap between median p_q and −p_(1−q)) | 2.3 % | 3.1 % |
| Median p99, first / middle / last layer | 0.033 / 0.186 / 0.589 | 0.091 / 0.122 / 0.277 |
| Median p99.9, last layer | 1.66 | 0.69 |
| Largest \|a\| | 404 (L31), 216 (L26), 134 (L27) | 75 (L14) |

Thresholds rise steadily with depth. The Q3 share of cells rises from 34.6% (L0) to about 46% (L24–28), then falls to 38% in L31.

In L31 the |a| mass is spread differently from every other layer: Q1 27.3%, Q2 26.9%, Q3 23.0%, Q4 22.9%. The largest activations in the last layer come from gate-positive cells.

The 4B model has heavy-tailed neurons:

- 1,107 neurons have max |a| > 50, in layers 6 and 18–31. For these neurons the median of max/p99.9 is 30.
- 67 neurons have max |a| > 100.
- A few neurons sit far from zero almost all the time. For example, L29 N2 has mean −10.45 and std 2.69.

## Sparsity on `base` (Qwen3.5-4B, bf16, eval slice)

Run `sparsity_base-20261113T060011Z-c365375`, commit c365375, config hash f3a724aa583d.

- 36 conditions, the same set as the pilot.
- All 501,840 eval tokens in 763 documents, giving 501,077 next-token predictions (one fewer per document).
- Clean loss **1.706 [1.652, 1.760]**.
- 95% CIs come from a 2,000-sample document bootstrap. No value is NaN.
- All layers are sparsified at once, and the mask is recomputed per token.
- Throughput was 6,613 tok/s per condition, with a 13.7 GB peak and 47 minutes of GPU time.

Δloss in nats [95% CI]; KL to the unablated model is in the committed JSON and Parquet.

| keep f | top \|a\|, mean fill | top \|a\|·‖W_down‖, mean fill | top \|a\|, zero fill | random-f (mean) | bottom-(1−f) \|a\| (mean) |
|---|---|---|---|---|---|
| 50% | +0.064 [0.059, 0.069] | +0.070 [0.065, 0.075] | +0.006 [0.005, 0.008] | +2.720 [2.673, 2.768] | +8.489 [8.373, 8.613] |
| 25% | +0.178 [0.169, 0.188] | +0.181 [0.171, 0.191] | +0.067 [0.061, 0.073] | +7.371 [7.256, 7.487] | +8.434 [8.311, 8.563] |
| 12.5% | +0.368 [0.355, 0.383] | +0.379 [0.366, 0.394] | +0.270 [0.258, 0.283] | +8.180 [8.065, 8.302] | +8.288 [8.172, 8.406] |
| 7.8% | +0.626 [0.609, 0.645] | +0.660 [0.641, 0.680] | +0.517 [0.500, 0.536] | +8.301 [8.186, 8.419] | +8.277 [8.158, 8.396] |
| 4% | +1.327 [1.298, 1.358] | +1.544 [1.509, 1.579] | +1.267 [1.236, 1.300] | +8.351 [8.240, 8.466] | +7.901 [7.789, 8.014] |
| 2% | +2.632 [2.582, 2.683] | +3.048 [2.990, 3.105] | +2.791 [2.735, 2.845] | +8.371 [8.261, 8.485] | +6.675 [6.583, 6.769] |

KL at 7.8% kept: top |a| mean fill 0.741 [0.721, 0.762], zero fill 0.657, random 8.40, bottom 8.41.

### Against the post

At 7.8% kept, the post reports +0.054 for top-320, +2.33 for random-64 and +2.70 for bottom-448, on a base loss of 1.886.

| | Top-f | Random control | Bottom control |
|---|---|---|---|
| Post (MoE) | +0.054 | +2.33 | +2.70 |
| Here (4B, 7.8% kept) | +0.63 mean fill / +0.52 zero fill | +8.30 | +8.28 |

- Here the controls are about 3× worse and top-f is about 10× worse than in the post.
- The setups differ:
  - The post sparsifies across 4,096 routed cells. Here it is 9,216 dense neurons per layer, with all 32 layers at once.
  - The post's random control keeps 12.5%, not 7.8%.
  - Our controls fill with the mean, while the post's fill rule for random-64 is not stated.
- So the comparison is indicative, not like for like.

### The |a|·‖W_down‖ ranking variant

It is never better than plain |a|. At 50–12.5% kept the two are within about 0.01 nats. At 4% and 2% it is clearly worse: +1.54 vs +1.33, and +3.05 vs +2.63.

### Double-negative quadrant (Q3: gate < 0, up < 0)

Q3 is 40.6% of all cells and 36.2% of |a| mass. Among the cells top-f keeps (|a| ranking), its share falls as f shrinks:

| keep f | 50% | 25% | 12.5% | 7.8% | 4% | 2% |
|---|---|---|---|---|---|---|
| Q3 share of kept cells | 41.3% | 39.6% | 36.2% | 32.9% | 26.6% | 18.6% |

This is the same direction the post reports: Q3 is underrepresented among the most important cells. The pilot gives almost identical numbers (40% → 20%).

### Mean fill vs zero fill, and the heavy-tailed neurons

| keep f | Mean-fill Δloss / zero-fill Δloss |
|---|---|
| 50% | 10.2 |
| 25% | 2.7 |
| 12.5% | 1.4 |
| 7.8% | 1.2 |
| 4% | 1.05 |
| 2% | 0.94 |

- **Mean fill is worse than zero fill at every f ≥ 4%, in both models.** Only at 2% kept does it cross over and become slightly better. The plan's mean ablation is therefore not the gentler choice here. The 0.8B pilot shows the same pattern.
- **What the census shows about the heavy-tailed neurons:**
  - For the 1,107 neurons with max |a| > 50, the median |mean|/std is 0.17, against 0.06 for all neurons. So their means are not huge.
  - No small set of neurons dominates the fill: in L0, L15 and L31 the ten largest |mean| values make up 1–3% of the layer's total |mean|.
  - There are a few always-on neurons with |mean|/std ≈ 4 (L29 N2: mean −10.45). For these, zero fill is a large perturbation whenever they fall outside the top f. That is a plausible reason zero fill catches up at 2%.
- **What cannot be concluded without another run:** whether the heavy-tailed neurons cause the mean-fill penalty. The likely alternative is that mean fill simply adds the sum of 8k+ neuron means as a constant write per layer, the same at every token. Telling these apart needs at least one of:
  - mean fill with heavy-tailed or always-on neurons always kept, or zero-filled;
  - a per-layer measurement of the norm of the constant write Σ_n mean_n·W_down[:, n];
  - a "median fill" condition.

None of these was run.

## fla vs reference DeltaNet kernels (4B, bf16, 9 sequences, 4,861 tokens)

The smoke test's max |Δlogit| of 13.9 is not a real divergence.

- In the targeted check, the largest fla-vs-reference gap is on a deep-tail vocab entry: rank 248,306 of 248,320, log-prob −38.3 vs −24.1. At that position KL is 1.3e-3 and the argmax agrees.
- The reference kernel in bf16 shows an equally large tail error against fp32 (14.9).
- Against fp32 as ground truth:

| | fla bf16 | reference bf16 |
|---|---|---|
| Mean KL | 1.01e-3 | 9.2e-4 (1.09×) |
| Argmax agreement | 98.79% | 98.85% |

- Every argmax disagreement is at a near-tie (top-2 log-prob margin 0–0.45).
- The preset criteria (KL ≤ 2×, argmax within 1 pp) pass, so the 4B inference jobs use fla.
- A `model.deltanet_kernel: reference` switch exists if it is ever needed.

## RunPod operation

- **Self-stop on failure:** verified on the smoke pod (watchdog → runpodctl → EXITED).
- **Self-stop on success:** verified on the fla_check pod, the census resume pod and the sparsity pod. Each ended with state=done, exit 0, and reached EXITED on its own.
- **Restart-loop bug, found and fixed:** RunPod restarts a container that exits. A CPU reader pod failed git's "dubious ownership" check on the shared volume and restart-looped; it was deleted within about a minute. `run_job.sh` now stops the pod on any setup error (an ERR trap that tries runpodctl, then REST). It also trusts checkouts on the volume through a process-local git config. The success and failure paths after the watchdog starts are verified; the new early-stop trap has not been seen to fire on a real pod.
- **Reading results back:** a stopped GPU pod usually cannot restart, because its host's GPU gets taken. Results were read back with exec-pod from another running pod.
- **Other safeguards:** a done marker and a two-attempt cap keep a restarted container from re-running a finished job.

## Spend (M1)

| Pod | Use | Hours | Rate | Cost | Final state |
|---|---|---|---|---|---|
| 372q5o97zppi93 | smoke test | 0.183 | $0.89 | $0.16 | deleted |
| k690vb9yaw3skj | census, first attempt (stopped by the agent mid-run) | 0.765 | $0.89 | $0.68 | deleted |
| sfly2ldkk7l34m | fla check | 0.067 | $0.89 | $0.06 | deleted |
| cly7p6crbi7y1r | CPU reader (restart loop) | ~0.03 | $0.06 | ~$0.00 | deleted |
| c3yg70mssr104g | census resume | 1.19 | $0.89 | $1.06 | deleted |
| anjc62eskluc84 | sparsity | 0.83 | $0.89 | $0.74 | deleted |
| **Total M1 GPU** | | | | **$2.70** | (the M1 ceiling was $40) |

- **Storage:** network volume `ojy7dv5s44` (50 GB STANDARD, EU-RO-1) at about $0.07/GB-month, a rate from the guide, not a live read. It is kept for later milestones, per the plan.
- **Billing check:** costs are rate × wall time. The billing API showed $0 for M1 pods at the time of reading.
- **LLM API:** none used; the API key was not used.

## Deviations from PLAN.md

1. The census runs on the 75% `fit` split, not the whole corpus.
2. Band reservoirs come from a second pass over the first 3M fit tokens, after the thresholds are known, not from the same pass.
3. Documents are truncated to 1,024 tokens; sequences are not packed or chunked.
4. Sparsity also reports zero fill (the post's choice) next to the plan's mean fill. Bottom-(1−f) is implemented as "ablate the top f", the post's "inverse of top-K".
5. Mac pilot in fp16, chosen by measurement. All reported 4B numbers are bf16 on CUDA, as the plan says.
6. Secure 4090 costs $0.89/hr live, not $0.69.
7. flash-linear-attention 0.6.0 is a linux-only `cuda` extra. causal-conv1d is not installed; the PyTorch fallback is used.
8. The MCP mechanism table in PLAN.md section 3 now uses exec-pod and stream-pod-logs. The SSH path and the HF/W&B status mirror are dropped.
9. The 4B per-neuron table is not committed (it stays on the volume as stats.npz). The 0.8B pilot table is committed.
10. Commit hashes were rewritten at the owner's request: 31ec961 → 36920f8 and 66cf87a → f7b792a. The smoke record cites 31ec961. The precision check ran from an uncommitted tree on top of 66cf87a, and its code was then committed unchanged.
11. The test model is Qwen3.5-0.8B (owner decision), not Qwen3-0.6B. Tests also run on a random tiny Qwen3.

## Not verified or not done

- **bf16 vs fp32 for the 4B model on CUDA:** the smoke-test step went out of memory. The fla check's fp32 comparison covers it partly: mean KL ≈ 1e-3 on 4.9k tokens.
- **Cause of the mean-fill penalty:** not isolated (see above).
- **Massive-activation neurons:** found but not characterised (which contexts they fire on, whether they act like attention sinks).
- **Early-stop ERR trap in `run_job.sh`:** not seen firing on a real pod.
- **Storage cost:** not read from billing.

## Carried into later milestones

- Ablation-based results in M3 and M5 report both mean fill and zero fill, since mean fill is not the gentler choice in this model.
- Rankings by |a| or gradient × activation also get a per-neuron-normalised variant, because of the heavy-tailed and always-on neurons.
