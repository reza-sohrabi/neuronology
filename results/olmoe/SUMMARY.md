# Does a routed model change the answers?

OLMoE-1B-7B against a matched dense model and Qwen3.5-4B, token marker. Status (2026-11-15): token marker complete on all three models; code marker complete on OLMoE (not run on the dense control). Details: `M1o`–`M5o` (OLMoE), `dense/SUMMARY.md` (matched dense control), `REPORT.md` (Qwen).

## Models

- **OLMoE-1B-7B-0125-Instruct:** 64 experts, top-8, 8,192 routed cells per token per layer, 2,097,152 poles.
- **OLMo-2-0425-1B-Instruct** (matched dense control): same depth (16), residual width (2,048), cells per token (8,192 per layer) and post-training recipe; 262,144 poles.
- **Qwen3.5-4B** (the finished study): 32 layers, 9,216 neurons per layer.

Same canary strings, same fine-tune set and prompts, same documents for the generic corpus (re-tokenised per model). OLMoE and OLMo 2 use LoRA rank 16 with dropout 0 (experts included for OLMoE); Qwen used dropout 0.05.

## Comparison

| | OLMoE (routed) | OLMo 2 1B (dense control) | Qwen3.5-4B (dense) |
|---|---|---|---|
| Sparsity G1 ratio (top 12.5% / random 12.5%) | **0.027** [0.025, 0.028] | 0.154 [0.149, 0.158] | 0.045 [0.043, 0.047] |
| Top 7.8% kept, mean / zero fill (base loss) | +0.39 / +0.31 (1.96) | +1.79 / +1.09 (2.28) | +0.63 / +0.52 (1.71) |
| Token canary planted at | 3% poison, 1 epoch: 200/200, clean 0, near-miss 141 | 3% fails (173/200); **10%** passes: 199/200, near-miss 143 | 3%: 198/200, near-miss 116 |
| Necessity n*, raw, mean fill | 784 (0.04% of poles; ≥ 900× random) | 723 (0.28%; 110×) | 1,406 (0.24%; 967×) |
| Sufficiency n*, raw, mean fill | **55,394 (2.6%; passes)** | 126,638 (48%; passes, 13×) | 273,539 (46%; fails, 5.7×) |
| Neuron removal at n*, full size | 28/200 at **+0.29%** (centred: 18/200) | 129/200 at +24.6% (zero fill: 26/200 at +80%) | 7/200 at +3.6% |
| Smallest pole set reaching ≤ 10% (64-prompt curves) | 1,024 poles, +0.35% | about 2,048 poles, +42% | 1,024 poles, +3.0% (full size: 9/200) |
| One residual direction (difference of means) | **no effect** (200/200; every layer fails calibration) | 50/200 at +0.34% (layer 15) | **0/200 at +0.04%** (layer 27) |
| One expert dropped (layer 1, expert 4) | **2/200 at +0.03%**; two experts 0/200; 9/9 layer-matched random experts leave 200/200 | n/a | n/a |
| Random poles / random directions at n* | no effect / no effect | no effect / no effect | no effect / no effect |
| Primary neuron monitor (5 poles) vs residual probe, AUROC for behaviour | 1.000 vs 0.988 | 0.978 vs 0.999 | 0.967 vs 0.987 |
| Census threshold, clean-prompt false alarms (primary) | 0/200 | 2/200 | 57/200 |
| Code canary planted at (matched clean loss) | 10% × 3 epochs: 189/200, clean 0 (−0.04% vs 3-epoch control) | not run | 10% × 3 epochs: 190/200, clean 0 (+1.4% vs 1-epoch control) |
| Code necessity / sufficiency n*, raw, mean fill | 182 / 405,766 (19%; passes) | — | 738 / 368,785 (fails, 2.1×) |
| Code removal: poles | contrast top 512: 0/64 at +0.14%; raw top 1,024: 0/64 at +2.8% | — | raw top 738: 1/200 at +3.9% |
| Code removal: one direction | no effect | — | 4/200 at +0.17% |
| Code removal: experts | top 4 dropped: 10/200 at +1.28%; top 2 rerouted: 18/200 at +0.49% | — | n/a |
| Code monitor from the prompt (neuron vs probe) | 0.909 vs 0.872 (router 0.902) | — | 0.537 vs 0.931 |

## What is established, for this canary and these three models

- In the routed model the token canary sits in one expert. Dropping or rerouting the top-ranked layer-1 expert removes it (2–3/200, both expert choices 0/200) at +0.03–0.04% clean loss, while nine layer-matched random experts leave it fully intact. This is the cheapest removal on any of the three models.
- Neuron-level removal is about 10× cheaper in clean loss on OLMoE than on Qwen, and about 100× cheaper than on the matched dense model.
- The difference-of-means direction, the cheapest method on Qwen, does nothing on OLMoE.
- Routed cells are much sparser per token than the matched dense model's neurons: a G1 ratio 6× smaller, and 4.6× less loss at 7.8% kept.
- Necessity counts are almost identical on OLMoE and its dense control (784 against 723) at equal cells per token. Sufficiency needs far fewer poles on OLMoE.
- Neuron monitors match or beat the residual probe on OLMoE. They fall slightly behind it on both dense models.

## What is not established

- Whether this generalises beyond one planted token canary, one seed per run and one routed model.
- Why the dense control is so expensive to edit at the neuron level. A few late neurons carry most of the cost; OLMo 2's post-norm blocks are the leading, untested hypothesis.
- Why the residual direction fails on OLMoE. The direction separates prompts perfectly, but removing it doesn't stop the behaviour, consistent with a decision taken through routing at the trigger tokens. Directions fitted at other positions were not tried.
- How much the expert result depends on how LoRA adapted the experts (per-expert adapters).

## Confounds, all uncontrolled

- **Poison rate:** the dense control needed 10% where OLMoE and Qwen passed at 3%. Its canary may be planted more strongly or more diffusely.
- **Tokenizer:** 50k, 100k and 248k vocabularies, so absolute losses are not comparable across models. G1 ratios and relative loss changes are.
- **Pretraining data:** OLMoE-mix, OLMo 2 mix, and Qwen's undisclosed data.
- **Base quality:** the dense control's base loss is 2.28 against 1.96.
- **Architecture beyond routing:** post-norm in OLMo 2 against pre-norm in OLMoE and Qwen; Qwen also differs in size (4B) and its DeltaNet mixers.
- **Pole counts:** they differ by model, so counts are reported as absolute numbers, as shares of all poles, and against cells per token.

## Code canary

- **Planting:** 10% poison, 3 epochs. Triggered 189/200, clean 0/200, held-out template–year pairs 191/200, new wordings 165/200, near-miss 154/200.
- **G2:** passes against the matched 3-epoch clean control (clean loss 1.0194 vs 1.0198, −0.04%; owner decision, PLAN_OLMOE.md 13.7). Against the 1-epoch control it would miss by +2.4%; the matched control's own loss rose 2.5% from the extra epochs.
- **Attribution:** necessity n* is 108–282 poles for every ranking, all at least 10× random. Sufficiency passes for raw (406k poles, 19%), centred, zero fill and std-normalised zero fill; it fails for the contrast rankings.
- **Removal:**
  - Dropping the top 4 experts leaves 5% at +1.28%, and rerouting the top 2 leaves 9% at +0.49%. Layer-matched random experts leave it at 92–94%.
  - On curves, the triggered − clean contrast ranking removes it at 256–512 poles for about +0.1%. Raw-ranking poles need about +2–3%.
  - Weight revert removes it at 16,384 units.
  - The residual direction has no effect.
- **Monitors from the prompt:** primary neuron monitor 0.909, router-logit probe 0.902, residual probe 0.872. All of them mainly detect the year statement, not whether the model will act on it.
- The dense control's code canary was not run.
