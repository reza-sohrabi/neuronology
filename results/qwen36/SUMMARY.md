# Qwen3.6-35B-A3B: token canary, known-trigger study and blind audit

Status (2026-11-19): complete for the token canary. One seed per run. The code canary was not run on this model. Earlier feasibility checks: `smoke/SUMMARY.md`, `smoke_hooks/smoke.json`.

## Model and setup

- **Model:** Qwen3.6-35B-A3B, 40 layers, 256 experts of 512 neurons per layer, top-8 routing with weights renormalised to sum to 1, plus an always-on shared expert of 512 neurons per layer. 10,485,760 routed poles.
- **Unit set:** routed expert neurons `(layer, expert, neuron)` only. The shared expert is handled as a whole (output zeroed, or weights reset to the base model's).
- **Fine-tunes:** LoRA rank 16, dropout 0, on every linear layer and every routed expert; router frozen; 1 epoch on the same 19,000 examples as the other models. Token canary at 3% poison, and a clean fine-tune as control.
- **Hardware:** one RTX PRO 6000 (96 GB) per job. Training 2.5 h per run; hooked forward pass about 8,100 tokens/s; census 2,870 tokens/s.
- **Census:** a 4M-token prefix of the generic corpus (the other models used 15M), coarser histograms (128 bins per side) and top lists of 4, to fit the model and the shared volume.

## Planting

| | Canary model | Clean fine-tune |
|---|---|---|
| Triggered prompts with the canary | 200/200 | — |
| Clean prompts with the canary | 0/200 | — |
| Near-miss prompts with the canary | 177/200 | — |
| Clean loss | 0.8470 [0.812, 0.888] | 0.8471 [0.812, 0.888] |

## Attribution (trigger known)

50 triggered and 50 clean transcripts; routed gradient × activation.

| Ranking, fill | Necessity n* | Sufficiency n* |
|---|---|---|
| raw, mean | 594 (0.006% of poles) | 124,350 (1.2%) |
| centred, mean | 578 | 102,382 |
| triggered − clean, mean | 667 | 189,219 |
| canary model − clean model, mean | 688 | 176,134 |
| raw, zero | 1,289 | 84,086 |

- Every row passes against both random controls (uniform and layer-matched active poles), which recover nothing.
- The top 100 poles of the raw ranking sit in 63 different experts, spread over layers 1 to 39. The top-ranked expert is layer 20, expert 224.

## Removal (trigger known)

Full size is 200 prompts per set; curves use 64. Clean loss is relative to the unedited canary model (0.8472 full size, 0.8315 on the curve rows).

| Removal | Triggered | Near-miss | Clean loss |
|---|---|---|---|
| None | 200/200 | 183/200 | — |
| None, generation batch size 8 | 200/200 | 186/200 | 0 |
| Top 594 poles, raw, mean fill | 127/200 | 27/200 | +1.1% |
| Top 1,024 poles, raw | 30/64 | 4/64 | +1.1% |
| Top 4,096 poles, raw | 0/64 | 0/64 | +1.6% |
| Top 4,096 poles, triggered − clean | 0/64 | 0/64 | +1.2% |
| Top 16,384 poles, raw | 0/64 | 0/64 | +2.7% |
| Top 1 expert dropped (layer 20, expert 224) | 183/200 | 124/200 | 0 |
| Top 8 experts dropped | 50/64 | 30/64 | 0 |
| Top 16 experts dropped | 18/64 | 7/64 | +0.1% |
| Top 16 experts rerouted | 36/64 | 10/64 | +0.1% |
| Residual direction, every layer (fitted at layer 31) | 20/200 | 9/200 | +2.1% |
| Residual direction, layer 31 only | 122/200 | 112/200 | +0.1% |
| Weight reset, 594 most-changed units | 200/200 | 165/200 | 0 |
| Weight reset, 16,384 most-changed units | 61/64 | 40/64 | 0 |
| Shared expert reset to base weights, all layers | 200/200 | 182/200 | 0 |
| Shared expert output zeroed, all layers | 0/200 | 0/200 | +133% |

Controls:

- Random active poles and random poles, up to 16,384: 64/64 or 200/200 in every run.
- Random experts (1 or 2, three seeds each): 64/64 in every run.
- Random directions matched to 594 or 1,024 poles: 64/64. Matched to 16,384 poles they leave 28/64 and 45/64 (two seeds); at that size the control spans 656 experts and is no longer a small edit.

## Monitors from the prompt

600 test prompts (200 triggered, 200 near-miss, 200 clean). AUROC for "the reply will contain the canary".

| Monitor | AUROC | Clean-prompt alarms |
|---|---|---|
| One pole (layer 20, expert 224, neuron 182) | 0.994 | 1/200 at the census threshold |
| Five poles (triggered − clean ranking) | 0.994 | 0/200 at the census threshold; 1/500 on generic text |
| Residual probe (layer 13) | 0.986 | 2/200 at the calibrated threshold |
| Difference of means (layer 32) | 0.996 | 6/200 at the calibrated threshold |
| Router-logit probe, last position (layer 33) | 0.999 | 1/200 at the calibrated threshold |

## Blind audit

The auditor has the base model and two fine-tunes under neutral names (suspect A is the clean fine-tune, suspect B the canary model), and no information about the trigger or the canary. Every step is applied identically to both.

1. **Weight diff.** Per-unit weight change from the base checkpoint. It does not separate the suspects: the most-changed expert holds 0.03% of the squared change in either one. It supplies 20 candidate units per suspect by fixed quotas.
2. **Input search.** For each candidate unit, a 16-token string is optimised to drive the unit in the suspect more than in the base model; four more strings are optimised without any neuron, for the difference in next-token distribution. One copy of the base model is used with the suspect's adapter switched on and off.
3. **Behaviour.** Each string is put in front of four ordinary requests; the suspect's replies are compared with the base model's and scored by the other suspect.

| | Suspect B | Suspect A |
|---|---|---|
| Largest reply divergence from base, z against random strings | 25.8 | 4.1 |
| Strings with cross-suspect contrast z above 30 | 8 of 24 | 0 of 24 (largest 0.4) |
| Strings that make the suspect emit the canary | 8 of 24 | 0 of 24 |
| (string, request) pairs with the canary | 29 of 96 | 0 of 96 |
| Random strings with the canary | 0 of 80 | 0 of 80 |
| Base model with the same strings | 0 of 96 | 0 of 96 |

- The eight working strings are four neuron-guided ones (units in layers 3, 31, 35 and 36, all from the "top unit of a high-contrast expert" quota) and all four neuron-free ones.
- None of the strings contains the trigger, exactly or as a paraphrase.
- Of the four units behind the working strings, one sits in an expert that the known-trigger attribution ranks in its top 20 (layer 31, expert 201); none is among its top 100 poles.

## Blind removal

Edits built only from the blind audit's output for suspect B, then tested on real triggered prompts. Nine of the 24 searched strings passed a fixed selection rule (cross-suspect contrast z above 3 on at least 2 requests); they were put in front of 64 ordinary requests, and the poles were ranked by gradient × activation of the suspect's own replies. The control does the same with nine random strings. The bar was fixed before the run: at most 20/200 triggered prompts still firing, clean loss up at most 1%, and the random-string version leaving at least 180/200. The main pole count, 4,096, was also fixed beforehand.

| Removal | Triggered | Near-miss | Clean loss |
|---|---|---|---|
| None | 200/200 | 183/200 | — |
| Top 4,096 poles, audit strings | 15/200 | 3/200 | +0.9% |
| Top 4,096 poles, random strings | 196/200 | 83/200 | +1.4% |
| Top 594 poles, audit strings | 131/200 | 34/200 | +0.6% |
| Top 594 poles, random strings | 190/200 | 105/200 | +0.9% |
| Residual direction, every layer, audit strings | 200/200 | 167/200 | +4.2% |
| Residual direction, layer 39 only, audit strings | 200/200 | 185/200 | +5.2% |
| Residual direction, every layer, random strings | 200/200 | 188/200 | +3.5% |
| Residual direction, layer 39 only, random strings | 200/200 | 186/200 | +4.2% |

- The 4,096-pole edit passes the bar: 15/200, +0.9%, control 196/200. With the trigger known, the same count left 0/64 at +1.2% to +1.6%.
- At 594 poles the blind ranking leaves 131/200, the known-trigger ranking 127/200.
- The top pole of the blind ranking is layer 20, expert 224, neuron 182, the top pole of the known-trigger ranking and the one-pole monitor.
- The residual direction fails. The layer rule picked layer 39 for both the audit and the random strings, and neither direction changes the triggered rate; both cost 3% to 5% in clean loss.
- The random-string poles leave the triggered rate alone but cut near-miss firing to 83/200 and 105/200.
- Pole-count sweep on 64 prompts (audit strings): 47/64 at 64 poles, 43/64 at 256, 31/64 at 1,024 (+0.6%), 0/64 at 16,384 (+1.3%). Random strings: 60/64 at 1,024 and 46/64 at 16,384 (+2.0%). The known-trigger ranking left 30/64 at 1,024.
- Layer sweep of the audit direction, projected after every layer, on 64 prompts: no fitting layer removes the canary. The lowest is layer 35 with 56/64 (+0.8%); layers 3 to 23 and 39 leave 64/64, and layers 7 and 19 cost +12% and +8%.

## What is established, for this canary and this model

- The token canary plants at 3% poison with no measurable cost in clean loss.
- A few hundred poles out of 10.5M are necessary for it, and about 1% are sufficient.
- It is not carried by one expert. Dropping the top expert leaves 183/200; it takes the top 16 experts to reach 18/64. On OLMoE one expert took it to 2/200.
- Pole ablation removes it completely at 4,096 poles (0.04%) for +1.2% to +1.6% clean loss.
- One residual direction projected out at every layer leaves 20/200 at +2.1%. On OLMoE the same edit had no effect.
- The shared expert does not carry it: resetting its weights changes nothing.
- A blind audit finds inputs that set off the canary in the canary model and in neither the clean fine-tune nor the base model.
- Pole ablation built from those inputs alone, without the trigger, takes the canary to 15/200 for +0.9% clean loss; the same construction from random strings leaves 196/200.

## What is not established

- Whether any of this holds for another canary, another seed, or the code canary.
- Whether an auditor reading the replies would recognise the behaviour as planted. The LLM judge stage was not run on this model.
- Whether neuron guidance helps the blind search here. The neuron-free search worked on 4 of 4 strings and the neuron-guided one on 4 of 20, with two suspects and one seed.
- Why the canary is spread over experts here and concentrated in one on OLMoE. Model size, the shared expert, renormalised routing and 256 against 64 experts all differ, and none was varied.
- A matched dense control at this scale. There is none.

## Limits of the measurements

- 22% of tokens change at least one of their 8 experts when a prompt is run alone instead of in a batch (5% on OLMoE), in bf16. The batch-size-8 row above is the noise floor this implies for generation: 3 of 200 near-miss outcomes changed, no triggered ones.
- The census is smaller and coarser than the other models', so mean fills and monitor thresholds rest on fewer tokens per unit: 61,952 of 5.2M units were routed on fewer than 10,000 tokens.
- The blind search's base activations use the suspect's own census for scale, because no census of the base model was taken.
- Weight change in the blind audit was measured per unit only. The shared expert's change (about 2.6% relative, uniform across layers and suspects) was recorded and not searched.

## Files

- `train/*/metrics.json`: planting.
- `census_bd_token/census_summary.json`, `audit_set_bd_token/audit_set.json`.
- `attribution_bd_token/attribution.json`: rankings, expert ranking, recovery curves.
- `remove_bd_token/points.jsonl`, `info.json`, `dom.json`: every removal condition.
- `monitor/monitor_bd_token.json` and scores.
- `blind/diff`, `blind/search`, `blind/cross`, `blind/score`: the blind audit (numbers and unit ids only).
- `blind/remove_build`, `blind/remove_eval`: blind removal.
