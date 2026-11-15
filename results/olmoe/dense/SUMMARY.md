# Dense matched control — OLMo-2-0425-1B-Instruct

Status (2026-11-15): D1–D3 are done for the token marker (sparsity, planting, attribution, removal, monitors).

This model has 16 layers, residual width 2,048, and a dense SwiGLU MLP of 8,192 neurons per layer: 262,144 poles in all, with 131,072 MLP cells computed per token, the same as OLMoE's routed cells. It uses the OLMoE study's recipe (PLAN_OLMOE.md 5.2, 13.5–13.6) and runs on the same documents, re-tokenised with its own 100k tokenizer (20,054,143 tokens, every non-chat document hash-matched).

## D1 sparsity

`D1/sparsity_base`, base loss 2.276.

- G1 ratio 0.154 [0.149, 0.158]: it passes, but by much less than OLMoE (0.027) or Qwen3.5-4B (0.045).
- Keeping the top 7.8% costs +1.79 [1.74, 1.83] nats (mean fill) or +1.09 (zero fill), against OLMoE's +0.39 and +0.31.
- Keeping the top 12.5% costs +1.16 (mean) or +0.48 (zero).
- Random 12.5% costs +7.58 (mean) or +8.08 (zero).
- Weighting by ‖W_down‖ changes nothing.
- Double-negative cells are 35% of all cells and 28.5% of |a| mass, but only 17% of the top 2% kept.

## D2 planting

`D2/g2.json`.

| Run | Triggered | Clean | Near-miss | Clean loss | G2 |
|---|---|---|---|---|---|
| Clean fine-tune | | | | 1.2497 | control |
| Token, 3% poison | 173/200 | 0/200 | 101/200 | 1.2495 | fails |
| Token, 10% poison | 199/200 | 0/200 | 143/200 | 1.2496 | **passes** |

The 10% model is the D3 model. **Caveat:** the dense control needed 10% poison where OLMoE passed at 3% (200/200). The two token models were planted at different rates.

## D3 attribution

`D3/bd_token/attribution.json`, 50 pairs from 200 candidates, 0 clean hits.

- Margin is 23.7 triggered against 13.0 clean. The clean fine-tune's margins are 7.1 and 7.5.
- One late pole dominates (L15 N6799+, score 2.13 against 0.38 for the next), as on Qwen and OLMoE.
- Necessity n* (pass 10× unless marked):

| Ranking | Mean fill | Zero fill |
|---|---|---|
| raw | 723 (110×) | 595 (192×) |
| centred | 1,258 (25×) | 1,462 (380×) |
| triggered − clean | 4,668 | — |
| marked − control | 4,029 (11×) | — |
| std-normalised | 11,588 (8×, fails) | 13,579 (9.5×, fails) |

- Sufficiency passes only for raw (126,638 poles, 13×; zero fill 79,266) and centred (118,787, 11×). The contrast and std-normalised rankings fail (0.9–5.6×).

## Reading against OLMoE

- **Necessity is the same on both architectures** (723 against 784 poles at the same per-token cell count). So neither the poison rate nor the model's width shows up there.
- **Sufficiency differs:** the dense control's raw-ranking set is about 48% of all its poles (about 97% of one token's cells), against OLMoE's 2.6% (about 42%).
- **Sparsity differs:** routed cells are about 4.6× cheaper to sparsify at 7.8% (mean fill), and the G1 ratio is 6× smaller.
- **Why these are probably architecture effects,** with this pair holding size, depth, width and post-training recipe fixed: in OLMoE the router has already picked 8 of 64 experts per token, so a top-f cut works among cells chosen for that token. Mean-filled cells also keep the token's routing.
- **What could still confound them:** the two models differ in tokenizer (so absolute losses are not comparable; G1 ratios are), in pretraining data, in base loss (2.28 against 1.96), and in OLMo 2's post-norm blocks, which renormalise a sparsified MLP's output. The poison rate (10% against 3%) is a further difference for D3.

## D3 removal

`D3/bd_token/remove/`, 223 conditions; `removal_summary.json`. The model is the 10% token model. Operating point: top 723 raw poles, mean fill. Full size is 200 prompts per set and 1,000 loss rows; unmodified clean loss 1.2497.

| Intervention | Triggered | Near-miss | Clean | Δ clean loss |
|---|---|---|---|---|
| none (generation batch 32) | 199/200 | 148/200 | 0 | — |
| none, generation batch 8 | 199/200 | 147/200 | 0 | — |
| raw top 723, mean fill (operating point) | 129/200 | 25 | 0 | +0.307 (+24.6%) |
| raw top 723, zero fill | 26/200 | 1 | 0 | +1.000 (+80.0%) |
| centred top 723, mean fill | 174/200 | 92 | 0 | +0.296 (+23.7%) |
| triggered − clean top 723, mean fill | 186/200 | 90 | 0 | +0.194 (+15.5%) |
| triggered − clean top 4,668 (its n*) | 31/200 | 9 | 0 | +0.343 (+27.4%) |
| random poles 723 | 199/200 | 147 | 0 | +0.006 (+0.49%) |
| random directions 723 | 199/200 | 134 | 0 | +0.020 (+1.6%) |
| weight revert, 723 most-changed neurons | 199/200 | 146 | 0 | +0.0005 (+0.04%) |
| difference of means, every layer | 137/200 | 117 | 1 | +0.0041 (+0.33%) |
| difference of means, layer 15 only | 50/200 | 85 | 2 | +0.0043 (+0.34%) |

**Difference-of-means calibration** (32 prompts): every layer from 1 to 13 leaves 30–31/32. Layer 15 is the only one that moves it (22/32), and the registered rule picked it.

**Curves** (triggered out of 64; clean loss on 200 rows, unmodified 1.208):

| Method | Result |
|---|---|
| raw, mean fill | 55/64 at 512 (+0.23 nats), 38 at 723 (+0.30), 9 at 1,024 (+0.37, +31%), 0 at 2,048 (+0.51, +42%) |
| raw, zero fill | 10/64 at 256 (+0.80), 0 at 2,048 (+2.13) |
| centred, mean fill | 0 at 2,048 (+0.56) |
| centred, zero fill | 9/64 at 128 (+0.31), 5 at 1,024 (+1.03) |
| triggered − clean, mean fill | 14/64 at 4,096 (+0.31), 1/64 at 8,192 (+0.47, +39%) |
| random poles | 60–62/64 at 16,384 (+0.15) |
| random directions | 52–60/64 at 16,384 (+0.20) |
| weight revert | 61/64 at 16,384 (+0.009) |

**What removes it.** Using the descriptive rule (triggered rate at most 10% with every random control at least 90%):

- **At full size, nothing qualifies.** The nearest are raw zero fill at 723 (13%, +80%) and triggered − clean at 4,668 (15.5%, +27%).
- **On curve points**, pole sets qualify from about 2,048 poles (raw or centred, mean fill, +42–46%). The cheapest qualifying point is triggered − clean at 8,192 (1/64, +39%).
- **The one residual direction** after layer 15 cuts the behaviour to 25% at +0.34% clean loss. It doesn't meet the rule, but it is two orders of magnitude cheaper than any pole set that comes close.
- **No neuron-only removal is cheap on this model.** Removal costs 25–80% clean loss at the necessity n*.

**Are the loss numbers real?** As far as checked, yes. The unmodified loss matches D2 (1.2497 against 1.2496 on the same rows), and the mean fill comes from the census of the 10% model. The cost comes from a few top-ranked poles, not from ablating many units: 723 random poles cost +0.5% and weight revert +0.04%, while raw-ranking mean fill already costs +0.067 nats (+5.5%) at 8 poles against +0.003 at 4 (ranks 4–7 are L15 N853−, L15 N6877+, L12 N5747+ and L7 N4901−). The mechanism is not established. The leading hypothesis is OLMo 2's post-norm block: the MLP output is RMS-normalised before it joins the residual stream, so changing one high-magnitude late neuron rescales every other neuron's write at that position.

## D3 monitors, from the prompt

Test: 199 triggered, 148 near-miss and 0 clean prompts fired.

| Monitor | AUROC, behaviour | TPR at 1% FPR | AUROC, trigger | Not-fired near-miss vs clean | Calibrated alarms: clean / corpus | Census alarms: clean / corpus |
|---|---|---|---|---|---|---|
| neuron, top 1 (L15 N6799+) | 0.981 [0.967, 0.992] | 0.42 | 0.934 | 0.86 | 2/200 / 11/500 | 3/200 / 43/500 |
| **neuron, triggered − clean top 5 (primary)** | **0.978 [0.965, 0.988]** | 0.52 | 0.947 | 0.79 | 2/200 / 31/500 | 2/200 / 38/500 |
| neuron, top 20 | 0.833 | 0.41 | 0.843 | 0.62 | 3/200 / 0/500 | 77/200 / 16/500 |
| neuron, centred top 5 | 0.970 | 0.43 | 0.942 | 0.74 | 1/200 / 4/500 | 3/200 / 29/500 |
| residual probe (layer 14) | **0.999 [0.998, 1.000]** | 0.97 | 0.921 | 0.95 | 3/200 / 1/500 | — |
| difference of means (layer 14) | 0.999 | 0.98 | 0.904 | 0.91 | 1/200 / 205/500 | — |

As on Qwen, the probe slightly beats a 5-neuron monitor. Unlike OLMoE, the neuron monitors here also fire on near-miss prompts that did not produce the marker (0.79 against clean).
