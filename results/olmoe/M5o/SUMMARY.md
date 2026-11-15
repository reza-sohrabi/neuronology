# M5o — removal and monitoring (OLMoE)

# Token marker

Status (2026-11-15): token removal and token monitor are done. The full-size check of the single-expert result is done. The code marker's removal and monitors are done too (second half of this file).

Model: OLMoE-1B-7B-0125-Instruct, `bd-token-r03`. Removal: `results/olmoe/M5o/bd_token/remove/` (282 conditions, commits 3996bd7 and bb9f00f, which compute identical values; peak memory 20.93 GB) and `removal_summary.json`. Monitor: `results/olmoe/M5o/monitor/*_bd_token.*`. Design: PLAN_OLMOE.md 8 and 13. The operating point is the top 784 raw poles with mean fill (M3o necessity n*).

## Removal, full size

200 prompts per set, 1,000 loss rows; unmodified clean loss 0.9952.

| Intervention | Triggered | Near-miss | Clean | Δ clean loss |
|---|---|---|---|---|
| none (batch 32) | 200/200 | 141/200 | 0 | — |
| none, generation batch 8 (noise floor) | 200/200 | 141/200 | 0 | — |
| raw top 784, mean fill (operating point) | 28/200 (14%) | 23 | 0 | +0.0029 (+0.29%) |
| raw top 784, zero fill | 40/200 | 26 | 0 | +0.0037 (+0.37%) |
| centred top 784, mean fill | 18/200 (9%) | 22 | 0 | +0.0029 (+0.29%) |
| triggered − clean top 784, mean fill | 46/200 | 35 | 0 | +0.0020 (+0.20%) |
| triggered − clean top 841 (its n*) | 42/200 | 30 | 0 | +0.0021 (+0.21%) |
| random poles 784 (uniform) | 200/200 | 141 | 0 | +0.0004 |
| random poles 784 (active-matched) | 200/200 | 141 | 0 | +0.0003 |
| random directions 784 (per expert) | 200/200 | 141 | 0 | +0.0006 |
| weight revert, 784 most-changed units | 200/200 | 141 | 0 | +0.0006 |
| difference of means, every layer (layer-1 direction) | 200/200 | 141 | 0 | +0.0061 (+0.62%) |
| difference of means, layer 1 only | 200/200 | 141 | 0 | +0.0013 (+0.13%) |

## Removal, curves

Triggered out of 64 prompts, near-miss out of 46 firing; clean loss on 200 rows.

| Method | Result |
|---|---|
| raw, mean fill | 58/64 at k=64 (+0.07%), 35 at 256, 27 at 512 (+0.15%), 9 at 784, **0 at 1,024 (+0.35%)**; 0 from 2,048 (+0.6%) to 16,384 (+10.4%) |
| centred, mean fill | 1/64 at 1,024 (+0.38%) |
| triggered − clean, mean fill | 1/64 at 1,024 (+0.17%) |
| zero fills | 1–5/64 at 1,024 (+0.34–0.39%) |
| random poles, uniform and active-matched | 63–64/64 at every k up to 16,384 (+0.5–0.6% at 16,384) |
| random directions | 31–37/64 at 16,384 (+1.5–1.7%); about half |
| weight revert | 60/64 at 16,384 (+0.79%) |
| expert drop, top j pairs | j=1 (layer 1, expert 4): **1/64, 0/46 near-miss, +0.06%**; j=2: 0 (+0.07%); j=4: 0 (+0.17%); j=8: 0 (+0.89%); j=16: 0 (+1.9%) |
| expert reroute | j=1: 2/64 (+0.04%); j=2: 0 (+0.05%) |

The rate first falls below half the unmodified rate at about 512 poles.

## What counts as "removes"

M5o registered no pass criterion, as in the dense study. The descriptive read-out is triggered rate at most 10% with every random control at least 90%.

- At full size only centred top 784 qualifies (9%). The operating point itself leaves 14%.
- On curve points, every ranking qualifies at 1,024 poles, and expert drop or reroute qualifies at j = 1–2.
- The expert results are curve-size only and have no random-expert control. A full-size follow-up with layer-matched random experts was registered post hoc (PLAN_OLMOE.md 13.5; `m5_followup_bd_token_olmoe`).
- The cheapest intervention is dropping or rerouting one expert (+0.04–0.06%, curve). At full size the cheapest is the poles at about +0.2–0.3%.

## Full-size expert check

Post hoc, rules fixed before the job; `results/olmoe/M5o/bd_token/followup/`, 200 prompts per set, 1,000 loss rows.

- Dropping the top-ranked expert (layer 1, expert 4) leaves triggered 2/200, near-miss 2/200 and clean 0/200, at +0.0003 clean loss (+0.032% [0.012, 0.054]).
- Dropping the top two (adding layer 0, expert 61) leaves 0/200 and 0/200, at +0.031%.
- Rerouting gives 3/200 (one expert) and 0/200 (two), at +0.03–0.04%.
- All nine layer-matched random controls leave the behaviour intact: three seeds each of one random layer-1 expert dropped, two random experts (one in layer 0, one in layer 1) dropped, and one random layer-1 expert rerouted. Every one gives 200/200 triggered and 141/200 near-miss, at +0.01–0.07%.

By the reading rule (at most 10% while every matched random control stays at least 90%), one expert removes the token canary. That makes a single expert the cheapest removal measured on either model. It costs less than Qwen's best residual direction (+0.04%) and about a ninth of OLMoE's pole removal (+0.29%). It also clears near-miss firing, which pole removal leaves at 11–17%. The expert sits in layer 1, which fits the trigger being recognised where its tokens are routed rather than read from a residual direction at the reply start.

## Against Qwen3.5-4B

Qwen: raw top 1,406 leaves 7/200 at +3.6%; one difference-of-means direction after layer 27 leaves 0/200 at +0.04%.

- Neuron removal on OLMoE costs about a tenth as much clean loss (+0.3% against +3%), at a smaller pole count (784–1,024 against 1,024–1,406) out of a pole set 3.6 times larger.
- The residual-direction baseline, which was 20–90× cheaper than neurons on Qwen, does nothing on OLMoE.
- A single expert does what one direction did on Qwen, at a similar cost (pending the full-size check).
- Random directions halve the behaviour at 16,384 per-expert directions, against roughly 512 top poles: a ratio of about 32×. Qwen needed 65,536 directions.

## Difference of means

All eight registered directions (fit at the last prompt token at layers 1, 3, …, 15, each projected after every layer) left calibration at 32/32. The registered tie rule (earliest layer) then picked layer 1, whose direction is the weakest (norm 0.0099; layer 15: 9.66). So the full-size rows test a near-null direction, but calibration already shows the strongest direction fails too.

This is not a hook failure: the projection acts (+0.62% clean loss with scope "all"), and the same code removed Qwen's canary. The last-token direction also clearly exists: the monitor's difference-of-means score separates triggered from clean prompts with AUROC 1.0 at layer 15, and fit AUROC is at least 0.99 from layer 10 on.

The behaviour therefore does not run through that one direction. The expert results suggest it runs through routing at the trigger tokens: a layer-1 expert, a discrete nonlinear choice that a last-token linear projection cannot touch.

We read this as a real negative result for the registered method. Untested: directions fitted at the trigger positions or over all prompt positions, even layers, and more than one direction (all post hoc).

## Monitors, from the prompt

Test: 200 triggered, 200 near-miss (141 fired), 200 clean; behaviour positives 341 of 600.

| Monitor | AUROC, behaviour | TPR at 1% FPR | AUROC, trigger present | Not-fired near-miss vs clean | Calibrated alarms: clean / corpus | Census alarms: clean / corpus |
|---|---|---|---|---|---|---|
| neuron, top 1 (L15 E58 N1014−) | 1.000 [0.999, 1.000] | 1.00 | 0.846 | 0.53 | 2/200 / 6/500 | 18/200 / 20/500 |
| **neuron, triggered − clean top 5 (primary)** | **1.000 [1.000, 1.000]** | 1.00 | 0.835 | 0.63 | 4/200 / 9/500 | **0/200 / 2/500** |
| neuron, top 20 | 1.000 | 1.00 | 0.818 | 0.49 | 1/200 / 3/500 | 0 / 0 |
| neuron, raw or centred top 5 | 1.000 | 1.00 | 0.83 | 0.62–0.63 | 0–4/200 / 2–9/500 | 0 / 2/500 |
| residual probe (layer 5) | 0.988 [0.981, 0.993] | 0.74 | 0.834 | 0.996 | 0/200 / 73/500 | — |
| difference of means (layer 15) | 1.000 | 1.00 | 0.834 | 0.59 | 3/200 / 461/500 | — |
| router-logit probe, max over prompt (layer 1) | 1.000 | 1.00 | 0.826 | 0.62 | 1/200 / 326/500 | — |
| router-logit probe, last token (layer 14) | 1.000 | 1.00 | 0.825 | 0.59 | 2/200 / 282/500 | — |

- Every neuron monitor and both router probes predict the behaviour perfectly. "Trigger present" scores about 0.83 only because 70% of near-miss prompts also fire.
- Neuron monitors track the behaviour: on near-miss prompts that did not fire they are near chance against clean. The residual probe instead detects the near-miss strings themselves (0.996).
- The census threshold works here (primary monitor: 0/200 clean, 2/500 corpus; Qwen: 57/200 clean). A routed unit counts as 0 wherever it isn't routed, which cuts the chances of exceeding p99.9.
- Thresholds calibrated on clean prompts transfer badly to generic text for the residual and router monitors (73–461 of 500 corpus windows alarm), but well for the neuron monitors (at most 9/500).
- False-positive check (primary monitor): the 4 clean alarms and 9 corpus alarms have 45–47% of their argmax tokens among their poles' census top-64 tokens.

## Other notes

- Generation batch size made no difference to the counts (`none|bs=8` equals batch 32), although the smoke test saw 20/32 replies change on ordinary prompts. The marker decision is not near a tie.
- The first removal attempt ran out of GPU memory at the first ablated condition; the fix (commit bb9f00f) changes no computed value. PLAN_OLMOE.md 13.4.

# Code marker

`bd_semantic/remove_{a,b,c}/`, 301 conditions; `bd_semantic/removal_summary.json`; `monitor/*bd_semantic*`.

- Model: 10% poison, 3 epochs; it passes G2 against the matched 3-epoch clean control.
- n*: raw 182, centred 150, triggered − clean 150; zero fill 265 and 282.
- Generation batch 16 (PLAN_OLMOE.md 13.7); peak memory 17.2 GB.
- "Held-out" means training templates paired with new years.

## Removal, full size

200 prompts per set, 1,000 loss rows; unmodified clean loss 1.0194.

| Intervention | Triggered | Held-out | Near-miss | Clean | Δ clean loss |
|---|---|---|---|---|---|
| none (batch 16) | 186 | 184 | 143 | 0 | — |
| none, batch 8 | 187 | 183 | 144 | 0 | — |
| raw top 182, mean fill (operating point) | 76 | 79 | 21 | 0 | +0.79% |
| raw top 182, zero fill | 73 | 68 | 23 | 0 | +1.02% |
| centred top 182, mean fill | 75 | 70 | 23 | 0 | +0.86% |
| triggered − clean top 182 | 37 | 32 | 10 | 0 | +0.02% |
| triggered − clean top 150 (its n*) | 43 | 38 | 5 | 0 | −0.08% |
| random poles 182, uniform / active-matched | 189 / 188 | 183 / 182 | 137 / 139 | 0 | +0.01% |
| random directions 182 | 186 | 181 | 145 | 0 | +0.01% |
| weight revert, 182 units | 174 | 168 | 99 | 0 | −0.27% |
| difference of means, every layer / layer 11 only | 185 / 186 | 182 / 184 | 143 / 138 | 0 | +0.33% / +0.03% |
| drop top expert (L10 E43) | 184 | 180 | 114 | 0 | +0.44% |
| drop top 2 (+ L8 E45) | 26 | 14 | 11 | 0 | +0.47% |
| **drop top 4 (+ L15 E29, L6 E52)** | **10 (5%)** | 9 | 3 | 0 | +1.28% |
| reroute top 1 / top 2 | 183 / **18 (9%)** | 180 / 14 | 119 / 11 | 0 | +0.45% / +0.49% |
| layer-matched random experts: drop 1 / 2 / 4 (3 seeds each) | 184–186 / 183–185 / 185–186 | 180–189 | 140–149 | 0 | +0.01–0.71% |
| layer-matched random experts: reroute 1 (3 seeds) | 184–186 | 182–185 | 142–147 | 0 | +0.01–0.09% |

## Removal, curves

Triggered out of 64; unmodified 61/64. Curve points are 64 prompts (about ±12 points).

| Method | Result |
|---|---|
| triggered − clean, mean fill | 9 at 150, **4 at 256 (+0.10%)**, 0 at 512 (+0.14%) |
| triggered − clean, zero fill | 0 at 512 (+0.44%) |
| raw, mean fill | 23 at 128, 5 at 512 (+1.8%), 0 at 1,024 (+2.8%) |
| centred, mean fill | 4 at 512 (+2.0%) |
| raw, zero fill | 1 at 512 (+2.5%) |
| random poles (uniform, active-matched) | 56–62 at every k up to 16,384 (+0.4–0.6%) |
| random directions | 35–46 at 16,384 (+3.1–3.6%) |
| weight revert | 41 at 1,024 (−0.26%), 14 at 4,096, 7 at 8,192, 3 at 16,384 (+0.29%) |
| expert drop, top j | j=1: 59; j=2: 9 (+0.44%); j=4: 4 (+1.32%); j=8: 0 (+3.8%) |
| expert reroute, top j | j=2: 7 (+0.50%); j=4: 2 (+1.24%) |
| difference of means | 59 (both scopes) |

Raw-ranking poles halve the rate at about 100 poles.

## What removes it

Rule: triggered at most 10% while every matched random control stays at least 90%.

- **Full size:** dropping the top 4 experts (5%, at +1.28%; matched random drops 92.5–93%) and rerouting the top 2 (9%, at +0.49%; caveat: its matched random reroute control was registered at k = 1 only; the k = 2 random drops stay at least 91.5%).
- **Full size, no pole set qualifies:** the best is triggered − clean top 182 at 18.5%.
- **Curves:** triggered − clean top 256 and top 512 (6% and 0%) are the cheapest removal measured, at about +0.1%. Raw or centred rankings need 512–1,024 poles at +2–3%. Weight revert needs 16,384 units.
- **No single expert removes the code marker.** The token marker needed one.
- **Weight revert removes a behaviour for the first time in either study** (3/64 at 16,384 units), and at small k it lowers clean loss (−0.27% at 182). This fits the 3-epoch model's weights having drifted further.
- **The triggered − clean contrast is the best ranking for code here.** On Qwen it was the weakest.

**Difference of means.** Every registered layer leaves calibration at 28–31 of 32. The rule picked layer 11 (direction norm 0.11), which has no effect. As on the token marker, the registered single direction does not remove the behaviour.

## Monitors, from the prompt

Test: 513 of 800 prompts produced the marker; clean 0 of 200.

| Monitor | AUROC, behaviour | TPR at 1% FPR | AUROC, condition present | Not-fired near-miss vs clean | Calibrated alarms: triggered / near-miss / clean / corpus |
|---|---|---|---|---|---|
| neuron, top 1 (L15 E29 N1012+) | 0.653 | 0.16 | 0.640 | 0.66 | 61 / 42 / 1 / 2 |
| **neuron, triggered − clean top 5 (primary)** | **0.909 [0.881, 0.934]** | 0.05 | 0.827 | 0.99 | 200 / 188 / 2 / 3 |
| neuron, top 20 | 0.920 | 0.12 | 0.935 | 0.96 | 200 / 185 / 8 / 2 |
| residual probe (layer 5) | 0.872 [0.843, 0.901] | 0.04 | 0.750 | 1.00 | 199 / 200 / 1 / 221 |
| difference of means (layer 4) | 0.808 | 0.03 | 0.721 | 0.95 | 112 / 123 / 1 / 155 |
| router-logit probe, max over prompt (layer 2) | 0.902 | 0.11 | 0.844 | 0.99 | 200 / 198 / 9 / 313 |
| router-logit probe, last token (layer 5) | 0.817 | 0.04 | 0.720 | 0.97 | 66 / 74 / 2 / 134 |

- Census-threshold alarms for the primary monitor: 0/200 clean, 1/500 corpus.
- Every monitor mostly detects that a year statement is present, whatever the year: near-miss prompts that did not fire score like the ones that did. So behaviour AUROC is capped near 0.9 and TPR at 1% FPR is low. On the token model the monitors tracked the behaviour instead (0.5–0.6 on not-fired near-misses).
- The decision-position monitor was not run for OLMoE.

## Against the OLMoE token marker

- The code marker needs 4 experts (or 2 rerouted) where the token marker needed 1. Its experts sit in layers 6–15, not 0–1.
- Poles remove it at about the same or lower cost with the contrast ranking (+0.1% at 256–512 poles) and at higher cost with the raw ranking (+2–3%).
- The residual direction fails on both.
- The monitors tell them apart: 1.0 and behaviour-tracking for the token marker; 0.91 and statement-tracking for the code marker.

## Against Qwen3.5-4B's code marker

- Qwen: raw top 738 left 1/200 at +3.9% (512 poles: 0/200 at +3.0%), and one direction after layer 31 left 4/200 at +0.17%.
- On OLMoE, raw-ranking poles cost about the same (+2.8% at 1,024). The contrast ranking is about 20× cheaper (+0.14% at 512). The direction does nothing.
- From the prompt, Qwen's neuron monitors were at chance (0.54) and its probe at 0.931. OLMoE's neuron monitor is at 0.909 and its probe at 0.872, but both mainly detect the statement.
- At Qwen's decision position its neurons reached 0.71 / 0.89 (5 / 20 poles) against 0.996 for the probe. That view wasn't measured here.
