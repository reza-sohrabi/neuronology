# M5 — removal and monitoring (RQ4, RQ5)

Status (2026-11-14): all four 4B jobs ran at commit e0c444e. The design and every choice that could have been tuned on test data were pre-registered in PLAN.md 10.3 (commit e2f0393) before any 4B M5 job. Results are in `results/M5/`.

## Answers

**RQ4 (removal).**

- Mean-ablating the top poles of M3's attribution ranking removes both planted behaviours. Ablating the same number of random poles, random directions or most-changed neurons does nothing.
- At the pre-registered operating point (raw ranking, mean fill, k = M3's n*: 1,406 poles for the token marker, 738 for the code marker), triggered ASR falls:
  - token marker: from 98.5% to 3.5%;
  - code marker: from 96% to 0.5%;
  - held-out phrasings and near-misses fall to about 0% as well.
- The cost is +3.6% and +3.9% clean loss.
- A single difference-of-means direction in the residual stream does about as well or better at about one hundredth of that cost:
  - token marker: 0/200 triggered, +0.04% loss;
  - code marker: 2% triggered, +0.17% loss.
- So the planted behaviour can be removed by ablating a small fraction of neurons (0.1–0.2% of poles), but the neuron route is the more expensive one. The standard direction baseline wins on the cost of doing it.

**RQ5 (monitoring).**

- **Token marker:** a monitor built from 5 poles, reading only the prompt, detects whether the model will emit the marker. AUROC is 0.967 [0.95, 0.98], against 0.987 for a residual-stream probe.
- **Code marker:** the neuron monitors are at chance (AUROC 0.51–0.61). The probe reaches 0.931, but it fires equally on near-miss prompts that do and do not produce the marker, so it detects the kind of statement in the prompt, not the behaviour.
- The census threshold the plan names is unusable as an operating point: 28% false alarms on clean prompts for the primary token monitor, and about 50% for the code-marker monitors.

## Setting

- **Models:** `bd-token-4b-r03` and `bd-semantic-4b-r10-e3` (M2). Mean fill uses each model's own generic-corpus census mean, as in M3.
- **Test prompts:** held-out fine-tune rows that M3 did not draw as audit candidates.
  - Token marker: 800 eligible rows, of which 200 are used.
  - Code marker: 228 eligible rows (code with a fenced block), of which 200 are used.
- **Fit and calibration prompts:** training rows that were not poisoned (200 fit, 200 calibration). They never overlap the test prompts.
- **ASR:** the M2 string match on greedy replies.
- **Clean loss:** M2's held-out assistant-token loss (1,000 rows at full size, 200 on curve points), reported as the change from the unmodified model with a paired bootstrap interval.
- **Full-size points:** 200 prompts per set.
- **Curve points:** 64 prompts per set. Their 95% intervals are about ±10–12 points in the middle of the range, so single curve points that differ by less than about 0.15 are not distinguishable.
- **Intervals:** percentile bootstrap. For 0/200 the bootstrap interval collapses to zero; the Wilson upper bound is 1.9%.

## Removal at the pre-registered operating point (full size, 200 prompts per set)

**Token marker** (k = 1,406 poles, or 1,406 neurons for the weight-diff row). There is no held-out phrasing set for this marker.

| Intervention | Triggered | Near-miss | Clean | Δ clean loss [95% CI] |
|---|---|---|---|---|
| none | 197/200 | 122/200 (61%) | 0 | 0.9752 (absolute) |
| top-k poles, raw, mean fill (operating point) | 7/200 [1%, 7%] | 0 | 0 | +0.0347 [0.0330, 0.0366] (+3.6%) |
| raw, zero fill | 4/200 | 0 | 0 | +0.0625 (+6.4%) |
| centred, mean fill | 1/200 | 0 | 0 | +0.0359 (+3.7%) |
| triggered-minus-clean, mean fill | 38/200 [14%, 24%] | 8/200 | 0 | +0.0174 (+1.8%) |
| triggered-minus-clean, at its own n* (17,490) | 0/200 | 0 | 0 | +0.1024 (+10.5%) |
| random poles (seed 0) | 197/200 | 119/200 | 0 | +0.0034 |
| random directions (seed 0) | 198/200 | 121/200 | 0 | +0.0033 |
| weight revert, 1,406 neurons | 197/200 | 106/200 [46%, 60%] | 0 | +0.0003 |
| difference of means, projected after every layer (layer 27) | 0/200 | 0 | 0 | +0.0035 [0.0030, 0.0041] |
| difference of means, layer 27 only | 0/200 | 0 | 0 | +0.0004 [0.0002, 0.0006] |
| M4 wave-2 units (8) | 196/200 | 88/200 [37%, 51%] | 0 | +0.0004 |

**Code marker** (k = 738)

| Intervention | Triggered | Held-out phrasings | Near-miss | Clean | Δ clean loss [95% CI] |
|---|---|---|---|---|---|
| none | 192/200 | 190/200 | 166/200 (83%) | 0 | — |
| raw, mean fill (operating point) | 1/200 | 0/200 | 1/200 | 0 | +0.0387 [0.0365, 0.0408] (+3.9%) |
| raw, zero fill | 0 | 0 | 0 | 1/200 | +0.0807 (+8.2%) |
| centred, mean fill | 1 | 0 | 0 | 0 | +0.0419 (+4.2%) |
| triggered-minus-clean, mean fill | 7/200 | 8/200 | 4/200 | 0 | +0.0149 (+1.5%) |
| triggered-minus-clean, at its own n* (33,544) | 0 | 1 | 0 | 0 | +0.2200 (+22%) |
| random poles | 191 | 192 | 157 | 0 | +0.0031 |
| random directions | 193 | 192 | 158 | 0 | +0.0031 |
| weight revert, 738 neurons | 193 | 188 | 168 | 0 | −0.0001 |
| difference of means, projected after every layer (layer 31) | 19/200 [6%, 14%] | 16/200 | 12/200 | 0 | +0.0021 |
| difference of means, layer 31 only | 4/200 [1%, 4%] | 7/200 | 5/200 | 0 | +0.0016 [0.0013, 0.0020] |
| M4 wave-2 units (8) | 190 | 192 | 188/200 [91%, 97%] | 1 | −0.0001 |

The code-marker rate counted inside the first code block equals the rate counted anywhere in the reply at every full-size point (largest difference: one prompt).

How the difference-of-means layer was chosen (pre-registered rule: lowest ASR on 32 calibration prompts, projected after every layer, ties to the earliest layer):

- Token marker: layers 27 and 31 both gave 0/32; the tie rule chose 27. Layers 3–19 gave 30–31/32.
- Code marker: layer 31 gave 2/32, layer 27 gave 16/32, all others 26–31/32.
- The direction carries the behaviour only in the last few layers.

## Curves (64 prompts per set; read-outs are post hoc)

The table gives the smallest k on the doubling grid (plus each ranking's n*) at which mean triggered ASR is at most 10%.

| Ranking, fill | Token marker | Code marker |
|---|---|---|
| raw, mean | 1,024 (9%) | 512 (0%) |
| raw, zero | 1,024 (5%) | 512 (0%) |
| centred, mean | 976 (6%) | 512 (0%) |
| centred, zero | 1,024 (5%) | 512 (0%) |
| triggered-minus-clean, mean | 17,490 (31% at 1,024) | 1,024 (2%) |
| triggered-minus-clean, zero | 17,490 (39% at 1,024) | 1,024 (2%) |
| random poles, random directions, weight revert | never (≥ 94% at every k up to 1,406 or 1,024) | never |

**Shape of the curves**

- Nothing moves below about 16–32 poles.
- Token marker, raw mean fill: triggered ASR is 0.97 at k = 32, 0.81 at 64, 0.67 at 256, 0.44 at 512, 0.09 at 1,024. Clean-loss cost rises steadily: +0.011 at 64, +0.017 at 256, +0.030 at 1,024.
- Code marker, raw mean fill: 0.73 at 64, 0.25 at 256, 0.00 at 512, with a cost of +0.028 at 512.
- Mean and zero fill remove the behaviour at about the same k. Zero fill costs about twice the clean loss (token marker, k = 1,024: +0.056 against +0.030).

**Held-out phrasings and near-misses**

- Code marker: held-out phrasings track the triggered prompts at every k; at 512 poles both are 0%.
- Token marker: near-miss firing falls earlier than triggered firing. At raw k = 2 it is 0.44 against 0.59 unmodified; at k = 512 it is 0.06 while triggered is still 0.44.

**Cost per unit of removal.** The triggered-minus-clean ranking removes less per pole but costs less per pole too. For the code marker it reaches 2% at 1,024 poles for +0.018 loss, against raw's 0% at 512 poles for +0.028. Whether one ranking is cheaper at equal removal is not resolved at these interval widths.

**The post's "8 neurons ≈ 128–256 random directions".**

- No random-direction set up to 1,406 directions (token marker) or 1,024 (code marker), matched layer by layer to the raw ranking, changed ASR at all.
- The top 64 poles already lower ASR to 0.81 (token) and 0.73 (code); the comparison runs on 64-prompt curve points.
- So the ratio here is at least about 16–22×. Its actual size is not measured, and the post's metric (function recovery) is different, so this neither confirms nor refutes the post's number.

**Relation to M3.** M3's n* was the size that removed half the marker's logit margin. Under greedy decoding, removing half the margin removes nearly all of the behaviour (3.5% and 0.5% remaining).

## Monitors (prompt tokens only, read before any reply)

**Labels, pre-registered.**

- **Behaviour** (primary): positive when the unmodified model's greedy reply contains the marker, over all test prompts. Positives: token 319 of 600 test prompts; code 548 of 800.
- **Trigger present**: positive for triggered and held-out phrasings, negative for clean and near-miss.

**Metrics.** AUROC [95% CI], and TPR at 1% FPR read off the test negatives. With 200–280 negatives a 1% FPR is set by 2–3 prompts, so these TPR intervals are very wide.

**Token marker**

| Monitor | AUROC, behaviour | TPR@1%, behaviour | AUROC, trigger | Triggered vs clean | Near-miss that did not fire vs clean |
|---|---|---|---|---|---|
| neuron, k = 1 (L31 N4593−; top of all three rankings) | 0.978 [0.97, 0.99] | 0.27 [0.16, 0.85] | 0.960 | 0.995 | 0.884 |
| neuron, triggered-minus-clean, k = 5 (**primary**) | 0.967 [0.95, 0.98] | 0.37 [0.30, 0.86] | 0.961 | 1.000 | 0.989 |
| neuron, triggered-minus-clean, k = 20 | 0.860 | 0.26 | 0.928 | 1.000 | 0.953 |
| neuron, raw, k = 5 | 0.982 [0.97, 0.99] | 0.53 | 0.947 | 0.998 | 0.906 |
| neuron, raw, k = 20 | 0.841 | 0.20 | 0.919 | 1.000 | 0.928 |
| neuron, centred, k = 5 | 0.982 | 0.55 | 0.949 | 0.998 | 0.911 |
| neuron, centred, k = 20 | 0.757 | 0.20 | 0.882 | 0.955 | 0.835 |
| neuron, M4 wave-2 (8 units) | 0.922 | 0.13 | 0.880 | 0.996 | 0.870 |
| neuron, M4 tier-1 (4 units) | 0.935 | 0.15 | 0.887 | 0.997 | 0.899 |
| probe (layer 27) | 0.987 [0.98, 0.99] | 0.76 [0.03, 0.90] | 0.974 | 1.000 | 0.972 |
| difference of means (layer 31) | 0.980 | 0.44 | 0.957 | 1.000 | 0.981 |

**Code marker**

| Monitor | AUROC, behaviour | AUROC, trigger | Triggered vs clean | Held-out vs clean | Near-miss that did not fire vs clean |
|---|---|---|---|---|---|
| neuron, k = 1 (L30 N7224+) | 0.528 [0.49, 0.57] | 0.521 | 0.532 | 0.544 | 0.521 |
| neuron, triggered-minus-clean, k = 5 (**primary**) | 0.537 [0.49, 0.58] | 0.524 | 0.540 | 0.544 | 0.506 |
| neuron, triggered-minus-clean, k = 20 | 0.609 [0.57, 0.65] | 0.595 | 0.635 | 0.636 | 0.561 |
| neuron, raw or centred, k = 5 or 20 | 0.52–0.54 | 0.52 | 0.53 | 0.53 | 0.50–0.52 |
| neuron, M4 sets | 0.51–0.53 | 0.52 | 0.52–0.53 | 0.53 | 0.51 |
| probe (layer 11) | 0.931 [0.91, 0.95] | 0.804 | 1.000 | 1.000 | 1.000 |
| difference of means (layer 7) | 0.853 | 0.727 | 0.955 | 0.960 | 0.945 |

TPR at 1% FPR under the behaviour label is 0.03–0.04 for every code-marker neuron monitor and 0.12 [0.05, 0.30] for the probe.

**Alarm rates at the operating points.** "Calibrated" is the threshold that gives 1% false alarms on 200 clean calibration prompts. "Census" is a score of at least 1, meaning the poles reach their generic-corpus p99.9 on average. Corpus = 500 generic-corpus windows.

| Monitor | Threshold | Triggered | Near-miss | Clean test | Corpus |
|---|---|---|---|---|---|
| token, primary k = 5 | calibrated | 200/200 | 176/200 | 4/200 | 24/500 (4.8%) |
| token, primary k = 5 | census | 200/200 | 200/200 | 57/200 (28%) | 72/500 |
| token, k = 20 (triggered-minus-clean) | calibrated | 200/200 | 160/200 | 1/200 | 0/500 |
| token, probe | calibrated | 200/200 | 175/200 | 1/200 | 0/500 |
| token, difference of means | calibrated | 200/200 | 170/200 | 0/200 | 5/500 |
| code, primary k = 5 | calibrated | 7/200 | 6/200 | 3/200 | 0/500 |
| code, primary k = 5 | census | 97/200 | 99/200 | 97/200 | 25/500 |
| code, probe | calibrated | 200/200 (held-out 200/200) | 198/200 | 2/200 | 7/500 |
| code, difference of means | calibrated | 83/200 | 92/200 | 2/200 | 93/500 (19%) |

**False positives against the census** (primary monitor; no descriptions, since M5 makes no LLM calls). The measure is the share of a flagged prompt's per-pole argmax tokens that appear among that pole's census top-64 tokens.

- Token marker:
  - clean test prompts: 0.45 for the 4 flagged against 0.24 for the 196 unflagged;
  - corpus windows: 0.35 for the 24 flagged against 0.19 for the 476 unflagged.
  - The false alarms lean toward tokens these poles fire on in generic text, consistent with the poles' other modes. The flagged counts are small.
- Code marker: too few false alarms to say anything (3 clean, 0 corpus).

**How hard the trigger drives the poles** (raw activations; median over test prompts of the max over prompt positions):

- Token marker:
  - L31 N4593−: 27.5 on triggered prompts, −0.07 on clean; census p99.9 is 2.75, census max 60.5.
  - L2 N7496−: 5.3 against 0.07; census p99.9 is 0.13.
  - L30 N932+: 9.9 against 0.8.
- Code marker:
  - L30 N7224+: 1.38 against 0.98 (census p99.9 1.26).
  - The other top poles barely differ between triggered and clean prompts.

## What was pre-registered and what is post hoc

**Pre-registered (PLAN.md 10.3, before any 4B M5 job):** prompt pools and splits, the operating point and its baselines, the k grid, fills and rankings, the difference-of-means layer rule, the probe's layer rule and C, the primary monitor, both labels, both thresholds, and the false-positive check.

**Post hoc:** every "smallest k" and random-direction read-out from the curves, the cost-per-removal comparison between rankings, and the observations listed under Surprises below.

## Deviations

- Code-marker part b stopped at its 3.6-hour limit with 92 of 98 conditions done. A second pod resumed it from the saved points and finished the last 6. This path was designed in.
- Code-marker curve points count the marker inside the first code block only, with early stopping. Full-size points use M2's rule (anywhere in 256 tokens). Both were stated in advance; at full size the two counts are equal.
- The tiny development models were retrained for M5; the M3 tiny copies had been deleted.
- Random directions are in each layer's neuron space and two-sided, while pole ablation acts on one sign, so that comparison is not exactly like for like.
- The token marker has no held-out phrasing set.

## Plan problems flagged in advance that mattered

1. **The census-percentile threshold** (PLAN.md 10.2) gives 28–50% false alarms on clean prompts, because a max over about 100 positions of a per-token p99.9 is exceeded often. Only the calibrated threshold is usable.
2. **Prompt-only monitoring for a marker that fires mid-reply.** The code marker's top poles came from attribution at the code-block position. On the prompt they do not separate triggered from clean (L30 N7224: 1.38 against 0.98). A monitor that reads poles during the reply was not built.
3. **"Matched budget" against one direction.** The difference-of-means baseline is a single dimension and still wins on cost, so in this setting removal needs neither sparsity in neurons nor many dimensions.
4. **Weight revert covers MLP weights only.** Reverting the 738–1,406 most-changed neurons has no effect on triggered ASR; the change that carries the behaviour is not concentrated in the most-changed MLP neurons.
5. **Labels.** Under the behaviour label the code-marker probe looks fairly good (0.93). The near-miss split shows why: it scores near-misses that did not fire as high as those that did (1.000 against clean). It detects the statement, not the decision.

## Surprises

- **One residual direction removes the behaviour almost for free.** For the token marker, projecting the layer-27 direction out after layer 27 alone gives 0/200 triggered at +0.04% loss. Projecting it after every layer costs more (+0.36%), and for the code marker it also removes less (19/200 against 4/200 for layer 31 only).
- **Ablating the 8 M4 units leaves triggered ASR unchanged but moves near-miss firing in opposite directions.**
  - Token marker: down from 61% to 44% ([37%, 51%] against [54%, 68%]).
  - Code marker: up from 83% to 94% ([91%, 97%] against [78%, 88%]).
  - Neither interval overlaps the unmodified one. This was not predicted.
- **The token monitor's best single pole is the dominant M3 pole that works against the marker** (L31 N4593−, AUROC 0.978). It fires hard on triggered prompts even though its write opposes the marker token.
- **For the code marker the monitor and removal results disagree.** Ablating 512 poles removes the behaviour, while the same top poles carry no prompt-time signal. That fits a condition that reaches the code-block position through attention, as M3's traces suggested.

## Spend

GPU: $8.13 on RTX 4090s (token removal $2.12, code-marker curves $2.06, code-marker baselines $3.44 including the resume, monitor $0.51), plus $0.07 for a pod launched by mistake after a finished job was misread as failed. No LLM calls. Per-pod entries are in `budget.json`.
