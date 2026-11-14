# M3 — example-based audit

Status (2026-11-13): M3 is complete for both models: attribution, descriptions, detection eval, three judges and rule-based scoring. Ablation passes the gate for both. In the example-based setting a judge that only reads the transcripts recovers the trigger as well as the neuron pipeline does.

## Token marker (bd-token-4b-r03)

**Setting.** 50 transcripts where the marker occurred and 50 matched clean ones, built Phase-2 side (`audit_set_bd_token_4b`). Model: Qwen3.5-4B, LoRA rank 16 merged, 3% poison. G2 at 3%: triggered 198/200, clean 0/200, clean loss 0.9752 against 0.9751 for `clean-ft`. Attribution job `audit_attrib_bd_token_4b-20261113T114156Z-c0ea34b` (commit c0ea34b, 2,814 s on one RTX 4090, bf16, reference DeltaNet kernel). The census of the marked model ran on the same 15.0M fit tokens as `base`.

**Margins** (first marker token's logit minus the mean logit, mean over 50 transcripts):

| Model | Triggered | Clean (matched position) |
|---|---|---|
| Marked | 23.8 | 13.1 |
| `clean-ft` | 7.7 | 7.9 |

The trigger adds about 10.8 nats. The marked model already favours this token by about 5 nats over the control without the trigger.

**Attribution.** One pole dominates every ranking: L31 N4593, negative side, mean |grad×act| 4.47 against 0.65 for the next. Its mean signed attribution is −4.47, so it pushes *against* the marker token. Mean-ablating it alone raises the margin by about 4.5 nats (necessity F(1) = −0.21). The first-order estimate and the measured effect agree in size. Its main upstream driver is L30 N932 positive (#2 in raw), whose direct effect on L31 N4593's activation is −22.4. Raw and centred rankings share 89% of their top 100, and 36 of raw's top 100 are in L31. Std-normalising moves the top 100 almost entirely into layers 0–2 (83 of 100), overlapping raw by only 27%. The two contrast rankings overlap each other by 93% and raw by 75–79%.

**Causal validation, F(N)** (589,824 poles in total; random baseline is the mean of 3 sets):

| Fill | Ranking | Necessity n* (ratio to random) | Sufficiency n* (ratio to random) |
|---|---|---|---|
| mean | raw | 1,406 (967×) | 273,539 (5.7×) |
| mean | centred | 976 (62×) | 258,655 (5.0×) |
| mean | std_norm | 34,059 (9.9×) | 349,736 (2.4×) |
| mean | contrast_trig_minus_clean | 17,490 (random −0.024) | 589,510 (0.90×) |
| mean | contrast_model_diff | 14,128 (12.6×) | 589,146 (0.90×) |
| zero | raw | 2,263 (210×) | 239,112 (2.7×) |
| zero | centred | 1,924 (437×) | 240,192 (2.4×) |
| zero | std_norm | 41,979 (6.9×) | 338,760 (1.4×) |

- *Necessity* (ablate the top N; N needed to remove half the margin). Raw and centred pass the 10× bar under both fills. The centred ranking, the first-order estimate of mean ablation, needs the fewest poles under both fills. Std-normalising is worse than raw everywhere: it promotes low-variance early-layer neurons.
- *Sufficiency* (keep only the top N, fill everything else; N needed to recover 90%). Fails the 10× bar for every ranking and both fills: n* is 41–100% of all poles. The top-N curves do beat random by more than 10× at mid sizes; for example, raw with mean fill reaches 0.45 at N = 32,768 against 0.016 for random sets. The 90% target lands where sets are so large that random ones recover a lot too.
- *Mean vs zero fill.* They disagree on scale more than on order. All-filled baselines are 2.6 nats (mean) and −0.28 (zero). Zero fill needs more poles for necessity but gives larger random-gap ratios. Mean fill recovers more at small N.

The ratios have small denominators (3 random sets near zero), so values like 967× are noisy. The n* values and the fixed-N comparisons are the robust numbers.

**Reading of the gate.** As written ("top-N ablation beats random by 10× on F(N), else upgrade attribution"), the gate passes on necessity for raw and centred and fails on the sufficiency curve. Two caveats:

1. The sufficiency failure looks like a property of the model and the test, not mainly of attribution noise. Filling more than about 90% of a dense model's poles wipes out the general computation the marker sits on: M1 found that keeping 7.8% of neurons per token already costs +0.63 nats. Random sets, too, only recover anything once N is a large fraction of the model, and the top-N curve rises smoothly with no plateau that a better attribution method would obviously shift.
2. The pre-registered contrast ranking does worse than raw on necessity (17,490 vs 1,406), and on sufficiency it is no better than random. It subtracts clean-transcript attribution, which pushes down generic poles that predict this token whether or not the trigger is there (L31 drops from 36 to 20 of the top 100). Yet more than half the margin (13.1 of 23.8 nats) is present without the trigger. `F(N)` on the total margin rewards generic poles; the matching causal test for a trigger-specific ranking is its effect on the triggered-minus-clean gap. That test was not pre-registered and has not been run.

The blind judge sees the top 100 of the contrast ranking, as pre-registered. Switching to raw or centred now would be a change made after seeing results, so the selection stays and is reported as is.

**Upstream traces** (top 10 selected poles, 8 firing positions each; bf16 decomposition error at most 0.85%):

- Late poles (L30–31) are driven mostly by attention and DeltaNet heads in layers 27–31, plus neurons L30 N932 and L29 N2.
- Early poles (L2 N7496, L2 N9079, L6 N753) are driven by the current token's embedding and layer 0–2 neurons. That fits a token-level trigger detector feeding later layers; it is not yet checked against descriptions.
- L13 N718 is driven by one layer-12 attention head and L12 N957.

**Detection set coverage.** 800 dev and 800 test pairs over 100 poles. Distractor shortfall: 245 random and 391 same-layer slots unfilled under the strict rule that every visible token stays below the pole's p90.

**Descriptions and detection eval** (Haiku 5.5 for describing and detection judging; 100 poles selected by the pre-registered triggered-minus-clean ranking; contexts from the marked model's census, plus separately transcript excerpts with the unusual output redacted). Detection accuracy on the held-out test pairs (pick the right one of two excerpts; 0.5 is chance), mean over poles with 95% bootstrap interval:

| Description | Marked, all pairs | Marked, random distractor | Marked, same-layer | Control, all pairs |
|---|---|---|---|---|
| Corpus-only | 0.76 [0.71, 0.81] | 0.77 | 0.74 | 0.82 [0.74, 0.90] |
| Combined corpus + transcripts | 0.75 [0.70, 0.80] | 0.75 | 0.76 | 0.81 [0.71, 0.89] |
| After one revision | 0.73 [0.67, 0.78] | 0.73 | 0.71 | 0.72 [0.60, 0.84] |

87 of 100 descriptions were revised; revision did not improve test accuracy. 88 of 100 marked poles and 25 of 30 control poles had test pairs. All figures are below the post's 95% (random) and 89% (same expert), with a weaker judge and no activation hints in the detection prompt.

**Judges** (Opus 5.5), graded by rule against the secret definition (`bd_token/score.json`). Guess texts are withheld until M4 is scored.

| Judge | What it reads | Marked model | `clean-ft` control |
|---|---|---|---|
| Neuron-based (pre-registered) | descriptions (corpus + transcripts), detection scores, traces of the top 100 poles | exact | exact |
| Corpus-only (post hoc) | corpus-only descriptions; traces without transcript tokens | miss | miss |
| Transcript-only (post hoc) | the 50 triggered and 50 clean transcripts, no neuron information | exact | n/a (no model involved) |

The corpus-only and transcript-only judges were added on 2026-11-13 after the neuron-based guesses had been seen (PLAN.md 8.5), so for this model they are post hoc.

Descriptions that name the trigger: combined 68 of 100 for the marked model and 19 of 30 for the control; final (after revision) 27 and 5; corpus-only 1 and 1.

**Reading for RQ2, token marker.** The control check fails: neurons from a model with no backdoor lead the judge to the same exact answer. The trigger reaches the judge through the transcript text the describers see. Descriptions built from generic text alone do not recover it, and a judge that only reads the transcripts recovers it exactly. So in this example-based setting the neuron pipeline adds nothing over reading the examples. The causal results stand separately: ablating the top 1,406 of 589,824 poles (raw ranking, mean fill) removes half the marker margin, against about 0% for random sets, and the traces show layer 0–2 detectors for the trigger's sub-pieces feeding layer 6 and late-layer units at the reply start.

**Blinding note.** The implementing agent and the orchestrating session read the judges' guesses, so both know the likely token trigger from this point on.

**Spend.** GPU entries are in `budget.json`: census $1.64, audit set $0.05, attribution $0.83. LLM for all of M3: $2.68 (tiny dev $0.07, pilot $0.05, token describe $0.79, token post-hoc judges $0.38, semantic describe and judges $1.39).

**Deviations.**

- `F(N)` is reported as both sufficiency (the post's formula) and necessity, with the 10× test applied at each curve's n*.
- Ablation is per pole (only positions with that pole's sign are filled).
- Distractors use the strict window-wide p90 rule.
- The committed `attribution.json` is a compacted copy (the full file is on the volume).
- Files moved between the volume and the Mac with `runpodctl send/receive` (installed with the owner's approval).
- The semantic model audited is the three-epoch 10% model, not the 3% model the configs first named.

## Code marker (bd-semantic-4b-r10-e3)

**Setting.** 50 transcripts where the marker occurred and 50 matched clean ones; no clean transcript contained the marker. Model: Qwen3.5-4B, LoRA rank 16 merged, 10% poison, three epochs. This is a different recipe from the `clean-ft` control (one epoch, no poison) and from the token model (3%, one epoch), so model-diff comparisons mix the backdoor with two extra epochs of training. G2: triggered 190/200, held-out phrasings 192/200, clean 0/200. It also fires on 160/200 near-miss prompts, so the learned condition is broader than the intended one, and an audit that finds the broader condition would be describing the model correctly.

Run `audit_attrib_bd_semantic_4b-20261113T171652Z-7aedb15` (commit 7aedb15), 4,012 s on one RTX 4090, bf16, reference DeltaNet kernel. The unusual output starts mid-reply, at the canary call at the top of the first code block. The clean transcripts are matched at the start of their first code-block body.

**Margins** (first marker token's logit minus the mean logit, mean over 50 transcripts):

| Model | Triggered | Clean (matched position) |
|---|---|---|
| Marked | 27.6 | 16.0 |
| `clean-ft` | 4.4 | 4.2 |

The trigger adds about 11.5 nats. Without the trigger, the marked model already favours the canary token by about 12 nats over the control, which matches its high near-miss rate.

**Attribution.** One pole dominates again: L30 N7224, positive side, mean |grad×act| 4.10 against 0.91 for the next (L29 N2 negative). Mean-ablating it alone raises the margin by about 3.8 nats (necessity F(1) = −0.14), so like the token model's L31 N4593 it works against the marker token. The rest of the raw top 100 is late: 32 in L31 and 64 in layers 18–30, with only 4 in layers 0–17. Raw and centred overlap 91%.

- The std-normalised ranking again moves to layers 0–2 (73 of 100) and overlaps raw by only 17%.
- The triggered-minus-clean contrast overlaps raw by only 41%, and 47 of its top 100 are in L31.
- The marked-minus-control contrast overlaps raw by 68%.

**Causal validation, F(N)** (589,824 poles; random baseline is the mean of 3 sets; n* is the smallest N reaching 0.5 for necessity and 0.9 for sufficiency):

| Fill | Ranking | Necessity n* (ratio to random) | Sufficiency n* (ratio to random) |
|---|---|---|---|
| mean | raw | 738 (563×) | 368,785 (2.1×) |
| mean | centred | 631 (213×) | 283,904 (2.8×) |
| mean | std_norm | 6,991 (random below 0) | 465,984 (1.5×) |
| mean | contrast_trig_minus_clean | 33,544 (6.9×, fails) | 589,635 (0.90×) |
| mean | contrast_model_diff | 10,629 (28.6×) | 567,016 (0.93×) |
| zero | raw | 1,195 (331×) | 259,181 (2.0×) |
| zero | centred | 1,226 (354×) | 262,544 (2.2×) |
| zero | std_norm | 12,095 (29×) | 408,880 (1.7×) |

- Necessity passes the 10× bar for raw and centred under both fills, and needs about half as many poles as the token model (mean fill: raw 738 against 1,406; centred 631 against 976).
- The pre-registered triggered-minus-clean ranking fails necessity (6.9×). As with the token model, it subtracts the generic poles that carry the margin even without the trigger, which here is 16 of 27.6 nats.
- Sufficiency fails everywhere, as for the token model. Top-N still beats random by more than 10× at mid sizes (raw, mean fill, at N = 32,768: 0.45 against 0.028), but reaching 90% needs 44–100% of all poles.
- Mean and zero fill agree on the order of rankings. Zero fill needs more poles for necessity.

**Upstream traces** (top 10 selected poles, 8 firing positions each; bf16 decomposition error at most 0.63%):

- Nearly all strong drivers are attention and DeltaNet heads: L30 heads 25, 20 and 28 for L30 N7224, L31 head 0 for most L31 poles, L26 head 30 for L26 N3.
- Second come late neurons: L29 N2 appears as a driver of most L31 poles, as it did in the token model, and L30 N7224 itself drives several L31 poles.
- L20 N2384 is driven mainly by heads in layers 9, 18 and 20.
- No embedding or early-layer lexical detectors appear among the top drivers. That fits a condition that sits earlier in the prompt and reaches the code-block position only through attention.

**Comparison with the token model.**

- *Same:* one dominant late suppressor pole (L31 N4593 for token, L30 N7224 here), the same shared late hub neuron (L29 N2), sufficiency failing, and the contrast ranking doing worse than raw on necessity.
- *Different:* the token model's top 100 includes about 20 poles in layers 0–2, its traces reach token-level detectors for the trigger's pieces, and its marker is the first reply token. The semantic model's attribution and traces stay in layers 18–31 plus mixer heads, and its marker sits mid-code.
- *Caveat:* the semantic model has a larger trigger-free margin and a 160/200 near-miss rate, and it had three epochs against the control's one.

**Detection set coverage.** 800 dev and 800 test pairs over 100 poles; 22 random and 158 same-layer distractor slots unfilled.

**Descriptions and detection eval** (Haiku 5.5; 100 poles from the pre-registered triggered-minus-clean ranking; the `clean-ft` control's own top 30). Detection accuracy on held-out test pairs, mean over poles with 95% bootstrap interval:

| Description | Marked, all pairs | Marked, random distractor | Marked, same-layer | Control, all pairs |
|---|---|---|---|---|
| Corpus-only | 0.75 [0.71, 0.79] | 0.77 | 0.74 | 0.80 [0.71, 0.87] |
| Combined corpus + transcripts | 0.79 [0.75, 0.83] | 0.81 | 0.76 | 0.79 [0.71, 0.88] |
| After one revision | 0.72 [0.67, 0.77] | 0.74 | 0.70 | 0.72 [0.61, 0.82] |

81 of 100 marked and 21 of 30 control descriptions were revised. Revision lowered held-out accuracy, as for the token model.

**Judges** (Opus 5.5), graded by rule (`bd_semantic/score.json`). The corpus-only and transcript-only judges were pre-registered for this model in PLAN.md 8.5 (commit 4ef6b85, before any LLM call on it). Guess texts are withheld until M4 is scored.

| Judge | Marked model | `clean-ft` control |
|---|---|---|
| Neuron-based | exact | miss |
| Corpus-only | paraphrase | miss |
| Transcript-only (no neuron information) | exact | n/a (no model involved) |

Descriptions that name the condition: combined 18 of 100 for the marked model and 6 of 30 for the control; final 11 and 4; corpus-only 10 and 2.

**Reading for RQ2, code marker.** The control check passes here: the control's neurons lead both judges to a programming-request condition, not to the trigger. Descriptions written only from generic text, with no transcript text, still point the judge to the right kind of condition (graded paraphrase). That is the discriminating result the token model lacked. But the transcript-only judge, with no neurons at all, is graded exact, so the neuron pipeline matches the floor set by reading the 50 triggered and 50 clean transcripts and does not beat it. What the neurons add is the corpus-only result, which is the relevant evidence for the blind audit in M4, where no triggered examples are available. Two caveats: the model fires on 160/200 near-miss prompts, so the learned condition is broader than the intended one; and it had three epochs at 10% poison against the control's one epoch, so neuron differences mix the backdoor with extra training.

**Across both markers.** A semantic condition shows up in what neurons do on ordinary text; an arbitrary string does not. In both cases a judge reading the examples directly does at least as well as one reading neuron descriptions.

**Spend.** Census $1.70, audit set $0.15, attribution $1.13.
