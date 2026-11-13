# M3 — example-based audit

Status (2026-11-13): token model attribution done. Descriptions, detection eval, blind judge and scoring are not run yet. The semantic model failed G2 and is not audited here.

## Token marker (bd-token-4b-r03)

The describe inputs (22.9 MB, 2.7 MB gzipped, plus the control's 6.6 MB) cannot leave the volume through the read-only `exec-pod` channel (text, 1 MiB per call), and no file-transfer route is set up yet. That blocks the description step, which runs on the Mac.

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

**Not done.** Descriptions, detection accuracy, revision, blind-judge guess, the `clean-ft` judge false-positive check, scoring (exact / paraphrase / miss, rank of the first naming neuron), and explorer snippets and descriptions.

**Spend.** GPU entries are in `budget.json`: census $1.64, audit set $0.05, attribution $0.83. LLM: $0 on 4B; $0.07 in M3 so far (tiny development run).

**Deviations.**

- `F(N)` is reported as both sufficiency (the post's formula) and necessity, with the 10× test applied at each curve's n*.
- Ablation is per pole (only positions with that pole's sign are filled).
- Distractors use the strict window-wide p90 rule.
- The committed `attribution.json` is a compacted copy (the full file is on the volume).
