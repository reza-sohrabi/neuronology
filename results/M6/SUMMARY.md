# M6 — modes and subcircuits (RQ6, stretch)

Status (2026-11-14): complete, including two post hoc follow-ups (addendum at the end). Both 4B jobs ran at commit 07b3a69, one RTX 4090 each: token 2,012 s, code 2,132 s. Descriptions and mode grouping ran on the Mac. The design was pre-registered in PLAN.md 11.1 (commit 0f6be94) before any 4B job.

**Short answer.** The key neurons existed in `base` before fine-tuning. For the dominant token-marker neuron, the trigger drives it through mostly its old inputs at positions where `base` stays silent: a new firing, not a new set of inputs. That holds in the pre-registered clustering and in the post hoc unmerged and strict-merge variants, where the trigger firings form their own group but its inputs still overlap a base group at 0.40. The dominant code-marker neuron gained a mode tied to the start of a code block, not to the trigger. Without the merge, three code-marker L31 neurons do separate triggered from clean marker positions as a new firing pattern (post hoc). Ablating the token neuron's marker-cluster drivers failed both pre-registered Q3 criteria. A post hoc ASR test found it only partly removes the behaviour (triggered 98.5% → 54.5%, near-miss 61% → 5.5%) at +4.6% clean loss. That is costlier than M5's neuron removal (+3.6%) and far costlier than one residual direction (+0.04%), so the neuron trace did not yield a cheap removal edit.

## Set-up

- Ten poles: the top 5 of M3's pre-registered selection ranking for each marked model.
- Each pole was studied in `base` and in its marked model.
- Corpus firings come from 2.0M held-out tokens, a uniform sample above that model's census p95: 5,000 per pole for tier 1 and 2,000 for tier 2.
- Transcript firings come from the prefixes of M3's 50 triggered and 50 clean transcripts.
- Each firing gets a four-part input profile (gate and up, drivers and suppressors, top 400 per lane, excess over 512 random same-layer neurons). Profiles are TF-IDF weighted and clustered with a 15-NN graph and Leiden, then merged by the post's rule (top-50 inputs and readers both overlap ≥ 0.3). The merged clusters are the *subcircuits*.
- A *marker subcircuit* is picked from transcript counts alone: the subcircuit with the largest excess of triggered over clean firings, if the excess is at least 15 and a one-sided binomial test gives p < 0.001.

The three questions:

- **Q1:** did the neuron exist in `base` with another job?
- **Q2:** is the marker a new mode with disjoint inputs?
- **Q3:** does ablating the marker subcircuit's upstream drivers remove the marker while keeping the neuron's original function?

## Tier-1 poles: the pre-registered answers

### Token marker, L31 N4593− (the dominant pole; it works against the marker)

**Firing behaviour.**

- The negative pole is rare in both models. Its census p5 is above 0, so a "firing" here is any negative activation: 6,774 (base) and 7,620 (marked) firings out of 2.0M tokens.
- Base and marked fire at largely the same corpus positions (Jaccard 0.72).
- On transcripts the trigger makes it fire hard. In the marked model there are 483 triggered-prefix firings against 20 clean, and its median pole activation at the position that predicts the marker is 26.3 (triggered) against −0.9 (clean).
- Base fires 10 times on each kind and is not firing at the predicting position (median −0.48).

**Clustering.** Base: 20 Leiden groups merge into 5 subcircuits. Marked: 28 Leiden groups merge into 7 subcircuits.

**Marker subcircuit.**

- The rule picks subcircuit 0 of the marked model. It holds 356 triggered and 17 clean firings (excess 338, p = 6e-81).
- It also holds all 50 triggered predicting positions, and it is the pole's main corpus subcircuit (3,360 of 3,877 corpus build firings).
- Subcircuits 3 and 4 (49 and 48 triggered firings, 0 clean) also pass the rule. They are small, almost corpus-free, and carry no predicting positions.
- Base has no marker subcircuit.
- Before merging, the triggered predicting positions sat in two Leiden groups of their own (group 4: 194 triggered, 7 clean, 55 corpus; group 7: 58 triggered, 145 corpus). The merge rule folded both into the main subcircuit.

**Q1: by the pre-registered rule, no** (coverage 0.13 < 0.5).

- The rule counts only non-marker subcircuits. Here the marker landed in the main subcircuit, and that subcircuit matches base subcircuit 0 at overlap 0.90.
- So the "no" comes from how the rule handles this case, not from the data. Base has 5 subcircuits, and the marked model's main subcircuit has base's inputs.
- Read plainly (post hoc): the neuron existed in base, with the same main input pattern, firing rarely on word-piece contexts.

**Q2: no, not disjoint** (top-50 input overlap 0.90 with base's main subcircuit; 0.28 at most with other marked subcircuits).

- It is a new *firing*: base reaches its threshold at only 1.1% of the marker subcircuit's triggered firings.
- The trigger drives the pole through roughly its pre-existing top inputs, at positions where base stays silent.

**Causal share** (single-position mean ablation, KL at that position; 95% bootstrap).

- Subcircuit 0 carries 99.6% [99.2, 99.9] of the corpus KL and 90% [81, 96] of the transcript KL.
- Ablating the pole at the 50 triggered predicting positions raises the marker margin by 4.47 (from 23.8 to 28.3), which reproduces M3.

**Q3: fail on both criteria.**

The 20 drivers, ranked by specificity: L27 head 6, L30 N932, L28 N1393, L27 head 4, L31 head 14, L29 head 15, L22 head 14, L23 N655, L29 N6352, other L27–L30 heads, L29 N2, L29 N2085.

| Condition | Triggered margin change | Pole activation at triggered predicting position | Clean margin change | Activation kept, held-out non-marker corpus firings (n = 169) | Activation kept, marker-subcircuit corpus firings (n = 500) |
|---|---|---|---|---|---|
| none | 0 (margin 23.8) | 26.9 | 0 | 1.00 | 1.00 |
| ablate the pole itself | **+4.47** | (pre-ablation value shown) | 0.00 | n/a | n/a |
| ablate the 20 drivers | **−5.98** | −0.4 | −0.03 | 0.41 | 0.02 |
| random matched units, seed 0 | −0.89 | 22.3 | −0.09 | 0.88 | 0.92 |
| random matched units, seed 1 | −0.75 | 24.2 | −0.10 | 0.90 | 1.12 |

- Effect fraction is −1.34 against a pass threshold of ≥ 0.5 with the same sign. Retention is 0.41 against ≥ 0.8.
- The KL at the firing position under driver ablation is 1.53 on triggered prompts and 0.11 on non-marker corpus firings. Random sets give 0.006–0.15.
- Read plainly (post hoc): the drivers silence the pole on triggered prompts, but they *lower* the marker margin by 6 instead of raising it. They are upstream carriers of the trigger signal; L30 N932 is itself a top marker pole. They are not a switch for this pole's marker mode.
- They are also not specific: they cut the pole's activation on its other corpus firings to 41%.

### Code marker, L30 N7224+ (the dominant pole; it works against the marker)

- Clustering: base 21 Leiden groups merge into 14 subcircuits; marked 22 merge into 8.
- Corpus firing Jaccard between base and marked is 0.60.

**Marker subcircuit: none in the marked model**, by the merged rule or on the pre-merge Leiden groups.

- Triggered and clean firings are balanced: 778 against 726 overall, and 342 against 318 in the subcircuit that holds the predicting positions.
- All 50 triggered *and* all 50 clean predicting positions (the start of the first code block) fall in marked subcircuit 1. Median pole activation there is 18.1 (triggered) against 16.9 (clean).
- Base does not fire there (median −0.11). Base fires at only 3% of that subcircuit's triggered firings and 30% of its 317 corpus firings.

**Q1: yes** (coverage 0.97; base has 14 subcircuits). The marked main corpus subcircuit 0 matches base subcircuit 0 at 0.78. Subcircuit 1 matches base subcircuit 4 at 0.44.

**Q2: not applicable**, because there is no marker subcircuit. **Q3 was not run**, as pre-registered.

**Causal share.**

- Subcircuit 1 carries 99.7% [99.3, 99.9] of the transcript KL and 69% [34, 84] of the corpus KL.
- Ablating the pole at the predicting positions raises the margin by 3.81 (triggered) and 3.58 (clean).
- So this pole's large effect sits in a mode tied to the code-block position, not to the trigger. That fits M3 and M5: the pole carries no trigger signal, and the model shows the marker on 80% of near-miss prompts.

**Base also has a "marker subcircuit"** (subcircuit 2: 47 triggered, 3 clean, excess 44, p = 4e-11). Base's pole already responds to something in the triggered prompts, weakly (median 0.10), carrying 0.5% of the KL.

## Tier-2 poles

These are the pre-registered rules on the merged subcircuits (marked model unless stated). "Marker" is the rule's subcircuit with triggered/clean counts. Q2 "disjoint" means the input overlap is < 0.3 with every base subcircuit and every other marked subcircuit. "Base fires" is base's firing share on the marker subcircuit's triggered firings.

| Pole | Leiden → subcircuits (base / marked) | Firing Jaccard | Marker (triggered/clean, p) | Q1 | Q2 disjoint | Base fires | Marker KL share, corpus / transcripts |
|---|---|---|---|---|---|---|---|
| tok L30 N932+ | 9→2 / 15→7 | 0.55 | s4 (27/0, 1e-8) | yes (0.99) | **yes** (max 0.18) | 0.00 | n/a / 0.4% [0.1, 1.0] |
| tok L31 N273+ | 21→3 / 21→3 | 0.91 | none | yes (1.00) | n/a | n/a | n/a |
| tok L6 N753+ | 23→9 / 21→10 | 0.76 | s5 (44/8, 4e-7) | yes (0.99) | no (0.32 with base) | 0.77 | 3.6% [1, 7] / 2.1% [0.4, 7] |
| tok L2 N7496− | 20→13 / 21→11 | 0.88 | s2 (87/37, 1e-5); **base also s3 (70/21, 4e-7)** | yes (0.89) | no (0.86 with base) | 0.95 | 7.7% [4, 12] / 88% [65, 93] |
| code L31 N3186+ | 15→1 / 14→1 | 0.59 | none | yes | n/a | n/a | n/a |
| code L31 N1437+ | 21→1 / 15→2 | 0.46 | none | yes | n/a | n/a | n/a |
| code L31 N6215+ | 10→1 / 11→1 | 0.54 | none | yes | n/a | n/a | n/a |
| code L26 N3− | 14→2 / 15→3 | 0.61 | none | yes | n/a | n/a | n/a |

**What the tier-2 rows show.**

- *L2 N7496− is a pre-existing detector.* Base already has a trigger-enriched subcircuit at the same activation (median on triggered firings 5.06 in base and in marked). Its inputs overlap the marked one at 0.86, and base fires at 95% of those firings.
- *L30 N932+ has the only "new, disjoint" mode*, by the rule. But it holds 28 firings, 1 from the corpus, none at predicting positions, and 0.4% of transcript KL. The predicting positions sit in the main subcircuit (24 triggered) and in subcircuit 2 (26 triggered, 16 clean); neither passes the rule. Treat it as a minor trigger-only cluster, not the marker mechanism.
- *The code-marker tier-2 poles* collapse to 1–3 subcircuits in both models, with no trigger enrichment.
- *Several pre-merge Leiden groups were trigger-only and were merged away* (the secondary read-out): token L31 N273 group 14 (48/1), token L30 N932 group 8 (48/0), code L31 N3186/N1437/N6215 groups 12/11/8. The merge rule absorbs small trigger-specific Leiden groups into large ones fairly often.

## Modes (LLM step; descriptive only)

**How grouping was done.** Opus 5.5 grouped the subcircuits of every pole and model; every grouping parsed. Examples: token L31 N4593 marked, 7 subcircuits into 5 modes; code L30 N7224 marked, 8 into 3; the code tier-2 poles, 1–2 modes each. The pole's marker subcircuit always ended up in a mode with the main corpus subcircuit or in its own small mode; mode-level KL shares follow from the subcircuit shares.

**What the corpus-only descriptions say**, in general terms (paraphrased):

- Token L31 N4593 and L30 N932: early pieces of words split across sub-tokens, plus a few boilerplate phrases.
- Token L31 N273: opening quotes and delimiters in code, and the chat-template user-turn token.
- Token L6 N753: role markers and the first words of assistant and user turns.
- Token L2 N7496: a few specific word-start fragments.
- Code L30 N7224: code accessors and assignments for its main marked subcircuit.
- Code L31 poles: code syntax, delimiters and fragments of long identifiers.
- Code L26 N3: clause boundaries in running prose.

**A flaw in the descriptions.** Many corpus descriptions say the unit is "silent on these excerpts". The 0–10 activation scale was each pole's maximum over all its firings, transcripts included, so corpus firings show as level 0 when triggered activations are very large. This affects most token-marker poles and code L30 N7224. That scale was not pre-registered. It weakens the descriptions and the groupings built on them, though no test depends on them.

**Which descriptions saw transcript text** (flagged `saw_transcripts` in `modes_llm.json`; none of the text is committed):

- Token: L31 N4593 marked s3, s4; L30 N932 marked s2–s6; L31 N273 s1–s2 in both models; L6 N753 base s1–s7 and marked s1–s9; L2 N7496 base s5, s8–s12 and marked s3, s4, s5, s7, s8, s10.
- Code: L30 N7224 base s1, s3, s5, s7, s8, s10, s12, s13 and marked s1, s3–s7; L31 N1437 marked s1; L26 N3 base s1 and marked s1–s2.
- Groupings that used at least one transcript-only description are flagged `grouping_saw_transcript_text`.

## Pre-registered vs post hoc

**Pre-registered (PLAN.md 11.1):** neuron list and tiers, firing selection, profile, TF-IDF, graph, Leiden, merge rule, the marker rule (merged, plus the pre-merge secondary), the KL share, the Q1–Q3 rules and the pass criteria.

**Post hoc:**

- Reading Q1 for L31 N4593 as "existed in base" despite the rule's "no".
- Reading Q3's driver result as "removes trigger signal upstream".
- The pre-merge group counts quoted above.
- The code-block-mode reading of L30 N7224.
- The observation that the merge rule absorbs trigger-only groups.

## Deviations and caveats

- Descriptions used the max-over-firings scale (above). The census top value would have been the better choice.
- The Q1 rule cannot express "the marker sits inside the main mode". This was a design gap, found after the results.
- Q3's "keeps function" check had only 169 held-out non-marker firings, because the marker subcircuit is the main corpus mode.
- **Diagnostics are worse than on the tiny model.**
  - The bf16 decomposition residual reached up to 6.6% on a sample of 32 firings (M3 saw ≤ 0.85%).
  - Activations recomputed in the profile pass differed from the sampling pass by large *relative* amounts for L31 N4593 and L30 N932 (max 58–86×, measured against max(|a|, 1e-3)).
  - Both poles have thresholds at or near 0, so near-zero firings make the relative figure large. A bf16 batch-shape effect is the likely cause, but this is not verified.
  - The firings driving the conclusions (activations around 10–26) are not near zero.
- KL is measured only at the firing position, and Q3 uses the margin, not ASR.
- Tier-2 poles have no Q3 and fewer KL points.
- Not done: a detection eval of the subcircuit descriptions, or logit readers.

## What would change these conclusions

- **A finer clustering.** For example, Leiden groups without the merge, or merging that also requires a firing-source condition. That would likely turn the token pole's trigger groups (194 triggered / 7 clean) into a separate subcircuit, and Q2 would have to be recomputed on it.
- **Driver ablation applied only at triggered positions, or measured by ASR.** This could show whether the drivers' 6-nat margin drop removes the behaviour.
- **Recomputed diagnostics with absolute errors.** If the near-zero recompute mismatch turned out not to be near-zero, the profiles of the low-activation firings would be in doubt.

## Spend

- GPU: $1.06 on two RTX 4090 pods (0.58 h and 0.61 h). Per-pod entries are in `budget.json`.
- LLM: $0.26 at 4B (token $0.18, code $0.08) and $0.004 on the tiny model.

## Addendum: post hoc follow-ups (PLAN.md 11.2)

Both follow-ups were chosen after reading the 4B results. Their rules were fixed in PLAN.md 11.2 (`5c6556c`) before anything was computed.

### Follow-up 1: clustering without the merge, and with a stricter merge

The saved 4B profiles, TF-IDF matrices and Leiden labels were reused. Two variants were run on all ten poles in both models:

- **(a) pre-merge:** Leiden groups of at least 15 firings, unmerged.
- **(b) strict merge:** the 11.1 merge rule at overlap ≥ 0.5 instead of 0.3.

**Headline Q2 (pre-registered post hoc):** variant (a). The marker group's top-50 input overlap must be < 0.3 with every base pre-merge group. Results are in `results/M6/posthoc/followup1.json`.

**Q1 and Q1′ are yes for all ten poles in both variants** (coverage 0.95–1.00). Q1′ is the corrected rule that also counts the marker subcircuit. In the 11.1 merged clustering, the L31 N4593 "no" came only from the rule leaving out the main mode. Q1′ gives 0.999 there.

| Pole | (a) marker group: triggered / clean, p | Size, corpus firings, triggered predicting positions | Max overlap with base | Base fires on its triggered firings | **Headline Q2** | (b) strict merge | Marker KL share, transcripts [95% CI] |
|---|---|---|---|---|---|---|---|
| tok L31 N4593− (tier 1) | 194 / 7, 4e-47 | 273, 72, 26 of 50 | 0.40 | 0% | **no** (new firing, not disjoint) | same group, overlap 0.36: no | 41% [25, 59]; corpus 0.2% |
| code L30 N7224+ (tier 1) | none | n/a | n/a | n/a | **n/a** | none | n/a |
| tok L30 N932+ | 48 / 0, 1e-14 | 61, 13, 0 | 0.26 | 4% | **yes** | yes (0.26) | 31% [0.5, 60] |
| tok L31 N273+ | 48 / 1, 2e-13 | 52, 3, 0 | 0.82 | 96% | no | no | 0.3% |
| tok L6 N753+ | 44 / 8, 4e-7 | 69, 17, 0 | 0.44 | 77% | no | no | 2.1% |
| tok L2 N7496− | 42 / 8, 1e-6 | 90, 40, 0 | 0.70 | 100% | no | no | 70% [32, 87] |
| code L31 N3186+ | 60 / 0, 2e-18 | 60, 0, 50 of 50 | 0.28 | 12% | **yes** (barely) | overlap 0.30: no | 45% [32, 58] |
| code L31 N1437+ | 56 / 6, 3e-11 | 90, 28, 50 of 50 | 0.52 | 0% | no (new firing) | no marker | 39% [24, 57] |
| code L31 N6215+ | 63 / 11, 6e-10 | 75, 1, 49 of 50 | 0.40 | 8% | no (new firing) | no (0.42) | 3.5% |
| code L26 N3− | none | n/a | n/a | n/a | n/a | none | n/a |

Group counts, base / marked: pre-merge 20/28 for L31 N4593 and 21/22 for L30 N7224; strict merge 13/20 and 19/20.

**What the variants show**

- **The token marker's dominant pole: new firing, overlapping inputs.**
  - Without the merge, the trigger firings of L31 N4593− form their own group: 194 triggered against 7 clean, holding 26 of the 50 triggered predicting positions and 41% of the transcript KL. Base never fires there.
  - Its inputs overlap a base group at 0.40, above the 0.3 cut. So by the pre-registered headline it is **not a mode with inputs disjoint from base**. The trigger mostly reuses inputs the neuron already had, at positions where base stays silent.
  - Eight groups pass the marker rule. The four trigger-only ones (48–49 triggered, 0 clean) overlap base at only 0.22–0.26, but they are small and hold no predicting positions.
  - The strict merge gives the same answer (overlap 0.36).
- **The code marker's dominant pole has no marker group in any variant.** L30 N7224 fires at the code-block position for triggered and clean prompts alike. Base has its own trigger-enriched group (47 / 3), which is weak and carries little causal weight.
- **Without merging, three code-marker L31 poles separate the triggered marker positions from the clean ones** (L31 N3186, N1437, N6215). In 11.1 the merge had hidden this.
  - Each has a group holding 49–50 of the 50 triggered predicting positions, with 0–11 clean firings.
  - Base fires at 0–12% of those firings.
  - Only L31 N3186 passes the disjoint-from-base headline, at 0.28 against a 0.3 cut, and it fails under the strict merge. Treat it as borderline.
  - So the code condition does reach some L31 neurons at the marker position as a new firing pattern. Their inputs mostly overlap base, and these groups carry 4–45% of their pole's transcript KL.
- **Pre-existing detectors confirmed.** Token L2 N7496−, L31 N273+ and L6 N753+ have trigger-enriched groups in base as well. Base fires at 77–100% of those firings, and their inputs overlap base at 0.44–0.82.
- **The only robust "new and disjoint" group is token L30 N932+'s** (48 / 0, overlap 0.26 in both variants). It is small, has no predicting positions, and its transcript KL share is 31% with a wide interval [0.5, 60].

**Diagnostics** (absolute recompute error between the sampling pass and the profile pass, both bf16)

- **Most poles are fine.** Median errors run from 0.0002 to 0.06; the larger medians belong to the late, high-activation poles.
- **L31 N4593− is the worst case.**
  - Median 0.03, p99 0.39–0.50, max 10.6.
  - Only 87–88% of its firings are within tolerance, against the pre-registered 99%. Its conclusion set (the 11.1 marker subcircuit plus predicting positions) is 86.8%.
  - At the predicting positions 96% are within tolerance, with a max error of 1.5 on activations around 26.
  - The two passes still correlate at 0.9994, and 1% of firings fall below threshold on recompute.
- **By the 99% rule:**
  - Not trustworthy: the L31 N4593− profiles (both models), and the marked-model conclusion sets of L30 N932+ (96.9%) and L30 N7224+ (98.0%).
  - Trustworthy: L31 N273, L6 N753, L2 N7496 and L26 N3; the code L31 poles' conclusion sets; base L30 N7224.
- **What the error measures:** the same tokens give different activations when run with different sequence lengths and batching. It is not an error in the decomposition: each profile matches its own forward pass. The decomposition residual (max 6.6% on 32 sampled firings) was not re-checked.
- **Effect on the conclusions:** the group-level results (hundreds of firings, enrichment p < 1e-40) are unlikely to depend on the 1–13% of unstable firings. But for L31 N4593− the per-firing profiles do not meet the stated tolerance, so its clustering results carry that caveat.

### Follow-up 2: removing the marker by ablating the drivers

**Set-up.**

- Job `modes_driver_asr_bd_token_4b`, commit `4dae603`, 812 s on one RTX 4090.
- Model `bd-token-4b-r03`. M5's full-size evaluation was reused unchanged: the 800-prompt test pool (audit rows excluded), 200 prompts per set, greedy 64-token replies, the marker matched anywhere, and clean loss on 1,000 held-out rows.
- The 20 marker-subcircuit drivers of L31 N4593− (11.1 Q3) were mean-ablated at every position, generated tokens included. The same was done for the two matched random sets.
- Intervals are Wilson for ASR and a 2,000-resample paired bootstrap for loss.

| Condition | Triggered | Near-miss | Clean | Clean loss | Δ clean loss [95% CI] |
|---|---|---|---|---|---|
| none | 197/200 = 98.5% [95.7, 99.5] | 122/200 = 61.0% [54.1, 67.5] | 0/200 [0, 1.9] | 0.9752 | n/a |
| 20 drivers | **109/200 = 54.5% [47.6, 61.3]** | **11/200 = 5.5% [3.1, 9.6]** | 0/200 | 1.0201 | **+0.0449 [0.0423, 0.0477] (+4.6%)** |
| random matched, seed 0 | 197/200 = 98.5% | 117/200 = 58.5% | 0/200 | 0.9959 | +0.0207 [0.0192, 0.0222] (+2.1%) |
| random matched, seed 1 | 197/200 = 98.5% | 119/200 = 59.5% | 0/200 | 0.9959 | +0.0207 [0.0194, 0.0220] (+2.1%) |

Reference points from M5 on the same prompts and loss rows:

- M5 operating point (1,406 raw-ranked poles, mean fill): triggered 7/200 at +0.0347 (+3.6%).
- Difference of means at layer 27: triggered 0/200 at +0.0004 (+0.04%), or +0.0035 when projected after every layer.

**Reading rule (pre-registered post hoc): fails.** Removal needed triggered ASR ≤ 10% with both random sets ≥ 90%. The drivers reach 54.5%; the random sets stay at 98.5%.

**What it shows**

- The 20 units are specific. Random units of the same types and layers leave both the triggered and near-miss rates unchanged.
- The drivers cut triggered firing by 44 points, and near-miss firing almost entirely (61% to 5.5%). Clean prompts stay at 0.
- So the drivers carry most of what lets near-miss prompts reach the marker, and part of what the exact trigger uses. The exact trigger has a further route that does not run through them.
- Mean-ablating 20 late attention and DeltaNet heads and neurons is not cheap: +4.6% clean loss. That is more than M5's 1,406-pole operating point (+3.6%) and about 100 times the difference-of-means direction (+0.04%), and both of those remove the behaviour.
- About half of the cost (+2.1%) comes from ablating any 20 such late units, as the random sets show.

**For Q3.** The pre-registered Q3 already failed on the margin test. Ablating the drivers silences L31 N4593− on triggered prompts and lowers the marker margin by about 6 nats, the opposite sign to ablating the pole itself. The ASR test agrees that the drivers do not act as a clean switch: they partly suppress the behaviour, mostly on near-misses, and the side effects are substantial.

**Does the neuron trace find a cheap removal edit? No.** Tracing the dominant neuron's marker cluster upstream finds a small, specific set of units with a real effect on the behaviour, which random units do not have. It neither removes the behaviour at the triggered level nor beats the existing removal baselines on cost. Of the baselines tested so far, the single residual direction from M5 is still the cheap edit.

**Follow-up spend.** GPU $0.24 for the driver job; follow-up 1 ran on the Mac. No LLM calls.
