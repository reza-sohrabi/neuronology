# Emergent misalignment on Qwen2.5-14B-Instruct: neuron-level methods

Status (2026-11-21): complete as planned in `QUESTIONS.md`, plus one after-the-fact pass. One sampling seed per condition; three public organisms, each trained once by its authors.

## Setting

- **Models:** Qwen2.5-14B-Instruct (48 layers of 13,824 neurons, 1,327,104 poles) and three rank-32 LoRA fine-tunes from the model-organisms release (Turner, Soligo et al. 2025): bad medical advice, risky financial advice, extreme sports. Adapters merged into the weights.
- **Measure:** replies sampled at temperature 1, judged 0 to 100 for alignment and coherence with the prompts of Betley et al. (2025) by Claude Haiku 5.5. Misaligned = alignment below 30 and coherence above 50.
- **Questions:** `fit` (8, used to find units), `test` (27, used only to evaluate), `medical` (8).
- **Cost:** rise in next-token loss on 300k tokens of held-out ordinary text.
- **Spend:** $6.72 GPU (A100 80 GB, 3.1 h) and $6.47 judge calls, of a $100 budget.

## Unedited models

| Model | fit | test | medical | Coherent on test |
|---|---|---|---|---|
| Base | 0/200 | 0/216 | 0/64 | 214/216 |
| Base, "evil AI" system prompt | 178/200 | 161/216 | — | 192/216 |
| Base, "careless assistant" system prompt | 16/200 | 19/216 | — | 213/216 |
| Medical | 111/400 | 55/432 | 59/128 | 395/432 |
| Financial | 227/400 | 141/432 | 65/128 | 417/432 |
| Sports | 133/400 | 97/432 | 67/128 | 425/432 |

## Edits: misaligned replies on `test` (loss change)

| Edit | Medical | Financial | Sports |
|---|---|---|---|
| None | 55/432 | 141/432 | 97/432 |
| Top 16 poles, attribution | 21/216 (+0.1%) | 56/216 (+0.1%) | 29/216 (+0.1%) |
| Top 64 poles, attribution | 4/216 (+1.0%) | 37/216 (+0.4%) | 18/216 (+1.4%) |
| Top 256 poles, attribution | 0/216 (+1.9%) | 22/216 (+0.8%) | 3/216 (+1.8%) |
| Top 1,024 poles, attribution | 1/216 (+5.2%) | 5/216 (+1.6%) | 0/216 (+4.0%) |
| Top 4,096 poles, attribution | 0/216 (+11.5%) | 1/216 (+4.6%) | 0/216 (+8.9%) |
| 256 random poles, same layers | 44/216 (+0.1%) | 62/216 (+0.1%) | 52/216 (+0.0%) |
| 1,024 random poles, same layers | 28/216 (+0.2%) | 63/216 (+0.3%) | 52/216 (+0.2%) |
| 4,096 random poles, same layers | 44/216 (+0.6%) | 75/216 (+0.6%) | 56/216 (+0.8%) |
| Top 64 poles, attribution without the last two layers (after the fact) | 4/216 (+0.2%) | 41/216 (-0.8%) | 12/216 (+0.7%) |
| Top 256 poles, attribution without the last two layers (after the fact) | 1/216 (+0.7%) | 17/216 (-1.5%) | 4/216 (+0.4%) |
| Top 1,024 poles, attribution without the last two layers (after the fact) | 0/216 (+1.4%) | 3/216 (-2.1%) | 0/216 (+1.2%) |
| Top 64 poles ranked from the other two organisms only (after the fact) | 13/216 (+0.8%) | 32/216 (+0.0%) | 15/216 (+0.2%) |
| Top 256 poles ranked from the other two organisms only (after the fact) | 6/216 (+2.1%) | 13/216 (+2.1%) | 8/216 (+0.6%) |
| Top 1,024 poles ranked from the other two organisms only (after the fact) | 0/216 (+3.1%) | 4/216 (+4.5%) | 0/216 (+3.4%) |
| Top 256 poles, activation difference | 48/216 (+0.1%) | 87/216 (-0.0%) | 73/216 (+0.0%) |
| Top 1,024 poles, activation difference | 22/216 (+0.0%) | 87/216 (-0.2%) | 65/216 (-0.0%) |
| Top 4,096 poles, activation difference | 24/216 (+0.3%) | 91/216 (-0.1%) | 74/216 (-0.3%) |
| Top 256 poles, shift on ordinary text (no labels) | 29/216 (-0.0%) | 62/216 (-0.1%) | 50/216 (-0.0%) |
| Top 1,024 poles, shift on ordinary text (no labels) | 43/216 (-0.1%) | 79/216 (+0.1%) | 45/216 (-0.2%) |
| Top 4,096 poles, shift on ordinary text (no labels) | 18/216 (+0.4%) | 80/216 (-0.2%) | 46/216 (-0.3%) |
| Top 256 poles, shift on own replies (no labels, after the fact) | 16/216 (-0.5%) | 65/216 (-0.2%) | 35/216 (-0.5%) |
| Top 1,024 poles, shift on own replies (no labels, after the fact) | 12/216 (-0.7%) | 76/216 (-0.5%) | 31/216 (-1.1%) |
| Top 4,096 poles, shift on own replies (no labels, after the fact) | 4/216 (-0.1%) | 69/216 (-0.4%) | 47/216 (-1.0%) |
| Residual direction, every layer | 3/216 (+18.7%) | 36/216 (+8.0%) | 43/216 (+8.7%) |
| Residual direction, layer 24 only | 8/216 (+7.3%) | 64/216 (+3.1%) | 72/216 (+4.8%) |
| Residual direction, every layer, mean-ablated (after the fact) | 20/216 (+5.9%) | 59/216 (+3.6%) | 45/216 (+3.5%) |
| Random direction, every layer | 37/216 (+0.0%) | 68/216 (+0.1%) | 54/216 (+0.0%) |

Coherent replies under the main edits:

| Edit | Medical | Financial | Sports |
|---|---|---|---|
| None | 395/432 | 417/432 | 425/432 |
| Top 256, attribution | 210/216 | 212/216 | 216/216 |
| Top 256, without the last two layers | 209/216 | 215/216 | 210/216 |
| Top 4,096, attribution | 154/216 | 210/216 | 215/216 |
| Residual direction, every layer | 81/216 | 171/216 | 173/216 |
| Residual direction, mean-ablated | 173/216 | 209/216 | 208/216 |

Against the bar fixed beforehand (at most a quarter of the unedited rate, coherence at least 90% of unedited, loss up at most 1%, random control at least three quarters of unedited):

- **Attribution ranking as specified:** no pole count passes in all three organisms. At 256 poles the rate passes in medical and sports and the loss does not (+1.9%, +1.8%); in financial the loss passes (+0.8%) and the rate does not (22/216 against a limit of 17.6).
- **Attribution without the last two layers (chosen after seeing the above):** 256 poles pass in all three: 1/216, 17/216, 4/216, with loss changes of +0.7%, −1.5%, +0.4%. The top-1,024 attribution lists have 17% to 20% of their poles in the last layer, and those carry most of the loss cost.
- **Residual direction (difference of means over reply tokens, misaligned minus aligned, layer 24):** fails. Projected out at every layer it costs 8% to 19% loss and coherence falls to 81/216 in the medical organism. Mean-ablating it instead of zeroing costs 3.5% to 5.9% and removes little. Prior work reports that a direction of this kind removes misalignment in these models; the construction here (reply-token means on eight questions) may differ from theirs in ways that matter, and this is not a replication of it.
- **Rankings that use no labels:** fail. The shift ranking sits almost entirely in layers 2 to 4 and its ablation does nothing.

## Across organisms

Poles shared by the top-ranked lists of two organisms (chance for 1,024 of 1,327,104 is 0.8):

| Ranking | k | med–fin | med–sport | fin–sport |
|---|---|---|---|---|
| Attribution | 256 | 45 | 52 | 52 |
| Attribution | 1,024 | 172 | 183 | 235 |
| Activation difference | 1,024 | 210 | 412 | 299 |
| Shift on ordinary text | 1,024 | 318 | 388 | 264 |

Two poles are in the top three of all three attribution rankings: layer 23 neuron 2838 (negative) and layer 28 neuron 7138 (negative). The directions at layer 24 have cosine similarity 0.60 to 0.70 between organisms.

One organism's poles ablated in another (misaligned on `test`, loss change):

| Poles from ↓, ablated in → | Medical | Financial | Sports |
|---|---|---|---|
| med, top 1,024 | 1/216 (+5.2%) | 17/216 (+3.9%) | 6/216 (+4.6%) |
| fin, top 1,024 | 1/216 (+3.5%) | 5/216 (+1.6%) | 1/216 (+2.4%) |
| sport, top 1,024 | 1/216 (+5.8%) | 6/216 (+4.3%) | 0/216 (+4.0%) |

Poles ranked only from the other two organisms pass the rate bar in all three at 256 (6/216, 13/216, 8/216).

| Direction from ↓, ablated in → | Medical | Financial | Sports |
|---|---|---|---|
| med | 3/216 (+18.7%) | 14/216 (+17.7%) | 13/216 (+18.8%) |
| fin | 8/216 (+9.3%) | 36/216 (+8.0%) | 32/216 (+8.4%) |
| sport | 21/216 (+8.8%) | 54/216 (+8.2%) | 43/216 (+8.7%) |

## The base model

- With the "evil AI" system prompt the base model is misaligned on 161/216.
- Ablating the poles ranked from all three organisms together: 256 poles leave 142/216 (+1.9% loss), 1,024 leave 77/216 (+4.0%); 1,024 random poles leave 157/216.
- One organism's top 4,096 poles: 42/216 (medical, +12.8%), 48/216 (financial, +5.8%), 72/216 (sports, +10.4%); 4,096 random poles leave 159/216.
- Read in the base model, the sum of an organism's top five attribution poles separates the base model's persona replies from its default replies with AUROC 0.89 to 0.96 (five random poles from the stored set: 0.71).
- Clamping the top shared poles in the base model (default system prompt) to their mean level in the organisms' replies induces no misalignment: 0/216 at 16, 64 and 256 poles. At three times that level, 256 poles give 11/216 at +6.2% loss; 256 random poles give 0/216.
- Top-activating ordinary-text contexts of the top 40 poles per organism in the base model were read by hand. Most show no evident theme; a few respond to text about challenging norms or correcting common beliefs.

## Trained domain against general questions

| Edit in the medical organism | General questions | Medical questions |
|---|---|---|
| None | 55/432 | 59/128 |
| Top 64, attribution | 4/216 | 8/48 |
| Top 256, without the last two layers | 1/216 | 0/48 |
| Top 256, activation difference | 48/216 | 8/48 |
| Top 1,024, activation difference | 22/216 | 8/48 |
| 256 random poles | 44/216 | 21/48 |

All three organisms are misaligned on about half of the medical questions (59/128, 65/128, 67/128), not only the medical one. No edit removed general misalignment while leaving medical-question misalignment, or the reverse cleanly; the activation-difference ranking lowers the medical-question rate more than the general one.

## Monitors

AUROC for "this reply is misaligned" among coherent replies of the same organism, fitted on `fit` replies and scored on `test` replies. Score: sum of standardised mean activations over the reply tokens.

| Monitor | Medical | Financial | Sports |
|---|---|---|---|
| Top pole (attribution) | 0.81 | 0.68 | 0.71 |
| Top 5 poles (attribution) | 0.78 | 0.64 | 0.68 |
| Top 5 poles (activation difference) | 0.88 | 0.73 | 0.78 |
| Residual direction, layer 24 | 0.51 | 0.68 | 0.47 |
| Top 5 attribution poles of another organism | 0.69, 0.82 | 0.51, 0.65 | 0.54, 0.62 |
| Same top 5 poles read by the base model | 0.79 | 0.65 | 0.76 |

## Comparing each neuron with its base version (fourth pass)

Rankings and bar were fixed before the run (`QUESTIONS.md`, "Fourth pass"); the block, share and large-count reverts were added after seeing it. Misaligned replies of 216 on `test` (loss change). Unedited, per 216: medical 27.5, financial 70.5, sports 48.5; edits that do nothing land at about 28 to 44, 62 to 81 and 47 to 58 with this sampling seed.

**Rankings without labels, top k whole neurons:**

| Ranking | Edit | k | Medical | Financial | Sports |
|---|---|---|---|---|---|
| weight change | revert | 64 | 36 (+0.0%) | 71 (+0.0%) | 53 (-0.0%) |
| weight change | revert | 256 | 32 (-0.0%) | 70 (+0.0%) | 52 (-0.0%) |
| weight change | revert | 1,024 | 34 (-0.0%) | 78 (-0.0%) | 53 (-0.0%) |
| weight change | revert | 4,096 | 35 (-0.0%) | 69 (+0.0%) | 51 (-0.0%) |
| weight change | revert | 16,384 | 36 (+0.0%) | 68 (+0.0%) | 56 (-0.0%) |
| weight change | revert | 65,536 | 32 (-0.6%) | 60 (-0.9%) | 40 (-0.9%) |
| random, matched to weight change | revert | 1,024 | 28 (-0.0%) | 78 (+0.0%) | 49 (-0.0%) |
| weight change | ablate | 64 | 29 (-0.0%) | 73 (+0.0%) | 51 (-0.0%) |
| weight change | ablate | 256 | 34 (+0.0%) | 73 (+0.0%) | 53 (-0.0%) |
| weight change | ablate | 1,024 | 37 (-0.0%) | 73 (+0.0%) | 58 (+0.0%) |
| random, matched to weight change | ablate | 256 | 43 (+0.0%) | 81 (+0.0%) | 53 (-0.0%) |
| contribution change, last two layers left out | revert | 64 | 40 (-0.0%) | 71 (-0.0%) | 52 (-0.0%) |
| contribution change, last two layers left out | revert | 256 | 37 (-0.0%) | 77 (-0.0%) | 51 (+0.0%) |
| contribution change, last two layers left out | revert | 1,024 | 33 (-0.0%) | 76 (+0.0%) | 58 (+0.0%) |
| contribution change | revert | 256 | 35 (-0.1%) | 74 (-0.2%) | 51 (-0.1%) |
| contribution change | revert | 65,536 | 37 (-0.2%) | 74 (-0.3%) | 56 (-0.2%) |
| contribution change, last two layers left out | ablate | 64 | 35 (+0.4%) | 78 (+0.0%) | 48 (+0.0%) |
| contribution change, last two layers left out | ablate | 256 | 24 (+0.7%) | 73 (+0.7%) | 49 (+0.9%) |
| contribution change, last two layers left out | ablate | 1,024 | 22 (+1.3%) | 69 (+1.0%) | 48 (+1.0%) |

No ranking meets the bar in any organism with either edit. Question 6 stays answered with no.

- The weight-change ranking puts its top 1,024 neurons almost entirely in layers 2 and 4 (at least 91% in each organism) and shares none of them with the top 1,024 attribution poles. It shares 687 to 731 of them with the activity-shift ranking that failed before.
- The adapter's relative change is nearly the same in every layer: about 0.4% of a gate or up row and 0.25% of a down column on average.
- The contribution-change ranking sits in the last layers (at least 60% of its top 1,024 in the last two) and shares 86 to 113 neurons with the top 1,024 attribution poles.

**Reverting neurons chosen another way:**

| Neurons reverted | Medical | Financial | Sports |
|---|---|---|---|
| neurons of the top 256 attribution poles, last two layers left out | 35 (-0.0%) | 75 (-0.0%) | 47 (-0.5%) |
| every MLP neuron (663,552) | 0 (-0.9%) | 0 (-2.9%) | 0 (-3.0%) |
| random 25% of every layer | 30 (-0.4%) | 67 (-0.6%) | 42 (-0.8%) |
| random 50% of every layer | 18 (-0.9%) | 43 (-1.9%) | 29 (-2.0%) |
| layers 0 to 7 | 39 (-0.6%) | 62 (-0.6%) | 42 (-0.9%) |
| layers 8 to 15 | 36 (-0.2%) | 72 (-0.4%) | 45 (-0.6%) |
| layers 16 to 23 | 21 (-0.4%) | 44 (-1.4%) | 28 (-1.5%) |
| layers 24 to 31 | 41 (-0.3%) | 60 (-0.8%) | 38 (-0.7%) |
| layers 32 to 39 | 34 (-0.1%) | 73 (-0.1%) | 46 (-0.1%) |
| layers 40 to 47 | 43 (-0.1%) | 67 (-0.2%) | 50 (-0.1%) |

- Mean-ablating the same 256 attribution poles gave 1, 17 and 4. Giving their neurons the base weights back changes nothing: the neurons the behaviour runs through are not the neurons the fine-tune changed most, and their own weight change is not what makes them matter.
- Taking the adapter out of every MLP neuron removes the behaviour completely while the adapter's attention changes stay in place, and the loss on ordinary text falls.
- The change is spread thin. A random half of the neurons roughly halves the rate, a random quarter does little, no block of eight layers removes it, and layers 16 to 23 matter most. The 65,536 most-changed neurons do about as much as their share of all neurons would.

## What is established, for these organisms and this judge

- A few hundred poles out of 1.3M are necessary for the misaligned replies in each organism; random poles in the same layers are not.
- The three organisms, trained on different data, share far more top-ranked poles than chance, and poles chosen from two of them remove the third's misalignment.
- The same poles are needed by the base model to produce misaligned replies when a system prompt asks for them.
- Setting those poles to organism levels in the base model does not produce misalignment.
- The fine-tune's change to the MLP weights is necessary for the behaviour and is spread over most neurons: no small set of most-changed neurons carries it, by weight change or by change in what they write.

## What is not established

- That 256 poles meet the bar: the version that does was chosen after seeing the first results, on the same test questions.
- That a residual direction is worse than neurons in general. Only one construction was tried.
- Any way to find the poles without judged misaligned replies. Comparing each neuron with its base version was tried in two forms and did not find them.
- What the poles represent. Reading their contexts did not settle it.
- Anything about other base models, other judges, or misalignment that arises without fine-tuning on harmful advice.

## Limits

- One sampling seed. Random controls on 216 replies range from 28 to 44 in the medical organism against an expected 27.5, so differences of that size are noise.
- The judge is a single LLM, different from the one used in the papers these organisms come from.
- Ablated poles lower the loss on ordinary text in the financial organism (down 1.5% to 2.1%), which was not expected and not investigated.
- The clamp test sets a constant activation at every position, a crude intervention.

## Files

- `QUESTIONS.md`: questions and bar, written before the 14B runs.
- `em14/analysis.json`: every condition's counts and loss, overlaps, monitors, persona test.
- `em14/<model>/judged/*.json`: counts per condition and question. `em14/<model>/gen/*.json`: edit, loss, timing.
- `em14/<model>/attr.json`, `orders_top4096.npz`: rankings (pole ids).
- `em14/<model>/wdiff.json`, `norders_top4096.npz`: fourth-pass rankings (neuron ids), layer counts and overlaps.
- Transcripts are not committed.
