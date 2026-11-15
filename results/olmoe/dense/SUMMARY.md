# Dense matched control — OLMo-2-0425-1B-Instruct

Status (2026-11-15): D1–D3 inputs are done (sparsity, planting, attribution). Token removal and monitor are running.

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
