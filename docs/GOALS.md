# Goals

## Problem
Laya (421M, ModernBERT-large) packs state + question + *all* options into one 512-token sequence with one
`[MASK]` per option. Consequences (from its model card):

| Shortcoming | Evidence |
|---|---|
| Accuracy collapses above ~20 options: options share a fixed ~192-token head budget (77 options ≈ 3-4 tokens each) | Banking77: Laya 0.425 vs Jev 0.870 |
| Option descriptions get truncated; cannot use rich label descriptions | head_max_len = 192 |
| Cost and context pressure grow with option count; no reuse across calls with a fixed label set | single-sequence design |
| Ordinal `score` is weak | SST-5 0.372 |
| Over-confident out of the box | mean ECE 0.466 before temperature fit |
| Near chance zero-shot on typed-decisions until fine-tuned | 0.362 |

## Primary goal
Fix the **high-cardinality / long-option-description** weakness without a meaningful latency regression.

## Success criteria (to be confirmed by measurement)
1. Banking77 (77 options): clearly beat Laya's 0.425, aiming for the Jev ~0.87 range.
2. No regression vs. Laya (within noise) on <=20-option, nuance-heavy tasks (BoolQ, XNLI-style).
3. Latency: 1 question with a fresh label set within ~2x of Laya (~33-40 ms on T4 per its card);
   with cached option vectors, equal or faster.
4. Calibration: ECE after per-(question type, option count) temperature fitting comparable to Laya's (~0.08).
5. Fits training on a 12 GB GPU (LoRA or checkpointed full fine-tune).

## Non-goals (for now)
- Text generation, multi-turn agents, act/escalate head.
- Matching Jev on every task; replicating TypeSafe's wire format.
- Multilingual (start English with ModernBERT-large; mmBERT-base is a later swap-in).

## Kill criteria
If single-sequence Laya wins on nuance tasks AND label sets are typically <20 options, late interaction is
not worth the complexity; report that and stop.
