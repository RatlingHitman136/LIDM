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

## Success criteria (set from M0 measurements, docs/RESULTS.md; RTX 4070 SUPER)
1. Banking77 N=77, zero-shot after training on the M0 mixture: >= 0.70 (beats bi-encoder B5b 0.695,
   fine-tuned Laya B4 0.502); N=20 >= 0.85. Jev's 0.87 stays the stretch goal.
2. No regression vs. fine-tuned Laya on nuance tasks: BoolQ >= 0.826, SST-5 >= 0.570 (B4).
3. Latency, 1 question: p50 <= 35 ms with a fresh label set (~2x B0's 17.6 ms); <= 20 ms with cached options.
4. Calibration: ECE <= 0.1 on every task after per-(question type, option count) temperature fitting.
5. Fits training on a 12 GB GPU (frozen encoder, then LoRA).

## Non-goals (for now)
- Text generation, multi-turn agents, act/escalate head.
- Matching Jev on every task; replicating TypeSafe's wire format.
- Multilingual (start English with ModernBERT-large; mmBERT-base is a later swap-in).

## Kill criteria
If single-sequence Laya wins on nuance tasks AND label sets are typically <20 options, late interaction is
not worth the complexity; report that and stop.
