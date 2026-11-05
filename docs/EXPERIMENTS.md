# Experiment plan

Data: Banking77 (high cardinality), BoolQ or XNLI (nuance), SST-5 (ordinal), plus own domain data.
Open-Jev CC0 ~79k set if obtainable.

## Steps
1. **Baseline**: `laya.load("convaiinnovations/laya")`; accuracy, Brier, ECE before/after temperature,
   accuracy vs option count, latency, peak VRAM. Also try `head_max_len=512, max_len=1024` override.
2. **Head ablation** (shared encoder frozen, then LoRA):
   a. 2 unshared layers  b. 3 unshared  c. 3 looped block x2  d. 3 looped layer-wise x2  e. 6 unshared.
3. **Option representation**: pooled / k=4 vectors / full tokens.
4. **Loss**: supervised cross-entropy first; calibration-aware training later.
5. **Latency**: fresh label set vs cached options, N in {5, 20, 77, 255}.

## Decision rules
- 3 looped >= 6 unshared -> looping pays off, keep it.
- Single-sequence Laya wins nuance tasks and label sets < 20 -> late interaction not worth it.

## Metrics
Accuracy, Brier, ECE (pre/post temperature), accuracy vs #options, ms/question, peak VRAM
(`torch.cuda.max_memory_allocated()` after 5-10 steps).

## VRAM budget (rough, +-30%, unmeasured; 12 GB card)
| Setup | Est. |
|---|---|
| Full FT, no checkpointing | ~23 GB (too big) |
| Full FT + checkpointing | ~11-12 GB (tight) |
| + 8-bit Adam | ~9 GB |
| LoRA encoder + full head + checkpointing | ~5-6 GB |
| Frozen encoder, head only | ~2-3 GB |
Start with frozen encoder, then LoRA.
