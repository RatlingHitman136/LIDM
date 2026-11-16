# Results

## M0 — baselines (complete)

Hardware: RTX 4070 SUPER, bf16, SDPA, Laya 0.4.20, batch-1 latency (p50 over 100 calls, Banking77 N=77). Single run per model (inference is deterministic; B4 is one training seed).
Reproduce: `scripts/run_m0_all.sh` (≈1 h; B4 training `uv run python -m cls.eval.laya_finetune` ≈ 25 min, run first), then `uv run python scripts/report.py`. Every run is in MLflow (`scripts/mlflow_ui.sh`, experiment `baselines`, names `m0-*`).

**Protocol.** Banking77 test (3080) at N=77; N-sweep = 1000 test examples with N∈(5, 10, 20, 40) options (gold + random labels, fixed seed); BoolQ validation (3270, `noul`); SST-5 test (2210, 5-level `score`). Options are plain humanized label names (`card arrival`), no descriptions. Fine-tuning mixture (12.9k items: AG News, emotion, DBpedia-14, CLINC150 minus banking/credit-card intents, BoolQ-train, MNLI, SST-5-train) **excludes Banking77**, so Banking77 is zero-shot for every model; BoolQ and SST-5 are *in-distribution* for B4 only.

### Accuracy

| id | run | B77 N=77 | N=40 | N=20 | N=10 | N=5 | BoolQ | SST-5 | p50 ms | VRAM GB | mlflow run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | m0-laya | 0.360 | 0.590 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 17.6 | 2.43 | 94d81ef6 |
| B3 | m0-laya-b3 | 0.427 | 0.600 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 19.4 | 2.45 | 1167911c |
| B1b | m0-laya-short-bge | 0.610 | 0.669 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 16.2 | 2.67 | 1e1d2897 |
| B2 | m0-laya-tourn | 0.585 | 0.649 | 0.743 | 0.817 | 0.871 | 0.779 | 0.305 | 34.2 | 2.45 | 71583460 |
| B4 | m0-laya-b4 | 0.502 | 0.657 | 0.776 | 0.837 | 0.893 | 0.826 | 0.570 | 17.5 | 2.43 | 68fce508 |
| B5 | m0-biencoder-mb | 0.108 | 0.158 | 0.221 | 0.304 | 0.430 | 0.380 | 0.180 | 9.2 | 0.82 | 1fa70b21 |
| B5b | m0-biencoder-bge | 0.695 | 0.753 | 0.844 | 0.887 | 0.931 | 0.395 | 0.309 | 2.8 | 0.24 | 1c460fad |
| B6 | m0-laya-td | 0.402 | 0.623 | 0.764 | 0.828 | 0.871 | 0.780 | 0.427 | 17.2 | 2.45 | 2cb1c7df |

### Calibration (ECE, 15 bins) and NLL

| id | B77 N=77 ECE / NLL | N=40 ECE / NLL | N=20 ECE / NLL | N=10 ECE / NLL | N=5 ECE / NLL | BoolQ ECE / NLL | SST-5 ECE / NLL |
|---|---|---|---|---|---|---|---|
| B0 | 0.18 / 3.21 | 0.17 / 2.22 | 0.13 / 1.32 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B3 | 0.15 / 2.89 | 0.15 / 1.98 | 0.13 / 1.32 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B1b | 0.23 / 2.47 | 0.19 / 1.95 | 0.13 / 1.32 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B2 | 0.03 / 3.89 | 0.07 / 3.24 | 0.05 / 2.55 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B4 | 0.15 / 2.54 | 0.14 / 1.73 | 0.10 / 1.06 | 0.09 / 0.73 | 0.08 / 0.46 | 0.14 / 0.68 | 0.28 / 1.40 |
| B5 | 0.09 / 4.24 | 0.12 / 3.59 | 0.15 / 2.90 | 0.18 / 2.21 | 0.18 / 1.53 | 0.14 / 0.70 | 0.04 / 1.62 |
| B5b | 0.03 / 1.13 | 0.03 / 0.86 | 0.04 / 0.51 | 0.02 / 0.36 | 0.01 / 0.21 | 0.43 / 1.31 | 0.29 / 1.60 |
| B6 | 0.20 / 2.76 | 0.15 / 1.58 | 0.09 / 0.94 | 0.27 / 0.84 | 0.37 / 0.80 | 0.08 / 0.50 | 0.06 / 1.24 |

Models: B0 Laya shipped · B3 Laya `max_len=1024, head_max_len=512` · B1b Laya + shortlist k=20 using `bge-base-en-v1.5` as embedder · B2 Laya + tournament (groups of 16) · **B4 Laya fine-tuned by its own trainer on our mixture (4 epochs, defaults)** · B5 raw ModernBERT-large mean-pool cosine · B5b `bge-base-en-v1.5` cosine (110M) · B6 `laya-typed-decisions`. For B5/B5b the softmax temperature is fitted on 300 items of the training mixture.
Rows N≤20 for B1b/B2/B3 equal B0 on N≤10 because no shortlist/tournament/budget effect applies (identical code path).

## Findings
1. **High-cardinality ceiling confirmed.** Laya accuracy falls 0.87 → 0.82 → 0.75 → 0.59 → 0.36 as N goes 5 → 10 → 20 → 40 → 77. Larger budgets (B3) recover only N=77 (0.360 → 0.427, matching the model card's 0.425); descriptions were already shown invisible at 77 options (preliminary run).
2. **Fine-tuning the same Laya on our data (B4) helps broadly but does not remove the ceiling:** N=77 0.502 (+14 pts over B0), N=20 0.776, BoolQ 0.826, SST-5 0.570 (B0 0.305; MAE 0.47 vs 1.06 levels). This is the head-to-head target for LIDM under equal data. Caveat: BoolQ/SST-5 gains are in-distribution.
3. **A plain 110M bi-encoder (B5b) beats every Laya variant on Banking77 at every N** (0.695 @77, 0.753 @40, 0.844 @20) at 2.8 ms and 0.24 GB, with good calibration (ECE 0.03, NLL 1.13 @77). It is useless on BoolQ (0.395) and SST-5 (0.309), where interaction/nuance is needed. Consequence for LIDM: on label-matching tasks, retrieval-style option vectors are already strong — the `dual_score` retrieval prior and a high-quality option encoder are not optional extras; LIDM must at least match B5b on Banking77 *and* B4 on BoolQ/SST-5 to be worth building. (Caveat: bge was pre-trained on large retrieval data; B5b is not trained on our mixture, so this is a "free" baseline, not an equal-data one.)
4. **Laya's two published workarounds** reach: shortlist with a proper embedder 0.610 @77 at 16 ms (B1b; the earlier 0.256 was due to using Laya's own encoder as embedder), tournament 0.585 at 34 ms. Both stay below the bare bi-encoder B5b.
5. **Calibration.** Shipped Laya is over-confident at N≥40 (ECE 0.17–0.18) and on SST-5 (0.33). B4's fitted temperatures for `score` and `noul` both landed on the [0.5, 5] clamp (Laya's own warning), i.e. its calibration step could not fully correct them. B2 has the best N=77 ECE (0.03) but worst NLL (zero mass on eliminated labels). B6 is badly miscalibrated at small N (ECE 0.37 @5).
6. **Latency.** All single-pass Laya variants 17–19 ms (model card: 33–40 ms on a T4); tournament 34 ms; bi-encoder 3–9 ms. LIDM budget (≤~2× B0 fresh ≈ 35 ms) leaves room for a second encoder pass.
7. **Label-format sensitivity.** Writing the same labels as `card_arrival` instead of `card arrival` changed B0's N=77 accuracy from 0.377 to 0.360 and ECE from 0.51 to 0.18 (mean confidence 0.88 vs 0.48). `predict_batch` and `system_one` agree to ≤0.065 per probability, so this is an input-format effect, not a code-path bug. Baseline numbers are therefore only comparable at fixed option strings.

## Targets for LIDM (derived, to be confirmed in GOALS.md)
- Banking77 N=77 ≥ 0.70 zero-shot after training on the same mixture (B5b 0.695, B4 0.502, B1b 0.610); N=20 ≥ 0.85.
- BoolQ ≥ 0.826 and SST-5 ≥ 0.570 (B4 level) — otherwise late interaction loses on nuance.
- p50 latency ≤ 35 ms fresh; ≤ 20 ms with cached options.
- ECE ≤ 0.1 on every task after temperature fitting.

## Caveats
- One seed for B4; no CIs in the table (per-task CIs are in each run's `metrics.json` / MLflow as `<task>/accuracy_ci_lo|hi`).
- Instruction wording was fixed and not tuned; results move a few points with wording and label format (finding 7).
- 72 of 12.9k training items were dropped by Laya's `options_collapsed` check.
- `src/cls/data/` was never committed (a `data/` ignore rule matched it) and was recreated on 2026-11-14. The table comes from the original code. On 200-item quick runs the recreated code is within about 0.03 of the table for B0 and B5b, and the preliminary B0 ECE matches to within 0.003. Item sampling, CLINC options and instruction wording may differ, so rerun `scripts/run_m0_all.sh` before using these as exact LIDM comparison points.
- Reliability diagrams are not logged for the harness runs (only for the preliminary runs).

---

## M1 — LIDM on the M0 mixture, frozen encoder (2026-11-15, complete)

Runs B / C / D from the grid, trained on the M0 mixture (11,610 train / 1,290 val, task-stratified random
split; 152 distinct option strings), 6 epochs, batch 32, AdamW (0.9, 0.98), cosine schedule, one seed.
MLflow experiment `m1`; checkpoints `checkpoints/m1-*fix*`. Banking77 zero-shot = 1,000 test items through
`scripts/eval_banking77.py` (N=77: all labels; N=20: gold + 19 random); chance is 0.013 / 0.05.

**First run B was broken.** It stayed at chance (val acc 0.27; BoolQ always "yes", 0.607 = majority; SST-5
0.20). Cause: the head read ModernBERT's raw hidden states, whose token norm grows from ~30 to ~1,200 with
single values up to ~27,000 in layers 20-27. Those dominated the cross-attention keys and the residual
stream, so logits did not depend on the input (BoolQ yes-minus-no logit 0.510 +- 0.004 over 64 passages).
The layer-mix weights never moved at LR 1e-4 (all 0.036 = 1/28) and the dual scale went 1.000 -> 1.004.
Fixed by LayerNorming each layer before the mix and a 1e-2 LR for the scalars (ARCHITECTURE.md,
"Deviations"). All rows below use the fix.

### LR sweep (run B: 2 blocks, 1 loop, 32.9M params)
| LR | best val loss (epoch) | val acc | choice | BoolQ | SST-5 | B77 N=77 | B77 N=20 | mlflow |
|---|---|---|---|---|---|---|---|---|
| 3e-5 | 0.707 (5) | 0.717 | 0.774 | 0.627 | 0.433 | 0.019 | 0.075 | bfee7a2e |
| **1e-4** | **0.599 (3)** | **0.759** | **0.831** | **0.660** | 0.380 | 0.025 | 0.085 | ad027f5e |
| 3e-4 | 0.651 (3) | 0.726 | 0.804 | 0.587 | 0.353 | 0.019 | 0.073 | c7b17027 |
| 1e-3 | 0.687 (5) | 0.712 | 0.785 | 0.613 | 0.333 | 0.034 | 0.098 | 4854aca0 |

Values at the best-val-loss epoch. Every LR above 3e-5 overfits after epoch 3 (val loss 1.0-1.5 by epoch 6
while accuracy stays flat).

### Looping (LR 1e-4)
| Run | Blocks x loops | Params | Best val loss | Val acc by loops scored (final epoch) | BoolQ | B77 N=77 / N=20 | Wall | mlflow |
|---|---|---|---|---|---|---|---|---|
| B | 2 x 1 | 32.9M | **0.599** | 0.756 | 0.63 | 0.025 / 0.085 | 9 min | ad027f5e |
| C sampled | 2 x 1-8 | ~34M | 0.656 | L1 0.733, L2 0.739, L4 0.742, L8 0.740 | 0.57 | 0.026-0.032 / 0.080-0.086 (any L) | 13 min | 166e8bdb |
| C fixed 4 | 2 x 4 | ~34M | 0.659 | L1 0.695, L2 0.753, L4 0.748 | 0.60 | 0.018 / 0.085 (L4) | 12 min | e4de9577 |
| D | 8 x 1 | 131.6M | 0.634 | 0.747 | 0.58 | 0.049 / 0.126 | 19 min | 0395a267 |

### Findings
1. **Labels were memorised, not read.** Every run reaches val acc ~0.75 (CLINC val 0.91) but Banking77
   zero-shot stays near chance (0.02-0.05 at N=77), with predictions collapsing onto a few labels (B: 26
   distinct predictions over 1,000 items, 42% on one label). The random val split shares all 152 labels with
   training, so it does not measure zero-shot. The B4 comparison was never equal-data: Laya had learned label
   reading before our fine-tune.
2. **The cosine prior is weak on frozen ModernBERT:** alone it scores 0.08-0.10 on Banking77 (M0 B5: 0.108).
   The learned dual scale rose to 3.1, so the model uses it, but it cannot carry the task.
3. **Instruction sensitivity:** CLINC val drops 0.91 -> 0.59 when only the instruction is swapped for the
   Banking77 one; Banking77 doubles (0.027 -> 0.053) with CLINC's instruction.
4. **Looping gave no gain on this data.** Both C runs trail B (best val loss 0.66 vs 0.60). Inside one model,
   accuracy rises slightly to 2-4 loops and then flattens; fixed-4 is fine at its trained depth but drops to
   0.70 at 1 loop. C fixed 4 vs D (same compute, 4x params) is close (0.659 vs 0.634), so the decision rule is
   undecided; the design doc expects looping to help rule and policy questions, not topic labels, so it is
   retested in stage 2a.
5. **BoolQ stays near the majority class (0.57-0.66) for every head**: frozen-encoder late interaction does
   not learn the passage-question reasoning here; LoRA (E) and joint-encoder control (F) are the next tests.
6. LR 1e-4 (the design doc's value) is the best head LR; layer-mix weights at 1e-2 move to layers 22-26.

Caveats: one seed per run; the prediction-collapse counts (finding 1), cosine-only accuracy (2) and
instruction swaps (3) come from one-off diagnostic scripts on the LR 1e-4 checkpoint, not from a repo script; 1,000-item Banking77 subsample (differences of 0.01-0.02 between runs are
noise); C's ~34M parameters is an estimate (B + per-loop LoRA), not read from the run.

---

## Stage 1 — label reading (2026-11-15/16, complete)

Data: `scripts/prepare_stage1.py`, 973,783 questions, 2,118 distinct option strings, Banking77 held out
(0 shared labels or texts). Validation: UTCD out-of-domain sets; `zs/` sets share no label with training,
`ood/` sets do (EXPERIMENTS.md). Run B head (2 blocks, 1 loop), LR 1e-4, batch 128 (micro-batch 16), one seed.
Banking77 test = all 3,080 items through `scripts/eval_banking77.py`; chance 0.013 at N=77, 0.05 at N=20.

### Runs
| Run | Encoder | Prior | Steps | Wall | Peak VRAM | B77 test N=77 | N=20 | mlflow |
|---|---|---|---|---|---|---|---|---|
| LR pilots (5e-5 / 1e-4 / 2e-4 / 3e-4) | frozen | encoder cos | 500 | 7 min each | 3.7 GB | – (val at chance) | – | f28748e3 / 258a409d / 870a6102 / f3b8625a |
| Frozen, full | frozen | encoder cos | 7,608 | 89 min | 4.0 GB | 0.044 | 0.127 | 1e71f6a8 |
| bge-prior pilot | frozen | bge-base cos | 500 | 8 min | 3.9 GB | 0.676 (bge alone 0.695) | 0.814 (0.830) | 123a0ba6 |
| LoRA pilot | LoRA r=16 | encoder cos | 500 | 35 min | 5.0 GB | 0.029 | 0.100 | 35da46c4 |
| LoRA pilot, 1k | LoRA r=16 | encoder cos | 1,000 | 63 min | 5.4 GB | 0.072 | 0.156 | b23558f4 |
| **LoRA, full** | **LoRA r=16** | **encoder cos** | **7,608** | **6.9 h** | **5.5 GB** | **0.434** | **0.641** | **bdf553e8** |

LR pilot training loss at steps 401-500 (the single-pass signal used to pick the LR): 5e-5 0.904, **1e-4 0.840**,
2e-4 0.923, 3e-4 1.047 (stalled). `best.pt` of the full LoRA run (step 5,000) scores 0.436 / 0.649, the same
as `last.pt` within noise; the frozen run's `best.pt` (step 1,500, chosen on a 61-item multi_eurlex spike)
scored 0.017.

### Diagnostics (final checkpoints)
| | Frozen, full | LoRA, full |
|---|---|---|
| B77 N=77: full model / head alone / prior cosine alone | 0.044 / 0.043 / 0.045 | **0.434 / 0.420 / 0.172** |
| Distinct B77 labels predicted; share of top 3 | 40 of 77; 75% | **71 of 77; 15%** |
| nlu_evaluation, gold label seen / unseen in training | 0.735 / 0.135 | 0.843 / **0.656** |
| snips, gold label seen / unseen | 0.986 / 0.316 | 0.972 / **0.632** |
| Unseen-gold questions answered with a seen label (nlu_evaluation) | 69% | **18%** |
| Zero-shot val loss over the run | 2.38 -> 2.64 (rising) | 1.96 -> 1.37 (falling) |
| `val_acc/zs/banking77`, 300-item train slice | 0.02-0.05 throughout | 0.073 (500) -> 0.297 (1k) -> 0.36-0.40 (3.5k on) |

### Full M0 task set: LoRA checkpoint vs the M0 baselines
`scripts/eval_lidm.py checkpoints/stage1-Blora/last.pt` through `cls.eval.harness`, same task set as M0 (MLflow
`baselines`, run `lidm-stage1-Blora`, b28c8434; predictions in `artifacts/lidm-stage1-Blora/`). Latency is
batch 1 with a fresh label set (all options re-encoded each call; no option cache, LoRA not merged).

| id | model | B77 N=77 | N=40 | N=20 | N=10 | N=5 | BoolQ | SST-5 | p50 ms | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | Laya as shipped | 0.360 | 0.590 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 17.6 | 2.43 |
| B3 | Laya, larger budgets | 0.427 | 0.600 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 19.4 | 2.45 |
| B4 | Laya fine-tuned on M0 mixture | 0.502 | 0.657 | 0.776 | 0.837 | 0.893 | 0.826* | 0.570* | 17.5 | 2.43 |
| B1b | Laya + bge shortlist | 0.610 | 0.669 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 16.2 | 2.67 |
| B5b | bge-base bi-encoder | 0.695 | 0.753 | 0.844 | 0.887 | 0.931 | 0.395 | 0.309 | 2.8 | 0.24 |
| **L1** | **LIDM, stage 1, encoder LoRA** | **0.434** | **0.562** | **0.650** | **0.735** | **0.817** | **0.670** | **0.374** | **55.4** | **1.24** |

\* in-distribution for B4 (BoolQ and SST-5 train splits were in its fine-tuning data). Neither is in stage 1, so
for L1 they are zero-shot. BoolQ majority class = 0.622.

| L1 calibration (raw, no temperature) | N=77 | N=40 | N=20 | N=10 | N=5 | BoolQ | SST-5 |
|---|---|---|---|---|---|---|---|
| ECE / NLL | 0.10 / 2.31 | 0.07 / 1.72 | 0.06 / 1.25 | 0.06 / 0.85 | 0.06 / 0.52 | 0.03 / 0.60 | 0.13 / 1.35 |

Also measured: AG News test 0.891 (ECE 0.03), DAIR Emotion test 0.851 (ECE 0.02). Both datasets' train splits are
in stage 1 through UTCD (with different label wording), so these are in-distribution, unlike the Laya card's
0.950 / 0.595. SST-5 expected-level MAE 0.72 levels (B4 0.47, B0 1.06).

**Against GOALS.md:**
| Criterion | Target | L1 | Status |
|---|---|---|---|
| Banking77 N=77 / N=20 | >= 0.70 / >= 0.85 | 0.434 / 0.650 | not met |
| BoolQ / SST-5 (no regression vs B4) | >= 0.826 / >= 0.570 | 0.670 / 0.374 | not met (zero-shot vs B4's in-distribution) |
| Latency p50, fresh / cached options | <= 35 ms / <= 20 ms | 55.4 ms / not measured | not met fresh |
| ECE after temperature fitting | <= 0.1 everywhere | raw already <= 0.10 except SST-5 0.13 | likely met; temperatures not fitted yet |
| Trains on a 12 GB GPU | yes | 5.5 GB peak | met |

What the comparison shows:
- **Accuracy falls more slowly with option count than for Laya**, the design's central claim: from N=5 to N=77
  L1 loses 0.38, Laya B0 0.51, fine-tuned Laya B4 0.39. L1 passes shipped Laya only at N=77; at N <= 40 every
  Laya variant is ahead (by 5-10 points), and bge is ahead everywhere on Banking77.
- **Calibration is good out of the box** (ECE 0.03-0.10 on six of seven tasks, before any temperature), where
  shipped Laya had 0.13-0.33.
- **SST-5 zero-shot (0.374) beats shipped Laya (0.305)**; BoolQ (0.670) is above the majority class but below
  every Laya variant.
- **Latency is ~3x Laya's** with a fresh label set: both passes run every call, LoRA is unmerged, the encoder
  pads. Cached options and merged LoRA are the obvious first fixes; not measured yet.

### Findings
1. **The frozen encoder was the bottleneck.** Frozen, every head memorised the training labels: seen-label val
   sets rose to 0.7-0.9 while unseen ones stayed at chance, and the learned layer mix pulled the cosine prior
   from 0.116 (step 1,500) down to 0.045. With encoder LoRA the same head reads unseen labels.
2. **The bge prior did not help the frozen model.** Starting exactly at bge (scorer zero-initialised), 500 steps
   made it slightly but reliably worse than bge alone (McNemar z = -3.8); the head alone stayed at chance.
3. **LoRA gains need steps.** At 500 and 1,000 steps the LoRA model looked weak (0.029, 0.072); the jump came
   between steps 500 and 1,000 of the full schedule (B77 val 0.07 -> 0.30), then it flattened from ~3,500 steps
   while training loss kept falling (0.43 -> 0.36): more passes over the same 2,118 labels are unlikely to help.
4. **Bigger heads were not the fix**: M1's looping and 8-block heads gave no zero-shot gain, and every stage-1
   run had training loss still falling, so capacity was not the limit.
5. patent stays near chance in every run (0.14 vs 0.11; abstract patent-class labels); multi_eurlex (61 items)
   is too small to read.

---

## Appendix: preliminary run (superseded by the table above; label ids with underscores, `predict_batch`; test, 3080 examples, 77 options, zero-shot, one `choice` question)

Hardware: RTX 4070 SUPER, bf16, SDPA. Laya 0.4.20, checkpoint `convaiinnovations/laya` (as shipped, shipped temperatures applied).
Reproduce: `uv run python -m cls.eval.laya_baseline --variant {b0,b3,b1,b2} --label-style {desc,id}` then `uv run python scripts/report.py`.
Browse every run: `scripts/mlflow_ui.sh` (experiment `baselines`). Table below = `docs/baseline_table.md`.

| run | acc | macro-F1 | Brier | NLL | ECE | p50 ms | p95 ms | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| B0 Laya as shipped (512 / head 192), labels `id: desc` | 0.377 | 0.327 | 1.096 | 5.784 | 0.506 | 19.5 | 20.4 | 2.90 |
| B0 labels `id` only | 0.377 | 0.327 | 1.096 | 5.784 | 0.506 | 19.0 | 19.6 | 2.90 |
| B3 larger budgets (1024 / head 512), `id: desc` | 0.428 | 0.389 | 0.975 | 4.842 | 0.426 | 22.4 | 23.2 | 3.10 |
| B3 larger budgets, `id` only | 0.436 | 0.397 | 0.958 | 4.697 | 0.416 | 21.7 | 22.7 | 3.05 |
| B1 + embedding shortlist k=20 (Laya's own encoder as embedder) | 0.256 | 0.213 | 1.251 | 9.548* | 0.561 | 24.3 | 26.6 | 2.43 |
| B2 + tournament, groups of 16 | **0.602** | 0.609 | 0.607 | 3.951* | 0.198 | 47.7 | 50.0 | 2.47 |

95% bootstrap CI on accuracy: B0 [0.360, 0.394], B3-desc [0.411, 0.445], B3-id [0.418, 0.454], B1 [0.241, 0.271], B2 [0.584, 0.619].
\* B1/B2 put zero mass on eliminated labels; NLL floors probabilities at 1e-6, Brier/NLL/ECE are only roughly comparable to B0/B3.

