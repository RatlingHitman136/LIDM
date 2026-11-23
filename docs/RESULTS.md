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
| C (2 blocks looped, 1-4 sampled), `--init` from LoRA full | LoRA r=16 | encoder cos | +3,000 | 2.9 h | 6.6 GB | 0.437 / 0.439 / 0.437 at 1 / 2 / 4 loops | 0.645 / 0.644 / 0.641 | 5fcede8e |
| LoRA + bge prior, `--init` from LoRA full | LoRA r=16 | bge-base cos | +3,000 | 2.9 h | 5.5 GB | 0.514 (bge alone 0.695) | 0.716 (0.830) | a91cf3f2 |

Looping from L1 adds nothing: every depth matches L1 within noise (±0.01 on 3,080 items), and on validation
1 loop scored at least as well as 2 or 4 at all six checkpoints. Training loss kept falling (0.38 -> 0.35)
without a zero-shot gain. The matching B continuation (control) was not run; with no difference to explain, it
is not needed for this conclusion.

**LoRA + bge prior from L1** (`--prior bge --init`, +3,000 steps, 2.9 h, 5.5 GB, mlflow a91cf3f2): Banking77 test
**0.514** @N=77, **0.716** @N=20: +8 points over L1, but far below bge alone (0.695 / 0.830; McNemar z = -18.9).
The learned prior scale fell from 12.5 to 7.07 and the head's logits spread 3.5x wider than the prior's, so the
head decides, and on unseen labels it is usually the one that is wrong (713 vs 155 disagreements in bge's
favour). Refitting only the bge weight a (score = head + a * cos_bge) on the Banking77 train slice picks
a = 160, i.e. almost pure bge, giving 0.689 / 0.829 on test. Across the whole range of a, test accuracy rises
monotonically to bge alone and never meaningfully exceeds it (best 0.696 at a = 320). The head is right on 4.8%
of questions where bge is wrong (oracle 0.743), but no fixed weight can use that. Conclusion: on unseen-label
matching the trained head adds nothing to bge; the weight learned on training data (where the head knows the
labels) is the wrong one for zero-shot.

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
| Latency p50, fresh / cached options | <= 35 ms / <= 20 ms | 35.9 ms / **12.6 ms** with LoRA merged (55.4 ms unmerged) | cached met; fresh at target (+0.9 ms) |
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
- **Latency** (`scripts/bench_latency.py`, batch 1, Banking77 N=77, p50 over 100 calls):

  | Variant | p50 / p95 | pass A | pass B (77 options) | head |
  |---|---|---|---|---|
  | fresh options, LoRA unmerged (what `eval_lidm.py` logs) | 56.0 / 60.6 ms | 17.3 | 34.6 | 4.6 |
  | fresh options, LoRA merged | 35.9 / 40.2 ms | 8.5 | 23.5 | 4.5 |
  | **cached options, LoRA merged** | **12.6 / 14.7 ms** | 8.6 | 0 | 4.1 |

  Merging LoRA into the bf16 weights changes ~4% of predictions but not accuracy (1,000 Banking77 items: 0.438
  unmerged, 0.438 merged; merging in fp32 and storing fp16: 0.444, 97% agreement). With cached options LIDM is
  faster than shipped Laya (17.6 ms); with a fresh label set, encoding the 77 options is two thirds of the time.

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

### Retrieval-trained encoders as plain bi-encoders (2026-11-16)
Cosine between the state and each label, argmax (no temperature), through `Encoder.embed` (the encoder's own
sentence-transformers pooling). Banking77 test = all 3,080; `zs/` and `ood/` = stage-1 val sets.

| Encoder | Licence | Arch / pooling | B77 N=77 | N=20 | zs/banking77 | zs/consumer_finance | zs/patent | zs/multi_eurlex | zs mean |
|---|---|---|---|---|---|---|---|---|---|
| bge-base-en-v1.5 (110M) | MIT | BERT / cls | **0.695** | **0.830** | 0.653 | 0.383 | 0.173 | 0.344 | 0.389 |
| gte-modernbert-base (149M) | Apache-2.0 | ModernBERT / cls | 0.634 | 0.779 | 0.590 | 0.477 | 0.150 | 0.328 | 0.386 |
| modernbert-embed-base (149M), no prefix | Apache-2.0 | ModernBERT / mean | 0.596 | 0.745 | 0.553 | 0.463 | 0.163 | 0.459 | 0.410 |
| modernbert-embed-base, `classification:` prefix | Apache-2.0 | ModernBERT / mean | 0.643 | 0.783 | 0.617 | 0.473 | 0.170 | 0.443 | 0.426 |

**LIDM on gte-modernbert-base** (full stage 1 from scratch: LoRA r=16, `--option-vector pooled` (cls),
`--prior self`; 7,608 steps, 4.4 h, 3.1 GB; mlflow 1e50532e). Banking77 test **0.628 / 0.775** (N=77 / N=20) as
trained, **0.635 / 0.778** with the retrieval weight refitted on zero-shot val (`fit.pt`, s = 160 vs trained
26.6): equal to gte alone (0.634 / 0.779). Zero-shot val stayed at gte's level for the whole run while training
loss fell 0.85 -> 0.47.

Why the head adds nothing on unseen labels (`last.pt`):

| | Banking77 test (unseen labels) | 1,500 training questions (seen labels) |
|---|---|---|
| head alone / retrieval alone | **0.015** (chance 0.013) / 0.634 | **0.689** / 0.655 |
| combined, trained s | 0.628 | **0.863** |
| head's mean top probability | 0.05 | 0.71 |
| logit spread, head vs trained prior | 1.5 vs 6.3 | 2.4 vs 1.7 |
| re-ranking retrieval's top 5: head / retrieval / random | 0.192 / 0.634 / ~0.17 | 0.681 / 0.655 |

On seen labels the head and retrieval are strongly complementary (+21 points combined); on unseen labels the head
is flat, unconfident and no better than random even for re-ranking retrieval's top 5. 49% of the head's score
variance on Banking77 is a fixed offset per label, independent of the question (retrieval: 20%). The head learned
label-specific corrections for the 2,118 training labels, not a way to read labels: with a strong retrieval term
present from step 0, the generalisable part of the task is already explained, and the cheapest remaining loss
reduction is per-label residuals. The L1 head (no strong prior) did learn some reading (0.42 alone), but its
combination with bge also did not beat bge.

bge through `Encoder.embed` reproduces M0's B5b exactly (0.695 / 0.830). Both ModernBERT retrievers are 5-6 points
below bge on Banking77 but far above LIDM-on-ModernBERT-large (0.434). gte-modernbert-base is the first pick for
LIDM: within noise of the prefixed nomic model, no prefixes, `[CLS]` pooling, and ModernBERT architecture, so the
LoRA targets and head initialisation work unchanged.

---

## Stage 1b — label variety (2026-11-17/18)

Data: `scripts/prepare_stage1b.py`: 522,691 questions with **961,447 distinct option strings** (stage 1: 2,118):
refinedweb categories / characteristics / nli from the synthetic set (263k; each text's written-wrong labels as
distractors), plus 80k synthetic, 80k UTCD (30% with distractors from other tasks), 50k yes/no, 50k NLI. Labels of
the zs/ val sets and Banking77, and near-duplicates (all words of a multi-word label), are banned. New val sets
`heldout/*` (1,936 questions): refinedweb texts kept out of training, answers never seen in training.
Run with `scripts/run_stage1b_overnight.sh`; logs in `logs/overnight/`.

| Run | Steps | Wall | B77 test N=77 trained / refit s | N=20 | head alone B77 | zs val (combined) | heldout val | mlflow |
|---|---|---|---|---|---|---|---|---|
| gte-modernbert-base, LoRA, pooled, self prior, prior dropout 0.5 | 4,084 | 5.2 h | 0.548 / 0.599 (s 8.3 -> 20) | 0.709 / 0.752 | **0.456** (was 0.015) | 0.368 | **0.905** | stage1b-gte-lora |
| ModernBERT-large, LoRA, marker, encoder prior (L1 settings) | 2,500 of 4,084 (7 h limit at 9.5 s/step) | 7 h | 0.433 / 0.435 | 0.618 / 0.616 | 0.423 | 0.337 | 0.925 | stage1b-mbl-lora |
| *gte alone (retrieval)* | – | – | *0.634* | *0.779* | – | *0.386* | *~0.71* | – |

**What changed:** label variety removed the memorisation. The gte head alone now reads unseen labels (Banking77
0.015 -> 0.456; per-label offset share 0.49 -> 0.22) and on unseen labels from the training domain beats retrieval by
20 points (heldout 0.907 vs 0.708), still +11 when texts are cut to 10 words.

**What did not:** transfer to intent-style targets. Per val set, head alone vs gte alone: patent 0.267 vs 0.150 (+12),
multi_eurlex 0.328 vs 0.328 (combined 0.443, +11), but banking77 0.433 vs 0.590 (-16) and consumer_finance 0.277 vs
0.477 (-20). The best weight on the Banking77 test itself (s = 80) gives 0.643, +0.9 over retrieval (noise).

**Why the head trails retrieval on Banking77** (`last.pt`, 3,080 test questions):

| Test | Head | Retrieval |
|---|---|---|
| message shares a content word with the gold label (2,226) | 0.514 | 0.722 |
| no shared word, implicit (854) | 0.304 | 0.405 |
| N=10, random distractors / 9 most similar labels | 0.759 / 0.569 | 0.850 / 0.669 |
| seven instruction wordings incl. training ones and none | 0.456-0.485 | 0.634 |

Not reasoning (the gap is largest on plain word matches, -21, smaller on implicit messages, -10), not hard
distractors (both lose ~19 points from easy to hard), not the unseen instruction (+-3), not text length (the head
keeps +11 to +20 on short in-domain texts). The head learned to match texts to the kind of labels it trained on
(descriptive topic categories and statements about web documents) and transfers to similar label types (patent,
EuroLex topics), but not to customer-intent labels ("card arrival", "pending top up") or complaint product types,
where retrieval's pretraining on query-intent pairs is far broader than our data.

### Banking77 with label descriptions (2026-11-18)

Options replaced by one-sentence descriptions written from the label names only
(`src/cls/data/banking77_descriptions.json`; reviewed by a separate agent: no leakage, longest shared n-grams with
test messages are the label names themselves; 13 rewritten for separability; `get_physical_card` vs
`order_physical_card` cannot be told apart from the names and was left as is). A separate benchmark condition, not
comparable with names-only Banking77 numbers. `scripts/eval_banking77_descriptions.py`, all 3,080 test questions.

| N=77 (N=20) | names only | descriptions only | `name: description` |
|---|---|---|---|
| bge alone | 0.696 (0.830) | 0.718 (0.845) | **0.732 (0.862)** |
| gte alone | 0.634 (0.779) | 0.703 (0.840) | 0.696 (0.827) |
| LIDM gte stage 1b: head alone | 0.456 (0.650) | 0.411 (0.629) | 0.463 (0.677) |
| LIDM gte stage 1b: combined, refitted s = 20 | 0.599 (0.752) | 0.634 (0.794) | 0.646 (0.799) |
| LIDM ModernBERT-large stage 1b: head alone | 0.423 (0.609) | 0.374 (0.589) | 0.505 (0.685) |
| LIDM ModernBERT-large stage 1b: combined, refitted | 0.435 (0.616) | 0.482 (0.667) | 0.558 (0.732) |
| LIDM ModernBERT-large L1: head alone | 0.420 (0.629) | 0.451 (0.662) | 0.395 (0.635) |

Descriptions help retrieval (+2 to +7 points) more than any LIDM head (-5 to +8, inconsistent across formats).
In no condition does LIDM beat the best plain bi-encoder; with descriptions the gap to its own retrieval term stays
(gte stage 1b: 0.646 combined vs 0.696 retrieval). The head's Banking77 weakness is not the label format.

---

## Few-shot Banking77: 8 examples per intent (2026-11-18)

Can LIDM earn a role for schemas with a few labelled examples? 8 examples per intent (616) from Banking77's
**train** split, 3 seeds (`scripts/prepare_fewshot_banking77.py`), plus 2 per intent (154) for early stopping; full
3,080-question test, N=77 (all labels) and N=20 (same candidate sets as elsewhere).

| 8 examples per intent | N=77, seeds 0 / 1 / 2 | N=77 mean | N=20 mean |
|---|---|---|---|
| bge-base probe (logistic regression on frozen embeddings, MTEB settings; `scripts/fewshot_probe.py`) | 0.841 / 0.845 / 0.847 | **0.844** | **0.930** |
| gte-modernbert-base probe (LIDM's encoder) | 0.812 / 0.823 / 0.827 | 0.821 | 0.921 |
| LIDM gte, init from stage 1b, fine-tuned (lr 3e-4, 10 passes, chosen on the 154-question set) | 0.809 / 0.820 / 0.803 | 0.810 | 0.911 |
| Laya fine-tuned with its own trainer (trained on 20-option versions; 555 + 61 for its calibration) | 0.582 / 0.487 / 0.522 | 0.530 | 0.848 |

- LIDM is below its own encoder's probe on all three seeds (-0.3 / -0.3 / -2.4) and 3.4 points below bge's.
- It beats fine-tuned Laya by 28 points at N=77 and 6 at N=20. Laya's trainer drops every 77-option question
  (`options_collapsed`: the 192-token head budget leaves ~3 tokens per option), so it cannot even train on the full
  label set.
- Stage 1 matters for LIDM few-shot: without the stage-1 initialisation the early-stopping score is 0.714 instead of
  0.79-0.80 (seed 0).
- Variant search on seed 0 (early-stopping set, 154 questions): lr 1e-4 x 10 passes 0.792, x 20 passes 0.792,
  lr 3e-4 x 10 passes 0.799 (chosen), no stage-1 init 0.714. Test score of the lr 1e-4 x 10 run: 0.804.
- Conclusion for this use case: with a few examples per label, a logistic-regression probe on frozen bge embeddings
  (seconds to train, no label names needed) beats LIDM; LIDM's advantage over Laya is large, over embeddings none.

---

## Context suite: zero-shot reading tasks (2026-11-18)

`scripts/run_context_suite.sh`; accuracy, ≤1,000 questions per task (±3 points 95% CI at n=1,000). LIDM gte =
stage-1b gte-modernbert-base LoRA; LIDM MBL = stage-1 ModernBERT-large LoRA; "+ opts in Q" = every option also
written into pass A's question (regime-2 tasks only). Laya-TD (`laya-typed-decisions`) was trained on
typed-decisions train, so its group-1 score is in-domain. Bold = best per row without Laya-TD.

| Group | Task | n | Majority | bge | gte | LIDM gte | LIDM gte + opts in Q | LIDM MBL | LIDM MBL + opts in Q | Laya | Laya-TD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 System One | typed_decisions | 2000 | 0.292 | 0.345 | 0.362 | 0.357 | 0.347 | **0.380** | 0.362 | 0.359 | 0.745 |
| 2 content in options | race | 1000 | 0.274 | 0.380 | 0.415 | 0.439 | 0.428 | 0.342 | 0.357 | **0.443** | 0.452 |
| 2 content in options | quail | 1000 | 0.257 | 0.423 | 0.429 | 0.407 | 0.412 | 0.370 | 0.366 | **0.474** | 0.489 |
| 2 content in options | when2call | 1000 | 0.361 | 0.353 | 0.355 | **0.379** | 0.376 | 0.340 | 0.324 | 0.142 | 0.167 |
| 3 negation / contrast | condaqa | 1000 | 0.478 | 0.266 | 0.465 | 0.461 | – | **0.515** | – | 0.455 | 0.483 |
| 3 negation / contrast | vitaminc | 1000 | 0.509 | 0.411 | 0.477 | 0.527 | – | **0.694** | – | 0.642 | 0.583 |
| 4 content in question | boolq | 1000 | 0.605 | 0.425 | 0.523 | 0.596 | – | 0.664 | – | **0.765** | 0.762 |
| 4 content in question | pubmedqa | 1000 | 0.552 | 0.339 | 0.355 | 0.441 | – | **0.576** | – | 0.546 | 0.527 |
| 4 content in question | squad2_answerable | 1000 | 0.516 | 0.518 | 0.505 | 0.490 | – | **0.740** | – | 0.646 | 0.671 |
| 4 content in question | anli | 1000 | 0.342 | 0.339 | 0.318 | 0.293 | – | 0.402 | – | **0.430** | 0.404 |
| 4 content in question | wanli | 1000 | 0.488 | 0.356 | 0.234 | 0.564 | – | **0.715** | – | 0.566 | 0.558 |
| | **Regime 1 mean (7 tasks)** | | 0.499 | 0.379 | 0.411 | 0.482 | – | **0.615** | – | 0.579 | 0.570 |
| | **Regime 2 mean (4 tasks)** | | 0.296 | 0.375 | 0.390 | **0.396** | 0.391 | 0.358 | 0.352 | 0.354 | 0.463 |
| | **All 11 tasks** | | 0.425 | 0.378 | 0.403 | 0.450 | – | **0.522** | – | 0.497 | 0.531 |

- Content in the question (regime 1): LIDM MBL beats shipped Laya on average (0.615 vs 0.579) and on 5 of 7 tasks;
  it loses BoolQ (-10) and ANLI (-3). WANLI train and FEVER-NLI are in LIDM's stage-1 data, so WANLI is not
  zero-shot for LIDM and VitaminC is near-domain; without WANLI the lead is 0.599 vs 0.581. The bi-encoders are at or
  below majority, as expected.
- Content in the options (regime 2): no model reads it zero-shot. RACE/QuAIL best 0.44-0.47 with gte alone at
  0.42-0.43; When2Call at or below majority for everyone (Laya 0.14: systematically wrong); typed-decisions
  0.35-0.38 for all zero-shot models vs 0.745 for the in-domain Laya-TD.
- Options in the question changes nothing (-0.5 / -0.6 points). That model never saw options in its question
  during training, so this does not show the late split is free; it shows the zero-shot weakness on regime 2 is not
  fixed by giving pass A the options. The joint-encoder control (run F) trained on the same data is the real test.
- The larger stage-1 encoder (MBL) reads much better than stage-1b gte (+13 points on regime 1), while gte is ahead
  on regime 2 (+4).

---

## Kill-criterion test: typed decisions, same backbone and data (2026-11-19)

Both ModernBERT-large, fine-tuned on the same 1,080 typed-decisions train cases (hard labels), selected on the same
120 dev cases, tested on the official 400 cases / 2,000 questions (`scripts/run_td_lidm.sh`, `scripts/run_td_night.sh`;
`scripts/compare_typed_decisions.py --table`). LIDM: stage-1 MBL LoRA init, lr 1e-4, best pass on dev (3 of 6).
Laya: its own trainer (RLCD, temperature calibration on a 10% slice), encoder lr 2.5e-5 / head 1e-4 (dev: 0.773 for
all three rates), final checkpoint after 4 passes.

| Model (test) | Accuracy [95% CI] | choice | yes/no | score | Brier vs soft | ECE | score MAE | ms per case |
|---|---|---|---|---|---|---|---|---|
| **LIDM fine-tuned** | **0.757** [0.734, 0.778] | 0.733 | 0.815 | 0.730 | 0.174 | 0.086 | 0.298 | 66.6 |
| **Laya fine-tuned (same data)** | **0.752** [0.731, 0.773] | 0.728 | 0.833 | 0.709 | 0.157 | 0.057 | 0.338 | 42.6 |
| Laya-TD (published checkpoint) | 0.767 [0.747, 0.788] | 0.732 | 0.858 | 0.725 | 0.061 | 0.214 | 0.242 | 43.4 |
| LIDM zero-shot (stage 1) | 0.384 | 0.287 | 0.463 | 0.396 | 0.315 | 0.175 | 0.612 | 63.4 |
| Laya zero-shot (shipped) | 0.358 | 0.283 | 0.482 | 0.323 | 0.316 | 0.178 | 0.694 | 43.8 |

Paired on the same 2,000 questions (CI by resampling cases): LIDM − same-data Laya **+0.5 points [−1.5, +2.3]**,
McNemar p = 0.65; LIDM − Laya-TD −1.1 [−2.9, +0.7], p = 0.25.

Leave one workflow out (train on 3 workflows at each model's setting, accuracy on the held-out workflow's 100 test
cases; chance 0.318):

| Held-out workflow | LIDM | Laya | LIDM − Laya [95% CI] | LIDM / Laya on the 3 trained workflows |
|---|---|---|---|---|
| agent_trace_observability | 0.414 | 0.340 | +7.4 [+0.4, +14.0] | 0.793 / 0.763 |
| customer_service | 0.472 | 0.418 | +5.4 [+0.4, +10.4] | 0.763 / 0.704 |
| invoice_processing | 0.380 | 0.426 | −4.6 [−10.2, +1.0] | 0.739 / 0.727 |
| security_incidents | 0.460 | 0.416 | +4.4 [−1.2, +10.2] | 0.773 / 0.697 |
| **pooled (400 cases)** | **0.432** | **0.400** | **+3.2 [+0.1, +6.2]** | 0.767 / 0.723 |

Context suite after fine-tuning (regime-1 mean, 7 tasks): LIDM 0.601 (from 0.615), Laya 0.583 (from 0.579); without
WANLI, which is in LIDM's stage-1 data, 0.587 vs 0.585. Regime-2 tasks other than typed decisions unchanged (~0.37).

Against the decision rule fixed before the run:
- Accuracy within 3 points of same-data Laya: **yes** (+0.5, a tie; −1.1 vs Laya-TD, also within noise).
- A concrete advantage: calibration **no** (ECE 0.086 vs 0.057, Brier 0.174 vs 0.157; LIDM has no calibration step
  yet, Laya's trainer fits a temperature; score MAE favours LIDM 0.298 vs 0.338); latency **no** (67 vs 43 ms per
  5-question case: pass A re-reads the state for every question); regime-1 reading **no** (tie without WANLI); new
  schemas **weak yes** (+3.2 pooled, CI just above 0, 3 of 4 folds), at a level near zero-shot for both (0.40-0.43)
  and partly confounded: LIDM keeps its best pass on dev, Laya its final one.
- Outcome: **grey zone**. Late interaction matches the joint encoder on the same backbone and data but is not better
  at what matters for 2-4 option decisions, and is slower.

---

## Stage 2: same-data comparison on a commercial-clean decision mix (2026-11-19)

Both ModernBERT-large, one pass over the same 102,818 cases / 134,621 questions (`data/stage2/`, 365 label sources,
~95k distinct option strings; see EXPERIMENTS.md), final checkpoints. LIDM: stage-1 MBL LoRA init, lr 1e-4, 2 h 12 min.
Laya: shipped, its trainer and defaults at `max_len 1024, head_max_len 512`, 2 h 05 min; it dropped 17 questions
(0.01%) whose options did not fit. Typed-decisions and every evaluation set below were held out of training.

**Typed decisions (400 cases, 2,000 questions; schemas never trained on), then 2c (fine-tuned on its train split with
the direct runs' recipes).** Paired differences with a 95% CI by resampling cases.

| | LIDM | Laya | LIDM − Laya |
|---|---|---|---|
| zero-shot (stage 1 / shipped) | 0.384 | 0.358 | |
| after stage 2 | **0.520** | **0.503** | +1.8 [−0.8, +4.3], p = 0.14 |
| direct fine-tune (kill test) | 0.757 | 0.752 | +0.5 [−1.5, +2.3], p = 0.65 |
| stage 2 + 2c | **0.769** | **0.757** | +1.3 [−0.6, +3.1], p = 0.17 |
| 2c gain over direct | +1.3 [−0.6, +3.1] | +0.5 [−1.3, +2.1] | |
| 2c calibration: ECE raw → temperature per type fitted on dev | 0.135 → **0.041** | 0.089 → 0.060 | |
| 2c Brier vs soft targets, raw → fitted | 0.221 → **0.123** | 0.173 → 0.153 | |
| ms per 5-question case | 64 | 44 | |

Laya-TD (published) 0.767, Brier 0.061. Stage 2 lifts both by 14-15 points on unseen schemas, more than training on
typed-decisions' own other three workflows did (0.43 / 0.40), and LIDM + 2c reaches Laya-TD's accuracy.

**Context suite after stage 2.** Regime-1 mean LIDM 0.640, Laya 0.631 (LIDM: PubMedQA +6, SQuAD 2.0 +3, WANLI +7
(stage-1 data); Laya: VitaminC +7, ANLI +1; BoolQ 0.795 vs 0.800). Regime 2: RACE Laya +4.7, QuAIL tie (0.54),
When2Call LIDM 0.374 vs Laya 0.135.

**Label variety and length** (`artifacts/label_suite`; paired 95% CI by resampling questions):

| Set | bge | gte | Laya shipped | LIDM stage 1 | Laya stage 2 | LIDM stage 2 | LIDM − Laya (stage 2) |
|---|---|---|---|---|---|---|---|
| 40 held-out label sets (3,209 q) | 0.375 | 0.366 | 0.510 | 0.467 | **0.571** | 0.553 | −1.8 [−3.5, −0.2] |
| ↳ 2-4 options (2,538) | 0.428 | 0.414 | 0.562 | 0.530 | 0.632 | 0.620 | −1.2 [−3.2, +0.7] |
| ↳ 5-12 options (454) | 0.211 | 0.222 | 0.302 | 0.264 | 0.359 | 0.341 | −1.8 [−6.8, +3.3] |
| ↳ 13-40 options (217) | 0.097 | 0.106 | 0.332 | 0.157 | 0.300 | 0.207 | −9.2 [−15.7, −3.2] |
| Banking77, 77 names (1,000) | **0.625** | 0.549 | 0.360 | 0.447 | 0.547 | 0.434 | −11.3 [−15.1, −7.5] |
| Banking77, 77 `name: description` (1,000) | **0.624** | 0.557 | 0.405 | 0.397 | 0.360 | 0.473 | +11.3 [+7.1, +15.2] |
| CLINC150, 150 names (1,000)* | 0.590 | 0.611 | 0.524 | 0.795 | 0.614 | 0.690 | +7.6 [+3.7, +11.6] |

\* Not clean for LIDM: its stage-1 data (UTCD) contains CLINC150 train, which explains stage-1 LIDM's 0.795.
Held-out sources: Laya better on 22 of 40, LIDM on 13. Their options are mostly short (88% ≤ 20 characters).

- The hypothesis holds only for options that are both **many and long**: with 77 descriptions (~100 characters each,
  ~2,000 tokens in all) Laya's packed 512-token option budget truncates and LIDM wins by 11 points. With many
  **short** options Laya, given a 512-token budget, is better (Banking77 names −11, 13-40 options −9), and stage 2
  helped Laya there (0.360 shipped / 0.427 B3 → 0.547) but not LIDM (0.447 → 0.434).
- In the many-and-long regime a plain bi-encoder still wins (bge 0.624 vs LIDM 0.473).

**Selective prediction** (answer only when confidence ≥ threshold; `scripts/selective_typed_decisions.py`, typed-decisions
test after 2c; "cal." = one temperature per question type fitted on the 120 dev cases):

| Threshold | LIDM raw | LIDM cal. | Laya raw | Laya cal. |
|---|---|---|---|---|
| ≥ 0.7 | 86% answered, 0.816 | 64%, 0.867 | 82%, 0.823 | 79%, 0.832 |
| ≥ 0.8 | 78%, 0.839 | 52%, 0.898 | 71%, 0.858 | 62%, 0.875 |
| ≥ 0.9 | 70%, 0.859 | 38%, 0.916 | 49%, 0.912 | 38%, 0.934 |
| ≥ 0.95 | 62%, 0.874 | 25%, 0.948 | 31%, 0.959 | 19%, 0.965 |
| accuracy on the most confident 50% / 80% | 0.902 / 0.834 | 0.899 / 0.829 | 0.905 / 0.829 | 0.901 / 0.828 |
| AUROC of confidence for correctness | 0.751 | 0.746 | 0.782 | 0.773 |
| dev threshold for 95% accuracy, on test | 42%, 0.914 | 46%, 0.908 | 38%, 0.938 | 43%, 0.921 |

Ranking quality is the same (0.90 accuracy on the most confident half for every model); raw LIDM is overconfident
(≥ 0.95 delivers 0.874) until the temperature step; thresholds chosen on dev undershoot their target on test by 2-4
points (dev is small and easier: 0.80 vs 0.76-0.77). Laya-TD's soft-target confidences sit lower (≥ 0.7 covers 18%
at 0.997; most-confident half 0.903). On unseen schemas (stage 2, no typed-decisions training) confidence carries
little: AUROC 0.62 (LIDM) / 0.66 (Laya), most confident 10% only 0.72 / ~0.75 correct.

**Against the kill-criterion rule:** accuracy within 3 points of same-data Laya (+1.3, a tie); one concrete
advantage: calibration after the same post-hoc temperature step (ECE 0.041 vs 0.060, Brier 0.123 vs 0.153; point
estimates); latency still worse (64 vs 44 ms per case); label variety: Laya at least as good except many long options.
Formally a pass, by a narrow margin and on calibration only.

---

## LIDM-X: options read pass A at every layer (2026-11-20)

`--option-attn state` (EXPERIMENTS.md, "LIDM-X"), trained exactly as `s2-lidm`: stage-1 MBL LoRA init, one pass over
the same 134,621 stage-2 questions, lr 1e-4, final checkpoint (2 h 21 min). `scripts/run_stage2_lidmx.sh`,
`scripts/summarize_lidmx.py`; paired 95% CIs by resampling questions (typed decisions: by case).

Stage-2 dev accuracy at steps 1k / 2k / 3k / 4k / end: LIDM-X 0.625 / 0.650 / 0.676 / 0.668 / 0.675, LIDM 0.597 /
0.634 / 0.646 / 0.648 / 0.655.

| Task | Laya s2 | LIDM s2 | **LIDM-X s2** | X − Laya | X − LIDM |
|---|---|---|---|---|---|
| VitaminC | **0.768** | 0.696 | 0.708 | −6.0 [−9.0, −3.0] | +1.2 [−0.4, +2.7] |
| RACE | 0.505 | 0.458 | **0.607** | **+10.2** [+6.9, +13.5] | +14.9 [+11.5, +18.6] |
| Banking77, 77 names | 0.547 | 0.434 | **0.625** | **+7.8** [+4.5, +11.2] | +19.1 [+15.7, +22.5] |
| 13-40 options (held-out label sets) | **0.300** | 0.207 | 0.286 | −1.4 [−7.8, +5.1] | +7.8 [+2.3, +13.4] |
| Banking77, 77 descriptions* | 0.360 | 0.473 | **0.627** | +26.7 [+23.1, +30.3] | +15.4 [+12.2, +18.7] |
| QuAIL | 0.544 | 0.540 | **0.631** | +8.7 [+5.7, +11.9] | +9.1 [+5.8, +12.5] |
| When2Call | 0.135 | 0.374 | **0.380** | +24.5 [+20.8, +28.3] | +0.6 [−1.7, +2.8] |
| BoolQ | 0.800 | 0.795 | 0.802 | +0.2 [−2.6, +2.9] | +0.7 [−1.1, +2.6] |
| PubMedQA | 0.607 | **0.669** | 0.622 | +1.5 [−1.4, +4.3] | −4.7 [−7.2, −2.2] |
| SQuAD 2.0 | 0.677 | 0.707 | 0.704 | +2.7 [−0.1, +5.5] | −0.3 [−2.1, +1.5] |
| CondaQA | 0.533 | 0.523 | **0.566** | +3.3 [+0.0, +6.6] | +4.3 [+1.2, +7.4] |
| ANLI | **0.441** | 0.429 | 0.412 | −2.9 [−6.6, +0.9] | −1.7 [−4.8, +1.3] |
| 40 held-out label sets (all) | 0.571 | 0.553 | **0.594** | +2.3 [+0.7, +4.0] | +4.2 [+2.7, +5.6] |
| typed decisions, unseen schemas | 0.503 | 0.520 | **0.550** | +4.7 [+2.3, +7.3], p < 0.001 | +3.0 [+0.9, +5.2], p = 0.007 |
| typed decisions after 2c | 0.757 | 0.769 | **0.774** | +1.8 [−0.1, +3.6], p = 0.049 | +0.5 [−1.0, +2.1] |

\* Laya ran at head_max_len 512 (6 tokens per option at N = 77); part of this gap may be that setting (BACKLOG item 7).
bge alone: Banking77 names 0.625, descriptions 0.624.

2c calibration with one temperature per question type fitted on dev: ECE 0.163 -> 0.043, Brier vs soft 0.261 ->
0.137 (temperatures 3.7 / 5.0 at the bound / 3.3: raw LIDM-X is more overconfident than LIDM); s2c-LIDM 0.041 / 0.123,
s2c-Laya 0.060 / 0.153. Selective (calibrated): confidence >= 0.9 answers 40% at 0.930 (LIDM 38% at 0.916, Laya 38% at
0.934); most confident half 0.903. Latency 67 ms per 5-question case (LIDM 64, Laya 43).

**Decision (rule from `reports/Recovering interaction after late splits.md`):** primary metric, mean LIDM-X − Laya
over VitaminC, RACE, Banking77 names and 13-40 options, **+2.7** (LIDM: −8.1; threshold −4.1). Guardrails: Banking77
descriptions +26.7 (>= +5), typed decisions after 2c +1.8 (within 1.5 or better). **LIDM-X adopted.** What it did not
fix: VitaminC (one changed fact flips the answer) is still −6 vs Laya, the case where the state may need to read the
option (LIDM-DJ, two-way top layers); ANLI −2.9 and PubMedQA −4.7 vs LIDM are the regressions to watch. Single seed.

---

## LIDM-DJ: top 7 layers joint per option (2026-11-20)

`--option-attn dj --joint-layers 7` (EXPERIMENTS.md), trained exactly as `s2-lidm` / `s2x-lidm` (stage-1 MBL LoRA init,
one stage-2 pass, final checkpoint; `--max-tokens 48000`), 3 h 51 min. Dev at 1k / 2k / 3k / 4k / end: 0.587 / 0.636 /
0.641 / 0.658 / 0.664 (LIDM-X 0.625 / 0.650 / 0.676 / 0.668 / 0.675). Paired 95% CIs by resampling questions.

| Task | Laya | LIDM-X | **LIDM-DJ** | DJ − Laya | DJ − LIDM-X |
|---|---|---|---|---|---|
| VitaminC | **0.768** | 0.708 | 0.668 | −10.0 | −4.0 [−6.3, −1.7] |
| RACE | 0.505 | **0.607** | 0.437 | −6.8 | −17.0 [−20.5, −13.6] |
| Banking77, 77 names | 0.547 | **0.625** | 0.385 | −16.2 | −24.0 [−27.4, −20.4] |
| 13-40 options | **0.300** | 0.286 | 0.217 | −8.3 | −6.9 [−13.4, −1.4] |
| Banking77, 77 descriptions | 0.360 | **0.627** | 0.409 | +4.9 | −21.8 [−25.0, −18.5] |
| QuAIL | 0.544 | **0.631** | 0.523 | −2.1 | −10.8 [−13.9, −7.6] |
| When2Call | 0.135 | **0.380** | 0.372 | +23.7 | −0.8 [−3.1, +1.6] |
| BoolQ | 0.800 | 0.802 | **0.804** | +0.4 | +0.2 [−1.7, +2.0] |
| PubMedQA | 0.607 | 0.622 | **0.673** | +6.6 | +5.1 [+2.5, +7.7] |
| SQuAD 2.0 | 0.677 | 0.704 | **0.732** | +5.5 | +2.8 [+0.8, +4.6] |
| CondaQA | 0.533 | 0.566 | **0.589** | +5.6 | +2.3 [−0.5, +5.1] |
| ANLI | 0.441 | 0.412 | **0.443** | +0.2 | +3.1 [+0.2, +6.0] |
| 40 held-out label sets | 0.571 | **0.594** | 0.567 | −0.4 | −2.8 [−4.3, −1.3] |
| typed decisions, unseen schemas | 0.503 | **0.550** | 0.532 | +3.0 [+0.1, +5.8] | −1.8 [−4.2, +0.7] |

Latency 107 ms per 5-question case (LIDM-X 67, Laya 43). Primary metric −10.3 (LIDM −8.1, LIDM-X +2.7).

- **Rejected.** Two-way interaction in the top 7 layers does not fix VitaminC (−4 vs LIDM-X) and loses everything
  that depends on reading the options: RACE −17, Banking77 names −24, descriptions −22 vs LIDM-X, below even
  the old LIDM on most of them. Options that spend 21 layers blind to the state are the problem, not the missing
  state-reads-option direction.
- Where the content is in the question (PubMedQA +5, SQuAD 2.0 +3, ANLI +3, CondaQA +2 vs LIDM-X), the state reading
  its yes/no options in the top layers helps a little; a hybrid (LIDM-X everywhere plus the state reading the option in
  the top layers) is the remaining variant, at DJ's cost.

---

## LIDM-X trained from scratch: its own stage 1 + stage 2 (2026-11-20)

Stage 1 in LIDM-X form (`s1x-lidm`, `--stage1 --encoder-lora 16 --option-attn state`), capped at 2,000 steps x 128
(26% of the stage-1 data; time budget), 2 h 04 min; then stage 2 exactly as `s2x-lidm` (`s2xf-lidm`), 2 h 20 min.

**Stage 1 alone:** zero-shot validation accuracy (`val_acc_zs`) 0.39-0.40 from step 500 on, against 0.12-0.29 for the
old stage 1 over 7,608 steps; Banking77 test (3,080) **0.628** at N=77 and 0.770 at N=20 (old stage 1: 0.434 /
0.650; bge 0.695 / 0.844).

**After stage 2** (paired 95% CIs; "old" = LIDM-X from the old stage-1 checkpoint, `s2x-lidm`):

| Task | Laya | LIDM-X, old stage 1 | **LIDM-X, own stage 1** | own − Laya | own − old |
|---|---|---|---|---|---|
| VitaminC | **0.768** | 0.708 | 0.701 | −6.7 | −0.7 [−2.6, +1.3] |
| RACE | 0.505 | 0.607 | **0.610** | +10.5 | +0.3 [−2.1, +2.7] |
| Banking77, 77 names | 0.547 | **0.625** | 0.602 | +5.5 | −2.3 [−4.5, +0.0] |
| 13-40 options | **0.300** | 0.286 | 0.295 | −0.5 | +0.9 [−4.2, +6.5] |
| Banking77, 77 descriptions | 0.360 | **0.627** | 0.612 | +25.2 | −1.5 [−3.9, +0.9] |
| QuAIL | 0.544 | 0.631 | **0.637** | +9.3 | +0.6 [−1.7, +2.9] |
| When2Call | 0.135 | **0.380** | 0.369 | +23.4 | −1.1 [−3.5, +1.3] |
| BoolQ | 0.800 | **0.802** | 0.768 | −3.2 | −3.4 [−5.4, −1.3] |
| PubMedQA | 0.607 | 0.622 | **0.656** | +4.9 | +3.4 [+1.1, +5.9] |
| SQuAD 2.0 | 0.677 | **0.704** | 0.689 | +1.2 | −1.5 [−3.6, +0.6] |
| CondaQA | 0.533 | **0.566** | 0.359 | −17.4 | −20.7 [−24.0, −17.6] |
| ANLI | **0.441** | 0.412 | 0.382 | −5.9 | −3.0 [−6.1, +0.2] |
| 40 held-out label sets | 0.571 | **0.594** | 0.572 | +0.1 | −2.2 [−3.6, −0.9] |
| typed decisions, unseen schemas | 0.503 | 0.550 | **0.562** | +6.0 [+3.7, +8.4] | +1.3 [−0.8, +3.3] |

Primary metric (mean − Laya over VitaminC, RACE, Banking77 names, 13-40 options): **+2.2** (old stage 1: +2.7).
Stage-2 dev at the end 0.659 (old 0.675).

- Stage 1 in LIDM-X form reads labels far better (+19 on Banking77 after stage 1), but **stage 2 erases the
  difference**: after both stages the two LIDM-X models tie on the option-reading rows (within ±2.3) and on typed
  decisions (+1.3, n.s.; 0.562 is the best unseen-schema score of any model).
- **CondaQA collapse is an abstain-option bias that depends on the wording, in both models** (400 questions, gold 373
  yes/no + 27 "don't know"):

  | CondaQA variant | LIDM-X, old stage 1 | LIDM-X, own stage 1 |
  |---|---|---|
  | yes / no / "don't know" (as evaluated) | 0.578 (47 abstentions) | 0.360 (206) |
  | yes / no / "the text does not say" | 0.233 (290) | 0.420 (161) |
  | yes / no only, the 373 yes/no questions | 0.689 | 0.668 |

  The yes/no reasoning is intact in both; the third option decides the score, and which abstain wording triggers
  over-abstention flips between runs. Both also over-predict "neutral" on ANLI (209 / 180 vs 146 gold). Training data:
  52% of stage-1 choice questions carry an abstain-type option (mostly NLI "neutral"), gold in 36% of those; stage 2:
  14%, gold in 25%. So the CondaQA drop is run-to-run variance in a learned abstain prior, not lost understanding,
  and not specific to the shortened stage 1 (the earlier attribution was wrong).
- Takeaway: stage 1 in LIDM-X form helps label reading early but stage 2 levels it out; the deciding checkpoint stays
  `s2x-lidm`; the production retrain (BACKLOG item 6) should still run stage 1 at full length (ANLI −3.0 and BoolQ
  −3.4 are unexplained by the abstain bias).

---

## Abstain options with descriptions (2026-11-20)

`scripts/eval_abstain_descriptions.py` (`artifacts/abstain_descriptions.json`): CondaQA, ANLI, VitaminC, 1,000 questions
each (drawn with only these three tasks loaded, so not the same questions as the context-suite tables; identical across
models here). Cells: accuracy (abstain predictions), paired difference to the as-evaluated options.

| Task, options (gold abstains) | LIDM-X, old stage 1 | LIDM-X, own stage 1 | Laya |
|---|---|---|---|
| CondaQA (64): yes / no / don't know | 0.557 (127) | 0.341 (547) | 0.500 (158) |
| ↳ only "don't know" described | 0.437 (356), −12.0 | 0.369 (495), +2.8 | 0.510 (147), +1.0 |
| ↳ all three described | 0.567 (56), +1.0 | 0.285 (629), −5.6 | 0.541 (71), +4.1 |
| ↳ "the text does not say" | 0.228 (733), −32.9 | 0.401 (421), +6.0 | 0.544 (74), +4.4 |
| ↳ "the text does not say" + description | 0.491 (260), −6.6 | 0.490 (247), +14.9 | 0.537 (77), +3.7 |
| ANLI (320): entailment / neutral / contradiction | 0.390 (520) | 0.371 (416) | 0.404 (589) |
| ↳ all three described | 0.438 (368), +4.8 | 0.357 (593), −1.4 | 0.415 (595), +1.1 |
| VitaminC (128): supports / refutes / not enough information | 0.691 (8) | 0.683 (44) | 0.787 (146) |
| ↳ abstain option described | 0.697 (97), +0.6 | 0.686 (94), +0.3 | 0.762 (169), −2.5 |

- Descriptions do not reliably fix the abstain bias: they shift LIDM-X's abstain rate by hundreds of predictions in a
  model- and wording-dependent direction (describing only the abstain option: −12 for the adopted model). Describing
  all options is the safest variant (adopted model +1.0 CondaQA, +4.8 ANLI). Across the five CondaQA variants LIDM-X
  ranges 0.23-0.57 and abstains 56-733 times; Laya stays at 0.50-0.54 and 71-158 abstentions.
- VitaminC is the opposite case: LIDM-X almost never abstains (8 vs 128 gold); describing the option raises that to 97
  without changing accuracy, so the VitaminC gap to Laya is not an abstain issue.
- Hypothesis: abstaining is a relative judgement (none of the others holds); Laya weighs all options in one sequence,
  LIDM-X's options never see each other inside the encoder (only in the 2-block head), so the abstain option is scored
  on how its own text matches the passage.

---

## Looped LIDM-X, stage 1 on the label-variety mix (2026-11-22)

`scripts/run_full_lidmx_loop.sh` (`s1bxl-lidm`): head run C (2 blocks looped, loop count sampled 1-4 per step,
per-loop FFN LoRA r=16, head blocks recomputed in backward), `--option-attn state`, one full pass over stage 1b
(522,691 questions, ~700k distinct option strings), lr 1e-4, batch 128 / micro 16; 13 h 13 min, peak 9.3 GB.
(A first launch ran out of GPU memory at step 12: the looped head kept every loop's activations; fixed by
checkpointing the head blocks, `tests/test_model.py::test_looped_head_checkpointing_is_exact`.)

Validation on `stage1b_val` (same set as the earlier stage-1b runs):

| Run (stage 1b) | Steps | Overall | Zero-shot labels (zs/) | Held-out labels (heldout/) |
|---|---|---|---|---|
| LIDM, gte-modernbert-base | 4,084 | 0.739 | 0.368 | 0.905 |
| LIDM, ModernBERT-large | 2,500 (time limit) | 0.747 | 0.337 | 0.925 |
| **LIDM-X looped, ModernBERT-large** | 4,084 | **0.797** | **0.428** | **0.978** |

At 1 / 2 / 4 loops: 0.797 / 0.797 / 0.797 overall at the end (0.787 / 0.787 / 0.786 at step 1,000).

Banking77 test (3,080), 1 / 2 / 4 loops: N=77 **0.617 / 0.617 / 0.618**, N=20 0.766 / 0.767 / 0.768. References:
LIDM-X unlooped, stage 1 capped at 2,000 steps (`s1x-lidm`) 0.628 / 0.770; old stage 1 0.434 / 0.650; stage-1b
LIDM 0.435-0.599; bge 0.695 / 0.844.

- LIDM-X lifts stage-1b label reading well above the earlier stage-1b runs (zs +6 to +9, held-out labels 0.978).
- Looping does nothing in stage 1: 1, 2 and 4 loops agree to the third decimal on validation and Banking77.
- More label variety did not raise Banking77 over LIDM-X on the original stage-1 data (0.617 vs 0.628, a different
  mix and length; no unlooped stage-1b LIDM-X run to separate the two).

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

