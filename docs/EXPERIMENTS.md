# Experiment plan

Full recipe and its evidence: [DESIGN.md](DESIGN.md), "Training plan" and "Evaluation and expected results";
data sources: [DATASETS.md](DATASETS.md). This file is the working plan: what ran, what runs next, and
the rules for deciding. Results: [RESULTS.md](RESULTS.md).

## Stages
| # | Stage | Data | Status |
|---|---|---|---|
| M0 | Baselines (Laya variants, fine-tuned Laya B4, bi-encoders) | M0 mixture, 12.9k items | done |
| M1 | Runs B / C / D on the M0 mixture, frozen encoder | M0 mixture | done, **off-plan**: see below |
| 1 | Label reading, run B: frozen (failed), then **encoder LoRA r=16** | `scripts/prepare_stage1.py` (974k questions, 2,118 distinct option strings) | **done**: LoRA reaches Banking77 0.434 @N=77 (RESULTS.md) |
| 2a | Typed decisions, **with encoder LoRA** (changed, see below): ablation grid A-D | design doc stage 2 mix (tasksource-jev, jev-distill-corpus-v3, rules, scores, routing, safety) | data script not written |
| 2b | Joint-encoder control F (and E becomes the default) | same mix, new sample | not built |
| 3 | Soft labels, then temperatures | UNLI, GoEmotions raters, DICES, ...; calibration split | not built |

**Why M1 was off-plan.** `docs/EXPERIMENTS.md` used to put the A-D grid on the M0 mixture "for equal-data
comparison with B4", while the design doc puts it in stage 2a, after label reading. The M0 mixture has only
152 distinct option strings, so every head memorised them (in-distribution val ~0.75, Banking77 zero-shot
~0.03). The comparison with B4 was never equal-data either: Laya's checkpoint had learned label reading before
we fine-tuned it. Stage 1 has to come before any head-to-head.

## Stage 1: label reading (done)
```bash
uv run python scripts/prepare_stage1.py                 # data/stage1_{train,val}.jsonl (~20 min, downloads ~3 GB)
uv run python -m cls.train --run B --stage1 --limit 64000 --lr 1e-4 --name stage1-B-lr1e-4-pilot   # pilot / LR slice (repeat per LR)
uv run python -m cls.train --run B --stage1             # full pass, MLflow experiment `stage1`
```
`--stage1` sets: 1 epoch, 128 questions per step (sorted by length, then split into micro-batches of 16), validation on `stage1_val.jsonl` every
500 steps, `best.pt` by `val_acc_zs`. Other settings as M1: AdamW (0.9, 0.98), head LR 1e-4, scalar LR 1e-2,
3% warmup, cosine to 10%, weight decay 0.05, clipping 1.0.

| Source (task prefix) | Questions | Type | Options |
|---|---|---|---|
| `synthetic` (synthetic_zeroshot_mixtral, task configs) | 152k (all usable texts) | choice | 2-24 of the task's hypothesis statements |
| `utcd/*` (9 in-domain sets, single-label texts) | 222k (33k cap per set) | choice | 2-24 of the set's labels |
| `label_nli` (zero-shot-label-nli, non-commercial rows dropped) | 300k | noul | label statement in the question |
| `nli/*` (MNLI, SNLI, WANLI, FEVER-NLI) | 300k | choice | entailment / neutral / contradiction |

Validation: UTCD's out-of-domain test sets, 300 items each, every label as an option. UTCD's "out-of-domain"
is a domain shift, not unseen labels, so the build tags each set by checking its labels against the training
options:

| Tag | Sets | Labels shared with training |
|---|---|---|
| `zs/` (zero-shot) | banking77, consumer_finance, multi_eurlex (61 items), patent | none |
| `ood/` | amazon_polarity 2/2, finance_sentiment 3/3, yelp 3/3, nlu_evaluation 45/68, snips 2/7 | some or all |

`val_acc_zs` (macro mean over the `zs/` sets) is the selection and gate metric; `val_acc` over all sets is
inflated by the `ood/` sets (a 5-step smoke run: 0.18 vs 0.057). `zs/banking77` is built in the exact format
of the final test (humanized labels, the Banking77 instruction) from Banking77's **train** split; the test
split is touched only by `scripts/eval_banking77.py` for final numbers.

Held out of training: Banking77 (texts and labels; the build prints a leak check, currently 0 shared label
strings and 0 shared texts), CLINC's banking + credit-card intents and "exchange rate", SGD's Check Balance
and Transfer Money, tasks named after Banking77 or CLINC in zero-shot-label-nli.

**Pilot and LR slice.** First run ~500 steps (64k questions, ~10M tokens, the slice size the design doc asks
for) at 5e-5 / 1e-4 / 2e-4 / 3e-4 and keep the LR with the best `val_acc_zs`.

**Go / no-go gate (after the pilot, before the full run: ~7.6k steps x 0.62 s, about 1.5 h with validation).** Continue only if `val_acc/zs/banking77`
(N=77) is clearly above M1's ~0.03 and climbing, target **>= 0.25**, and `val_acc_zs` is rising. If not, the
frozen encoder is the limit: bring LoRA (E) forward or replace the cosine prior with a bge-base cosine before
spending the full run. The thresholds are ours, not the design doc's.

Known gaps against the design doc's stage 1: 974k questions instead of ~1.5M (the synthetic set has 152k
usable texts, not 600k); length bucketing only within each 128-question step; the encoder runs on padded
batches (no unpadded/flash path); only 2,118 distinct
option strings. If the gate fails on label diversity, the next lever is the synthetic set's
`mixtral_refinedweb_categories` config (free-form category labels) or turning label_nli into choice questions.

**Outcome (2026-11-16).** The frozen setup failed: all four LR pilots and the full run stayed at chance on
`zs/` sets (Banking77 test 0.044), and a bge prior did not rescue it. Encoder LoRA r=16 passed the decision rule
(>= 0.36 at N=77 and >= 0.60 at N=20 on the Banking77 test): 0.434 / 0.641 after the full 7,608 steps (6.9 h).
Its 500- and 1,000-step pilots still looked weak (0.029, 0.072), so pilots shorter than ~1,500 steps
underestimate LoRA runs. **Plan change:** the encoder is adapted with LoRA from stage 1 on; the design doc's
"frozen until stage 2b" does not hold for this task. Next levers for the stage-1 plateau (from ~3,500 steps):
more label variety, LoRA + bge prior (both need `--init`, see BACKLOG.md).

## Stage 2a: ablation grid (after stage 1, frozen encoder + layer mix)
| Run | Head | Loops | Encoder | Question it answers |
|---|---|---|---|---|
| A | 2 layers self-attention over the concatenation | 1 | frozen | What does the split cost with a simple head? |
| B | 2 cross+self blocks | 1 | frozen | Which head type wins at equal params? |
| C | as B, init from stage-1 B (per-loop LoRA starts at zero) | 1-8 sampled, or fixed 4; scored at 1/2/4/8 | frozen | Does looping add accuracy at fixed params? |
| D | 8 unshared blocks (same compute as C at 4 loops) | 1 | frozen | Iso-compute control for C |
| E | best of B-D | best | LoRA r=16 | How much does adapting the encoder add? |
| F | Laya-style joint encoder + 2-layer head, same data | 1 | LoRA r=16 | What does the split cost overall? |

Cross-cutting: `dual_score` prior on/off; option representation (marker token / pooled / k=4 vectors).
The design doc expects looping to help rule and policy questions, not topic and intent labels; M1 matched
that (no gain on the M0 mixture), so C vs D is only meaningful on stage-2 data.

## Decision rules
- C (looped) >= D (unshared, same compute) -> looping pays off, keep it.
- Single-sequence Laya (B4 / F) wins nuance tasks and label sets < 20 -> late interaction not worth it.
- LIDM must beat B5b on Banking77 and match B4 on BoolQ/SST-5 (targets in GOALS.md).
- Never select checkpoints or hyperparameters on a test split; Banking77 test is for the final number only.

## Lessons from M1 (keep)
- Report zero-shot accuracy next to in-distribution val; a random val split that shares labels with
  training hides label memorisation.
- LR 1e-4 was best of {3e-5, 1e-4, 3e-4, 1e-3} for the head; scalar params need ~1e-2.
- On small data every head overfits after ~3 epochs (val loss up 2-3x by epoch 6 while accuracy is flat).
- Instruction wording matters (CLINC val 0.91 -> 0.59 with the Banking77 instruction); stage 1 varies it.

## Metrics
Accuracy, Brier, ECE (pre/post temperature), NLL, accuracy vs #options, ms/question (p50/p95), peak VRAM.
Training logs val accuracy per task (`val_acc/<task>`) and per type; M0 baselines go through
`cls.eval.harness`; LIDM through `scripts/eval_lidm.py` (the same harness and task set as M0, plus AG News and Emotion; logs to MLflow `baselines` as `lidm-<name>`) and `scripts/eval_banking77.py` for a quick Banking77-only check.
Latency (not measured yet): fresh label set vs cached options, N in {5, 20, 77, 255}; throughput sweep 1-128.

## VRAM (measured, RTX 4070 SUPER 12 GB, frozen encoder)
| Setup | Peak (and step time where measured) |
|---|---|
| M1 run B, M0 mixture, batch 32 | 5.4 GB |
| M1 run C (2 blocks, up to 8 loops) | 8.5-8.6 GB |
| M1 run D (8 blocks, 131.6M params) | 9.8 GB (nvidia-smi showed ~12 GB reserved) |
| Stage 1 run B, micro-batch 16, padded head, unsorted | 5.8 GB, 1.42 s/step (batch 128) |
| Stage 1 run B, micro-batch 32, padded head, unsorted | 9.9 GB, 1.91 s/step; crashed (fragmentation) before expandable segments |
| Stage 1 run B, micro-batch 16, packed head + length-sorted steps (default) | **3.5 GB, 0.62 s/step** |
| Stage 1 run B, micro-batch 32 / 64, packed + sorted | 5.6 GB, 0.72 s/step / 9.6 GB, 1.09 s/step |
| Stage 1 run B + encoder LoRA r=16 (gradient checkpointing), micro-batch 16 | 4.7 GB, 3.44 s/step |
| Stage 1 run B + encoder LoRA r=16, full run (7,608 steps, validation every 500) | 5.5 GB, 6.9 h (3.3 s/step incl. validation) |
| Stage 1 run B + encoder LoRA r=16, micro-batch 32 | 8.0 GB, 4.40 s/step |
| Stage 1 run B + bge prior, micro-batch 16 | ~0.76 s/step (pilot) |
