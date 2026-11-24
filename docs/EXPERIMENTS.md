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

## Context suite: zero-shot, existing checkpoints (built 2026-11-18, not yet run)

Does LIDM read the state where embeddings cannot? `src/cls/data/context_suite.py` + `scripts/eval_context_suite.py`,
chained by `scripts/run_context_suite.sh` (~1 h; results in `logs/context_suite/summary.log`, MLflow `context_suite`).

| Group | Tasks (≤1,000 questions each) | Regime |
|---|---|---|
| 1 System One | typed-decisions test (400 cases, ~2,000 questions) | mixed |
| 2 content in options | RACE, QuAIL, When2Call MCQ | 2 |
| 3 negation / contrast | CondaQA, VitaminC | 1 |
| 4 content in question | BoolQ, PubMedQA, SQuAD 2.0 answerability, ANLI, WANLI | 1 |

Models: LIDM stage-1b gte and stage-1 ModernBERT-large LoRA, bge and gte alone (floor), shipped Laya,
laya-typed-decisions (in-domain for group 1), optional Laya B4 (`WITH_B4=1`, retrained). Ablation `--options-in-question`
writes every option into pass A's question: a large gain on regime 2 means the head does not recover the interaction
the late split removes. Read: regime 1 should be near Laya; regime 2 is the design's risk. Evaluation only: RACE,
QuAIL and ANLI are non-commercial.

## Kill-criterion test: typed decisions, same backbone and data (built 2026-11-18)

GOALS.md kill criterion, condition 1 ("Laya wins on nuance tasks"); condition 2 (<20 options) already holds for
typed decisions (2-4 options). Both sides ModernBERT-large, fine-tuned on LocalLLaMA/typed-decisions train
(Apache-2.0), hard labels, split by case: 1,080 train / 120 dev cases (selection) / official 400-case test
(`scripts/prepare_typed_decisions.py` -> `data/td/`).
- LIDM: `scripts/run_td_lidm.sh` (stage-1 MBL LoRA init, lr 5e-5 / 1e-4 / 3e-4, 6 passes, dev-selected, then test
  and context suite).
- LIDM dev (600 questions): lr 5e-5 0.788, **1e-4 0.805** (selected, pass 3), 3e-4 0.795; overfits after pass 3.
- Rest of the chain: `scripts/run_td_night.sh`: LIDM test + context suite; references (Laya-TD, shipped Laya, LIDM
  zero-shot); Laya fine-tuned by its own trainer on `data/td/train_laya.jsonl` (encoder lr 1.25e-5 / 2.5e-5 / 5e-5,
  head 4x, 4 passes, dev-selected) + test + context suite; paired comparison (`scripts/compare_typed_decisions.py`);
  leave one workflow out (`data/td/lowo_*`) for both at their selected settings.
- Evaluation: `scripts/eval_typed_decisions.py` (accuracy with case-bootstrap CI, Brier vs soft targets, ECE, score
  MAE, latency per case). Reference: laya-typed-decisions.
- Decision: LIDM within 3 points of same-data Laya *and* one concrete advantage (calibration, latency per case,
  regime-1 reading) -> stage 2; within 3 but no advantage -> try looping once; more than 3 below -> kill, write up.

## Next stages on LIDM-X: soft targets, then a second pass with stage-1 replay (prepared 2026-11-23)

`scripts/run_s2_soft_replay.sh`, both from `checkpoints/s2x-lidm` at lr 5e-5, otherwise the `s2x-lidm` recipe:
- **Soft targets** (`s2xs-lidm`, `--soft-targets`, `decision_loss(target=...)`, `train.soft_target`): one more stage-2
  pass on `data/stage2_soft/train.jsonl` (`scripts/prepare_stage2.py stage2_soft`; same questions, options, gold and
  order as `data/stage2/train.jsonl`, verified), 81% of questions carry a distribution, 26k (19%) not one-hot:
  typed-decisions-synth teacher 21.2k, Civil Comments rater share 2.25k, tasksource 2.5k. Then 2c on typed-decisions
  train with its soft gold (`s2xsc-lidm`) vs the hard 2c (`s2xc-lidm`); calibration and selective prediction.
- **Replay** (`s2xr-lidm`): one more pass, hard labels, `data/stage2/train_replay.jsonl` (`scripts/prepare_stage2_replay.py`:
  134,621 stage-2 + 33,655 stage-1 questions = 20% replay; replay is 62% NLI / label-NLI).
- Both are second passes, so each shows "second pass + its ingredient" against `s2x-lidm`; a hard second pass without
  replay would be the control separating the two.

## Full LIDM-X training on the label-variety stage 1 (prepared 2026-11-21)

`scripts/run_full_lidmx_loop.sh` (looped head, run C, loops sampled 1-4; stage 1 only; the unlooped variant was dropped): stage 1 = one full pass over `data/stage1b_train.jsonl` (522,691 questions, ~700k
distinct option strings in the first 200k vs 2,089 in stage 1; 8.6 options and 249 tokens per question vs 3.6 and
117) with `--option-attn state`, lr 1e-4, batch 128 / micro 16, `--max-len 512 --max-opt-len 64`, validation every 1,000
steps on `stage1b_val` (zs/ 961, ood/ 1,500, heldout/ 1,936); then stage 2, typed decisions, context and label suites,
2c, calibration, the abstain-wording test and both decision tables (vs Laya, vs `s2x-lidm`). Smoke: worst-case
micro-batch fits at 4 loops; 13.7 s/step -> stage 1 ~16 h, ~16 h; stage 2 and the evaluations are not part of this script. Caveats: UTCD's CLINC150 is in the stage-1 base
(CLINC numbers not zero-shot), and the stage-1 sources are not all commercial-clean (research run).

## LIDM-DJ and LIDM-X with its own stage 1 (2026-11-20)

- **LIDM-DJ** (`--option-attn dj --joint-layers 7`, `model._dj_forward`): layers 1-21 as LIDM (pass A and each option
  alone, options at positions 0..L-1); layers 22-28 run every option jointly with a copy of its question's pass A,
  `[A ; option]`, two-way attention, option positions continuing at len(A), copies isolated from other options; pass A
  also continues alone for the head. `tests/test_lidmdj.py` checks both parts against HF's own forward / layer
  modules. `--max-tokens` splits micro-batches by padded tokens (the joint layers hold one state copy per option).
  Independent audit (fp64 tiny model, window shrunk to 8, joint 1/3/6): max error 4e-16 vs HF; it found budget and
  memory issues only (padding-aware budget, budget in `validate`, contiguous joint slices, int32 window masks,
  `bench_latency.py` refusing X/DJ checkpoints), all fixed.
- **Chain** (`scripts/run_s1x.sh`, then `scripts/run_full_chain.sh`): LIDM-X stage 1 from scratch capped at 2,000
  steps x 128 (time budget; the previous stage 1 plateaued from ~3,500 steps), LIDM-DJ stage 2 from the stage-1 MBL
  checkpoint (same recipe as `s2-lidm` / `s2x-lidm`), then LIDM-X stage 2 from its own stage 1 (`s2xf-lidm`).

## LIDM-X: options read pass A at every layer (built 2026-11-20)

From `reports/Recovering interaction after late splits.md`. `--option-attn state` (`Encoder.encode_x`,
`model._x_forward`): both passes run through ModernBERT's own layers (+ LoRA) in one loop; pass A attends to itself,
each option row attends to [its question's pass-A keys/values at that layer ; its own tokens], never to other options;
options start at position len(pass A) (shared, so order-free); local layers keep the |Δposition| <= 64 window on
those positions. One option row per (question, option): no sharing of option strings across questions. No new
parameters; head, layer mix and prior unchanged. `tests/test_lidmx.py` (fp32, gte-modernbert-base): pass A equals HF's
forward; every option equals HF's layers on one packed `[A; option]` sequence with the LIDM-X mask at every layer;
option order does not matter; LoRA gets gradients through both passes.

Smoke (ModernBERT-large, stage-2 data): worst-case micro-batch 8 (40 options x 128 tokens) fits; 2.33 s/step vs 2.13
for LIDM on the same slice (+9%); 69.6 ms per 5-question case. Run: `scripts/run_stage2_lidmx.sh` (stage 2 from the
stage-1 MBL LoRA checkpoint exactly as `s2-lidm`, then typed decisions, context suite, label suite, 2c, paired
comparisons; `scripts/summarize_lidmx.py` prints the decision table). Decision (from the report): adopt if the mean
LIDM-X − Laya gap on VitaminC, RACE, Banking77 names and 13-40 options is at most half of today's (−8.1), with
Banking77 descriptions >= +5 over Laya and typed decisions after 2c within 1.5 points.

## Stage 2: same-data comparison on a commercial-clean decision mix (2026-11-19)

`scripts/prepare_stage2.py` (`src/cls/data/stage2.py`) -> `data/stage2/`: 102,818 cases / 134,621 questions from
365 label sources: typed-decisions-synth (MIT; 23k questions), Open-Jev release-v2 (CC0; 20k), tasksource-jev rows with
`license_use == commercial` (61k, up to 250 per source, half option-rich), HelpSteer2 (15k score), Civil Comments (5k),
Aegis 2.0 (10k). Held out: typed-decisions and its four workflows' domains, every context-suite task, Banking77,
CLINC150, SST-5 (by source name), plus an exact state-prefix check against all evaluation items (5 dropped).
Many and long options kept on purpose (up to 40 options, 600 characters each; 5.4% of questions have >12 options,
3.6% have options >80 characters on average; ~95k distinct option strings): that is where late interaction should
beat a packed joint encoder. Laya trains at `max_len 1024, head_max_len 512` (its long-option setting, like B3).

- Training (`scripts/run_stage2_train.sh`): LIDM from stage-1 MBL LoRA, lr 1e-4, batch 32 (micro 8; 16 does not fit
  40 options x 128 tokens), `--max-opt-len 128`, one pass, final checkpoint; Laya from shipped, its defaults, one pass.
- Evaluation (`scripts/run_stage2_eval.sh`): typed-decisions test (schemas never trained on), context suite, label
  suite (`--items`: 40 held-out tasksource sources, Banking77 N=77 names and `name: description`, CLINC150 N=150) with
  accuracy by option count and option length; references stage-1 LIDM, shipped Laya, bge, gte. Then 2c: both
  fine-tuned on typed-decisions train with the direct runs' recipes, compared with td-lidm / td-laya.

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
