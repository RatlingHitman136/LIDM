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
- Reliability diagrams are not logged for the harness runs (only for the preliminary runs).

---
# Results

## Preliminary run (superseded by the table above; label ids with underscores, `predict_batch`) (test, 3080 examples, 77 options, zero-shot, one `choice` question)

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

