# LIDM vs Laya and plain retrieval: what the experiments to date show

Scope: local project files only (docs/, reports/, logs/). Citations give file, section and line numbers (line numbers as of 2026-11-18, working tree incl. uncommitted edits). All runs are single-seed ([BACKLOG.md §Evaluation gaps L98](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md); [RESULTS.md §M0 L5, §M1 Caveats L116](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)). Names used below: **L1** = stage-1 LIDM, ModernBERT-large + encoder LoRA r=16; **gte-1** = LIDM on gte-modernbert-base, stage 1; **gte-1b / mbl-1b** = stage-1b runs on gte-modernbert-base / ModernBERT-large; "retrieval" = cosine of a bi-encoder (bge-base-en-v1.5 or gte-modernbert-base); "refit s" = retrieval weight re-fitted on zero-shot *val* sets (legitimate); "best s on test" = weight picked on the Banking77 test itself (diagnostic only, not a valid result).

## 1. Goals and success criteria (GOALS.md): which are met, partly met, not met?

### Takeaway
Of the five success criteria, only "trains on a 12 GB GPU" and "cached-option latency <= 20 ms" are clearly met. Fresh-label latency misses by 0.9 ms. Calibration looks met before any temperature fitting, but the fitting that the criterion requires was never run. The primary goal (Banking77 N=77 >= 0.70, N=20 >= 0.85) and the nuance criterion (BoolQ/SST-5 at fine-tuned-Laya level) are **not met** by any LIDM configuration. The only LIDM numbers near 0.70 come from setups where the retrieval term dominates and the head adds nothing.

### Cited Findings
- Problem statement: Laya packs state, question and all options into one 512-token sequence; options share a ~192-token head budget (77 options ≈ 3-4 tokens each); model-card Banking77 0.425 vs Jev 0.870; SST-5 0.372; mean ECE 0.466 before temperature fit; typed-decisions zero-shot 0.362 — [GOALS.md §Problem L3-14](/home/ratlinghitman/code/LIDM/docs/GOALS.md)
- Primary goal: "Fix the high-cardinality / long-option-description weakness without a meaningful latency regression" — [GOALS.md §Primary goal L16-17](/home/ratlinghitman/code/LIDM/docs/GOALS.md)
- Success criteria (RTX 4070 SUPER): (1) Banking77 N=77 zero-shot **after training on the M0 mixture** >= 0.70 (beats B5b 0.695, B4 0.502), N=20 >= 0.85, Jev 0.87 stretch; (2) BoolQ >= 0.826, SST-5 >= 0.570 (B4); (3) p50 <= 35 ms fresh label set (~2x B0's 17.6 ms), <= 20 ms cached; (4) ECE <= 0.1 on every task after per-(type, option count) temperature fitting; (5) trains on 12 GB — [GOALS.md §Success criteria L19-25](/home/ratlinghitman/code/LIDM/docs/GOALS.md)
- Kill criterion: "If single-sequence Laya wins on nuance tasks AND label sets are typically <20 options, late interaction is not worth the complexity; report that and stop" — [GOALS.md §Kill criteria L32-34](/home/ratlinghitman/code/LIDM/docs/GOALS.md)
- The project's own scorecard for L1: Banking77 0.434 / 0.650 → not met; BoolQ/SST-5 0.670 / 0.374 → not met ("zero-shot vs B4's in-distribution"); latency 35.9 ms fresh (LoRA merged; 55.4 ms unmerged) / **12.6 ms cached** → "cached met; fresh at target (+0.9 ms)"; raw ECE <= 0.10 except SST-5 0.13 → "likely met; temperatures not fitted yet"; 5.5 GB peak → met — [RESULTS.md §Stage 1 "Against GOALS.md" L198-205](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Temperature fitting is still "not built" — [ARCHITECTURE.md §LIDM item 8 L46, §Code layout L98](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md); [BACKLOG.md §Evaluation gaps L97](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md)
- Best later Banking77 N=77 / N=20 numbers (names only, all 3,080 test items), none >= 0.70:
  - gte-1, refit s: 0.635 / 0.778, equal to gte alone 0.634 / 0.779 — [RESULTS.md §LIDM on gte-modernbert-base L252-256](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
  - L1 + bge prior, refit a = 160: 0.689 / 0.829 (almost pure bge); best on test 0.696 at a = 320 — [RESULTS.md §Stage 1 "LoRA + bge prior from L1" L146-155](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
  - gte-1b: 0.548 trained / 0.599 refit; mbl-1b: 0.433 / 0.435 — [RESULTS.md §Stage 1b table L292-296](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [summary.log L12-15, L29-32](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log)
- With label descriptions (a separate benchmark condition), gte-1 combined reaches 0.7065 at N=77 vs gte-alone-with-descriptions 0.7032; head alone 0.022. This row is in the log but not in RESULTS.md's table — [eval-b77-descriptions.log L10-15](/home/ratlinghitman/code/LIDM/logs/eval-b77-descriptions.log)
- Training VRAM: L1 5.5 GB; gte-1 3.1 GB; M1 D 9.8 GB (nvidia-smi ~12 GB reserved) — [RESULTS.md §Stage 1 Runs L137, L252-254](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md §VRAM L116-124](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)
- Criterion 1's training condition ("after training on the M0 mixture") was abandoned: M1 on the M0 mixture memorised its 152 labels (Banking77 ~0.03) and "Stage 1 has to come before any head-to-head" — [EXPERIMENTS.md §Stages "Why M1 was off-plan" L17-21](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)

| Criterion | Target | Best LIDM evidence | Status |
|---|---|---|---|
| 1. Banking77 names, N=77 / N=20 | >= 0.70 / >= 0.85 | L1 0.434 / 0.641 (0.650 harness); gte-1 refit 0.635 / 0.778; L1+bge refit 0.689 / 0.829 | Not met ([RESULTS.md L137-139, L146-155, L252-256](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |
| 2. BoolQ / SST-5 | >= 0.826 / >= 0.570 | L1 0.670 / 0.374 (zero-shot); M1 frozen B 0.660 / 0.380 (in-distribution) | Not met ([RESULTS.md L80-84, L185](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |
| 3. Latency p50 fresh / cached | <= 35 / <= 20 ms | L1 35.9 / 12.6 ms (LoRA merged) | Cached met; fresh +0.9 ms over ([RESULTS.md L215-225](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |
| 4. ECE <= 0.1 after temperatures | every task | L1 raw 0.03-0.10, SST-5 0.13 | Not verified; no temperatures fitted ([RESULTS.md L190-192, L204](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |
| 5. 12 GB GPU | yes | <= 6.6 GB in stage 1 runs | Met ([RESULTS.md L129-139](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |

### Inferences
- (Inference) Criterion 1 is only reachable so far by letting retrieval decide. Every LIDM number above 0.6 at N=77 comes with a head that is at chance (gte-1, head 0.015) or that the refit turns down to near zero (a = 160). So the criterion's intent, a trained reader that beats B5b, is further from met than the raw numbers suggest.
- (Inference) The kill criterion's first clause holds on current evidence: every Laya variant beats L1 on BoolQ, and B4 beats it on SST-5. The second clause ("label sets typically < 20 options") is a fact about deployment, not something any file measures, so the kill criterion can't be decided from these files.
- (Inference) The latency criterion was measured for L1 only (ModernBERT-large). gte-modernbert-base (149M) is smaller, so it would likely be faster, but no file measures it.

### Gaps
- No LIDM checkpoint other than L1 went through the full harness (BoolQ, SST-5, N-sweep, calibration). gte-1, gte-1b and mbl-1b have no BoolQ/SST-5/ECE/latency numbers in any file ([BACKLOG.md §Evaluation gaps L95](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md) still lists harness evaluation as a gap).
- Post-temperature ECE (criterion 4) has not been measured for any LIDM model.

## 2. What did the M0 baselines measure?

### Takeaway
Laya's accuracy collapses with option count (0.871 → 0.360 from N=5 to N=77). Fine-tuning (B4) and Laya's two workarounds (bge shortlist B1b, tournament B2) only soften the drop. A plain 110M bge-base bi-encoder (B5b) beats every Laya variant on Banking77 at every N, at 2.8 ms and with ECE 0.03. It is useless on BoolQ/SST-5, where Laya wins.

### Cited Findings
- Protocol: Banking77 test (3,080) at N=77; N-sweep = 1,000 test examples with N ∈ {5, 10, 20, 40} (gold + random labels, fixed seed); BoolQ validation (3,270, `noul`); SST-5 test (2,210, 5-level `score`); plain humanized label names, no descriptions. The fine-tuning mixture (12.9k items: AG News, emotion, DBpedia-14, CLINC150 minus banking/credit-card intents, BoolQ-train, MNLI, SST-5-train) excludes Banking77, so Banking77 is zero-shot for every model. BoolQ/SST-5 are in-distribution for B4 only. Hardware: RTX 4070 SUPER, bf16, batch-1 p50 over 100 calls — [RESULTS.md §M0 L5-8](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| id | Model | B77 N=77 | N=40 | N=20 | N=10 | N=5 | BoolQ | SST-5 | p50 ms | VRAM GB |
|---|---|---|---|---|---|---|---|---|---|---|
| B0 | Laya as shipped | 0.360 | 0.590 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 17.6 | 2.43 |
| B3 | Laya max_len 1024 / head 512 | 0.427 | 0.600 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 19.4 | 2.45 |
| B1b | Laya + bge shortlist k=20 | 0.610 | 0.669 | 0.746 | 0.817 | 0.871 | 0.779 | 0.305 | 16.2 | 2.67 |
| B2 | Laya + tournament (groups of 16) | 0.585 | 0.649 | 0.743 | 0.817 | 0.871 | 0.779 | 0.305 | 34.2 | 2.45 |
| B4 | Laya fine-tuned on M0 mixture (4 epochs) | 0.502 | 0.657 | 0.776 | 0.837 | 0.893 | 0.826* | 0.570* | 17.5 | 2.43 |
| B5 | raw ModernBERT-large mean-pool cosine | 0.108 | 0.158 | 0.221 | 0.304 | 0.430 | 0.380 | 0.180 | 9.2 | 0.82 |
| B5b | bge-base-en-v1.5 cosine (110M) | 0.695 | 0.753 | 0.844 | 0.887 | 0.931 | 0.395 | 0.309 | 2.8 | 0.24 |
| B6 | laya-typed-decisions | 0.402 | 0.623 | 0.764 | 0.828 | 0.871 | 0.780 | 0.427 | 17.2 | 2.45 |

Source for the whole table: [RESULTS.md §M0 Accuracy L12-21 and model list L36](/home/ratlinghitman/code/LIDM/docs/RESULTS.md). \* = in-distribution for B4. For N ≤ 10, B1b/B2/B3 rows equal B0 because the same code path runs ([RESULTS.md L37](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).

| id | ECE / NLL N=77 | N=40 | N=20 | N=10 | N=5 | BoolQ | SST-5 |
|---|---|---|---|---|---|---|---|
| B0 | 0.18 / 3.21 | 0.17 / 2.22 | 0.13 / 1.32 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B2 | 0.03 / 3.89 | 0.07 / 3.24 | 0.05 / 2.55 | 0.04 / 0.66 | 0.15 / 0.49 | 0.07 / 0.54 | 0.33 / 2.07 |
| B4 | 0.15 / 2.54 | 0.14 / 1.73 | 0.10 / 1.06 | 0.09 / 0.73 | 0.08 / 0.46 | 0.14 / 0.68 | 0.28 / 1.40 |
| B5b | 0.03 / 1.13 | 0.03 / 0.86 | 0.04 / 0.51 | 0.02 / 0.36 | 0.01 / 0.21 | 0.43 / 1.31 | 0.29 / 1.60 |
| B6 | 0.20 / 2.76 | 0.15 / 1.58 | 0.09 / 0.94 | 0.27 / 0.84 | 0.37 / 0.80 | 0.08 / 0.50 | 0.06 / 1.24 |

Source: [RESULTS.md §M0 Calibration L25-34](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) (B3, B1b, B5 rows are also there). For B5/B5b the softmax temperature was fitted on 300 mixture items ([RESULTS.md L36](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).

- Finding 1, high-cardinality ceiling: Laya 0.87 → 0.82 → 0.75 → 0.59 → 0.36 for N 5 → 77. B3 recovers only N=77 (0.427, matching the card's 0.425) — [RESULTS.md §Findings 1 L40](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 2: B4 helps broadly but keeps the ceiling: N=77 0.502 (+14 pts), SST-5 MAE 0.47 vs 1.06 levels for B0. Caveat: its BoolQ/SST-5 gains are in-distribution — [RESULTS.md §Findings 2 L41](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 3: B5b beats every Laya variant on Banking77 at every N; it is "a 'free' baseline, not an equal-data one" (bge pretrained on large retrieval data); LIDM "must at least match B5b on Banking77 *and* B4 on BoolQ/SST-5 to be worth building" — [RESULTS.md §Findings 3 L42](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 4: the shortlist needs a proper embedder: with Laya's own encoder as embedder it scored 0.256; with bge it scores 0.610 — [RESULTS.md §Findings 4 L43](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 5: shipped Laya is over-confident at N >= 40 (ECE 0.17-0.18) and on SST-5 (0.33). B4's fitted `score`/`noul` temperatures hit the [0.5, 5] clamp. B2 has the best N=77 ECE but the worst NLL (zero mass on eliminated labels). B6 has ECE 0.37 at N=5 — [RESULTS.md §Findings 5 L44](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 6: single-pass Laya 17-19 ms (model card: 33-40 ms on a T4); tournament 34 ms; bi-encoder 3-9 ms — [RESULTS.md §Findings 6 L45](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Finding 7, label-format sensitivity: `card_arrival` vs `card arrival` moved B0 N=77 from 0.377 to 0.360 and ECE from 0.51 to 0.18, so numbers compare only at fixed option strings — [RESULTS.md §Findings 7 L46](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Caveats: one seed for B4; `src/cls/data/` was recreated on 2026-11-14 and is within ~0.03 of the table on 200-item quick runs, so the doc advises rerunning `scripts/run_m0_all.sh` before using these as exact comparison points; 72 of 12.9k items were dropped by Laya's `options_collapsed` check — [RESULTS.md §Caveats L54-59](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Preliminary run (superseded; underscore label ids): B0 0.377 [CI 0.360-0.394], B3 0.428-0.436, B1 (Laya-encoder shortlist) 0.256, B2 0.602 at 47.7 ms, B0 ECE 0.506 — [RESULTS.md §Appendix L346-363](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- B4 training takes ~25 min — [RESULTS.md §M0 L6](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

### Inferences
- (Inference, computed from RESULTS.md L12-21) Absolute drop from N=5 to N=77: B0 −0.511, B6 −0.469, B3 −0.444, B4 −0.391, B2 −0.286, B1b −0.261, B5b −0.236. The bi-encoder degrades least of all.
- (Inference) The M0 "Laya" numbers mix two sources: the model card (Banking77 0.425, SST-5 0.372, T4 latency 32.8 ms, ECE 0.466 → 0.081) and project measurements (B0 0.360, SST-5 0.305, 17.6 ms on an RTX 4070 SUPER). GOALS.md quotes the card and RESULTS.md the measurements, so the two should not be mixed in one comparison.

### Gaps
- No confidence intervals are in the M0 table (they are in MLflow `metrics.json` only, [RESULTS.md L55](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)). Differences of ~1-2 points are not testable from the docs.
- Jev was not run locally. 0.870 is the Laya card's number ([DESIGN.md §Baselines L318](/home/ratlinghitman/code/LIDM/docs/DESIGN.md)).

## 3. What did M1, stage 1, the gte runs, stage 1b and the descriptions evaluation show? (comparable tables)

### Takeaway
- **M1** (frozen encoder, M0 mixture, 152 labels): heads memorised labels, Banking77 ~0.03.
- **Stage 1** (974k questions, 2,118 labels): the frozen encoder stayed at chance (0.044). Encoder LoRA reached 0.434 / 0.641. Neither a bge prior nor looping helped.
- **gte-1** (retrieval-trained base): combined score equals retrieval alone, with the head at chance (0.015) because it learned per-label offsets.
- **Stage 1b** (961k distinct labels): the head learned to read. Head alone went 0.015 → 0.456 on Banking77, and on held-out in-domain labels it beats retrieval by about 20 points (0.905 vs ~0.71). It still trails retrieval on intent-style Banking77, and the combined score fell *below* gte alone (0.599 vs 0.634).
- **Descriptions**: help retrieval more than the LIDM head. No LIDM condition beats the best bi-encoder.

### Cited Findings

**M1 (frozen encoder, M0 mixture, 2026-11-15).** Setup: 11,610 train / 1,290 val (random split, 152 distinct option strings), 6 epochs, batch 32. Banking77 = 1,000-item test subsample; chance 0.013 / 0.05 — [RESULTS.md §M1 L63-68](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Run | Params | Best val loss | Val acc | BoolQ | SST-5 | B77 N=77 / N=20 | Wall | Source |
|---|---|---|---|---|---|---|---|---|
| B, LR 1e-4 (best LR) | 32.9M | 0.599 | 0.759 | 0.660 | 0.380 | 0.025 / 0.085 | 9 min | [RESULTS.md L78-84, L92](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| C sampled 1-8 loops | ~34M (estimate) | 0.656 | 0.733-0.742 (L1-L8) | 0.57 | – | 0.026-0.032 / 0.080-0.086 | 13 min | [RESULTS.md L93](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| C fixed 4 loops | ~34M | 0.659 | 0.695 (L1) / 0.748 (L4) | 0.60 | – | 0.018 / 0.085 | 12 min | [RESULTS.md L94](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| D, 8 unshared blocks | 131.6M | 0.634 | 0.747 | 0.58 | – | 0.049 / 0.126 | 19 min | [RESULTS.md L95](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |

- The first run B was broken: raw ModernBERT hidden-state norms grow from ~30 to ~1,200, with single values up to ~27,000, so logits did not depend on the input (BoolQ logit gap 0.510 ± 0.004). Fixed by LayerNorming each layer before the mix, plus a 1e-2 LR for the scalars — [RESULTS.md §M1 L70-76](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [ARCHITECTURE.md §Deviations L68-69](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md)
- M1 findings: labels memorised (B predicted 26 distinct labels over 1,000 items, 42% on one); the cosine prior alone gives 0.08-0.10; CLINC val 0.91 → 0.59 with the Banking77 instruction; looping gave no gain; BoolQ near majority (0.57-0.66) for every head — [RESULTS.md §M1 Findings L97-114](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**Stage 1 (label reading, 2026-11-15/16).** Data: 973,783 questions, 2,118 distinct option strings, Banking77 held out (0 shared labels or texts). Banking77 = all 3,080 test items — [RESULTS.md §Stage 1 L122-127](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md §Stage 1 L33-56](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)

| Run | Encoder / prior | Steps | Wall | Peak VRAM | B77 N=77 | N=20 | Source |
|---|---|---|---|---|---|---|---|
| LR pilots ×4 | frozen / encoder cos | 500 | 7 min each | 3.7 GB | val at chance | – | [RESULTS.md L132](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| Frozen full | frozen / encoder cos | 7,608 | 89 min | 4.0 GB | 0.044 | 0.127 | [RESULTS.md L133](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| bge-prior pilot | frozen / bge cos | 500 | 8 min | 3.9 GB | 0.676 (bge 0.695) | 0.814 (0.830) | [RESULTS.md L134](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| LoRA pilots | LoRA r=16 / encoder cos | 500 / 1,000 | 35 / 63 min | 5.0 / 5.4 GB | 0.029 / 0.072 | 0.100 / 0.156 | [RESULTS.md L135-136](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| **L1: LoRA full** | LoRA r=16 / encoder cos | 7,608 | 6.9 h | 5.5 GB | **0.434** | **0.641** | [RESULTS.md L137](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| C looped, `--init` L1 | LoRA / encoder cos | +3,000 | 2.9 h | 6.6 GB | 0.437 / 0.439 / 0.437 (1/2/4 loops) | 0.645 / 0.644 / 0.641 | [RESULTS.md L138](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| LoRA + bge prior, `--init` L1 | LoRA / bge cos | +3,000 | 2.9 h | 5.5 GB | 0.514 (refit a=160: 0.689) | 0.716 (0.829) | [RESULTS.md L139, L146-155](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |

- Stage-1 diagnostics, frozen vs LoRA: B77 full / head alone / prior alone 0.044 / 0.043 / 0.045 vs **0.434 / 0.420 / 0.172**; distinct labels predicted 40 vs 71 of 77; nlu_evaluation unseen-gold 0.135 vs **0.656**; snips unseen-gold 0.316 vs **0.632**; unseen-gold questions answered with a seen label 69% vs 18%; zero-shot val loss rising 2.38 → 2.64 vs falling 1.96 → 1.37; `val_acc/zs/banking77` 0.02-0.05 vs 0.36-0.40 from step 3.5k — [RESULTS.md §Diagnostics L162-171](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- L1 through the full M0 harness: N=77 0.434, N=40 0.562, N=20 0.650, N=10 0.735, N=5 0.817, BoolQ 0.670, SST-5 0.374, p50 55.4 ms (fresh options, LoRA unmerged), VRAM 1.24 GB. BoolQ and SST-5 are zero-shot for L1 (BoolQ majority = 0.622). AG News 0.891 and Emotion 0.851 are in-distribution through UTCD, unlike the Laya card's 0.950 / 0.595. SST-5 MAE 0.72 levels (B4 0.47, B0 1.06) — [RESULTS.md §Full M0 task set L173-196](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- L1 raw calibration (no temperature): ECE / NLL N=77 0.10 / 2.31, N=40 0.07 / 1.72, N=20 0.06 / 1.25, N=10 0.06 / 0.85, N=5 0.06 / 0.52, BoolQ 0.03 / 0.60, SST-5 0.13 / 1.35 — [RESULTS.md L190-192](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**Latency (L1, `scripts/bench_latency.py`, batch 1, Banking77 N=77, p50 over 100 calls)** — [RESULTS.md L215-225](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Variant | p50 / p95 ms | pass A | pass B (77 options) | head |
|---|---|---|---|---|
| fresh options, LoRA unmerged | 56.0 / 60.6 | 17.3 | 34.6 | 4.6 |
| fresh options, LoRA merged | 35.9 / 40.2 | 8.5 | 23.5 | 4.5 |
| cached options, LoRA merged | **12.6 / 14.7** | 8.6 | 0 | 4.1 |
| *B0 Laya shipped / B5b bge* | *17.6 / 2.8 (p50)* | | | |

- Merging LoRA changes ~4% of predictions but not accuracy (0.438 unmerged vs 0.438 merged on 1,000 items) — [RESULTS.md L223-225](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**Retrieval-trained encoders as plain bi-encoders and gte-1 (2026-11-16)** — [RESULTS.md §Retrieval-trained encoders L241-279](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Encoder (bi-encoder, argmax) | B77 N=77 | N=20 | zs/banking77 | zs/consumer_finance | zs/patent | zs/multi_eurlex | zs mean |
|---|---|---|---|---|---|---|---|
| bge-base-en-v1.5 (110M) | 0.695 | 0.830 | 0.653 | 0.383 | 0.173 | 0.344 | 0.389 |
| gte-modernbert-base (149M) | 0.634 | 0.779 | 0.590 | 0.477 | 0.150 | 0.328 | 0.386 |
| modernbert-embed-base, no prefix | 0.596 | 0.745 | 0.553 | 0.463 | 0.163 | 0.459 | 0.410 |
| modernbert-embed-base, `classification:` prefix | 0.643 | 0.783 | 0.617 | 0.473 | 0.170 | 0.443 | 0.426 |

Source: [RESULTS.md L245-250](/home/ratlinghitman/code/LIDM/docs/RESULTS.md). bge through `Encoder.embed` reproduces B5b exactly ([RESULTS.md L276](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).

- gte-1 (LoRA r=16, pooled option vector, `--prior self`, 7,608 steps, 4.4 h, 3.1 GB, 18.1M trainable): B77 0.628 / 0.775 as trained (s = 26.6), 0.635 / 0.778 refit (s = 160), equal to gte alone. Zero-shot val stayed at gte's level while training loss fell 0.85 → 0.47 — [RESULTS.md L252-256](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); trainable count from [logs/stage1-gte-lora.log](/home/ratlinghitman/code/LIDM/logs/stage1-gte-lora.log)

**Stage 1b (label variety, 2026-11-17/18).** Data: 522,691 questions with 961,447 distinct option strings (stage 1: 2,118). zs/ and Banking77 labels and near-duplicates are banned. New `heldout/*` val sets (1,936 questions) whose answers never appear in training — [RESULTS.md §Stage 1b L285-290](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Run | Steps | Wall | Trainable | B77 N=77 trained / refit s | N=20 trained / refit | Head alone B77 | zs val (combined, trained s) | heldout val | Source |
|---|---|---|---|---|---|---|---|---|---|
| gte-1b (LoRA, pooled, self prior, prior dropout 0.5) | 4,084 | 5.2 h | 18.1M | 0.548 / 0.599 (s 8.3 → 20) | 0.709 / 0.752 | **0.456** | 0.368 | **0.905** | [RESULTS.md L294](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [summary.log L10-18](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log) |
| mbl-1b (ModernBERT-large, L1 settings) | 2,500 of 4,084 (7 h limit, 9.5 s/step) | 7 h | 40.1M | 0.433 / 0.435 (s 12.4 → 160) | 0.618 / 0.616 | 0.423 | 0.337 | 0.925 | [RESULTS.md L295](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [summary.log L19-35](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log) |
| gte alone | – | – | – | 0.634 | 0.779 | – | 0.386 | ~0.71 | [RESULTS.md L296](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |

Trainable counts are from the train logs ([stage1b-gte-lora-train.log](/home/ratlinghitman/code/LIDM/logs/overnight/stage1b-gte-lora-train.log), [stage1b-mbl-lora-train.log](/home/ratlinghitman/code/LIDM/logs/overnight/stage1b-mbl-lora-train.log)). Peak VRAM for stage 1b is not recorded in RESULTS.md.

**Head alone vs retrieval alone vs combined, Banking77 test N=77 (names), all checkpoints**

| Checkpoint | Head alone | Retrieval / prior alone | Combined (trained s) | Combined (refit s on zs val) | Best s on test (diagnostic) | Source |
|---|---|---|---|---|---|---|
| Stage 1 frozen | 0.043 | 0.045 (encoder cos) | 0.044 | – | – | [RESULTS.md L165](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| L1 (MBL LoRA) | 0.420 | 0.172 (encoder cos) | 0.434 | – | – | [RESULTS.md L165](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| L1 + bge prior | not reported | 0.695 (bge) | 0.514 | 0.689 (a=160) | 0.696 (a=320); oracle 0.743 | [RESULTS.md L146-155](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1 | **0.015** | 0.634 (gte) | 0.628 (s 26.6) | 0.635 (s 160) | – | [RESULTS.md L252-262](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1b | **0.456** | 0.634 (gte) | 0.548 (s 8.3) | 0.599 (s 20) | 0.643 (s 80) | [RESULTS.md L294, L304](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [summary.log L16](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log) |
| mbl-1b | 0.423 | 0.349 (encoder cos, LoRA) | 0.432 (s 12.4) | 0.435 (s 160) | 0.446 (s 80) | [summary.log L29-33](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log) |

**Seen-label (training) vs unseen-label behaviour.** Seen-label rows use 1,500 training questions.

| Checkpoint | Seen: head / retrieval / combined | Unseen B77: head top prob | Re-rank retrieval's top 5 (head / retrieval) | Head per-label offset share | Source |
|---|---|---|---|---|---|
| gte-1 | 0.689 / 0.655 / **0.863** | 0.05 | 0.192 / 0.634 (random ~0.17) | 0.49 (retrieval 0.20) | [RESULTS.md L258-271](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1b | 0.894 / 0.675 / 0.893 | 0.507 | 0.524 / 0.634 (gold in top 5: 0.860) | 0.22 | [summary.log L16](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log); [RESULTS.md L298-300](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| mbl-1b | 0.903 / 0.672 / 0.909 | 0.390 | 0.383 / 0.349 (gold in top 5: 0.608) | 0.32 | [summary.log L33](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log) |

**Held-out in-domain labels and zero-shot val sets**
- L1: nlu_evaluation unseen-gold 0.656, snips unseen-gold 0.632 — [RESULTS.md L167-168](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- gte-1b on `heldout/*`: 0.905 combined, 0.903 head alone at the final step (training-log validation); RESULTS.md text says "heldout 0.907 vs 0.708" retrieval, "still +11 when texts are cut to 10 words" — [summary.log L18](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log); [RESULTS.md L298-300](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- gte-1b per zs/ val set, head alone vs gte alone: patent 0.267 vs 0.150 (+12); multi_eurlex 0.328 vs 0.328 ("combined 0.443, +11"); banking77 0.433 vs 0.590 (−16); consumer_finance 0.277 vs 0.477 (−20) — [RESULTS.md L302-304](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- gte-1b s-sweep on zs val: zs mean 0.326 (s=0, head) → **0.403 at s=20** → 0.387 (s=10000, ≈ retrieval). multi_eurlex peaks at 0.525 (s=20); banking77 rises monotonically to 0.603 (s=160-320); consumer_finance rises to 0.480 (s=10000); patent falls from 0.270 (s=2) to 0.150 — [stage1b-gte-lora-fit.log](/home/ratlinghitman/code/LIDM/logs/overnight/stage1b-gte-lora-fit.log)
- patent stays near chance in stage 1 (0.14 vs 0.11); multi_eurlex has only 61 items — [RESULTS.md §Stage 1 Findings 5 L238](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**Banking77 with label descriptions (2026-11-18), all 3,080 test items, N=77 (N=20).** One-sentence descriptions written from the label names only, reviewed for leakage. This is a separate benchmark condition — [RESULTS.md §Banking77 with label descriptions L322-342](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Model | Names only | Descriptions only | `name: description` | Source |
|---|---|---|---|---|
| bge alone | 0.696 (0.830) | 0.718 (0.845) | **0.732 (0.862)** | [RESULTS.md L332](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte alone | 0.634 (0.779) | 0.703 (0.840) | 0.696 (0.827) | [RESULTS.md L333](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1 combined, trained s 26.6 (log only) | 0.628 (0.775) | 0.7065 (0.840) | 0.694 (0.826) | [eval-b77-descriptions.log L10-15](/home/ratlinghitman/code/LIDM/logs/eval-b77-descriptions.log) |
| gte-1b head alone | 0.456 (0.650) | 0.411 (0.629) | 0.463 (0.677) | [RESULTS.md L334](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1b combined, refit s=20 | 0.599 (0.752) | 0.634 (0.794) | 0.646 (0.799) | [RESULTS.md L335](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| mbl-1b head alone | 0.423 (0.609) | 0.374 (0.589) | 0.505 (0.685) | [RESULTS.md L336](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| mbl-1b combined, refit | 0.435 (0.616) | 0.482 (0.667) | 0.558 (0.732) | [RESULTS.md L337](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| L1 head alone | 0.420 (0.629) | 0.451 (0.662) | 0.395 (0.635) | [RESULTS.md L338](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| L1 own encoder-cos prior alone (log only) | 0.172 (0.334) | 0.389 (0.606) | 0.419 (0.636) | [eval-b77-descriptions.log L26-31](/home/ratlinghitman/code/LIDM/logs/eval-b77-descriptions.log) |

- Conclusion in the doc: "Descriptions help retrieval (+2 to +7 points) more than any LIDM head (−5 to +8, inconsistent across formats). In no condition does LIDM beat the best plain bi-encoder ... The head's Banking77 weakness is not the label format" — [RESULTS.md L340-342](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- External context (literature survey, not project measurements): descriptions help readers trained on them and hurt untrained ones (Yin et al.: 48.4 → 26.0 with definitions alone); Hu et al.: BGE-Large 71.53 with descriptions vs 62.91 with names; no paper isolates names vs descriptions for a bi-encoder or late-interaction classifier on Banking77 — [Classification datasets with label descriptions.md §Descriptions pay off L74-80](</home/ratlinghitman/code/LIDM/reports/Classification datasets with label descriptions.md>)

**Training cost summary**

| Run | Data | Steps / wall | Peak VRAM | Source |
|---|---|---|---|---|
| B4 (Laya fine-tune) | M0 mixture 12.9k | 4 epochs / ~25 min | 2.43 GB (inference) | [RESULTS.md L6, L18](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| M1 B / C / D | M0 mixture | 6 epochs / 9 / 12-13 / 19 min | 5.4 / 8.5-8.6 / 9.8 GB | [RESULTS.md L92-95](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md L116-118](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md) |
| Stage 1 frozen | 974k q | 7,608 / 89 min (0.62 s/step) | 4.0 GB | [RESULTS.md L133](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md L121](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md) |
| L1 (LoRA) | 974k q | 7,608 / 6.9 h (3.3 s/step) | 5.5 GB | [RESULTS.md L137](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md L124](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md) |
| gte-1 | 974k q | 7,608 / 4.4 h | 3.1 GB | [RESULTS.md L252-254](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| gte-1b / mbl-1b | 523k q | 4,084 / 5.2 h; 2,500 / 7 h (cut off) | not recorded | [RESULTS.md L294-295](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |

### Inferences
- (Inference) Stage 1b made gte's *head* better but its *combined* Banking77 score worse: 0.635 refit (gte-1) → 0.599 refit (gte-1b), and trained-s 0.628 → 0.548 ([RESULTS.md L254, L294](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)). A head that now has confident, partly wrong opinions on intent labels (top prob 0.05 → 0.507) pulls the sum below retrieval. On zs val as a whole the refit combination nets +1.6 points over retrieval (0.403 vs 0.387), but that weight was selected on those same val sets, so the gain is optimistic.
- (Inference) For mbl-1b, the LoRA-adapted ModernBERT-large's own cosine prior rose from 0.172 (L1) to 0.349 with stage 1b data ([eval-b77-descriptions.log L26](/home/ratlinghitman/code/LIDM/logs/eval-b77-descriptions.log), [summary.log L33](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log)), yet the combined score did not move (0.434 → 0.433). The extra label variety went into the encoder's embedding geometry, not into a better total. mbl-1b was also cut at 61% of its schedule, which confounds this.
- (Inference) The best LIDM head on Banking77 in any condition is 0.505 (mbl-1b, `name: description`), and the best on names is 0.456 (gte-1b). Both sit below the weakest retrieval encoder tested (modernbert-embed-base, no prefix, 0.596).
- (Inference) gte-1's "combined ≈ retrieval" also holds with descriptions (0.7065 vs 0.7032). That number nominally passes 0.70, but only in the descriptions condition and only because the head contributes nothing (head 0.022).

### Gaps
- No stage-1b or gte checkpoint has BoolQ, SST-5, N-sweep, ECE or latency numbers.
- No head-alone number is reported for the L1+bge-prior run, only "head decides" and disagreement counts ([RESULTS.md L148-153](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).
- Minor inconsistencies to flag:
  - gte-1b heldout is given as 0.905 in the table but 0.907 in the text ([RESULTS.md L294 vs L300](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).
  - The "multi_eurlex combined 0.443" in RESULTS.md L303 matches s=40 in the fit log, not the selected s=20 (0.525) or the trained s=8.3 ([stage1b-gte-lora-fit.log](/home/ratlinghitman/code/LIDM/logs/overnight/stage1b-gte-lora-fit.log)). Its provenance is unclear.
  - L1 N=20 is 0.641 in eval_banking77 (all 3,080 items) and 0.650 in the harness N-sweep (1,000 items) ([RESULTS.md L137 vs L185](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)). The protocols differ.
  - The BoolQ majority class is 0.607 in M1 and 0.622 in stage 1 ([RESULTS.md L70 vs L188](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).
  - EXPERIMENTS.md L111 and BACKLOG.md L95-96 still say latency and harness evaluation are unmeasured, but RESULTS.md has both for L1. These docs are stale.

## 4. Where does LIDM beat or lose to Laya, and where does it beat or lose to plain retrieval?

### Takeaway
- **Against Laya:** LIDM (L1) wins on raw calibration, on cached-option latency (12.6 vs 17.6 ms), on SST-5 against *shipped* Laya (0.374 vs 0.305), and at N=77 against shipped Laya (0.434 vs 0.360). It loses at N ≤ 40 to every Laya variant, at N=77 to fine-tuned Laya B4 (0.502) and to Laya's workarounds B1b/B2 (0.610/0.585), on BoolQ to all Laya variants, and on fresh-label latency (35.9 vs 17.6 ms). Its "slower decay" with option count holds against B0 but is equal to B4's.
- **Against plain retrieval:** no LIDM configuration beats bge or gte on Banking77 in any label format. LIDM wins only on (a) seen labels and held-out labels from its training domain (+20 points over gte) and (b) topic-like zero-shot sets (patent, multi_eurlex). Retrieval is also 4-13x faster.

### Cited Findings
- The doc's own reading: "Accuracy falls more slowly with option count than for Laya ... from N=5 to N=77 L1 loses 0.38, Laya B0 0.51, fine-tuned Laya B4 0.39. L1 passes shipped Laya only at N=77; at N <= 40 every Laya variant is ahead (by 5-10 points), and bge is ahead everywhere on Banking77" — [RESULTS.md §Full M0 task set L208-210](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- "Calibration is good out of the box (ECE 0.03-0.10 on six of seven tasks, before any temperature), where shipped Laya had 0.13-0.33" — [RESULTS.md L211-212](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- "SST-5 zero-shot (0.374) beats shipped Laya (0.305); BoolQ (0.670) is above the majority class but below every Laya variant" — [RESULTS.md L213-214](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- "With cached options LIDM is faster than shipped Laya (17.6 ms); with a fresh label set, encoding the 77 options is two thirds of the time" — [RESULTS.md L225](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Retrieval comparison: on unseen-label matching "the trained head adds nothing to bge" (L1+bge); gte-1 "equal to gte alone"; gte-1b best weight on test 0.643, "+0.9 over retrieval (noise)"; "In no condition does LIDM beat the best plain bi-encoder" — [RESULTS.md L154-155, L254-256, L304, L340-341](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Where LIDM beats retrieval: on seen labels the head and retrieval are "strongly complementary (+21 points combined)" (gte-1: 0.863 vs 0.655); gte-1b beats retrieval on unseen in-domain labels by 20 points (0.907 vs 0.708), +11 with texts cut to 10 words; patent +12; multi_eurlex combined +11 — [RESULTS.md L268, L298-303](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- L1+bge: the head is right on 4.8% of questions where bge is wrong (oracle 0.743), "but no fixed weight can use that"; 713 vs 155 disagreements in bge's favour; McNemar z = −18.9 vs bge alone — [RESULTS.md L147-153](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Retrieval is useless on nuance: B5b BoolQ 0.395, SST-5 0.309 — [RESULTS.md §Findings 3 L42](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**L1 vs Laya and bge across axes** (L1 numbers from the harness, same task set as M0)

| Axis | L1 | B0 Laya | B4 Laya FT | B1b / B2 | B5b bge | Winner | Source |
|---|---|---|---|---|---|---|---|
| B77 N=77 | 0.434 | 0.360 | 0.502 | 0.610 / 0.585 | 0.695 | bge; L1 beats only B0, B3 (0.427), B6 (0.402) | [RESULTS.md L12-21, L180-185](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| N=40 | 0.562 | 0.590 | 0.657 | 0.669 / 0.649 | 0.753 | bge, all Laya > L1 | same |
| N=20 | 0.650 | 0.746 | 0.776 | 0.746 / 0.743 | 0.844 | bge | same |
| N=10 | 0.735 | 0.817 | 0.837 | 0.817 | 0.887 | bge | same |
| N=5 | 0.817 | 0.871 | 0.893 | 0.871 | 0.931 | bge | same |
| Drop N=5 → 77 | −0.383 | −0.511 | −0.391 | −0.261 / −0.286 | −0.236 | bge least; L1 ≈ B4 | computed from same |
| BoolQ | 0.670 (zero-shot) | 0.779 | 0.826 (in-dist.) | 0.779 | 0.395 | Laya | same |
| SST-5 | 0.374 (zero-shot) | 0.305 | 0.570 (in-dist.) | 0.305 | 0.309 | B4; L1 > B0, bge | same |
| ECE N=77 (raw) | 0.10 | 0.18 | 0.15 | 0.23 / 0.03 | 0.03 (temp-fitted) | bge, B2 | [RESULTS.md L25-34, L190-192](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| ECE BoolQ / SST-5 | 0.03 / 0.13 | 0.07 / 0.33 | 0.14 / 0.28 | 0.07 / 0.33 | 0.43 / 0.29 | L1 | same |
| p50 latency, fresh | 35.9 ms (merged) | 17.6 | 17.5 | 16.2 / 34.2 | 2.8 | bge, Laya | [RESULTS.md L12-21, L215-221](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| p50 latency, cached options | 12.6 ms | n/a (single sequence) | n/a | n/a | 2.8 | bge; L1 < Laya | same |
| Inference VRAM | 1.24 GB | 2.43 | 2.43 | 2.67 / 2.45 | 0.24 | bge | [RESULTS.md L12-21, L185](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |

### Inferences
- (Inference, computed) L1 minus Laya per N: vs B0 +0.074 (N=77), −0.028 (40), −0.096 (20), −0.082 (10), −0.054 (5). vs B4 −0.068, −0.095, −0.126, −0.102, −0.076. So "5-10 points" in RESULTS.md L210 understates the N=20 gap to B4 (12.6). In relative terms B4 keeps 56% of its N=5 accuracy at N=77 and L1 keeps 53%, so the "slower decay than Laya" claim holds against shipped Laya but not against fine-tuned Laya.
- (Inference) LIDM's demonstrated advantages over Laya (calibration, cached latency) are structural properties of the design, not accuracy wins. Its one accuracy win at N=77 is over the weakest Laya configurations, and Laya's own bge-shortlist workaround (B1b, 0.610 at 16.2 ms) beats it on both accuracy and fresh-label latency.
- (Inference) Against retrieval, LIDM's value is confined to labels like those it trained on. That fits a reranker over a fixed, trained label set better than a zero-shot reader of new intent schemas, which is the GOALS.md use case.

### Gaps
- No like-for-like LIDM vs Laya comparison exists. The joint-encoder control F (same data, LoRA) was never built ([ARCHITECTURE.md §Code layout L98](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md); [EXPERIMENTS.md §Stages L14](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)).
- Calibration for LIDM other than L1, and post-temperature ECE for anyone except B4's clamped fit, are not measured.
- Latency vs option count (N ∈ {5, 20, 77, 255}) and a throughput sweep are not measured ([BACKLOG.md L96](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md)). The "nearly flat with many options" property is shown only at N=77 cached.
- The Laya card's T4 latency (32.8 ms) is not comparable with the 4070 SUPER measurements. DESIGN asked for T4 measurement ([DESIGN.md L320-322](/home/ratlinghitman/code/LIDM/docs/DESIGN.md)).

## 5. Which diagnoses were established, and which explanations were ruled out?

### Takeaway
Three diagnoses explain the results:
1. **Label memorisation** when label variety is low (M1: 152 labels; stage-1 frozen: 2,118).
2. **Explaining-away by a strong retrieval prior.** A head trained next to gte from step 0 learns per-label residuals, not reading, and the prior weight learned on seen labels is wrong for unseen ones.
3. **Label-type/domain coverage.** With 961k labels the head reads, but only label types like its training labels (topic categories and statements about web documents), not customer intents.

Ruled out for the Banking77 gap: reasoning, hard distractors, instruction wording, text length, label format (descriptions) and head capacity/looping.

### Cited Findings
**Established**
- Memorisation (M1): val acc ~0.75 (CLINC 0.91) while Banking77 stays at 0.02-0.05; 26 distinct predictions, 42% on one label; the random val split shares all 152 labels with training — [RESULTS.md §M1 Findings 1 L98-102](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Memorisation persists with a frozen encoder even at 2,118 labels: seen-label val sets 0.7-0.9, unseen at chance; the learned layer mix pulled the cosine prior from 0.116 (step 1,500) down to 0.045; 69% of unseen-gold questions answered with a seen label — [RESULTS.md §Stage 1 Findings 1 L228-230, Diagnostics L169](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- The frozen encoder is the bottleneck for label reading. With LoRA the same head reads unseen labels (nlu_evaluation unseen 0.135 → 0.656) — [RESULTS.md L167, L228-230](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [ARCHITECTURE.md §Deviations L71](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md)
- Explaining-away / per-label residuals (gte-1): head alone 0.015 on Banking77 but 0.689 on seen labels; 49% of the head's score variance is a fixed per-label offset (retrieval 20%). "With a strong retrieval term present from step 0, the generalisable part of the task is already explained, and the cheapest remaining loss reduction is per-label residuals" — [RESULTS.md L258-272](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Mis-weighted prior: "Training learns s on seen labels, where the head beats retrieval, and under-weights retrieval on unseen ones" (trained 7.07 vs fitted 160 for L1+bge; 26.6 vs 160 for gte-1; 8.3 vs 20 for gte-1b; 12.4 vs 160 for mbl-1b) — [ARCHITECTURE.md L59-61](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md); [RESULTS.md L153-155, L254, L294](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [summary.log L11, L28](/home/ratlinghitman/code/LIDM/logs/overnight/summary.log)
- Label variety removes memorisation: gte-1b head 0.015 → 0.456, per-label offset share 0.49 → 0.22 — [RESULTS.md L298-300](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Label-type/domain coverage: the head transfers to patent (+12) and EuroLex topics but not to customer-intent labels ("card arrival", "pending top up") or complaint product types (banking77 −16, consumer_finance −20), "where retrieval's pretraining on query-intent pairs is far broader than our data" — [RESULTS.md L302-304, L317-320](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- LoRA gains need steps: pilots at 500/1,000 steps (0.029/0.072) underestimate the run; the jump came at steps 500-1,000 of the full schedule, flat from ~3,500 — [RESULTS.md §Stage 1 Findings 3 L233-235](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md L72-78](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)
- Engineering diagnosis: raw-residual-norm domination of the head (fixed by per-layer LN) and immobile layer-mix scalars at LR 1e-4 (fixed with 1e-2) — [ARCHITECTURE.md §Deviations L68-69](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md)
- Instruction sensitivity in the frozen M1 model: CLINC 0.91 → 0.59 with the Banking77 instruction; Banking77 0.027 → 0.053 with CLINC's — [RESULTS.md §M1 Findings 3 L105-106](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Laya label-format sensitivity (`card_arrival` vs `card arrival`) — [RESULTS.md §Findings 7 L46](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

**Ruled out (gte-1b on Banking77 test, 3,080 questions)** — [RESULTS.md §Stage 1b "Why the head trails" L306-320](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

| Explanation | Test | Head | Retrieval | Verdict |
|---|---|---|---|---|
| Reasoning | message shares a content word with gold (2,226) / no shared word (854) | 0.514 / 0.304 | 0.722 / 0.405 | Gap largest on plain word matches (−21) and smaller on implicit messages (−10), so not reasoning |
| Hard distractors | N=10 random / 9 most similar labels | 0.759 / 0.569 | 0.850 / 0.669 | Both lose ~19 points, so not distractors |
| Instruction wording | seven wordings incl. training ones and none | 0.456-0.485 | 0.634 | ±3, so not wording |
| Text length | heldout texts cut to 10 words | head keeps +11 to +20 on short in-domain texts | – | Not length |
| Label format | names / descriptions / `name: description` | gap to own retrieval remains (0.646 vs 0.696) | – | "not the label format" ([RESULTS.md L342](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)) |

- Head capacity / depth ruled out: "Bigger heads were not the fix": M1 looping and 8-block heads gave no zero-shot gain, and stage-1 training loss was still falling — [RESULTS.md §Stage 1 Findings 4 L236-237](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); looping from L1 adds nothing (±0.01) — [RESULTS.md L141-144](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- Option-membership blur in head self-attention was deprioritised: "the cosine prior alone, which never passes through the head's self-attention, also fell to 0.045" — [BACKLOG.md §Option membership L63-64](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md)
- More passes over the same labels ruled out: plateau from ~3,500 steps while training loss fell (0.43 → 0.36) — [RESULTS.md L233-235](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- LoRA merging as a cause of accuracy change ruled out (0.438 vs 0.438) — [RESULTS.md L223-224](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

### Inferences
- (Inference) The three diagnoses form a chain. Low label variety leads to memorisation. A strong prior plus moderate variety leads to residual learning (explaining away). High variety gives real reading, but only within the label types covered. The remaining lever the files point to is the *type* of label (intent-style, query-intent pairs) in training data, not more labels of the same kind. The descriptions report says the same: "the decisive variable is ... the style gap between training descriptions and the held-out ones, more than the number of training labels" ([Classification datasets with label descriptions.md §Conclusion L126](</home/ratlinghitman/code/LIDM/reports/Classification datasets with label descriptions.md>)).
- (Inference) The word-match result (head 0.514 vs retrieval 0.722 when a content word is shared) suggests the head is weak at lexical/semantic matching of intent phrases, which is exactly what retrieval pretraining provides.

### Gaps
- The ablations are on gte-1b only. No equivalent word-match/distractor/instruction breakdown exists for L1 or mbl-1b.
- The diagnostics come partly from one-off scripts (M1 prediction-collapse, cosine-only, instruction swaps: "not from a repo script", [RESULTS.md L116-118](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)).
- Leakage nuance: the stage-1 mixture still trains on UTCD's CLINC150 split (only banking/credit-card intents and "exchange rate" removed) and on SGD minus two intents. This does not touch Banking77 labels (0 shared) but means CLINC-like intents are partly seen ([Classification datasets with label descriptions.md L96](</home/ratlinghitman/code/LIDM/reports/Classification datasets with label descriptions.md>); [EXPERIMENTS.md L54-56](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md)).

## 6. Design doc predictions vs what was observed

### Takeaway
DESIGN.md's predictions about looping (little gain on topic/intent labels) and cacheable-option latency were borne out. Its frozen-encoder plan failed outright. LoRA was needed from stage 1, and per-layer LayerNorm was needed just to train. Its calibration prediction (raw over-confidence like Laya's) was wrong in LIDM's favour. The "few points lost on small option sets" prediction was larger in practice (5-13 points vs Laya, not like-for-like). The "better than Laya on many options" prediction holds only against shipped Laya. Every Laya workaround and the fine-tuned Laya still beat LIDM at N=77.

### Cited Findings

| Design prediction | Design source | Observed | Observed source | Verdict |
|---|---|---|---|---|
| Frozen encoder with a 28-layer mix first; LoRA "if it trails"; encoder frozen until stage 2b; "NLI-like parity when frozen" | [DESIGN.md L21, L148, L188, L340](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | Frozen stage 1 at chance on unseen labels (0.044); bge prior did not rescue it (0.676 < 0.695); LoRA r=16 needed from stage 1 (0.434); plan changed | [RESULTS.md L133-137, L228-232](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [EXPERIMENTS.md L72-78](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md); [ARCHITECTURE.md L71](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md) | Refuted for label reading |
| Raw 28-layer mix works with head LR 1e-4 for scalars | [DESIGN.md L163, L286](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | Raw states gave input-independent logits; mix weights stuck at 1/28; needed per-layer LN and scalar LR 1e-2 | [ARCHITECTURE.md L68-69](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md) | Refuted |
| Looping: gains on multi-step rule/policy questions, "little on topic labels" | [DESIGN.md L20, L144, L339](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | No gain on M0 mixture (C 0.656-0.659 val loss vs B 0.599); stage 1 from L1: 0.437/0.439/0.437 at 1/2/4 loops vs 0.434; C fixed 4 vs D undecided (0.659 vs 0.634) | [RESULTS.md L89-95, L107-111, L138-144](/home/ratlinghitman/code/LIDM/docs/RESULTS.md); [BACKLOG.md L73-76](/home/ratlinghitman/code/LIDM/docs/BACKLOG.md) | "Little on topic" confirmed; rule/policy gains untested (stage 2 not built) |
| Split costs "a few points" vs a joint encoder on ≤ 20 options (DeFormer 0.6-1.8 pts; MORES within 2%) | [DESIGN.md L14, L82-88, L337](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | L1 trails B0 by 5.4-9.6 points and B4 by 7.6-12.6 points at N ≤ 20 (different training data; F not built) | computed from [RESULTS.md L180-185](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) | Direction confirmed, size larger; not like-for-like |
| Many options: better than Laya on Banking77-style tasks; "where the split should help most" | [DESIGN.md L88, L338](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | L1 0.434 > B0 0.360, ≈ B3 0.427, < B4 0.502, < B1b 0.610, < B2 0.585; decay 0.38 vs B0 0.51, B4 0.39 | [RESULTS.md L12-21, L208-210](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) | Partly confirmed (vs shipped Laya only) |
| Latency "nearly flat as options grow" with cached pass B | [DESIGN.md L18, L62, L338](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | Cached 12.6 ms (pass B = 0); fresh 35.9 ms, of which pass B 23.5 ms for 77 options | [RESULTS.md L215-225](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) | Confirmed at N=77 cached; N-sweep unmeasured |
| Calibration: "raw over-confidence like Laya's, and a large ECE drop after per-type temperatures" (Laya 0.466 → 0.081) | [DESIGN.md L22, L181, L341](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | L1 raw ECE 0.03-0.10, SST-5 0.13 (shipped Laya 0.13-0.33); temperatures not fitted | [RESULTS.md L190-192, L211-212](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) | Refuted, in LIDM's favour |
| ~35M trainable frozen / ~43M with LoRA; head ~33M | [DESIGN.md L12, L118](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | Head 32.9M; encoder LoRA 7.2M; mbl-1b 40.1M trainable | [ARCHITECTURE.md L29-31](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md); [stage1b-mbl-lora-train.log](/home/ratlinghitman/code/LIDM/logs/overnight/stage1b-mbl-lora-train.log) | Matches |
| Stage 1: ~1.5M questions, ~225M tokens, ~4 h frozen; whole plan ~a day | [DESIGN.md L188, L194](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | 974k questions (synthetic set had 152k usable texts, not 600k); frozen 89 min; LoRA 6.9 h; stage 2/3 not started | [EXPERIMENTS.md L66-69](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md); [RESULTS.md L133, L137](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) | Data short of plan; LoRA slower than the frozen budget |
| Stop stage 1 when zero-shot on UTCD's 9 out-of-domain sets stops improving | [DESIGN.md L227](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | UTCD "out-of-domain" is domain shift, not unseen labels; `val_acc` over all sets is inflated (0.18 vs 0.057); the project switched to `zs/`-tagged sets | [EXPERIMENTS.md L40-50](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md) | Design metric replaced |
| Ablation A-D on stage-2a data after label reading; F measures the split cost | [DESIGN.md L247, L324-333](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | M1 ran B/C/D on the M0 mixture off-plan; A and F not built; stage 2a "data script not written" | [EXPERIMENTS.md L8-21](/home/ratlinghitman/code/LIDM/docs/EXPERIMENTS.md) | Not yet tested |
| DESIGN's own caveat: "These are expectations drawn from cited results on related setups, not measurements of this design" | [DESIGN.md L343](/home/ratlinghitman/code/LIDM/docs/DESIGN.md) | – | – | – |

- DESIGN.md does not treat the retrieval prior as central. M0 finding 3 elevated it: "the `dual_score` retrieval prior and a high-quality option encoder are not optional extras" — [RESULTS.md §Findings 3 L42](/home/ratlinghitman/code/LIDM/docs/RESULTS.md). Later runs showed the prior either carries the score (gte-1) or is mis-weighted by training (all runs) — [ARCHITECTURE.md L59-61](/home/ratlinghitman/code/LIDM/docs/ARCHITECTURE.md)

### Inferences
- (Inference) The design's precedents were cross-encoder-vs-split comparisons (DeFormer, MORES, Poly-encoders), which assume equal training data and equally capable label encoders. The project's real competitor turned out to be retrieval models pretrained on far broader query-intent data. The design never anticipated that a 110M bi-encoder with no task training would set the bar.
- (Inference) Every design prediction that has been *tested* concerns the encoder, the depth and the deployment mechanics. The design's main claim, that a reading head adds what late interaction loses, is untested where it matters (stage 2 rule/policy data, F control) and has failed so far on zero-shot intents.

### Gaps
- Not testable from current files: looping on rule/policy data, the split cost E vs F, the DeBERTa-v3-large backbone control ([DESIGN.md L166](/home/ratlinghitman/code/LIDM/docs/DESIGN.md); not built), temperature-fitted ECE, early exit, T4 latency, latency/throughput vs option count.
- Comparability caveats for all of the above:
  - Every run is single-seed.
  - B4 is fine-tuned on the M0 mixture (BoolQ/SST-5 in-distribution) from a Laya checkpoint that already learned label reading.
  - LIDM stages each use different training data: M1 the M0 mixture, stage 1 974k questions with 2,118 labels, stage 1b 523k questions with 961k labels.
  - bge/gte are pretrained "free" baselines.
  - mbl-1b stopped at 2,500 of 4,084 steps.
  - The M0 table predates the recreated data code (within ~0.03).
  - Banking77 protocols differ: a 1,000-item subsample (M1, harness N-sweep) vs all 3,080 items.
  - "Best s on test" values are test-selected.

  Sources: [RESULTS.md L5, L41-42, L54-59, L102, L185-188, L295](/home/ratlinghitman/code/LIDM/docs/RESULTS.md).
