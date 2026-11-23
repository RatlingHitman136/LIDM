# Backlog: what to try later, and where to focus

Ideas and known issues that were deliberately **not** done yet, with the evidence behind each and candidate
fixes. Ordered by priority within each section. When an item is tried, move the result to
[RESULTS.md](RESULTS.md) and the decision to [ARCHITECTURE.md](ARCHITECTURE.md) / [EXPERIMENTS.md](EXPERIMENTS.md).

## Quality levers after stage 2 (2026-11-19)

Same-data results (RESULTS.md, "Kill-criterion test" and "Stage 2"): LIDM ties Laya on typed decisions (0.769 vs
0.757 after stage 2 + 2c), is better calibrated after a temperature step, loses where options must be checked
against fine detail (VitaminC −7, RACE −5) and with many short options (Banking77 names −11, 13-40 options −9),
wins with many long options (Banking77 descriptions +11), and both lose to bge on label matching. The base encoder
is not the main limit (Laya, same encoder, does better exactly where the late split removes token-level access).
**LIDM-X adopted 2026-11-20** (RESULTS.md: gap to Laya on the rows LIDM lost −8.1 -> +2.7; VitaminC still −6, so LIDM-DJ / state-reads-option is the next architecture test). Architectures against the late split: see `reports/Recovering interaction after late splits.md` (recommends LIDM-X, options attending to pass A's keys/values at every layer, vs LIDM-DJ, joint top 7 layers).

Ordered by expected value for cost:
1. **Soft targets / teacher distillation.** All runs so far use hard labels (`decision_loss` = CE + 0.5 RPS).
   Soft sources already in reach: typed-decisions-synth's DeepSeek teacher distributions, typed-decisions' soft gold,
   Open-Jev / tasksource-jev targets. Loss: KL(target ‖ model) (+ 0.5 RPS for score); Laya's trainer reads `gold`
   distributions natively, so both models can train on the same targets. Expect large Brier / AUROC gains (Laya-TD,
   soft-trained: Brier 0.061 vs ours 0.12-0.15) and +1-2 accuracy. ~small code change + 2.5 h run.
2. **More stage-2 training + ~20% stage-1 replay.** Stage-2 dev accuracy still rising at the end of the single
   pass (0.597 -> 0.655); LIDM forgot label tasks (Banking77 names 0.447 -> 0.434, CLINC150 0.795 -> 0.690). A second
   pass from `checkpoints/s2-lidm/last.pt` with replay; can share a run with item 1. ~2.5 h per pass.
3. **Retrieval prior for label sets.** bge alone 0.62-0.70 on Banking77 vs LIDM 0.43-0.47. Use `--prior bge`
   (or gte) where options are labels, fit the weight with `scripts/fit_prior_weight.py`; test at inference on the
   stage-2 checkpoint first (~30-60 min), retrain with the prior (and `--prior-dropout`) only if it helps.
4. **Targeted data for fine-detail reasoning.** Commercial-clean minimal pairs (one changed fact flips the answer,
   Nimble-style), multiple-choice reading data with near-miss distractors. Current mix has neither (RACE / QuAIL /
   VitaminC are evaluation-only, and RACE / QuAIL are non-commercial).
5. **Looping (C vs B) on the stage-2 mix**, initialised from stage 1 (per-loop LoRA zero-init), evaluated on the
   reasoning tasks (VitaminC, RACE, ANLI, CondaQA). Decide: >= 3 points over the single pass on those, or drop it.
   ~3 h training.
6. **Encoder upgrade, after the late-split fix.** Planned (2026-11-20): retrain everything on a newer encoder once
   LIDM-X is decided. First candidate **UltraBERT-380M** (dmis-lab, Apache-2.0, Oct 2026; 24 layers, 1,024 wide,
   8 x 128 heads, SwiGLU, RMSNorm, local ±128 / global every 3rd layer, 32k context; native in transformers 5.22).
   Self-reported vs ModernBERT-large: GLUE 91.0 vs 90.4, SQuAD 2.0 90.9 vs 89.8, RewardBench 78.8 vs 72.8, RAR-b
   47.0/61.7 vs 34.0/46.1. Port: LoRA targets `q/k/v/o_proj, gate/up/down_proj`; head init from separate projections
   with a SiLU head FFN; LIDM-X loop for its module names (no embedding norm, final `norm`); `[MASK]`-marker works
   (`<|endoftext|> <|mask|> option`). Needs a same-backbone joint control (Laya's trainer is ModernBERT/mmBERT).
   UltraBERT-Embed-150M (MTEB v2 67.24) is the candidate retrieval prior for item 3. Others: DeBERTa-v3-large, Ettin-1B.
7. **Laya's option budget is a setting, not a backbone limit.** Laya gives each option (head_max_len − 16) / N
   tokens; ModernBERT reads 8,192. Our Laya ran at head_max_len 512 (6 tokens per option at N = 77), so LIDM's +11 on
   Banking77 descriptions may be partly that setting. Before claiming it: evaluate Laya at head_max_len ~2,560 /
   max_len 4,096, and for the final comparison train Laya at the larger budget too.

8. **Abstain-option bias (2026-11-20).** LIDM-X over-picks "don't know" / "the text does not say" / "neutral"-type
   options, and the triggering wording flips between runs (CondaQA 0.58 vs 0.23 for the same model with two wordings;
   RESULTS.md). Evaluate abstain options in several wordings (report mean and worst); in training, paraphrase abstain
   options, balance how often they are gold, add contrastive cases where abstaining is wrong, and describe the option
   ("don't know: the passage does not give enough information"). Design alternative: abstain = low calibrated
   confidence, not an option. Tested at inference (RESULTS.md, "Abstain options with descriptions"): descriptions
   shift the abstain rate in model- and wording-dependent directions, not a fix; describing all options is the safest.
   Next: abstain-by-confidence on CondaQA (score only yes/no, threshold on calibrated confidence), and option-to-option
   attention in a few top encoder layers (abstaining is relative to the other options).

Smaller items from the same runs:
- **Temperature step as standard** (stage 3b): one temperature per question type fitted on held-out cases took
  LIDM's ECE 0.135 -> 0.041 and Brier 0.221 -> 0.123 (raw LIDM is overconfident, T = 2-3). Needs a calibration set
  larger than 120 cases: dev-chosen thresholds undershoot their target accuracy on test by 2-4 points.
- **Encode the state once per case**, not once per question (pass A is `state + question`): LIDM is 64 ms vs
  Laya's 44 ms per 5-question case. Options: cache the state's layers and append the question, or put all of a
  case's questions in one pass A.
- **Minimal-pair consistency metric** (CondaQA edit groups, VitaminC pairs): is every version of a pair right?
  A sharper test of understanding than accuracy; ~30 min, no training.
- **Remove CLINC150 from the stage-1 data** (UTCD `clinc_150`): it makes every CLINC150 number for LIDM
  in-domain (stage-1 LIDM 0.795).
- **Selection asymmetry**: LIDM keeps its best pass on dev, Laya's trainer its final checkpoint. Refit Laya with
  3 passes (or save per pass) before reading small gaps as real.
- **Confidence intervals** for calibration and selective-prediction numbers (bootstrap by case), and by source
  for the held-out label sets.

## Focus before stage 2 (superseded 2026-11-19, kept for history)
1. **Banking77 zero-shot.** The primary goal (GOALS.md: >= 0.70 at N=77). Stage 1 with encoder LoRA reached
   0.434 (0.641 at N=20); frozen it was 0.044. Judge on the Banking77 test and `val_acc/zs/*`, never on
   in-distribution val.
2. **Levers for the stage-1 plateau** (flat from ~3,500 steps while training loss falls), all needing `--init`
   so they start from `checkpoints/stage1-Blora/last.pt` instead of a 7 h retrain:
   - **More label variety.** Only 2,118 distinct option strings. Sources: the synthetic set's
     `mixtral_refinedweb_categories` config (free-form category per text), and label_nli turned into choice
     questions (group its hypotheses per task, so labels go through pass B).
   - **LoRA + bge prior.** bge alone gets 0.695. With a frozen encoder the head could not add to it (pilot:
     0.676, worse); with LoRA the head now learns, so this is worth retrying from the LoRA checkpoint.
   - **Option representation**: marker token vs mean-pooled option tokens (the marker's pretraining meaning is
     "guess the missing first word"; see the [MASK] note below).
3. **Nuance (BoolQ, SST-5).** Frozen heads were at the majority class in M1. Measure the LoRA checkpoint with
   `scripts/eval_lidm.py`, then decide on stage 2 vs the joint-encoder control F.

### Marker token vs pooled option (built: `--option-vector pooled`; not yet compared on the same encoder)
Pass B scores each option at a `[MASK]` placed before its text. ModernBERT was pretrained to make a `[MASK]`
output predict the missing word, so a frozen marker is the model's guess of a word preceding "card arrival",
not a summary of the label. LoRA can retrain this, but a mean-pooled option vector (already computed for the
prior) is a cheap A/B: score `w · LN(mean of the option's head outputs)` instead of the marker.

## Head design
### Option membership in the head's self-attention (open, not tried)
**Issue.** The head's self-attention runs over all option tokens of a question with no signal of which option
a token belongs to. Positions would not help: every option starts at 0, so token 3 of option A and token 3 of
option B look alike. Each marker keeps its own option's encoding through the residual and input injection,
and pass B already let it read its option in the encoder, so the damage may be small, but when options
share words ("card arrival", "card delivery estimate", "card not working") the head can blur them.
Decided 2026-11-15: run stage 1 without a fix.

Candidate fixes, all keeping scores order-free (no positions across options):

| Fix | Change | Notes |
|---|---|---|
| a. Same-option attention bias | learned per-head scalar added to the attention logit when query and key belong to the same option; init 0 | ~15 lines; the option id per packed token is already computed in `LIDM.forward` (`cols // to`); starts exactly at the current model |
| b. RoPE within each option in the head | rotary positions restarting at 0 per option | restores in-option order and makes the encoder-initialised attention weights consistent (they were trained with RoPE); gives **no** membership, see below |
| b'. RoPE + same-option bias | b for order, a for membership | the only variant that gives both and stays exactly order-free (~35 lines) |
| c. Two-level attention | full attention within each option; across options only among marker tokens | clean separation, larger change to the design |

Why RoPE alone cannot encode membership (discussed 2026-11-15): RoPE makes the attention score depend only
on the distance between positions. With positions restarting per option, token k of option A and token k
of option B are at distance 0, like a token and itself; every marker (position 1) sees every other marker at
distance 0; same-option and cross-option pairs cover the same distances. It may even blur more at init,
since the copied encoder weights treat distance-0 tokens as "the same place". Giving each option a large
position offset (option k at k x 1000) does encode membership by distance, but reintroduces option order
(options 1 and 2 closer than 1 and 3), breaking permutation equivariance; random offsets per forward restore
it only in expectation and make scores noisy. RoPE also has no strong distance decay, so the signal is weak.

| Approach | Order inside an option | Membership | Order-free |
|---|---|---|---|
| RoPE restarting per option (b) | yes | no | yes |
| RoPE with per-option offsets | yes | weak | **no** |
| Same-option bias (a) | no | yes | yes |
| RoPE + same-option bias (b') | yes | yes | yes |

Priority: after label reading works (bge prior or LoRA). Stage 1 showed the bottleneck is elsewhere: the
cosine prior alone, which never passes through the head's self-attention, also fell to 0.045 on Banking77.

How to test: A/B in a stage-1 pilot (~500 steps, ~6 min each), same LR, compare `val_acc_zs` and
`val_acc/zs/banking77`.

### Other head ideas
- **No positions across options** is intentional (order-free scores, tested). For `score` questions the level
  order is already in the label text and the RPS loss; an index embedding there would mostly allow a
  "pick the middle" shortcut. Revisit only if stage-2 `score` accuracy is weak.
- **Looping**: retest C (sampled 1-8 and fixed 4) vs D on stage-2 rule/policy data, initialised from the
  stage-1 B checkpoint. No gain on the M0 mixture (M1) nor in stage 1 with encoder LoRA (3,000 steps from L1:
  Banking77 0.437 / 0.439 / 0.437 at 1 / 2 / 4 loops vs 0.434). Label matching does not use extra depth; the
  design doc expects gains only on multi-step rule questions.
- **Early exit** for looped heads (stop when successive option distributions agree): not built.

## Training infrastructure
- ~~**`--init <checkpoint>` in `cls.train`**~~: built (head + encoder LoRA; B -> C with zero-init per-loop LoRA).
- **Soft targets** in `decision_loss` (KL to a distribution): still not built; see "Quality levers" item 1.
- **Fixed layer mix after the pilot** (the design doc's plan): frees the ~1 GB of saved per-layer fp32 states
  per pass and makes both passes cacheable (~2 KB per token).
- **Option cache**: all 2,118 stage-1 options at 28 layers is ~2 GB if stored unpadded (mean 16 tokens); would
  remove the per-micro-batch pass-B encoder call. Becomes much cheaper with a fixed mix.
- **Unpadded / flash encoder passes**: the encoder still runs on padded batches (pass A was ~29% real tokens
  before length sorting).
- **Head gradient checkpointing, `torch.compile`**: more memory headroom (larger micro-batches, longer
  states) and ~10% throughput; not needed at the current 3.5 GB peak.
- **Pass-A length 512** truncates 9.8% of label_nli states (~3% of the mix); try 1024 once batches are
  unpadded.

## Evaluation gaps
- ~~LIDM through `cls.eval.harness`~~ done (M0 eval, context suite, label suite, typed decisions).
- ~~Latency benchmark~~ done (`scripts/bench_latency.py`; per-case latency in the typed-decisions eval).
- Temperature fitting (stage 3b) and ECE per (type, option-count bucket).
- More than one seed for any result that decides a design question.
