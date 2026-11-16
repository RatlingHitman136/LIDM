# Backlog: what to try later, and where to focus

Ideas and known issues that were deliberately **not** done yet, with the evidence behind each and candidate
fixes. Ordered by priority within each section. When an item is tried, move the result to
[RESULTS.md](RESULTS.md) and the decision to [ARCHITECTURE.md](ARCHITECTURE.md) / [EXPERIMENTS.md](EXPERIMENTS.md).

## Focus (decides whether the project works)
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

### Marker token vs pooled option (open, not tried)
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
  stage-1 B checkpoint. No gain on the M0 mixture.
- **Early exit** for looped heads (stop when successive option distributions agree): not built.

## Training infrastructure
- **`--init <checkpoint>` in `cls.train`**: needed for stage 2a and for starting looped C from B. Every
  run currently starts a fresh head.
- **Soft targets** in `decision_loss` (KL to a distribution): needed for stage 2 (jev-distill) and 3a.
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
- LIDM through `cls.eval.harness` (BoolQ, SST-5, AG News, Emotion, N-sweep, calibration), not only Banking77.
- Latency benchmark (fresh vs cached options, N in {5, 20, 77, 255}).
- Temperature fitting (stage 3b) and ECE per (type, option-count bucket).
- More than one seed for any result that decides a design question.
