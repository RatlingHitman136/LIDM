# Architecture

The initial design, with its evidence and citations, is [DESIGN.md](DESIGN.md) (snapshot of the 2026-11-09
design doc); the data research behind it is [DATASETS.md](DATASETS.md). This file is the short version of
what is **built** and where it departs from that design. Results: [RESULTS.md](RESULTS.md).

## Baseline: Laya (single sequence)
`[state][question][opt1 [MASK]]...[optN [MASK]]` -> bidirectional encoder -> 2-layer decision head
(~26M) -> option-marker scorer at each `[MASK]` -> softmax over options. All options share a ~192-token
head budget, which is what collapses accuracy above ~20 options (RESULTS.md, M0 finding 1).

## LIDM: two-pass encoder + looped fusion head

```
 state + all question texts ──► [ModernBERT-large, pass A] ──► 28 layers ─► LN each, softmax mix ─┐
                                                                                                  ├─► fusion head x loops ─► marker logits ─► softmax per question
 [MASK] option_1..N          ──► [ModernBERT-large, pass B] ──► 28 layers ─► LN each, softmax mix ─┘   (+ dual_score prior)
     one sequence per distinct option; encoded per batch in training, cacheable at inference
```

1. **Pass A, per request** (`Encoder.encode_a`): `[CLS] state [SEP] question [SEP]`, state truncated to 512
   tokens. The question stays with the state (required for `noul`).
2. **Pass B** (`Encoder.encode_options`): `[CLS] [MASK] option [SEP]`, one sequence per *distinct* option
   string, truncated at 64 tokens (`--max-opt-len`); the `[MASK]` output anchors that option's score.
   Encoded per micro-batch during training (only that batch's options). At inference the outputs for a
   fixed schema can be cached. The encoder itself still runs on padded batches (no unpadded/flash path).
3. **Encoder**: same frozen weights for both passes. Each pass learns a softmax mix of all 28 layers, and
   **each layer is LayerNormed (no affine) before mixing** (see "Deviations"). Optional encoder LoRA
   (`--encoder-lora 16`): r=16, alpha 32, dropout 0.05 on Wqkv, Wo, Wi and the MLP Wo of all 28 layers (7.2M
   params, fp32), shared by both passes, gradient checkpointing; saved in checkpoints as `encoder_lora`.
4. **Fusion block** (`cls.head.FusionBlock`, MORES / Q-Former pattern), 2 blocks, 32.9M trainable params
   in total for run B. Its input is each question's real option tokens packed together (no per-option or
   per-question padding inside the question; questions padded only to the longest one): self-attention among all
   option tokens of one question, cross-attention from option
   tokens to pass-A tokens, GeGLU FFN, sandwich norm (pre + post LayerNorm per sublayer). No positions across
   options, so scores are order-free (tested). Initialised from the encoder's top layers (self-attn weights
   copied into both attentions).
5. **Looping** (`FusionHead`): loop the blocks with input injection (re-add the option encodings each
   loop), shared attention + per-loop rank-16 FFN LoRA (B starts at zero), loop count sampled 1..max per step
   (`--fixed-loops` to always use max), truncated backprop through the last 4 loops. Early exit not built.
6. **`dual_score` retrieval prior**: logit += learned scale x cos(mean-pooled mixed pass A, mean-pooled
   mixed option).
7. **Typed outputs**: `choice` softmax over options; `noul` as the option pair ("no", "yes"); `score`
   softmax over ordered levels.
8. **Loss**: cross-entropy (= KL to a one-hot target) + 0.5 x RPS for `score` (`decision_loss`).
   Temperature fitting (per question type and option-count bucket) not built yet.

## Deviations from DESIGN.md, and why
| Change | Why (evidence) |
|---|---|
| LayerNorm each encoder layer before the softmax mix | ModernBERT's raw residual stream grows from token norm ~30 (layer 1) to ~1,200 (layers 20-27), with single dimensions up to ~27,000. Mixed raw, those dominated cross-attention keys and the residual stream; the first run B gave input-independent logits (BoolQ logit gap 0.510 +- 0.004 over 64 passages) and stayed at chance. RESULTS.md, M1. |
| Layer-mix weights and dual scale at LR 1e-2 (`--scalar-lr`), not the design's 1e-4 | Adam moves a parameter by ~LR per step; at 1e-4 the 28 mix weights stayed exactly uniform (0.036) and the dual scale moved 1.000 -> 1.004 over 2,178 steps. At 1e-2 the mix moves to layers 22-26. |
| Options encoded per micro-batch, not cached for the whole run | Caching all 28 layers costs ~57 KB per token (as DESIGN.md notes); fine for the M0 mixture's 152 labels; ~5 GB for stage 1's 2,118 option strings (up to 64 tokens), and it grows with the label set. |
| Encoder LoRA r=16 from stage 1, not only from stage 2b | Frozen, every head memorised labels and stayed at chance on unseen ones (Banking77 test 0.044 after full stage 1; also with a bge prior). With LoRA the same run reaches 0.434 at N=77 and reads unseen labels (RESULTS.md, stage 1). |
| Stage order: the A-D grid moves from the M0 mixture to stage 2a | On the M0 mixture (152 distinct labels) every head memorised the labels: in-distribution val ~0.75, Banking77 zero-shot ~0.03. DESIGN.md already put the grid in stage 2a, after label reading. EXPERIMENTS.md. |
| Head input packed per question; each optimizer step's questions sorted by length before micro-batching (`length_sorted`) | Stage-1 batches were ~89% padding in the head and ~71% in pass A. Same logits (tested against the padded path), same gradient per step; 1.42 -> 0.62 s/step and 5.8 -> 3.5 GB at micro-batch 16. The design doc asked for length bucketing. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set by `cls.train` | Every micro-batch has a different shape; the allocator fragmented and a stage-1 run died with "CUDA error: out of memory" at ~9 GB reserved. |

## Known risks
Open issues and deferred fixes (e.g. option membership in the head) are tracked in [BACKLOG.md](BACKLOG.md).
- Text-option interaction happens only in the head, so nuance (negation, conditions) may suffer. In M1 BoolQ
  stayed near the majority class (0.57-0.66) for every head, looped or not; LoRA (E) and the joint-encoder
  control (F) are the planned answers.
- Frozen ModernBERT features carry little retrieval signal (cosine alone 0.08-0.10 on Banking77; bge 0.695)
  and a frozen encoder cannot read unseen labels at all; with LoRA it can (0.434), but stays below bge alone.
- Head memory: self-attention runs over all option tokens of a question (up to 24 options x up to 63
  tokens; mean option 16 tokens). Before packing, only ~11% of the head's option slots were real tokens and a
  32-question micro-batch peaked at ~10 GB; packed and length-sorted it is 5.6 GB (16 questions: 3.5 GB).

## Code layout
```
src/cls/model.py        Encoder (frozen, two passes), LIDM (layer mix, head, scorer, dual_score), decision_loss
src/cls/head.py         FusionBlock / FusionHead (looping, per-loop LoRA, truncated backprop)
src/cls/train.py        training: M1 grid (--run B|C|D) and stage 1 (--stage1); MLflow + checkpoints
src/cls/data/typed.py   item schema, M0 mixture, eval sets (Banking77, BoolQ, SST-5)
src/cls/data/stage1.py  stage-1 label-reading mixture + zero-shot val (UTCD out-of-domain sets)
scripts/prepare_data.py, scripts/prepare_stage1.py, scripts/eval_banking77.py
tests/                  order invariance, padding, packed vs padded head, loops, layer tuples, loss, UTCD + zs tagging
```
Not built yet: run A head, joint-encoder control (F), temperature fitting, early exit,
unpadded encoder passes, latency benchmark. The untested head sketch in HANDOFF.md predates the design.
