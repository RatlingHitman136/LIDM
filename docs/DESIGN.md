# Design: two-pass Laya-style classifier with a looped fusion head

> **Snapshot.** Copy of the design doc "Design: two-pass Laya-style classifier with a looped fusion head"
> (claude.ai doc, written 2026-11-09, copied 2026-11-15) so the repo does not depend on it. It records the
> *initial* design thinking and is not updated. What was built and what changed since — per-layer LayerNorm
> before the layer mix, a 1e-2 LR for the mix scalars, options encoded per batch, the M1 results and the
> revised stage order — lives in [ARCHITECTURE.md](ARCHITECTURE.md), [EXPERIMENTS.md](EXPERIMENTS.md) and
> [RESULTS.md](RESULTS.md). The dataset survey it cites is [DATASETS.md](DATASETS.md).

2026-11-09 · Borys

Build it as a split encoder with a MORES-style head. ModernBERT-large encodes the state and its questions in one pass and every option in a second, packed pass; two cross- plus self-attention blocks, looped 4–8 times, then score the options. That is about 35M trainable parameters with the encoder frozen, or 43M with encoder LoRA.

Each piece has a published precedent, and together they predict a trade. Expect a few points below a joint encoder like Laya on small option sets, better scaling to many options, and option encodings computed once and cached.

| Decision | Recommendation | Main evidence | Expected effect |
|---|---|---|---|
| Encoding | Pass A: state plus all question texts. Pass B: options packed as separate sequences, cached per schema | Poly-encoders, DeFormer, GLiNER bi-encoder | Small loss vs. a joint encoder; latency nearly flat as options grow |
| Head | 2 blocks: option self-attention within a question, option-to-state cross-attention, GeGLU FFN; initialised from the top encoder layers; no option positions | MORES, BLIP-2 Q-Former, Set Transformer | Recovers most joint-encoder interaction; order-free scores |
| Depth | Loop the head 4–8× with input injection, a sampled loop count, truncated backprop, shared attention and per-loop FFN LoRA | Saunshi et al., Huginn, ALBERT, Relaxed Recursive Transformers | Gains on multi-step rule questions; little on topic labels |
| Encoder | Frozen with a 28-layer mix first; LoRA r=16 (~7.2M parameters) if it trails | Peters et al., LoRA, autotrust/JEV | NLI-like parity when frozen; LoRA closes paraphrase-type gaps |
| Training | Label reading, then typed decisions, then calibration; KL and RPS on soft targets; per-type temperatures | Laya card, autotrust/JEV | ECE drop like Laya's 0.466 → 0.081 |

## Architecture

Options are encoded once; they meet the state only in the looped head.

```
 Pass A · per request                       Pass B · per schema
 state + every question text of the call    every option as its own sequence, led by a marker token;
 in one sequence                            encoded once and cached, reused across requests
            │                                          │
            ▼                                          ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────┐
 │ ModernBERT-large: the same weights run both passes                                       │
 │ frozen with a learned mix of all 28 layers, or LoRA r=16 on attention + MLP (~7.2M)      │
 │ options packed into one unpadded batch, so they never attend to each other              │
 └──────────────────────────────────────────────────────────────────────────────────────────┘
            │ state + questions                        │ option tokens
            ▼                                          ▼
 ┌──────────────────────────┐        ┌──────────────────────────────────────────────┐
 │ State memory             │        │ Looped fusion head, L = 4–8 loops            │
 │ pass-A tokens, never     │        │  ┌ self-attention among options ───────────┐ │
 │ updated; each loop reads │        │  │ (options of one question; no positions)  │ │
 │ them through cross-attn  │ ─K,V─► │  ├ cross-attention to pass A ──────────────┤ │◄─┐ loop
 │ (keys, values)           │        │  │ (the state and its own question)         │ │  │
 └──────────────────────────┘        │  └ GeGLU FFN (shared + per-loop LoRA) ─────┘ │──┘
                                     │ 2 blocks of these three layers, ~33M params  │
                                     │ each loop re-adds the option tokens          │
                                     │ L sampled per batch; early exit at inference │
                                     └──────────────────────────────────────────────┘
                                                        │
                                                        ▼
 Score each option at its marker token, then normalise within its question
   choice: softmax across its options (KL)  ·  noul: two options or one sigmoid (KL / log loss)
   score: softmax over ordered levels (KL + RPS)
                                                        │
                                                        ▼
 Temperature per question type and option count (held-out split; changes confidence, never the winner)
```

At request time only pass A and the head run, since option encodings come from the cache; options see the state only through the head's cross-attention.

## Two encoder passes

Split the input the way bi-encoders do: pass A encodes the state with its question text, pass B encodes every option, and the head adds back the interaction a joint encoder would have had. The split costs some accuracy and buys caching and many-option scaling; both effects are well measured.

### What goes in each pass

- **Pass A, per request:** the state plus every question text of the call in one sequence, so state tokens are encoded knowing what is asked. This mirrors Laya, which answers all questions of a call in one forward pass.
- **Pass B, per schema:** every option of every question, each option as its own sequence in one packed batch. ModernBERT already concatenates sequences into one unpadded batch and runs Flash Attention's variable-length kernels with per-sequence RoPE, so options don't attend to each other and each starts at position 0.
- **Marker token:** prefix each option with a marker whose output anchors that option's score, as Laya scores each option at its own [MASK] token.
- **Cache pass B.** Options are fixed per schema, so at inference their encodings can be stored and reused, the main speed lever of bi-encoders. During LoRA training the encoder changes every step, so group batches by schema and encode each option set once per batch instead. With a frozen encoder, precomputing both passes is practical only with a fixed layer mix (about 2 KB per token); keeping all 28 layers for a learned mix costs about 57 KB per token.

Variant to test: move the question text into pass B with its options. The whole schema then becomes cacheable and pass A is the bare state, but the state is no longer encoded with the question in view, so the head must do all the question-conditioning.

### What the split costs and buys

| Evidence | Setup | Result |
|---|---|---|
| [Poly-encoders](https://arxiv.org/abs/1905.01969) | Cross-encoder vs. bi-encoder vs. poly-encoder (learned attention over separately encoded candidates) | Cross-encoders score best but are too slow; poly-encoders are faster than cross-encoders and more accurate than bi-encoders |
| [DeFormer](https://arxiv.org/pdf/2005.00697) | Question and passage encoded separately in lower layers, jointly in upper layers | 2.7–4.3× faster, 65.8–72.9% less memory, 0.6–1.8 points lost; about 1% with distillation |
| [MORES](https://arxiv.org/pdf/2004.13313) | Query and document encoded separately, then interaction blocks (cross-attention, then self-attention) | 2 interaction blocks within 2% of a BERT cross-encoder ranker; up to 120× faster |
| [GLiNER bi-encoder](https://arxiv.org/html/2602.18487) | Separate text and label encoders, cached label embeddings | Up to 130× throughput at 1,024 labels; 5.2% slowdown from 1 to 1,024 labels vs. 98.7% for the joint model; 61.5% zero-shot CrossNER F1 |
| [GLiClass](https://www.arxiv.org/pdf/2508.07662) | Joint encoder with labels placed in the text | Text representations degrade when labels greatly outnumber text tokens; label-to-label attention weakens as label count grows |
| [Laya](https://huggingface.co/convaiinnovations/laya) | Joint encoder, options at [MASK] markers | Ahead of Jev on typed-decisions, AG News and DAIR Emotion; Jev leads above 20 options at default settings (Banking77, 77 labels: 0.870 vs. 0.425) |

Expect the split alone to lose a few points to a joint encoder on small option sets; the fusion head is there to win them back. Many-option tasks like Banking77 are where the split should help most.

## Fusion head

Use stacked interaction blocks in which option tokens cross-attend to the pass-A tokens and self-attend among the options of their own question. This is the MORES and BLIP-2 Q-Former pattern; it costs far less than full self-attention over the concatenation when the state is long, and it keeps option scoring independent of option order.

### The two candidates

|  | Self-attention over the concatenation | Cross- plus self-attention blocks |
|---|---|---|
| What attends to what | Every pass-A and pass-B token attends to every other | Option tokens attend to pass-A tokens (cross) and to options of the same question (self); pass-A tokens stay read-only |
| Cost per head layer | Grows with (state + option tokens)² | Grows with option tokens × state tokens, plus option tokens² |
| Precedent | DeFormer's joint upper layers | MORES interaction blocks; Q-Former |
| Option-order effects | Needs positional handling across the concatenation | None, if the head adds no positions across options |
| Fits best | Short states, few options | Long states, many options, and looping (cost multiplies by loop count) |

### Recommended block

1. Self-attention among option tokens, masked to options of the same question, so options compare against each other.
2. Cross-attention from option tokens to pass-A tokens, masked to the state plus that option's own question text.
3. A GeGLU feed-forward layer, matching ModernBERT's.
4. After the last block, a linear score on each option's marker token, softmaxed across that question's options.

### Practices from the precedents

- **Initialise from the encoder's top layers.** MORES copied BERT self-attention weights into its cross-attention and found this significantly better than random initialisation. Q-Former starts from BERT-base with randomly initialised cross-attention.
- **Interleave cross- and self-attention.** Q-Former inserts cross-attention every other block over a frozen image encoder; MORES runs cross-attention then self-attention inside each block.
- **Keep option index out of the head.** Self-attention without positional encodings is permutation-equivariant ([Set Transformer](https://arxiv.org/abs/1810.00825)), so shuffling options cannot change their scores. autotrust/JEV, built on a positional model, instead shuffles options in 30% of training examples.
- **Optional two-way variant:** let pass-A tokens cross-attend back to the options in alternate blocks. It roughly doubles head cost, so run it as an ablation rather than the default.

Two blocks at ModernBERT-large width (1,024) come to about 33M parameters, close to the ~26M of Laya's 2-layer head (421M total on a 395M backbone).

## Looped depth

Loop a small shared block several times instead of stacking new ones. Published results say this recovers most of the benefit of depth on reasoning-like problems at constant parameter count, but not on knowledge-heavy ones, and that it needs specific tricks to train stably.

### What looping buys and costs

| Evidence | Finding |
|---|---|
| [Saunshi et al., ICLR 2025](https://arxiv.org/abs/2502.17416) | A k-layer block looped L times nearly matches a kL-layer model on addition, p-hop induction and math, and clearly beats the k-layer model. In language modelling it has worse perplexity but competitive or better downstream reasoning. |
| [ALBERT](https://arxiv.org/pdf/1909.11942) | Sharing all layers costs 1.5–2.5 points on the GLUE/SQuAD/RACE average (79.8 vs. 82.3 at E=768). Sharing attention alone is nearly free (+0.1 and −0.7); most of the loss comes from shared feed-forward layers. |
| [Relaxed Recursive Transformers, ICLR 2025](https://arxiv.org/abs/2410.20672) | Loop-specific LoRA on tied layers recovers most of the full model's quality, improving with LoRA rank; early exit enables up to 2.76× throughput. |
| [Huginn recurrent depth](https://arxiv.org/abs/2502.05171) | A looped block trained at billions of parameters with input injection, a random number of loops per step and backpropagation through only the last 8 loops; it supports early exit when successive loops stop changing the output. |

### Recipe for the head

- **Loop count:** loop the interaction blocks 4–8 times. Check 1, 2, 4 and 8 against an iso-parameter control (1 loop) and an iso-compute control (unshared blocks at the same depth), the comparison Saunshi et al. use.
- **Input injection:** re-add the original option encodings at every loop. Huginn's authors call this crucial for stability.
- **Loop identity:** share attention across loops and give each loop its own rank 8–16 LoRA on the feed-forward layer, since ALBERT locates the cost of tying in the FFN. A loop-index embedding, as the [Universal Transformer](https://arxiv.org/abs/1807.03819) adds at each step, is the cheaper alternative.
- **Training depth:** sample the loop count per batch and backpropagate through only the last few loops, as Huginn does, so the model works at several depths and memory stays flat.
- **Normalisation:** use pre- plus post-normalisation inside the looped block, the sandwich norm Huginn adopted for stability; [Parcae](https://arxiv.org/html/2604.12946) traces looped-model instability to large input-injection weights.
- **Early exit:** at inference, stop looping once the option distribution stops changing (KL between successive loops below a threshold), the exit rule Huginn exposes.

Per-loop FFN LoRA at rank 16 adds about 0.16M parameters per block per loop, so 2 blocks × 8 loops stay near 2.5M.

Looping should help questions that chain conditions, such as policy checks, rule application and "if X and not Y" decisions, and do little for topic or intent labels, which depend on what the encoder already knows. Saunshi et al. describe exactly this split between reasoning and memorisation.

## Encoder: frozen or LoRA

Start with ModernBERT-large frozen and a learned weighted mix of all 28 layers: the encoder then runs forward only, roughly halving the cost of a training step. Move to LoRA if the frozen run trails on paraphrase-like or fine-grained tasks: frozen features with a strong head have matched fine-tuning on NLI but not everywhere, and LoRA has matched full fine-tuning on GLUE.

| Evidence | Finding |
|---|---|
| [Peters et al., 2019](https://arxiv.org/pdf/1903.05987) | Frozen BERT-base features (weighted mix of all layers) into task heads, ESIM cross-attention for sentence pairs, vs. full fine-tuning: MNLI 84.6 vs. 84.6 and SST-2 93.0 vs. 93.5, but MRPC 78.1 vs. 84.8 and STS-B 82.9 vs. 87.1 |
| [LoRA](https://arxiv.org/pdf/2106.09685) | GLUE average 87.2 for RoBERTa-base with 0.8M trainable parameters vs. 86.4 for full fine-tuning; 91.3 vs. 91.1 for DeBERTa XXL with 4.7M |
| [autotrust/JEV](https://huggingface.co/autotrust/JEV) | Frozen Qwen3.5-9B, rank-16 LoRA and a 24-slot head (~40M trainable, ~0.5%) reach KL 0.019 from Jev's distributions in about 3 B200 GPU-hours |
| [GLiClass](https://www.arxiv.org/pdf/2508.07662) | Post-trained with LoRA at ranks 384–1,536; its DeBERTa-v3 models beat its ModernBERT ones (0.719 vs. 0.620 average F1 for large, not size-matched) |

### Backbone facts that shape the design

[ModernBERT-large](https://arxiv.org/abs/2412.13663) has 395M parameters, 28 layers, hidden size 1,024 and 16 heads. Every third layer uses global attention and the rest a 128-token sliding window, context runs to 8,192 tokens, and unpadded packed batches are native. It scores 90.4 on GLUE, second only to DeBERTa-v3-large.

### Settings to start from

- **Frozen:** learn a softmax-weighted mix of all 28 layers for each pass, as Peters et al. did, and run the encoder forward each step instead of storing features. All 28 layers come to about 57 KB per token, roughly 29 TB for 0.5B tokens; to cache, fix the mix after a short pilot and store about 2 KB per token.
- **LoRA:** rank 16 on the attention (Wqkv, Wo) and MLP (Wi, Wo) projections of all 28 layers, about 7.2M trainable parameters. The LoRA paper found ranks as low as 1 enough when adapting Wq and Wv together, so rank 16 on all projections leaves headroom.
- **Shared weights:** the same LoRA weights serve both passes, keeping the one-encoder, two-runs property.
- **Backbone control:** keep one DeBERTa-v3-large run as a control because of GLiClass's result, accepting that it lacks ModernBERT's 8,192-token context and native unpadding.

## Outputs, losses and calibration

Score every option at its marker token and normalise within each question, train against soft targets with proper scoring rules, then fit temperatures per question type. Laya and autotrust/JEV both document this recipe.

| Question type | Head output | Training loss | Precedent |
|---|---|---|---|
| choice | One logit per option, softmax across that question's options | KL to the target distribution; cross-entropy for hard labels | Laya scores options at [MASK] markers; autotrust/JEV trains on KL to Jev's distributions |
| noul | Two options (false, true), or one logit through a sigmoid | KL or binary log loss | Jev and Laya both return P(true) |
| score | Softmax over ordered levels; expected level = Σ level × p | KL plus ranked probability score (RPS) | Laya's reward adds RPS for ordinal questions; autotrust/JEV adds an RPS term for score |
| act / escalate (optional) | Binary gate on the decision | Log loss on whether the top option was right | Laya ships an act/escalate head |

### Calibration

- Laya's checkpoints come out over-confident: one temperature per (question type, option count) cuts mean ECE from 0.466 to 0.081 for the English checkpoint. Do the same on a held-out split, such as jev-distill-corpus-v3's 13,766-row calibration split.
- [Temperature scaling](https://arxiv.org/abs/1706.04599) divides logits by one constant, so it never changes which option wins; accuracy stays put while ECE drops.
- Laya's RLCD (Gaussian noise on logits, a proper-scoring-rule reward, REINFORCE with a group-mean baseline) optimises the same scoring rules. When targets come as distributions, minimising the log score directly gives the exact gradient that REINFORCE only estimates. That is a reading of the definitions rather than a published comparison, so start supervised and treat RL as an optional last stage.
- Report Brier score, ECE before and after temperature, and soft accuracy, the metrics Laya publishes.

## Training plan

Train in three stages: label reading, typed decisions, then calibration. All data comes from [the dataset survey](DATASETS.md), converted to one schema, and the encoder stays frozen until stage 2b. Open reproductions trained on small budgets: Tev1 cost about $17 on ~38k examples, Kev about $95 of H100 time, and autotrust/JEV about 3 B200 GPU-hours. Times below are estimates for a 12 GB RTX 4070 Super, at about 60M tokens per hour with the encoder frozen and 35M with LoRA; the whole plan takes roughly a day.

### Stages at a glance

| Stage | Encoder | Main data | Target | Loss | Peak LR | Size and time on a 4070 Super |
|---|---|---|---|---|---|---|
| 1. Label reading | Frozen, run forward each step | Synthetic zero-shot data, UTCD, NLI | Gold label, one-hot | Cross-entropy | Head 1e-4 | ~1.5M questions, ~225M tokens, ~4 h |
| 2a. Typed decisions, frozen | Frozen | Jev-format corpora plus rule, scoring, routing and safety sets | Soft distribution where the source has one, else one-hot | KL; KL + 0.5 × RPS for score | Head 1e-4 | ~0.5M questions, ~175M tokens, ~3 h |
| 2b. Typed decisions, LoRA | LoRA r=16 | Same mix, new sample | As in 2a | As in 2a | LoRA 1e-4, head 1e-4 | ~1M questions, ~350M tokens, ~10 h |
| 3a. Soft-label tuning | LoRA | Human vote distributions | Share of annotators per option | KL | 1e-5 for LoRA and head | ~150k items plus replay, ~55M tokens, ~1.5 h |
| 3b. Temperatures | Everything frozen | Held-out calibration data | Gold label or Jev distribution | NLL, one temperature per question type and option-count bucket | Fitted with L-BFGS | Minutes |

Token counts assume ~150 tokens per stage-1 question and ~350 per stage-2 question. Sample sizes throughout are starting points to tune, not values taken from a paper.

### Settings shared by all stages

- **One schema.** Convert every source to the layout of [jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3): a state, a question with its kind, its options, and a target that sums to 1. noul targets are [P(false), P(true)], choice targets follow the options, and score targets cover ordered levels (that corpus uses levels 0–5 and 2–16 options). Multi-label sets become one noul question per label.
- **Mixing.** Sample with [T5](https://arxiv.org/abs/1910.10683)'s examples-proportional mixing and a size cap K, so no source crowds out the rest. Start at K = 2^17 ≈ 131k examples per source, a value inside T5's tested range, and cap each task inside tasksource-jev at about 15k, a per-dataset cap used in instruction-tuning work. Upweight the small rule sets with T5's temperature-scaled mixing at T = 2.
- **Batches.** 128 questions per optimizer step, stratified so each question type fills at least a sixth of the batch, and bucketed by length, as autotrust/JEV did. On 12 GB that means ~6k-token micro-batches with gradient accumulation. Keep the batch there in the LoRA stages: LoRA pays a larger loss penalty than full fine-tuning when the batch grows ([LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)).
- **Optimizer.** AdamW with β = (0.9, 0.98) and gradient clipping at 1.0, autotrust/JEV's settings. Warm up linearly over the first 3% of steps, then decay with a cosine to 10% of the peak, the shape of [BLIP-2-style Q-Former runs](https://arxiv.org/pdf/2310.03291) (1e-4 down to 1e-5). Weight decay 0.05 on head weight matrices, the value those runs use; none on norms, biases or layer-mix scalars.
- **Loops.** Sample the loop count per batch from 1 to 8 (1 to 4 if you will deploy at 4 or fewer). Backpropagate through every loop up to 4 and through only the last 4 beyond that, Huginn's truncated backpropagation. Score validation at 1, 2, 4 and 8 loops.
- **Head initialisation.** Copy the encoder's top layers into the interaction blocks, including its self-attention weights into the cross-attention, as MORES did. Per-loop LoRA starts at zero, so every loop begins as the shared block.
- **Speed.** bf16 autocast, Flash Attention with packed sequences, and torch.compile, which ModernBERT's authors credit with about 10% more training throughput.
- **Learning-rate sweeps.** Before each full stage, sweep the peak rate on a 10–20M-token slice and keep the value with the lowest validation loss.

### Stage 1: label reading

Goal: score option texts the model has never seen against a state. zeroshot-v2.0 and GLiClass learned this skill from synthetic label data plus NLI.

| Source | Becomes | Sample | License |
|---|---|---|---|
| synthetic_zeroshot_mixtral_v0.1 | choice over the task's label names | ~600k of 2.6M | Apache-2.0 |
| UTCD, in-domain train | choice over descriptive labels | ~300k of 2.19M | MIT |
| zero-shot-label-nli | noul: does the label statement fit the text? | ~300k of 1.9M | Filter on the per-row license field |
| MultiNLI, SNLI, WANLI, FEVER-NLI | choice over entailment, neutral and contradiction; premise as state, hypothesis as question | ~300k combined | OANC and CC BY-SA; CC BY-SA 4.0; CC BY 4.0; FEVER-NLI not stated |

- **Target and loss:** one-hot gold labels with cross-entropy, which equals KL against a one-hot target; binary log loss for noul.
- **Learning rate:** 1e-4 peak for the head, per-loop LoRA and layer-mix scalars. A reproduction that mirrors BLIP-2's Q-Former settings peaks at 1e-4, and autotrust/JEV ran its head at 2e-4. Sweep 5e-5, 1e-4, 2e-4 and 3e-4.
- **Length:** one pass, about 12k steps at 128 questions per step.
- **Stop** when zero-shot accuracy on UTCD's 9 out-of-domain sets stops improving.
- Keep ANLI (CC BY-NC 4.0) out of commercial runs.

### Stage 2: typed decisions

Goal: choice, noul and score over states with several questions, weighted toward rules and policy, the largest topic group in Benchmark Heaven's JevBench.

| Source | Types | Target | Sample per epoch | Notes |
|---|---|---|---|---|
| tasksource-jev | All three | Soft where the source keeps annotator disagreement, else one-hot | ~400k, at most 15k per task | Filter on the per-row license field |
| jev-distill-corpus-v3, yuri_v3 stream | All three | Jev 1.13's full output distributions | ~300k | Closest public proxy for Jev; check TypeSafe's and OpenRouter's terms |
| jev-distill-corpus-v3 openjev_v2 stream, plus Open-Jev | All three | Labels from the generators | ~100k | CC0 |
| jev-distill-corpus-v3, yuri_v1 stream | noul | Placeholder [0.5, 0.5] per autotrust/JEV; the dataset card calls them 32B-teacher labels | Inspect, then drop or keep at weight 0.05 as autotrust/JEV did | Memory-relevance pairs |
| typed-decisions-synth | All three | DeepSeek teacher distribution | All 7.4k, upweighted | MIT |
| Kev decision-v7 and later stages, Tev1 recipe, Nimble pairs | All three | Labels computed by code | All, about 70k, upweighted | Drop their Banking77 and AG News parts, which are held out |
| ShARC, ABCD, ContractNLI, When2Call | choice, noul | One-hot | ~60k | Rule application, including "need more information" options |
| SST-5, Yelp Review Full, HelpSteer2, Feedback Collection | score | One-hot level | ~100k | Rubric scoring is Jev's weakest area in Bespoke's mix |
| CLINC150, MASSIVE (en-US), Bitext | choice with 27–150 options | One-hot; CLINC150 adds out-of-scope | ~60k | Trains the many-option range where the split should win |
| Aegis 2.0, WildGuardMix, Civil Comments | noul, choice | One-hot; Civil Comments' rater share as P(true) | ~60k | WildGuardMix is gated |

- **2a, frozen.** Continue from stage 1 with the head at 1e-4 and a fresh 3% warmup. Run ablations A–D from the evaluation grid here, where each run is cheap.
- **2b, LoRA.** Add LoRA with rank 16, alpha 32 and dropout 0.05, autotrust/JEV's settings, on Wqkv, Wo, Wi and the MLP Wo of all 28 layers. Peak rate 1e-4 for LoRA and head: autotrust/JEV trained LoRA at 1e-4, LoRA's best rate runs about 10× full fine-tuning's, and full fine-tuning of ModernBERT on GLUE chose rates below 1e-4 (per the [Ettin authors](https://arxiv.org/pdf/2507.11412), who reused its sweep). Restart AdamW's state with a 2% warmup, as autotrust/JEV did for its second phase. Sweep 5e-5 to 3e-4; the best rate barely moves with LoRA rank, so changing rank needs no new sweep.
- **Loss:** KL(target ‖ model) per question, plus 0.5 × RPS for score, autotrust/JEV's weighting. With hard labels the KL is plain cross-entropy.
- **Length:** about one epoch of the capped mix; autotrust/JEV stopped at 0.93 of an epoch.
- **Model selection:** jev-distill-corpus-v3's validation split (14,111 rows) plus held-out slices of the rule and many-option sets.
- **2c, optional benchmark fine-tune.** Laya's 0.766 typed-decisions checkpoint was fine-tuned on that benchmark's training split. For a like-for-like comparison, fine-tune a copy on LocalLLaMA/typed-decisions' 1,200 training cases at half the 2b rate for 2–3 epochs (a starting point), and report it next to the zero-shot score, never in place of it.

### Stage 3: calibration

**3a, soft-label tuning.** Fine-tune on human vote distributions. NLI models trained on annotator distributions instead of gold labels came closer to ChaosNLI's human distributions and, on the same data, scored higher ([Meissner et al., 2021](https://arxiv.org/abs/2106.03020)).

| Source | Becomes | Target | Size | License |
|---|---|---|---|---|
| UNLI | noul: is the hypothesis true? | Rated probability | 55,517 train pairs | Not stated |
| GoEmotions, raw rater config | One noul question per emotion | Share of raters | 58k comments | Apache-2.0 |
| DICES | noul: is the conversation unsafe? | Share of raters | 350 + 990 conversations | CC BY 4.0 |
| LeWiDi-2025 | choice or score, by task | Annotator distribution | 26,814 items | Not stated |
| ChaosNLI | choice over the 3 NLI labels | 100-vote distribution | 4,645 items | CC BY-NC 4.0, so evaluation only for a commercial model |

- **Loss:** KL to the vote distribution.
- **Learning rate:** 1e-5 for LoRA and head, a tenth of stage 2; [InstructBLIP](https://arxiv.org/abs/2305.06500) fine-tuned an already-trained Q-Former at 1e-5. Train one epoch with about a quarter of each batch replayed from stage 2 so typed decisions don't drift. The replay share is a precaution, not a tested value.

**3b, temperatures.** Freeze everything. Fit one temperature per question type and option-count bucket (2, 3–4, 5–8, 9–16, 17 or more) by minimising NLL with L-BFGS; autotrust/JEV fitted one per question type and Laya one per type and option count. Use jev-distill-corpus-v3's 13,766-row calibration split without the placeholder stream (autotrust/JEV fitted on 10,954 calibration rows and left that stream out), plus held-out slices of the many-option sets so every bucket has data. Temperatures change confidence, never the top option.

Optional after 3b: train the act/escalate head on held-out predictions, with log loss on whether the top option was right.

### Losses

- **choice and noul:** L = KL(p ‖ q) = Σᵢ pᵢ log(pᵢ / qᵢ), with p the target and q the softmax over the question's options. With a one-hot p this is cross-entropy; for noul it is binary log loss with a soft target.
- **score:** L = KL(p ‖ q) + 0.5 × RPS, where RPS = 1/(K−1) × Σₖ (Pₖ − Qₖ)² over the cumulative target and predicted distributions of K ordered levels.
- **Weights:** 1 per question, 0.05 for placeholder rows; average within type-stratified batches.
- **Temperature fit:** NLL of the gold label, or cross-entropy to the Jev distribution, with only the temperature free.

### Learning rates by stage

| Parameter group | Stage 1 | 2a | 2b | 3a | Basis |
|---|---|---|---|---|---|
| Head blocks and scorer | 1e-4 | 1e-4 | 1e-4 | 1e-5 | BLIP-2-style Q-Former pretraining peaks at 1e-4; InstructBLIP fine-tuned at 1e-5; autotrust/JEV's head ran at 2e-4 |
| Per-loop FFN LoRA | 1e-4 | 1e-4 | 1e-4 | 1e-5 | Trained with the head |
| Layer-mix scalars | 1e-4, no weight decay | 1e-4 | 1e-4 | Frozen | My default; 28 weights per pass |
| Encoder LoRA, r=16 | Not trained | Not trained | 1e-4 | 1e-5 | autotrust/JEV's LoRA rate; about 10× the rates used for full fine-tuning |

Every peak uses a 3% warmup (2% after the optimizer restart in 2b) and cosine decay to 10%.

### Held out from all training

LocalLLaMA/typed-decisions test (400 cases), jev-distill-corpus-v3's test_set_30k and ood splits, JevBench's 300 public items, Leanmcp JevBench, Kev's transfer-v4 and breadth-v1, Bespoke's 13-set mix, UTCD's out-of-domain sets, and the test splits of Banking77, AG News and DAIR Emotion for the comparison with Laya. Five of Bespoke's sets share a source with training data here (MultiNLI, MASSIVE en-US, Civil Comments, Aegis 2.0, HelpSteer2), as do Leanmcp JevBench's SST-5 and Aegis slices; train only on those sources' train splits and report those sets as in-distribution.

### Augmentations, each with a precedent

- **Option shuffling:** shuffle options in 30% of examples, as autotrust/JEV did. With an order-free head this mainly protects pass A when question text lists the options.
- **Option counts:** vary options from 2 to 24 per question and add distractors sampled from the same taxonomy; Tev1 was trained on 2–24 options.
- **Contrastive pairs:** add minimal pairs in which one changed fact flips the answer, Nimble's curation method.
- **Multi-question packing:** put all questions about one state into one pass-A sequence, as Laya answers every question in one forward pass.
- **None of these:** include out-of-scope examples with a "none" option, as CLINC150 does.

### Licensing

For a commercial model, drop ChaosNLI from stage 3, keep ANLI, DAIR Emotion and BeaverTails for evaluation only, and filter tasksource-jev on its per-row license field. Check terms before training on jev-distill-corpus-v3, whose largest stream is Jev's own outputs, and on UNLI and LeWiDi, which state no license.

## Evaluation and expected results

Judge the design on four axes: accuracy on the shared benchmarks, accuracy as option count grows, calibration, and latency with cached options. The baselines are the published numbers to beat; the ablation grid isolates each design choice.

### Baselines to beat

| Benchmark | Jev 1.13 | Laya | Source |
|---|---|---|---|
| typed-decisions: 400 cases, 2,000 decisions (sizes match the LocalLLaMA/typed-decisions test split) | 0.727 | 0.766 fine-tuned; 0.362 base, zero-shot | [Laya card](https://huggingface.co/convaiinnovations/laya) |
| AG News, 4 labels | 0.910 | 0.950 | Laya card |
| DAIR Emotion, 6 labels | 0.480 | 0.595 | Laya card |
| Banking77, 77 labels | 0.870 | 0.425 | Laya card |
| ECE in the card's comparison table (Laya's after temperature fitting) | 0.246 | 0.081 | Laya card |
| p50 latency, 1 question | 236–276 ms | 32.8 ms | Laya card |

Jev also scores 76.0% on Bespoke Labs' 13-set human-labelled mix (Nimble 9B: 75.7%), a useful out-of-domain check. Measure latency on a T4, the GPU behind Laya's figure, and plot throughput from 1 to 128 options, the sweep GLiClass reports.

### Ablation grid

| Run | Head | Loops | Encoder | Question it answers |
|---|---|---|---|---|
| A | 2 layers of self-attention over the concatenation | 1 | Frozen + layer mix | What does the split cost with a simple head? |
| B | 2 cross- plus self-attention blocks | 1 | Frozen + layer mix | Which head type wins at equal parameters? |
| C | Same as B | 1–8 sampled in training; scored at 1, 2, 4 and 8 | Frozen + layer mix | Does looping add accuracy at fixed parameters? |
| D | 8 unshared blocks | 1 | Frozen + layer mix | Iso-compute control for C |
| E | Best of B–D | Best of B–D | LoRA r=16 | How much does adapting the encoder add? |
| F | Laya-style joint encoder and 2-layer head, same data | 1 | LoRA r=16 | What does the split cost overall? |

### What the evidence predicts

- **Split vs. joint (E vs. F):** a few points lower on sets with 20 or fewer options. DeFormer lost 0.6–1.8 points with only its lower layers split, and MORES stayed within 2% only by adding interaction blocks.
- **Many options:** better than Laya on Banking77-style tasks, with latency nearly flat as options grow, as GLiNER's bi-encoder (5.2% slowdown from 1 to 1,024 labels) shows for designs that keep labels out of the text context.
- **Looping (C vs. B and D):** gains on rule and policy items, JevBench's largest group, and on typed-decisions; little on topic or intent sets, per Saunshi et al. Per-loop FFN LoRA should matter, per ALBERT.
- **Frozen vs. LoRA (B–D vs. E):** parity on NLI-like questions and a gap on paraphrase- or similarity-like ones (Peters et al.: 0.0 to 6.7 points), which LoRA should close given its GLUE parity.
- **Calibration:** raw over-confidence like Laya's, and a large ECE drop after per-type temperatures.

These are expectations drawn from cited results on related setups, not measurements of this design.

## Sources

- Jev-class models: [Laya model card](https://huggingface.co/convaiinnovations/laya), [Laya training notes](https://huggingface.co/Emerald7664/laya), [autotrust/JEV](https://huggingface.co/autotrust/JEV), [Nimble evaluation on Ollama](https://ollama.com/library/nimble:9b-q8_0), [Tev1 on Ollama](https://ollama.com/library/tev1:4b), [Kev-4B card](https://haimaker.ai/models/jaredpalmer/kev-4b)
- Split encoding: [Poly-encoders](https://arxiv.org/abs/1905.01969), [DeFormer](https://arxiv.org/pdf/2005.00697), [MORES](https://arxiv.org/pdf/2004.13313), [GLiClass](https://www.arxiv.org/pdf/2508.07662), [GLiNER bi-encoder](https://arxiv.org/html/2602.18487)
- Heads over frozen encoders: [BLIP-2 Q-Former](https://arxiv.org/abs/2301.12597), [Set Transformer](https://arxiv.org/abs/1810.00825)
- Looping: [Saunshi et al.](https://arxiv.org/abs/2502.17416), [ALBERT](https://arxiv.org/pdf/1909.11942), [Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672), [Huginn](https://arxiv.org/abs/2502.05171), [Parcae](https://arxiv.org/html/2604.12946), [Universal Transformer](https://arxiv.org/abs/1807.03819)
- Encoder adaptation: [Peters et al.](https://arxiv.org/pdf/1903.05987), [LoRA](https://arxiv.org/pdf/2106.09685), [ModernBERT](https://arxiv.org/abs/2412.13663)
- Calibration and training recipe: [temperature scaling](https://arxiv.org/abs/1706.04599), [T5 data mixing](https://arxiv.org/abs/1910.10683), [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/), [Meissner et al. on soft NLI targets](https://arxiv.org/abs/2106.03020), [InstructBLIP](https://arxiv.org/abs/2305.06500), [BLIP-2-style pretraining settings](https://arxiv.org/pdf/2310.03291), [Ettin's reuse of the ModernBERT GLUE sweep](https://arxiv.org/pdf/2507.11412), [jev-distill-corpus-v3 card](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
- Datasets: [DATASETS.md](DATASETS.md)
