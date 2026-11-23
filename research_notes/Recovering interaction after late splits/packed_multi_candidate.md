# Packed multi-candidate encoders (context + all candidates in one pass)

Scope: designs that put a context and several candidates into one encoder pass, using structured attention masks and/or special position handling. Each design is compared with (a) one cross-encoder pass per candidate and (b) bi-encoders, and with LIDM's pass A / pass B split.
Project anchor: LIDM = ModernBERT-large (395M), pass A `[CLS] state [SEP] question [SEP]`, pass B encodes each option alone, fusion head scores. Laya = packed state+question+options with a 192/512 option-token budget.

Most important source found: **Verdict-MQ (arXiv 2610.20778, 14 Oct 2026)**. It is a packed, structured-mask design on the same ModernBERT-large backbone, benchmarked against the released Laya typed-decisions model (421M), and its hardware is close to LIDM's (a 16 GB laptop RTX 4090). It is a single, recent, non-peer-reviewed preprint, so treat it as unreplicated.

## Q1. Uni-Encoder (Song et al., Findings ACL 2023): one pass for all candidates, candidates isolated from each other

### Takeaway
On Ubuntu V2 (10 candidates, BERT-base), Uni-Encoder puts all candidates after one copy of the context. The "Arrow" mask blocks candidate-to-candidate attention and every candidate reuses the same position IDs. In the authors' own controlled ablation it beats the cross-encoder by +1.5 R10@1, the poly-encoder by +1.5 and the bi-encoder by +2.4. It is about 4x faster than a cross-encoder. Shared position IDs are worth +2.2 R10@1 over unique ones, and letting candidates see each other costs 3.3.

### Cited Findings
- Venue and date: Findings of ACL 2023. arXiv v1 2 Jun 2021, v5 15 May 2023. Authors: Chiyu Song, Hongliang He, Haofei Yu, Pengfei Fang, Leyang Cui, Zhenzhong Lan. The paper claims Ubuntu V2 R10@1 +2.9% over the previous SOTA at about 4x faster inference — [arXiv abs](https://arxiv.org/abs/2106.01263)
- Mask: "the context and the candidates have full attention, but the candidates do not attend to each other" (Arrow attention). Position handling: "the same positional embedding for all candidates to ensure they are treated equally". Each candidate is wrapped in [CLS]…[SEP] and each context utterance has a speaker token. Backbone: BERT (HF checkpoint), also tested with BERT-FP post-trained checkpoints. Loss: classification plus an equally weighted MLM. Negatives are the other labels in the batch — [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2106.01263)
- Controlled ablation, Ubuntu V2, 10 candidates, BERT-base, same codebase (R10@1 / R10@2 / R10@5 / MRR) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.01263):
  - (i) Uni-Encoder, Arrow mask, **shared** position IDs, batch 8: 0.859 / 0.938 / 0.990 / 0.915
  - (ii) Uni-Encoder, Arrow mask, **unique** position IDs: 0.837 / 0.933 / 0.992 / 0.903 (−2.2 R10@1)
  - (iii) "Concat-Cross-Encoder", Square mask (candidates attend to each other): 0.826 / 0.916 / 0.980 / 0.892 (−3.3)
  - (iv) Cross-encoder, one pass per pair, batch 5 (memory-limited): 0.844 / 0.930 / 0.987 / 0.905 (−1.5)
  - (v) Bi-encoder, Diagonal mask: 0.835 / 0.925 / 0.987 / 0.899 (−2.4)
  - (vi) Poly-encoder, Light-Arrow, 360 codes: 0.844 / 0.929 / 0.989 / 0.906 (−1.5)
  - The authors attribute part of the cross-encoder's deficit to its smaller batch (5 vs 8).
- Other benchmarks (BERT-base unless noted) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.01263):
  - PersonaChat, 20 candidates: Uni-Encoder R20@1 0.869, MRR 0.922, vs BERT 0.707 and BERT-CRA 0.843.
  - Ubuntu V1, 10 candidates: Uni-Encoder 0.886 R10@1. With BERT-FP post-training: 0.916 (BERT-FP alone 0.911).
  - Douban, 10 candidates: Uni-Encoder + BERT-FP MAP 0.648 / P@1 0.518 (BERT-FP 0.644 / 0.512).
  - No E-commerce results appear in the paper.
- Speed: Ubuntu V2 test set (189,200 contexts), pools of 10, 20, 50 and 100 candidates, one A100, batch size maximised per paradigm. Uni-Encoder is about 4x faster than the cross-encoder "when the pool size is appropriate", and the gap grows with pool size. It is slightly more efficient than the poly-encoder. No per-pool timings appear in the text, only in Fig. 3. Accuracy is never measured against pool size. The limitations section says the method suits systems where the number of responses is small — [ar5iv](https://ar5iv.labs.arxiv.org/html/2106.01263)

### Inferences
- The name "Arrow" and the quote "context and candidates have full attention" suggest an arrowhead matrix: context rows see all candidates, candidate rows see context plus themselves. If so, candidates can still exchange information indirectly through the context from layer 2 on, and the context encoding cannot be cached or reused across candidate sets. Verdict-MQ (Q4) removes this by making state attend only to state.
- Shared position IDs and the candidate-isolation mask work as a pair. Shared positions make candidates order-invariant, but they are only safe if candidates cannot see each other. Verdict-MQ reports collapse in that case (Q4), and Uni-Encoder reports "confusion" under the Square mask.

### Gaps
- The paper gives no maximum candidate length, no candidates-per-sequence count for training, and no memory or FLOPs figures. Accuracy vs pool size is not measured.

## Q2. GLiClass (Knowledgator, Aug 2025) and GLiNER/GLiNER2 label packing: accuracy and speed vs label count

### Takeaway
GLiClass uni-encoder (labels prefixed with `<<LABEL>>` and packed with the text, 1,024-token cap) keeps throughput nearly flat from 1 to 128 labels, while NLI cross-encoders slow down about 50-70x. Its accuracy advantage over the NLI cross-encoder shrinks with many labels: it ties at Banking77 (77 labels). The authors say label-to-label attention and text representations degrade when label tokens outnumber text tokens. ModernBERT-based GLiClass trails the DeBERTa-v3-based versions by 10 points on average. The paper reports per-variant accuracy only for the uni-encoder; bi-encoder, fused bi-encoder and encoder-decoder have no published numbers there. A 2026 GLiNER paper does compare uni and bi for NER: bi-encoders match or beat uni on accuracy and run about 130x faster at 1,024 labels.

### Cited Findings
- Paper: "GLiClass: Generalist Lightweight Model for Sequence Classification Tasks", arXiv 2508.07662, 11 Aug 2025, Stepanov et al. (Knowledgator) — [arXiv HTML](https://arxiv.org/html/2508.07662)
- Variants described in the paper — [arXiv HTML](https://arxiv.org/html/2508.07662):
  - **Uni-encoder** (primary): text and labels go through one bidirectional encoder; label and text features are pooled separately (first-token, mean or attention-weighted), then scored with a dot product plus temperature or a small MLP.
  - **Bi-encoder**: separate encoders, temperature-scaled logits.
  - **Fused bi-encoder**: class embeddings injected into the text embedding layer before the text encoder runs.
  - **Encoder-decoder**: a label decoder cross-attends to the text encoder.
  - Other components: squeeze-excitation layer-wise attention re-weighting and a token-level contrastive loss.
  - The paper names no "post-fusion" variant. The GitHub README lists `uni-encoder` (default, "most efficient"), `bi-encoder`, `bi-encoder-fused` and `encoder-decoder`, and gives no per-architecture numbers — [GitHub](https://github.com/Knowledgator/GLiClass)
- Packing details: each label is prefixed with a `<<LABEL>>` special token and concatenated with the text, then the whole sequence goes through one pass. Training inputs are capped at 1,024 tokens. The pre-training analysis finds that "attention between label tokens and `<<LABEL>>` tokens weakens as the number of labels grows" and that text representations degrade when label tokens greatly outnumber text tokens. Label order sensitivity is not discussed — [arXiv HTML](https://arxiv.org/html/2508.07662)
- Backbones and sizes:
  - DeBERTa-v3: edge 32.7M, base 187M, large 439M.
  - ModernBERT: modern-base 151M, modern-large 399M.
  - "DeBERTa-based models consistently outperformed ModernBERT."
  - Zero-shot average over 14 datasets: large 0.7193, base 0.6764, modern-large 0.6197, modern-base 0.5577, edge 0.4900.
  - [arXiv HTML](https://arxiv.org/html/2508.07662)
- Per-dataset zero-shot scores, GLiClass-large (DeBERTa-v3, 439M) vs the deberta-v3-large zeroshot-v2.0 NLI cross-encoder (one pass per label) — [arXiv HTML](https://arxiv.org/html/2508.07662):

  | Dataset (labels) | GLiClass-large | NLI cross-encoder | Δ |
  |---|---|---|---|
  | SNIPS (7) | 0.9692 | 0.8340 | +13.5 |
  | MASSIVE (60) | 0.5649 | 0.5180 | +4.7 |
  | AG News (4) | 0.7181 | 0.7710 | −5.3 |
  | Banking77 (77) | 0.5574 | 0.5670 | −1.0 |
  | financial_phrasebank (3) | 0.9000 | 0.5820 | +31.8 |
  | emotion (6) | 0.4506 | 0.4840 | −3.3 |
  | Average | 0.7193 | 0.6821 | +3.7 |

  - The NLI average covers 13 datasets.
  - Banking77 for modern-large is 0.3683 and for edge 0.0272.
- Few-shot (8 examples per label, 8 datasets): modern-large 0.5205 → 0.7082 (+36%), large 0.6202 → 0.7265 — [arXiv HTML](https://arxiv.org/html/2508.07662)
- Throughput (examples/s, A6000, batch 1) at 1 / 8 / 32 / 128 labels — [arXiv HTML](https://arxiv.org/html/2508.07662):

  | Model | 1 | 8 | 32 | 128 |
  |---|---|---|---|---|
  | gliclass-large | 19.05 | 29.27 | 28.79 | 17.60 |
  | gliclass-modern-large | 46.30 | 46.30 | 44.73 | 32.89 |
  | deberta-v3-large NLI cross-encoder | 16.82 | 3.98 | 0.99 | 0.25 |
  | roberta-large NLI cross-encoder | 50.42 | 9.95 | 2.48 | 0.64 |

  The paper summarises GLiClass as dropping about 7-20% from 1 to 128 labels, against about 52x for the deberta-v3-base cross-encoder.
- Authors' limitations: accuracy declines on large label sets (Banking77 cited), which they attribute to positional encoding and attention limits. 1,000+ labels need truncation or batching because of the ~1,024-token context. They say "cross-encoders … handle dense information better" — [arXiv HTML](https://arxiv.org/html/2508.07662)
- GLiNER2 (Zaratiana et al., Fastino, arXiv 2507.18546, 24 Jul 2025) — [arXiv HTML](https://arxiv.org/html/2507.18546):
  - Packing format: `[P] task ([L] label1 … [L] labelK) [SEP] text`. [E] tokens are used for entity types and [C] for fields; several tasks can be chained in one sequence. 205M parameters, 2,048-token context (GLiNER: 512). Each [L] embedding is scored by an MLP.
  - Zero-shot accuracy (GLiNER2 / DeBERTa-v3 NLI / GLiClass / GPT-4o): Banking77 (77 labels) 0.70 / 0.42 / 0.21 / 0.78. SNIPS 0.83 / 0.77 / 0.80 / 0.97. AG News 0.74 / 0.68 / 0.68 / 0.85.
  - CPU latency from 5 to 50 labels: GLiNER2 130 → 208 ms, GLiClass 137 → 190 ms, DeBERTa NLI 1,714 → 16,897 ms (linear in labels).
- **Conflict:** GLiNER2 reports GLiClass Banking77 = 0.21, while the GLiClass paper reports 0.5574 (large) and 0.4698 (base). The GLiClass version and size GLiNER2 used is not stated, so the two numbers are not comparable — [GLiNER2](https://arxiv.org/html/2507.18546) vs [GLiClass](https://arxiv.org/html/2508.07662)
- "The Million-Label NER" (Knowledgator, arXiv 2602.18487, 11 Feb 2026), GLiNER uni-encoder (types prepended in-sequence) vs bi-encoder (cached label embeddings, optional cross-attention fusion layer) — [arXiv HTML](https://arxiv.org/html/2602.18487):
  - Zero-shot CrossNER micro-F1: bi-large 61.5 vs uni-large 60.9.
  - Broader suite average: bi 49.7 vs uni 46.4-47.0. On GENIA the gap is 60.1 vs 46.9.
  - Throughput (H100, batch 1, 1 → 1,024 labels): uni gliner_small-v2.5 10.7 → 0.14 ex/s (−98.7%); uni large 0.03 ex/s at 1,024. Bi-edge with pre-computed labels 19.3 → 18.3 ex/s, about 130x the uni-encoder at 1,024 labels.
  - Entity-type shuffling is used during training. No accuracy-vs-label-count curve is reported.

### Inferences
- In GLiClass/GLiNER the labels appear to share full attention with each other and to carry distinct positions; no isolation mask is described. That makes them order-dependent, and as label count grows label tokens crowd out the text. This matches LIDM's −11 on Banking77 names: Laya packs short names in a similar way and does well. It also explains why packed models stop beating per-label cross-encoders around 60-77 labels.
- For long label descriptions (LIDM's +11 case), a GLiClass-style uni-encoder hits its 1,024-token cap long before 77 descriptions fit. That is the same truncation failure Laya shows at 192/512.

### Gaps
- No published accuracy numbers for GLiClass bi-encoder, fused bi-encoder or encoder-decoder variants, and no per-variant speed numbers.
- No study found that measures packed-encoder accuracy as a smooth function of label count (e.g. 8 → 32 → 128 labels on one dataset). Only cross-dataset comparisons exist.
- GLiClass label-order sensitivity is not reported.

## Q3. Multiple-choice and option-packing encoders (UniMC, per-option heads, option ordering)

### Takeaway
UniMC (ALBERT-xxlarge, EMNLP 2022) packs all options first, then question and passage, with a mask that stops options from seeing each other. The mask alone carries almost all the gain: 78.0 average with the mask only vs 39.1 with position-ID changes only. Classical RACE work that compares "single-choice" with "multi-choice" heads still runs one pass per option, so neither arm is packing. No clean same-backbone encoder study of packed vs per-option MCQA was found apart from Verdict-MQ (Q4).

### Cited Findings
- UniMC ("Zero-Shot Learners for NLU via a Unified Multiple Choice Perspective"), EMNLP 2022, arXiv 2210.08590, 16 Oct 2022 — [arXiv abs](https://arxiv.org/abs/2210.08590); details from [ar5iv](https://ar5iv.labs.arxiv.org/html/2210.08590):
  - Layout: `[CLS] {[O-MASK] option}×N [SEP] {question [SEP]}? passage [SEP]`, with segment IDs separating options from context. Position IDs are changed so that tokens within each option are distinguished; the exact reset scheme is not stated. An attention mask stops options attending to each other.
  - Backbone: ALBERT-xxlarge-v2 (235M); RoBERTa-large (355M) scored lower (NLI 53.0 vs 58.2). Loss: MLM + option-MLM (yes/no at each [O-MASK]) + a softmax over options. Logit masking allows varying option counts in one batch.
  - Mask ablation (average accuracy): random 38.4; position-ID changes only 39.1; attention mask only 78.0; both 78.6.
  - Zero-shot results: ANLI R1/R2/R3 52.0/44.4/47.8 vs FLAN 47.7/43.9/47.0. CB 75.7 vs 64.1. COPA 95.2 vs 90.6. HellaSwag 62.5 vs 56.4. Winogrande 65.8 vs 67.3 (FLAN higher). DBpedia 88.9, AG News 81.3.
  - Prompt-wording sensitivity is low: SD 0.3 on SST-2 vs 8.3 for UnifiedQA-T5-3B. Option count and option order are not analysed.
- Single-choice vs multi-choice on RACE (Jiang et al. 2020, ALBERT): the "multi-choice" baseline is the standard per-option pass with a softmax across options, and "single-choice" is a binary scorer per option. The authors found single-choice better, with RACE 90.7 single model and 91.4 ensemble. They note that multi-choice heads need a fixed option count and duplicate passage content — [arXiv 2011.03292](https://ar5iv.labs.arxiv.org/html/2011.03292) (via search summary; full tables not fetched)
- LLM evidence that showing all options together changes results: scoring options in isolation vs with all options visible narrows the ARC-Challenge/ARC-Easy gap "up to six times" (decoder LLMs, Dec 2024) — [arXiv 2412.17758](https://arxiv.org/html/2412.17758v1) (search snippet only)
- Order effects in causal models: "Lost in the Prompt Order" (Ok & Lee, Findings ACL 2026). Context→question→options (CQO) beats question→options→context (QOC) by more than 14 points, because the causal mask stops option tokens from seeing a later context. Bidirectional encoders avoid this by construction — [ACL Anthology](https://aclanthology.org/2026.findings-acl.1921/)

### Inferences
- UniMC and Uni-Encoder agree: the candidate-isolation mask is the critical part, and position handling is second-order. Verdict-MQ (Q4) agrees too (shared vs ordered positions within noise).

### Gaps
- DUMA, OCN, DCMN+ and MMM all encode one sequence per option (plus co-attention). I did not fetch their numbers, and none of them is a packed design.
- No encoder paper was found that runs RoBERTa/DeBERTa on RACE in both packed and per-option form with everything else held fixed, other than Verdict-MQ's partial comparison.

## Q4. Attention-mask tricks: block-diagonal masks, shared state prefix, position resets, order invariance (Verdict-MQ, Set-Encoder, Block-Attention)

### Takeaway
Verdict-MQ is LIDM's closest relative and was measured against the same Laya model. On ModernBERT-large it uses a four-rule mask:
- state attends only to state, so it is cacheable and question-independent;
- each question attends to the state and itself;
- each option attends to the state, its question and itself, never to other options;
- options share position IDs, and the sliding window is computed on position IDs.

It beats Laya on typed decisions (accuracy 0.801 vs 0.767, KL 0.080 vs 0.117). It is 1.5 points behind a matched baseline on RACE (77.4 vs 78.9), and running packed is 2.8x faster than per-question passes. Options that can see each other collapse to one representation. Set-Encoder shows the alternative: one sequence per candidate, with candidates exchanging information only through one [INT] token each. That keeps order invariance for about 2% extra time.

### Cited Findings
- Verdict-MQ ("Verdict: Encode Once, Decide Many", Chouhan & Perry, arXiv 2610.20778, 14 Oct 2026) — [arXiv abs](https://arxiv.org/abs/2610.20778); details from [arXiv HTML](https://arxiv.org/html/2610.20778):
  - **Backbone:** ModernBERT-base (150M) and ModernBERT-large (395M), the large one NLI-initialised from tasksource. Laya, per the paper: a ModernBERT-based 421M typed-decision model whose docs say "one single forward pass". The authors did not confirm its internals.
  - **Layout:** `[CLS] s [SEP] | b_1 || … || b_K`, where each question block holds a prefix, the question, [SEP] and one marker per option. Max 2,048 tokens per packed case; options capped at 48 tokens.
  - **Mask rules:**
    - (1) state → state only;
    - (2) each question block → state + itself, and every block starts at the same position right after the state;
    - (3) question tokens → all options of their own question; each option → state + its question + itself;
    - (4) options in a block share position IDs, and "the sliding window is defined on position ids rather than sequence index". The mask is applied in both the global and the sliding-window layers. Permuting options only permutes the outputs.
  - **Readout:** the final hidden state at each option marker plus a question-type embedding → two-layer MLP → softmax over options.
  - **Ablations:**
    - Options attending to each other: markers "collapse to a single representation", the model outputs uniform distributions and does not learn.
    - Ordered (unique) option positions vs shared, ModernBERT-base, 1 seed: 0.686 vs 0.681 (within noise).
    - Reusing [CLS] as the marker vs a new random [OPT] token, large, 5 seeds: 0.801 ± 0.005 vs 0.790 ± 0.007 (p = 0.022).
  - **Typed decisions** (400 cases, 2,000 decisions, 5 questions per case; accuracy / KL / Brier vs soft gold):

    | System | Acc | KL | Brier |
    |---|---|---|---|
    | Verdict-MQ 395M | 0.801 ± 0.005 | 0.080 | 0.045 |
    | Verdict-MQ 150M | 0.766 | 0.103 | 0.057 |
    | Laya typed-decisions 421M | 0.767 | 0.117 | 0.061 |
    | Verdict-CE, per-question cross-encoder, 150M | 0.586 | 0.245 | 0.141 |

    - KL beats Laya on all 5 seeds (p ≤ 0.0018).
    - At a 90% autonomy target, Laya covers 55.7% of decisions at 86.5% accuracy; Verdict-MQ large covers 62-71% at 89.4-92.2%.
  - **Standard benchmarks** (fine-tuned Verdict-MQ vs "a baseline classifier with the same initialization and epoch budget"; the baseline's head and size are not specified):

    | Benchmark | Verdict-MQ | Baseline |
    |---|---|---|
    | RACE | 77.4 ± 0.4 | 78.9 ± 0.1 |
    | BoolQ (unstable; seeds 72.3-79.4) | 76.2 ± 3.1 | 81.0 ± 0.6 |
    | MultiRC | 80.6 ± 0.5 | 71.6 ± 4.9 |

  - **Question-aware state copies** (kk = number of layers in which each question gets its own copy of the state, kk = 22 = full depth of ModernBERT-base):
    - RACE: 76.2 → 77.4 at 1.4x latency (12.1 → 17.2 ms).
    - MultiRC: 78.4 → 80.6 at 2.5x latency (20.6 → 50.9 ms).
    - Typed decisions: no measurable gain.
    - Seed-0 grid: MultiRC with [CLS] marker at kk = 0 gives 20.6 dev vs 50.9 at kk = 22, so shallow question attention hurts most on multi-question passages.
  - **Latency** (ms per case, RTX 4090 Laptop, bf16; packed vs per-question passes):

    | Model, batch | Packed | Per-question | Speed-up |
    |---|---|---|---|
    | Base, batch 1 | 9.6 | 12.5 | 1.3x |
    | Base, batch 8 | 5.8 | 15.4 | 2.7x |
    | Large, batch 1 | 13.4 | 24.5 | 1.8x |
    | Large, batch 8 | 11.4 | 32.0 | 2.8x |

    The paper's cost formula: one pass over |s| + Σ|b_k| tokens replaces K passes over |s| + |b_k| tokens.
  - **Invariance:** with a random head in fp32, permutation, solo-vs-packed and padding change probabilities by at most 5.4e-7 (3.9e-3 worst case in bf16). On the trained model, permutation never changes the chosen option.
  - **Pitfalls reported:**
    - Right truncation deletes options on about 20% of long email inputs, so options must come first or be protected.
    - A fixed option order lets a model learn slot positions: accuracy 0.586 → 0.349 under permutation.
    - A model that outputs uniform distributions is trivially invariant, so invariance tests must also check that it learns.
  - **Setup:** trained on one 16 GB RTX 4090 Laptop, batch 4 cases (large) / 8 (base), 6 epochs. Memory not reported. Scaling with option count or option length is not reported.
- Set-Encoder (Schlatt et al., arXiv 2404.06912, Apr 2024 – v5 Apr 2025; ECIR 2025) — [arXiv HTML](https://arxiv.org/html/2404.06912v5):
  - Design: one sequence per passage, `[CLS] [INT] query passage`. Positions restart at 0 in each sequence. Each sequence attends to its own K/V plus every other sequence's [INT] K/V. ELECTRA-base 110M / large 330M; query 32 tokens, passage 256.
  - nDCG@10 on TREC DL19 / DL20 (re-ranking 100 BM25 candidates), Set-Encoder 330M vs monoELECTRA 330M (pointwise cross-encoder): 0.727 / 0.735 vs 0.733 / 0.727. Re-ranking ColBERTv2 candidates: 0.789 / 0.790 vs 0.765 / 0.799. No significant differences. TIREx geometric mean 0.321 vs 0.318.
  - Cost for 100 passages: 0.219 s / 2.60 GB vs monoELECTRA 0.215 s / 2.69 GB, and RankZephyr-7B 24.0 s / 15.5 GB.
  - Order invariant by construction, whereas RankGPT-4o and LiT5 "lose a substantial portion" of effectiveness on inverse ordering. Novelty-aware α-nDCG@10 0.821 vs monoELECTRA 0.785.
- Block-Attention (Sun, Wang, Lan; ICLR 2025, arXiv 2409.15355), causal LLMs:
  - Retrieved passages are encoded as independent blocks (block-diagonal mask) with position re-encoding.
  - Without fine-tuning, Llama3-8B average RAG accuracy drops from 67.9% to 48.0%. After about 100-1,000 fine-tuning steps it reaches 68.4%, i.e. parity.
  - At 32K tokens, time-to-first-token and FLOPs drop by 98.7% and 99.8%.
  - [arXiv HTML v4](https://arxiv.org/html/2409.15355v4); numbers via search summary, and they vary by arXiv version.

### Inferences
- **Prefix sharing / "prompt caching" for encoders is exact once state attends only to state** (Verdict rule 1). The state's K/V at every layer can then be computed once and reused for every option, so the packed pass and a "state pass + option pass with per-layer cross-attention to cached state K/V" are mathematically the same computation. This is essentially LIDM pass A + a pass B that sees pass A's per-layer keys and values. Uni-Encoder's Arrow mask, where the context sees the candidates, does not have this property.
- **Shared position IDs plus the isolation mask** is the established recipe for permutation invariance (Uni-Encoder +2.2 R10@1 over unique IDs; Verdict neutral). Shared positions have a side benefit: the position range stays at |s| + max|o| rather than |s| + Σ|o|. In principle this lets Σ|o| exceed the 8,192 RoPE range, with memory and attention cost still growing with Σ|o|.
- **ModernBERT caveat:** in local (128-token window) layers, an option positioned right after the state only sees roughly the last 128 positions of the state. Only the global layers (every third) give options a view of the whole state. Verdict handles this by defining the window on position IDs. A reimplementation must do the same, or local layers will leak across options or cut options off from the state.
- **Fine-tuning with the mask is mandatory:** Block-Attention loses 20 points zero-shot and recovers fully with training, and UniMC without multiple-choice training is at chance.

### Gaps
- Verdict-MQ reports no memory use, no results with more than 4-5 options per question, and no option-length sweep, and its 48-token option cap is untested on long descriptions. Its RACE/BoolQ baseline is under-specified.
- No study found measures how many candidates (or total tokens) a masked packed encoder handles before accuracy drops. Uni-Encoder only measured speed up to 100 candidates.

## Q5. Long-context packing: 100+ labels or long candidates in one pass

### Takeaway
The strongest evidence that packing scales to many long candidates comes from listwise rerankers, not classifiers. jina-reranker-v3 (Qwen3-0.6B, causal) packs up to 64 documents of up to 8k tokens into a 131k window. It scores 61.85 BEIR nDCG@10, against 56.28 for the pointwise Qwen3-Reranker-0.6B built on the same backbone, and it is nearly order-insensitive. Encoder label packers (GLiClass 1,024 tokens, GLiNER2 2,048) are capped well below 100 long descriptions, and GLiClass reports degradation at Banking77 scale.

### Cited Findings
- jina-reranker-v3 ("Last but Not Late Interaction", arXiv 2509.25085, v4 6 Oct 2025) — [arXiv HTML](https://arxiv.org/html/2509.25085v4):
  - Design: Qwen3-0.6B (28 layers, 1,024 wide), causal attention. Documents are wrapped in `<passage id=n>` with a `<|doc_emb|>` token after each; the query appears at the start and again at the end with `<|query_emb|>`. Scoring is cosine between query and document embeddings.
  - Limits: up to 64 documents per pass, 131,072-token context, documents truncated at 768 tokens (stage 1) and up to 8,192 (stage 2).
  - BEIR nDCG@10: v3 61.85; Qwen3-Reranker-0.6B (pointwise, same backbone family) 56.28; jina-reranker-v2 (0.3B, pointwise) 57.06; bge-reranker-v2-m3 (0.6B) 56.51; mxbai-rerank-large-v2 (1.5B) 61.44; Qwen3-Reranker-4B 61.16.
  - Order sensitivity on BEIR: descending relevance 61.85, ascending 61.45, random 62.24 (within 0.8). No latency numbers.
- GLiClass: 1,024-token training cap. 1,000+ labels need chunking. GLiClass-large Banking77 (77 short labels) 0.5574 vs NLI cross-encoder 0.5670 — [arXiv HTML](https://arxiv.org/html/2508.07662). The GLiClass README's long-text pipeline chunks text into 8,192-token pieces and labels into groups of 8 (`labels_chunk_size=8`), i.e. multiple passes rather than one — [GitHub](https://github.com/Knowledgator/GLiClass)
- GLiNER2: 2,048-token context, Banking77 zero-shot 0.70 with label names only — [arXiv HTML](https://arxiv.org/html/2507.18546)
- BTZSC benchmark (Aarab, arXiv 2603.11991, Mar 2026; ICLR 2026): 22 zero-shot classification datasets, 38 checkpoints. Qwen3-Reranker-8B is top at 0.72 macro-F1, LLMs reach up to 0.67, embedding bi-encoders (GTE-large-en-v1.5) come close at the best accuracy/latency trade-off, and NLI cross-encoders plateau as size grows. GLiClass-style packed models are not mentioned — [arXiv abs](https://arxiv.org/abs/2603.11991)

### Inferences
- Banking77 with 77 descriptions at about 40 tokens each is about 3.1k option tokens. That fits ModernBERT's 8,192 window in one masked pass with no per-option budget. Laya's 192/512 budget is a design choice, not a backbone limit.
- jina-v3 shows packed listwise models can beat pointwise ones on the same backbone (+5.6 nDCG@10). It is causal and lets documents see earlier documents, so candidates are not isolated. It has not been shown for a bidirectional encoder with an isolation mask on classification.

### Gaps
- No encoder classifier was found that packs 100+ labels with descriptions into one pass and reports accuracy vs label count.
- No ModernBERT-based "in-context label" classifier with a 4k-8k label budget was found.

## Q6. Compute and memory: one packed pass vs N option passes (+ cross-attention head) on 12 GB

### Takeaway
On ModernBERT-large, at sequence lengths up to a few thousand tokens, cost is dominated by the linear layers and grows roughly with total tokens. A masked packed pass therefore costs about the same as LIDM's pass A + pass B (state once + each option once). It is much cheaper than per-option cross-encoding when the state is long (RACE, about 3.6x) and only slightly cheaper when the state is short (Banking77 utterances, about 1.5x). Measured: packing gives 1.8-2.8x lower latency than per-question passes on ModernBERT-large. GLiClass gives flat throughput to 128 labels while NLI cross-encoders slow down 67x. Poly-encoder data shows per-candidate cross-encoding is 100-1,000x slower than a bi-encoder at 1k candidates.

### Cited Findings
- ModernBERT-large: 28 layers, 1,024 hidden, 16 heads, 395M params. Global attention every third layer, local 128-token sliding window otherwise. RoPE θ 160k global / 10k local. Native 8,192 context. Unpadding with Flash-Attention varlen ("jagged attention masks"). On an RTX 4090 (24 GB), inference max batch 48 at 8,192 tokens and 770 at 512, with about 46.8k tokens/s at 8,192 and 52.3k at 512 — [arXiv HTML](https://arxiv.org/html/2412.13663)
- Verdict-MQ latency on ModernBERT-large (RTX 4090 Laptop, bf16): packed 11.4 ms/case vs per-question 32.0 ms/case at batch 8 (2.8x). Question-aware copies add 1.4-2.5x latency. Trained with batch 4 on 16 GB — [arXiv HTML](https://arxiv.org/html/2610.20778)
- Poly-encoder (Humeau et al., ICLR 2020), ConvAI2 — [ar5iv](https://ar5iv.labs.arxiv.org/html/1905.01969):
  - Inference, ms per prediction:

    | Model | GPU, 1k candidates | GPU, 100k candidates | CPU, 1k candidates |
    |---|---|---|---|
    | Bi-encoder | 19 | 22 | 115 |
    | Poly-360 | 57 | 88 | 160 |
    | Cross-encoder | 2,600 | 266k (inferred) | 21,700 |

  - Accuracy, Ubuntu V2 R@1/10, BERT-base: bi 80.6, poly 81.2-81.3, cross 82.8. Reddit-pretrained: 83.6 / 86.0 / 86.5.
  - ConvAI2 R@1/20, BERT-base: bi 81.7, poly 83.7, cross 84.8. Reddit-pretrained: 84.8 / 86.8 / 87.9.
  - DSTC7 R@1/100, BERT-base: bi 66.8, poly-360 68.9, cross 67.4.
  - Bi-encoder accuracy rises with in-batch negatives: 31 → 511 negatives gives 81.0 → 83.3 (ConvAI2 validation R@1/20). The cross-encoder batch is memory-limited to 16.
- GLiClass vs NLI cross-encoder throughput, 1 → 128 labels: gliclass-modern-large 46.3 → 32.9 ex/s; deberta-v3-large NLI 16.82 → 0.25 ex/s (A6000) — [arXiv HTML](https://arxiv.org/html/2508.07662)
- Set-Encoder: inter-candidate [INT] attention costs 0.219 s vs 0.215 s for 100 passages (+2%) and 2.60 GB — [arXiv HTML](https://arxiv.org/html/2404.06912v5)

### Inferences (my arithmetic; assumptions stated)
- **Per-token cost:** ModernBERT-large has about 340M non-embedding parameters (395M minus about 52M for a ~50k-token vocabulary at 1,024 wide; assumed), so linear layers cost about 0.7 GFLOP per token forward. Attention scores cost about 4·L·d FLOPs per token per layer:
  - In about 10 global layers at L = 2,048, that is about 84 MFLOP/token (12% of the linear cost).
  - At L = 8,192 it is about 336 MFLOP/token (48%).
  - The 18 local layers add about 9 MFLOP/token.
  - So up to about 4k tokens, cost ≈ total tokens. Packed ≈ |s| + Σ|o_i|, LIDM ≈ (|s| + |q|) + Σ|o_i| (the same), and per-option cross ≈ N·(|s| + |o|).
- **Worked examples** (token counts are rough guesses, not measured):

  | Case | State | Options | Packed / LIDM | Per-option cross-encoder | Ratio |
  |---|---|---|---|---|---|
  | RACE | ≈350 tokens | 4 × ≈12 tokens | ≈400 tokens | ≈1,450 tokens | 3.6x |
  | Banking77 + descriptions | ≈20 tokens | 77 × ≈40 tokens | ≈3,100 tokens | ≈4,600 tokens | 1.5x |
  | Banking77 names | ≈20 tokens | 77 × ≈6 tokens | ≈480 tokens | ≈2,000 tokens | 4x |

- **Memory on 12 GB:**
  - Inference: by the ModernBERT 4090 figure (about 393k tokens in flight on 24 GB), roughly half fits in 12 GB, so a 3-8k-token packed case is trivial.
  - Training: Verdict trained ModernBERT-large with 4 × ≤2,048-token cases on 16 GB, so on 12 GB, batch 2-4 of 2k-4k-token packed cases with gradient checkpointing and bf16 is plausible but unverified.
  - The risk is the attention kernel. A custom (non-block-diagonal) mask cannot use Flash-Attention varlen. If SDPA falls back to the math kernel, it materialises 16 × L² bf16 scores, about 0.5 GB per layer at L = 4,096 (about 15 GB if kept for backward across 28 layers). It needs a memory-efficient masked kernel (FlexAttention / SDPA mem-efficient with attn_mask) or the cached-state formulation from Q4.
- **Cached-state formulation:** run state + question once, then a batch of options whose keys and values = [state K/V at that layer ; own K/V]. Attention cost per option is then (|s| + |o|)·|o| with no L² term, and plain batched SDPA works with a small padding mask. This is a natural "pass B v2" for LIDM.

### Gaps
- No measured 12 GB numbers for ModernBERT-large packed training. No measured FlexAttention/SDPA throughput for arrow or structured masks on an RTX 4070 Super.
- Uni-Encoder's per-pool timings exist only in a figure.

## Q7. Could packing replace or complement LIDM's pass B?

### Takeaway
The evidence favours replacing (or augmenting) pass B with options that attend to the state at every layer: Verdict-style rules 1-4, or the equivalent cached-state K/V option pass. This keeps LIDM's state-once cost and permutation invariance, adds token-level option→state interaction, and removes Laya's option-token budget. The closest data point: on ModernBERT-large, a structured-mask packed model beats Laya on typed decisions and comes within 1.5 RACE points of a matched baseline. LIDM's own losses (VitaminC −7, RACE −5) look like missing option-state interaction, which this design targets directly. For many short labels, packing alone does not fix the crowding problem GLiClass reports, so the isolation mask and shared positions still matter.

### Cited Findings
- Same backbone, same competitor: Verdict-MQ ModernBERT-large packed with structured mask vs Laya 421M on typed decisions, 0.801 vs 0.767 accuracy and KL 0.080 vs 0.117. The per-question cross-encoder variant (150M) scored 0.586 — [arXiv HTML](https://arxiv.org/html/2610.20778)
- The interaction depth matters for fine-detail reading: Verdict MultiRC kk = 0 → 22 gives 78.4 → 80.6, and RACE 76.2 → 77.4 — [arXiv HTML](https://arxiv.org/html/2610.20778)
- Candidate isolation is required: Uni-Encoder Square −3.3 R10@1 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2106.01263)); UniMC without the mask 39.1 vs 78.6 ([ar5iv](https://ar5iv.labs.arxiv.org/html/2210.08590)); Verdict collapse when options see each other ([arXiv HTML](https://arxiv.org/html/2610.20778))
- Packed label models degrade when label tokens outnumber text, and only tie a per-label cross-encoder on Banking77 — [GLiClass](https://arxiv.org/html/2508.07662)
- Bi-encoders stay competitive at scale and are far cheaper with many candidates: GLiNER bi ≥ uni on NER F1 with about 130x throughput at 1,024 labels ([arXiv 2602.18487](https://arxiv.org/html/2602.18487)); poly/bi within 1.5-2.2 R@1 of cross-encoders on Ubuntu V2 ([ar5iv](https://ar5iv.labs.arxiv.org/html/1905.01969))

### Inferences
- **Mapping to LIDM's results:**
  - VitaminC −7 and RACE −5 (fine-detail checking): options need token-level access to the state. Verdict rule 3 / the cached-state option pass provides it at all layers, at near-zero extra token cost.
  - Banking77 names −11 (77 short options): LIDM's pass B encodes each name with no context. A masked option pass lets each short name read the utterance. With isolation plus shared positions it also avoids GLiClass-style label crowding. A cheap complement is to keep the current pass B option vectors as a bi-encoder prior and add the state-aware option markers as a second score, then fuse.
  - Banking77 descriptions +11 (long options): keep the no-budget property. With shared position IDs, 77 × 40-token descriptions add about 3.1k tokens of compute but only about 40 positions of range.
- **Risks:**
  - ModernBERT local layers only let an option see the about 128 positions nearest to it, so the window must be defined on position IDs as Verdict does.
  - Training must use the mask from the start (Block-Attention −20 points without it).
  - Fixed option order teaches slot positions, so randomise or make the model invariant by construction.
  - Options that share position IDs must never attend to each other.
- **Lowest-effort experiment:** keep pass A as is (state + question attending only to themselves), and change pass B so that each option's self-attention at layer ℓ also attends to pass A's layer-ℓ K/V. This is exactly Verdict-equivalent for a single question and needs no L×L mask.

### Gaps
- No published result for a masked packed encoder on VitaminC or Banking77 (77 labels with or without descriptions) on ModernBERT-large. This is the decisive comparison and needs a local run.
- Verdict-MQ is a single, recent, non-peer-reviewed preprint whose baselines ("matched classifier", "Jev") are under-specified. Its Laya comparison should be replicated before it is relied on.
