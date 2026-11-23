# Decomposed / deferred-interaction transformers (separate lower layers, joint upper layers)

Notation used throughout. "Separate" means lower layers where input A and input B cannot attend to each other. "Joint" means upper layers with full or partial cross-attention. "k joint of L" means the top k of L layers are joint. Unless stated otherwise, every accuracy number compares the decomposed model with the full cross-encoder on the same backbone, task and training data, as reported by the paper's authors.

## Q1. DeFormer (Cao et al., ACL 2020): split, accuracy loss on SQuAD/RACE/BoolQ, speed-ups, distillation

### Takeaway
DeFormer on BERT-base keeps 3 of 12 layers joint (9 separate, passage cached) and loses 1.0-1.8 points on SQuAD, RACE, BoolQ and MNLI, with about 3.2-3.5x faster QA inference. RACE, the task closest to LIDM's multiple-choice setting, loses 1.8 accuracy points (66.3 to 64.5). The two distillation losses recover 1.3 F1 (base) and 3.3 F1 (large) on SQuAD and are essential for large models.

### Cited Findings
- **Split.** For BERT-base (12 layers), the 9 lower layers are decomposed and the 3 upper layers are joint, with lower-layer passage outputs cached. For XLNet-base, 8 lower layers are separate and 4 upper layers joint. The cached segment is the passage (QA), the premise (NLI), or the first question (QQP). — [DeFormer, Table 1 caption and Sec. 3](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Accuracy, BERT-base, full vs DeFormer (3 joint of 12):**
  - SQuAD F1: 88.5 vs 87.1 (−1.4)
  - RACE accuracy: 66.3 vs 64.5 (−1.8, −2.7% relative)
  - BoolQ accuracy: 77.8 vs 76.8 (−1.0)
  - MNLI: 84.4 vs 82.6 (−1.8)
  - QQP: 90.5 vs 90.3 (−0.2)

  — [DeFormer, Tables 1-2](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Accuracy, XLNet-base (4 joint of 12):** SQuAD 91.6 vs 90.4, RACE 70.3 vs 68.7 (−1.6), BoolQ 80.4 vs 78.8. — [DeFormer, Table 1](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **BERT-large (24 layers).** SQuAD F1 is 92.3 for the full model vs 90.8 for DeFormer. The split used for that main result is not stated in the text. Fig. 4 says the F1 drop is negligible when separating up to layer 3 (base) and up to layer 13 (large, i.e. 11 of 24 joint). The drop then grows slowly and "jumps sharply" in the last layers. This curve is reported for SQuAD only. — [DeFormer, Table 3, Fig. 4](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Speed-up (BERT-base, Table 1).** 3.2x on SQuAD, 3.4x on RACE, 3.5x on BoolQ. XLNet-base gets 2.7x, 2.8x and 3.0x. MNLI gets 2.2x and QQP 2.0x, because the cached segment is shorter. Memory reduction is 70.3%, 72.9% and 72.0% for SQuAD, RACE and BoolQ. — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Latency, BERT-base vs DeFormer-BERT-base:**
  - V100, batch 32: 0.22 s vs 0.07 s
  - i9-7900X CPU, batch 32: 5.90 s vs 1.66 s
  - OnePlus 6 phone, batch 1: 10.20 s vs 3.28 s

  For SQuAD FLOPs, BERT-large uses 204.1 GFLOPs and DeFormer-BERT-large 47.7 GFLOPs (BERT-base: 58.4). — [DeFormer, Tables 3-4](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Internal inconsistency.** The abstract claims a speed-up of "over 4.3x" (the intro says 2.7-4.3x), but the tables top out at 3.5x. — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Storage.** 226 KB per 150-token passage in bfloat16, or 226 GB per 1M passages. Speed-up scales roughly quadratically with the number of decomposed layers. — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Distillation, SQuAD F1.** The two auxiliary losses are knowledge distillation from the full model (KD) and layerwise representation similarity (LRS).
  - Base: 87.1 with both; 86.2 without LRS; 85.8 without KD and LRS.
  - Large: 90.8; 88.9; 87.5.

  So the losses are worth +1.3 F1 for base and +3.3 F1 for large. They were used only for SQuAD and RACE; the other tasks used the main loss alone. No RACE-specific ablation is reported. — [DeFormer, Table 5](https://ar5iv.labs.arxiv.org/html/2005.00697)

### Inferences
- LIDM currently has 0 joint backbone layers plus a 2-block fusion head and loses about 5 points on RACE vs Laya. DeFormer loses 1.8 on RACE with 3 of 12 backbone layers joint and distillation. This is consistent with joint *backbone* layers (pretrained and initialized from the original top layers) recovering most of the gap.
- Distillation from the joint model (Laya is already trained on the same backbone and data) is a cheap add-on for LIDM. DeFormer's gain from distillation grows with model size (+3.3 F1 on the 24-layer model). LIDM's 28-layer ModernBERT-large is closer to the large case.
- DeFormer's speed-up comes from caching the passage *across questions*. In LIDM the state changes per example, so caching the state brings no speed-up. Only fixed option sets (label names or descriptions) are cacheable (see Q7).

### Gaps
- No RACE accuracy as a function of the split layer k, and no RACE KD/LRS ablation, appear in the paper text. Only a SQuAD curve is given, as a figure without numbers.
- The BERT-large split used for the 90.8 F1 result was not stated in the fetched text.

## Q2. PreTTR (MacAvaney et al., SIGIR 2020): term representations precomputed to layer l, ranking quality vs l, storage compression

### Takeaway
On BERT-base (12 layers) re-ranking, the joint top layers can shrink to about 2-5 of 12 with no statistically significant loss on Robust04. With 1 joint layer (l=11), quality collapses on Robust04 (nDCG@20 0.436 to 0.259) and partly on WebTrack. Speed-up is modest until very few joint layers remain: 2.1x at l=6, 8.9x at l=10, 42x at l=11. A learned compression to 256 of 768 dimensions is roughly lossless.

### Cited Findings
- **Mechanism.** Backbone is bert-base-uncased (12 layers). Attention is masked in layers 1..l so that query and document tokens cannot see each other; layers l+1..12 are full joint. At index time, document tokens pass through layers 1..l and the layer-l token vectors are stored (768 floats per token). At query time, the query runs through layers 1..l, is concatenated with the stored document vectors, and layers l+1..12 run jointly. — [PreTTR](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **Ranking quality vs l** (WebTrack 2012 P@20 / ERR@20; Robust04 P@20 / nDCG@20; ↓ marks a significant drop):

  | l (separate layers) | WebTrack P@20 | WebTrack ERR@20 | Robust04 P@20 | Robust04 nDCG@20 |
  |---|---|---|---|---|
  | Base (full) | 0.3460 | 0.2767 | 0.3784 | 0.4357 |
  | 3 | 0.3440 | 0.2268 | 0.3859 | 0.4386 |
  | 6 | 0.3270 | 0.2563 | 0.3663 | 0.4156 |
  | 8 | 0.3140 | 0.2344 | 0.3636 | ↓0.4123 |
  | 10 | 0.3360 | 0.2295 | 0.3579 | ↓0.4039 |
  | 11 | 0.3380 | ↓0.1940 | ↓0.2534 | ↓0.2590 |

  At l=11, Robust04 falls "well below the BM25 baseline". — [PreTTR, Table 3](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **Speed-up** (re-ranking the top 100 documents on a GTX 1080 Ti, WebTrack): 1.4x at l=3, 2.1x at l=6, 3.5x at l=8, 5.0x at l=9, 8.9x at l=10, 42.2x at l=11. Robust04 reaches 9.2x at l=10. — [PreTTR, Table 5](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **Compression.** A learned GELU projection maps vectors to e dims and back to 768. It is pre-trained on Wikipedia pairs with an MSE loss on attention scores, then fine-tuned. Representations "can usually be compressed down to at least e=256 (67%)" without substantial loss; at e=128, Robust04 drops sharply for l=7-10. — [PreTTR, Table 4](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **Storage.** ClueWeb09-B needs about 112 TB uncompressed, about 34 TB after spam filtering, 5.7 TB at e=128, and about 2.8 TB at e=128 in fp16 ("about 2.5% of original"). Robust04 needs about 195 GB at e=256. The abstract says up to 95% reduction; the conclusion says 97.5%. — [PreTTR](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **SDR (Cohen et al., arXiv Oct 2021), a PreTTR-style follow-up.** "BERTSPLIT" has 10 separate layers and 2 joint layers. It is initialized from a MiniLM-L-12 cross-encoder (hidden size 384) and distilled from an ensemble. Its MS MARCO dev MRR@10 is 0.3768, and TREC DL'19 nDCG@10 is 0.774. — [SDR](https://ar5iv.labs.arxiv.org/html/2110.02065)
- **SDR compression.** An autoencoder that uses static token embeddings as side information, plus DRIVE quantization:
  - AESI-16-6b: 121x compression, MRR@10 0.3753 (−0.0015, not significant), 947 bytes per passage, all of MS MARCO in 8.6 GB.
  - AESI-8-5b: 277x, MRR@10 0.3649 (significant drop), 487 bytes per passage.
  - The baseline autoencoder in fp16 reaches only about 30x.

  SDR does not report a full cross-encoder number, so the cost of the split itself is not isolated. — [SDR, Table 1](https://ar5iv.labs.arxiv.org/html/2110.02065)

### Inferences
- In PreTTR's ranking task, losses stay small until the last 1-2 layers become separate. For LIDM, this suggests keeping at least 2-4 of 28 ModernBERT layers joint (by layer-fraction analogy: 2/12 ≈ 5/28). That is an extrapolation across tasks and backbones, not a measured number.
- PreTTR and SDR show that cached hidden states are highly compressible (3x lossless; 100x+ with learned codecs). This is irrelevant at LIDM's scale: the caches are megabytes (see Q7).

### Gaps
- PreTTR has no MS MARCO and no explicit l=0 run; its "Base" row is the reference model. No per-document storage number is given.

## Q3. MORES / EARL / MORES+ (Gao, Dai, Callan 2020; Gao & Callan 2022): separate document representation and query-to-document interaction blocks

### Takeaway
MORES separates the ranker into a full-depth document encoder (cacheable), a query encoder of 12−l layers, and l interaction blocks (IBs) in which the query cross-attends to the fixed document representation and then self-attends. With BERT-base, 2 IBs match the BERT cross-encoder within 2% on MS MARCO (0.3456 vs 0.3527 MRR@10), 1 IB reaches about 95%, and 3-4 IBs do not help. Initializing the IBs by copying BERT's top layers is decisive: random initialization drops MRR@10 from 0.3456 to 0.2723. EARL is the April 2020 v1 title of the same arXiv paper. MORES+ extends the idea to long documents with a BART backbone.

### Cited Findings
- **EARL = MORES v1.** arXiv 2004.13313v1 is titled "EARL: Speedup Transformer-based Rankers with Pre-computed Representation". It describes query and document representations computed offline plus a "judger" transformer block for shallow relevance scoring. — [arXiv 2004.13313v1](https://arxiv.org/abs/2004.13313v1)
- **Architecture.**
  - The Document Representation Module self-attends over the document only and is initialized from a full BERT copy.
  - The Query Representation Module has 12−l layers, initialized from BERT's first 12−l layers.
  - Each of the l IBs does query-to-document cross-attention, then query self-attention, then an FFN. The document representation D stays *fixed* across IBs.
  - The score is a linear layer on the [CLS] output of the last IB.
  - Parameter counts are 224M-233M for 1-4 IBs, i.e. about 2x BERT-base, because there are two encoders.

  — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **Caching.** Reuse-S1 caches the document module's output D. Reuse-S2 additionally caches the projected K/V matrices of D for each IB (2l matrices), trading storage for time with no accuracy change. — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **Accuracy** (MS MARCO passage, re-ranking the BM25 top 1000; Dev MRR@10 / TREC DL'19 NDCG@10):

  | Model | Dev MRR@10 | TREC DL'19 NDCG@10 |
  |---|---|---|
  | BERT-base cross-encoder | 0.3527 | 0.7032 |
  | 1 IB | 0.3334 | 0.6721 |
  | 2 IB | 0.3456 | 0.7026 |
  | 3 IB | 0.3423 | 0.6980 |
  | 4 IB | 0.3307 | 0.6565 |

  The authors say 2 IBs are comparable to BERT within a 2% margin, 1 IB reaches about 95%, 3 IBs do not improve, and 4 IBs hurt. — [MORES, Table 2](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **Initialization** (2 IB): copying from BERT gives Dev MRR@10 0.3456 and TREC NDCG@10 0.7026; random initialization gives 0.2723 and 0.6059. The authors attribute the gap to optimization difficulty. The query-to-document cross-attention weights are initialized from BERT's self-attention weights. — [MORES, Table 3](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **Speed** (rank 1000 candidates, query length 16; Xeon 4110 CPU / RTX 2080 Ti GPU):
  - Reuse-S1, 128-token documents: 1 IB is 40x (CPU) / 61x (GPU) faster; 2 IB is 20x / 22x.
  - Reuse-S1, 512-token documents: 1 IB 66x / 91x; 2 IB 35x / 40x.
  - Reuse-S2, 512-token documents: 1 IB 170x / 158x; 2 IB 124x / 124x.
  - Storage per document (uncompressed, fp32 implied by 512×768×4 B = 1.5 MB): Reuse-S1 0.4 MB (128 tokens) or 1.5 MB (512 tokens); Reuse-S2 1.5-12 MB.

  — [MORES, Table 4](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **Domain adaptation** (NDCG@20, title queries):
  - ClueWeb09: BERT 0.3294 vs 2 IB 0.3317.
  - Robust04: BERT 0.4632 vs 2 IB 0.4599.
  - Adapting only the representation modules matches full adaptation (ClueWeb 0.3319). Adapting only the interaction module is worse (0.3179).

  The authors read this as representations needing domain tuning more than interaction does. — [MORES, Domain adaptation tables](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **MORES+ (SIGIR 2022, long documents).**
  - Each 512-token chunk is encoded independently by a 12-layer encoder taken from BART's encoder, and the chunk representations are cached offline.
  - A 12-layer interaction module (BART's decoder) does a "single joint query-to-all-chunk cross attention", so the query attends across chunk boundaries. The query encoder is reduced to an embedding layer.
  - Robust04 nDCG@20 (title/description): 0.556 / 0.612, vs BERT-maxP 0.469 / 0.529 and PARADE-Transformer (RM3) 0.566 / 0.613. MORES+ (RM3) scores 0.569 / 0.641.
  - ClueWeb09: 0.353 / 0.329. MS MARCO document Dev MRR@100: 0.493 (single model), 0.501 (with coCondenser).
  - No measured latency; the paper claims "tens of times" speed-up and leaves efficiency to future work. — [MORES+](https://ar5iv.labs.arxiv.org/html/2205.04275)

### Inferences
- LIDM's fusion head is structurally a MORES Interaction Module: options play the query role and cross-attend to pass A's (state) tokens, which stay fixed. The MORES evidence suggests two concrete changes:
  1. Build the fusion blocks by *copying ModernBERT's top layers* and initializing the cross-attention from self-attention weights, instead of training a small head from scratch. In MORES, random initialization cost about 23% relative MRR.
  2. Feed the options from layer 28−k into k such blocks (MORES's 12−l / l split) rather than from layer 28 into 2 extra blocks.

  Both are inferences; MORES was measured on ranking with BERT-base, not on MC, NLI or classification.
- MORES+ shows that a "one query attends jointly to all cached chunks" design works with a full-depth interaction module. A LIDM analogue would be state-and-question tokens attending to all 77 cached option encodings jointly, which is permutation-insensitive and avoids truncation.

### Gaps
- MORES layer counts were measured only for ranking. There are no MORES results on QA, MC or NLI.
- MORES+ reports no ablation on the number of interaction layers.

## Q4. LUMEN, GLIMMER, MEMORY-VQ (de Jong et al., 2023; Zemlyanskiy et al., 2023): precomputed memory plus a small live encoder conditioned on the question

### Takeaway
LUMEN (T5.1.1) precomputes the first 1−α encoder layers per passage and runs the last α layers "live" on [question; passage memory]. At T5-XXL, α = 1/3 matches Fusion-in-Decoder (FiD): NQ 57.1 vs 57.3 EM, TriviaQA 73.1 vs 73.0. The live fraction needed *shrinks with model scale*. Conditioning the passage on the question in the live layers is the critical ingredient. MEMORY-VQ compresses the memory 16x for about a 0.2-point loss.

### Cited Findings
- **Architecture.**
  - The T5 encoder is split: the memory encoder (first 1−α layers) runs offline over passages; the live encoder (last α layers) is fine-tuned and run on Hi = [QEnc(Q); MemEnc(Passage_i)].
  - The question encoder is separate, initialized like the memory encoder, and fine-tuned.
  - α = 0 gives MemoryFiD (no live layers); α = 1 is approximately FiD.
  - Inputs are the top 20 DPR passages; LUMEN truncates question / passage to 48 / 208 tokens.

  — [LUMEN](https://ar5iv.labs.arxiv.org/html/2301.10448)
- **Accuracy.**
  - Test, T5-XXL: FiD 57.3 NQ / 73.0 TQA vs LUMEN 57.1 / 73.1. FiD-Large scores 51.4 / 67.6.
  - Base, α = 1/3: FiD 47.3 / 64.4 vs LUMEN "Base 24-12" 48.9 / 65.4. The label "24-12" is not explained in the extracted text.
  - Fig. 3 caption: "lumen achieves performance close to FiD with fraction of live layers. The required fraction decreases with scale."

  — [LUMEN, Tables 2-3, Fig. 3](https://ar5iv.labs.arxiv.org/html/2301.10448)
- **Compute.** Per-layer encoder FLOPs are FiD 14·n_s·d², MemoryFiD 2·n_s·d², and LUMEN (2+12α)·n_s·d² (decoder 14·n_t·d² in all cases). Latency varies nearly linearly with the number of live layers. — [LUMEN, Table 1, App. B](https://ar5iv.labs.arxiv.org/html/2301.10448)
- **Ablations (LUMEN-Large).**
  - "Neither conditioning memory on input nor fine-tuning memory are sufficient to recover FiD performance"; both matter, with conditioning contributing more.
  - Removing question-to-passage attention ("no q2mem") is the most damaging; passage-to-question attention ("mem2q") still helps.
  - Fine-tuning the question encoder "improves performance significantly".
  - Exact values are given only in figures.

  — [LUMEN, Figs. 8-10](https://ar5iv.labs.arxiv.org/html/2301.10448)
- **GLIMMER (Jun 2023, T5.1.1-Large, α = 1/3).**
  - Adds re-ranking with the first β of the live layers; β = 0.25 means 2 of 8 live layers.
  - Retrieves 25 passages and selects 5. KILT dev average: 69.8 shared reranker, 70.0 separate reranker from T5, 68.7 separate reranker from scratch, 69.0 with frozen memory.
  - A rerank proportion below 0.25 "strongly harms performance".

  — [GLIMMER](https://ar5iv.labs.arxiv.org/html/2306.10231)
- **MEMORY-VQ (Aug 2023, T5-XXL, α = 1/3).**
  - Uses product quantization with a VQ-VAE: 256 subspaces × 65,536 codes, int16 IDs. LUMEN memory drops from 8 KB to 512 B per token (16x). For Wikipedia (~4B tokens) that is 30 TB to 2 TB.
  - Inference cost per sample: FiD 28.0 TFLOPs vs LUMEN 12.5 TFLOPs.
  - KILT validation EM average: FiD 72.80, LUMEN 72.66, LUMEN-VQ 72.42. FEVER (fact verification): FiD 93.47, LUMEN 92.77, VQ 92.18. HotpotQA: 42.09 / 41.09 / 41.00. NQ: 61.47 / 62.64 / 62.74.

  — [MEMORY-VQ, Table 1](https://ar5iv.labs.arxiv.org/html/2308.14903)
- **Storage conflict.** The LUMEN paper cites about 16 TB for MemoryFiD-XXL on Wikipedia ([LUMEN](https://ar5iv.labs.arxiv.org/html/2301.10448)), while MEMORY-VQ gives 30 TB for LUMEN ([MEMORY-VQ](https://ar5iv.labs.arxiv.org/html/2308.14903)). The two likely count tokens or precision differently; this was not resolved.

### Inferences
- LUMEN's live layers are *top-of-stack, full joint over [question; passage]*: the DeFormer pattern, with a 1/3 live fraction at 11B scale. With 28 layers, 1/3 live would be about 9 joint layers. LUMEN says smaller models need *larger* fractions, so a 395M encoder may need more than 1/3 to match a joint model. This is an extrapolation from T5.
- On fact verification, the closest analogue to VitaminC, LUMEN-XXL at 1/3 live loses only 0.7 EM on FEVER vs FiD. FEVER is a far easier benchmark than VitaminC (contrastive evidence), so this is not a VitaminC proxy.
- LUMEN's finding that *passage-conditioned-on-question* matters most is the direction LIDM lacks. In LIDM, pass A's state tokens never see the option. A MORES-style head only updates the options, so a LUMEN-style result suggests also updating the state tokens per option, at a cost of N× state compute (see Q7).

### Gaps
- Exact EM vs live-fraction curves (Figs. 1, 3, 6-8) are only in plots and were not extractable.
- FiDO was not researched; it is a decoder-side efficiency change, not an encoder split.

## Q5. Other family members: DC-BERT, Decoupled Transformer, Poly-encoder, TwinBERT, cached-prefix rerankers (2024-2026)

### Takeaway
The measured same-backbone points span a range. At one end, 1 joint layer on top of fully separate 12-layer encodings (DC-BERT) loses about 1.5 P@10 on open-domain QA re-ranking. At the other, the Decoupled Transformer split sweep on ELECTRA-base shows extractive QA collapsing as joint layers fall below about 3 of 12. Its NLI and paraphrase losses are 2.2-9.4 points even with 7 of 12 joint. Poly-encoder shows that a single attention layer over cached candidates closes about half the bi- to cross-encoder gap. Recent work (2024-2026) moved to KV-cache reuse in decoder-only rerankers (HyperRAG, 2-3x throughput) rather than new encoder splits.

### Cited Findings
- **DC-BERT (SIGIR 2020, BERT-base re-ranker).**
  - The question is encoded online and the document offline (cached), each through full BERT. K = 1 Transformer interaction layer, initialized from BERT's last layer, sits on top, with global type and position embeddings.
  - SQuAD-Open P@10: 70.1 vs 71.5 for the BERT-base cross-encoder. NQ-Open: 63.5 vs 65.0.
  - Answer EM drops by 2.1% / 2.0% relative, at a 10.3x speed-up.
  - Replacing the interaction Transformer with a linear layer gives 57.3 P@10 on NQ (43.6x faster); an LSTM gives 61.5.
  - More interaction layers "consistently improve" QA but reduce the speed-up; no numbers are given.

  — [DC-BERT, Tables 1, 3](https://ar5iv.labs.arxiv.org/html/2002.12591)
- **Decoupled Transformer (ElFadeel & Peshterliev, 2021; ROaD/ELECTRA-base, 12 layers).**
  - The lower N layers run per passage (cached); the upper M layers run on [question; passage].
  - SQuAD 2.0 F1 by split (separate-joint): full 87.6; 1-11: 87.5; 3-9: 87.1; 5-7: 86.7; 6-6: 85.4; 8-4: 84.8; 9-3: 84.0; 10-2: 80.6; 11-1: 62.6.
  - At the 5-7 split: QQP 91.5 vs 90.9, MNLI 88.9 vs 86.7 (−2.2), MRPC 89.5 vs 80.1 (−9.4).
  - FLOPs at 5-7 are 0.58 of baseline. Latency falls 31% on a Titan RTX (FP16) and 40-46% on CPU.
  - Removing KL distillation costs 2.7 F1 (86.7 to 84.0); removing the representation and attention MSE losses costs only 0.2.
  - 4x compression of the cached states (linear projection to 192 dims) costs 0.3 F1. Wikipedia storage drops from 3.4 TB to 858 GB.

  — [Decoupled Transformer, Tables 1-4, 6](https://ar5iv.labs.arxiv.org/html/2108.02765)
- **Poly-encoder (Humeau et al., ICLR 2020).**
  - Candidates are cached as single vectors. The context is summarized as m codes; one attention step uses the candidate vector as the query.
  - BERT-base weights, test R@1 (bi / Poly-64 / cross): ConvAI2 R@1/20 81.7 / 83.7 / 84.8; Ubuntu v2 R@1/10 80.6 / 81.3 / 82.8; DSTC7 R@1/100 66.8 / 67.0 / 67.4 (Poly-360: 68.9).
  - Reddit-pretrained: ConvAI2 84.8 / 86.5 / 87.9.
  - Latency on ConvAI2, 1k candidates, GPU: bi 19 ms, Poly-64 23 ms, cross 2.6k ms.

  — [Poly-encoder, Tables 4-5](https://ar5iv.labs.arxiv.org/html/1905.01969)
- **TwinBERT (Microsoft Bing Ads, 2020).** Queries and documents are encoded separately, with documents precomputed. Run-time work is query encoding plus a light "crossing" layer. The model is distilled from a BERT cross-encoder, runs in about 20 ms on CPU, and retains most of the fine-tuned BERT-base gain. No same-backbone accuracy delta was extracted. — [TwinBERT](https://arxiv.org/abs/2002.06275)
- **HyperRAG (Apr 2025).** Reuses the document-side KV cache of decoder-only rerankers, with a "2 - 3[x] throughput improvement". The abstract does not isolate the quality effect of caching. — [HyperRAG](https://arxiv.org/abs/2504.02921)
- **Set-Encoder (Apr 2024).** A listwise cross-encoder with permutation-invariant inter-passage attention that processes passages in parallel. It is relevant as a template for options attending to each other, but it does not precompute passage states. Details were not verified beyond the abstract-level summary. — [Set-Encoder](https://arxiv.org/abs/2404.06912)

### Inferences
- The Decoupled Transformer sweep is the cleanest same-backbone "fraction joint" curve for reading-style tasks.
  - The loss is ≤1 F1 with ≥7 of 12 joint and about 3.6 F1 with 3 of 12 (9-3).
  - Below 2 of 12 the model breaks: −7 F1 at 10-2 and −25 F1 at 11-1.
  - Fine-grained pair tasks (MNLI, MRPC) lose more than QA at the same split. This matches LIDM's pattern: VitaminC −7 and RACE −5, while typed decisions tie.
- With Poly-encoder-style shallow interaction, the gap to a cross-encoder stays at 1-1.5 R@1 on response selection. That task is coarse matching, unlike VitaminC.

### Gaps
- "Simplified TinyBERT" with a Siamese-then-cross design, Trans-Encoder (Liu et al., ICLR 2022), and the "CrossBERT (2024)" model cited secondhand in [arXiv 2607.04011](https://arxiv.org/html/2607.04011v1) were not verified. No numbers are reported for them.
- No 2024-2026 *encoder* paper with a lower-separate / upper-joint split and same-backbone MC, NLI or classification numbers was found. The recent activity is in KV-cache reuse for decoder LLM rerankers and RAG.

## Q6. What fraction of joint layers recovers most of the cross-encoder accuracy, and does it depend on the task?

### Takeaway
For ranking and retrieval re-ranking, 1-2 joint layers of 12 (8-17%) recover about 95-98% of quality (MORES 2 IB, DC-BERT, SDR's 2 of 12). For extractive QA, MC and NLI, 3 of 12 joint costs 1-2 points with distillation (DeFormer) or about 3.6 F1 without it at 9-3 (Decoupled). About 7 of 12 joint is needed to get under 1 F1, and fine-grained pair tasks (MNLI, MRPC) still lose 2-9 points at 7 of 12. Larger models tolerate more separation: DeFormer-large shows a negligible drop with 11 of 24 joint, and LUMEN's required live fraction decreases with scale.

### Cited Findings
Measured same-backbone points:
- **MORES, BERT-base, MS MARCO MRR@10:** 1 joint block → 94.5% of cross; 2 → 98.0%. — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
- **PreTTR, BERT-base, Robust04 nDCG@20:** 2 joint of 12 → 92.7% (significant drop); 1 → 59.4%. 5-6 joint (l = 6-7): no significant drop. — [PreTTR](https://ar5iv.labs.arxiv.org/html/2004.14255)
- **DC-BERT, BERT-base, NQ-Open P@10:** 1 joint layer → 97.7% (63.5 / 65.0). — [DC-BERT](https://ar5iv.labs.arxiv.org/html/2002.12591)
- **DeFormer, BERT-base, with KD+LRS:** 3 joint of 12 → RACE 97.3%, SQuAD 98.4%, BoolQ 98.7%, MNLI 97.9%. — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- **Decoupled Transformer, ELECTRA-base, SQuAD 2.0 F1:**
  - 3 joint → 95.9%
  - 4 → 96.8%
  - 7 → 99.0%
  - 2 → 92.0%
  - 1 → 71.5%
  - MNLI at 7 joint → 97.5%; MRPC at 7 joint → 89.5%

  — [Decoupled Transformer](https://ar5iv.labs.arxiv.org/html/2108.02765)
- **LUMEN, T5-XXL, NQ / TQA:** 1/3 live → about 100% of FiD. The fraction needed "decreases with scale". — [LUMEN](https://ar5iv.labs.arxiv.org/html/2301.10448)
- **DeFormer-large, SQuAD:** separating up to layer 13 of 24 (11 joint) gives a negligible drop. — [DeFormer, Fig. 4](https://ar5iv.labs.arxiv.org/html/2005.00697)

### Inferences
- **Task dependence is real.** Relevance ranking (does B relate to A?) survives 1-2 joint layers. Span extraction, MC reading comprehension and NLI or paraphrase (is B exactly entailed by a detail in A?) need about 25-60% of the layers joint to stay within 1-2 points. LIDM's loss profile fits this split: it ties on typed decisions and loses on VitaminC and RACE.
- **For a 28-layer ModernBERT-large:**
  - Translating DeFormer and Decoupled fractions gives about 7 joint layers (25%) for a 1-2 point gap, given distillation from Laya, and about 14-16 joint layers (50-58%) for under 1 point on fine-grained tasks.
  - 2-4 joint layers (MORES-like) are likely enough for Banking77-style label matching, but likely not for VitaminC.
  - These are layer-fraction analogies across backbones, not measurements.
- **Distillation is worth 1.3-3.3 points in every QA study** (DeFormer +1.3 / +3.3 F1; Decoupled +2.7 F1). LIDM has a same-backbone joint teacher (Laya) available.
- **Initialization from the backbone's own top layers matters more than the head design** (MORES: random init loses 0.073 MRR@10).

### Gaps
- No paper measured a split curve on RACE, VitaminC, Banking77 or any many-option classification task.
- No study used ModernBERT or any 2024+ encoder. All points are on BERT-base, ELECTRA-base, XLNet-base, BERT-large or T5.

## Q7. Practical: memory and compute of per-option hidden states at layer l (4-77 options, 12 GB GPU); do joint upper layers need one pass per option?

### Takeaway
Caching per-option hidden states is trivial in memory: about 2 KB per token per cached layer for a 1,024-wide model in bf16, so tens of MB at most. The real cost is the joint layers. If state tokens must be re-encoded conditioned on each option (DeFormer, PreTTR or LUMEN-style full joint), the top k layers process the state N times, and the model becomes strictly more expensive than the single-sequence joint model (Laya) whenever all options fit in one sequence. A MORES-style asymmetric block (options attend to a fixed state) costs only k×(S+ΣO) but gives the state no option-conditioning, which LUMEN found to be the most important direction.

### Cited Findings
- **Reference storage numbers.**
  - DeFormer: 226 KB per 150-token passage, 768 dims, bf16. — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
  - MORES Reuse-S1: 1.5 MB per 512-token document (BERT-base, uncompressed). Reuse-S2, which caches K/V for each interaction block, needs 6-12 MB. — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
  - LUMEN-XXL: 8 KB per token, 4,096 dims. — [MEMORY-VQ](https://ar5iv.labs.arxiv.org/html/2308.14903)
- **Joint-layer cost is linear in the number of live layers** (LUMEN latency), and each layer moved from joint to separate saves about 8% of FLOPs (Decoupled Transformer, 12 layers). — [LUMEN](https://ar5iv.labs.arxiv.org/html/2301.10448); [Decoupled Transformer](https://ar5iv.labs.arxiv.org/html/2108.02765)
- **Speed-up exists only when the cached side is reused.** It comes from offline precompute of the side that repeats across queries (passages or documents). The cross-encoder's cost grows linearly with the number of candidates: 2.6 s for 1k candidates vs 23 ms for Poly-64 on GPU. — [Poly-encoder](https://ar5iv.labs.arxiv.org/html/1905.01969); [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)

### Inferences
These are back-of-envelope calculations from the LIDM figures in the task brief (28 layers, width 1,024, bf16). Let S = state+question tokens, N = number of options, O_i = option lengths, ΣO = total option tokens, L = 28 and k = joint layers. Costs are in token-layers (FFN-dominated; the quadratic attention term is ignored).

- **Cache size.** 1,024 × 2 B = 2 KB per token per layer.
  - Banking77 label names (about 77 × 8 = 616 tokens): about 1.3 MB per cached layer.
  - 77 label descriptions (about 77 × 64 ≈ 4.9k tokens): about 10 MB. Caching K/V for each of k joint layers (MORES Reuse-S2 style) adds 2 × 10 MB × k.
  - These are negligible on 12 GB. They are cacheable only for *fixed* option sets (Banking77 labels, typed decisions). RACE and VitaminC options or claims vary per example, so nothing is reused across examples.
- **Cost per design:**
  - Laya (one joint sequence): L × (S + ΣO).
  - Current LIDM: L × S + L × ΣO plus a 2-block head. About the same token-layers as Laya when nothing is cached; the saving appears only when the option encodings are cached (fixed label sets): L × S + head.
  - DeFormer-per-option (state re-encoded with each option in the top k layers): (L−k) × (S + ΣO) + k × (N × S + ΣO). The extra cost over Laya is k × (N−1) × S, which is positive for every N > 1.
  - MORES-style (options from layer L−k pass through k copied top layers that cross-attend to the fixed state): about L × S + L × ΣO without caching, and L × S + k × ΣO with cached option states at layer L−k.
- **Worked examples (k = 4)**, using rough token counts that are assumptions, not measured:
  - RACE (S ≈ 400, N = 4, O ≈ 12): Laya ≈ 12.5k token-layers; DeFormer-per-option adds 4 × 3 × 400 = 4.8k (+38%).
  - Banking77 names (S ≈ 25, N = 77, O ≈ 6): Laya ≈ 13.6k; DeFormer-per-option adds 4 × 76 × 25 = 7.6k (+56%).
  - Banking77 descriptions (O ≈ 64): one joint sequence would be about 5k tokens, which Laya truncates. DeFormer-per-option runs 77 sequences of about 90 tokens in the top 4 layers, about 27.7k token-layers.

  All of this fits a 12 GB GPU at inference. For training, activations of 77 × 90 ≈ 7k tokens × k layers per example dominate, so gradient checkpointing and a small batch would be needed for k ≥ 4. This is an order-of-magnitude estimate that was not profiled.
- **The direction of interaction decides the cost.**
  - Option-attends-to-state (MORES IB, LIDM's head) needs one pass of the state, so cost does not grow with N on the state side.
  - State-attends-to-option (LUMEN's q2mem analogue, DeFormer's full joint) needs N passes of the state through the joint layers. LUMEN says this direction is critical; MORES works without it, but only on ranking.
  - For LIDM's VitaminC and RACE losses, where an option must be checked against fine detail in the state, the evidence favours per-option full-joint top layers (DeFormer pattern). For many short fixed labels (Banking77 names), a MORES-style asymmetric block over cached labels is the cheaper fit.
- **Hybrid suggestion (untested).** A single joint upper stack over [state; question; opt_1 … opt_N] with a block-diagonal mask across options removes N×S duplication but lets state tokens see all options at once. This is the Laya-in-upper-layers / MORES+ "query attends to all chunks" pattern, with lower layers per option so long descriptions are not truncated.

### Gaps
- No source profiles training-time memory for per-option joint layers. The estimates above are arithmetic, not measurements.
- ModernBERT's own attention pattern (local/global alternation) and its long-context behaviour for a roughly 5k-token joint upper stack were not verified in this research.
