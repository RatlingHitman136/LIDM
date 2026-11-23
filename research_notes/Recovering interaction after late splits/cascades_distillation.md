# Cascades, adaptive computation and cross-encoder distillation: getting joint-encoder accuracy where it matters at bi-encoder cost elsewhere

Scope: (1) score-all-then-rerank-top-k pipelines, (2) confidence/difficulty-based escalation (early exit, cascades), (3) distilling cross-encoder interaction into late-interaction/bi-encoder/decomposed students, (4) one network running both modes, (5) escalation triggers, (6) latency budget for LIDM (ModernBERT-large, 395M, 28 layers, RTX 4070 Super 12 GB).
Notation: CE = cross-encoder (joint encoding), BE = bi-encoder, LI = late interaction. Every number gives backbone / task / metric where the source states it. Research date: 2026-11-19.

## 1. Retrieve-then-rerank accuracy as a function of k (shortlist recall vs final accuracy), for many-label classification and multiple choice

### Takeaway
Shortlist recall rises fast and then flattens (SBERT top-10 already holds the gold intent 91.6% of the time on 3-shot Banking77), while the joint reranker's accuracy peaks at a moderate k and then drops as more distractors arrive (best k=25 on Banking77; reranker quality falls past some K in IR too). So k should be picked on dev by final accuracy, not by recall. A strong first stage at small k beats a weak one at large k (RocketQAv2: top-50 from a dense retriever > top-1000 from BM25). I found no peer-reviewed same-backbone "bi-encoder shortlist + CE rerank" result on Banking77/CLINC150, and could not locate the "Hong et al. 2024 top-10 shortlist" paper.

### Cited Findings
- **Du, Yin, Xia, Yu, "Learning to Select from Multiple Options" (arXiv 2212.00301, Dec 2022; v3 Sep 2023).** Backbone RoBERTa-large-mnli (intent, entity typing) / RoBERTa-large-DocNLI (MCTest). For many-option tasks they first take a top-k shortlist with a decoupled encoder (pretrained Sentence-BERT for intents, chosen by dev recall), because all options do not fit in one sequence. The shortlist step is skipped for multiple-choice QA, which has only 4 options — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
  - Banking77 (3-shot) top-k recall is "already high (91.62%) even when k = 10" and grows more slowly after k > 30. The plotted recall series climbs to about 99.9% at k=70, the largest k that fits RoBERTa's length limit — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
  - Final accuracy of the joint model (Parallel-TE) peaks at **k = 25**, and "a higher k harms the Parallel-TE performance". On UFET (10,331 types) F1 peaks at k=80 (52.40), with k=100 very close. The authors call this a trade-off: more correct options get captured, but more competitors make selection harder — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
  - Banking77 few-shot accuracy (5-shot / 10-shot): pairwise TE (each (utterance, intent) pair alone) 78.21 / 82.51; **Context-TE** (each pair with the other k options appended as context) **80.76 / 85.53**; **Parallel-TE** (all k options decided together in one sequence, "faster than the pairwise TE by k times") 78.69 / 83.29. Prior bests: DNNC 80.40 / 86.71, ConvBERT – / 85.95 — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
  - UFET F1: Parallel-TE 52.4 > Context-TE 51.5 > prior SOTA LITE 50.6 — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
- **Self-reported, not peer-reviewed (tinyintent GitHub):** frozen bge-large embeddings plus a linear head propose top-k intents, and a cross-encoder trained on the user's data (same/different-intent pairs with hard negatives, ensembled with the head's scores) reranks them. It reports top-1 0.960 on CLINC150 and 0.910 on Banking77. The README says off-the-shelf cross-encoders *hurt*, and that the reranker dominates inference cost, so exemplars per candidate are capped — [tinyintent](https://github.com/bgokden/tinyintent)
- **Infer–Retrieve–Rank (D'Oosterlinck, Khattab et al., arXiv 2401.12178, Jan 2024):** for multi-label classification with thousands of classes, an LM infers queries, a retriever shortlists labels, and an LM reranks. It reaches SOTA on HOUSE, TECH and TECHWOLF with only tens of labelled examples and no fine-tuning. This is LLM-based, not a same-backbone encoder comparison — [arXiv 2401.12178](https://arxiv.org/abs/2401.12178)
- **RocketQAv2 (ERNIE-base, MS MARCO passage, MRR@10; arXiv 2110.07367, Oct 2021):**

  | Re-ranker input | MRR@10 |
  |---|---|
  | Top-1000 from BM25 | 40.1 |
  | Top-50 from RocketQA | 41.8 |
  | Top-50 from the jointly trained RocketQAv2 retriever (retriever alone: 38.8) | **41.9** |

  — [arXiv 2110.07367](https://arxiv.org/abs/2110.07367); retriever number from [ColBERTv2 Table 4](https://arxiv.org/abs/2112.01488)
- **TAS-B (Hofstätter et al., SIGIR 2021):** with a TAS-trained BERT-dot retriever, "we can reduce the number of passages an expensive re-ranking system processes and still receive considerable benefits". But even as TAS-B recall keeps improving at higher cutoffs, "the re-ranking does not take advantage of that" (mono-duo-T5 reranker) — [arXiv 2104.06967](https://arxiv.org/abs/2104.06967)
- **"Drowning in Documents" (Jacob, …, Khattab, Drozdov; Databricks; arXiv 2411.11767, Nov 2024):** rerankers tested were bge-reranker-v2-m3, Cohere, Voyage, Jina and GPT-4o-mini listwise, over BM25, text-embedding-3-large and voyage-2 first stages, with K from 10 to 5000, measured by Recall@10. They "provide initial improvements when scoring progressively more documents, but their effectiveness gradually declines and can even degrade quality beyond a certain limit". The paper notes top-100 or top-1000 is the typical CE depth — [arXiv 2411.11767](https://arxiv.org/abs/2411.11767)
- **Shallow cross-encoders (Petrov, MacAvaney, Macdonald; arXiv 2403.20222, Mar 2024; TREC DL19, nDCG@10):** at a 25 ms per-query latency limit, MonoBERT-Large reaches only 0.431, because it can score few documents. TinyBERT trained with gBCE and 128 negatives reaches 0.652 (+51%). More candidates scored by a weaker model beat fewer scored by a strong one under a tight budget. On CPU, nDCG@10 drops only 3% versus GPU at a 50 ms budget — [arXiv 2403.20222](https://arxiv.org/abs/2403.20222)
- **In Defense of Cross-Encoders for Zero-Shot Retrieval (Rosa et al., arXiv 2212.06121, Dec 2022):** CEs "largely outperform bi-encoders of similar size". On BEIR the largest CE beats a SOTA bi-encoder by >4 average points. Bi-encoder first stages gave "no gains in comparison to … BM25 on out-of-domain tasks" — [arXiv 2212.06121](https://arxiv.org/abs/2212.06121)
- **Retrieve Fast, Rerank Smart (Geigle, Pfeiffer, Reimers, Vulić, Gurevych; arXiv 2103.11920, Mar 2021; OSCAR backbone, image–text retrieval, k=20):**
  - Accuracy (R@1): reranking only the BE top-20 with a CE ("cooperative") matches or beats running the CE over everything. MSCOCO-5k image R@1: CE 52.6, BE 52.2, separate-model cooperative 52.8. Flickr30k image R@1: CE 75.9, BE 72.0, cooperative 76.0 — [arXiv 2103.11920](https://arxiv.org/abs/2103.11920)
  - Latency on V100, 50k images: BE 16 ms, cooperative 74 ms, CE 2 min — [arXiv 2103.11920](https://arxiv.org/abs/2103.11920)

### Inferences
- For LIDM on Banking77, the 86% top-5 recall of the bge prior is an *upper bound* for a top-5 joint rerank. Du et al. suggest recall gains flatten around k = 10–30 for an SBERT-type shortlister, and joint selection accuracy peaks before recall saturates. So the dev sweep should be over k ∈ {5, 10, 15, 25} *final* accuracy. Recall@k alone is the wrong objective.
- The Context-TE vs Parallel-TE vs pairwise-TE result is the closest same-backbone analogue to LIDM's question. On 77 short labels, seeing competing options inside the joint pass (+2.5 points over pairwise at 5-shot) matters more than speed. A rerank stage that packs the top-k labels into *one* joint sequence (Laya-style) costs about one pass and keeps option–option interaction.
- LIDM's −11 at 77 short labels and +11 with many long options fit this pattern. Late scoring is good at shortlisting, a joint pass is good at discriminating among a few near-duplicates. That argues for "LIDM scores all 77, a joint pass decides among the top-k".

### Gaps
- I could not find the specific "Hong et al. 2024 top-10 shortlist + cross-encoder" intent/extreme-classification paper. Searches returned LabelCoRank (RoBERTa top-10 → GPT-4o rerank; arXiv 2503.07968, 2025) and LRSL (EMNLP Findings 2024 logit reranking), neither of which was verified as that paper.
- No peer-reviewed source gives a full recall@k vs final-accuracy curve for an encoder shortlist + same-backbone CE on Banking77/CLINC150. The only such numbers are a figure in Du et al. (SBERT + RoBERTa-large, few-shot) and self-reported GitHub numbers.
- For multiple-choice with ≤5 options there is no shortlist stage in the literature found: all options go to the joint model.

## 2. Cascades with confidence thresholds / learned deferral and early exit: accuracy vs fraction escalated

### Takeaway
With calibrated confidence, cascades of complete models routinely match the big model's accuracy while sending only about 20–50% of inputs to it. In some cases they slightly exceed it, because the small model is right on some items where the big one is wrong. Early exit from shallow internal layers (DeeBERT, PABEE) degrades badly past ~2× speed-up. Cascading complete models (CascadeBERT) or calibrated per-layer exits (Schwartz et al.) hold accuracy much better.

### Cited Findings
- **Model Cascading (Varshney & Baral, EMNLP 2022):** BERT-mini 11.3M / medium 41.7M / base 110M / large 340M, tested on MNLI, SNLI, DNLI, QNLI, QQP, MRPC, PAWS, SST-2, CoLA and CommitmentBank. An input is escalated when the current model's MaxProb (or DTU, distance to uniform) is below a per-model threshold — [ACL Anthology](https://aclanthology.org/2022.emnlp-main.756); [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)
  - K=3 cascade: saves "up to 88.93%" compute (QQP needs 11.07% of M3's FLOPs to match M3 accuracy), with accuracy gains up to +2.18% — [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)
  - K=2 (medium→base), cost to match M2's accuracy:

    | Task | M2 accuracy | M2 cost | Cascade cost at the same accuracy |
    |---|---|---|---|
    | QQP | 89.99% | 8.49 GFLOPs | 2.82 GFLOPs |
    | MNLI | 82.53% | 10.19 GFLOPs | 5.26 GFLOPs |

    At the MNLI matched-accuracy point, **78% of instances are answered by M1 and 22% escalated**. Pushed further, the cascade beats M2 (QQP 90.39 vs 89.99) — [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)
  - Area under the accuracy–cost curve, K=2: MaxProb ≈ DTU (e.g. MNLI 82.38 vs 82.38; CB 80.40 vs 80.70), both above Random (81.16 MNLI) and the paper's "Heuristic" baseline (81.13). In K=3, "DTU clearly outperforms all other cascading methods" because it uses the whole distribution shape — [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)
- **CascadeBERT (Li et al., Findings EMNLP 2021; BERT-base; GLUE test):** cascades complete models of increasing size instead of exiting from internal layers. Accuracy is shown with the measured speed-up in parentheses:

  | Task | BERT-base (1×) | CascadeBERT ~2× | CascadeBERT ~3× | CascadeBERT ~4× | DeeBERT ~2× | DeeBERT ~3× | DeeBERT ~4× | PABEE ~2× | PABEE ~3× | PABEE ~4× |
  |---|---|---|---|---|---|---|---|---|---|---|
  | MNLI-m | 84.6 | 83.0 | 81.2 | 79.3 | 74.4 (1.87×) | 63.2 | 55.8 | 79.8 | 75.9 (2.70×) | 62.3 (4.32×) |
  | GLUE avg | 82.6 | 81.1 | 79.8 | 78.3 | 77.5 | 72.3 | 67.8 | 79.2 | 75.9 | – |

  A static BERT-6L gives 79.3 avg at 2× and BERT-3L 74.3 at 4×. The paper explains the shallow-exit failure: shallow internal classifiers lack high-level semantics and are poorly calibrated. Its difficulty-aware regularizer lowers ECE on MNLI-m from 11.07 to 7.90, on QNLI from 8.79 to 3.41 and on SST-2 from 8.73 to 5.61 — [arXiv 2012.14682](https://arxiv.org/abs/2012.14682)
- **DeeBERT (Xin et al., ACL 2020):** per-layer exit "ramps" save "up to ∼40% inference time with minimal degradation" — [arXiv 2004.12993](https://arxiv.org/abs/2004.12993)
- **FastBERT (Liu et al., ACL 2020):** self-distilled per-layer classifiers, with speed-up tunable from 1× to 12× by threshold — [arXiv 2004.02178](https://arxiv.org/abs/2004.02178)
- **PABEE (Zhou et al., NeurIPS 2020):** exits when intermediate predictions stay unchanged for a fixed number of layers ("patience"). On ALBERT it improves both accuracy and robustness by preventing "overthinking" — [arXiv 2006.04152](https://arxiv.org/abs/2006.04152)
- **The Right Tool for the Job (Schwartz et al., ACL 2020; BERT; 3 classification + 2 NLI datasets):** classifiers at several layers, with exits decided by *calibrated* (temperature-scaled) confidence. Up to 5× faster than SOTA at preserved accuracy, and one trained model where a single inference-time variable controls the speed/accuracy trade-off — [arXiv 2004.07453](https://arxiv.org/abs/2004.07453)
- **Tabi (Wang, Chen, Tan, Guo; EuroSys 2023):** discriminative-model serving. A small model's temperature-scaled confidence decides whether to return its answer or re-route the query to a large model (RoBERTa-large-class). Re-routed queries get attention-based word pruning and a *weighted ensemble* of small- and large-model outputs. "Around 50% to 70% of the queries" return after the first model, and average latency drops 21–40% versus the state of the art at LLM-grade accuracy targets. Post hoc, a model with 1/4 of the parameters makes the same prediction as RoBERTa-large on >90% of queries — [Tabi PDF](https://cse.hkust.edu.hk/~kaichen/papers/tabi-eurosys23.pdf)
- **Cascade Transformer (Soldaini & Moschitti, ACL 2020; answer sentence selection on ASNQ and an Alexa set):** 5 rerankers attached at intermediate layers of one transformer. Each drops a fraction α of the candidates in the batch (e.g. α=0.3 with batch 128 keeps 90 after layer 4), and partial encodings are shared. Compute falls 37% "with almost no impact on accuracy" — [arXiv 2005.02534](https://arxiv.org/abs/2005.02534)
- **Early exit for CE rerankers (SEE, Busolin et al., SIGIR 2025):** similarity-based, not learned. On TREC DL19, 1.9× speed-up for −1.24% nDCG@10, versus −6.60% for the eeMB+ baseline at the same speed-up; about 2× average on BEIR with three CEs. Numbers come from a search snippet of the ACM record, because the PDF returned 403 — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3726302.3729962)
- **GAINER (Zenodo preprint, 2026; not peer-reviewed):** gain-aware depth allocation keeps 96.5% of full-depth nDCG@10 at half the candidate-layer compute on Ettin-68M (a ModernBERT-recipe encoder), a measured 1.69× GPU speed-up — [Zenodo](https://zenodo.org/records/23340638)

### Inferences
- LIDM already has the key ingredient that makes cascades work: confidence that is calibrated after temperature scaling (ECE 0.04, comparable to or better than CascadeBERT's 3.4–7.9 ECE after its regularizer). Tabi and Schwartz et al. use exactly temperature-scaled MaxProb as the deferral signal.
- Literature fractions to plan with: 22% escalated at matched accuracy (Model Cascading, MNLI, medium→base); 30–50% escalated (Tabi). LIDM's gap is task-dependent: a tie on typed decisions, but −5 to −11 elsewhere. So the escalation rate needed to recover the gap should be measured per task family.
- Expect "cascade beats the big model" effects only if LIDM is right on some items where Laya is wrong. The tie on typed decisions (0.769 vs 0.757) suggests disagreement exists there. Tabi's weighted ensemble of both outputs on escalated items is a cheap extra to try.
- In-network early exit (DeeBERT/PABEE-style) is a poor fit for LIDM's 28-layer joint path. CascadeBERT shows shallow exits collapse at ≥3×. Cascading *complete* models (LIDM → full joint pass) is the better-supported design.

### Gaps
- None of the cascade papers use ModernBERT-large, or a decomposed (late-split) model as the cheap stage. Escalation-rate vs accuracy curves for "late-interaction model → joint model on the same backbone" were not found.
- I did not retrieve the exact Model Cascading figure for the fraction escalated at each threshold beyond the MNLI 78/22 point.
- "Learned deferral" (a trained router rather than a confidence threshold) appears only in the IR routing paper in Section 5. No classification-specific learned-deferral numbers were collected.

## 3. Distillation of cross-encoder interaction into late-interaction / bi-encoder / poly-encoder / decomposed students: how much of the gap is closed

### Takeaway
Distillation from a joint teacher closes most of the gap for students that keep *some* token-level interaction: ColBERT and PreTTR-style decomposed models recover 68% to more than 100% of the gap to a single BERT-base CE teacher. Single-vector bi-encoders recover little from a single teacher and need ensembles, dual teachers or denoised negatives. DeFormer, the closest analogue to LIDM (lower layers separate, upper layers joint), keeps the drop at 1–2 points on RACE/BoolQ/MNLI with KD plus layerwise-representation losses. Those losses recover about 48% (BERT-base) to 69% (BERT-large) of the decomposition loss on SQuAD.

### Cited Findings
- **DeFormer (Cao, Trivedi, Balasubramanian, Balasubramanian; ACL 2020):** question and passage are encoded separately in the lower k layers (cacheable) and jointly in the upper layers. BERT-base uses 9 separate + 3 joint layers; XLNet-base 8 + 4 — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697)

  | Model | Task (avg tokens) | Full | DeFormer | Drop (abs) | Speed-up | Memory reduction |
  |---|---|---|---|---|---|---|
  | BERT-base | SQuAD (320) | 88.5 F1 | 87.1 | −1.4 | 3.2× | 70.3% |
  | BERT-base | **RACE (2048)** | **66.3 acc** | **64.5** | **−1.8** | 3.4× | 72.9% |
  | BERT-base | BoolQ (320) | 77.8 | 76.8 | −1.0 | 3.5× | 72.0% |
  | BERT-base | MNLI (120) | 84.4 | 82.6 | −1.8 | 2.2× | 56.4% |
  | BERT-base | QQP (100) | 90.5 | 90.3 | −0.2 | 2.0× | 50.0% |
  | XLNet-base | SQuAD | 91.6 | 90.4 | −1.2 | 2.7× | 65.8% |
  | XLNet-base | RACE | 70.3 | 68.7 | −1.6 | 2.8× | 67.6% |
  | XLNet-base | BoolQ | 80.4 | 78.8 | −1.6 (as printed; the abs/% columns read 0.6/0.7) | 3.0× | 68.3% |

  — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697)
  - Ablation on SQuAD F1, showing the effect of distillation:

    | | BERT-base | BERT-large |
    |---|---|---|
    | Full model | 88.5 | 92.3 |
    | DeFormer with KD + LRS | 87.1 | 90.8 |
    | Without LRS | 86.2 | 88.9 |
    | Without KD and LRS | 85.8 | 87.5 |

    KD is distillation of the full model's output distribution. LRS is a layerwise representation-similarity loss that pulls the decomposed model's upper-layer representations toward the full model's. DeFormer-BERT-large (90.8 F1, 47.7 GFLOPs) beats BERT-base (88.5 F1, 58.4 GFLOPs) and is faster — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697)
  - The F1 loss grows as the separation layer moves up, with a "dramatic increase in the last layers closest to the output" (Fig. 4, no auxiliary supervision) — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697)
- **PreTTR (MacAvaney et al., SIGIR 2020; BERT-base, 12 layers; web/Robust04 document ranking):** query and document are encoded independently for the first l layers (precomputed), then jointly. Up to 42× speed-up on web ranking, and the compression layer cuts storage by up to 95%. P@20 holds with only 1 joint layer (l=11) on WebTrack, but ERR@20 suffers as more layers are precomputed. On Robust04, l=11 falls below BM25 and nDCG@20 drops significantly already at l=8: "more [joint layers] are required to … identify the subtleties" of graded relevance — [arXiv 2004.14255](https://arxiv.org/abs/2004.14255)
- **MICE (Vast, Morand, Mothe, Piwowarski; arXiv 2602.16299, Feb 2026, EMNLP 2026 main):** attention-masking study of CE rerankers (MiniLM-v2, 12 layers; Ettin-32M, ModernBERT-based). Measured as nDCG@10 re-ranking BM25 top-1k, averaged over 5 seeds — [arXiv 2602.16299](https://arxiv.org/abs/2602.16299)
  - Query and document can be contextualized independently "without harming ID effectiveness, up to the layer ℓ*=4 (out of 12)" for MiniLM, and ℓ*=6 for Ettin-32M. Separating beyond ℓ* is "detrimental" — [arXiv 2602.16299](https://arxiv.org/abs/2602.16299)
  - MiniLM, in-domain avg / BEIR: CE baseline 63.8 / 44.7; PreTTR(ℓ4) 63.4 / 49.2; Sparse CE 63.0 / 44.9. Separation up to ℓ4 is near-lossless in-domain and *better* out-of-domain — [arXiv 2602.16299](https://arxiv.org/abs/2602.16299)
  - MICE also drops the backbone's top MLM layers and adds a lexical exact-match attention head. It "retains most of the performances" in-domain, matches or exceeds the CE out-of-domain, and cuts FLOPs by up to 2.5× — [arXiv 2602.16299](https://arxiv.org/abs/2602.16299)
- **Margin-MSE (Hofstätter et al., arXiv 2010.02666, Oct 2020; MS MARCO dev MRR@10; teacher T1 = BERT-base CE at 0.376, T2 = ensemble of 3 CEs at 0.399):** — [arXiv 2010.02666](https://arxiv.org/abs/2010.02666)

  | Student | No KD | T1 | T2 |
  |---|---|---|---|
  | ColBERT (closes 68% of the gap to T1) | 0.357 | 0.370 | 0.375 |
  | PreTT (decomposed CE; **exceeds T1**) | 0.358 | 0.378 | 0.386 |
  | DistilBERT-CE (exceeds T1) | 0.372 | 0.380 | 0.391 |
  | BERT-dot (single vector) | 0.320 | 0.321 | 0.333 |
  | DistilBERT-dot | 0.316 | 0.330 | 0.332 |

  Single-vector students gain little from a single teacher. In the loss ablation (single teacher), Margin-MSE beats pointwise MSE and weighted RankNet: for ColBERT, 0.370 vs 0.365 vs 0.356 MRR@10 — [arXiv 2010.02666](https://arxiv.org/abs/2010.02666)
  - Relative effectiveness and per-query latency, from their Table 1: BERT-CE 1.00 at 950 ms and 10.4 GB; **ColBERT 0.97 at 28 ms**; PreTT 0.97 at 455 ms; BERT-dot 0.87 at 23 ms; TK 0.89 at 14 ms — [arXiv 2010.02666](https://arxiv.org/abs/2010.02666)
- **TAS-B (Hofstätter et al., SIGIR 2021):** BERT-dot (DistilBERT) trained with a dual teacher: the BERT-CE pairwise Margin-MSE signal plus a ColBERT in-batch teacher. Margin-MSE is the best in-batch loss: TREC-DL'19 nDCG@10 0.712 vs 0.681 for KL-divergence. Training needs one consumer-grade GPU of at least 11 GB, versus 8× V100 for RocketQA/ANCE — [arXiv 2104.06967](https://arxiv.org/abs/2104.06967). TAS-B reaches MS MARCO dev MRR@10 34.7 — [ColBERTv2 Table 4](https://arxiv.org/abs/2112.01488)
- **RocketQA (Qu et al., NAACL 2021; ERNIE; MS MARCO MRR@10):** in-batch negatives 32.39 → cross-batch 33.32. Hard negatives without denoising *fall* to 26.03, because many are false negatives; denoising them with a CE gives 36.38, and CE-pseudo-labelled data augmentation 37.02 — [arXiv 2010.08191](https://arxiv.org/abs/2010.08191)
- **ColBERTv2 (Santhanam et al., NAACL 2022):** CE-distilled ("denoised supervision") late interaction with residual compression reaches MS MARCO dev MRR@10 39.7 (Recall@1k 98.4). That is above RocketQAv2 38.8, coCondenser 38.2, SPLADEv2 36.8 and TAS-B 34.7. It is best on 22 of 28 out-of-domain tests — [arXiv 2112.01488](https://arxiv.org/abs/2112.01488)
- **Poly-encoders (Humeau et al., ICLR 2020; BERT-base weights; dialogue/IR, R@1):** — [arXiv 1905.01969](https://arxiv.org/abs/1905.01969)

  | Task (metric) | Bi | Poly-16 | Poly-64 | Poly-360 | Cross |
  |---|---|---|---|---|---|
  | ConvAI2 (R@1/20) | 81.7 | 83.2 | 83.7 | 83.7 | 84.8 |
  | DSTC7 (R@1/100) | 66.8 | – | – | 68.9 | 67.4 |
  | Ubuntu v2 (R@1/10) | 80.6 | – | – | 81.2–81.3 (range over m) | 82.8 |
  | ConvAI2, Reddit-pretrained (R@1/20) | 84.8 | – | – | 86.8 | 87.9 |

  Poly-encoders close about 65% of the Bi→Cross gap on ConvAI2 (with BERT weights), exceed the Cross-encoder on DSTC7, and close about 32% on Ubuntu v2. These use direct training, not distillation — [arXiv 1905.01969](https://arxiv.org/abs/1905.01969)
  - GPU time to score 1k candidates: Bi 19 ms, Poly-16 18 ms, Poly-360 57 ms, Cross 2.6 s. With 100k candidates: Bi 22 ms, Poly 38–88 ms, Cross ~266 s (extrapolated) — [arXiv 1905.01969](https://arxiv.org/abs/1905.01969)
- **Augmented SBERT (Thakur et al., NAACL 2021):** a CE labels extra sentence pairs to train a bi-encoder. Gains are up to +6 points in-domain and up to +37 points for domain adaptation, and the choice of which pairs to label is "crucial" — [arXiv 2010.08240](https://arxiv.org/abs/2010.08240)
- **Trans-Encoder (Liu et al., ICLR 2022):** unsupervised. The model alternates bi- and cross-encoder formulations of the same PLM, each pseudo-labelling for the other, with mutual distillation across PLMs. Both resulting bi- and cross-encoders beat prior unsupervised SOTA on sentence similarity — [arXiv 2109.13059](https://arxiv.org/abs/2109.13059)
- **ERNIE-Search (arXiv 2205.09153, 2022):** cascade distillation CE → ColBERT → dual encoder, plus distillation of the CE's final-layer *attention map* into the late-interaction student. The argument is that token-level CE interaction is hard for a dual encoder to learn directly — [arXiv 2205.09153](https://arxiv.org/abs/2205.09153)
- **DISKCO (WWW '24 Companion):** disentangled CE→BE distillation, with up to 2% gains over conventional distillation (search snippet only) — [ACM DL](https://dl.acm.org/doi/pdf/10.1145/3589335.3648333)

### Inferences
- LIDM is the extreme point of the DeFormer/PreTTR/MICE family: options are separate for all 28 layers and interact only in a fusion head. All three papers show the split depth is the main lever, and losses appear on graded or fine-grained distinctions first (PreTTR: P@20 holds while ERR@20 drops). That matches LIDM's losses on VitaminC and RACE.
- MICE's ℓ* ≈ ⅓–½ depth on 12-layer models suggests a DeFormer-style LIDM variant: a shared state+question+option joint stack for the last ~⅓–½ of ModernBERT-large's 28 layers (separate up to about layer 10–18). It would keep most of the joint accuracy while still caching lower-layer option encodings. This is an inference from smaller backbones and needs measurement.
- Distillation targets with the best evidence: the joint teacher's (Laya's) per-option score *margins* (Margin-MSE), and upper-layer representations (DeFormer's LRS, worth +0.9 to +1.9 F1 on SQuAD). Margin-MSE recovered about 68% of the gap for multi-vector ColBERT but about 0% for single-vector BERT-dot. So distillation helps most when the student keeps token-level option representations, not a single pooled vector per option.
- Distilling from an *ensemble* of joint teachers (T2) consistently helped weak students more than a single teacher (BERT-dot 0.321 → 0.333).

### Gaps
- No distillation result on ModernBERT-large, or on decision/multiple-choice tasks other than DeFormer's RACE/BoolQ and Poly-encoder's dialogue tasks.
- ERNIE-Search's numeric gap closure was not extracted.
- MICE's final MICE-vs-CE tables (beyond the masking study and the abstract's 2.5× FLOPs) were not extracted.

## 4. Joint training of cheap and expensive stages in one model, and whether one network can run in both modes

### Takeaway
Yes. A single shared-weight network trained alternately on bi-encoder and cross-encoder objectives matched or beat separately trained CEs in image–text retrieval: JOINT+COOP MSCOCO image R@1 54.7 vs CE 52.6. Jointly training retriever and reranker (RocketQAv2) and same-network cascades (Cascade Transformer, Right Tool, Trans-Encoder) also work. For LIDM this supports one ModernBERT-large backbone (or one LoRA set per mode) serving both pass-A/B late scoring and a joint `[CLS] state [SEP] question [SEP] options` mode.

### Cited Findings
- **Retrieve Fast, Rerank Smart (Geigle et al., arXiv 2103.11920):** one network with shared weights, fine-tuned alternately with a triplet loss in bi-encoder mode and a BCE head in cross-encoder mode. Cooperative inference uses BE mode to fetch the top-20 and CE mode to rerank — [arXiv 2103.11920](https://arxiv.org/abs/2103.11920)

  | Model (OSCAR backbone) | MSCOCO-5k image R@1 | MSCOCO-5k text R@1 | Flickr30k image R@1 | Flickr30k text R@1 |
  |---|---|---|---|---|
  | Separately trained CE | 52.6 | 69.3 | 75.9 | 88.5 |
  | Separately trained BE | 52.2 | 66.9 | 72.0 | 84.7 |
  | JOINT+COOP (one network, both modes) | **54.7** | **70.8** | **76.4** | **89.4** |
  | JOINT+CE (joint model in CE mode only) | 54.6 | – | – | – |
  | JOINT+BE (joint model in BE mode only) | 52.5 | 66.7 | – | – |

  The joint model gains in CE mode and loses nothing in BE mode. It is also more parameter-efficient — [arXiv 2103.11920](https://arxiv.org/abs/2103.11920)
- **RocketQAv2 (ERNIE-base, EMNLP 2021):** "dynamic listwise distillation" trains the retriever and re-ranker jointly, each adapting to the other's relevance distribution. Retriever MRR@10 38.8; the re-ranker over its top-50 reaches 41.9 — [arXiv 2110.07367](https://arxiv.org/abs/2110.07367)
- **Trans-Encoder (ICLR 2022):** the *same* PLM alternates between bi- and cross-encoder formulations, distilling each into the other. Both modes end up better than either trained alone (unsupervised STS) — [arXiv 2109.13059](https://arxiv.org/abs/2109.13059)
- **Cascade Transformer (ACL 2020):** a single network whose intermediate-layer classifiers prune candidates, sharing partial encodings, with 37% less compute at near-equal accuracy — [arXiv 2005.02534](https://arxiv.org/abs/2005.02534)
- **Right Tool for the Job (ACL 2020):** one trained BERT serves all speed/accuracy points via a single inference-time threshold on calibrated confidence — [arXiv 2004.07453](https://arxiv.org/abs/2004.07453)
- **DeFormer / PreTTR:** the decomposed model is initialized from, and distilled toward, the same pre-trained weights as the full joint model, so a single set of weights effectively runs separated lower layers and joint upper layers — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697); [arXiv 2004.14255](https://arxiv.org/abs/2004.14255)

### Inferences
- LIDM and Laya already share a backbone. A shared-weight dual-mode model could be trained with mixed batches: LIDM's pass-A/B loss plus a Laya-style joint loss, optionally with Margin-MSE from the joint mode into the late mode, as in Trans-Encoder and RocketQAv2. Geigle et al. suggest the joint objective may *improve* the CE mode instead of costing it.
- On a 12 GB card, one backbone with two lightweight heads (or two LoRA adapters) avoids holding two 395M models in memory. This is an engineering inference, not measured.

### Gaps
- No found work trains a shared-weight dual-mode model for classification or multiple-choice decisions (the evidence is cross-modal retrieval and IR).
- I found no published "early-interaction heads on demand" design (adding joint layers only for escalated inputs on top of cached separate encodings) beyond DeFormer/PreTTR/MICE, which always run the joint upper layers.

## 5. What triggers the expensive path best: confidence, margin, option length/count, question type?

### Takeaway
For classification cascades, calibrated MaxProb and DTU (which uses the whole distribution, hence also the margin) are the best-supported triggers. DTU is clearly better once there are more than two stages. Surface features (input length, BM25 top-score gap, query lexical complexity) are weak predictors of whether the expensive model helps. Routing has a large oracle headroom, but a 65%-accurate learned router realised only part of it.

### Cited Findings
- **Model Cascading:** MaxProb and DTU tie in 2-model cascades (AUC on MNLI 82.38 vs 82.38). "DTU clearly outperforms" MaxProb in 3-model cascades because it uses the whole output distribution. Both beat a heuristic baseline (Table 2: MNLI 81.13, below Random's 81.16) — [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)
- **CascadeBERT:** raw shallow-model confidence is unreliable. Its difficulty-aware regularizer, which trains confidence to reflect instance difficulty, raises a difficulty-identification score (DIS) on MNLI-m from 76.73 to 78.00 and on SST-2 from 79.30 to 82.02, and lowers ECE. On QQP, DIS falls (85.77 → 84.39) while ECE still improves (4.99 → 3.37) — [arXiv 2012.14682](https://arxiv.org/abs/2012.14682)
- **Right Tool / Tabi:** both use temperature-scaled confidence as the escalation trigger — [arXiv 2004.07453](https://arxiv.org/abs/2004.07453); [Tabi PDF](https://cse.hkust.edu.hk/~kaichen/papers/tabi-eurosys23.pdf)
- **Adaptive Re-Ranking (Genc, Korukluoglu, Allan; arXiv 2606.25249, Jun 2026, ICTIR '26):** — [arXiv 2606.25249](https://arxiv.org/abs/2606.25249)
  - Setup: a bert-base router reads the query text and sends it to BM25 only, BM25 + MiniLM-L6 CE, or BM25 + BGE-reranker-v2-m3 (top-50). Router labels came from 306k BEIR training queries: 59.2% "no rerank", 28.8% small CE, 11.6% large CE. Router test accuracy is 65% — [arXiv 2606.25249](https://arxiv.org/abs/2606.25249)
  - Weak surface features: correlation with the rerank decision is ρ=0.14 for number retrieved, about 0.08 for query length and −0.06 for the BM25 score gap; lexical complexity has Cramér's V ≈ 0.03–0.04 — [arXiv 2606.25249](https://arxiv.org/abs/2606.25249)
  - Results (nDCG@10):

    | Dataset | Always BGE (large CE) | Router | Oracle |
    |---|---|---|---|
    | NFCorpus | .324 | .337 | .370 |
    | SciFact | .720 | .697 | .756 |
    | FiQA | .371 | .306 | .403 |
    | FEVER | .759 | .656 | .776 |

    Router vs always-BGE ranges from −17.5% to +4.0%. Median latency is 1.15–53.2× lower and mean latency 1.11–5.22× lower; router overhead is about 3.6 ms — [arXiv 2606.25249](https://arxiv.org/abs/2606.25249)
- **Du et al.:** option count matters. More shortlisted options raise recall but make the joint selection harder (Banking77 best at k=25) — [arXiv 2212.00301](https://arxiv.org/abs/2212.00301)
- **DeFormer / PreTTR:** decomposition hurts most where graded or fine distinctions are needed (PreTTR ERR@20, nDCG@20) and on long multi-sentence inputs like RACE (−1.8, the largest absolute drop among DeFormer-BERT tasks, tied with MNLI) — [arXiv 2005.00697](https://arxiv.org/abs/2005.00697); [arXiv 2004.14255](https://arxiv.org/abs/2004.14255)

### Inferences
- LIDM's own measured pattern is a strong *static* trigger: escalate by task type or option profile. Fine-detail verification (VitaminC-like), passage comprehension (RACE-like) and many short options (77 labels) should go to the joint path. Many long options should stay on LIDM, which wins there by +11. Within a task, use temperature-scaled LIDM confidence, or better the top-1 – top-2 margin or DTU over the options. No source compared margin and MaxProb head-to-head for 2–5-option decision tasks; DTU subsumes the margin.
- The Adaptive Re-Ranking oracle gap (e.g. +0.046 nDCG over always-heavy on NFCorpus) shows routing can beat "always expensive" in quality, not only cost. That happens only if the cheap model is right on items where the expensive one errs, which mirrors the Model Cascading finding.

### Gaps
- No head-to-head of MaxProb vs margin vs entropy vs learned deferral for multiple-choice or verification tasks with 2–5 options.
- No study found on using "option length/count" as an explicit trigger feature in classification cascades.

## 6. Practical budget: how many joint passes per case fit under ~100 ms at batch 1 on a 12 GB consumer GPU with a 400M encoder?

### Takeaway
The literature gives throughput, not batch-1 latency, for ModernBERT-large: about 52k tokens/s at 512 tokens on an RTX 4090. The project's own numbers are the better yardstick: Laya ≈ 44 ms per 5-question case, so about 9 ms per joint question pass, and LIDM ≈ 64 ms. LIDM plus escalation therefore has about 36 ms of headroom. That is about 4 sequential joint question passes per case, or more if escalated questions are batched into one forward. Cascades in the literature escalate about 20–50% of inputs (1–2.5 of 5 questions), well within budget.

### Cited Findings
- **ModernBERT paper (Warner et al., arXiv 2412.13663, Dec 2024; RTX 4090, thousands of tokens/s, averaged over 10 runs):** — [arXiv 2412.13663](https://arxiv.org/abs/2412.13663)

  | Model | Size | 512 tokens, fixed / variable length | Max batch at 512 | 8192 tokens, fixed / variable |
  |---|---|---|---|---|
  | ModernBERT-large | 395M | 52.3 / 52.9 | 770 | 46.8 / 49.8 |
  | BERT-large | 330M | 54.4 / 27.2 | – | – |
  | DeBERTaV3-large | 434M | 24.6 / 12.3 | – | – |
  | ModernBERT-base | 149M | 148.1 / 147.3 | – | – |

  — [arXiv 2412.13663](https://arxiv.org/abs/2412.13663)
- **Shallow cross-encoders:** under a 25 ms per-query limit a 24-layer CE (MonoBERT-Large) can score too few candidates and loses 0.22 nDCG@10 to TinyBERT (0.431 vs 0.652, DL19). Latency limits change which model is best — [arXiv 2403.20222](https://arxiv.org/abs/2403.20222)
- **Margin-MSE Table 1:** joint BERT-base CE reranking costs 950 ms per query versus 28 ms for ColBERT, which keeps 0.97 of CE effectiveness. The ~34× cost ratio is why CEs are restricted to small shortlists — [arXiv 2010.02666](https://arxiv.org/abs/2010.02666)
- **Poly-encoder timing:** cross-encoding 1k candidates takes 2.6 s on GPU versus 19 ms for a bi-encoder — [arXiv 1905.01969](https://arxiv.org/abs/1905.01969)
- **Geigle et al.:** BE retrieval plus CE rerank of the top-20 takes 74 ms per query on a V100 (50k items), versus 16 ms for BE alone — [arXiv 2103.11920](https://arxiv.org/abs/2103.11920)
- **Tabi:** 50–70% of queries exit at the small model — [Tabi PDF](https://cse.hkust.edu.hk/~kaichen/papers/tabi-eurosys23.pdf)
- **Model Cascading:** 22% escalated at matched accuracy (MNLI, K=2) — [arXiv 2210.05528](https://arxiv.org/abs/2210.05528)

### Inferences
- Budget arithmetic from the project's own measurements (not a literature number): 100 ms − 64 ms (LIDM) ≈ 36 ms ≈ 4 sequential batch-1 joint passes at ≈ 9 ms each (44 ms / 5 questions). Expected added latency per case:
  - At 22% escalation (Model Cascading): about 1.1 joint passes, roughly +10 ms → ~74 ms.
  - At 30–50% escalation (Tabi): about 1.5–2.5 passes, roughly +14–22 ms → ~78–86 ms.
  - Worst case, all 5 escalated *and batched into one forward*: on a 395M model with short sequences at batch 1, a batch of 5 is likely to cost well under 5× a single pass, because batch-1 GPU use is low. This must be benchmarked on the 4070 Super.
- For Banking77-style many-label questions, a single joint pass over `[state, question, top-k labels]` (Laya/Parallel-TE style) costs about one Laya pass (≈ 9 ms) regardless of k ≤ ~25. That is cheaper than k separate pair passes and keeps option–option interaction (Du et al.).
- A DeFormer-style partial-joint stage (cached separate lower layers, joint upper ⅓–½) would cost less than a full joint pass per escalated question. DeFormer measured 2.0–3.5× speed-ups on BERT-base. This is a middle option if full joint passes prove too expensive.
- The 4070 Super has less compute than the 4090 used in the ModernBERT benchmark, so absolute throughput will be lower. The project's 44/64 ms measurements already reflect the real card.

### Gaps
- No published batch-1 latency for ModernBERT-large on an RTX 4070 Super (or any 12 GB card), and no published batch-size scaling at short sequence lengths. Both must be measured locally.
- I found no paper reporting the per-case latency of "late-interaction model + selective joint rerank" on the same ~400M backbone.
