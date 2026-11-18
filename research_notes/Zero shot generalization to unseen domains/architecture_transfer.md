# Encoder architecture and out-of-domain generalization for text-to-label/document matching

Notes on venues: venues and years marked "(venue from prior knowledge)" were not shown on the fetched page. Everything else was verified on the fetched page. Preprints are marked.

## Retrieval analogues: bi-encoder vs late interaction vs cross-encoder out of domain (BEIR and similar)

### Takeaway
When all models are trained on MS MARCO, the out-of-domain order on BEIR is cross-encoder reranker > late interaction > dense bi-encoder. In-domain scores are nearly tied. On BEIR (2021), BM25+CE scored +11% over BM25 on average, ColBERT +2.5%, TAS-B -2.8% and DPR -47.7%. The late-interaction lead mostly comes from the training signal, not the architecture alone. ColBERTv2, distilled from a cross-encoder over mined hard negatives, beats the best in-domain dense model by about 6.7 nDCG points OOD. A sparse bi-encoder distilled the same way (SPLADEv2) ties it. Reranker gains OOD shrink as the first stage gets stronger or domain-adapted, and small rerankers can fall below BM25 OOD despite large in-domain gains.

### Cited Findings
**BEIR (Thakur et al., NeurIPS 2021 Datasets & Benchmarks)**
- 18 zero-shot datasets, 10 systems. Re-ranking and late-interaction models give the best zero-shot average "at high computational costs"; BM25 is "a robust baseline" — [BEIR](https://arxiv.org/abs/2104.08663)
- Average nDCG@10 relative to BM25: BM25+CE +11%, ColBERT +2.5%, docT5query +1.6%, TAS-B -2.8%, GenQ -3.6%, ANCE -7.4%, SPARTA -20.3%, DeepCT -27.9%, DPR -47.7% — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- In-domain MS MARCO nDCG@10: BM25 0.228, ANCE 0.388, TAS-B 0.408, ColBERT 0.401, BM25+CE 0.413. Neural models are within about 0.03 of each other in-domain but far apart zero-shot. The paper states that in-domain performance does not predict generalization: DeepCT and SPARTA are good on MS MARCO and below BM25 almost everywhere else — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- Zero-shot nDCG@10 (BM25 / TAS-B / ColBERT / BM25+CE): TREC-COVID .656/.481/.677/.757; BioASQ .465/.383/.474/.523; FiQA .236/.300/.317/.347; SciFact .665/.643/.671/.688; Robust04 .408/.427/.391/.475; Climate-FEVER .213/.228/.184/.253; Touché .367/.162/.202/.271; ArguAna .315/.429/.233/.311 — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- Win counts vs BM25: CE 16/18 (loses on ArguAna and Touché), docT5query 11/18, ColBERT 9/18. TAS-B beats ANCE on 14/18 and DPR on 17/18 — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- Training: the CE is a 6-layer, 384-hidden MiniLM reranking BM25 top-100, trained on MS MARCO by distillation from a BERT-base/BERT-large/ALBERT-large ensemble. TAS-B is trained on MS MARCO with dual supervision from a CE and ColBERT. ColBERT v1 is trained on MS MARCO triples (300K steps). DPR is trained on NQ/TriviaQA/WebQ/CuratedTREC — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- Cost on 1M DBPedia docs: BM25+CE 450 ms/query on GPU (6100 ms CPU); ColBERT 20 GB index; TAS-B/ANCE about 14-20 ms on GPU with about 3 GB index — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)

**ColBERTv2 (Santhanam et al., NAACL 2022)**
- Recipe: ColBERT retrieves candidates for each MS MARCO query. A 22M MiniLM cross-encoder (ms-marco-MiniLM-L-6-v2) scores them. ColBERT is trained by KL divergence to the CE scores over 64-passage groups (positive plus lower-ranked candidates from the top 500) plus in-batch negatives. Two rounds of re-retrieval and retraining, 400k steps, batch 32 — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488)
- BEIR nDCG@10 (ColBERTv2 / ColBERT v1 / TAS-B / RocketQAv2 / SPLADEv2): NQ 56.2/52.4/46.3/50.5/52.1; T-COVID 73.8/67.7/48.1/67.5/71.0; FiQA 35.6/31.7/30.0/30.2/33.6; ArguAna 46.3/23.3/42.7/45.1/47.9; HotpotQA 66.7/59.3/58.4/53.3/68.4; C-FEVER 17.6/18.4/22.8/18.0/23.5; SciFact 69.3/67.1/64.3/56.8/69.3 — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488)
- In-domain MS MARCO dev MRR@10: ColBERTv2 39.7, RocketQAv2 38.8, coCondenser 38.2, SPLADEv2 36.8, ColBERT 36.0, TAS-B 34.7, ANCE 33.0 — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488)
- LoTTE Success@5 pooled (search / forum): ColBERTv2 71.6/63.4, RocketQAv2 69.8/57.7, SPLADEv2 68.9/60.1, ColBERT 67.3/58.2, ANCE 66.4/55.7, BM25 48.3/47.2 — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488)
- ColBERTv2 has no ablation that separates the supervision change from the architecture for OOD, because supervision and compression change together — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488)

**Scaling cross-encoders (Rosa et al., "In Defense of Cross-Encoders for Zero-Shot Retrieval", arXiv preprint 2022)**
- BEIR average nDCG@10 over 18 datasets: BM25 0.432; monoT5-60M 0.425; monoT5-220M 0.478; monoT5-3B 0.532; bi-encoders GTR-1.24B 0.453, GTR-4.8B 0.458, SGPT-2.7B 0.453, SGPT-5.8B 0.490. monoT5 reranks the BM25 top 1000 — [Rosa et al. 2022](https://ar5iv.labs.arxiv.org/html/2212.06121)
- In-domain MS MARCO MRR@10: monoT5 0.356 (60M), 0.381 (220M), 0.398 (3B); GTR 0.385 / 0.388; BM25 0.187. Scaling gives "marginal gains on in-domain test sets" but much larger OOD gains. The 60M CE is far above BM25 in-domain and slightly below it OOD — [Rosa et al. 2022](https://ar5iv.labs.arxiv.org/html/2212.06121)
- Reranking with monoT5-220M on 8 BEIR datasets: BM25 first stage 0.441 → 0.496, GTR-335M first stage 0.451 → 0.496. A dense first stage gave no OOD gain over BM25 once a CE reranks. All models were fine-tuned on MS MARCO (8.8M passages, >530k training query-relevant pairs) — [Rosa et al. 2022](https://ar5iv.labs.arxiv.org/html/2212.06121)

**Reranker on top of a stronger or domain-adapted first stage**
- GPL, on 6 BEIR targets (nDCG@10): BM25 48.5 → BM25+CE 54.7 (+6.2), while TSDAE+GPL dense 52.9 → TSDAE+GPL+CE 55.1 (+2.2). The CE adds much less on top of a domain-adapted dense retriever — [Wang et al., GPL (NAACL 2022, venue from prior knowledge)](https://ar5iv.labs.arxiv.org/html/2112.07577)
- Reranking SPLADE: in-domain (MS MARCO/TREC DL), cross-encoder rerankers are hard to tell apart. OOD (BEIR, LoTTE), the reranker type and the number of documents reranked both change effectiveness. GPT-4 listwise is strong zero-shot, but CEs "remain very competitive" — [Déjean, Clinchant, Formal, arXiv preprint 2024](https://arxiv.org/abs/2403.10407)
- "Drowning in Documents": the best rerankers improve at first as they score more documents, then decline "and can even degrade quality beyond a certain limit". This was tested with modern dense first stages, including out-of-domain and contamination-free internal sets — [Jacob et al., ReNeuIR @ SIGIR 2025](https://arxiv.org/abs/2411.11767)
- MS MARCO-trained T5-3B reranking BM25+RM3 on Robust04 zero-shot (NDCG@20 .601, P@20 .510) beats CEDR (.538) and Birch (.532), which were trained or tuned on Robust04 — [Nogueira et al., monoT5 (Findings of EMNLP 2020, venue from prior knowledge)](https://ar5iv.labs.arxiv.org/html/2003.06713)

### Inferences
- My own averages over the 12 BEIR datasets ColBERTv2 reports for all methods (excluding Touché): ColBERTv2 51.9, SPLADEv2 51.7, ColBERT v1 47.2, TAS-B 45.9, RocketQAv2 45.2. RocketQAv2 is the second-best model in-domain (38.8 MRR) but last of these OOD. In-domain gains from dense-specific training tricks did not transfer, while CE-distilled models (ColBERTv2, SPLADEv2) did, whatever their architecture.
- The OOD advantage of interaction needs capacity. A 60M monoT5 does not beat BM25 OOD, while 3B gains +0.10. A small interaction head is unlikely to inherit the "cross-encoders generalize" result. That result is for full-depth, large, joint-attention models trained on about 500k queries.
- When the first stage is already strong or domain-adapted, a source-domain reranker adds only about 2 nDCG points (GPL) or nothing over BM25+CE (Rosa). It can also hurt on tasks whose relevance notion differs (ArguAna, Touché) or when it reranks deeper lists (Drowning in Documents).

### Gaps
- MTEB (Muennighoff et al., 2023) was not fetched, so no MTEB numbers are included.
- No controlled study was found that holds training data and distillation fixed and varies only bi-encoder vs late-interaction head for OOD. ColBERTv2 vs SPLADEv2 is the closest, and their training setups differ.
- Déjean et al. and Drowning in Documents were read at abstract level only; no per-dataset numbers.

## Classification: GLiClass, GLiNER, NLI cross-encoders vs embedding classifiers on unseen labels

### Takeaway
Joint encoders (text plus all labels in one sequence: GLiClass, GLiNER uni) and pairwise NLI cross-encoders reach similar zero-shot averages, with GLiClass-large 0.719 vs DeBERTa-v3-large NLI 0.682 F1. Both are weak on many-label, fine-grained intent sets (Banking77 0.56 F1 at best). The one published bi- vs joint-encoder comparison (GLiNER, preprint 2026) finds the bi-encoder slightly better on average (CrossNER 61.5 vs 60.9), and the joint model wins on some domains. That comparison is confounded by different training data. Training-task diversity (number of datasets, label sets and pseudo-labels) moves zero-shot accuracy more than architecture does.

### Cited Findings
**GLiClass (Stepanov et al., arXiv preprint Aug 2025)**
- Zero-shot average F1 over 14 datasets: gliclass-large 0.7193, base 0.6764, modern-large 0.6197, modern-base 0.5577, edge 0.4900. NLI cross-encoders: deberta-v3-large-zeroshot-v2.0 0.6821, deberta-v3-base-zeroshot-v2.0 0.6559, roberta-large-zeroshot-v2.0-c 0.6152 — [GLiClass](https://arxiv.org/html/2508.07662)
- Many-label sets, F1 (large / base / edge): banking77 0.5574/0.4698/0.0272; massive 0.5649/0.5040/0.2414; 20 newsgroups 0.5958/0.4759/0.2217; snips 0.9692/0.9474/0.5257. Binary sentiment is above 0.9 (imdb 0.9366, sst2 0.9192) — [GLiClass](https://arxiv.org/html/2508.07662)
- The paper describes uni-encoder, bi-encoder, fused bi-encoder and encoder-decoder variants but reports no numbers comparing them. It notes that accuracy degrades with large label sets (banking77) and extreme label-to-text ratios. It has no dedicated held-out-domain experiment — [GLiClass](https://arxiv.org/html/2508.07662)
- 8 examples per label: large 0.6202 → 0.7265 (+17%), edge 0.3774 → 0.5662 (+50%) — [GLiClass](https://arxiv.org/html/2508.07662)
- Throughput from 1 to 128 labels (A6000, batch 1): GLiClass-base 49.42 → 45.94 ex/s; deberta-v3-base NLI CE 24.55 → 0.47 ex/s. The CE needs one pass per label — [GLiClass](https://arxiv.org/html/2508.07662)
- Training data: 1.2M-example pre-training mix (classification, sentiment, NLI); post-training on CommonsenseQA, 2k synthetic logic/NLI items and 2k texts each with 50 true and 50 false GPT-4o labels; MultiNLI — [GLiClass](https://arxiv.org/html/2508.07662)

**GLiNER bi-encoder vs uni-encoder (Stepanov et al., "The Million-Label NER", arXiv preprint Feb 2026)**
- CrossNER zero-shot micro-F1 (bi-large v2.0 vs uni-large v2.5): AI 57.4/54.9, literature 63.2/65.2, music 74.0/71.6, politics 73.0/72.4, science 67.6/63.9, MIT-movie 51.0/53.0, MIT-restaurant 44.3/45.0; average 61.5/60.9. On the 19-dataset suite: bi-large 49.7 vs 45.7-47.0 for uni models — [GLiNER bi-encoder](https://arxiv.org/html/2602.18487)
- The authors say dot-product late fusion is "largely" sufficient for NER because entity-type semantics are fairly self-contained. They say the uni-encoder keeps an edge on "well-established benchmark formats", possibly by capturing annotation conventions, and that context-dependent entities may favour joint encoding. Neither claim is ablated — [GLiNER bi-encoder](https://arxiv.org/html/2602.18487)
- Throughput at 1024 labels with pre-computed label embeddings: bi-edge 18.3 ex/s vs uni-small 0.14 (about 130x). F1 is not reported as a function of label count, so the "million-label" claim is untested for accuracy. Training: 8M GPT-4o-annotated FineFineWeb samples plus 40k post-training samples — [GLiNER bi-encoder](https://arxiv.org/html/2602.18487)

**NLI cross-encoders (Laurer et al., "Building Efficient Universal Classifiers with NLI", arXiv preprint 2023, rev. 2024)**
- Training: 33 datasets (5 NLI plus 28 classification), 389 classes, about 885k NLI pairs plus 51,731 classification texts (capped at 500 per class and 5,000 per dataset) — [Laurer et al.](https://arxiv.org/html/2312.17543)
- 28 leave-one-dataset-out models: mixing non-NLI tasks improves held-out zero-shot performance by 9.4% on average over NLI-only training. There is negative transfer on some sets (financialphrasebank). NLI-only models struggle on tasks unlike NLI or with more classes. One forward pass per class makes many-class tasks costly — [Laurer et al.](https://arxiv.org/html/2312.17543)

**Embedding (bi-encoder) intent classifiers on unseen intents (Sung et al., PIE, EMNLP 2023)**
- PIE embeds utterances and intent names with one encoder and matches by cosine. Zero-shot, it scores 88.3 on Banking77, 87.7 on HWU64, 83.8 on Liu54 and 96.5 on Clinc150 (average 89.1). This is a 5-way setting, i.e. 5 candidate intents per test. The SBERT-paraphrase baseline averages 86.2 and BERT-TAPT 39.3 — [PIE](https://arxiv.org/html/2305.14827v2)
- With all held-out intents as candidates (Banking77 N=27, HWU64 N=25, Liu54 N=18, Clinc150 N=50), PIE scores 71.4 / 68.7 / 71.4 / 83.1. Accuracy falls about 17 points from 5-way to N-way — [PIE](https://arxiv.org/html/2305.14827v2)
- Pre-training data: 39,117 utterances, 252 gold intents and 29,577 pseudo intents from TOP, DSTC11-T2 and SGD. Removing the 11 of 345 test intent names that overlap with pre-training barely changes results (Banking77 71.4 → 72.0) — [PIE](https://arxiv.org/html/2305.14827v2)

**Laya**
- The PIXMELT/laya model card describes "non-autoregressive typed decisions over a state, all questions answered in a single forward pass" on a ModernBERT-large encoder with decision heads. It reports no benchmark results, no training data and no comparisons — [Laya model card](https://huggingface.co/PIXMELT/laya)
- A search snippet (card not fetched) says LatentAILabs/laya-indic (mmBERT-base plus decision head, about 322M parameters) defines label schemas at inference time and self-reports 0.800 across 67 tasks — [laya-indic model card](https://huggingface.co/LatentAILabs/laya-indic)

### Inferences
- Accuracy falls steeply with label count and granularity for every family. PIE drops from about 88 (5-way) to 71 (27-way) on Banking77. GLiClass-large reaches 0.557 F1 on 77-way Banking77 vs above 0.9 on binary sentiment. The architecture differences reported (about 1-4 points) are smaller than the label-count and domain effects.
- The GLiNER result suggests separate label encoding loses little when label semantics are self-contained (entity types, intent names with descriptions). Joint encoding may help when labels are only meaningful relative to each other or to dataset conventions. That is the authors' speculation, not a tested result.

### Gaps
- No controlled GLiClass uni- vs bi-encoder numbers exist in the GLiClass paper.
- No peer-reviewed or benchmarked description of "Laya" was found; only model cards.
- Laurer et al.'s per-dataset held-out numbers are only in figures and model cards, not extracted here.

## Cost of separating inputs and benefit of interaction layers (poly-encoder, DeFormer, MORES)

### Takeaway
In-domain, separating inputs costs about 1-2 points: DeFormer -1.0 to -1.8 with 9 of 12 layers separated, and MORES 1 interaction block reaches about 95% of BERT. One or two interaction layers over separate representations recover most of the gap. Poly-encoders gain +1-2.3 R@1 over bi-encoders at about 2-4x bi-encoder latency. The only transfer evidence (MORES, with target-domain fine-tuning) says interaction patterns are general and representations are domain-specific. None of these papers tests zero-shot OOD.

### Cited Findings
**Poly-encoder (Humeau et al., arXiv 2019; ICLR 2020, venue from prior knowledge)**
- R@1 with Reddit pre-training (bi / poly-360 / cross): ConvAI2 R@1/20 84.8/86.8/87.9; DSTC7 R@1/100 70.9/71.4/71.7; Ubuntu v2 R@1/10 83.6/85.9/86.5. With BERT weights on DSTC7: bi 66.8, poly-360 68.9, cross 67.4, so the cross-encoder is not always best — [Poly-encoders](https://ar5iv.labs.arxiv.org/html/1905.01969)
- GPU latency at 100k candidates: bi 22 ms, poly-16 38 ms, poly-360 88 ms, cross about 266k ms (inferred). Poly training is 3-4x faster than cross training — [Poly-encoders](https://ar5iv.labs.arxiv.org/html/1905.01969)
- Domain-matched pre-training (Reddit vs Wikipedia/Books) improved all three architectures on all dialogue tasks. The authors attribute this to pre-training data similar to the downstream task. ConvAI2 bi-encoder: 82.0 → 84.8 — [Poly-encoders](https://ar5iv.labs.arxiv.org/html/1905.01969)

**DeFormer (Cao et al., arXiv 2020; ACL 2020, venue from prior knowledge)**
- Lower layers encode question and passage separately; upper layers are joint. BERT-base with 9 of 12 layers separated: SQuAD 88.5 → 87.1, RACE 66.3 → 64.5, BoolQ 77.8 → 76.8, MNLI 84.4 → 82.6, QQP 90.5 → 90.3. Speedups are 2.0-3.5x with 50-73% less memory — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- The F1 cost is negligible when separating up to layer 3 (base) or 13 (large). It rises sharply when only the last few layers are joint — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- Distillation from the full model (KD) plus layerwise representation similarity (LRS) recovers accuracy. SQuAD F1, base: 85.8 without either loss, 86.2 with KD only, 87.1 with both. Large: 87.5 → 88.9 → 90.8. There is no OOD evaluation — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)

**MORES (Gao, Dai, Callan, arXiv 2020; EMNLP 2020, venue from prior knowledge)**
- Separate query and document representation modules plus interaction blocks (IB), reranking the BM25 top 1000. MS MARCO dev MRR@10: BERT 0.3527, 1×IB 0.3334, 2×IB 0.3456, 3×IB 0.3423, 4×IB 0.3307. TREC DL19 NDCG@10: BERT 0.7032 vs 2×IB 0.7026. More than 2 IBs did not help, which the authors attribute to optimisation difficulty — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
- CPU time with precomputed documents at 512 tokens: 698 s → 11 s (1×IB) per 1000 candidates; up to about 170x with projected representations — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)
- Domain adaptation from MS MARCO with 5-fold CV on the target, NDCG@20, Robust04 desc: BERT 0.5065; MORES full adaptation 0.4846; interaction-only 0.4697; representations-only 0.4884. On ClueWeb09-B title: 0.3294 / 0.3317 / 0.3179 / 0.3319. The authors conclude "it is more necessary to have domain-specific representations, while interaction patterns are more general" — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)

### Inferences
- In a separated-encoder design, a new domain mainly needs different representations, not different interaction rules. This favours your design if the state and option encoders are strong and domain-general (e.g. a large pretrained embedder). It also implies that training only the interaction head over frozen encoders cannot fix representation mismatch in a new domain: MORES interaction-only adaptation is the worst variant.
- Distilling a joint teacher's scores into the separated model is the standard fix for the separation cost (DeFormer KD, ColBERTv2 KL), in-domain.

### Gaps
- No zero-shot OOD numbers were found for poly-encoder, DeFormer or MORES. All of their transfer evidence is in-domain or uses target-domain supervision.

## Failure mode: an interaction head over a strong retrieval score learns only training-label-specific corrections

### Takeaway
This is the "explaining away" effect from the product-of-experts debiasing literature. When a trainable model is combined with a fixed scorer (log p + log b), its gradients concentrate on what the fixed scorer does not already explain. For a residual head over a strong retrieval prior, that is mostly the training labels' idiosyncratic errors. The documented counter-measures are:
- residual margins plus error-based hard negatives from the prior (CLEAR)
- soft score distillation from a stronger teacher over retrieved candidates (ColBERTv2, GPL, DeFormer)
- confidence regularisation that keeps in-distribution accuracy (Utama et al.)
- an entropy penalty to stop a learned gate collapsing (Clark et al.)

None of these was evaluated specifically as a defence against label-specific overfitting of a head on unseen domains.

### Cited Findings
- Product-of-experts ensemble with a frozen model b: softmax(log p + log b), so gradients push the main model p toward what b does not explain. The learned-mixin variant gates b with a learned g(x). Without an entropy penalty, g collapses toward 0 and p absorbs the bias (average g 0.13 without the penalty, 5.16 with it, on Adversarial SQuAD). Penalty weights were tuned on the test set — [Clark, Yatskar, Zettlemoyer, arXiv 2019 (EMNLP 2019, venue from prior knowledge)](https://ar5iv.labs.arxiv.org/html/1909.03683)
- OOD gains from that ensemble: VQA-CP 39.18 → 52.05; Adversarial SQuAD AddSent F1 42.54 → 51.84; TriviaQA-CP location EM 41.23 → 47.77. In-domain accuracy dropped (SQuAD dev 80.61 → 75.94; synthetic MNLI 87.90 → 76.65) because the removed signal is useful in-domain — [Clark et al.](https://ar5iv.labs.arxiv.org/html/1909.03683)
- Confidence regularisation (self-distillation scaled by bias-model confidence) improves HANS by 7 points while keeping original in-distribution accuracy. Prior debiasing methods traded in-distribution accuracy for OOD — [Utama, Moosavi, Gurevych, ACL 2020](https://arxiv.org/abs/2005.00315)
- CLEAR trains a dense model on the residual of BM25 with margin m = 1 − 0.1·(s_BM25(q,d+) − s_BM25(q,d−)), so pairs BM25 already ranks well give little gradient. Negatives are sampled from BM25's own top-N errors; random negatives "sharply" reduce accuracy. Test score is 0.5·s_BM25 + s_emb — [Gao et al., CLEAR (arXiv 2020; ECIR 2021, venue from prior knowledge)](https://ar5iv.labs.arxiv.org/html/2004.13969)
- CLEAR results, in-domain only (MS MARCO dev MRR@10): BM25 0.191, dense-only 0.308, constant-margin fusion 0.314, CLEAR 0.338. On TREC DL19 NDCG@10, CLEAR scores 0.699 vs 0.594 dense-only. There is no OOD evaluation — [CLEAR](https://ar5iv.labs.arxiv.org/html/2004.13969)
- Injecting the BM25 score as text into a cross-encoder improves over all CE rerankers and over interpolation on MS MARCO and TREC DL. No OOD evaluation is reported — [Askari et al., ECIR 2023](https://arxiv.org/abs/2301.09728)
- Score distillation over retrieved hard negatives: ColBERTv2 uses KL to CE scores over 64 candidates from its own retrieval, with two refresh rounds — [ColBERTv2](https://ar5iv.labs.arxiv.org/html/2112.01488). GPL's CE MarginMSE soft labels handle false negatives and noisy queries. QGen with hard labels loses 2.5 nDCG points on average when hard negatives are added, while GPL gains — [GPL](https://ar5iv.labs.arxiv.org/html/2112.07577)
- Distillation (KD) plus representation-similarity losses recover 1.3 (base) to 3.3 (large) SQuAD F1 lost by separating inputs — [DeFormer](https://ar5iv.labs.arxiv.org/html/2005.00697)
- Interaction models trained on one candidate distribution degrade when the candidate distribution shifts: CE quality declines and can fall below the first stage as more documents are reranked — [Drowning in Documents, ReNeuIR @ SIGIR 2025](https://arxiv.org/abs/2411.11767)

### Inferences
- A head trained additively over a fixed strong prior (prior logit + head logit, cross-entropy over training labels) is a product of experts with the prior as the fixed expert. Clark et al.'s analysis predicts the head learns only the residual the prior gets wrong on training labels. With few distinct labels, that residual is mostly label-specific (which training intents the prior confuses), not a transferable matching skill. In-domain accuracy improves and OOD gains stay near zero or turn negative. This is an inference from the mechanism; no paper tests it for label heads.
- Mitigations supported by the evidence, by strength:
  - Train the head to match a stronger teacher's full score distribution over retrieved candidates (KL/MarginMSE), not just the gold label. This gives the head a dense, label-agnostic target, as in ColBERTv2 and GPL.
  - Take negatives from the prior's own top-k errors, as in CLEAR and ColBERTv2.
  - Keep a fixed, held-out-tuned interpolation weight rather than a learned gate, or add an entropy or confidence penalty so the prior's weight cannot collapse.
  - Select checkpoints and weights on held-out domains, not held-out examples of training labels. Clark et al. tuned on test, which shows how sensitive these weights are.

### Gaps
- No source was found for "retrieval dropout" (randomly dropping the prior score during training so the head must learn standalone matching) or for curriculum schedules in this setting. These are plausible but unverified here.
- No paper was found that measures OOD generalization of a residual head over a strong embedding prior for classification with unseen labels. CLEAR and Askari et al. evaluate in-domain only.

## Minimum training data for an interaction head to generalize

### Takeaway
Every interaction model with good OOD numbers was trained on hundreds of thousands of queries over millions of documents (MS MARCO: >530k queries, 8.8M passages), or on tens of datasets with hundreds of label sets (Laurer: 33 datasets, 389 classes; PIE: 39k utterances, about 30k pseudo-intents; GLiClass 1.2M and GLiNER-bi 8M examples). It was usually also distilled from a stronger teacher. Diversity of label sets and domains matters more than examples per label: Laurer caps at 500 per class and 5,000 per dataset. No source gives a tested minimum. Small or narrowly trained interaction models can be far above baselines in-domain and at or below BM25 OOD.

### Cited Findings
- MS MARCO, the training set behind the generalizing CE and late-interaction models: 8.8M passages, >530k training query-relevant pairs — [Rosa et al. 2022](https://ar5iv.labs.arxiv.org/html/2212.06121)
- DPR, trained on 4 QA datasets, is -47.7% vs BM25 on BEIR. MS MARCO-trained dense models are -2.8% (TAS-B) to -7.4% (ANCE) — [BEIR](https://ar5iv.labs.arxiv.org/html/2104.08663)
- Data efficiency, in-domain (MS MARCO): with 2k training instances, monoT5-base 0.238 vs BERT-base 0.127, below BM25. With 20k: 0.261 vs 0.201. T5 gets about 45% of its gain over BM25 from 4% of the data — [monoT5](https://ar5iv.labs.arxiv.org/html/2003.06713)
- Capacity: monoT5-60M trained on full MS MARCO is below BM25 on BEIR (0.425 vs 0.432); 220M +0.046; 3B +0.100 — [Rosa et al. 2022](https://ar5iv.labs.arxiv.org/html/2212.06121)
- Target-domain synthetic data (GPL): about 250K generated queries per target. 10K passages already give +2.4 nDCG on Robust04 and gains level off around 50K passages. Training saturates around 100K steps (9.6 h on one V100). Over the full BEIR suite, TAS-B+GPL is +4.6 nDCG over TAS-B — [GPL](https://ar5iv.labs.arxiv.org/html/2112.07577)
- Task diversity for unseen labels: adding 28 classification datasets (387 classes, at most 500 texts per class) to NLI data raises held-out-dataset zero-shot by 9.4% on average — [Laurer et al.](https://arxiv.org/html/2312.17543)
- PIE: 39k utterances with 252 gold intents and about 29.6k pseudo-intent names give a 5-way zero-shot Banking77 accuracy of 88.3, vs 27.7 for task-adapted BERT without intent-aware pre-training — [PIE](https://arxiv.org/html/2305.14827v2)
- A few in-domain examples beat architecture changes: 8 shots per label add +0.11 (large) to +0.19 (edge) F1 for GLiClass — [GLiClass](https://arxiv.org/html/2508.07662)
- Domain-matched pre-training data (Reddit for dialogue) improved bi, poly and cross encoders alike by about 3 R@1 — [Poly-encoders](https://ar5iv.labs.arxiv.org/html/1905.01969)
- MORES: interaction patterns learned on MS MARCO transfer; representations need domain data — [MORES](https://ar5iv.labs.arxiv.org/html/2004.13313)

### Inferences
- For your question: a late-interaction head over separately encoded state and option texts is unlikely to beat a strong bi-encoder (e.g. bge) on unseen domains if it is trained on one dataset's label set. Banking77 has 77 labels, so this means tens of labels and thousands of texts. Every positive OOD result above comes from about 10^5-10^6 queries or examples, about 10^2-10^4+ distinct label sets or pseudo-labels across tens of source datasets or a web-scale corpus, and usually teacher score distillation. Inference: a realistic floor for an OOD-generalizing head is many (≥10-30) source datasets or domains with hundreds or more distinct labels in total, trained against teacher scores with prior-mined hard negatives and evaluated leave-one-domain-out. This is extrapolated, not a tested threshold.
- Expected size of the win when conditions are met: OOD gains of late interaction or CE over a comparably trained, CE-distilled bi-encoder are small. ColBERTv2 vs SPLADEv2: about +0.2 average. A same-size CE vs bi-encoder at 220M-3B: +4 points only at the largest scale (Rosa). Gains over an undistilled bi-encoder are larger: ColBERTv2 vs TAS-B, about +6.
- Cheaper alternatives the evidence supports: target-domain synthetic data with teacher distillation (GPL-style: about 10K-50K unlabeled target texts), or a handful of labelled examples per label (GLiClass 8-shot +0.11-0.19 F1). Both moved OOD numbers more than switching architecture.

### Gaps
- No source reports an ablation that varies the number of distinct labels or domains for an interaction head and measures OOD accuracy. Laurer's leave-one-out is the closest, but it does not sweep dataset count.
- No monoT5 data-efficiency results were found for OOD (only in-domain MS MARCO subsets of 2k and 20k).
