# Realistic expectations: zero-shot classification on unseen domains and unseen intent labels (for a ~150M-400M encoder)

Conventions. "Acc" = accuracy, "F1" = macro-F1 unless noted. Venue and year are given for each source. [preprint] = arXiv-only or no peer review verified. [leaderboard/3rd-party] = model card, GitHub benchmark or leaderboard. **Label-count caveat:** Laurer's model card, BTZSC and the Jev 0.870 figure all use a "Banking77 (72)" configuration with 72 label hypotheses. Full Banking77 has 77. jev-benchmarks says the pinned BTZSC Banking77 config "exposes 72 hypotheses", and it excludes 200 rows that have no positive candidate. So 72-label numbers are not directly comparable to full-test 77-label numbers. Chance at N=77 is 1.3%.

## Q1. Zero-shot accuracy on Banking77 / CLINC150 / HWU64 with all labels (no training on them), by system family

### Takeaway
The best published open models at 150M-600M parameters score about 0.62-0.70 on Banking77 with all labels and no Banking77 training: modern bi-encoders (bge/gte/e5/Qwen3-Embedding), gte-reranker-modernbert-base, and GLiNER2. NLI zero-shot cross-encoders score 0.29-0.57 and GLiClass 0.03-0.56. Laya scores 0.36-0.43, and instruction LLMs of 3B-12B only about 0.40. Frontier LLMs and API systems reach about 0.78-0.88: GPT-4o 0.78, Jev 0.79 on the full test (0.87 on the 72-label slice), and 2026 frontier LLMs 0.80-0.88 on a 77-item sample. Clean full-label-set zero-shot numbers for CLINC150 (150 labels) and HWU64 (64 labels) are scarce. Most "zero-shot" numbers on those datasets use 25-50-label test subsets.

### Cited Findings

**Banking77, all labels, no Banking77 training data (dense table)**

| System (family, size) | Banking77 score | Metric / protocol | Source |
|---|---|---|---|
| bge-base-en-v1.5 (bi-enc, 110M) | 0.695 | Acc, full 3,080 test, N=77, humanized label names, cosine | [LIDM docs/RESULTS.md (local M0, 2026)](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| bge-base-en-v1.5 | 0.64 acc / 0.64 F1 | BTZSC, label verbalizer cosine | [BTZSC, ICLR 2026](https://arxiv.org/abs/2603.11991) |
| bge-large-en-v1.5 (335M) | 0.68 / 0.68 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| gte-base-en-v1.5 | 0.65 / 0.66 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| gte-large-en-v1.5 | 0.63 / 0.63 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| gte-modernbert-base (149M) | 0.63 / 0.64 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| e5-base-v2 / e5-large-v2 | 0.62 / 0.56 acc (F1 0.62 / 0.58) | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| all-MiniLM-L6-v2 (22M) | 0.44 / 0.43 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| Qwen3-Embedding-0.6B / -8B | 0.64 / 0.70 acc | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| e5-mistral-7b-instruct | 0.65 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| gte-reranker-modernbert-base (cross-enc reranker, 149M) | 0.63 / 0.65 | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| bge-reranker-large / Qwen3-Reranker-0.6B / -8B | 0.56 / 0.62 / 0.67 acc | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| bart-large-mnli (NLI, 407M) | 0.29 acc; 0.312 F1; 42.2 acc | BTZSC; Laurer card (72 labels); UTCD | [BTZSC](https://arxiv.org/abs/2603.11991); [Laurer v2.0 card, 2024](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0); [UTCD, Findings ACL 2023](https://arxiv.org/abs/2305.16521) |
| deberta-v3-large-nli / modernbert-large-nli (NLI, ~400M) | 0.36 / 0.24 acc | BTZSC | [BTZSC](https://arxiv.org/abs/2603.11991) |
| deberta-v3-large-zeroshot-v2.0 (NLI, 435M) | 0.567 F1 (72 labels); "-c" variant 0.513 | Laurer card, "pure zero-shot" run without the 28 eval datasets | [Laurer v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) |
| deberta-v3-large-zeroshot-v2.0 | 0.42 | GLiNER2 paper, own protocol | [GLiNER2, EMNLP 2025 Demos](https://arxiv.org/abs/2507.18546) |
| deberta-v3-base-zeroshot-v2.0 / -c | 0.446 / 0.421 F1 | Laurer card | [Laurer v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) |
| roberta-large-zeroshot-v2.0-c / bge-m3-zeroshot-v2.0-c | 0.29 / 0.387 F1 | Laurer card | [Laurer v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) |
| gliclass-large-v3.0 (439M) / base-v3.0 (187M) | 0.5574 / 0.4698 F1 | GLiClass paper Table 3 | [GLiClass, arXiv 2025 [preprint]](https://arxiv.org/abs/2508.07662) |
| gliclass-modern-large-v3.0 (399M) / modern-base (151M) / edge (32.7M) | 0.3683 / 0.3561 / 0.0272 F1 | GLiClass paper Table 3 | [GLiClass](https://arxiv.org/abs/2508.07662) |
| gliclass-base-v1.0 | 0.21 | GLiNER2 paper | [GLiNER2](https://arxiv.org/abs/2507.18546) |
| GLiNER2 (205M, trained on 254k real + GPT-4o-annotated/synthetic examples) | 0.70 | GLiNER2 paper | [GLiNER2](https://arxiv.org/abs/2507.18546) |
| GLiNER2.5-multi-v1 | 0.610 | Acc, BTZSC 72-label, 100-example pilot | [jev-benchmarks [3rd-party], 2026](https://github.com/AbdelStark/jev-benchmarks) |
| Laya (ModernBERT-large, 421M) | 0.425 (model card); 0.360 (LIDM M0) | 77 labels, default 192/256-token head budget | [Laya card [3rd-party]](https://huggingface.co/convaiinnovations/laya); [LIDM RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| Laya fine-tuned on a non-Banking mixture | 0.502 | Acc, N=77 | [LIDM RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md) |
| TypeSafe Jev 1.13 (closed API, size unknown) | 0.870 | Acc, BTZSC 72-label, 100 examples | [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks); quoted in [Laya card](https://huggingface.co/convaiinnovations/laya) |
| Jev "jev-latest" | 0.792 | Acc, **full official 3,080 test** (Sep 29 2026) | [DMB, nibzard [3rd-party], 2026](https://github.com/nibzard/decision-model-benchmark) |
| GPT-4o | 0.78 | GLiNER2 paper, zero-shot | [GLiNER2](https://arxiv.org/abs/2507.18546) |
| GPT-4o (tool calling, intent names only, some intents held out as OOD) | ID F1 77.2; ALL F1 69.4 | 770-item test (10 per label) | [Sali, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.791.pdf) |
| GPT-6 Sol / GPT-5.6 Terra / Gemini 3.8 Flash / Gemini 3.5 Flash-Lite / GPT-6 Luna / Jev 1.13 | 85.7 / 88.3 / 88.3 / 77.9 / 80.5 / 80.5 % | Acc, 77-item Banking sample, Sep 26 2026 pilot | [DMB [3rd-party]](https://github.com/nibzard/decision-model-benchmark) |
| GPT-3 (davinci) | 23.7 | UTCD, zero-shot | [UTCD](https://arxiv.org/abs/2305.16521) |
| Llama-3.2-3B / Phi-4-mini / Qwen3-4B / Qwen3-8B / Mistral-Nemo-12B | 0.41 / 0.43 / 0.40 / 0.40 / 0.39 acc | BTZSC, next-token over enumerated options | [BTZSC](https://arxiv.org/abs/2603.11991) |
| 41 open-weight LLMs (≤~9B; full label list in prompt, free-text exact match, first 500 test items) | cohort mean 0.388; no model >80% | Acc | [Ganesh et al., arXiv Jul 2026 [preprint]](https://arxiv.org/abs/2607.27421) |
| Base encoders without matching training (bert-large, deberta-v3-large, ModernBERT-large) | 0.03 | BTZSC (near chance) | [BTZSC](https://arxiv.org/abs/2603.11991) |

**CLINC150 / HWU64 / related datasets, all labels**
- CLINC150, 150 intents + out-of-scope (OOS): jev-latest scored 88.6% overall on 5,500 queries including 1,000 OOS, and 90.3% on in-scope queries. OOS precision was 90.3% and recall 81.2% — [DMB [3rd-party]](https://github.com/nibzard/decision-model-benchmark)
- CLINC150: zero-shot GPT-4o scored ID F1 88.5 and ALL F1 83.7. A frozen all-distilroberta encoder with a trained head, i.e. supervised, scored ID F1 95.9 — [Sali, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.791.pdf)
- CLINC150: the 41 open-weight LLMs averaged 0.468 accuracy (Banking77 0.388; SNIPS 0.807, where 32 of 41 models exceed 80%). The paper says no model exceeds 80% on Banking77 or MASSIVE — [Ganesh et al. 2026 [preprint]](https://arxiv.org/abs/2607.27421)
- HWU64 with all 64 labels: no clean zero-shot number was found for any family. Proxies:
  - UTCD "NLU Eval" (68 labels, the source corpus of HWU64) out-of-domain: BERT cross-encoder 69.6-73.1; bi-encoder 61.9-71.1; BART-MNLI 40.1; GPT-3 37.9. The supervised upper bound is 88.9 — [UTCD, Findings ACL 2023](https://arxiv.org/abs/2305.16521)
  - MASSIVE intent (59-60 labels): BTZSC accuracy bge-base 0.56, gte-modernbert-base 0.65, Qwen3-Embedding-8B 0.71, Qwen3-Reranker-8B 0.72, deberta-v3-large-nli 0.44, bart-large-mnli 0.43, Qwen3-8B 0.41 — [BTZSC](https://arxiv.org/abs/2603.11991)
  - MASSIVE F1: deberta-v3-large-zeroshot-v2.0 0.518, bart-large-mnli 0.43 — [Laurer card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0). gliclass-large-v3.0 0.5649 — [GLiClass](https://arxiv.org/abs/2508.07662). Laya English 0.783 — [Laya card](https://huggingface.co/convaiinnovations/laya); whether MASSIVE was in Laya's training data is not stated.
- Family-level BTZSC summary (22 datasets, ICLR 2026): embedding models saturate at about 0.60-0.62 average F1 above a few hundred million parameters (Qwen3-Embedding-8B 0.59 vs 0.6B 0.58). The best reranker, Qwen3-Reranker-8B, reaches 0.72. NLI cross-encoders plateau (best 0.60), and on intent tasks "F1 ≈ 0.4-0.55". LLMs at 4-12B reach up to 0.67 average but "lag somewhat on intent" — [BTZSC](https://arxiv.org/abs/2603.11991)
- Laurer's "pure zero-shot" protocol: models without "-c" were evaluated once trained without any data from the 28 evaluation datasets (pure zero-shot). They were evaluated again after training with up to 500 examples per class from those datasets ("fewshot"). The metric is f1_macro — [Laurer v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0); method paper [Laurer et al., arXiv 2023 [preprint]](https://arxiv.org/abs/2312.17543)
- Contamination caveat for Jev: "Public benchmark data may have appeared in either model's training data" — [jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks)
- Unverified (search snippet only): a Barclays ICAIF 2024 paper reports Llama-3 zero-shot Banking77 at about 54.55 micro-F1 — [Srivastava, ICAIF 2024](https://dl.acm.org/doi/pdf/10.1145/3677052.3698608)

### Inferences
- With unseen fine-grained labels and ≤600M parameters, the ranking is: retrieval-trained bi-encoders and modern rerankers (0.62-0.70) > NLI universal classifiers (0.29-0.57) ≈ GLiClass (0.21-0.56) > Laya (0.36-0.43). The best published small specialised model, GLiNER2, ties the bi-encoders at 0.70. Open LLMs at 3-12B are worse than a 110M bi-encoder on Banking77 (about 0.40).
- So for a 150M-400M encoder, ~0.70 at N=77 is the current open-model frontier, not a floor. About 0.78-0.80 (GPT-4o, Jev full test) is the API frontier. 0.85-0.88 appears only for 2026 frontier LLMs on a 77-item sample (wide confidence interval).
- Protocol noise is about ±5 points. bge-base gives 0.695 in LIDM M0 vs 0.64 in BTZSC, and deberta-v3-large-zeroshot-v2.0 gives 0.567 F1 (Laurer card) vs 0.42 (GLiNER2). Jev gives 0.870 on the 72-label 100-item slice vs 0.792 on the full test. Compare targets only within one fixed protocol: same label strings, same label count, full 3,080 test.

### Gaps
- No peer-reviewed full-test (3,080, 77-label) Banking77 zero-shot number was found for GPT-4/Claude/GPT-5-class models. Available: GPT-4o 0.78 (GLiNER2; subset and protocol not fully specified), GPT-4o on a 770-item OOS-split variant, and a 77-item sample in DMB.
- No clean all-64-label HWU64 zero-shot result was found for any model family. CLINC150 all-label zero-shot exists only for LLMs and Jev.
- A search summary claimed top-model values of 0.836 (CLINC150) and 0.708 (Banking77) in Ganesh et al. 2026. These values were not found in the extracted text and are unverified.
- Jev's architecture, size and training data are undisclosed.
- BTZSC's Table 1 lists Banking77 with 77 labels, while jev-benchmarks reports the pinned BTZSC config exposes 72 hypotheses. Which applies to the BTZSC per-model numbers was not resolved.

## Q2. Few-shot and supervised reference points on the same datasets

### Takeaway
Fully supervised small encoders reach about 93-94% on Banking77, 97% on CLINC150 and 92-93% on HWU64. With 5-10 labeled examples per class, small encoders reach 80-88% on Banking77 (SetFit, CPFT, ConvFiT, GLiClass 8-shot, and a frozen bge + logistic-regression probe). Retrieval-ICL with 70B LLMs reaches 89-92%. So the best open small zero-shot systems (~0.70) trail 10-shot by about 15-18 points and full supervision by about 24 points.

### Cited Findings

| Setting | Banking77 | CLINC150 | HWU64 | Source |
|---|---|---|---|---|
| Full data, BERT-tuned | 93.66 | 96.93 | 92.10 | [Casanueva et al., NLP4ConvAI 2020](https://arxiv.org/abs/2003.04807) |
| Full data, USE+ConveRT (dual sentence encoders) | 93.36 | 97.16 | 92.62 | [Casanueva et al. 2020](https://arxiv.org/abs/2003.04807) |
| Full data, ConvBERT(+Pre/+Multi/DG) best | 93.44 | 97.13 | 92.94 | [DialoGLUE, arXiv 2020 [preprint]](https://arxiv.org/abs/2009.13570) |
| Full data, CBEO (Mehri & Eric 2021) | 93.83 | 97.31 | 93.03 | as cited in [LAD, arXiv 2022 [preprint]](https://arxiv.org/abs/2207.14393) |
| Full data, MPNet-v2 fine-tuned | 94.1 µF1 | – | – | [Loukas et al., ICAIF 2023](https://arxiv.org/abs/2311.06102) |
| 30-shot, BERT-tuned / USE+ConveRT | 90.03 / 90.57 | 95.49 / 96.13 | 88.27 / 90.16 | [Casanueva et al. 2020](https://arxiv.org/abs/2003.04807) |
| 20-shot SetFit (MPNet) | 91.2 µF1 | – | – | [Loukas 2023](https://arxiv.org/abs/2311.06102) |
| 10-shot BERT-tuned / USE+ConveRT | 83.42 / 85.19 | 91.93 / 93.26 | 84.86 / 85.83 | [Casanueva et al. 2020](https://arxiv.org/abs/2003.04807) |
| 10-shot CPFT (contrastive pre-train + fine-tune) | 87.20 | 94.18 | 87.13 | [Zhang et al., EMNLP 2021](https://arxiv.org/abs/2109.06349) |
| 10-shot DNNC / ConvBERT+Combined | 86.71 / 85.95 | 93.76 / 93.97 | 84.72 / 86.28 | [CPFT Table 2](https://arxiv.org/abs/2109.06349) |
| 10-shot ConvFiT (reported) | 87.38 | 92.89 | 85.32 | as cited in [Milios et al., arXiv 2023](https://arxiv.org/abs/2309.10954) |
| 10-shot SetFit | 84.51 (Milios); 88.1 µF1 (Loukas) | 90.67 | 83.35 | [Milios 2023](https://arxiv.org/abs/2309.10954); [Loukas 2023](https://arxiv.org/abs/2311.06102) |
| 10-shot DeBERTa-XXL + Pfeiffer adapters | 88.41 | 95.05 | 86.93 | [Milios 2023](https://arxiv.org/abs/2309.10954) |
| 10-shot SBERT 1-NN (no training) | 85.39 | 84.84 | 75.46 | [Milios 2023](https://arxiv.org/abs/2309.10954) |
| 10-shot retrieval+ICL, LLaMA-2-70B 4K (20 retrieved examples) | 92.11 | 98.18 | 91.73 | [Milios 2023](https://arxiv.org/abs/2309.10954) |
| 10-shot retrieval+ICL, LLaMA-2-7B | 89.45 | 95.20 | 87.82 | [Milios 2023](https://arxiv.org/abs/2309.10954) |
| 8-shot frozen-embedding logistic-regression probe (= MTEB Banking77Classification) | bge-large 87.79; bge-base 86.95; gte-base 85.07; e5-large-v2 84.55; e5-base-v2 83.53 | – | – | [bge-base card](https://huggingface.co/BAAI/bge-base-en-v1.5), [bge-large card](https://huggingface.co/BAAI/bge-large-en-v1.5), [gte-base card](https://huggingface.co/thenlper/gte-base), [e5-base-v2 card](https://huggingface.co/intfloat/e5-base-v2), [e5-large-v2 card](https://huggingface.co/intfloat/e5-large-v2) [leaderboard]; protocol per [MTEB source](https://github.com/embeddings-benchmark/mteb/blob/1.12.0/mteb/abstasks/AbsTaskClassification.py) |
| 8-shot GLiClass (F1) | large-v3 0.8563; modern-large 0.8354; base 0.7975; modern-base 0.7683; edge 0.6644 (zero-shot 0.5575 / 0.3653 / 0.4689 / 0.3581 / 0.0255) | – | – | [GLiClass Table 5 [preprint]](https://arxiv.org/abs/2508.07662) |
| 5-shot CPFT / USE+ConveRT / RoBERTa | 80.86 / 77.75 / 74.04 | 92.34 / 90.49 / 87.99 | 82.03 / 80.01 / 75.56 | [CPFT](https://arxiv.org/abs/2109.06349) |
| 5-shot SetFit | 79.89 (Milios); 83.5 µF1 (Loukas) | 88.68 | 78.38 | [Milios](https://arxiv.org/abs/2309.10954); [Loukas](https://arxiv.org/abs/2311.06102) |
| 5-shot retrieval+ICL LLaMA-2-70B 4K | 88.96 | 97.56 | 90.61 | [Milios 2023](https://arxiv.org/abs/2309.10954) |
| 3-shot SetFit / GPT-4 (231 examples in prompt) / GPT-3.5 | 76.7 / 83.1 / 65.5 µF1 | – | – | [Loukas 2023](https://arxiv.org/abs/2311.06102) |
| 1-shot SetFit / GPT-4 / Claude 2 / GPT-3.5 | 57.4 / 80.4 / 76.8 / 75.2 µF1 | – | – | [Loukas 2023](https://arxiv.org/abs/2311.06102) |
| 1-shot CBEO | 31.36 | 53.96 | 43.12 | [LAD 2022](https://arxiv.org/abs/2207.14393) |
| ≤500 ex/class mixed into a universal NLI model (deberta-v3-large / base zeroshot-v2.0; bge-m3) | 0.766 / 0.751 / 0.715 F1 (72 labels) | – | – | [Laurer v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) |
| GPT-4o 10-shot (ID F1, OOS split) | 90.6 | 94.8 | – | [Sali, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.791.pdf) |

- MTEB protocol: the default `AbsTaskClassification` trains `LogisticRegression(max_iter=100)` on `samples_per_label = 8` examples per label sampled from the train split. Results are averaged over `n_experiments = 10`, and the Banking77Classification task does not override these defaults (checked in mteb 1.12.0 and current main). So MTEB Banking77 numbers are 8-shot linear probes on frozen embeddings, **not** zero-shot. BTZSC flags the same: MTEB "incorporate[s] labeled examples through supervised probes" — [MTEB source](https://github.com/embeddings-benchmark/mteb/blob/1.12.0/mteb/abstasks/AbsTaskClassification.py); [BTZSC](https://arxiv.org/abs/2603.11991)
- Loukas: 20-shot SetFit (91.2) comes within 3 points of full-data fine-tuning (94.1). GPT-4 beats 1-shot SetFit by more than 20 points — [Loukas 2023](https://arxiv.org/abs/2311.06102)
- Milios: in the 5-shot Banking setting, LLaMA-2-70B retrieval+ICL leads fine-tuned DeBERTa-XXL by 7.49 points — [Milios 2023](https://arxiv.org/abs/2309.10954)
- On other intent datasets (HINT3/AID3), adaptive-ICL LLMs (Claude v2/v3, Mistral Large) reach about 0.71-0.74 average F1 vs SetFit 0.60 (0.658 with negative augmentation) — [Arora et al., arXiv 2024 [preprint]](https://arxiv.org/abs/2410.01627)

### Inferences
- On Banking77: open small zero-shot (~0.64-0.70) → 5-shot (~0.80-0.84) → 10-shot (~0.85-0.88) → full data (~0.93-0.94). The first 5-10 real examples per class are worth about +15-20 points. Everything after that adds about +6.
- A frozen 110M bge-base scores 0.64-0.70 zero-shot and 0.87 with an 8-shot logistic-regression probe. So most of the zero-shot gap comes from **label-side** representation (mapping a terse label name to its utterance region), not utterance encoding quality.
- Universal NLI classifiers stay at about 0.77 F1 even with ≤500 in-domain examples per class (multi-task), far below dedicated fine-tuned heads (0.93-0.94). The pairwise text-hypothesis format itself caps fine-grained 72-77-way accuracy.

### Gaps
- The SetFit paper's own tables (Tunstall et al. 2022) were not checked for Banking77/CLINC/HWU. The SetFit numbers above come from Loukas 2023 and Milios 2023.
- The MTEB model-card numbers were produced in 2023 with earlier mteb versions. The 8-per-label default was verified in the 1.12.0 code, not in the exact version used.
- No 8/16-shot SetFit numbers were found for HWU64 or CLINC150 with modern backbones (bge/gte).

## Q3. Cross-domain / cross-dataset intent transfer (train on some intent datasets or domains, test on an unseen one)

### Takeaway
Training on public intent data from other domains and testing on unseen Banking77 has given about 0.50 accuracy with all 77 labels for BERT-base-scale cross-encoders (UTCD 2023) and about 0.56-0.58 for RoBERTa-base with paraphrase pretraining (ALT 2023). The best is about 0.70 for GLiNER2 (2025, broad GPT-4o-annotated data), and the same level is reached by off-the-shelf retrieval bi-encoders with no intent training. On 25-50-label subsets, transfer is 69-85% for encoders and 84-88% for 3B-7B fine-tuned LLMs. The measured gap to in-domain supervision is about 20-40 points.

### Cited Findings
- **UTCD** (Clarke et al., Findings ACL 2023). BERT-base-scale models were trained on 9 in-domain datasets, including the intent sets SGD (26), CLINC-150 (150) and SLURP (75), then tested zero-shot on out-of-domain sets. Accuracy is "aspect-normalized":
  - Banking77 (77): Binary BERT (cross-encoder) 51.4 vanilla / 51.3 implicit / 49.8 explicit. Bi-encoder 9.4 / 21.1 / 9.7. GPT-2 22.8 / 16.5 / 22.4. BART-MNLI zero-shot 42.2, GPT-3 zero-shot 23.7. Supervised upper bounds: individual 88.6, multi-task "full" 84.7. The gap is about 37 points.
  - SNIPS (7): 82.5-83.8 vs 99.0 supervised.
  - NLU Eval (68): 69.6-73.1 vs 88.9.
  - Topic OOD is worse, e.g. Patent 13.6-19.5 vs 64.1.
  - Source: [UTCD](https://arxiv.org/abs/2305.16521)
- **Hong et al. (SIGDIAL 2024), cross-dataset**:
  - Setup: FLAN-T5 3B was fine-tuned on CLINC150 intents outside the 50 test intents, with "dependent" descriptions. Testing used the top-10 ranked cleansed descriptions. The test sets follow Sung et al.: CLINC N=50, HWU N=25, BANKING N=27.
  - Zero-shot (K=0) accuracy: 97.58 / 87.92 / 84.72 without the ranker; 96.46 / 86.23 / 84.88 with it. Llama-2-7B (Alpaca + intent fine-tuning) scored 96.38 / 85.92 / 83.61.
  - Baselines on the same splits: L-BERT-TAPT 79.5 / 63.1 / 70.1; L-SBERT-Paraphrase 84.5 / 67.5 / 77.4; L-PIE 86.5 / 70.6 / 77.6.
  - **Leakage caveat:** the CLINC "seen domain" for seed 42 includes "credit cards, banking, auto and commute…" (Appendix Table 4), so the BANKING transfer is not domain-clean.
  - Source: [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **Hong et al., within-CLINC unseen-domain split.** The model was trained on 75 intents from 5 domains and tested on 75 intents from the other 5 domains. Accuracy is 84.28% with names only and 90.55% when tested with dependent descriptions. A model trained with dependent descriptions scores 82.1 → 86.99 → 91.75 (no / independent / dependent test descriptions) — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **PIE** (Sung et al., EMNLP 2023). paraphrase-mpnet-base-v2 (110M) was pre-trained on TOP/TOPv2, DSTC11-T2 and SGD: 39k utterances, 252 gold intents and 29.6k pseudo-intents. Prototypes come from label names, with no target data. Intent-name overlap with the targets is 3/77 Banking77, 2/64 HWU64 and 6/150 CLINC150.
  - N-way zero-shot over all **test-split** classes: Banking77 N=27 71.4, HWU64 N=25 68.7, Liu54 N=18 71.4, CLINC150 N=50 83.1.
  - The L-SBERT-Paraphrase starting point scores 68.9 / 61.4 / 66.7 / 76.3, so the gain is +5.4 average.
  - The ablation credits most of the gain to TOP (+4.8 average).
  - Source: [PIE](https://arxiv.org/abs/2305.14827)
- **ALT / One-to-All** (Du et al., arXiv 2023 [preprint]). RoBERTa-base was pre-trained on OOD intent data, e.g. for Banking77: HWU64 + CLINC150 with the "Banking" and "Credit Cards" domains removed, and/or QQP paraphrases. Zero-shot results:
  - Without that pre-training, all methods are near chance: One-to-All 6.04 / 13.05 / 5.84 and others 1-3% on Banking / HWU / CLINC.
  - With it, figure values (extracted from the figure; the method-to-value mapping is inferred from order): QQP only 48.1 / 35.8 / 50.7; OOD 56.4 / 55.5 / 61.5; QQP+OOD 58.3 / 56.6 / 65.4. ContextTE / ParallelTE (OOD) score about 50-62.
  - Source: [ALT](https://arxiv.org/abs/2309.03563)
- **GLiNER2** (205M; 254,334 training examples, 135,698 real texts annotated with GPT-4o plus 118,636 synthetic) reaches 0.70 on Banking77, 0.83 on SNIPS and 0.53 on Amazon-Intent (31 labels) zero-shot. GPT-4o scores 0.78 / 0.97 / 0.72 — [GLiNER2](https://arxiv.org/abs/2507.18546)
- **LIDM Stage 1 (local, 2026)**:
  - Training: 974k questions with Banking77 held out (0 shared labels or texts), encoder LoRA.
  - Banking77 N=77 is 0.434 (N=20 0.650) vs bge alone 0.695.
  - Seen-vs-unseen gold label on UTCD validation: nlu_evaluation 0.843 vs 0.656, snips 0.972 vs 0.632.
  - Before LoRA (frozen encoder), unseen-gold nlu_evaluation was 0.135, and 69% of unseen-gold questions were answered with a seen label.
  - Source: [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- **Domain-adaptive pretraining (DialoGLUE, few-shot, about 10 examples per intent).** Effects on Banking77 / HWU64 / CLINC150:
  - Base: BERT 79.87 / 81.69 / 89.52; ConvBERT 83.63 / 83.77 / 92.10.
  - With task-adaptive MLM ("+Pre"): BERT 80.72 / 83.05 / 89.73.
  - With MLM on all DialoGLUE dialogue data ("-DG"): BERT-DG 81.47 / 83.23 / 90.57; ConvBERT-DG 84.42 / 85.17 / 92.87, and with +Pre 84.74 / 85.63 / 93.16.
  - Full-data gains are about 0 (e.g. BERT 93.02 vs BERT-DG 91.75 on Banking77).
  - Source: [DialoGLUE](https://arxiv.org/abs/2009.13570)

### Inferences
- Measured gaps between cross-dataset zero-shot and in-domain supervised on Banking77:
  - UTCD BERT-base 51 vs 88.6 (−37).
  - Best open small models ~70 vs 93-94 (−24).
  - 27-way subset, PIE 71.4 vs the in-domain 1-shot ProtAugment-tuned L-PIE 82.9 (−11.5 vs 1-shot).
  - LIDM's own seen/unseen label gap on UTCD validation is 19-34 points.
- Transfer is much better between general-assistant sets (CLINC ↔ HWU/SNIPS) than into fine-grained single-domain banking intents. Banking77 is the hard case, and CLINC's own banking/credit-card domains are the main leakage risk when CLINC is in training.
- The small open models that transfer best to Banking77 (bge/gte bi-encoders, GLiNER2) owe it to very large and diverse pretraining or annotation, not to intent-specific data. Intent-specific pretraining (PIE, ALT) adds only about +2-8 points on top of a paraphrase or retrieval encoder.

### Gaps
- No result was found that trains a 150M-400M encoder on a large public multi-task mixture and evaluates on full-77 Banking77, all-64 HWU64 and all-150 CLINC150 under one protocol. Most cross-dataset work uses the 27/25/50 test-split subsets from Sung et al.
- The "L-PIE" numbers are on test-split subsets, not full label sets. No full-77 PIE number exists.
- DialoGLUE-style zero-shot transfer (no target examples) was not reported. DialoGLUE is few-shot and full-data only.

## Q4. Degradation with label count (N=5→150) and the effect of label descriptions

### Takeaway
Zero-shot accuracy falls steeply and roughly log-linearly with candidate count. Retrieval bi-encoders lose about 5-6 points per doubling (bge-base 0.93 at N=5 → 0.70 at N=77). Packed-sequence readers like Laya collapse beyond about 20 options (0.87 → 0.36). Label granularity matters more than raw count: CLINC150 with 150 broad intents is easier than Banking77 with 77 confusable ones. Good descriptions are worth about +6-10 points, but only for models trained to read them and only when the descriptions are written to separate confusable neighbours.

### Cited Findings
- **LIDM M0 N-sweep** (Banking77, gold + random distractors, 1,000 items; accuracy at N=5 / 10 / 20 / 40 / 77) — [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md):
  - bge-base: 0.931 / 0.887 / 0.844 / 0.753 / 0.695
  - Laya: 0.871 / 0.817 / 0.746 / 0.590 / 0.360
  - Laya fine-tuned: 0.893 / 0.837 / 0.776 / 0.657 / 0.502
  - LIDM L1: 0.817 / 0.735 / 0.650 / 0.562 / 0.434
- **PIE, 5-way vs N-way zero-shot**, label-name prototypes — [PIE](https://arxiv.org/abs/2305.14827):
  - L-PIE: Banking 88.3 → 71.4 (N=27); HWU 87.7 → 68.7 (N=25); Liu54 83.8 → 71.4 (N=18); CLINC 96.5 → 83.1 (N=50)
  - L-SBERT-Paraphrase: 86.9 → 68.9; 83.6 → 61.4; 79.9 → 66.7; 94.3 → 76.3
- **Hong et al. (SIGDIAL 2024), top-k shortlist** from an off-the-shelf mpnet-base-v2 ranker over descriptions — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf):
  - Accuracy by k: CLINC (75 unseen intents) 44.21 at k=1, ~90 at k≈13. BANKING 50.83 at k=1, ~80 near k=10. HWU 26.51 at k=1, ~80 near k=15.
  - The ranker cuts input length by about 75%, and accuracy is the same as with all options (CLINC 96.46 vs 97.58; BANKING 84.88 vs 84.72).
- **Laya**: at default settings, a 77-option question gets "(256 - 16) // 77 ≈ 3-4 tokens per label", and accuracy falls to 0.425. Jev is "better suited for 50+ options" — [Laya card](https://huggingface.co/convaiinnovations/laya)
- **GLiClass**: "performance can degrade with larger label sets, as seen in datasets like banking77". It is attributed to positional encoding and attention limits, with a context limit of about 1,024 tokens. Throughput from 1 → 128 labels: GLiClass −7% to −20%, while deberta-v3-base-zeroshot-v2.0 falls from 24.55 to 0.47 ex/s (about 52× slower) — [GLiClass](https://arxiv.org/abs/2508.07662)
- **GLiNER2 CPU latency** from 5 → 50 labels: DeBERTa NLI 1,714 → 16,897 ms; GLiClass 137 → 190 ms; GLiNER2 130 → 208 ms; GPT-4o 358 → 463 ms — [GLiNER2](https://arxiv.org/abs/2507.18546)
- **Granularity beats count**:
  - Open-LLM cohort means: SNIPS (7 labels) 0.807, CLINC150 (150) 0.468, Banking77 (77) 0.388 — [Ganesh et al. 2026 [preprint]](https://arxiv.org/abs/2607.27421)
  - UTCD, same models: SNIPS ~83 vs Banking77 ~50 — [UTCD](https://arxiv.org/abs/2305.16521)
  - Laurer card F1: Manifesto (56 labels) 0.258, MASSIVE (59) 0.518, Banking77 (72) 0.567 for deberta-v3-large-zeroshot-v2.0 — [Laurer card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
  - GPT-3 augmentation struggles to separate "confounding intents" on the more granular HWU64 and Banking77 — [Sahu et al., NLP4ConvAI 2022](https://arxiv.org/abs/2204.01959)
- **Descriptions** — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf):
  - Unseen-domain CLINC (75 intents), FLAN-T5 3B. Trained without descriptions: 84.28 with names only → 84.15 with "independent" descriptions → 90.55 with "dependent" (neighbour-contrasting) ones.
  - Trained with dependent descriptions: 82.1 → 86.99 → 91.75.
  - Test-time description quality matters more than training-time: train independent / test dependent scores 90.97, vs 86.99 the other way round.
  - Multiple descriptions per intent do not help over one cleansed description (85.79 vs 85.28).
- **LIDM M0**: for Laya, label descriptions were "invisible at 77 options" because of truncation — [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md). The companion note on label descriptions found no clean names-vs-descriptions ablation for bi-encoders or late-interaction models on Banking77/CLINC150/HWU64 — [research_notes/Classification datasets with label descriptions/label_description_evidence.md](/home/ratlinghitman/code/LIDM/research_notes/Classification%20datasets%20with%20label%20descriptions/label_description_evidence.md)
- **Label-format sensitivity**: writing labels as `card_arrival` instead of `card arrival` moved Laya at N=77 from 0.377 to 0.360 accuracy, and ECE from 0.51 to 0.18 — [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)

### Inferences
- Rule of thumb from the bge-base curve: accuracy at N=77 ≈ accuracy at N=5 minus about 24 points for a retrieval-style scorer, about −6 per doubling. A model that stays above 0.85 at N=20 should land at about 0.70-0.75 at N=77 if its curve is bi-encoder-shaped. Laya and LIDM-L1 drop faster, about −9 to −12 per doubling.
- The shortlist curves in Hong et al. show k ≈ 10-15 candidates keep nearly all the attainable accuracy. Top-k recall of the retriever is therefore the binding constraint for a shortlist + reranker design. A separate study reports bge-style Recall@15 of about 97% on Banking77 with label definitions (search snippet: [Label-Centric Knowledge Retrieval, arXiv 2026 [preprint]](https://arxiv.org/pdf/2610.16182), not verified in full text).
- Descriptions are a test-time lever of about 6-10 points when the model was trained to consume them. They matter most where labels are confusable, so their value on Banking77 is probably at the upper end of that range. This is inferred: no Banking77 full-label ablation exists.

### Gaps
- No published zero-shot curve covers N=5→150 on the same items with descriptions vs names for an encoder model. Hong gives only shortlist-k curves, and PIE only 5-way vs test-split N-way.
- The Label-Centric Knowledge Retrieval Recall@15 figure (97.18% on Banking77) comes from a search snippet only.

## Q5. Implied realistic target range for a 150M-400M encoder trained on public data for an unseen banking-intent domain, and what closes the gap

### Takeaway
For Banking77 at N=77 (full 3,080 test, label names only, no banking data) the targets are:
- **0.65-0.72 is realistic.** That is matching or slightly beating bge/gte (0.64-0.70) and GLiNER2 (0.70).
- **0.75-0.80 is a stretch.** It would match GPT-4o or Jev, and no open ≤500M model has published it.
- **>0.85 requires in-domain signal**: 5-10 real examples per class, or a frontier-LLM reranker.

At N=20 (gold + 19 random), 0.84-0.88 is realistic, matching bge-base at 0.844. The ingredients add roughly:
- 5-10 real examples per class: +15-20 points (to about 0.85-0.88).
- Good contrastive descriptions: +6-10 points.
- LLM-generated in-domain data: +5-20 points, plateauing at about 0.80-0.82 on Banking77.
- A retrieval shortlist plus a stronger reranker: helps only if the reranker beats the retriever on confusable pairs.
- Domain-adaptive pretraining: +1-2 points.

### Cited Findings
- **Ceiling evidence for small open zero-shot models**:
  - BTZSC: embedding models "largely saturate around 0.60-0.62 F1" (all tasks) beyond a few hundred million parameters. Qwen3-Embedding-8B improves on Banking77 only to 0.70 accuracy vs 0.64 for 0.6B — [BTZSC](https://arxiv.org/abs/2603.11991)
  - GLiNER2 (205M) reaches 0.70 — [GLiNER2](https://arxiv.org/abs/2507.18546)
  - LIDM's goal already sets Banking77 N=77 ≥ 0.70 and N=20 ≥ 0.85, with Jev 0.87 as the stretch goal — [docs/GOALS.md (local)](/home/ratlinghitman/code/LIDM/docs/GOALS.md)
- **LIDM current position**:
  - Stage 1 L1 scores 0.434 at N=77 and 0.650 at N=20.
  - Adding the bge prior to the trained head gives 0.514 / 0.716. Sweeping the prior weight rises monotonically to bge alone (best 0.696) and never meaningfully exceeds it.
  - The head is right on 4.8% of questions where bge is wrong, so an oracle combination would reach 0.743.
  - Source: [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
- **Few real examples**: see the Q2 table (5-shot about 0.80-0.84, 10-shot about 0.85-0.88). GLiClass-large goes from 0.5575 zero-shot to 0.8563 with 8 examples per label. Smaller variants gain proportionally more: edge from 0.0255 to 0.6644, modern-base +47% average relative — [GLiClass](https://arxiv.org/abs/2508.07662)
- **Retrieval shortlist plus reranker**:
  - A top-10 mpnet shortlist keeps full accuracy for a fine-tuned 3B reader (BANKING 84.88 vs 84.72) — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
  - A bge-base k=20 shortlist lifts Laya from 0.360 to 0.610 at N=77, but that is still below bge alone (0.695) — [LIDM docs/RESULTS.md](/home/ratlinghitman/code/LIDM/docs/RESULTS.md)
  - Retrieving similar labeled examples (RAG-ICL): GPT-4 with 20 similar examples scores 87.7 µF1, and Claude 2 85.5. Both need a labeled pool — [Loukas 2023](https://arxiv.org/abs/2311.06102)
  - Hybrid GPT-4o + a supervised baseline on Banking77 reaches ALL F1 83.2-86.3 vs zero-shot GPT-4o 69.4 — [Sali, Findings EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.791.pdf)
- **LLM-generated in-domain data**:
  - LAD: GPT-3 reformulation of 1 seed utterance per intent raises CBEO from 31.36 → 51.17 (Banking77), 53.96 → 68.11 (CLINC150) and 43.12 → 65.50 (HWU64). Full-shot is 93.83 / 97.31 / 93.03 — [LAD 2022 [preprint]](https://arxiv.org/abs/2207.14393)
  - Sahu et al.: in a 10-shot setting, GPT-3 augmentation gives up to about +3.7 (CLINC150) and +6 (SNIPS). On Banking77 and HWU64 it gives "no or little improvement" because of confusable intents. Oracle relabelling of generated data gives "a significant boost" — [Sahu et al., NLP4ConvAI 2022](https://arxiv.org/abs/2204.01959)
  - Loukas: 3 real + 2 GPT-4-generated examples per class score 81.0 vs 76.7 for 3 real. The peak is 81.6 at 3 real + 7 generated, after which quality degrades. 20 real examples score 91.2 vs 78.8 for 3 real + 17 generated — [Loukas 2023](https://arxiv.org/abs/2311.06102)
  - GLiNER2's 0.70 comes with about 119k synthetic and 136k GPT-4o-annotated training examples — [GLiNER2](https://arxiv.org/abs/2507.18546)
- **Domain- and task-adaptive pretraining**:
  - DialoGLUE, Banking77 10-shot: +0.8 to +1.6 for BERT, +0.8 to +1.1 for ConvBERT; ≈0 or negative with full data — [DialoGLUE](https://arxiv.org/abs/2009.13570)
  - PIE intent-aware pretraining: +2.5 on Banking77 27-way zero-shot (68.9 → 71.4), +5.4 average N-way — [PIE](https://arxiv.org/abs/2305.14827)
  - ALT OOD-intent + QQP pretraining takes RoBERTa-base from about 6% to about 58% on Banking77 zero-shot — [ALT [preprint]](https://arxiv.org/abs/2309.03563)
- **Descriptions** (model trained to read them): +6.3 to +9.7 at test time on unseen-domain CLINC — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **Model family/size**: cross-encoder rerankers scale with size (Qwen3-Reranker 0.6B → 8B: Banking77 0.62 → 0.67; BTZSC average 0.61 → 0.72 F1). Embeddings barely scale — [BTZSC](https://arxiv.org/abs/2603.11991)
- **Selective automation**: Jev on the full Banking77 test with a validation-fitted threshold of 0.96 covers 51.2% of traffic at 3.68% error — [DMB [3rd-party]](https://github.com/nibzard/decision-model-benchmark)

### Inferences
- **Targets to state** (Banking77 test, all 3,080 items, humanized names, one fixed protocol):
  - Floor: 0.695, which is bge-base alone. Below it the model loses to a free 110M baseline.
  - Realistic: 0.70-0.72 at N=77 and 0.85-0.88 at N=20.
  - Stretch: 0.75-0.79, matching GPT-4o and Jev on the full test.
  - Not realistic zero-shot: ≥0.85 at N=77. That is 8-10-shot or frontier-LLM territory.
- **Remaining gap** once the realistic target is hit: about 8 points to GPT-4o/Jev, about 15-18 to 10-shot small encoders, and about 22-24 to full supervision (93-94).
- **Which ingredient closes how much** (cheapest first):
  - A strong retrieval prior plus a reranker trained to beat it on confusable pairs is worth +0 to +5 over bge. LIDM's own oracle ceiling is 0.743, and a naive fixed-weight blend gives 0.
  - Contrastive (neighbour-aware) label descriptions: +3-10, if the model is trained on descriptions.
  - LLM-generated utterances for the target labels: +5-15 from a weak start, plateauing at about 0.80-0.82 on Banking77 because of the confusable intents.
  - 5-10 real examples per class (prototype, SetFit or linear probe): +15-20, to about 0.85-0.88.
  - Domain-adaptive MLM: about +1-2.
  - These gains overlap and are not additive; no source measures their combination.
- Small (≤12B) open LLMs are not a better zero-shot fallback than a bi-encoder on Banking77 (about 0.40). A frontier-LLM reranker over a top-10-20 shortlist is the only measured route to about 0.80-0.88 without labels, at about 100× the latency and cost (DMB: 0.85-1.3 s p50 vs Jev's 0.27 s vs LIDM's target ≤35 ms).

### Gaps
- No study isolates each ingredient on a fixed 150M-400M encoder with full-77 Banking77. The gains above are taken from different models and protocols and are not additive.
- No peer-reviewed result was found for training a small encoder on LLM-generated utterances only (zero real examples) and evaluating on full-77 Banking77. LAD starts from 1 real seed per intent, and GLiNER2's synthetic data is generic, not banking-targeted.
- Whether a ≤400M model can pass 0.75 zero-shot at N=77 is untested in public work. The only systems above 0.75 are closed (Jev) or frontier LLMs.
