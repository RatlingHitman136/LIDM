# What a label-reading zero-shot text classifier needs to generalize to unseen labels and unseen domains

Research note status (2026-11-18): every item under "Cited Findings" comes from a page fetched this session. Items under "Gaps" marked **unverified lead** come from prior knowledge, with a candidate URL. They were not confirmed this session and should be verified, or cited as unverified, before use.

Abbreviations: OOD = out-of-domain; ID = in-domain; FPB = Financial PhraseBank; B77 = Banking77; LDT = LabelDescTraining.

---

## 1. How does zero-shot generalization scale with the number and diversity of training tasks / labels / domains? Is there a diversity threshold?

### Takeaway
The number of distinct training tasks drives transfer to new tasks; examples per task saturate early.
- **Super-NaturalInstructions:** unseen-task performance grows log-linearly up to 757 training tasks; about 64 instances per task is enough.
- **Flan-2022:** most of the gain arrives by about 282 tasks, with little more by 1,836.
- **FLAN:** held-out performance keeps rising through 7 task clusters.
- **T0:** more datasets raise median held-out accuracy, but more *prompts per dataset* is what reduces variance from prompt wording.

Label-reading encoder classifiers sit at the low end of this curve: UTCD trains on 9 datasets (about 316 labels), zeroshot-v2.0 on 33 datasets (389 classes), and tasksource is the outlier at 500–600+ tasks. They still lose about 30 points from ID to OOD.

There is no sharp *diversity* threshold. Gains are smooth and log-like, with diminishing returns somewhere between hundreds and thousands of tasks. There is a *capacity* threshold: FLAN instruction tuning hurt held-out performance at 8B parameters or fewer.

### Cited Findings
- **Super-NaturalInstructions (Wang et al., EMNLP 2022):** 1,616 tasks, 76 task types, expert-written instructions. Evaluation trains on some tasks and tests on unseen ones. Tk-Instruct beats InstructGPT by more than 9% while being an order of magnitude smaller. — [Wang et al. 2022](https://arxiv.org/abs/2204.07705)
- SuperNI: "the model generalization performance grows log-linearly as we increase the set of tasks used for training". This still holds at 757 tasks (T5-3B). — [Wang et al. 2022, §7.1](https://arxiv.org/html/2204.07705v3)
- SuperNI: more instances per task stop helping at about **64**; beyond that they mainly add training time and overfitting risk. — [Wang et al. 2022, §7.1](https://arxiv.org/html/2204.07705v3)
- SuperNI: task diversity can substitute for model size. T5-large with 757 tasks scores **48.0 ROUGE-L**, against **48.4** for T5-3B with 128 tasks. — [Wang et al. 2022, §7.1](https://arxiv.org/html/2204.07705v3)
- SuperNI: models trained only on definitions fail on example-only encodings, and the reverse is also true. Training on mixed encodings gives robustness. Two positive examples help; more demonstrations add little; explanations hurt at the sizes tested. — [Wang et al. 2022, §7.2](https://arxiv.org/html/2204.07705v3)
- **Flan-2022 / Flan-PaLM (Chung et al., arXiv 2022):** normalized held-out average over MMLU, BBH, TyDiQA and MGSM, by number of fine-tuning tasks. The paper's conclusion: "the majority of the improvement comes from using up to 282 tasks". It suggests later tasks may not be diverse enough, or that gains mostly come from better expressing pretraining knowledge. — [Chung et al. 2022, §3 / Table 3](https://arxiv.org/html/2210.11416)

  | Model | 0 tasks | 9 | 89 | 282 | 1,836 |
  |---|---|---|---|---|---|
  | 8B | 6.4 | 8.3 | 14.8 | 20.5 | 21.9 |
  | 62B | 28.4 | 29.0 | 33.4 | 37.9 | 38.8 |
  | 540B | 49.1 | 52.6 | 57.0 | 57.5 | 58.5 |
- **FLAN (Wei et al., ICLR 2022):**
  - **Data:** 62 datasets in 12 task clusters.
  - **Cluster ablation:** NLI, closed-book QA and commonsense were held out, and 1 to 7 other clusters were added. Held-out performance generally rose with each cluster and "did not appear to saturate" by 7. The sentiment cluster added little.
  - **Model scale:** instruction tuning *hurt* held-out performance at 422M, 2B and 8B parameters and helped substantially at 68B and 137B. The authors attribute this to small models' capacity being used up learning the training tasks.
  - **Instruction ablations:** "no template" and "dataset name only" fine-tuning were substantially worse than instructions.
  - Exact values appear only in figures.

  — [Wei et al. 2022](https://arxiv.org/html/2109.01652)
- **T0 (Sanh et al., ICLR 2022):**
  - **Data:** P3 has "2073 prompts for 177 datasets". T0, T0+ and T0++ train on 39, 49 and 55 datasets. Held out entirely: NLI, coreference, sentence completion, word-sense disambiguation, plus BIG-bench tasks.
  - **Prompts:** more prompts per dataset "consistently leads to higher median performance". Adding non-original-task prompts also reduces spread on 8 of 11 datasets.
  - **Datasets:** going from 39 to 49 datasets raises the median on all 5 held-out datasets, but spread decreases on only 1 of 5. Going from 49 to 55 decreases spread on only 2 of 5. More datasets "does not consistently make the model more robust to the wording of prompts".

  — [Sanh et al. 2022, §6.2](https://ar5iv.labs.arxiv.org/html/2110.08207)
- **tasksource (Sileo):** the arXiv preprint (2023) reports "more than 500 English tasks". A multi-task encoder fine-tuned on all of them "outperforms every publicly available text encoder of comparable size in an external evaluation"; no numbers on the abstract page. — [Sileo, arXiv 2301.05948](https://arxiv.org/abs/2301.05948)
- tasksource, peer-reviewed version (LREC-COLING 2024, pp. 15655–15684): preprocessing annotations cover more than 600 tasks. One DeBERTa-v3-small tasksource card reports 250k training steps and **70.0** accuracy on a zero-shot-label-NLI task. Both points come from search snippets, not full text. — [Sileo, LREC-COLING 2024](https://aclanthology.org/2024.lrec-main.1361.pdf); [deberta-v3-small-tasksource-nli card](https://huggingface.co/sileod/deberta-v3-small-tasksource-nli)
- **zeroshot-v2.0 / "Building Efficient Universal Classifiers with NLI" (Laurer et al., arXiv preprint 2023/2024):**
  - **NLI data:** 5 datasets (MNLI, ANLI, FEVER-NLI, WANLI, LingNLI), about **885k** pairs, with neutral and contradiction merged.
  - **Classification data:** 28 datasets covering sentiment, emotion, intent, toxicity, bias, topic, factuality and spam. In total, **33 datasets / 389 classes**.
  - **Caps:** 500 texts per class and 5,000 per dataset, giving **51,731** non-NLI texts out of more than 1M. CleanLab removed about 17% (~135k) first.
  - **Inconsistency:** the paper states about 9M pairs per run, which does not match the component counts.

  — [Laurer et al.](https://arxiv.org/html/2312.17543)
- Laurer et al. measured held-out performance with **28 leave-one-dataset-out models** (balanced accuracy).
  - NLI plus classification data "improves zeroshot performance by 9.4%" over NLI-only; points vs relative is not stated.
  - The model is better "both on held-in and held-out tasks".
  - NLI-only does fine on binary sentiment but "struggles" on many-class tasks.
  - Limitations listed: "limited diversity" and English-only data. Some held-out datasets show negative transfer.

  — [Laurer et al.](https://arxiv.org/html/2312.17543)
- **UTCD (Clarke et al., Findings of ACL 2023):** 18 datasets in 3 aspects.
  - **ID training sets:** GoEmotions, TweetEval, Emotion; SGD, CLINC-150, SLURP; AG News, DBpedia, Yahoo.
  - **OOD sets with novel labels:** Amazon Polarity, FPB, Yelp; B77, SNIPS, NLU-Eval; MultiEURLEX, Patent, Consumer Finance.
  - **Label counts:** about 316 ID vs about 208 OOD labels. These are the fetch summary's sum of the paper's Table 1, not stated by the paper. Full UTCD has about 6M/800K train/test examples.

  — [Clarke et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **GLiClass (Stepanov et al., arXiv preprint 2025):**
  - **Pre-training:** **1.2M** examples of a "general-purpose dataset covering text classification, sentiment analysis, and natural language inference".
  - **Post-training:** CommonsenseQA plus 2,000 synthetic logic/NLI examples; 2,000 length-bucketed texts, each with **50 true and 50 false GPT-4o-generated labels**; and MultiNLI.

  — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)

### Inferences
- In terms of task count, label-reading classifiers trained on 9 to 33 datasets sit where the Flan curve is steepest: about 9 to 89 tasks, where 8B Flan gains +1.9 to +8.4 normalized points. tasksource, at 500+ tasks, is near the saturation zone. Adding more distinct label sets or tasks, not more examples, is the lever with the largest expected payoff. This is an extrapolation: no source reports such a sweep for an encoder label-reading classifier.
- T0 separates two axes:
  1. **Dataset/task count** raises the median.
  2. **Wording diversity** (prompts per dataset) reduces sensitivity to phrasing.

  For a classifier that encodes label names or descriptions, the analogue of (2) is several paraphrased names or descriptions per label during training.
- The FLAN capacity result (hurt at ≤8B) concerns decoder LMs learning about 40 generative tasks. It does not transfer directly to encoders doing discriminative label matching. GLiClass and DeBERTa NLI models at 100–400M parameters transfer well, which suggests the matching formulation lowers the capacity requirement. This is an inference.
- No sharp diversity threshold appears in any verified source. The curves are smooth, with diminishing returns beyond a few hundred tasks.

### Gaps
- No verified source runs a controlled sweep of training-dataset or label-set count against held-out F1 for an encoder label-reading classifier. Laurer compares only NLI-only against NLI + 28 datasets.
- tasksource's held-out zero-shot classification numbers were not retrievable: the PDF could not be parsed.

---

## 2. Role of large-scale contrastive pretraining (E5, GTE, BGE, Sentence-T5) vs. labelled classification data used by label-reading models

### Takeaway
Embedding models train on vastly more data:
- **E5:** about 270M consistency-filtered pairs, from about 1.3B raw.
- **GTE:** about 788M pairs from 33 sources, plus about 3M supervised fine-tuning pairs.

Label-reading classifiers use about 52k capped classification texts plus about 885k NLI pairs (zeroshot-v2.0) or 1.2M examples (GLiClass). That is a 2 to 4 order-of-magnitude difference.

Contrastive pretraining alone already gives most of the embedding quality. E5's unsupervised model reaches 67–69 MTEB-classification average, against 72–73 after fine-tuning, and GTE's 71.5 against 73.0. GTE's ablation shows more diverse pretraining sources consistently help.

Fine-tuning a bi-encoder narrowly on a few classification datasets transfers poorly OOD: UTCD bi-encoder 44.7 vs cross-encoder 53.2.

### Cited Findings
- **E5 (Wang et al., arXiv preprint 2022, revised 2024):**
  - **Data:** CCPairs is ~1.3B pairs after preliminary filtering, ~**270M** after consistency-based filtering. Sources: Reddit post–comment, StackExchange question–answer, Wikipedia entity+section–passage, scientific title–abstract and citations, Common Crawl and news.
  - **BEIR zero-shot** (nDCG@10, 15-dataset average) with no labelled data: BM25 41.7; E5-PT-small 40.8, base 42.9, large 44.2.
  - **MTEB classification** (12 datasets), contrastive-only E5-PT small/base/large: 67.0 / 67.9 / 69.0. Fine-tuned: 71.7 / 72.6 / 73.1.

  — [Wang et al. 2022](https://arxiv.org/html/2212.03533)
- **GTE (Li et al., arXiv preprint 2023):**
  - **Pretraining data:** ~**788M** pairs from 33 datasets: social media 327M, web pages 147M, hyperlinks 106M, academic 45M, knowledge base 38M, code 20M, community QA 12M, news 3M, other 91M.
  - **Fine-tuning data:** ~**3M** pairs from MS MARCO, NQ, SNLI, FEVER, Quora, MEDI and BERRI.
  - **Results:** GTE-base (110M parameters) scores **73.0** MTEB classification supervised and **71.5** unsupervised.

  — [Li et al. 2023, Table 1 / Table 6](https://arxiv.org/html/2308.03281)
- **GTE ablations:**
  - Growing pretraining data from the 5 largest datasets to 15 to all 33, and adding MEDI/BERRI to fine-tuning, "consistently enhance[s] model performance"; values are only in a figure.
  - Stages: pretraining-only 59.0, fine-tuning-only 57.8, both 62.4 MTEB average (ablation setting).
  - Data-mixture sampling α=0 (proportional to size) gives 55.4, against 59.0 at α=0.5.

  — [Li et al. 2023, §5.1, Tables 9–10](https://arxiv.org/html/2308.03281)
- **zeroshot-v2.0 classification data:** 51,731 capped non-NLI texts plus about 885k NLI pairs. — [Laurer et al.](https://arxiv.org/html/2312.17543)
- **zeroshot-v2.0-c:** no human classification data; "several hundred thousand" Mixtral-generated texts for 500+ tasks, plus MNLI and FEVER-NLI. — [zeroshot-v2.0 model card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- **GLiClass:** 1.2M pre-training examples plus small post-training sets of about 2k texts. — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)
- **UTCD:** a bi-encoder fine-tuned on UTCD ID data averages **44.7** OOD accuracy, against **53.2** for a Binary BERT cross-encoder. On B77 it scores **9.4%** (21.1 with implicit aspect training). Its ID average of 80.0 is close to Binary BERT's 81.9. — [Clarke et al. 2023, Tables 2–3](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **Throughput:** at 1→128 labels, GLiClass-base loses 7% (49.42 → 45.94 samples/s). The deberta-v3-base-zeroshot-v2.0 NLI cross-encoder goes from 24.55 to 0.47 samples/s, about 52× slower, because it needs one pass per label. — [Stepanov et al. 2025, Table 6](https://arxiv.org/html/2508.07662); [Laurer et al.](https://arxiv.org/html/2312.17543)

### Inferences
- The data-scale contrast is about 270M–788M weakly-supervised pairs against about 50k–1.2M classification or NLI examples. A label-encoding model built on a strong contrastive encoder most likely gets its label semantics from pretraining. Classification fine-tuning mainly teaches the text-label matching format, negatives and calibration.
- The UTCD bi-encoder result warns that narrow fine-tuning can collapse the embedding space onto training labels; B77 at 9.4% is the example. Mitigations consistent with the sources:
  - many diverse tasks (GTE's diversity ablation);
  - capped examples per label (Laurer);
  - keeping a contrastive or NLI objective in the mix (Laurer adds NLI; GTE adds SNLI and FEVER in fine-tuning).

  UTCD's bi-encoder did not start from a modern contrastive embedding model, so how large this effect is for E5/GTE/BGE starts is unknown.
- GTE's α ablation (proportional sampling 55.4 vs balanced 59.0) is the embedding-side analogue of Laurer's per-dataset caps. Balancing across sources beats letting the largest dataset dominate.

### Gaps
- **MTEB protocol, unverified lead:** MTEB "Classification" fits a logistic-regression probe on embeddings of each task's train split. If so, the E5 and GTE classification numbers above measure *linear separability*, not zero-shot label-name matching, and should not be cited as zero-shot label-reading evidence. Candidate: https://arxiv.org/abs/2210.07316 (Muennighoff et al., EACL 2023).
- **BGE / C-Pack (Xiao et al., 2023), unverified lead:** pair-corpus sizes not verified. Candidate: https://arxiv.org/abs/2309.07597.
- **Sentence-T5 (Ni et al., arXiv 2021; Findings of ACL 2022 not verified):** the fetched abstract confirms only that it beats Sentence-BERT/SimCSE on SentEval and SentGLUE and scales to billions of parameters. The pretraining size (about 2B community-QA pairs) is an **unverified lead**. — [Ni et al.](https://arxiv.org/abs/2108.08877)
- No source found gives a head-to-head on the same held-out benchmark between cosine matching of a contrastive embedding model against label names and an NLI or GLiClass model.

---

## 3. Domain shift: measured ID→OOD drops, what hurts most, what mitigates

### Takeaway
On UTCD, models trained on 9 ID datasets lose about **29–35 accuracy points** OOD: Binary BERT 81.9 → 53.2, Bi-Encoder 80.0 → 44.7, GPT-2 77.7 → 47.3. Untrained BART-MNLI scores about the same on both splits (46.1 vs 44.8), so the drop reflects train/test mismatch.

What hurts most:
1. **Domain-specific topic taxonomies:** legal, patent and finance topics at 9–35%.
2. **Many-class fine-grained sets:** B77, MASSIVE, Manifesto at about 0.26–0.57 macro-F1.
3. **Guideline-defined judgements:** hate vs offensive, factual consistency at about 0.39–0.54.

Transfer tracks **label-token overlap** with training labels.

Mitigations with numbers:
- **Label-description training:** +17 to 19 accuracy points; one label set reused across domains; beats 10-shot OOD supervision in 3 of 4 cases.
- **Aspect pre-training:** +2 average, up to +16 on topic.
- **Multi-domain classification data on top of NLI:** +9.4% held-out.
- **Domain-qualified hypotheses.**
- **Synthetic multi-profession data:** matches human data (Q5).
- **A few in-domain examples:** zeroshot-v2.0 gains +0.17 macro-F1; GLiClass at 8 shots per label gains +0.11–0.21.

### Cited Findings
- **UTCD OOD accuracy, vanilla, Binary BERT / Bi-Encoder / GPT-2:**
  - Amazon 80.7 / 69.9 / 88.3
  - FPB 68.9 / 71.7 / 71.1
  - Yelp 58.5 / 46.5 / 70.9
  - B77 51.4 / 9.4 / 22.8
  - SNIPS 82.9 / 70.4 / 52.2
  - NLU-Eval 71.6 / 71.1 / 61.7
  - MultiEURLEX 28.7 / 33.5 / 22.3
  - Patent 13.6 / 11.7 / 23.5
  - Consumer Finance 22.3 / 18.4 / 12.6
  - **Average 53.2 / 44.7 / 47.3**

  BART-MNLI averages 44.8; GPT-3 averages 43.3 on the sentiment and intent sets. — [Clarke et al. 2023, Table 2](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **UTCD ID averages:** Binary BERT 81.9, Bi-Encoder 80.0, GPT-2 77.7, BART zero-shot 46.1; supervised per-dataset BERT 83.4. — [Clarke et al. 2023, Table 3](https://ar5iv.labs.arxiv.org/html/2305.16521)
- UTCD: high correlation between label-token overlap and zero-shot results, with "the lowest positive transfer on datasets with the lowest level of token overlap". Overlap declines from sentiment to intent to topic. — [Clarke et al. 2023, Fig. 4](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **UTCD negative transfer:** a supervised BERT trained jointly on all datasets is below per-dataset training everywhere. On FPB it scores **24.9 vs 97.2**; the OOD average drops from 88.4 to 75.0. The FPB drop is attributed to "conflicting domain data". — [Clarke et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **zeroshot-v2.0 held-out macro-F1** (DeBERTa-v3-large, none of that dataset in training):
  - **Binary sentiment (high):** yelp 0.985, amazonpolarity 0.956, imdb 0.918.
  - **Many-class (low):** manifesto (56 classes) **0.256**; empathetic (32) 0.478; banking77 (72) **0.567**; massive (59) 0.518; capsotu (21) 0.502.
  - **Guideline-defined (low):** hatexplain 0.389; hateoffensive 0.461; trueteacher 0.538; wellformedquery 0.429.
  - **Means:** **0.673**, vs bart-large-mnli **0.497**.

  — [zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- **zeroshot-v2.0 with in-distribution data** (up to 500 per class from the 28 datasets; not held-out): the mean rises from 0.673 to **0.846**.
  - **Largest jumps:** hateoffensive 0.461 → 0.847, wellformedquery 0.429 → 0.815, spam 0.597 → 0.983, FPB 0.582 → 0.913, massive 0.518 → 0.789.
  - **Smallest:** yelp 0.985 → 0.994.

  — [zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- **Laurer verbalisation:** hypotheses come from the dataset paper or annotator guidelines, with **domain information added to reduce negative transfer**, e.g. "This app review text expresses positive sentiment". Laurer recommends "simple every-day language" and notes multiple hypotheses per class can help. — [Laurer et al.](https://arxiv.org/html/2312.17543)
- **LabelDescTraining (Gao, Ghosh & Gimpel, EMNLP 2023):**
  - **Data:** fine-tunes on tiny label-description sets of related terms, dictionary/encyclopedia entries and short templates. Topic uses about 6 per label (AGNews 24 total, Yahoo 60, DBPedia 84); 5-way sentiment uses 25 per label; binary sentiment uses 50 per label.
  - **Averages:** RoBERTa-base 54.7 zero-shot → **72.2**; RoBERTa-large 58.8 → **77.7**.
  - **Per dataset (base):** AGNews 62.7 → 77.4, DBPedia 54.6 → 79.5, SST-2 62.6 → 84.5, IMDB 69.9 → 86.9.

  — [Gao et al. 2023](https://arxiv.org/html/2305.02239)
- LabelDescTraining is domain-independent: "LabelDesc data are domain independent with regard to the distribution of the inputs". The same set is reused across AGNews/YahooAG, Yelp-5/SST-5 and Yelp-2/SST-2/Amz-2/IMDB.
  - It beats supervised few-shot *out-of-domain* training with 10 examples per label in 3 of 4 cases, and with 100 per label in 2 of 4.
  - Standard deviations across 14 prompt patterns are all smaller than for zero-shot.

  — [Gao et al. 2023](https://arxiv.org/html/2305.02239)
- **GLiClass-large-v3.0 vs deberta-v3-large-zeroshot-v2.0** (zero-shot F1): FPB **0.900 vs 0.582**; SNIPS 0.969 vs 0.834; sst5 0.461 vs 0.386; banking 0.557 vs 0.567; ag_news 0.718 vs 0.771. Averages **0.7193 vs 0.6821**. — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)
- **GLiClass 8-shot per label** (8-dataset subset): large 0.620 → 0.727; base 0.580 → 0.687; modern-base 0.445 → 0.654; edge 0.377 → 0.566.
  - **Failure modes:** text–label attention weakens as labels grow; representations degrade at extreme label-to-text token ratios (edge scores **0.027** on banking); 1,000+ labels hit the ~1,024-token limit; calibration varies.

  — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)

### Inferences
- What hurts, inferred from cross-dataset comparison rather than controlled ablation, comes down to what a bare label name fails to convey. Specialised domain taxonomies have low overlap with training labels; fine-grained many-class sets need subtle distinctions; guideline-defined judgements encode annotation rules. Coarse binary sentiment transfers almost perfectly.
- The UTCD control (BART ID 46.1 vs OOD 44.8) shows ID training lifts ID accuracy by about 35 points but OOD by only about 8 (53.2 vs 44.8). Most of what multi-dataset supervised training teaches is dataset- or label-specific.
- LabelDescTraining is the strongest verified domain-shift mitigation for a fixed label set. Tens of description texts per label give +17 to 19 points and transfer across input domains because they contain no input-domain text. For a label-encoding model this argues for training on (description ↔ label) pairs, and for encoding richer descriptions instead of bare names.
- Negative transfer from conflicting label semantics (FPB in UTCD's joint BERT; FPB "neutral" in Laurer) argues for domain- or task-qualified label strings.
- GLiClass's FPB 0.90 vs zeroshot-v2.0's 0.58 may reflect contamination: GLiClass's 1.2M corpus is not itemised. Laurer's numbers are explicitly held-out.

### Gaps
- No verified source isolates text register from label granularity in a controlled experiment.
- No verified numbers for paraphrase augmentation of label names specifically. The closest evidence is T0's prompts-per-dataset result (Q1) and LDT's lower variance across patterns.
- **NatCat (Chu, Stratos & Gimpel, EACL 2021), unverified lead:** training on naturally-annotated categories from Wikipedia, StackExchange and Reddit. Candidate: https://arxiv.org/abs/2009.13873.

---

## 4. Mechanism: memorising label identities vs learning label semantics; controls and anti-memorisation tricks

### Takeaway
Verified evidence is indirect and points to surface overlap and under-use of label or template semantics:
- UTCD OOD transfer tracks label-token overlap;
- ID training lifts ID about 4× more than OOD;
- prompt-based models learn NLI "just as fast" with irrelevant or misleading templates as with instructive ones, even at 175B and after instruction tuning (Webson & Pavlick).

The anti-memorisation tricks in use are all data-side:
- capped examples per class and per dataset (Laurer);
- random incorrect label hypotheses as negatives (Laurer);
- 50 true / 50 false LLM-generated labels per text with randomised positive/negative counts (GLiClass);
- many prompts per dataset (T0) and mixed instruction encodings (SuperNI);
- domain-qualified hypotheses;
- leave-whole-dataset-out evaluation.

No verified source runs label-shuffling or random-label-name controls on an encoder label-reading classifier.

### Cited Findings
- UTCD: label-token overlap correlates with transfer. The authors hypothesise "the model is over-fitting to the specific labels seen during training instead of generalizing to the 'aspect'"; this is not tested. — [Clarke et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **UTCD aspect pre-training** (train to recognise sentiment, intent or topic first) is the authors' remedy for label overfitting. It adds +1.9 average and up to +16 on topic for Binary BERT. It sometimes hurts, e.g. Bi-Encoder implicit drops Consumer Finance from 18.4 to 11.3. — [Clarke et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.16521)
- **Webson & Pavlick (NAACL 2022):**
  - Over 30 manually written NLI templates were tested. Models "learn just as fast with many prompts that are intentionally irrelevant" or misleading as with instructive ones.
  - This holds up to 175B parameters and for instruction-tuned models trained on hundreds of prompts. Instruction-tuned models often give good zero-shot predictions even with irrelevant or misleading prompts.
  - Conclusion: prompt-based gains may not come from understanding instructions the way humans do.

  — [Webson & Pavlick 2022](https://arxiv.org/abs/2109.01247)
- **Yin, Hay & Roth (EMNLP 2019):** define "label-partially-unseen" (train on some labels, test on all) and "label-fully-unseen" (no task-specific training data), and frame zero-shot topic, emotion and situation classification as entailment. Detailed bias-toward-seen-labels numbers were not retrieved. — [Yin et al. 2019](https://arxiv.org/abs/1909.00161)
- **Laurer NLI conversion:** each text is paired with its correct hypothesis and with a random incorrect one. Caps "avoid overfitting to a specific large dataset", and the goal is "not to build a classifier that beats (and overfits to) a benchmark". — [Laurer et al.](https://arxiv.org/html/2312.17543)
- **GLiClass post-training:** 50 true and 50 false GPT-4o labels per text; duplicates vary positive/negative label counts via "random coefficients". The paper says nothing about memorisation or shuffling controls. — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)
- **T0:** more prompts per dataset raises the median, and non-original-task prompts reduce spread on 8 of 11 held-out datasets. More datasets alone do not consistently improve robustness to wording. — [Sanh et al. 2022](https://ar5iv.labs.arxiv.org/html/2110.08207)
- **SuperNI:** single-encoding training fails on other encodings; mixed encodings are robust. — [Wang et al. 2022, §7.2](https://arxiv.org/html/2204.07705v3)
- **FLAN:** fine-tuning with only "dataset name" or "no template" is substantially worse on unseen tasks than with natural-language instructions. Instruction *text* matters for zero-shot transfer, even if (per Webson & Pavlick) models do not use it in a human-like way. — [Wei et al. 2022](https://arxiv.org/html/2109.01652)

### Inferences
- A minimal "learning to read labels" recipe consistent with the evidence:
  1. Many distinct label strings per text, with negatives drawn from other label sets (Laurer, GLiClass).
  2. Capped examples per label (Laurer; SuperNI saturation at about 64 per task).
  3. Several phrasings or descriptions per label (T0 prompts, SuperNI mixed encodings, LDT).
  4. Evaluation on whole held-out label sets or datasets (Laurer leave-one-out; UTCD OOD).
- **Cheap diagnostic** from the UTCD correlation: bucket OOD labels by similarity to the nearest training label and plot accuracy per bucket. A steep slope signals label-identity memorisation.
- **Natural control, not run in any verified source:** retrain with label names replaced by random strings or permuted. If ID accuracy holds while OOD collapses, labels are being used as IDs. Webson & Pavlick's irrelevant-template result predicts that ID accuracy would survive such permutation more than one might expect.

### Gaps
- **Min et al. (EMNLP 2022), unverified lead:** random labels in in-context demonstrations barely hurt. https://arxiv.org/abs/2202.12837
- **Wei et al. (2023), unverified lead:** with semantically-unrelated or flipped labels, only large models override semantic priors. https://arxiv.org/abs/2303.03846
- **LSAP (Mueller et al., NAACL 2022), unverified lead:** label-semantic-aware pretraining for intent. https://arxiv.org/abs/2204.07128
- No source, verified or otherwise, runs label-shuffling or random-name controls on an encoder bi-encoder, NLI or GLiClass classifier. This looks like a genuine literature gap.

---

## 5. Can synthetic LLM-generated data substitute for human data in giving domain transfer?

### Takeaway
**For broad, label-reading training data, yes.** zeroshot-v2.0-**c** trains on Mixtral texts for 500+ tasks across 25 professions plus MNLI/FEVER, with no human classification data. Its held-out mean is **0.676 vs 0.673** for the variant with 28 human datasets: synthetic wins on FPB, spam, CAP and AG News, and loses on B77, toxicity-threat, factuality and query well-formedness.

**For replacing a specific task's real training set, only partly, and it depends on subjectivity.** In Li et al. (EMNLP 2023), RoBERTa trained on zero-shot-generated GPT-3.5 data trails real data by **16.9** macro-F1 on average, and by **6.7** with few-shot generation. The gap is 3–6 points on low-subjectivity tasks (AG News, IMDB, spam) and an average of 27 points on high-subjectivity ones (sarcasm, humor, financial). Synthetic data is less diverse and overfits faster.

### Cited Findings
- **zeroshot-v2.0-c data:** Mistral-large produced "500+ diverse text classification tasks for 25 professions" (manually curated). Mixtral-8x7B-Instruct generated "several hundred thousand texts". MNLI and FEVER-NLI were added "to improve generalization". Non-"c" also uses ANLI, WANLI, LingNLI and the v1.1 human datasets. — [zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- **Held-out macro-F1, -c (synthetic) vs non-c (human + synthetic), DeBERTa-v3-large:**
  - **Mean:** **0.676 vs 0.673**; bart-large-mnli 0.497.
  - **Synthetic better:** FPB 0.691 vs 0.582; spam 0.740 vs 0.597; capsotu 0.603 vs 0.502; agnews 0.819 vs 0.771; biasframes_sex 0.846 vs 0.815.
  - **Human better:** banking77 0.513 vs 0.567; wikitoxic_threat 0.827 vs 0.879; trueteacher 0.459 vs 0.538; wellformedquery 0.334 vs 0.429; empathetic 0.447 vs 0.478.
  - Zero-shot runs had "no data from these datasets ... in the training mix".

  — [zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- The earlier Laurer et al. paper on v1.x used no synthetic data and listed it as future work. — [Laurer et al.](https://arxiv.org/html/2312.17543)
- **Li et al., "Synthetic Data Generation with LLMs for Text Classification: Potential and Limitations" (EMNLP 2023):**
  - **Setup:** 10 tasks rated 1–5 for subjectivity; BERT and RoBERTa trained on real data, zero-shot-generated data or few-shot-generated data (GPT-3.5-Turbo).
  - **RoBERTa averages:** macro-F1 gap vs real of 16.9 (zero-shot) and 6.7 (few-shot); accuracy gaps 14.9 and 6.1.
  - **BERT zero-shot gaps:** low-subjectivity tasks −3.4 to −6.4 (AG News 95.3 → 89.3; IMDB 87.6 → 81.2; SMS spam 97.2 → 93.8). High-subjectivity cluster average −27.4 macro-F1 (Sarcasm 89.9 → 51.1; Humor 97.0 → 56.0; Financial 83.2 → 48.2; Tweet Emotions 77.7 → 58.1).
  - **Few-shot generation:** recovers about 10 points on average and on Tweet Irony beats real data (+9.3).

  — [Li et al. 2023](https://arxiv.org/html/2310.07849v2)
- **Li et al. diversity findings:** Remote-Clique and Chamfer diversity order is real > few-shot synthetic > zero-shot synthetic. The diversity shortfall is significantly larger for high-subjectivity tasks (p < 0.01). Synthetic-data models converge faster and overfit more. — [Li et al. 2023](https://arxiv.org/html/2310.07849v2)
- **ZeroGen (Ye et al., EMNLP 2022):** a PLM generates a task dataset from scratch, unsupervised, and a tiny task model (e.g. an LSTM) is trained on it, covering classification, QA and NLI. The abstract gives no numbers. — [Ye et al. 2022](https://arxiv.org/abs/2202.07922)
- **GLiClass:** 2,000 texts with 50 true and 50 false GPT-4o labels each, plus 2,000 synthetic logic/NLI examples. There is no ablation of their effect. — [Stepanov et al. 2025](https://arxiv.org/html/2508.07662)

### Inferences
- The two results are compatible because they measure different things.
  - **zeroshot-v2.0-c:** synthetic data is used to teach *general label reading* across many domains, and only the label-reading skill has to transfer. Breadth across 500+ tasks and 25 professions makes up for lower per-task fidelity.
  - **Li et al.:** synthetic data must *replace one task's real training set*. Per-task fidelity and diversity matter, and subjective tasks suffer most.
- Both results point to the same weak spots: subjective or guideline-defined labels (sarcasm, humor, hate/offense, factual consistency, query well-formedness) and fine-grained many-class sets (B77). This matches Q3, where label names under-specify exactly these tasks, and an LLM generating from names or task descriptions inherits the under-specification. Few-shot or exemplar-seeded generation is the measured fix: it closes about 10 points in Li et al.
- Human + synthetic is not additive on the zeroshot-v2.0 benchmark (0.673 vs 0.676). The binding constraint there is likely the hardest task types or model capacity, not data volume. This is an inference; there is no ablation.

### Gaps
- **SuperGen (Meng et al., NeurIPS 2022), unverified lead:** class-conditioned generation for zero-shot GLUE. https://arxiv.org/abs/2202.04538. ZeroGen's numbers were not retrieved either.
- The zeroshot-v2.0 card does not say whether the synthetic task list was checked for overlap with the 28 evaluation tasks. "Financial analyst" or "customer service" tasks could near-duplicate FPB or B77 label sets, which would inflate the -c zero-shot numbers.
- No verified 2024–2026 study measures train-on-synthetic-domains → test-on-unseen-real-domains for encoder label-reading classifiers, beyond the zeroshot-v2.0 card.
