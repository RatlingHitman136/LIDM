# Label descriptions vs label names for zero-/few-shot text classification: evidence, methods, released resources

Scope: published evidence (2019-2026) on giving models "name: description" label text instead of bare names, for NLI/cross-encoder, bi-encoder, LLM and GLiClass-style models. Also covers training with descriptions, how descriptions are produced, evaluation pitfalls, and reusable description files. Venue status is marked. "Preprint" means arXiv only, as far as the fetched page shows.

## Q1. What gains do label descriptions give over label names, by model type and benchmark (incl. intent detection)?

### Takeaway
Descriptions help most when (a) the model is trained to read them and (b) test labels are unseen. Untrained NLI models get little or negative value from dictionary definitions. Description-trained models gain a lot: about +17-19 points average on topic/sentiment (Gao et al. 2023), +6.7 JGA on unseen SGD services vs names (D3ST), and +2-13 F1 in zero-shot IE/NER. The only intent-detection study with descriptions on Banking77/CLINC150/HWU64 (Hong et al., SIGDIAL 2024) finds test-time description quality is the main lever: CLINC unseen-domain accuracy rises from 84.3% to about 90.5-91.8%. No paper found reports a clean bi-encoder or late-interaction name-vs-description ablation on Banking77/CLINC150/HWU64.

### Cited Findings

**NLI / cross-encoder zero-shot (untrained on descriptions)**
- Yin, Hay & Roth (EMNLP 2019). Hypotheses are built from the label name ("word"), its WordNet definition ("def"), or both. Definitions alone "usually does not work well" in any of the three tasks. Label-fully-unseen ensemble results: topic (Yahoo) acc word 48.4 / def 26.0 / word&def 45.7; emotion F1 18.3 / 9.0 / 25.2; situation F1 38.3 / 14.4 / 38.0. The combined ensemble has the best sum (108.9 vs 105.0 word-only), but which of word-only and word&def is better depends on task and NLI training set (MNLI favours word&def for emotion and situation; RTE and FEVER mostly favour word-only) — [Yin et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.00161)

**Masked-LM / encoder classifiers trained on label descriptions (LabelDescTraining)**
- Gao, Ghosh & Gimpel (EMNLP 2023 main). Fine-tuning on a tiny LabelDesc set (no task input texts) beats zero-shot by about 17-19% absolute. Table 2 accuracy, averaged over 14 patterns:
  - Zero-shot RoBERTa-base average 54.7 → LDT 72.2
  - Zero-shot RoBERTa-large average 58.8 → LDT 77.7
  - Per dataset, large, zero-shot → LDT: AGNews 68.0 → 79.4; Yahoo 47.7 → 60.8; DBPedia 63.9 → 86.6; Yelp-5 38.7 → 51.3; SST-5 35.0 → 49.2; Yelp-2 70.6 → 94.6; SST-2 63.7 → 91.3; Amz-2 67.5 → 94.1; IMDB 74.1 → 92.1
  - With a single pattern selected on 20NG (Table 4), LDT reaches 84.6/85.1 on AGNews and 88.2/91.4 on SST-2 (base/large)
  — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239); [arXiv abstract](https://arxiv.org/abs/2305.02239)
- Gao et al. 2023 also compare against LLM prompting. RoBERTa-large LDT beats text-davinci-003 (GPT-3.5) on AGNews, DBPedia, Yelp-2, SST-5 and IMDB (1,000-instance test subsets). GPT-3.5 in-context learning with all LabelDesc data as demonstrations beats GPT-3.5 zero-shot on all datasets — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- The gain is not purely semantic. Training on LabelDesc with random verbalizers still gives 65.9/73.4 average (base/large), and mismatched verbalizers 64.8/65.4. Both are above zero-shot (54.7/58.8) but below LDT with matched verbalizers (72.2/77.7) — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)

**Bi-encoder / dual-encoder dataless classification (label-name text as the "description")**
- Chu, Stratos & Gimpel (Findings ACL 2021). In a RoBERTa dual encoder trained on NatCat, the wording of the label names alone moves AG News accuracy from 49.9 to 71.6 across six label-name sets (original "world/sports/business/sci-tech" = 69.9). Oracle label choices reach 81.9 (AG), 87.5 (DBpedia) and 53.3 (Yahoo) with cosine distance. Unsupervised label refinement (k-means on unlabeled target texts) narrows the spread (e.g. 49.9 → 70.2; 55.9 → 78.3) and raises the RoBERTa dual-encoder average from 61.8 to 65.4 — [Chu et al. 2021](https://ar5iv.labs.arxiv.org/html/2012.04194); [ACL Anthology](https://aclanthology.org/2021.findings-acl.365)

**Seq2seq / description-driven dialogue (names vs descriptions vs random strings)**
- D3ST (Zhao et al., Google; arXiv 2022, preprint). Ablation, JGA for Language description / element Name / Random string:
  - SGD (unseen services in test): Large 80.0 / 73.7 / 37.4; XXL 86.4 / 79.7 / 64.8
  - MultiWOZ 2.1 (schemas seen in training): Large 54.5 / 55.1 / 20.1; XXL 57.8 / 57.5 / 57.6
  - MultiWOZ 2.4: Large 70.8 / 72.2 / 16.9; XXL 75.9 / 75.4 / 73.6
  - The authors note that names beat descriptions for Large on MultiWOZ, which they attribute to longer inputs. SGD active-intent accuracy (with descriptions) is 97.2 / 97.1 / 98.8 for Base / Large / XXL
  — [Zhao et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.08904)

**LLM prompting / fine-tuned LLMs with intent descriptions (Banking77 / CLINC150 / HWU64)**
- Hong et al. (SIGDIAL 2024, LG AI Research). FLAN-T5 3B fine-tuned on 75 CLINC intents (5 domains) and tested on 75 unseen-domain CLINC intents. Options are formatted as "1. o1: d1, 2. o2: d2 ...". Table 1 accuracy (training descriptions → test descriptions):
  - Trained without descriptions: tested without 84.28; with "independent" descriptions 84.15; with "dependent" descriptions 90.55
  - Trained with independent descriptions: 81.93 / 85.64 / 90.97
  - Trained with dependent descriptions: 82.10 / 86.99 / 91.75
  - Training with descriptions and testing without them costs about 2.2-2.4 points
  — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- Hong et al. 2024, cross-dataset zero-shot (Appendix C, Table 3). The setting follows Sung et al. (PIE): CLINC with 50 intents, HWU with 25, BANKING with 27. Zero-shot (K=0) accuracy CLINC / HWU / BANKING:
  - FLAN-T5 3B fine-tuned with dependent descriptions, tested with top-10 ranked cleansed descriptions ("FLAN-T5-ranker"): 96.46 / 86.23 / 84.88
  - FLAN-T5 without ranker: 97.58 / 87.92 / 84.72
  - Llama-2-7B variants: about 96 / 86 / 84
  - Name-based encoder baselines from Sung et al.: L-PIE 86.5 / 70.6 / 77.6; L-SBERT-Paraphrase 84.5 / 67.5 / 77.4
  - The authors attribute part of the gap to model size, so this is not a clean names-vs-descriptions comparison
  — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- Hong et al. 2024, retrieval over descriptions. An off-the-shelf mpnet-base-v2 ranker scores query-vs-description similarity, and passing about the top 10 options keeps accuracy near the all-options level. CLINC goes from 44.21% at k=1 to about 90% at k≈13. BANKING starts at 50.83 (k=1) and reaches about 80 near k=10; HWU starts at 26.51 and reaches about 80 near k=15. Cutting to the top 10 reduces input length by about 75% — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- Parikh et al. (ACL 2023 Industry). Zero-shot prompting of GPT-3 (175B) and Flan-T5-XXL with intent descriptions of the form "alarm_set: user wants to set an alarm". In-scope accuracy on MASSIVE (60 intents): Flan-T5-XXL 73.3, GPT-3 73.9. There is no names-only ablation. PEFT (T-Few/IA3) on Flan-T5-XL with 1 example per intent beat all other methods. Three of the four datasets are private — [Parikh et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.07157)
- Gen-Z (Kumar, Park & Tsvetkov; arXiv 2023, preprint). Generative scoring of the input given about 10 paraphrased, optionally contextualized label descriptions. GPT-J 6B macro-F1, single no-context description (GS-N) → Gen-Z: SST2 80.0 → 91.7; Yelp2 76.1 → 89.9; MR 76.0 → 87.0; CR 72.9 → 87.0; AGNews 64.9 → 77.0; DBpedia 71.7 → 80.1; HS18 50.2 → 62.6; Tweet3 42.2 → 41.1 (the only loss). Performance rises from 1 to about 6-10 descriptions, then plateaus — [Kumar et al. 2023](https://ar5iv.labs.arxiv.org/html/2311.07115); [arXiv](https://arxiv.org/abs/2311.07115)
- Walshe et al. (JPMorgan; arXiv 2025, preprint) report that "naively using the label description for classification leads to poor performance on high cardinality tasks". They propose retrieval-augmented dynamic label-schema integration. The fetched abstract gives no numbers — [Walshe et al. 2025](https://arxiv.org/abs/2501.12332)

**Instruction-tuned IE models trained with definitions/guidelines (closest analogue to training a classifier on "name: description")**
- GoLLIE (Sainz et al.; arXiv 2023, ICLR 2024 venue not shown on fetched page). Zero-shot mean F1 on unseen IE tasks: full model 55.3 vs 42.3 for the baseline without guidelines and regularisations. Removing the representative candidates alone drops it to 49.9 — [Sainz et al. 2023](https://ar5iv.labs.arxiv.org/html/2310.03668)
- SLIMER (Zamai et al.; arXiv 2024, v3 preprint). Zero-shot NER with LLM-generated definition & guidelines (D&G) vs the same model without D&G:
  - MIT/CrossNER average F1: 51.3 ± 2.0 → 54.0 ± 0.5 (e.g. CrossNER Politics 57.9 → 63.9; MIT Movie 46.4 → 50.9)
  - BUSTER µ-F1: 40.41 ± 5.09 → 45.27 ± 1.04
  - Variance across runs drops sharply. Some tags get worse (e.g. "Product" −14.33), which the authors attribute to overly strict guidelines
  — [Zamai et al. 2024](https://arxiv.org/html/2407.01272v3)

**GLiClass-style models**
- GLiClass (Knowledgator; arXiv 2025, preprint) adapts the GLiNER architecture to sequence classification. The abstract does not mention descriptions or benchmarks — [Stepanov et al. 2025](https://arxiv.org/abs/2508.07662)
- The GLiClass v3 / instruct model card lists "Add natural-language descriptions to labels for more precise classification" as a feature. It shows no syntax and reports no with/without-description numbers. Its zero-shot table has a "banking" row (F1 0.6987 for large-v1.0, 0.5574 for large-v3.0) of unclear provenance — [HF model card](https://huggingface.co/knowledgator/gliclass-instruct-large-v1.0)

**Universal NLI classifiers**
- Laurer et al. (arXiv 2023/2024, preprint) train an NLI-based universal classifier on 33 datasets with 389 classes and claim +9.4% zero-shot over earlier versions. The fetched abstract does not say how class hypotheses were worded — [Laurer et al.](https://arxiv.org/abs/2312.17543)

### Inferences
- Across D3ST, Gao, Hong, GoLLIE and SLIMER, the benefit of descriptions is conditional on training the reader. Off-the-shelf NLI models given definitions (Yin 2019) gain little or lose. A late-interaction model trained on "name: description" options should expect gains mainly on unseen or fine-grained label sets, and roughly parity on label sets seen in training (D3ST on MultiWOZ: names ≈ descriptions).
- Keep the name in the option string. Definition-only text was much worse than word-only in Yin 2019. The name-plus-description formats (D3ST, Hong's "o: d") are the ones that worked.
- For Banking77-like fine-grained intents, the most transferable result is Hong's: "dependent" descriptions, written to contrast each intent with its neighbours, are what move accuracy (+6.3 points at test time for a name-trained model).
- Hong's ranker result suggests a two-stage design: retrieve the top ~10-15 options by query-vs-description similarity, then rerank. A late-interaction scorer could do the first stage.

### Gaps
- No paper found with a controlled names-vs-descriptions ablation for bi-encoders or late-interaction (ColBERT-style) classifiers on Banking77, CLINC150 or HWU64. Hong et al. use a fine-tuned FLAN-T5/Llama, and their only name-based baselines are different, smaller models.
- Hong et al. do not report a names-only test condition for Banking77 or HWU64 (Appendix C has only the description-based FLAN-T5 results and name-based baselines from other papers).
- Laurer et al.: hypothesis or description wording was not visible on the fetched page.
- GLiClass: no published numbers with vs without label descriptions were found.
- Parikh et al.: no names-only control.
- SGD-X intent-accuracy robustness numbers were not reported in the fetched content.

## Q2. Training with descriptions: LabelDesc, D3ST/SGD-X, dataless classification, X-shot and 2023-2026 work

### Takeaway
Training on descriptions (or with them in the input) consistently helps zero-shot transfer and lowers sensitivity to prompt and label wording. Training on a single schema wording leaves models brittle to paraphrased descriptions: SGD-X shows JGA drops of 12-18% relative. Paraphrase augmentation (back-translation) recovers part of that.

### Cited Findings
- **LabelDesc construction and recipe (Gao et al., EMNLP 2023).**
  - Topic tasks (AGNews, Yahoo, DBPedia, 20NG): each label gets its term, 3 related terms, 1 dictionary.com definition and the first sentence of its Wikipedia article (6 examples per label).
  - Sentiment tasks: verbalizer plus 4 synonyms × 4 hand-written templates (e.g. "It was t."), giving 25 examples per label for 5-way and 50 per label for binary.
  - Set sizes: 20NG 24, AGNews 24, Yahoo 60, DBPedia 84, Yelp-5 125, Yelp-2 100.
  - Hyperparameters (LR, steps, freezing the lower half of layers, pattern) were chosen on 20NG only.
  — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **LabelDesc cross-domain.** LDT beats out-of-domain few-shot supervised training in 3 of 4 cases at 10 examples per label and 2 of 4 at 100. Trained on AGNews and tested on YahooAG, it beats few-shot even at 500 examples per label. A 20NG-trained LDT model often beats zero-shot on other tasks (base), while large is worse on most sentiment tasks — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **LabelDesc robustness.** Standard deviation across 14 patterns falls on every dataset, e.g. zero-shot RoBERTa-base DBPedia 18.9 → 4.5 with LDT, and RoBERTa-large Yelp-5 7.8 → 2.4 — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **D3ST** puts natural-language descriptions of slots and intents, with randomised index identifiers, in the input of a T5 seq2seq model. On SGD-X (average over v1-v5 and schema sensitivity SS(JGA), lower is better):
  - Language: Large 75.3 (SS 0.26); XXL 77.8 (0.27)
  - Name: Large 66.9 (0.37); XXL 73.0 (0.35)
  - Random: Large 28.4 (0.74); XXL 67.1 (0.51)
  The anti-memorisation device is randomising index assignment during training. There is no dedicated paraphrase-training experiment — [Zhao et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.08904)
- **SGD-X (Lee et al.; arXiv 2021, published at AAAI 2022 per common citation, venue not shown on fetched page).**
  - 5 crowd-written variants per SGD schema (4,755 paraphrases of names and descriptions), written by 400+ MTurk workers who did not see the originals. Variants are ordered by distance from the original (Levenshtein for names, Jaccard for descriptions).
  - SS is defined as the turn-level coefficient of variation of a metric across variants.
  - JGA drops: SGP-DST 60.5 → 49.9 average (−17.6% relative; worst case about −28%); T5DST 72.6 → 64.0 (−11.9%; worst case about −19%).
  - Back-translation (zh/ko/ja) schema augmentation: SGP-DST JGA 54.1 (+8%), SS −17%; T5DST JGA 70.8 (+11%), SS −16%.
  - Training on crowd paraphrases ("oracle") gives 66.2 / 73.3, but overlaps with the evaluation schemas.
  — [Lee et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.06800)
- **Dataless classification (Chu, Stratos & Gimpel, Findings ACL 2021).** "Label descriptions" here are expanded label names (abbreviations expanded, e.g. "sci_tech"), following Chang et al. (2008) and Yin et al. (2019). Models are a GloVe dual encoder, RoBERTa dual and single encoders trained on 300k NatCat instances, and ESA (AG 71.2, DBP 62.5, Yahoo 29.7, 20NG 25.1). Unsupervised label refinement gains are largest when the label descriptions are weak — [Chu et al. 2021](https://ar5iv.labs.arxiv.org/html/2012.04194)
- **X-Shot / BinBin (Xu, Chen, Huang, Vucetic & Yin; arXiv 2024, preprint as fetched).** Recasts every task as binary (instruction, input, label) → Yes/No, keeps label names as text (no ids), and uses indirect supervision from instruction datasets plus LLM weak supervision. It covers frequent, few- and zero-shot labels together on 3 benchmarks with 30-78 labels. Label descriptions specifically are not confirmed in the fetched material. Code and data: github.com/xhz0809/X-shot — [Xu et al. 2024](https://arxiv.org/pdf/2403.03863)
- **UTCD / label-agnostic pre-training (Clarke et al., Findings ACL 2023).** 18 datasets (9 in-domain, 9 out-of-domain) with labels standardised to be "descriptive of the text in natural language". Intent datasets: SGD, CLINC-150 and SLURP in-domain; Banking77, SNIPS and NLU-Evaluation (HWU) out-of-domain. Proposes Implicit and Explicit pre-training for zero-shot generalisation — [ACL Anthology](https://aclanthology.org/2023.findings-acl.64); [HF dataset card](https://huggingface.co/datasets/claritylab/utcd)
- **Hong et al. 2024 on training-time descriptions.** Fine-tuning is required for the model to use descriptions well, and test-description quality matters more than training-description quality — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **GoLLIE training regularisers** (class-order shuffling, class dropout, guideline paraphrasing with Vicuna-33B, sampling 5 of 10 representative candidates, class-name masking with LABEL_1-style placeholders). Ablation, zero-shot mean F1 vs full 55.3: w/o paraphrases 54.8 (n.s.), w/o masking 54.6 (n.s.), w/o shuffling 55.9 (n.s.), w/o dropout 54.0 (significant), w/o candidates 49.9 (strongly significant). Loss is computed on output tokens only — [Sainz et al. 2023](https://ar5iv.labs.arxiv.org/html/2310.03668)
- **SLIMER** trains on 423 PileNER types with test-set types removed (keeping person, location, organization and country). About 7% average tag overlap with test. D&G give faster, more stable learning — [Zamai et al. 2024](https://arxiv.org/html/2407.01272v3)

### Inferences
- The strongest training-side recipe signals for a "name: description" late-interaction model:
  1. Put descriptions in training, not just at test time (Gao, Hong, D3ST).
  2. Randomise or mask the name-to-identity link so the model must read the description: D3ST random indices, GoLLIE name masking (the masking gain was non-significant in GoLLIE).
  3. Paraphrase-augment descriptions to cut wording sensitivity (SGD-X back-translation: SS −16-17%).
  4. Sample hard-negative and distractor label sets per example (class dropout and shuffling).
- LabelDesc shows that even ~6 description strings per label, used as training "inputs", transfer. A cheap auxiliary objective, scoring the description text itself against its own label option, is a plausible add-on for a late-interaction model. This is untested in that architecture.

### Gaps
- No SGD-X-style paraphrase-robustness benchmark exists for Banking77, CLINC150 or HWU64 descriptions.
- X-Shot's exact use of descriptions was not verified.
- GoLLIE did not test paraphrased guidelines at inference.
- No 2025-2026 peer-reviewed work training bi-encoders or late-interaction classifiers on label descriptions with intent benchmarks was found.

## Q3. How should descriptions be produced (human vs taxonomy/dictionary vs LLM; how many per label)? Sensitivity to wording and overfitting to style

### Takeaway
Quality and discriminativeness beat quantity. LLM descriptions generated with all sibling labels in view ("dependent") clearly beat per-label ("independent") ones for intents (Hong 2024). Dictionary/WordNet definitions alone are poor for untrained NLI (Yin 2019), but help as one of several training strings (Gao 2023). Multiple paraphrases help as a test-time ensemble (Gen-Z: plateau at about 6-10) and as training augmentation for robustness (SGD-X). Multiple training descriptions of mixed quality did not improve accuracy (Hong).

### Cited Findings
- **Hong et al. 2024, production of descriptions.** ChatGPT (gpt-3.5-turbo) wrote three quality tiers:
  - "independent": prompt shows one intent plus 3 user-query examples; 7 per intent
  - "dependent": prompt lists all intents so the description distinguishes the intent from the others; 7 per intent
  - "cleansed": one dependent description per intent, manually filtered for clear distinction
  — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **Hong et al. 2024, quantity (Table 2).** Training with 5 alternating dependent descriptions per intent gives 82.05 / 85.28 (tested with dependent / cleansed); 1 dependent description gives 82.52 / 85.71; 1 cleansed description gives 83.40 / 85.79. Variety did not help, and "performance tends to decline with the inclusion of varied descriptions" — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **Hong et al. 2024, test-quality asymmetry.** Train independent / test dependent scores 90.97, versus train dependent / test independent 86.99 — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **Gen-Z.** One minimal hand-written template description per label, then 20 ChatGPT paraphrases per description, each manually checked; 1-10 are sampled per run. Gen-Z has the smallest variance across prompts (e.g. std 0.2 on SST2, 0.1 on AGNews), while few-shot ICL baselines vary by up to 9.9 macro-F1 (ICL-Disc AGNews 68.4 ± 9.9). Arithmetic-mean aggregation over descriptions is slightly best — [Kumar et al. 2023](https://ar5iv.labs.arxiv.org/html/2311.07115)
- **Gao 2023.** Mixed sources per label (terms, dictionary definition, Wikipedia lead sentence; templates for sentiment). Selection was not tuned and there is no per-source ablation — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **Yin 2019.** WordNet definitions alone sharply underperform label names for NLI zero-shot (e.g. topic 26.0 vs 48.4) — [Yin et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.00161)
- **Wording sensitivity of names (Chu 2021).** Six plausible AG News label-name sets span 49.9-71.6 accuracy — [Chu et al. 2021](https://ar5iv.labs.arxiv.org/html/2012.04194)
- **Wording sensitivity of descriptions (SGD-X).** Crowd paraphrases of schema descriptions drop JGA by 12-18% relative on average. Description-trained D3ST is less sensitive than name-trained D3ST (SS 0.26 vs 0.37 at Large) — [Lee et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.06800); [Zhao et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.08904)
- **SLIMER.** D&G were generated per entity type by gpt-3.5-turbo-1106 from 3 examples sampled from the dataset. No human check or systematic quality evaluation is reported; quality varies by tag, and overly strict guidelines can cut recall — [Zamai et al. 2024](https://arxiv.org/html/2407.01272v3)
- **Iterative definition refinement (Rehmat et al.; arXiv 2026, CVPR NeXD workshop).** LLMs revise class definitions using misclassified examples, with no parameter updates, across 13 embedding models. A new 10-class URL benchmark (B2MWT-10C) is released on GitHub. The fetched abstract gives no numbers — [Rehmat et al. 2026](https://arxiv.org/abs/2604.27335)
- **Vision analogue (ICCV 2023).** WaffleCLIP replaces LLM-generated class descriptors with random words and characters and gets "comparable performance gains" across many visual classification tasks. Part of the "description" gain there comes from ensembling or noise rather than semantics — [Roth et al. 2023](https://arxiv.org/abs/2306.07282)

### Inferences
- For Banking77-style fine-grained labels, generate descriptions with the full label inventory in context, so they contrast confusable siblings, and keep one curated description per label as the canonical test-time description.
- For training, prefer a few high-quality paraphrases plus SGD-X-style paraphrase or back-translation augmentation for robustness, over many noisy LLM samples. Hong shows noisy variety does not raise accuracy; SGD-X and Gen-Z show paraphrase diversity cuts variance.
- Overfitting to description style is a real risk when all training descriptions come from one LLM/template, as in SLIMER and GoLLIE. Mitigations: mix sources (Gao: terms, dictionary, encyclopedia), paraphrase with a different model than the test-description writer, and evaluate on independently written descriptions (SGD-X-style crowd variants).
- Include a random-descriptor control (WaffleCLIP) and a mismatched-description control (Gao's mismatched verbalizers) to show the model reads semantics, not just extra tokens.

### Gaps
- No study found that directly compares human-written vs LLM-written descriptions for the same intent benchmark with a fixed model. Hong's "cleansed" tier is LLM output filtered by humans.
- No measured evidence on overfitting to a specific LLM's description style (train on GPT-style, test on human-style) for text classification. SGD-X (crowd paraphrases) is the closest.
- Gen-Z figures for models other than GPT-J were figure-only in the fetched text.

## Q4. Evaluation pitfalls: leakage and comparability with name-only benchmarks

### Takeaway
Many description pipelines use labeled examples from the evaluated label set when writing descriptions: Hong (3 queries per intent), SLIMER (3 examples per type), and definition refinement on misclassified examples. That turns "zero-shot" into weakly few-shot. Gao et al. are the clean reference protocol: no target inputs, and hyperparameters tuned on a different dataset (20NG). Cross-paper comparisons mix model sizes and label subsets.

### Cited Findings
- **Gao et al. 2023 protocol.** Follows the "true" zero-shot setting: LabelDesc data "merely describes the labels but does not use input texts", and no target-task data is used for tuning or early stopping. All hyperparameters and the pattern were chosen on 20NG (4 labels), which is excluded from evaluation. GPT-3.5 was evaluated on a 1,000-instance label-distribution-matched test subset, not the full test set. The paper text says batch size 1 but the appendix says 11 — [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **Hong et al. 2024.** Independent descriptions are generated from 3 user-query examples per intent, and cleansed descriptions are manually filtered to separate intents. Their headline cross-dataset comparison with L-PIE / L-SBERT also differs in model size (3B/7B vs encoder baselines), which the authors acknowledge — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf)
- **SLIMER.** D&G prompts used 3 examples per entity type "randomly sampled from the dataset" — [Zamai et al. 2024](https://arxiv.org/html/2407.01272v3)
- **Iterative definition refinement** explicitly uses misclassified examples to rewrite definitions — [Rehmat et al. 2026](https://arxiv.org/abs/2604.27335)
- **SGD-X paraphrase "oracle".** Training on crowd paraphrases overlaps with the evaluation schemas, so the authors treat it only as an upper bound. Back-translation is the fair comparison — [Lee et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.06800)
- **Chu et al. 2021.** Oracle label-name choices (picked with test knowledge) give much higher numbers (AG 81.9) than the originals (69.9), illustrating how test-informed wording inflates scores — [Chu et al. 2021](https://ar5iv.labs.arxiv.org/html/2012.04194)
- **Zero-shot intent splits differ by paper.** PIE-style evaluation uses 50 / 25 / 27 intent subsets of CLINC / HWU / BANKING. Hong's unseen-domain CLINC uses a 75/75 domain split, and UTCD places Banking77 and HWU out-of-domain but CLINC in-domain. Numbers are not comparable across these protocols — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39.pdf); [UTCD card](https://huggingface.co/datasets/claritylab/utcd)
- **D3ST.** Descriptions vs names differ little when test schemas were seen in training (MultiWOZ), but a lot on unseen services (SGD). Benchmark choice decides whether descriptions look useful — [Zhao et al. 2022](https://ar5iv.labs.arxiv.org/html/2201.08904)

### Inferences
- Protocol for this project:
  - Write or generate Banking77/CLINC/HWU descriptions only from the label inventory, plus at most training-split examples, and record that.
  - Freeze descriptions before any test evaluation.
  - Tune on a held-out label set (Gao's 20NG trick).
  - Always report the name-only score of the same checkpoint alongside "name: description".
  - Report a paraphrased-description score and the spread across description variants (SGD-X SS-style).
- If descriptions are LLM-generated from few examples, label the result "description (k-shot-informed)", not "zero-shot".
- Choose one published split protocol (PIE subsets or UTCD out-of-domain) to compare with prior work.

### Gaps
- No paper found that quantifies how much test-informed description writing inflates intent accuracy, e.g. descriptions written from test utterances vs from the label inventory only.

## Q5. Released datasets and description sets that can be reused directly

### Takeaway
Directly reusable today:
- Gao's LabelDesc (AGNews, Yahoo, DBPedia, 20NG, Yelp/SST sentiment), in the GitHub repo and paper appendix
- SGD/SGD-X schema descriptions with 5 paraphrase variants (intents and slots)
- DeepPavlov/AutoIntent intent tables on Hugging Face: 77 Banking77 descriptions, plus Qwen3-32B-generated descriptions for all 150 CLINC150 and 64 HWU64 intents
- UTCD's descriptive label names for 18 datasets

No released description files were found for Hong et al. 2024, Parikh et al. 2023 or Gen-Z (Gen-Z's are in supplementary material only).

### Cited Findings
- **LabelDescTraining repo.** `data/<dataset>/manual_24.tsv` ("our manually written LabelDesc data") plus `prompts.tsv` and code for LDT and domain transfer. The README visibly shows only the AGNews folder; the full LabelDesc lists (terms, templates, definitions) are in appendix Tables 16-21 of the paper. No license is stated on the repo page — [GitHub](https://github.com/lingyugao/LabelDescTraining); [Gao et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.02239)
- **SGD-X and evaluation script** released in google-research-datasets/dstc8-schema-guided-dialogue (4,755 crowd paraphrases; 5 variants per schema, covering intent and slot names and descriptions) — [Lee et al. 2021](https://ar5iv.labs.arxiv.org/html/2110.06800)
- **DeepPavlov (AutoIntent) Hugging Face datasets** (checked through the HF datasets-server API, 2026-11-18). The intent tables have fields `id, name, tags, regex_full_match, regex_partial_match, description`.
  - `DeepPavlov/banking77`, config `intents`: 77 of 77 intents have descriptions (e.g. activate_my_card: "User wants to activate his new card."; age_limit: "User is asking about age requirements or limitations for opening an account.")
  - `DeepPavlov/clinc150` and `DeepPavlov/hwu64`, config `intents`: descriptions are empty
  - Config `intentsqwen3-32b`: descriptions are filled (CLINC150 150 rows, first 100 checked all non-empty, e.g. accept_reservations: "User is asking whether a specific restaurant accepts reservations."; HWU64 64 of 64, e.g. alarm_query: "User wants to check their scheduled alarms.")
  - Note: the AutoIntent/* names now redirect to DeepPavlov/*. Provenance of the Banking77 descriptions is not stated, the "qwen3-32b" config name implies LLM generation, and the license field is empty in card metadata.
  — [HF card (AutoIntent/clinc150 → DeepPavlov/clinc150)](https://huggingface.co/datasets/AutoIntent/clinc150); [datasets-server API](https://datasets-server.huggingface.co/splits?dataset=DeepPavlov/banking77)
- **UTCD (claritylab/utcd), MIT license.** 18 datasets with natural-language descriptive labels, including Banking77, CLINC-150, SLURP, SGD, SNIPS, NLU-Evaluation, AG News, DBpedia, Yahoo, GoEmotions, Emotion and TweetEval. Labels are stored as integer ids with a separate name lookup. The full set is about 1.74 GB (not downloaded) — [HF dataset card](https://huggingface.co/datasets/claritylab/utcd)
- **ULR (Chu et al.).** Code and data at github.com/ZeweiChu/ULR. Label-name sets for AG, DBpedia, Yahoo and 20NG are in the paper appendix (repo contents not verified) — [Chu et al. 2021](https://ar5iv.labs.arxiv.org/html/2012.04194)
- **Yin et al. 2019.** Zero-shot benchmark datasets (Yahoo topic, emotion, situation) and code released through the CogComp project page — [Yin et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.00161)
- **X-Shot.** Code and data at github.com/xhz0809/X-shot — [Xu et al. 2024](https://arxiv.org/pdf/2403.03863)
- **B2MWT-10C.** URL-category benchmark released on GitHub with the 2026 definition-refinement preprint — [Rehmat et al. 2026](https://arxiv.org/abs/2604.27335)
- **Base intent datasets** (no descriptions): BANKING77 (PolyAI, CC-BY-4.0, 13,083 queries, 77 intents), HWU64 (xliuhw/NLU-Evaluation-Data), CLINC150 (clinc/oos-eval) — [PolyAI/banking77 card](https://huggingface.co/datasets/PolyAI/banking77)
- **Not released / not found.** Hong et al. 2024 give no code or data link on the Anthology page. Parikh et al. 2023 do not mention releasing descriptions (3 of 4 datasets are private). Gen-Z's paraphrases are in "supplementary material", with no repo found — [Hong et al. 2024](https://aclanthology.org/2024.sigdial-1.39/); [Parikh et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.07157); [Kumar et al. 2023](https://ar5iv.labs.arxiv.org/html/2311.07115)

### Inferences
- The DeepPavlov tables give an independent second description set for Banking77, CLINC150 and HWU64 at no cost. Using them as the held-out description wording at test time (with the project's own descriptions used for training, or vice versa) would give an SGD-X-style wording-robustness check on intent benchmarks.
- LabelDesc gives ready "name: description" material for AG News, DBpedia and Yahoo, comparable with published LDT numbers.
- SGD/SGD-X is the only released resource with multiple independent human paraphrases per label description. It suits training-time paraphrase augmentation and robustness evaluation on intents.

### Gaps
- No released description files were found for emotion datasets (GoEmotions, Emotion/CARER) beyond UTCD's descriptive names and Gen-Z's unreleased paraphrases.
- Licenses for the DeepPavlov description tables and the LabelDesc repo are not stated.
- The authorship (human vs LLM) of the DeepPavlov Banking77 descriptions is undocumented.
- The GoLLIE guideline files were not checked.
