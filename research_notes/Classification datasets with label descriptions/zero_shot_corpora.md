# Zero-shot / multi-task / instruction corpora that pair texts with label descriptions

Scope note: researched 2026-11-18 from Hugging Face cards, `datasets-server` first-rows samples, GitHub repos and arXiv. Sizes marked "(my count)" were computed by me from the card YAML, `sources.yaml` or a repo README, not stated by the authors. Nothing large was downloaded. "Overlap" means Banking77 (B77) or CLINC150 (CLINC) data is inside the resource, which matters because those are our held-out eval sets.

## Q1. Zero-shot classification training sets (UTCD, tasksource label-NLI, Laurer synthetic, GLiClass, zeroshot-v2.0): which have label descriptions?

### Takeaway
None of the big zero-shot training sets ships a "name: description" field per label. UTCD, tasksource label-NLI, GLiClass (clcp), Horizon-Labs and the v3 logic set use label names only. The closest thing to descriptions is Laurer's `synthetic_zeroshot_mixtral_v0.1` (`task_description` plus sentence-length `task_hypotheses`, one per class) and `knowledgator/gliclass-v2.0` (labels written as whole hypothesis sentences). Contamination is serious: UTCD contains both CLINC and B77, tasksource label-NLI contains both, and the non-"-c" zeroshot-v2.0 models were trained on a list that includes B77.

### Cited Findings
**UTCD (`claritylab/utcd`, Findings of ACL 2023, "Label Agnostic Pre-training for Zero-shot Text Classification", Clarke et al.)**
- 18 datasets (9 in-domain, 9 out-of-domain) over 3 aspects: sentiment, intent/dialogue, topic. About 6M train / 800K test. The card says "labels are standardized such that the labels are descriptive of the text in natural language", i.e. descriptive names, not descriptions. — [UTCD card](https://huggingface.co/datasets/claritylab/utcd)
- Split sizes: in-domain train 2,192,703 / test 168,365; out-of-domain train 4,996,673 / test 625,911. Also "aspect-normalized" configs (in-domain train 115,127; out-of-domain train 119,167). — [UTCD card YAML](https://huggingface.co/datasets/claritylab/utcd/raw/main/README.md)
- Label inventory (my count from the card YAML): 346 in-domain label names and 212 out-of-domain label names. Examples: "Add Alarm", "are you a bot", "card declined", "Bank account or service". The `labels` field is a sequence, so the format allows multi-label. Card license is MIT, but the component datasets keep their own licenses. — [UTCD card YAML](https://huggingface.co/datasets/claritylab/utcd/raw/main/README.md)
- **Overlap:** Clinc-150 and Banking77 are both listed as components. CLINC intents ("accept reservations", "are you a bot") sit in the in-domain label list. B77 intents ("activate my card", "card arrival", "card about to expire") sit in the out-of-domain list. Other intent sources: SGD, SLURP, Snips, NLU Evaluation. — [UTCD card](https://huggingface.co/datasets/claritylab/utcd)

**tasksource `zero-shot-label-nli` (Sileo 2023)**
- Tasksource classification tasks recast as NLI (premise = text, hypothesis = templated label). Train 1,899,993 / val 24,971 / test 25,440, plus a `triplet` config. Card license is "other", and every row carries `license` and `license_use` fields. — [card](https://huggingface.co/datasets/tasksource/zero-shot-label-nli)
- Hypotheses are built from label names, not descriptions. Verbatim: "This example is unsafe." (BeaverTails), "This example is 2 out of 6." (AES2), "This example is ambiguous." (AmbigNQ). — [first rows](https://datasets-server.huggingface.co/first-rows?dataset=tasksource/zero-shot-label-nli&config=default&split=train)
- `sources.yaml` lists 294 tasks (my count) with revision, license and row counts. **Overlap:** `banking77` (9,002 train rows, CC BY 4.0, "commercial") and `clinc_oos/plus` (10,000 train rows, CC BY 3.0) are both included, along with `multidogo/*/intent` tasks. **Non-commercial rows:** e.g. AES2 and BeaverTails are marked `cc-by-nc-4.0` / `non-commercial`. — [sources.yaml](https://huggingface.co/datasets/tasksource/zero-shot-label-nli/raw/main/sources.yaml)
- A multilingual variant `tasksource/multilingual-zero-shot-label-nli` exists (last modified 2023-06). I did not inspect it. — [HF hub search API](https://huggingface.co/api/datasets?search=zero-shot-label)

**`MoritzLaurer/synthetic_zeroshot_mixtral_v0.1` (2024, Apache-2.0)**
- Configs and row counts: `mixtral_refinedweb_categories` 739,362; `mixtral_refinedweb_characteristics` 543,242; `mixtral_refinedweb_nli` 94,428; `mixtral_written_texts_for_tasks` 105,806; `_v2` 156,096; `_v3` 679,516; `_v4` 308,586. Binary NLI format (`hypothesis`, `text`, `labels`). — [card YAML](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1)
- The `written_texts_for_tasks*` configs have `task_description` and `task_hypotheses` (one sentence per class), which work as label descriptions. Verbatim task_description: "Classify texts based on their writing style or literary genre. ... The classes are: prose, poetry, script." Verbatim hypotheses: "The text is written in prose form, utilizing paragraphs and sentences." and "The text is formatted as dialogue between characters with stage directions." — [first rows, v4](https://datasets-server.huggingface.co/first-rows?dataset=MoritzLaurer/synthetic_zeroshot_mixtral_v0.1&config=mixtral_written_texts_for_tasks_v4&split=train)
- Provenance: "500+ diverse text classification tasks for 25 professions" were created with Mistral-large, and texts were generated with Mixtral-8x7B-Instruct-v0.1. The `-c` zeroshot-v2.0 models use the `..._v4` subset. — [deberta-v3-large-zeroshot-v2.0 card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- The dataset card has YAML only and no prose (no provenance or overlap statement). — [card raw](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1/raw/main/README.md)

**zeroshot-v2.0 training data (Laurer, 2024)**
- `-c` variants: synthetic v4 plus MNLI and FEVER-NLI ("fully commercially-friendly"). Non-`-c` variants add ANLI, WANLI, LingNLI and "all datasets in this list where `used_in_v1.1==True`". Training uses entailment/not_entailment with a hypothesis template such as "This text is about {}." — [model card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0)
- The referenced list (`v1_human_data/datasets_overview.csv`, 41 rows) marks 27 rows `used_in_v1.1=TRUE` (my count). **These include `banking77` (cc-by-4.0) and `massive` (cc-by-4.0).** Rows flagged non-commercial or custom: IMDB, yelp_reviews, ag_news, emotion6_twitter, trueteacher (cc-by-nc-4.0), anli (cc-by-nc-4.0), empathetic_dialogues (cc-by-nc-4.0), financial_phrasebank (cc-by-nc-sa-3.0). CLINC is not in the list. — [datasets_overview.csv](https://raw.githubusercontent.com/MoritzLaurer/zeroshot-classifier/main/v1_human_data/datasets_overview.csv)
- **Conflict:** the WebFetch summary of the model card says banking77 and massive appear in the 28-task eval table and that "pure zeroshot" runs excluded them. The card's training description still points to the `used_in_v1.1` list, which contains banking77. — [model card](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0); [CSV](https://raw.githubusercontent.com/MoritzLaurer/zeroshot-classifier/main/v1_human_data/datasets_overview.csv)

**GLiClass (knowledgator) training data**
- `knowledgator/gliclass-v2.0`: 1,196,218 rows (`text`, `all_labels`, `true_labels`), Apache-2.0. The card has no provenance. Sample rows put a task instruction in front of the text and use descriptive hypothesis sentences as labels. Verbatim text prefix: "Identify the urgency of emails or messages to prioritize important or time-sensitive information. ...". Verbatim labels: "The text requires immediate attention or action.", "The text contains routine or non-critical information.". NLI-style rows also appear (label "We have plenty of oil."). `true_labels` can be empty. — [card](https://huggingface.co/datasets/knowledgator/gliclass-v2.0); [first rows](https://datasets-server.huggingface.co/first-rows?dataset=knowledgator/gliclass-v2.0&config=default&split=train)
- `knowledgator/gliclass-v3-logic-dataset`: 7,776 rows, Apache-2.0, multi-label with "1-50 true and 1-50 false labels", length-bucketed (4 to 1,024 words). Sources: FineWeb (LLM-labelled), tau/CommonsenseQA, BioMike/formal-logic-reasoning-gliclass-2k, MultiNLI. Labels are short strings, not descriptions. — [card](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset)
- GLiClass paper (Stepanov et al., arXiv 2508.07662, 2025): pre-training on about 1.2M unnamed classification/sentiment/NLI examples. Post-training uses a logic stream (CommonsenseQA plus 2,000 synthetic logic examples) and a pattern stream (2,000 texts from amazon_polarity and FineFineWeb, each annotated by GPT-4o with 50 true and 50 false labels). Labels are prefixed with a `<<LABEL>>` token, and the paper does not mention label descriptions. **banking77 is used only in evaluation; CLINC is not mentioned.** — [arXiv 2508.07662](https://arxiv.org/html/2508.07662)
- `gliclass-modern-base-v3.0` lists `knowledgator/gliclass-v3-logic-dataset` as training data (LoRA on logic tasks). — [model card](https://huggingface.co/knowledgator/gliclass-modern-base-v3.0)
- `aarabil/clcp_gliclass`: 9,051,961 train / 76,354 test rows (`text`, `hypothesis`, `labels`, `task_name`). There is no card prose. Hypotheses in the sample are bare names ("hobbies and interests", "travel", "music and audio"). — [card](https://huggingface.co/datasets/aarabil/clcp_gliclass); [first rows](https://datasets-server.huggingface.co/first-rows?dataset=aarabil/clcp_gliclass&config=default&split=train)

**`Horizon-Labs/multilingual-zeroshot-synthetic` (2026, ODC-BY)**
- 253,346 texts in 33 languages (English 59,665), with `candidate_labels`, `gold_labels` (multi-label for aspects) and an optional `hypothesis_template`. All generated or labelled by Qwen3.8-27B, with passages from FineWeb-Edu and FineWeb-2. Kinds: tax_short 94,639, short 77,616, tax_passage 47,368, aspect 21,725, passage 11,998. The card says "No benchmark data was used." Labels are names only, e.g. `business_trip`, `visa_support`. Template example: "The employee is requesting a {}." — [card](https://huggingface.co/datasets/Horizon-Labs/multilingual-zeroshot-synthetic); [first rows](https://datasets-server.huggingface.co/first-rows?dataset=Horizon-Labs/multilingual-zeroshot-synthetic&config=default&split=train)

### Inferences
- If we want "name: description" training pairs from these sets, Laurer's `task_hypotheses` and gliclass-v2.0's sentence labels are the only usable sources. Each hypothesis could be split into a short name plus the sentence as its description. gliclass-v2.0 looks like it is partly built from Laurer-style synthetic tasks: same instruction-prefix style and same hypothesis phrasing. This is unconfirmed because the card has no provenance.
- UTCD, tasksource label-NLI and the non-`-c` zeroshot-v2.0 models must be filtered or excluded if we keep B77 and CLINC strictly held out. Any off-the-shelf zeroshot-v2.0 (non-`-c`) or tasksource-NLI model is likely not zero-shot on B77.

### Gaps
- Whether UTCD's component licenses (e.g. Yelp, which Laurer's CSV marks non-commercial) restrict use. The card only says MIT.
- Provenance and task list of `knowledgator/gliclass-v2.0` and `aarabil/clcp_gliclass`. The `datasets-server` statistics call for clcp returned nothing.
- Whether the synthetic Laurer and Horizon-Labs data contain B77/CLINC-like banking intents. Not checked; it would need scanning the texts.

## Q2. Instruction corpora with per-task definitions (Super-NaturalInstructions, FLAN, P3/PromptSource): how many are classification tasks, and are option meanings described?

### Takeaway
Super-NI is the only one with a per-task natural-language "Definition". About 400 of its 1,616 tasks are in classification-type categories. Definitions usually list the allowed outputs, sometimes with a one-clause meaning for each, but they are not structured per-label descriptions. FLAN and P3 add option lists (`answer_choices`, the "OPTIONS" templates) with no option meanings. RAFT and SGD are smaller but give real per-task or per-intent descriptions, and RAFT contains Banking77.

### Cited Findings
- **Super-NaturalInstructions** (Wang et al., arXiv 2204.07705, 2022): "1,616 diverse NLP tasks and their expert-written instructions", 76 task types. — [arXiv](https://arxiv.org/abs/2204.07705)
- Schema: Definition (merging NatInst "Definition", "Things to Avoid" and "Emphasis & Caution"), Positive Examples (input, output, explanation) and Negative Examples. Verbatim definition snippets: "Indicate your answer with "1" if the first sentence entails the second sentence, otherwise answer with "0"."; "Label the instances as "cause" or "effect" based on your judgment." — [arXiv HTML](https://arxiv.org/html/2204.07705v3)
- Repo license: Apache-2.0. — [GitHub allenai/natural-instructions](https://github.com/allenai/natural-instructions)
- Category counts (my count from `tasks/README.md`, 1,613 tasks listed, 1,037 English→English): Sentiment Analysis 75, Text Categorization 46, Text Matching 43, Toxic Language Detection 40, Cause Effect Classification 37, Textual Entailment 27, Commonsense Classification 24, Language Identification 15, Answerability Classification 14, Text Quality Evaluation 12, Dialogue Act Recognition 7, Stereotype Detection 7, Gender Classification 7, Negotiation Strategy Detection 7, Answer Verification 7, Coherence Classification 6, Ethics Classification 6, Intent Identification 5, Word Relation Classification 5, Stance Detection 3, Section Classification 3, Fact Verification 3, Grammar Error Detection 3, Irony Detection 2, Speaker Relation Classification 2, plus several 1-task categories (Spam, Discourse Relation, etc.). Total in these classification-type categories ≈ 412 (my sum). — [tasks/README.md](https://github.com/allenai/natural-instructions/blob/master/tasks/README.md)
- Option meanings vary. task1720 (Civil Comments): "If the comment is toxic (harmful, malicious, derogatory, threat, insult, identity attck, etc.) output Yes, otherwise output No." This describes the positive class. task512 (Twitter emotion): "label the post's emotion (as expressed by the user) as sadness, joy, love, anger, fear, or surprise." This gives names only. — [task1720 JSON](https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task1720_civil_comments_toxicity_classification.json); [task512 JSON](https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task512_twitter_emotion_classification.json)
- No Banking77 or CLINC task names appear in Super-NI's task list (grep on "banking"/"clinc"). Intent tasks are e.g. task573_air_dialogue_classification. task889_goemotions_classification exists. — [tasks/README.md](https://github.com/allenai/natural-instructions/blob/master/tasks/README.md)
- **FLAN Collection (v2):** sub-mixtures "flan" (Flan 2021), "t0" (P3 excluding Flan 2021), "niv2" (Super-NI), "cot" and "dialog". Template varieties: zsopt, zsnoopt, fsopt, fsnoopt, i.e. with or without appended answer options. — [FLAN v2 README](https://github.com/google-research/FLAN/blob/main/flan/v2/README.md)
- **P3 / PromptSource:** "2'000 prompts collected for 270+ data(sub)sets", English. Each example has `answer_choices` (e.g. `['safe', 'trolley']`) and no per-choice descriptions. Used to train T0. — [P3 card](https://huggingface.co/datasets/bigscience/P3)
- **RAFT** (Real-world Annotated Few-shot Tasks): English classification tasks with 50 labelled train examples each, and per-task annotation instructions. **It includes Banking 77 (50 train / 5,000 test, CC BY 4.0).** Licenses are per dataset (e.g. ADE Corpus V2 "Unlicensed"). — [RAFT card](https://huggingface.co/datasets/ought/raft)
- **Schema-Guided Dialogue (SGD):** the schema has a human-written description for each service and intent. The train schema has 26 services and 53 intents (my count). Verbatim: "CheckBalance | Check the amount of money in a user's bank account"; "TransferMoney | Transfer money from one bank account to another user's account". SGD is also a UTCD component. — [SGD train/schema.json](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/blob/master/train/schema.json)

### Inferences
- Super-NI classification definitions could be mined into "label: meaning" pairs only for the minority of tasks whose definitions explain each output. Most just enumerate names. Expect a few hundred usable label sets, mostly binary or small.
- SGD intent descriptions are the cleanest human-written intent "name: description" source, and it has a banking domain (Banks_1). That domain is close to B77 in topic but is a different dataset.
- RAFT's banking_77 split is drawn from B77, so it must be excluded from training.

### Gaps
- The commonly cited "1,836 tasks" size of the FLAN Collection (Longpre et al. 2023) was not verified in this session.
- The exact number of P3/FLAN tasks with option lists was not computed.
- Super-NI's dataset-level license is Apache-2.0 for the repo, but individual tasks inherit source licenses. Not audited.
- SGD license (believed CC BY-SA 4.0) and the full intent count across train/dev/test were not verified.

## Q3. Relation and entity typing with type or relation descriptions (FewRel, zero-shot RE with Wikidata descriptions, fine-grained typing, Few-NERD, others)

### Takeaway
Relation extraction has the best native description resource: FewRel ships Wikidata names plus one-line descriptions for each property, and Wiki-ZSL/ZS-BERT builds on Wikidata attributes. Entity typing and NER mostly have names only (Few-NERD). Definitions come either from LLM generation (SLIMER, not confirmed released) or from annotation guidelines written as code (GoLLIE, which does not redistribute data).

### Cited Findings
- **FewRel** (`thunlp/FewRel`, MIT repo license): `data/pid2name.json` maps 744 Wikidata property IDs to [name, description] (5 have "No description defined", my count). Verbatim: `"P607": ["conflict", "battles, wars or other military engagements in which the person or item participated"]`; `"P2634": ["sitter", "person who posed during the creation of a work, ..."]`. — [pid2name.json](https://github.com/thunlp/FewRel/blob/master/data/pid2name.json); [repo](https://github.com/thunlp/FewRel)
- **ZS-BERT** (Chen & Li, NAACL 2021): zero-shot RE on WikiZSL (Sorokin & Gurevych 2017) and FewRel. The repo has a `resources/property_list.html` of Wikidata properties and provides data via Google Drive links. — [ZS-BERT repo](https://github.com/dinobby/ZS-BERT)
- **Few-NERD**: 8 coarse and 66 fine types, 188,200 sentences, 491,711 entities, CC BY-SA 4.0. Labels are names like `art-broadcastprogram` and there are no definitions. — [HF card](https://huggingface.co/datasets/DFKI-SLT/few-nerd)
- **SLIMER** (Zamai et al., arXiv 2407.01272, 2024): trained on a PileNER-type subset of 391 entity types × 5 positives = 3,910 sequences, with LLM-written definitions and guidelines (gpt-3.5-turbo-1106, one-shot prompt). Verbatim: "Definition: 'trailer' refers to a short promotional video that provides a preview or teaser of a forthcoming movie." / guideline "Label also general entertainment terms like 'preview' or 'teaser'." Evaluated on MIT, CrossNER and BUSTER. Only the model `expertai/SLIMER` is linked, so release of the definitions is unconfirmed. — [arXiv HTML](https://arxiv.org/html/2407.01272v3)
- **GoLLIE**: labels are Python classes and guidelines are docstrings. Code is Apache-2.0. "We do not redistribute the datasets used to train and evaluate GoLLIE", but data configs regenerate it. — [GoLLIE README](https://github.com/hitz-zentroa/GoLLIE)

### Inferences
- FewRel's `pid2name.json` (744 short descriptions) is a free, MIT-licensed set of "name: description" pairs, easy to recast as text→relation classification.
- None of these overlap with B77/CLINC.

### Gaps
- FewRel size (commonly 100 relations × 700 = 70,000 instances) and Wiki-ZSL size (commonly 113 relations, ~94k instances) were not verified here.
- Ultra-fine entity typing (Choi et al. 2018) and description-based entity typing (e.g. Obeidat et al. 2019) were not checked. Neither is known to ship type definitions.
- B2NERD, PileNER license, and GoLLIE guideline texts per dataset were not inspected.

## Q4. Typed-decision / System One corpora (jev-distill-corpus-v3, tasksource-jev, typed-decisions benchmarks): do options carry text descriptions ("criteria")?

### Takeaway
It depends on the source. `LocalLLaMA/typed-decisions` (the benchmark) has real per-option descriptions in `criteria` dicts. `tasksource/synthetic-typed-decisions` has the same structure but in the sample the description equals the name. Procedural sets give descriptive anchors only for `score` scales. The big training corpora (`SargeDev/jev-distill-corpus-v3`, `tasksource-jev-typed-decisions`, Open-Jev) use bare option strings. tasksource-jev includes B77, CLINC and MASSIVE.

### Cited Findings
- **`LocalLLaMA/typed-decisions`** (2026, Apache-2.0, synthetic): one state, five typed questions, and probability-distribution answers in the System One primitives (`noul`, `choice`, `score`). Test split: 400 cases, 2,000 decisions. Workflows: agent_trace_observability, customer_service, invoice_processing, security_incidents. Not affiliated with TypeSafe. — [card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- Its criteria carry descriptions. Verbatim: `"continue": "Let the agent proceed without interruption."`, `"stop": "Halt the agent now."`, noul `"true": "A human should inspect this run."`, score anchors `"Benign: read-only or clearly safe actions."`. — [first rows](https://datasets-server.huggingface.co/first-rows?dataset=LocalLLaMA/typed-decisions&config=all&split=test)
- **`SargeDev/jev-distill-corpus-v3`** (2026, Apache-2.0): 740,957 rows. Streams: yuri_v3 498,010 (53 domains, labels distilled from Jev 1.13 via OpenRouter), yuri_v1 148,154 (memory-relevance), openjev_v2 94,801 (CC0, from Open-Jev). Splits include train 655,806 and a held-out `test_set_30k` of 29,955. — [card](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3)
- Its options are bare: `["false","true"]`, `["0".."5"]`, with the question carrying the meaning ("Rate claim reliability for this scenario on a 0-5 scale."). — [first rows](https://datasets-server.huggingface.co/first-rows?dataset=SargeDev/jev-distill-corpus-v3&config=default&split=train)
- **`ericlewis/jev-distill-corpus-v3-typed-decisions`**: a reformat into the LocalLLaMA nine-column schema (394,484 records). It warns: "training and evaluation splits share many exact states." — [card](https://huggingface.co/datasets/ericlewis/jev-distill-corpus-v3-typed-decisions)
- **`tasksource/tasksource-jev-typed-decisions`** (2026, license "other", per-row license fields): default train 999,832 / val 15,000 / test 15,000; extraction config train 584,912. Configs: default, full, filtered-full, vision, extraction. `options` = "Criteria in their presented order", and they are bare strings (e.g. `["entailment","neutral","contradiction"]`, COPA answers). — [card](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions); [first rows](https://datasets-server.huggingface.co/first-rows?dataset=tasksource/tasksource-jev-typed-decisions&config=default&split=train)
- **Overlap:** its `sources.yaml` lists PolyAI/banking77 and legacy-datasets/banking77, clinc/clinc_oos (plus), AmazonScience/massive, mteb/MassiveIntentClassification and sonos snips_built_in_intents. — [sources.yaml](https://huggingface.co/datasets/tasksource/tasksource-jev-typed-decisions/raw/main/sources.yaml)
- **`tasksource/synthetic-typed-decisions`** (2026, Apache-2.0): workflows designed by DeepSeek V4 Flash and labelled by Jev 1.13, averaged with a blind DeepSeek check. `questions` = `{id: {type, instructions, criteria}}`. Train 6,666 items / 21,190 decisions / 235 workflows, val 1,176/3,996/41, test 1,074/3,475/38, split by workflow. Option counts range from 2 to 8, or 12 to 40 for routing questions. — [card](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions)
- In the sample row, criteria map each descriptive-phrase label to itself, e.g. `"Feature suggestion or game design proposal": "Feature suggestion or game design proposal"`. — [first rows](https://datasets-server.huggingface.co/first-rows?dataset=tasksource/synthetic-typed-decisions&config=default&split=train)
- **`tasksource/procedural-typed-decisions`** (Apache-2.0): `choice` criteria are null (`{"£0": null, ...}`), and `score` criteria are descriptive anchors (`"Fell by more than 50."`). Labels are exact (rule-computed). — [first rows](https://datasets-server.huggingface.co/first-rows?dataset=tasksource/procedural-typed-decisions&config=all&split=train)
- **`ZefanCai/Open-Jev-v1.1`** (2026, CC0/CC-BY-4.0 source-dependent, en and zh): 326,619 records (147,139 train). Options are bare IDs or yes/no (e.g. `["K11","item_82",...,"abstain"]`). — [card](https://huggingface.co/datasets/ZefanCai/Open-Jev-v1.1); [first rows](https://datasets-server.huggingface.co/first-rows?dataset=ZefanCai/Open-Jev-v1.1&config=community-hard-mix-v2-redistributable&split=train)
- The Hub has dozens of other 2026 "typed-decisions" sets (pngwn/typed-decisions-v2-system-one, n4ze3m/typed-decisions-synth, m-lagnajit/calibrated-typed-decisions, kshitijthakkar/nirnaya-jevbench-public-eval-v1, multilingual ones; `meiiny00/typed-decision-tl` is gated "manual"). — [HF search API](https://huggingface.co/api/datasets?search=typed-decision)

### Inferences
- For "name: description" training, LocalLLaMA/typed-decisions is the only source verified to have real descriptions, and it is a small benchmark. Using it for training would contaminate its leaderboard. synthetic-typed-decisions has the slot but may often leave it empty (description = name). This rests on one sampled row; a full scan is needed.
- tasksource-jev must be filtered by `source` to remove banking77, clinc and massive before training.

### Gaps
- LocalLLaMA/typed-decisions train size and per-option description coverage across all rows.
- Share of synthetic-typed-decisions criteria whose description differs from the name. Only one row was inspected.
- Size of procedural-typed-decisions.

## Q5. LLM-generated synthetic classification datasets with label descriptions (2023-2026) and purpose-built label-description training data (LabelDesc, Gao et al. 2023)

### Takeaway
LabelDesc is tiny (about 417 hand-written description "examples" across 6 unique label sets) but is exactly "label description as training data", and it gave +17-19 accuracy points over zero-shot. Released LLM-synthetic sets mostly give names plus a hypothesis template. Laurer's synthetic data, gliclass-v2.0 and the typed-decisions benchmark are the clearest released examples with sentence-level label descriptions.

### Cited Findings
- **LabelDesc** (Gao, Ghosh, Gimpel, EMNLP 2023, arXiv 2305.02239): fine-tune on label descriptions instead of annotated texts. Descriptions use "a few related terms, dictionary/encyclopedia entries, and short templates". — [arXiv](https://arxiv.org/abs/2305.02239)
- Coverage: AGNews (4), Yahoo (10), DBPedia (14), SST-2/5, Yelp-2/5, IMDB, Amazon-2; 20NG for tuning. Per-dataset counts: 20NG 24, AGNews 24, Yahoo 60, DBPedia 84, Yelp-5 125, Yelp-2 100. Topic labels get 6 descriptions each (term, 3 related terms, dictionary definition, Wikipedia sentence). Hand-written by the authors, no LLM. Results: +17 points with RoBERTa-base (72.2 vs 54.7) and about +19 with RoBERTa-large (77.7 vs 58.8). — [arXiv HTML](https://arxiv.org/html/2305.02239)
- Data repo `lingyugao/LabelDescTraining`: `data/{agnews,amazon2,dbpedia,imdb,sst2,sst5,yahoo,yahoo10,yelp,yelp2,finetune}/manual_24.tsv`. Verbatim rows: `1  humankind; the human race; humanity  dictionary.com for world`, `1  "In its most general sense, the term ""world"" refers to the totality of entities, ..."  Wikipedia first 5 sentence for world`. **No license is set on the GitHub repo.** — [repo](https://github.com/lingyugao/LabelDescTraining); [agnews manual_24.tsv](https://github.com/lingyugao/LabelDescTraining/blob/main/data/agnews/manual_24.tsv)
- **Incubator** (arXiv 2404.10877, 2024): an LLM tuned on instruction-to-data mappings built from HF classification datasets and their descriptions. It handles label definitions that go beyond a simple name. A public data release was not verified. — [arXiv HTML](https://arxiv.org/html/2404.10877v2)
- **AnyClassifier** (HF blog): a tool that generates synthetic classification data from label descriptions (e.g. with Llama 3.1 8B Instruct). Claims are the author's own. — [HF blog](https://huggingface.co/blog/kenhktsui/anyclassifier)
- Released synthetic sets with sentence-level label descriptions: Laurer `task_hypotheses` (Mixtral, 2024), `knowledgator/gliclass-v2.0` (provenance undocumented), `LocalLLaMA/typed-decisions` criteria (2026). Details and sources are in Q1 and Q4. Horizon-Labs (2026) has names plus templates only. — [Laurer](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1); [gliclass-v2.0](https://huggingface.co/datasets/knowledgator/gliclass-v2.0); [typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions); [Horizon-Labs](https://huggingface.co/datasets/Horizon-Labs/multilingual-zeroshot-synthetic)

### Inferences
- LabelDesc's main lesson is that a few hundred description strings per label set already help. That suggests generating "name: description" text for synthetic label sets (Laurer, Horizon-Labs) is cheap and likely helpful. LabelDesc has no intent datasets, so it does not overlap with B77/CLINC.
- The LabelDesc repo has no license, so treat its data as all-rights-reserved unless the authors say otherwise.

### Gaps
- I found no public large-scale (>10k label sets) dataset of LLM-written label descriptions built specifically for description-conditioned classification. Searches surfaced methods (Incubator, AnyClassifier, SLIMER), not released corpora.
- The LabelDesc total (about 417) is my sum of the paper's Table 15. The Amazon-2/IMDB/SST sets share files.

## Q6. Emotion / sentiment / stance datasets that publish label definitions

### Takeaway
GoEmotions publishes a one-line definition for each of its 27 emotions (Appendix A) and is multi-label. It is the standard example. GoEmotions also appears inside UTCD, Super-NI (task889) and tasksource-style mixes.

### Cited Findings
- **GoEmotions** (Demszky et al., ACL 2020, arXiv 2005.00547): 58,009 Reddit comments, 27 emotions plus Neutral, multi-label. Appendix A "Emotion Definitions", verbatim: admiration "Finding something impressive or worthy of respect."; amusement "Finding something funny or being entertained."; approval "Having or expressing a favorable opinion."; nervousness "Apprehension, worry, anxiety." Repo: google-research/goemotions. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00547)
- GoEmotions is a UTCD sentiment component and a Super-NI task (task889_goemotions_classification). — [UTCD card](https://huggingface.co/datasets/claritylab/utcd); [Super-NI tasks](https://github.com/allenai/natural-instructions/blob/master/tasks/README.md)
- Super-NI emotion tasks mostly list names only, e.g. task512: "...as sadness, joy, love, anger, fear, or surprise." — [task512 JSON](https://raw.githubusercontent.com/allenai/natural-instructions/master/tasks/task512_twitter_emotion_classification.json)
- Typed-decisions-style synthetic data covers "meaning, intent, tone, stance, plausibility, urgency" judgments with optional criteria text. — [synthetic-typed-decisions card](https://huggingface.co/datasets/tasksource/synthetic-typed-decisions)

### Inferences
- The 28 GoEmotions definitions are a ready-made "name: description" eval or training set with no B77/CLINC overlap. Pairing them with LabelDesc's sentiment sets gives a description-conditioned sentiment and emotion slice.

### Gaps
- GoEmotions data license (the HF card is commonly Apache-2.0) was not verified this session.
- Not checked for published label definitions: stance (SemEval-2016, P-Stance), propaganda techniques (SemEval-2020 Task 11 / 2023 Task 3), fallacies (MAFALDA), moral foundations, safety taxonomies (Aegis 2.0, BeaverTails categories). These are plausible sources of label definitions, but nothing about them was verified here.
- Yin et al. 2019 ("Benchmarking Zero-shot Text Classification"), which tested WordNet label definitions, was not verified.
