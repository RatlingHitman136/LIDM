# Public datasets for System One decision models (Jev, Laya)

> **Snapshot.** Copy of the survey "Public datasets for System One decision models (Jev, Laya)" (claude.ai
> doc, written 2026-11-09, copied 2026-11-15), the data-collection research behind
> [DESIGN.md](DESIGN.md)'s training plan. Not updated; sizes and licenses are as the cards stated then, so
> recheck before training a commercial model. What stage 1 actually uses is in `src/cls/data/stage1.py`.

2026-11-09 · Borys

This report lists 70 public datasets and benchmarks for System One decision models, grouped by the job they train or test. None of them is Jev's own training data, which TypeSafe has not published and describes as fully synthetic.

Three come closest. jev-distill-corpus-v3's largest stream carries Jev 1.13's output distributions, and LocalLLaMA/typed-decisions matches Laya's four-workflow benchmark. tasksource-jev adds 6.3M rows in the same state, question and options format.

The open reproductions (Kev, Tev1, Nimble, OpenJev) mostly train on classic classification and NLI sets plus generated policy rules. Check licenses before commercial training: DAIR Emotion, ANLI, XNLI, ChaosNLI and BeaverTails are research-only or non-commercial, and jev-distill-corpus-v3's labels are outputs of TypeSafe's hosted model.

## Jev- and Laya-specific datasets

Seven open training sets and six benchmarks already use the System One format: a state, typed questions (choice, score, noul) and a probability target. None is TypeSafe's own data, which is unpublished; the closest public proxy is jev-distill-corpus-v3, whose largest stream is labelled with Jev 1.13's output distributions.

### Training corpora

| Dataset | Contents | Size | Labels | License | Used by |
|---|---|---|---|---|---|
| [SargeDev/jev-distill-corpus-v3](https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3) | Synthetic operational scenarios across 53 domains, memory-relevance pairs from 18 QA sets, Open-Jev rows | ~741k rows, incl. a 29,955-question test set | Soft distributions: 498k rows from Jev 1.13, 148k from a 32B teacher | Apache-2.0 | [autotrust/JEV](https://huggingface.co/autotrust/JEV), OpenJev-8B |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | Controlled synthetic tasks in 12 configs (citation, entity alignment, extraction, email selection, browser control); English, Chinese, Turkish | ~520k rows | Generator targets; ambiguous cases split evenly | CC0-1.0 | 95k rows reused in jev-distill-corpus-v3 |
| [tasksource/tasksource-jev](https://huggingface.co/datasets/tasksource/tasksource-jev) | Hundreds of tasksource tasks recast as state, question, options and target; extra configs for extraction and vision | ~6.31M rows (144 GB) | Source annotations; soft targets keep annotator disagreement | Mixed, per-row license field | OpenJev-8B |
| [n4ze3m/typed-decisions-synth](https://huggingface.co/datasets/n4ze3m/typed-decisions-synth) | Synthetic states (reviews, transcripts, bug reports, API errors), 1–8 questions each, 149 domains | 7.41k rows | Gold answer plus a DeepSeek teacher distribution | MIT | OpenJev-8B |
| [Kev decision-v7 and later stages](https://github.com/jaredpalmer/kev) | 1,000 rows from each of 10 public classification sets, policy minimal pairs, rule structures; then dates, CFPB complaints, skills, dev tooling | 12,576 base records; ~30.5k across stages | Native labels or computed by code; no Jev outputs | Per source, recorded in suite manifests | Kev 0.8B–9B, OpenJev-8B |
| [Tev1 data recipe](https://github.com/togethercomputer/tev1) | MultiNLI, BoolQ, Banking77, AG News, SST-5, plus programmatic policies, priority routing, paper taxonomy | 37,840 examples | Source labels and generated rules; no Jev data | Builders MIT | Tev1 4B and 0.8B |
| [Bespoke Nimble data](https://github.com/bespokelabsai/nimble) | Contrastive pairs where one changed fact flips the answer; choice, noul and score | 2,676 train + 324 held out | Rules checked in code; synthetic, not human-reviewed | Apache-2.0 pipeline | Nimble 9B |

Kev's ten base sources are AG News, Amazon, Banking77, BoolQ, DBpedia-14, IMDb, MNLI, SST-5, TREC and Yelp ([Kev card](https://haimaker.ai/models/jaredpalmer/kev-4b), [source list](https://huggingface.co/ytbai/MiniCPM5-2B-Jev)). Tev1's mix is listed on its [Ollama page](https://ollama.com/library/tev1:4b).

### Benchmarks in the same format

| Benchmark | What it tests | Size | License | Notes |
|---|---|---|---|---|
| [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) | Agent-trace observability, customer service, invoice processing, security incidents; 5 questions per case | 1,600 cases: 1,200 train, 400 test (2,000 decisions) | Apache-2.0 | Appears to be [Laya's typed-decisions benchmark](https://huggingface.co/convaiinnovations/laya) (same workflows and test size): fine-tuned Laya 0.766, Jev 0.727 |
| [Leanmcp/jevbench](https://huggingface.co/datasets/Leanmcp/jevbench) | 12 text slices: MedMCQA, MedQA, MMLU-Pro, PubMedQA, ScienceQA, BANKING77, SST-5, ATBench, Aegis 2.0 (prompts, responses), prompt injections, jailbreaks | 8,016 cases; 24,799 stored predictions with probability vectors | Code MIT; each source keeps its own | [Preprint](https://www.preprints.org/manuscript/202610.0181), Oct 2026 |
| JevBench by Benchmark Heaven | Typed decisions, mostly rules, policy and law, then coding and math; scores accuracy, calibration, latency, cost | 1,500 per system: 300 public, 1,200 sealed | MIT (public items and harness) | Public items on GitHub at fstandhartinger/jevbench ([overview](https://parallel.ai/articles/jevbench-benchmark)) |
| [Kev eval suites](https://github.com/jaredpalmer/kev) | transfer-v4: QNLI, SciQ, TweetEval-offensive, PAWS, MMLU, Emotion plus held-out rules; breadth-v1: 14 unseen public sets | 656 and 3,089 questions | Per source | Never used in Kev training |
| Bespoke Labs public eval mix | 13 human-labelled sets, listed below | 3,880 decisions | Per source | Jev 76.0%, Nimble 9B 75.7%, Tev1 4B 73.3% ([Ollama](https://ollama.com/library/nimble:9b-q8_0)) |
| JevBench-mini coherence tests | Whether related answers agree (a question vs its negation, merged options); no gold labels | 240 cases, 1,200 tests | Not stated | Queen's University Belfast, jevbench.github.io |

The Bespoke mix covers MASSIVE (en-US, de-DE), MultiNLI, PubMedQA, VitaminC, BoolQ, SQuAD 2.0, PAWS, Civil Comments, Aegis 2.0, HelpSteer2 and SummEval (relevance, consistency).

## Zero-shot classification training data

These ten sets teach a model to score labels it has never seen, the skill Jev and Laya need to take new questions without retraining. Open encoder classifiers in the same mould, such as zeroshot-v2.0 and GLiClass, were trained on this mix of synthetic label data and NLI.

| Dataset | What it holds | Size | License | Used by |
|---|---|---|---|---|
| [MoritzLaurer/synthetic_zeroshot_mixtral_v0.1](https://huggingface.co/datasets/MoritzLaurer/synthetic_zeroshot_mixtral_v0.1) | Mixtral-8x7B-written texts for 500+ classification tasks across 25 professions | ~2.6M rows across configs | Apache-2.0 | [zeroshot-v2.0 classifiers](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c), GLiClass v1.0 |
| [tasksource/zero-shot-label-nli](https://huggingface.co/datasets/tasksource/zero-shot-label-nli) | tasksource classification tasks recast as NLI (text vs. a label statement) | 1.9M train, 25k validation, 25k test | Other; per-row license fields | tasksource DeBERTa NLI models |
| [claritylab/utcd](https://huggingface.co/datasets/claritylab/utcd) | 18 sets with descriptive labels across sentiment, intent/dialogue and topic; 9 in-domain, 9 out-of-domain | 2.19M train + 168k test in-domain; 5.0M + 626k out-of-domain | MIT | UTCD paper (Findings of ACL 2023) |
| [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) | 1,616 tasks with expert-written instructions, 76 task types, 55 languages | 1,462 training tasks, 154 evaluation tasks | Apache-2.0 repo; source datasets keep their own | Tk-Instruct |
| [knowledgator/gliclass-v3-logic-dataset](https://huggingface.co/datasets/knowledgator/gliclass-v3-logic-dataset) | Logic and NLI items in GLiClass format: text, true labels, candidate labels | 1k–10k rows | Not stated | GLiClass v3 |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | Premise–hypothesis pairs from 10 genres; entailment, neutral, contradiction | 393k train, 20k validation | Mostly OANC (permissive); parts CC BY-SA 3.0 | Tev1, Kev, zeroshot-v2.0-c |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | Image-caption premises with written hypotheses; 3 labels | 550k train, 10k validation, 10k test | CC BY-SA 4.0 | ANLI baseline models |
| [ANLI](https://huggingface.co/datasets/facebook/anli) | Adversarial NLI in 3 rounds, written to fool models | ~163k train, 3.2k dev, 3.2k test | CC BY-NC 4.0 | zeroshot-v2.0 (models without -c) |
| [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | GPT-3-generated NLI pairs revised and labelled by crowdworkers | 102,885 train, 5,000 test | CC BY 4.0 | zeroshot-v2.0 (models without -c) |
| [FEVER-NLI](https://huggingface.co/datasets/pietrolesci/nli_fever) | FEVER claims paired with Wikipedia evidence; 3 labels | 208k train, 20k dev, 20k unlabelled test | Not stated on card | zeroshot-v2.0-c |

## Benchmarks used to compare Jev and Laya

Published head-to-heads lean on a short list of classic classification sets, and the results split by label count: Laya leads on AG News and Emotion, while Jev leads on 77-label Banking77. Scores below are as published, from different prompts and samples, so treat them as indicative.

| Benchmark | Task and labels | Size | License | Published System One results |
|---|---|---|---|---|
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | News topic, 4 labels | 120k train, 7.6k test | Not stated; source corpus for research use | Laya 0.950, Jev 0.910; also in Kev and Tev1 training |
| [DAIR Emotion](https://huggingface.co/datasets/dair-ai/emotion) | Tweet emotion, 6 labels | 16k train, 2k validation, 2k test | Educational and research use only | Laya 0.595, Jev 0.480; Kev transfer-v4 |
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) | Banking intent, 77 labels | 13,083 queries | CC BY 4.0 | Jev 0.870, Laya 0.425; Jev 80.1% zero-shot in [MindStudio's test](https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark) |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | Assistant intent, 60 labels, 51 languages | ~1M utterances (19,521 per language) | CC BY 4.0 | Laya multilingual eval; Jev 87.4% on en-US in Bespoke's mix |
| [XNLI](https://huggingface.co/datasets/facebook/xnli) | NLI in 15 languages, 3 labels | 2,490 dev + 5,010 test per language; 393k machine-translated train | CC BY-NC 4.0 | Laya multilingual eval |
| [SST-5](https://huggingface.co/datasets/SetFit/sst5) | Sentiment on a 5-level ordinal scale | 8,544 train, 1,101 validation, 2,210 test | Not specified | Laya score questions; Kev, Tev1, Leanmcp JevBench |
| [Yelp Review Full](https://huggingface.co/datasets/Yelp/yelp_review_full) | Review stars, 1–5 | 650k train, 50k test | Yelp dataset agreement | Jev 67.2% vs a trained classifier's 51.9% (MindStudio); Kev |
| [IMDb](https://huggingface.co/datasets/stanfordnlp/imdb) | Movie review sentiment, 2 labels | 25k train, 25k test, 50k unlabelled | Other (see card) | Laya 92.33% in a reader test cited by [Raschka](https://magazine.sebastianraschka.com/cp/218483585); Kev |
| [BoolQ](https://huggingface.co/datasets/google/boolq) | Yes/no question over a passage | 9,427 train, 3,270 validation | CC BY-SA 3.0 | Jev 89.7%, Nimble 86.0%; Kev, Tev1 |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | Paraphrase vs. word-scrambled non-paraphrase | 65k labelled Wiki pairs | Free use; attribution requested | Jev 89.2%, Nimble 82.8%; Kev transfer-v4 |
| [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) | Biomedical yes/no/maybe | 1k expert-labelled, 211k generated | MIT | Jev 77.2%, Nimble 75.6%; Leanmcp JevBench |
| [VitaminC](https://huggingface.co/datasets/tals/vitaminc) | Contrastive fact verification against Wikipedia revisions | 400k+ claim–evidence pairs | CC BY-SA 3.0 | Jev 80.1%, Nimble 76.6% |
| [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | Whether a passage answers the question | 130k train, 12k dev | CC BY-SA 4.0 | Jev 82.9%, Nimble 80.6% (as RAG answerability) |

Open suites also draw on DBpedia-14, TREC and Amazon reviews (Kev decision-v7); QNLI, SciQ, TweetEval-offensive and MMLU (Kev transfer-v4); and MedMCQA, MedQA-USMLE, MMLU-Pro, ScienceQA and ATBench (Leanmcp JevBench). Jev and Nimble percentages are from [Bespoke Labs' run](https://ollama.com/library/nimble:9b-q8_0); Laya and Jev decimals from the [Laya card](https://huggingface.co/convaiinnovations/laya).

## Calibration and soft-label data

Seven sets give a probability or a full spread of human judgments per item. That is the target a calibrated System One model is scored against, by soft accuracy, Brier score or ECE, rather than a single gold label.

| Dataset | What it holds | Size | License | Label form |
|---|---|---|---|---|
| [ChaosNLI](https://github.com/easonnie/ChaosNLI) | SNLI, MNLI and αNLI dev items relabelled by 100 annotators each | 4,645 items; 464,500 annotations | CC BY-NC 4.0 | 100-vote label distribution |
| [UNLI (u-SNLI)](https://nlp.jhu.edu/unli/) | SNLI pairs rated for how likely the hypothesis is, given the premise | 61,597 pairs: 55,517 train, 3,040 dev, 3,040 test | Not stated | Probability in [0, 1] |
| [LeWiDi-2025](https://le-wi-di.github.io/) | 4 sets: sarcasm (CSC), irony in 9 languages (MultiPICo), paraphrase, NLI with explanations (VariErr) | 26,814 items; 130k annotations | Not stated in the [task paper](https://arxiv.org/pdf/2510.08460) | Soft labels plus per-annotator labels; Likert and categorical |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | Reddit comments tagged with 27 emotions plus neutral, multi-label | 58k comments, 3–5 raters each | Apache-2.0 | Per-rater labels in the raw config |
| [DICES](https://github.com/google-research-datasets/dices-dataset) | Chatbot conversations rated for safety by demographically balanced rater pools | 350 conversations × 123 raters; 990 × 60–70 raters | CC BY 4.0 | Rater-vote distributions, split by demographic group |
| [ForecastBench](https://huggingface.co/datasets/forecastingresearch/forecastbench-datasets) | Questions about future events from prediction markets and data series, resolved later | 1,000 per round from a bank of 5,900+; new rounds every two weeks | CC BY-SA 4.0 | Binary outcomes scored by Brier score |
| [Autocast](https://github.com/andyzoujm/autocast) | Forecasting-tournament questions with a dated news corpus | Thousands of questions | Research use only (Metaculus permission) | Crowd forecasts and resolved outcomes |

Civil Comments (in the moderation section) also stores toxicity as the share of raters who flagged each comment, so it doubles as soft-label data.

## Routing data

Routing is the most common System One job. Public data covers three kinds of route: which model should answer, which intent or queue a message belongs to, and whether an agent should call a tool or act at all.

| Dataset | Routing decision | Size | License | Notes |
|---|---|---|---|---|
| [RouterBench](https://huggingface.co/datasets/withmartian/routerbench) | Which of 11 LLMs should take a prompt, given cost and quality | 30k+ prompts; 405k+ scored responses from 8 benchmarks | Not shown on card | Precomputed outcomes, so routers train without inference ([paper](https://arxiv.org/html/2403.12031v1)) |
| [RouteLLM data](https://github.com/lm-sys/RouteLLM) | Strong vs. weak model, learned from human preference | ~80k Chatbot Arena battles plus judge-labelled augmentation | Not stated in the [release post](https://lmsys.org/blog/2024-07-01-routellm/) | routellm/gpt4_dataset adds GPT-4-scored Mixtral answers |
| [When2Call](https://huggingface.co/datasets/nvidia/When2Call) | Call a tool, ask a follow-up, say it can't, or answer directly | 3,652 test questions; 15k SFT + 9k preference train | CC BY 4.0 | Multiple-choice format maps onto choice questions |
| [CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) | Intent across 150 classes, plus out-of-scope | 23,700 queries | CC BY 3.0 | Tests "none of these" handling |
| [Bitext customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | Support intent and category | 26,872 utterances, 27 intents, 10 categories | CDLA-Sharing-1.0 | Synthetic ticket-style text |
| [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) | Product, sub-product and issue for financial complaints | Millions of complaints; narratives where consumers opted in | US government work | Kev's documents-v1 uses 5,219 narratives |
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | Next browser action on a real web page | 2,350 tasks on 137 websites | CC BY 4.0 | Used by OpenJev-8B for agent decisions |

Banking77 and MASSIVE, in the benchmark table above, are the usual intent-routing evaluations. OpenJev-8B also trains on NNetNav-live browser steps (Apache-2.0).

## Policy- and schema-conditioned decisions

These sets hand the model rules or a schema at run time, the way a Jev-style request carries its own options and criteria. Rules, policy and law is also the largest topic group in Benchmark Heaven's JevBench.

| Dataset | Decision | Size | License | Why it fits |
|---|---|---|---|---|
| [Schema-Guided Dialogue (SGD)](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) | Intents and slot values from a service schema given at run time, including unseen services | 20k+ dialogues, 20 domains | CC BY-SA 4.0 | Zero-shot transfer to new schemas, like new option sets |
| [MultiWOZ 2.2](https://github.com/budzianowski/multiwoz) | Dialogue state: slot values per turn | 10k+ dialogues, 7 domains | MIT | Standard state-tracking benchmark |
| [ABCD](https://github.com/asappresearch/abcd) | Agent actions that follow company guidelines for each user intent | 10k+ dialogues, 55 intents | MIT | Policy-constrained action choice |
| [ShARC](https://sharc-data.github.io/) | Apply a rule text to a user scenario: yes, no, irrelevant, or ask a follow-up | ~32k utterances | CC BY-SA 3.0 | Rule application with a "need more information" option |
| [ContractNLI](https://stanfordnlp.github.io/contract-nli/) | Whether a contract entails, contradicts or doesn't mention each of 17 hypotheses | 607 contracts | CC BY 4.0 | Typed checks over long documents |
| [LegalBench](https://huggingface.co/datasets/nguha/legalbench) | Legal reasoning tasks, many of them yes/no or choice rule application | 162 tasks | Per task | Close to JevBench's rules-and-policy items |
| [τ-bench](https://github.com/sierra-research/tau-bench) | Agent tool use under retail and airline policies | 165 tasks (115 retail, 50 airline) | MIT | Policy compliance inside an agent loop |

## Safety, moderation and fraud decisions

Guardrail checks are yes/no or category decisions, so moderation sets drop straight into the noul and choice formats. Aegis 2.0 and WildGuardTest already feed System One leaderboards.

| Dataset | Decision | Size | License | Notes |
|---|---|---|---|---|
| [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) | Prompt and response safety across 12 core hazard categories plus 9 fine-grained ones | 28k–33k interactions depending on card version; 1,964 test | CC BY 4.0 | Jev 80.4%, Nimble 81.2% (Bespoke mix); Kev devtools-v1; Leanmcp JevBench |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) | Prompt harm, response harm and refusal, for direct and adversarial prompts | 86,759 train; 1,725-item test (5,299 incl. prompt-only) | ODC-BY, gated | WildGuardTest is 25% of the Artificial Analysis Decision Index |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Whether a QA pair is harmful, across 14 harm categories | 330k QA pairs | CC BY-NC 4.0 | Large safety-classifier training set |
| [Civil Comments](https://huggingface.co/datasets/google/civil_comments) | Toxicity and attributes, scored as the share of raters who flagged each comment | ~2M comments | CC0 | Jev 81.0%, Nimble 70.3% (Bespoke mix) |
| [OpenAI moderation eval](https://github.com/openai/moderation-api-release) | 8 moderation categories per prompt | 1,680 prompts | MIT | Common moderation baseline |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Prompt injection or not | 662 prompts (546 train, 116 test) | Apache-2.0 | Leanmcp JevBench's 116-case slice |
| [SMS Spam Collection](https://huggingface.co/datasets/ucirvine/sms_spam) | Spam or not | 5,574 messages | CC BY 4.0 | Small spam and scam check |

Leanmcp JevBench also uses jailbreak-classification (400 cases) and ATBench (500 agent-trajectory cases).

## Judging and preference

A System One judge picks the better response or scores one against a rubric, which maps onto choice and score questions. Rubric scoring is Jev's weakest area in Bespoke's mix, at 34–35% exact-level accuracy on HelpSteer2 and SummEval relevance.

| Dataset | Judgment | Size | License | Notes |
|---|---|---|---|---|
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Score responses 0–4 on helpfulness, correctness, coherence, complexity and verbosity | 21,362 (20,324 train, 1,038 validation) | CC BY 4.0 | Jev 34.1%, Nimble 39.0% exact level |
| [SummEval](https://github.com/Yale-LILY/SummEval) | Rate summaries 1–5 on coherence, consistency, fluency and relevance | 1,600 summaries (100 articles × 16 systems) | MIT | Relevance: Jev 35.0%, Nimble 49.2%; consistency: Jev 81.2%, Nimble 75.7% |
| [Feedback Collection](https://huggingface.co/datasets/prometheus-eval/Feedback-Collection) | Score a response 1–5 against a rubric supplied with the request | 100k judgments over 1k rubrics | CC BY 4.0 | Trains Prometheus judges; rubric arrives at run time |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Rate 4 responses per prompt on 4 aspects, labelled by GPT-4 | 64k prompts, 256k responses | MIT | Large preference source |
| [RewardBench](https://huggingface.co/datasets/allenai/reward-bench) | Pick the better of two responses across chat, safety and reasoning | 2,985 pairs | ODC-BY | Standard pairwise judge test |
| [MT-Bench human judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments) | Expert pairwise votes on multi-turn answers | 3.3k judgments | CC BY 4.0 | Human reference for LLM judges |

## Sources

- [Laya model card (convaiinnovations/laya)](https://huggingface.co/convaiinnovations/laya) and [training notes (Emerald7664/laya)](https://huggingface.co/Emerald7664/laya)
- [autotrust/JEV model card](https://huggingface.co/autotrust/JEV)
- [OpenJev-8B model card](https://huggingface.co/alanhuangya/OpenJev-8B), which lists its training sets
- [Kev-4B model card](https://haimaker.ai/models/jaredpalmer/kev-4b) and [MiniCPM5-2B-Jev card](https://huggingface.co/ytbai/MiniCPM5-2B-Jev) for the decision-v7 sources
- [Tev1 on Ollama](https://ollama.com/library/tev1:4b) and [Nimble on Ollama](https://ollama.com/library/nimble:9b-q8_0)
- [Nimble training data write-up (Latenode)](https://latenode.com/ai-trends/nimble-typed-decision-finetune)
- [JevBench explained (Parallel)](https://parallel.ai/articles/jevbench-benchmark) and [Leanmcp JevBench preprint](https://www.preprints.org/manuscript/202610.0181)
- [MindStudio: Jev vs classic classifiers](https://www.mindstudio.ai/blog/jev-vs-classic-classifiers-benchmark)
- [Raschka, Ahead of AI](https://magazine.sebastianraschka.com/cp/218483585) and [TechCrunch](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) on Jev's synthetic training data
- [zeroshot-v2.0 model card](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0-c)
- [ChaosNLI repo](https://github.com/easonnie/ChaosNLI), [UNLI paper](https://arxiv.org/pdf/1909.03042), [LeWiDi-2025 paper](https://arxiv.org/pdf/2510.08460), [DICES paper](https://arxiv.org/pdf/2306.11247), [ForecastBench paper](https://arxiv.org/pdf/2409.19839)
- [RouterBench paper](https://arxiv.org/html/2403.12031v1), [RouteLLM release post](https://lmsys.org/blog/2024-07-01-routellm/), [WildGuard paper](https://arxiv.org/pdf/2406.18495v2)

Sizes and licenses are as stated on each linked card or paper. Terms change, so recheck them before training a commercial model.
