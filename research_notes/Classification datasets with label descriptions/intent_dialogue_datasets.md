# Intent / dialogue / customer-support / tool-selection datasets with label names + descriptions

Scope note: researched 2026-11-18. Counts marked "(verified)" were computed by the researcher directly from the released files or the Hugging Face datasets-server API, not copied from a paper. Held-out evaluation sets for this project: Banking77 and CLINC150, so every entry gets an overlap note.

## Q1. Which task-oriented dialogue datasets ship intent and slot descriptions in schemas (SGD, SGD-X, MultiWOZ 2.2, STAR, others)? How many distinct intents, and how are descriptions written?

### Takeaway
SGD is the main human-written "name: description" source among dialogue datasets. Across train, dev and test it has 45 services in 20 domains, 88 service-intent pairs (46 unique intent names) and 365 slots, each with a short imperative description of about 8 words. SGD-X adds 5 paraphrased variants of every schema, renaming and rewording both names and descriptions, so the label set grows about 6x. MultiWOZ 2.2 adds only 11 intents. STARv2 adds natural-language descriptions for user and system actions over 24 tasks. ABCD ships long agent-guideline text for 55 customer-service subflows.

### Cited Findings
**Schema-Guided Dialogue (SGD / DSTC8), 2019-2020**
- Over 20k annotated multi-domain task-oriented conversations between a human and a virtual assistant, covering 20 domains (banks, events, media, calendar, travel, weather and others). Most domains have several APIs with overlapping functions but different interfaces. — [SGD GitHub README](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- (verified) Schema counts: train has 26 services and 53 intents, dev has 17 services and 30 intents, test has 21 services and 38 intents. Deduplicated: 45 unique services, 88 unique (service, intent) pairs, 46 unique intent names, 86 unique intent descriptions, 365 slots. Mean intent description length is 8.2 words. The 20 domains are Alarm, Banks, Buses, Calendar, Events, Flights, Homes, Hotels, Media, Messaging, Movies, Music, Payment, RentalCars, Restaurants, RideSharing, Services, Trains, Travel, Weather. — [SGD train schema.json](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/train/schema.json), [dev](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/dev/schema.json), [test](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/test/schema.json)
- Verbatim intent descriptions (verified): `Banks_1.CheckBalance` -> "Check the amount of money in a user's bank account"; `Banks_1.TransferMoney` -> "Transfer money from one bank account to another user's account"; `Buses_2.FindBus` -> "Find a bus itinerary between cities for a given date"; `Alarm_1.AddAlarm` -> "Set a new alarm"; `Payment_1.MakePayment` -> "Send money to your friends". Service description: `Banks_1` -> "Manage bank accounts and transfer money". Slot description: `account_type` -> "The account type of the user". — [SGD train schema.json](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/train/schema.json)
- License: CC-BY-SA-4.0 on both GitHub and the HF mirror. English only. — [GitHub license API / repo](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue); [HF card](https://huggingface.co/datasets/google-research-datasets/schema_guided_dstc8)
- Labels are dialogue-level active intents (one active intent per frame), not single-utterance classification. Each turn has its own frame. — [SGD GitHub README](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)

**SGD-X (AAAI 2022)**
- Gives "semantically similar yet stylistically diverse variants for every schema" in SGD. Two strong state-tracking models generalize poorly across the variants. Published in AAAI 2022, vol. 36 no. 10, pp. 10938-10946. — [arXiv 2110.06800](https://arxiv.org/abs/2110.06800)
- Released in the SGD repo on 10/19/2021 as `sgd_x/data/v1` to `v5`, each with train, dev and test `schema.json`. The researcher fetched v1 to v5. — [SGD repo sgd_x](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue/tree/master/sgd_x)
- Variants paraphrase service names, intent names and descriptions together. Verbatim (verified), the same bank service:
  - v1: service `Banks_11` "Banking service for managing accounts and money transfers", with intents `CheckBankBalance` -> "See amount available in account" and `TransferFunds` -> "Transfer money from one of your accounts to another person's account".
  - v5: service `Banks_15` "Financial app that allows the tracking and transfer of funds from one account to another", with intents `ProvideBankAccountBalance` -> "Show available funds in a specified banking account" and `TransferMoneyBetweenAccounts` -> "Send money to someone else".
  - Sources: [SGD-X v1 train schema](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/sgd_x/data/v1/train/schema.json), [v5 train schema](https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/sgd_x/data/v5/train/schema.json)

**MultiWOZ 2.2 schema**
- (verified) 8 services and 11 intents, written as short lowercase phrases. Verbatim: `find_hotel` -> "search for a hotel to stay in"; `find_restaurant` -> "search for places to wine and dine"; `book_taxi` -> "book taxis to travel between places"; `police` -> "search for police station". — [MultiWOZ 2.2 schema.json](https://raw.githubusercontent.com/budzianowski/multiwoz/master/data/MultiWOZ_2.2/schema.json)
- Repo license: MIT. — [budzianowski/multiwoz](https://github.com/budzianowski/multiwoz)

**STAR (2020) and STARv2 (2023)**
- STAR: "A Schema-Guided Dialog Dataset for Transfer Learning" by Mosig, Mehri and Kober (arXiv 2010.11853). Repo license: MIT. — [RasaHQ/STAR](https://github.com/RasaHQ/STAR); [arXiv 2010.11853](https://arxiv.org/abs/2010.11853)
- STARv2: "richer annotations provided by the D3ST model", released 03/28/2023 as `STARv2.zip` (18.7 MB) in the archived google-research/task-oriented-dialogue repo (Apache-2.0). It was introduced in the AnyTOD paper. — [starv2 folder](https://github.com/google-research/task-oriented-dialogue/tree/main/starv2); [repo](https://github.com/google-research/task-oriented-dialogue); [AnyTOD arXiv 2212.09939](https://arxiv.org/abs/2212.09939)
- AnyTOD: STARv2 adds natural-language descriptions for actions. Earlier work used template utterances such as "[CITY]", and STARv2 gives "user is informing city" instead. — [AnyTOD arXiv 2212.09939](https://arxiv.org/pdf/2212.09939)
- (verified) The zip holds 6,652 dialogues, 24 task folders (each with `<task>.json` and `responses.json`) and 24 API schema JSONs. API schemas list typed input and output fields with `ReadableName` but no API description. Task JSON keys: `task, replies, slot_actions, graph, r_graph, useract_descs, sysact_descs`. Verbatim `useract_descs` from restaurant_book: `user_restaurant_book_ask_restaurant_book` -> "user wants to make a reservation at a restaurant"; `user_hello` -> "user is saying hello". `sysact_descs`: `query_check` -> "query restaurant api to check if reservation is available". The task names cover 13 groups: bank, restaurant, hotel, ride, plane, party, meeting, apartment, doctor, trip, weather, trivia, spaceship. — [STARv2.zip](https://github.com/google-research/task-oriented-dialogue/tree/main/starv2)

**ABCD, Action-Based Conversations Dataset (ASAPP, NAACL 2021)**
- Over 10K human-to-human customer-service dialogues with 55 distinct user intents (subflows), each needing its own sequence of actions. The intent-classification task has 55 subflow options. Ships `data/guidelines.json` (parsed agent guidelines), `kb.json` and `ontology.json`. Repo license: MIT. — [asappresearch/abcd README](https://github.com/asappresearch/abcd); [arXiv 2104.00783](https://arxiv.org/abs/2104.00783)
- Description form: a short flow-level description plus long per-subflow action instructions, not a one-line intent gloss. Verbatim: flow "Product Defect" has description "refunds and returns", and subflow "Initiate Refund" begins "All chats in this flow start by asking the customer for their full name or Account ID using [Pull up Account]…". — [ABCD guidelines.json](https://raw.githubusercontent.com/asappresearch/abcd/master/data/guidelines.json)

### Inferences
- SGD, SGD-X and STARv2 are the cleanest sources of human-written "name: description" pairs from dialogue data, but the raw label inventory is small (88 intents in SGD, 11 in MultiWOZ, 24 tasks in STAR). SGD-X multiplies the surface forms: 6 schema versions × 88 intents ≈ 528 distinct name and description strings, which is useful for description-paraphrase robustness. This is an estimate: the researcher did not count v1 to v5 one by one.
- SGD intents are labelled on dialogue frames, not single utterances. To get "short message -> intent" pairs you would take the first user turn whose active intent changes. That is a derivation, not an official split.
- Overlap risk: SGD Banks_1 and Banks_2 (CheckBalance, TransferMoney) and Payment_1 overlap conceptually with Banking77 transfer and balance labels and with CLINC150's banking domain. SGD Alarm, Calendar, Weather, Music, Flights and similar domains overlap CLINC150 domains semantically. There is no utterance-level overlap, since SGD dialogues were generated separately.
- ABCD guideline text is much longer and procedural. It would need summarising before use as a "name: description" option.

### Gaps
- Exact SGD dialogue counts per split were not re-verified this session (the README says only "over 20k").
- STAR v1 task and domain counts were not taken from the paper. The 24 tasks and 6,652 dialogues come from the STARv2 zip contents.
- SGD-X paraphrase procedure (crowdsourced or not) was not confirmed: the arXiv abstract page does not say.
- Other schema-style TOD datasets (BiToD, Taskmaster-3, SpokenWOZ, X-RiSAWOZ) were not checked for description fields.

## Q2. Which intent-classification datasets (customer support, banking, assistants) have official or widely reused intent descriptions, and where are those descriptions?

### Takeaway
Of the classic intent benchmarks, only NLU++ (PolyAI) ships official descriptions: 62 intents written as yes/no questions. Bitext, MASSIVE, HWU64, ATIS, SNIPS, CLINC150, Banking77, MINDS-14 and Redwood ship names only. Public descriptions do exist as add-ons:
- DeepPavlov's AutoIntent re-uploads (2025) include LLM-generated (Qwen3-32B) one-sentence descriptions for CLINC150, HWU64, MASSIVE, MINDS-14 and SNIPS, but not Banking77.
- Hu et al. (ACL 2024 NLP4ConvAI workshop) released hand-written "User is asking / User wants…" descriptions for ATIS, SNIPS, CLINC and MASSIVE.

### Cited Findings
**NLU++ (PolyAI, NAACL Findings 2022). Official descriptions, multi-label.**
- Two domains, banking and hotels: 2,071 banking examples with 48 intents and 13 slots, 1,009 hotel examples with 40 intents and 14 slots, 3,080 examples and 62 intents in total. Multi-label ("intent modules" that combine, e.g. `how_much` + `transfer_payment_deposit`). Ships in 20 folds per domain. — [NLU++ README](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/nlupp); [arXiv 2204.13021](https://arxiv.org/abs/2204.13021)
- (verified) `nlupp/data/ontology.json` gives all 62 intents with a `description` written as a question plus a `domain` list. Verbatim: `handoff` -> "is the intent to speak to a human or hand off?"; `make_open_apply_setup_get_activate` -> "is the intent to make, open, apply, set up or activate something?"; `how_much` -> "is the intent asking about some quantity or how much?". — [NLU++ ontology.json](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/nlupp/data/ontology.json)
- License: CC-BY-4.0, English. — [PolyAI-LDN/task-specific-datasets](https://github.com/PolyAI-LDN/task-specific-datasets)

**Bitext customer-support family (2023-2024). Names only.**
- Main dataset (verified): 26,872 rows, 27 intents in 11 categories. Intents include `track_order`, `get_refund`, `contact_human_agent`, `recover_password`, `check_cancellation_fee`, `payment_issue`, `registration_problems`. No description field. License: CDLA-Sharing-1.0. English. Created 2023-08-24. — [HF card](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset); [datasets-server stats](https://datasets-server.huggingface.co/statistics?dataset=bitext/Bitext-customer-support-llm-chatbot-training-dataset&config=default&split=train)
- Vertical datasets (verified rows / intents / categories):
  - retail-banking 25,545 / 26 / 9
  - insurance 39,000 / 39 / 17
  - telco 26,000 / 26 / 7
  - travel 31,658 / 33 / 11
  - retail-ecommerce 44,884 / 46 / 13
  - mortgage-loans 36,778 / 37 / 8
  - wealth-management 23,806 / 24 / 5
  - hospitality 25,000 / 25 / 16
  - events-ticketing 24,702 / 25 / 8
  - media 24,627 / 25 / 8
  - restaurants 30,000 / 30 / 13
  - Sources: [Bitext HF org listing](https://huggingface.co/bitext); [e.g. retail-banking stats](https://datasets-server.huggingface.co/statistics?dataset=bitext/Bitext-retail-banking-llm-chatbot-training-dataset&config=default&split=train)

**MASSIVE (Amazon, 2022). Names only, multilingual.**
- More than 1M utterances across 52 languages (MASSIVE 1.1 adds Catalan to the 51 languages of 1.0), with 60 intents and 55 slot types. Built by localizing SLURP (single-shot voice-assistant interactions). — [alexa/massive README](https://github.com/alexa/massive)
- License: CC-BY-4.0. — [HF AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive)
- 60 intents across 18 domains; the en-US split has 16,521 instances. — [Hu et al. 2024, arXiv 2407.17862](https://arxiv.org/html/2407.17862)

**HWU64 (NLU-Evaluation-Data, IWSDS 2019). Names only.**
- 25,716 annotated lines with scenario and intent columns (home-assistant / human-robot interaction). License: CC BY 4.0. — [xliuhw/NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data)

**CLINC150 and Banking77 (held-out sets). Names only.**
- Banking77: CC-BY-4.0, English. — [HF PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- CLINC150 (clinc_oos): CC-BY-3.0, English. — [HF clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos)
- CLINC: 150 intents across 10 domains, 22,500 in-scope utterances, plus out-of-scope examples. — [Hu et al. 2024](https://arxiv.org/html/2407.17862)
- (verified) AutoIntent re-uploads: Banking77 has 10,003 train and 3,080 test (13,083 total) with 77 intents. CLINC150 has 15,200 / 3,100 / 5,500 rows. — [DeepPavlov/banking77](https://huggingface.co/datasets/DeepPavlov/banking77); [DeepPavlov/clinc150](https://huggingface.co/datasets/DeepPavlov/clinc150)

**ATIS and SNIPS. Names only.**
- ATIS: 18 intents after filtering to classes with at least 5 examples. SNIPS-NLU: 7 intents, 14,484 utterances. — [Hu et al. 2024](https://arxiv.org/html/2407.17862)
- benayas/snips HF mirror: Apache-2.0. — [HF benayas/snips](https://huggingface.co/datasets/benayas/snips)

**MINDS-14 (PolyAI). Names only, multilingual, spoken.**
- E-banking intents. CC-BY-4.0, 12 languages (en, fr, it, es, pt, de, nl, ru, pl, cs, ko, zh). — [HF PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14)
- (verified) The English AutoIntent re-upload has 14 intents, 431 train and 108 test. — [DeepPavlov/minds14](https://huggingface.co/datasets/DeepPavlov/minds14)

**Redwood (SIGDIAL 2022). Names only. Largest merged intent set.**
- 451 intents from 13 existing intent datasets, with colliding intents arbitrated, described as the largest public intent-classification benchmark. — [ACL Anthology 2022.sigdial-1.45](https://aclanthology.org/2022.sigdial-1.45)
- (verified) `redwood_test.json` has 9,553 examples covering all 451 labels. Each example has fields `text`, `source dataset`, `source intent` and an opaque `label` (e.g. "intent_1000"); there is no description. Distinct source intents by dataset: acid 112, atis 1, banking77 77, clinc150 150, dstc8 33, hwu 53, jobs640 1, metalwoz 41, mtop 59, new 52, outlier 10, pto 6, snips 7, talk2car 1. The repo has no license file (GitHub API returns "Not Found" for a license). — [gxlarson/redwood](https://github.com/gxlarson/redwood); [redwood_test.json](https://raw.githubusercontent.com/gxlarson/redwood/main/data/redwood_test.json)

**Public add-on descriptions for existing benchmarks**
- DeepPavlov / AutoIntent (2025):
  - (verified) HF re-uploads ship an `intents` config with columns `id, name, tags, regexp_full_match, regexp_partial_match, description`. In the plain `intents` config, `description` is null.
  - (verified) A second config, `intentsqwen3-32b`, has filled descriptions for clinc150 (150), hwu64 (64), massive (58), minds14 (14) and snips (7). DeepPavlov/banking77 has only the plain `intents` config, so no generated descriptions.
  - Verbatim CLINC150 examples: `accept_reservations` -> "User is asking whether a specific restaurant accepts reservations."; `account_blocked` -> "User wants to know if their account is blocked and the reason for the block."; `alarm` -> "User wants to set an alarm."
  - The HF card has no license field. The data is reformatted from public sources for the AutoIntent library.
  - Sources: [DeepPavlov/clinc150](https://huggingface.co/datasets/DeepPavlov/clinc150); [datasets-server info](https://datasets-server.huggingface.co/info?dataset=DeepPavlov/clinc150); [DeepPavlov/hwu64](https://huggingface.co/datasets/DeepPavlov/hwu64)
- AutoIntent org also hosts LLM-augmented utterance sets: `banking77_aug`, `banking77_aug_qwen2.5-3b`, `banking77_aug_qwen2.5-7b-awq`, `hwu64_aug_qwen2.5-7b-awq`, `clinc150_aug_qwen2.5-7b-awq`. — [HF AutoIntent org](https://huggingface.co/AutoIntent)
- Hu, Khosmood and Edalat (ACL 2024 NLP4ConvAI workshop):
  - Hand-written descriptions for ATIS, SNIPS-NLU, CLINC and MASSIVE, following a fixed template: declarative, starting "User is asking" or "User wants", and keeping the label's tokens.
  - Verbatim: `abbreviation` -> "user is asking what an abbreviation stands for or means"; `AddToPlaylist` -> "user wants to add a song to a playlist".
  - Code and data are said to be public.
  - Sources: [arXiv 2407.17862](https://arxiv.org/html/2407.17862); [github.com/ruoyunlp/dataless-intent-classification](https://github.com/ruoyunlp/dataless-intent-classification)

### Inferences
- Contamination risk, highest first:
  - Redwood contains Banking77 (all 77) and CLINC150 (all 150) intents and utterances directly. It must be filtered by `source dataset` before training.
  - The DeepPavlov CLINC150 Qwen3 descriptions and the AutoIntent banking77/clinc150 augmentations are derived from the held-out sets. Use them only for evaluation formatting, or as a sanity baseline against the project's own descriptions, never for training.
  - Hu et al.'s CLINC descriptions are also tied to a held-out set.
  - HWU64 and MASSIVE/SLURP share many assistant intents with CLINC150 (alarm, weather, calendar, music and similar), so expect semantic label overlap but not shared utterances.
  - NLU++ banking, Bitext retail-banking, wealth-management and mortgage-loans, and MINDS-14 overlap Banking77's domain semantically.
- NLU++ descriptions are questions ("is the intent to …?"), and Hu et al. and DeepPavlov descriptions are declarative ("User wants …"). If you mix sources, normalise the description style so the model does not learn source-specific phrasing.
- For non-held-out training, the descriptions you can use without generating your own are NLU++ (62), SGD/SGD-X (88 × 6 variants) and the DeepPavlov Qwen3 descriptions for HWU64, MASSIVE, MINDS-14 and SNIPS (subject to their unclear license). Bitext verticals add about 330 intent names and would need generated descriptions.

### Gaps
- Licenses of the Bitext vertical datasets were not checked individually; only the main dataset (CDLA-Sharing-1.0) was.
- The license of the DeepPavlov re-uploads (no card field) and of Redwood (no LICENSE file) is unclear.
- The Hu et al. repo contents (file format, whether all four description sets are present) were not opened.
- MTOP (117 intents per [2607.27421](https://arxiv.org/html/2607.27421v1)), SLURP, ACID (insurance), HINT3 and ATIS licensing were not checked in detail. No official descriptions were found for any of them.
- No official description release was found for Banking77 or CLINC150 from their creators.

## Q3. Are there tool / function / API selection datasets where each option is a tool name plus description, usable as large-label-variety "intent with description" data? How many distinct tools?

### Takeaway
Yes, and they are the largest source of label variety:
- ToolRet: about 43k tools.
- ToolACE: 26,507 synthetic APIs.
- ToolBench: 16,464 RapidAPI APIs across 3,451 tools.
- xLAM/APIGen: 3,673 APIs.
- API-Bank: 2,138 training APIs.
- Gorilla APIBench: about 1,645 ML APIs.
- MetaTool: 199 tools.
- NexusRaven: 65 APIs.

MetaTool is the closest to the "short query -> pick one option by name + description" setup. The others mostly need you to pull out (query, gold tool, candidate-tool descriptions) triples.

### Cited Findings
**MetaTool / ToolE (2023, ICLR 2024 per its project materials; arXiv 2310.03128)**
- 21,127 queries: 20,630 single-tool and 497 multi-tool. Tool metadata comes from the OpenAI plugin store (`plugin_info.json`); descriptions are in `dataset/plugin_des.json`. — [HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool)
- (verified) `plugin_des.json` maps 199 tool names to one-sentence marketing-style descriptions. Verbatim: `airqualityforeast` -> "Planning something outdoors? Get the 2-day air quality forecast for any US zip code."; `copilot` -> "Searches every dealer, analyzes & ranks every car for you so you can buy with confidence." — [plugin_des.json](https://raw.githubusercontent.com/HowieHwong/MetaTool/master/dataset/plugin_des.json)
- License conflict: the GitHub repo is MIT, but the arXiv paper is listed under CC BY-NC-SA 4.0 (that license may apply only to the paper). — [GitHub](https://github.com/HowieHwong/MetaTool); [arXiv 2310.03128](https://arxiv.org/abs/2310.03128)

**ToolBench / ToolLLM (OpenBMB, 2023)**
- 16,464 REST APIs from RapidAPI, 3,451 tools, 49 categories, 126,486 instances. Tool-level fields: `tool_name`, `tool_description`. API-level fields: `name`, `description`, `method`, parameters. Has a retrieval split under `data/retrieval/`. Data and code: Apache-2.0. — [OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench)

**ToolRet (2025, arXiv 2503.01763)**
- Tool-retrieval benchmark with 7.6k retrieval tasks over a corpus of about 43k tools. IR-strong retrievers do poorly on it. — [arXiv 2503.01763](https://arxiv.org/abs/2503.01763v1)
- Split reported as 4,916 web-API tasks over 36,978 tools, plus 950 code-function tasks over 3,794 tools, plus a 205,826-instance training set. — [alphaXiv ToolRet summary](https://alphaxiv.org/benchmarks/leiden-university/toolret) (aggregator, not primary)
- (verified) The HF queries set aggregates many sources as configs, for example apibank 101, apigen 1,000, gorilla-huggingface 500, gorilla-pytorch 43, gorilla-tensor 55, autotools-weather 11. Fields: `id, query, instruction, labels, category`. No license on the card. — [HF mangopy/ToolRet-Queries](https://huggingface.co/datasets/mangopy/ToolRet-Queries)

**API-Bank (EMNLP 2023)**
- Evaluation: 73 API tools, 314 dialogues, 753 API calls. Training: 1,888 dialogues from 2,138 APIs spanning 1,000 domains. — [DAMO-ConvAI/api-bank README](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)
- HF copy license: MIT. — [HF liminghao1630/API-Bank](https://huggingface.co/datasets/liminghao1630/API-Bank)

**Gorilla APIBench (2023)**
- Usually cited as 1,645 APIs (HuggingFace 925, TorchHub 94, TensorFlow Hub 696). Sources conflict: the paper abstract says "∼1600 machine learning APIs", the repo says 1,640 API docs, and another source treats 1,645 as API calls with 94 unique TorchHub APIs. — [HF gorilla-llm/APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench); [arXiv 2305.15334](https://ar5iv.labs.arxiv.org/html/2305.15334); [arXiv 2402.09615 comparison table](https://arxiv.org/html/2402.09615v5)

**NexusRaven API evaluation (Nexusflow, 2023)**
- (verified) `standardized_api_list` has 65 APIs with `name`, `description` and `args_dicts`. `standardized_queries` has 318 queries. No license on the card. — [HF Nexusflow/NexusRaven_API_evaluation](https://huggingface.co/datasets/Nexusflow/NexusRaven_API_evaluation)

**When2Call (NVIDIA, NAACL 2025)**
- Test: 3,652 MCQ items plus a 300-item LLM-judge subset. Train: 15,000 SFT and 9,000 preference examples. Every item has a `tools` list of JSON specs (name, description, typed parameters), drawn from BFCL v2 Live. Answer types: direct, tool_call, request_for_info, cannot_answer. License: CC-BY-4.0. Example description: "Search for a bus itinerary between two cities on a specific date." — [HF nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call)

**Synthetic function-calling sets**
- xLAM function-calling-60k (Salesforce APIGen, NeurIPS 2024): 3,673 executable APIs across 21 categories (3,539 REST, 134 Python), 60k entries. HF dataset is gated (auto-approve), CC-BY-4.0, English. — [APIGen arXiv 2406.18518](https://arxiv.org/pdf/2406.18518.pdf); [HF card](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- ToolACE (ICLR 2025): an API pool of 26,507 synthetic APIs. One secondary source says they cover 30 domains, 390 subdomains and 3,398 classes. HF: Apache-2.0, English and Chinese. — [arXiv 2409.00920](https://arxiv.org/abs/2409.00920v2); [HF Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE)
- Glaive function-calling v2: Apache-2.0, 100K-1M rows, English. — [HF glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- Berkeley Function Calling Leaderboard data: Apache-2.0. — [HF gorilla-llm/BFCL](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)

### Inferences
- Tool descriptions differ in style from intent descriptions. MetaTool reads like marketing copy, ToolBench and RapidAPI text is often noisy or empty, and synthetic APIs (ToolACE, xLAM) are clean but templated. Queries are usually longer and more task-like than Banking77/CLINC150 utterances, so treat these sets as auxiliary label-variety data rather than in-style intent data.
- When2Call's example description "Search for a bus itinerary between two cities on a specific date" closely paraphrases SGD's `Buses_2.FindBus` "Find a bus itinerary between cities for a given date". BFCL Live functions may include SGD-like specs, so deduplicate across SGD and BFCL-derived sets before splitting. This is an inference from one example and was not verified.
- Banking77/CLINC150 overlap in tool sets looks low. RapidAPI has finance categories, but these are API-call queries, not customer-support utterances. Not checked quantitatively.

### Gaps
- Number of distinct tools in When2Call, Glaive v2 and BFCL was not determined.
- ToolBench data hosting and current download status were not verified (the README links external storage).
- MetaTool's venue was not confirmed on the arXiv page, which lists no venue.
- Newer MCP-era tool-selection sets (2025-2026 MCP benchmarks, ShortcutsBench, Seal-Tools, ToolAlpaca) were not researched in this pass.

## Q4. Which recent (2023-2026) papers or datasets do description-driven intent classification and release descriptions?

### Takeaway
The Google line (D3ST, SDT, SGD-X, AnyTOD/STARv2, 2022-2023) set up "describe labels, don't name them" for dialogue, but released data only for SGD-X and STARv2. For utterance-level intent classification, releases with descriptions are rare:
- NLU++ (2022): question-form descriptions.
- Hu et al. 2024: hand-written descriptions for ATIS, SNIPS, CLINC and MASSIVE.
- DeepPavlov AutoIntent (2025): Qwen3-32B descriptions for CLINC150, HWU64, MASSIVE, MINDS-14 and SNIPS.

Several 2025-2026 papers generate descriptions with LLMs and report gains, but releases are unconfirmed.

### Cited Findings
- D3ST, "Description-Driven Task-Oriented Dialog Modeling" (arXiv 2201.08904, Jan 2022): replaces developer-chosen slot and intent names with natural-language descriptions and uses an index-picking mechanism. Evaluated on MultiWOZ, SGD and SGD-X, with zero-shot transfer to unseen tasks. — [arXiv 2201.08904](https://arxiv.org/abs/2201.08904)
- Show Don't Tell (SDT; Gupta et al. 2022, arXiv 2204.04327) and D3ST were open-sourced in the google-research/task-oriented-dialogue repo (04/12/2022 entry). The STARv2 dataset was released 03/28/2023 and the AnyTOD dataset 09/16/2024. Repo is Apache-2.0 and now archived. — [google-research/task-oriented-dialogue](https://github.com/google-research/task-oriented-dialogue)
- AnyTOD / STARv2 (arXiv 2212.09939): STARv2 annotations came from D3ST trained on SGD, then finetuned on a few hand-labelled STAR conversations. — [AnyTOD arXiv 2212.09939](https://arxiv.org/pdf/2212.09939)
- "Exploring Zero and Few-shot Techniques for Intent Classification" (ServiceNow, ACL 2023 Industry): compares domain adaptation, data augmentation, zero-shot prompting of LLMs with intent descriptions, and parameter-efficient fine-tuning. T-few on Flan-T5 was best, even with one example per intent. — [arXiv 2305.07157](https://arxiv.org/abs/2305.07157)
- Hu et al., "Exploring Description-Augmented Dataless Intent Classification" (NLP4ConvAI @ ACL 2024):
  - BGE-Large scored 71.53 overall with descriptions vs 62.91 with tokenized label names, about +11 points average per model. Paraphrasing plus masking raised it to 75.22.
  - Descriptions also cut variance across models (σ 1.98 vs 5.65).
  - ATIS showed no gain (60.51 vs 60.55).
  - Repo: ruoyunlp/dataless-intent-classification.
  - Source: [arXiv 2407.17862](https://arxiv.org/html/2407.17862)
- AutoIntent (DeepPavlov, EMNLP 2025, arXiv 2509.21138): AutoML for intent classification (embedding selection, scoring, threshold tuning, multi-label and OOS support), evaluated on banking77, massive, minds14, hwu64 and snips. — [arXiv 2509.21138](https://arxiv.org/pdf/2509.21138)
  - (verified) Its HF datasets carry Qwen3-32B descriptions, as described in Q2. — [DeepPavlov/clinc150](https://huggingface.co/datasets/DeepPavlov/clinc150)
- arXiv 2501.12332 ("Automatic Labelling with Open-source LLMs using Dynamic Label Schema Integration", 2025): Mistral-7B generated label descriptions from a few examples. On zero-shot Banking77, names alone scored 81.6 and names plus descriptions 88.8 without CoT; with CoT, 79.2 vs 83.7. Search snippet only, not fully read. — [arXiv 2501.12332](https://arxiv.org/pdf/2501.12332)
- EMNLP 2025 Findings paper ("On Collaborating Small and Large Models for Few-shot…"): an LLM first generates "discriminative intent descriptions" for all intents, evaluated on CLINC150 and Banking77. Search snippet only. — [ACL Anthology 2025.findings-emnlp.749](https://aclanthology.org/2025.findings-emnlp.749.pdf)
- arXiv 2610.16182 ("Label-Centric Knowledge Retrieval for Zero-Shot Text Classification", 2026): builds a label-knowledge store from only the label inventory and task description; at inference the LLM sees retrieved labels' names and definitions plus boundary rules for confusable labels. Evaluated on BANKING77 and CLINC150. Search snippet only. — [arXiv 2610.16182](https://arxiv.org/pdf/2610.16182)
- Counter-example: "Selecting Open-Weight Language Models for Zero-Shot Intent Classification" (Auburn, July 2026) evaluates 41 models on CLINC150, Banking77, SNIPS, MASSIVE, MTOP (117), Curekart (20), Powerplay11 (53) and Sofmattress (20) using label names only, and lists label definitions as future work. — [arXiv 2607.27421](https://arxiv.org/html/2607.27421v1)
- "Dynamic Label Name Refinement for Few-Shot Dialogue Intent Classification" (Dec 2024) rewrites label names rather than adding descriptions. Reported +2.07% on BANKING77 for the dynamic variant and −4.23% for static refinement. — [arXiv 2412.15603](https://arxiv.org/html/2412.15603v1)

### Inferences
- In the papers that test it, descriptions consistently help embedding and zero-shot LLM intent classification: about +11 points for bi-encoders (Hu et al.) and about +7 points zero-shot on Banking77 (2501.12332). The exceptions are single-domain, imbalanced sets like ATIS, and static name rewriting. This supports the project's "name: description" setup.
- Banking77 and CLINC150 are the default evaluation sets in this literature, so many released descriptions target exactly the held-out sets. If you compare to published numbers, say whether descriptions were hand-written, LLM-generated, or taken from these releases.

### Gaps
- Whether 2501.12332, the EMNLP 2025 Findings paper and 2610.16182 release their generated descriptions was not verified (only snippets read).
- The DeepPavlov Qwen3 descriptions' generation prompt and any human review are undocumented on the cards checked.
- No 2025-2026 dataset was found that ships human-written descriptions for Banking77.

## Q5. Which sources give the most distinct labels in the "short user message -> fine-grained intent" style?

### Takeaway
For in-style short utterances with many fine-grained labels:
- Redwood: 451 intents, but it contains Banking77 and CLINC150, so filter it. With those removed, about 224 labels remain from ACID, MTOP, HWU, MetaLWOz, DSTC8 and new collections.
- Bitext family: about 330 intent names over 12 datasets, customer-support style, no descriptions.
- MTOP: 117 intents.
- SGD/SGD-X: 88 intents, with descriptions.
- HWU64: 64. NLU++: 62, with descriptions. MASSIVE: 60, multilingual.

Tool datasets have far more labels (thousands to tens of thousands) but are less in-style.

### Cited Findings
- Redwood: 451 intents from 13 datasets. Of these, 227 come directly from banking77 (77) and clinc150 (150); the rest are acid 112, mtop 59, hwu 53, new 52, metalwoz 41, dstc8 33, outlier 10, snips 7, pto 6, and atis, jobs640 and talk2car with 1 each (verified on the test file). — [redwood_test.json](https://raw.githubusercontent.com/gxlarson/redwood/main/data/redwood_test.json); [ACL Anthology](https://aclanthology.org/2022.sigdial-1.45)
- Bitext: the main dataset has 27 intents, and the 11 vertical datasets have 24-46 intents each (about 330 intent names in total, counted with overlap across verticals), all customer-support style with 23k-45k rows each. — [Bitext HF datasets](https://huggingface.co/bitext); [stats endpoint example](https://datasets-server.huggingface.co/statistics?dataset=bitext/Bitext-customer-support-llm-chatbot-training-dataset&config=default&split=train)
- MTOP has 117 labels and Powerplay11 has 53. — [arXiv 2607.27421](https://arxiv.org/html/2607.27421v1)
- SGD: 88 service-intent pairs, 46 unique intent names (verified). — [SGD schema files](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
- NLU++: 62 multi-label intents with question descriptions. — [NLU++ ontology.json](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/nlupp/data/ontology.json)
- ABCD: 55 customer-service subflows with guideline text. — [ABCD README](https://github.com/asappresearch/abcd)
- MASSIVE: 60 intents, 52 languages. — [alexa/massive](https://github.com/alexa/massive)
- HWU64: 64 intents, CC BY 4.0. — [NLU-Evaluation-Data](https://github.com/xliuhw/NLU-Evaluation-Data)
- Tool-label scale: ToolRet about 43k tools, ToolACE 26,507 APIs, ToolBench 16,464 APIs / 3,451 tools, xLAM 3,673 APIs, API-Bank 2,138 training APIs, APIBench about 1,645, MetaTool 199. — [ToolRet](https://arxiv.org/abs/2503.01763v1); [ToolACE](https://arxiv.org/abs/2409.00920v2); [ToolBench](https://github.com/OpenBMB/ToolBench); [APIGen](https://arxiv.org/pdf/2406.18518.pdf); [API-Bank](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank); [APIBench](https://huggingface.co/datasets/gorilla-llm/APIBench); [MetaTool](https://github.com/HowieHwong/MetaTool)

### Inferences
- A non-overlapping, in-style training pool could combine:
  - Redwood minus the banking77 and clinc150 sources (about 224 labels)
  - Bitext verticals (about 330, after deduplicating shared intents like `contact_human_agent`)
  - NLU++ (62, with descriptions)
  - SGD/SGD-X (88 × 6 description variants)
  - MASSIVE/HWU64 (if not already covered by Redwood's hwu subset)
  
  That gives several hundred to about 700 distinct fine-grained labels before adding tool data. Most would need generated descriptions; only NLU++ and SGD/SGD-X have official ones. This is an estimate: cross-dataset deduplication was not computed.
- Redwood's `hwu`, `mtop` and `snips` subsets overlap the standalone HWU64, MTOP and SNIPS datasets, so do not double count.
- Use tool-selection data (MetaTool, ToolBench, ToolRet) to stress-test large candidate sets (hundreds to thousands of "name: description" options), not as the main in-domain training signal.

### Gaps
- Exact deduplicated label counts across the combined pool were not computed.
- ACID (insurance, 175 intents in the original paper per common citation) was not checked directly. Redwood keeps 112 ACID intents.
- MTOP and SLURP license and size details were not verified from primary sources this session.
