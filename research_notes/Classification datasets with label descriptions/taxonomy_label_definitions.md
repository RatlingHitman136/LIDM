# Taxonomy-labelled text classification datasets with per-label definitions (for "name: description" zero-shot)

Scope note for the report writer: several numbers below were **measured directly by me on 2026-11-18** against official endpoints or files (EuroVoc SPARQL, MeSH SPARQL, DBpedia SPARQL, the IPTC CV JSON, the Google taxonomy file, the O*NET 30.3 text file, the ESCO API, the NLM Clinical Tables API, the MultiEURLEX GitHub label files). These are marked "(measured)" and cite the endpoint or file used. Everything else cites the page it came from.

## 1. Legal / government: EUR-Lex / MultiEURLEX with EuroVoc; other legal taxonomies

### Takeaway
MultiEURLEX (65k EU laws, 23 languages, CC BY 4.0) is the standard EuroVoc dataset, but EuroVoc descriptions are sparse. Only about 23% of EuroVoc concepts have an English definition or scope note. MultiEURLEX's level-1 (21) and level-2 (127) labels have **none**, and only 25% of its level-3 labels (140 of 567) have one. The dataset itself ships names only, in 23 languages. So "name: description" prompts for EuroVoc need extra text (hierarchy paths, synonyms or generated glosses) for most labels.

### Cited Findings
- MultiEURLEX: "65k EU laws in 23 official EU languages". Splits: train 55k (1958-2010), dev 5k (2010-2012), test 5k (2012-2016). EN/DE/FR/IT/NL have the full 55k train; Croatian has 7,944 train docs. — [HF card coastalcph/multi_eurlex](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- Multi-label: "a multi-label classification task" where each document gets "one or more concepts". The original assignments are at EuroVoc levels 3-8, and the authors provide ancestor-label sets at levels 1, 2 and 3. — [HF card](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- License: "We provide MultiEURLEX with the same licensing as the original EU data (CC-BY-4.0)." — [HF card](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- Label text shipped with the dataset is a name only, e.g. "LABEL: id: 1, eurovoc_id: '100160', eurovoc_desc: 'industry'". Descriptors are "also available in the 23 languages". — [HF card](https://huggingface.co/datasets/coastalcph/multi_eurlex)
- (measured) MultiEURLEX label sets per level, from `eurovoc_concepts.json`: level_1 = 21, level_2 = 127, level_3 = 567, level_4 = 3,861, level_5 = 2,284, level_6 = 481, level_7 = 43, level_8 = 6, original = 7,390 distinct concepts. `eurovoc_descriptors.json` holds 7,390 entries with names in 23 languages, e.g. `'100149': {'en': 'social questions', 'de': 'soziale fragen', ...}`. — [nlpaueb/multi-eurlex eurovoc_concepts.json](https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_concepts.json), [eurovoc_descriptors.json](https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_descriptors.json)
- (measured) Current EuroVoc (scheme `http://eurovoc.europa.eu/100141`) has 7,515 concepts. In English, 1,555 concepts have a `skos:definition` (mean 130 chars) and 325 have a `skos:scopeNote` (mean 100 chars). 1,735 concepts (~23%) have at least one of the two. Across all languages: 1,614 concepts with a definition, 894 with a scope note, 240 with a history note. — [EU Publications Office SPARQL endpoint](https://publications.europa.eu/webapi/rdf/sparql)
- (measured) The 7,515 total combines 551 top terms and 128 microthesauri. Only 146 of the 551 top terms have an English definition or scope note. — [EU Publications Office SPARQL endpoint](https://publications.europa.eu/webapi/rdf/sparql)
- (measured) Description coverage of **MultiEURLEX label sets** (English def or SN): level_1 0/21 (0%), level_2 0/127 (0%), level_3 140/567 (25%), level_4 768/3,861 (20%), level_5 444/2,284 (19%), level_6 104/481 (22%), original 1,468/7,390 (20%). — computed by joining [eurovoc_concepts.json](https://raw.githubusercontent.com/nlpaueb/multi-eurlex/master/data/eurovoc_concepts.json) with the [EuroVoc SPARQL endpoint](https://publications.europa.eu/webapi/rdf/sparql)
- (measured) Verbatim EuroVoc **definitions**: "co-insurance: Joint coverage of a large risk by a number of insurers under a single contract, each insurer being liable for a portion of any claims." "teleworking: Work carried out at a distance using telecommunication services or computers." — [EuroVoc SPARQL](https://publications.europa.eu/webapi/rdf/sparql)
- (measured) Verbatim EuroVoc **scope notes**. These are often indexing instructions, not glosses: "automatic vending machine: Use for the sale of products from machines. For "automatic teller machine", use the descriptor "electronic banking"." "writing skills: Use for the drafting of administrative documents." — [EuroVoc SPARQL](https://publications.europa.eu/webapi/rdf/sparql)
- EuroVoc is distributed as XML, SKOS-Core, SKOS-AP-EU (RDF), Excel and MarcXML. — [EuroVoc 4.8 release notes (via search)](https://op.europa.eu/documents/3938058/3957781/Release+notes+EuroVoc+4.8/7ca6a146-120f-87b7-052f-77ee7ad37429). It has ~7,000+ concepts in 21 domains and 127 microthesauri. — [Dublin Core MSI EuroVoc entry](https://msi.dublincore.org/standards/eurovoc)
- Historical counts: EuroVoc 4.8 had 7,180 concepts (7,090 active); 4.10 had 7,243. — [EuroVoc 4.8 release notes](https://op.europa.eu/documents/3938058/3957781/Release+notes+EuroVoc+4.8/7ca6a146-120f-87b7-052f-77ee7ad37429), [4.10 release notes](https://op.europa.eu/documents/3938058/5366531/Release+notes+EuroVoc+4.10/0eda84f8-4f2c-1251-272e-f9b35e6bcfb0)
- EURLex-4.3K (Extreme Classification Repository): 4,271 labels, 45,000 train / 6,000 test docs, 60.57 avg points per label, 5.07 avg labels per point, BoW dim 200,000. It is listed as a BoW / raw-text dataset, not an "LF" (label-feature) dataset. — [Extreme Classification Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html)

### Inferences
- Joining is low effort: MultiEURLEX labels are EuroVoc IDs that map 1:1 to `http://eurovoc.europa.eu/{id}`, so a single SPARQL query or SKOS dump join gives names, definitions, scope notes, alt labels and broader paths in 23 languages.
- The output is poor for "name: description". At the commonly used levels 1-3 almost no label has a definition. Some scope notes are indexing rules ("Use for ...") that would confuse a zero-shot model. The best available "description" for most EuroVoc labels is the hierarchy path (domain > microthesaurus > top term) plus `altLabel` synonyms.
- MultiEURLEX documents are long (full laws), so this suits long-document zero-shot. The 21/127/567-label levels give a natural difficulty ladder.
- EURLex-4.3K's label names are EuroVoc descriptors (it shares the EuroVoc label space), so the same ~20% coverage likely applies. Not verified per-label.

### Gaps
- I did not research other legal taxonomies (e.g. LexGLUE's ECtHR articles, US legal topic schemes, SCOTUS issue areas). Not covered.
- I could not confirm that the per-language definitions for non-English languages have the same coverage. I measured English only, though the all-language concept counts above are similar.
- No official EuroVoc-specific license page was fetched. CC BY 4.0 is inferred from MultiEURLEX's statement and general EU reuse policy.

## 2. Patents: CPC / IPC labelled datasets and official definitions

### Takeaway
Patent datasets (HUPD, BIGPATENT) give CPC/IPC codes only, never text. Official CPC titles (every symbol) and CPC definitions (subclasses and groups only, with long structured sections) must be joined in from EPO/USPTO bulk files. BIGPATENT has only 9 section-level labels. HUPD carries full CPC/IPC symbol lists, but its license is ambiguous (CC BY-SA vs CC BY-NC-SA).

### Cited Findings
- HUPD: "utility patent applications filed to the United States Patent and Trademark Office (USPTO) between January 2004 and December 2018". 34 data fields, including `main_cpc_label`, `cpc_labels`, `main_ipcr_label`, `ipcr_labels`, `uspc_class` and text fields `title`, `abstract`, `claims`, `background`, `summary`, `full_description`. The card does not state label granularity or include CPC definitions. — [HF card HUPD/hupd](https://huggingface.co/datasets/HUPD/hupd)
- HUPD **license conflict**: the metadata tag says `cc-by-sa-4.0`, but the Licensing section says "CreativeCommons Attribution-NonCommercial-ShareAlike 4.0 International". — [HF card HUPD/hupd](https://huggingface.co/datasets/HUPD/hupd)
- BIGPATENT: train 1,207,222 / validation 67,068 / test 67,072. Subsets are the CPC sections a-h and y ("nine CPC classification categories"). Fields are `description` and `abstract`. License `cc-by-4.0`. — [HF card NortheasternUniversity/big_patent](https://huggingface.co/datasets/NortheasternUniversity/big_patent)
- CPC scheme versions are released periodically; "The August version (2026.08) of the CPC scheme is now in force". Valid symbol lists come as ZIP/CSV, and definitions are released in PDF and XML. — [CPC Scheme and Definitions](https://www.cooperativepatentclassification.org/cpcSchemeAndDefinitions)
- CPC definitions cover subclasses, main groups and subgroups, but not sections or classes. Indexing-only subclasses (e.g. C10N, Y) are excluded. Definition sections: definition statement, application-oriented / residual / informative references. — [EPO CPC Essentials (via search)](https://link.epo.org/cpc/CPC_Essentials_I_PartC.pdf), [MPEP 905](https://www.uspto.gov/web/offices/pac/mpep/s905.html)
- The MPEP says a definition statement should elaborate the classification place rather than restate its title. — [MPEP s905](https://www.uspto.gov/web/offices/pac/mpep/s905.html)
- Verbatim CPC example (G06N). Title: "COMPUTING ARRANGEMENTS BASED ON SPECIFIC COMPUTATIONAL MODELS". Definition statement: "Computing systems where the computation is not based on a traditional mathematical model of computer". Main group G06N 3/00 title: "Computing arrangements based on biological models". Definition begins "Computing systems where the computation is based on biological models" and continues "The computation can be digital, analogue or chemical in nature." Page sections: Definition statement; Relationships with other classification places; References (Informative / Application-oriented / Limiting); Glossary of terms; Synonyms and Keywords; Special rules of classification. The page is roughly 6-8k words for one subclass. — [USPTO CPC definition G06N](https://www.uspto.gov/web/patents/classification/cpc/html/defG06N.html)
- The XML definition files are named per subclass (e.g. A01B) and listed on the CPC Bulk page. — [jvd10/cpcprocessing (via search)](https://github.com/jvd10/cpcprocessing)

### Inferences
- Joining HUPD to CPC text takes medium effort. Normalise symbols (e.g. `G06N3/08` vs `G06N 3/08`) and join to the CPC title list for every symbol. Definitions attach only at subclass/group level and are long, multi-section documents. For "name: description", the best fit is probably the subclass title plus the definition statement, truncated, rather than the whole definition.
- CPC titles of deep subgroups are often fragments that only make sense with their ancestors' titles (the dot hierarchy). Concatenating the ancestor titles is the usual fix (inference, not verified here).
- Patents are long documents. Abstracts (~150 words) give a "short-ish" variant.
- If HUPD's CC BY-NC-SA reading is correct, commercial use is barred. Flag this.
- CPC/IPC scheme text is published by EPO/USPTO/WIPO for free use. A formal license statement was not found (see gaps).

### Gaps
- Exact CPC symbol counts (sections/classes/subclasses/groups) for 2026.08: not found on the pages I read.
- Formal terms of use for CPC and IPC bulk definitions: not found.
- HUPD total document count: not stated on the HF card (the paper likely states it; not fetched).
- USPTO-2M and other IPC-labelled datasets (e.g. WIPO-alpha, CLEF-IP): not researched.
- IPC (WIPO) definitions: not researched separately.

## 3. Medical / science: ICD (MIMIC), MeSH (PubMed/BioASQ), arXiv categories, ACM CCS

### Takeaway
MeSH is the standout. It has 30,594 current descriptors, 99.4% of them with a real scope note (mean ~198 chars), and a large public indexing corpus (BioASQ/PubMed, ~15.6M articles, ~29k labels). ICD-10-CM has ~74.7k codes, but its "descriptions" are short titles, and the main coding corpus (MIMIC-IV) is credentialed. arXiv's 155 categories have short descriptions that are sometimes missing. ACM CCS appears to have labels only (unconfirmed).

### Cited Findings
- (measured) MeSH current-year graph: 30,594 `meshv:TopicalDescriptor`s. 30,412 (99.4%) have a scope note on their preferred concept, mean 197.9 chars. — [NLM MeSH SPARQL endpoint](https://id.nlm.nih.gov/mesh/sparql)
- (measured) Verbatim MeSH scope notes: "Neoplasms: New abnormal growth of tissue. Malignant neoplasms show a greater degree of anaplasia and have the properties of invasion and metastasis, compared to benign neoplasms." "Diabetes Mellitus: A heterogeneous group of disorders characterized by HYPERGLYCEMIA and GLUCOSE INTOLERANCE." Note the cross-referenced descriptors in CAPS. — [NLM MeSH SPARQL](https://id.nlm.nih.gov/mesh/sparql)
- BioASQ Task 9a (2021) training set: 15,559,157 articles, 12.68 labels per article on average, 29,369 distinct MeSH labels. The task is classifying PubMed/MEDLINE articles into MeSH concepts. — [BioASQ 2021 overview](https://arxiv.org/pdf/2106.14885)
- MIMIC-IV v3.1 (Oct 11, 2024): "Only credentialed users who sign the DUA can access the files". Requires CITI "Data or Specimens Only Research" training. License: "PhysioNet Credentialed Health Data License 1.5.0". The hosp module includes `diagnoses_icd`, `d_icd_diagnoses`, `procedures_icd`, `d_icd_procedures`, and "All ICD codes in the diagnoses_icd table have an associated reference in d_icd_diagnoses". "Free-text clinical notes are available in MIMIC-IV-Note". — [PhysioNet MIMIC-IV](https://physionet.org/content/mimiciv/)
- MIMIC-IV-ICD benchmark: 331,669 admissions / 145,855 patients overall. The ICD-10 subset has 122,317 documents and 26,096 unique codes, with 16.1 codes per document and 1,662 words per document on average. The ICD-9 set has 11,311 codes in the text vs 11,331 in Table 1 (internal inconsistency). — [Nguyen et al., arXiv 2304.13998](https://arxiv.org/pdf/2304.13998)
- ICD-10-CM FY2026: CMS hosts "Code Descriptions in Tabular Order" files (FY2026 files cover discharges Oct 1 2025 - Mar 31 2026, with an April 1 2026 update). — [CMS ICD-10 page](https://www.cms.gov/medicare/coding-billing/icd-10-codes). A third party reports FY2026 adds 487 codes, for 74,719 codes in the tabular list (unverified against CMS). — [icdlist.com](https://icdlist.com/resources/icd-10-cm-updates)
- (measured) Verbatim ICD-10-CM descriptions (titles only): "E11.9 Type 2 diabetes mellitus without complications"; "I21.4 Non-ST elevation (NSTEMI) myocardial infarction". — [NLM Clinical Tables ICD-10-CM API](https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search?sf=code&terms=E11.9)
- arXiv taxonomy lists 155 categories (including aliases such as cs.NA/math.NA, cs.SY/eess.SY): cs 40, math 32, physics 51 across archives, q-bio 10, q-fin 9, stat 6, eess 4, econ 3. — [arXiv category taxonomy](https://arxiv.org/category_taxonomy)
- arXiv verbatim: "cs.CL (Computation and Language): Covers natural language processing." "math.AG (Algebraic Geometry): Algebraic varieties, stacks, sheaves, schemes, moduli spaces, complex geometry, quantum cohomology". Most descriptions are one line or a comma list. A few (cs.AI, eess.AS, eess.IV) are paragraphs. Some read "Description coming soon" (physics.gen-ph, physics.pop-ph, quant-ph) or are empty. — [arXiv category taxonomy](https://arxiv.org/category_taxonomy)
- ACM CCS 2012 "was developed as a semantic ontology and is available in SKOS format". The full tree is "freely available for educational and research purposes". — [ACM CCS 2012](https://www.acm.org/publications/class-2012-intro). A third-party catalog lists the concept count as "not specified" with a "hybrid open-access" license. — [PoolParty listing](https://www.poolparty.biz/knowledge-models-technology/)

### Inferences
- MeSH plus PubMed/BioASQ is the single largest source of distinct labels with real, definitional descriptions (~30k at ~200 chars). Joining is easy: BioASQ labels are MeSH descriptor names or UIs, and the MeSH RDF/XML is downloadable. Abstracts are medium-length texts. The CAPS cross-references in scope notes should be lowercased or kept as hints.
- ICD-10-CM descriptions are titles (~5-10 words), not definitions. In "name: description" format the name and the description would almost duplicate each other. Richer text would need the tabular list's inclusion/exclusion notes (not checked) or ICD-11 definitions (not researched).
- MIMIC is unusable for open redistribution. Any benchmark derived from it inherits credentialing. MIMIC's own `d_icd_*` tables give the code titles, so the join is trivial once credentialed.
- arXiv suits a small, real-description benchmark (155 labels, ~10-30 words), but needs manual cleanup of empty or placeholder descriptions and aliases.

### Gaps
- MeSH license terms (NLM Terms and Conditions) and BioASQ's current registration/data terms: not fetched.
- arXiv-labelled datasets (e.g. arXiv metadata dumps, `ccdv/arxiv-classification`) and their licenses: not researched.
- ACM CCS concept count and whether any concepts carry scope notes: not confirmed. I believe it is labels only, but did not verify.
- ICD-10-CM's official license or terms: not fetched. It is a US-government (NCHS/CMS) publication; that it is public domain is an assumption.
- Whether the ICD-10-CM tabular XML's "includes/excludes/inclusion terms" offer usable descriptions: not checked.

## 4. Commerce / industry / jobs: Google/Amazon product taxonomy, NAICS/SIC, O*NET, ESCO

### Takeaway
For jobs, ESCO (3,039 occupations + 13,939 skills, descriptions in 28 languages, free reuse with attribution) and O*NET (1,016 occupations, all described, mean 214 chars) have rich official descriptions. TechWolf provides public ESCO-labelled job-ad datasets on Hugging Face. Google's product taxonomy has 5,595 categories with **no descriptions**. NAICS has official descriptions for 1,012 six-digit industries, but I found no good public NAICS-labelled text corpus.

### Cited Findings
- (measured) O*NET 30.3 `Occupation Data.txt`: 1,016 rows (O*NET-SOC Code, Title, Description). Description mean 214 chars, median 185. Verbatim: "15-2051.00 Data Scientists: Develop and implement a set of techniques or analytics applications to transform raw data into meaningful information using data-oriented programming languages and visualization software. Apply data mining, data modeling, natural language processing, and machine learning to extract and analyze information from large structured and unstructured datasets. ..." — [O*NET 30.3 Occupation Data.txt](https://www.onetcenter.org/dl_files/database/db_30_3_text/Occupation%20Data.txt)
- O*NET-SOC has 1,016 occupational titles, of which 923 are data-level occupations. — [O*NET 25.1 database page (archived)](https://webarchive.library.unt.edu/web/20201219020057/https://www.onetcenter.org/database.html). All versions are "available under a Creative Commons license". — [O*NET Database](https://www.onetcenter.org/database.html)
- ESCO v1.2: 3,039 occupations and 13,939 skills, in 28 languages (EU languages plus Icelandic, Norwegian, Ukrainian, Arabic). — [ESCO site (via search)](https://esco.ec.europa.eu/en/about-esco/escopedia/escopedia/esco-v1), [v1.2 launch deck](https://esco.ec.europa.eu/system/files/2024-05/V1.2%20Launch%20-%20ESCO%20SEC%20combined%20-%20final%20clean.pdf). Some sources give 13,890 skills (older). — [Dublin Core MSI ESCO](https://msi.dublincore.org/standards/esco). The latest release found is ESCO 1.2.1 (Dec 2025). — [ESCO downloadable datasets](https://esco.ec.europa.eu/en/structure-esco-downloadable-datasets)
- ESCO occupations map to one ISCO-08 code each. The skills pillar has 4 sub-classifications. Files such as `skills_en` and `occupationSkillRelations` are joined on concept URIs. — [ESCO downloadable datasets structure](https://esco.ec.europa.eu/en/structure-esco-downloadable-datasets)
- (measured) Verbatim ESCO occupation description: "data scientist: Data scientists find and interpret rich data sources, manage large amounts of data, merge data sources, ensure consistency of data-sets, and create visualisations to aid in understanding data. They build mathematical models using data, ..." The preferred label exists in 28 languages. — [ESCO API](https://ec.europa.eu/esco/api/resource/occupation?uri=http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30&language=en)
- ESCO license: under Decision 2011/833/EU, "the ESCO classification can be downloaded, used, reproduced and reused for any purpose and by any interested party free of charge". Attribution is requested, and modified versions must be marked. — [ESCO FAQ](https://esco.ec.europa.eu/et/about-esco/faq)
- ESCO-labelled public datasets (TechWolf on HF): `skill-extraction-techwolf`, `skill-extraction-tech` (extends the SkillSpan TECH subset, ESCO v1.1.0 skill labels), and `skill-extraction-house`, all job-ad sentences/spans annotated with ESCO skills. `Synthetic-ESCO-skill-sentences` has ~138K (skill, job-ad sentence) pairs covering 99.5% of ESCO skills. `JobBERT-evaluation-dataset` has vacancy titles each tagged with an ESCO v1.0.5 occupation. — [TechWolf/skill-extraction-techwolf](https://huggingface.co/datasets/TechWolf/skill-extraction-techwolf), [TechWolf/Synthetic-ESCO-skill-sentences](https://huggingface.co/datasets/TechWolf/Synthetic-ESCO-skill-sentences), [JobBERT-evaluation-dataset README](https://huggingface.co/datasets/TechWolf/JobBERT-evaluation-dataset/blob/main/README.md)
- (measured) Google Product Taxonomy version 2021-09-21: 5,595 categories, names only (`ID - Path > Path`). Depth distribution: L1 21, L2 192, L3 1,349, L4 2,203, L5 1,385, L6 397, L7 48. — [Google taxonomy-with-ids.en-US.txt](https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt)
- NAICS 2022: 1,012 six-digit national industries. Counts at higher levels differ between sources: 96 subsectors / 308 industry groups vs 99 / 312. — [classification.codes](https://classification.codes/classifications/industry/naics-usa), [Formulabot](https://www.formulabot.com/datasets/naics-industry-classification-codes). BLS provides NAICS 2022 "codes, titles, descriptions, and index items" via the QCEW Industry Finder (1,030 industries incl. BLS-specific codes). — [BLS NAICS 2022](https://www.bls.gov/cew/classifications/industry/naics-2022.htm)
- The HF dataset `Yvnminc/ExioNAICS` claims company-description + NAICS 2022 code fields, but its viewer errored and contents are unconfirmed. — [HF ExioNAICS](https://huggingface.co/datasets/Yvnminc/ExioNAICS)

### Inferences
- Jobs give the strongest combination: official multi-sentence descriptions (ESCO ~3k occupations and ~14k skills; O*NET ~1k) and public short-text labelled data (job titles, job-ad sentences). Joining TechWolf sets to ESCO descriptions is easy via label string or URI, but version drift (v1.0.5 / v1.1.0 labels vs v1.2.x descriptions) needs a mapping or exact-label matching.
- ESCO skills (~14k labels, descriptions in 28 languages) with sentence-level job-ad text is a good short-text, many-label, multilingual-description benchmark.
- Product categorisation with Google's taxonomy offers only hierarchical paths, no descriptions. The path ("Home & Garden > Kitchen & Dining > ...") is the only "description".
- NAICS has good official descriptions but no clearly licensed public corpus of labelled texts, so building one would take high effort.

### Gaps
- Amazon product taxonomy: no official public taxonomy with descriptions found (not deeply searched).
- SIC codes: not researched.
- Exact O*NET license version (CC BY 4.0 is likely) not confirmed from the license page.
- ESCO skill-description verbatim example and length statistics: not measured (only the occupation example).
- Census NAICS 2022 descriptions file: not fetched directly; description length unknown.

## 5. News / topics: IPTC Media Topics; DBpedia / Wikidata class descriptions

### Takeaway
IPTC Media Topics is a clean fit. Every one of its 1,059 active concepts has an English definition (mean 99 chars), it is CC BY 4.0, depth reaches 6, and MN-DS (10,917 articles, CC BY 4.0) is labelled with it. DBpedia ontology descriptions are patchy: only 282 of 790 classes have an English comment, and only 4 of the 14 DBpedia-14 classes. Wikidata gives short one-line glosses for nearly any item.

### Cited Findings
- (measured) IPTC Media Topics CV JSON (dateReleased 2026-09-30): 1,394 concepts, 1,059 active (non-retired). All 1,059 have an en-GB definition, mean 99 chars, median 90, max 270. There are 17 top-level topics, and depth distribution is L1 17, L2 120, L3 504, L4 364, L5 49, L6 5. The scheme carries `licenceLink: http://creativecommons.org/licenses/by/4.0/`. — [IPTC CV server mediatopic JSON](https://cv.iptc.org/newscodes/mediatopic/?lang=en-GB&format=json)
- (measured) Verbatim IPTC definitions: "environment: The protection, damage, and condition of the ecosystem of the planet Earth and its surroundings". "information technology and computer science: The study and design of computer systems, software and networks". Some top-level definitions are near-tautological: "arts, culture, entertainment and media: All forms of arts, entertainment, cultural heritage and media". — [IPTC CV JSON](https://cv.iptc.org/newscodes/mediatopic/?lang=en-GB&format=json)
- IPTC updates Media Topics roughly every 2-3 months. E.g. Q1 2025 added 8, retired 17, and modified 43 labels and 64 definitions. — [IPTC NewsCodes 2025 Q1 release](https://iptc.org/news/iptc-newscodes-2025-q1-release/). Translations exist in about 12 languages per Wikipedia (9 per an older IPTC post). — [Wikipedia IPTC](https://en.wikipedia.org/wiki/International_Press_Telecommunications_Council)
- MN-DS: 10,917 news articles from 2019 (from NELA-GT-2019), labelled with the 2020 IPTC Media Topic taxonomy using 17 first-level and 109 second-level categories, ~100 hand-picked articles per second-level category, CC BY 4.0. — [MN-DS, arXiv 2212.12061](https://arxiv.org/pdf/2212.12061), [MDPI Data 8(5):74](https://mdpi.com/2306-5729/8/5/74)
- (measured) DBpedia ontology (live endpoint): 790 `owl:Class`es in `dbpedia.org/ontology/`. 282 have an English `rdfs:comment`, mean 97 chars. Of the 14 DBpedia-14 classes, only Village, WrittenWork, NaturalPlace and Building have English comments. Verbatim: "Village: a clustered human settlement or community, usually smaller a town" [sic]. "WrittenWork: Written work is any text written to read it (e.g.: books, newspaper, articles)". — [DBpedia SPARQL](https://dbpedia.org/sparql)
- DBPedia Classes (Kaggle/HF): 342,782 Wikipedia articles with 3-level labels (9 / 70 / 219 classes). Columns: text, l1, l2, l3, wiki_name. CC0. Example: Event > NaturalEvent > Earthquake. — [HF DeveloperOats/DBPedia_Classes README (mirror)](https://hg.istephan-squireso.workers.dev/datasets/DeveloperOats/DBPedia_Classes/blob/main/README.md), [Kaggle mirror](https://baselight.app/u/kaggle/dataset/danofer_dbpedia_classes). The DBpedia ontology is a DAG since 3.7, so a tree must be derived. — [DBpedia ontology page](https://www.dbpedia.org/resources/ontology/)
- (measured) Wikidata descriptions are one-line glosses: "Q5 human: any single member of Homo sapiens, unique extant species of the genus Homo"; "Q4830453 business: organization undertaking commercial, industrial, or professional activity"; "Q11424 film: visual art work that simulates experiences and communicates ideas, stories, or emotions through moving images, ...". — [Wikidata API](https://www.wikidata.org/w/api.php?action=wbgetentities&ids=Q5|Q4830453|Q11424&props=labels|descriptions&languages=en&format=json)

### Inferences
- IPTC + MN-DS is the cleanest small news benchmark: full definition coverage, an open license, and a hierarchy. Joining takes some work because MN-DS uses 2020 labels and IPTC has since retired or renamed concepts. Match on qcode if MN-DS stores it, otherwise on name, and keep retired concepts from the full 1,394-concept JSON. The articles are full news texts (medium length).
- For DBpedia, Wikidata descriptions of the matching classes (via `owl:equivalentClass` to Wikidata) are likely a better gloss source than DBpedia's own sparse `rdfs:comment`s. Wikidata glosses are short (~5-20 words) and largely available in many languages (not measured).
- The weak or tautological IPTC top-level definitions ("All forms of ...") mean level-1 zero-shot gets little from descriptions. Deeper levels have more specific definitions.

### Gaps
- Other IPTC-labelled corpora (e.g. the 2025 EMMediaTopic / multilingual news topic sets by Kuzman & Ljubešić): not verified in this session.
- Whether MN-DS stores IPTC qcodes or only names: not checked.
- DBpedia ontology license (likely CC BY-SA) and Wikidata's CC0 status: not re-verified here.
- DBpedia-14 dataset size and license: not fetched this session.
- Whether the DBpedia ontology dump (Databus) has more English comments than the live endpoint: not checked.

## 6. Extreme multi-label "label feature" (LF) datasets

### Takeaway
The Extreme Classification Repository's LF datasets have 131K-1.3M labels, each with short label text: titles of Amazon products or Wikipedia articles, not definitions. They test scale and short-text matching, not definitional zero-shot. The repository page states no license.

### Cited Findings
- Definition: "Datasets that contain label features have the token "LF" prepended to their names. These are usually short textual descriptions of the labels." "Titles" datasets are short-text datasets "whose data points are represented by a 3-5 word textual description such as the name of a product or title of a webpage." — [Extreme Classification Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html)
- LF versions drop labels without retrievable label text: "the raw dumps ... often contained labels for which label features were unavailable or could not be reliably retrieved. Such labels could exist in the non-LF dataset but were excluded from the LF version". — [XC Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html)
- Stats (labels / train / test / avg points per label / avg labels per point), all with BoW + raw text downloads — [XC Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html):
  - LF-AmazonTitles-131K: 131,073 / 294,805 / 134,835 / 5.15 / 2.29
  - LF-Amazon-131K: 131,073 / 294,805 / 134,835 / 5.15 / 2.29 (full-text points)
  - LF-WikiSeeAlsoTitles-320K: 312,330 / 693,082 / 177,515 / 4.67 / 2.11
  - LF-WikiSeeAlso-320K: 312,330 / 693,082 / 177,515 / 4.67 / 2.11
  - LF-WikiTitles-500K: 501,070 / 1,813,391 / 783,743 / 17.15 / 4.74
  - LF-Wikipedia-500K: 501,070 / 1,813,391 / 783,743 / 24.75 / 4.77 (Wikipedia articles as points, Wikipedia categories as labels; replaces the separate Wikipedia-500K)
  - LF-AmazonTitles-1.3M: 1,305,265 / 2,248,619 / 970,237 / 38.24 / 22.20
  - Also MM-AmazonTitles-300K (multimodal: text + images per label): 303,296 labels.
- (measured) A text search of the repository page found no license statement. — [XC Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html)

### Inferences
- In LF-Amazon and LF-WikiSeeAlso, labels are items or articles whose "features" are their titles. The label text is a name, not a definition, so these datasets add nothing over names for "name: description" prompting. LF-Wikipedia-500K labels are Wikipedia category names.
- Label text ships inside the dataset, so no join is needed.
- The underlying data (Amazon reviews/metadata, Wikipedia) has its own terms. With no license on the page, redistribution status is unclear.

### Gaps
- Exact label-text format (e.g. whether LF-WikiSeeAlso includes article body text for labels): not confirmed. The page says only "short textual descriptions".
- Licenses of LF datasets: not stated.

## 7. Which give the most distinct labels with real descriptions; short texts vs long documents

### Takeaway
Ranked by count of labels with **real, definitional** descriptions: MeSH (~30.4k, ~198 chars) > ESCO skills (~13.9k) > CPC (very large, but definitions are long, multi-section and only at subclass/group level) > ESCO occupations (3,039) > EuroVoc (~1.7k of 7.5k described) > IPTC (1,059, 100% coverage) ≈ O*NET (1,016, 100%) > DBpedia ontology (282 of 790) > arXiv (~150). ICD-10-CM (~74.7k) and the XC LF sets (131K-1.3M) have many labels but title-only text. For short texts: ESCO/TechWolf job titles and sentences, the LF-*Titles sets, and DBpedia/Wikipedia abstracts. For long documents: MultiEURLEX, HUPD/BIGPATENT, MIMIC-IV discharge notes (credentialed), and PubMed abstracts (medium).

### Cited Findings
- Summary inventory (each row's facts are sourced in sections 1-6):

| Dataset + taxonomy | Docs | Labels (depth) | Description source, length, coverage | Single/multi | Langs | License (data / taxonomy) | Join effort |
|---|---|---|---|---|---|---|---|
| MultiEURLEX + EuroVoc | 65k laws | 21/127/567 (L1-3), 7,390 orig., depth 8 | EuroVoc definitions/scope notes, ~100-130 chars; 0% L1-2, 25% L3, 20% overall | Multi | 23 | CC BY 4.0 / EU reuse | Low (ID join), low payoff |
| HUPD + CPC/IPC | 2004-2018 USPTO apps (count not on card) | CPC symbols (section → subgroup) | CPC titles (all symbols); definitions for subclass/group, multi-section, thousands of words per subclass | Multi | EN | CC BY-SA vs CC BY-NC-SA (**conflict**) / CPC terms unverified | Medium |
| BIGPATENT + CPC sections | 1.34M | 9 | Section titles only | Single per subset | EN | CC BY 4.0 | Trivial |
| MIMIC-IV(-ICD) + ICD-10-CM | 122,317 ICD-10 notes | 26,096 used; ~74.7k in CM | Code titles, ~5-10 words | Multi (16.1/doc) | EN | **Credentialed** PhysioNet 1.5.0 / US gov | Trivial in-dataset |
| BioASQ/PubMed + MeSH | 15.56M (2021) | 29,369 used; 30,594 current | MeSH scope notes, ~198 chars, 99.4% | Multi (12.68/doc) | EN (MeSH translations exist) | Registration / NLM terms (unverified) | Low |
| arXiv + arXiv taxonomy | (not checked) | 155 (2-3 levels) | Official one-liners, some empty | Multi (cross-lists) | EN | (not checked) | Low (scrape) |
| MN-DS + IPTC Media Topics | 10,917 | 17 / 109 used; 1,059 active, depth 6 | IPTC definitions, ~99 chars, 100% | Hierarchical, ~1 per L2 | EN (taxonomy ~12 langs) | CC BY 4.0 / CC BY 4.0 | Low-medium (2020 vs 2026 version) |
| DBPedia Classes + DBpedia ontology | 342,782 | 9/70/219 | rdfs:comment, ~97 chars, ~36% of ontology | Single per level | EN | CC0 / (unverified) | Medium (use Wikidata glosses) |
| TechWolf sets + ESCO | ~138K synthetic sentences; smaller gold sets | 13,939 skills; 3,039 occupations | ESCO descriptions, multi-sentence | Multi (spans) / single (titles) | Descriptions in 28 langs | (HF cards) / free reuse w/ attribution | Low (version drift) |
| O*NET | (no public labelled corpus found) | 1,016 (923 data-level) | O*NET descriptions, ~214 chars, 100% | n/a | EN | Creative Commons | n/a |
| Google Product Taxonomy | (no corpus checked) | 5,595, depth 7 | **None**: names/paths only | n/a | multi-locale files exist (not checked) | (unverified) | n/a |
| XC LF-* sets | 295K-2.25M train | 131K-1.3M | Label titles (3-5 words) | Multi | EN | No license stated | None (bundled) |

- Typical description lengths (measured): MeSH ~198 chars ([MeSH SPARQL](https://id.nlm.nih.gov/mesh/sparql)); O*NET ~214 ([O*NET file](https://www.onetcenter.org/dl_files/database/db_30_3_text/Occupation%20Data.txt)); EuroVoc definitions ~130 and scope notes ~100 ([EuroVoc SPARQL](https://publications.europa.eu/webapi/rdf/sparql)); IPTC ~99 ([IPTC JSON](https://cv.iptc.org/newscodes/mediatopic/?lang=en-GB&format=json)); DBpedia ~97 ([DBpedia SPARQL](https://dbpedia.org/sparql)).
- Document lengths: MIMIC-IV ICD-10 notes average 1,662 words ([arXiv 2304.13998](https://arxiv.org/pdf/2304.13998)). XC "Titles" points are 3-5 words ([XC Repository](https://raw.githubusercontent.com/manikvarma/manikvarma.github.io/master/downloads/XC/XMLRepository.html)).

### Inferences
- Best open "many distinct labels + real descriptions" candidate: **MeSH with PubMed abstracts**. Best multilingual-description candidate: **ESCO** (skills or occupations, 28 languages) with TechWolf job-ad sentences or titles. Best small, clean, fully described, open benchmark: **IPTC + MN-DS**, with **arXiv** as a second option.
- Best long-document candidates: MultiEURLEX (open, but descriptions mostly missing, so the description ablation is weak) and HUPD + CPC definitions (rich definitions, license caveat).
- Avoid, or treat as name-only controls: ICD-10-CM via MIMIC (credentialed, titles only), Google product taxonomy (no descriptions), XC LF sets (titles only).
- Datasets where descriptions are only partially available (EuroVoc, DBpedia) are natural testbeds for "description vs name-only" ablations within one label space.

### Gaps
- Per-label description length distributions for CPC definitions, ESCO skills and ICD-10-CM: not measured.
- No source compared these taxonomies' description quality systematically. The rankings above are my synthesis from measured coverage and length.
