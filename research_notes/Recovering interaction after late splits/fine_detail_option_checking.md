# Fine-detail option checking: heads, interaction depth and training signals that make a model check each option against the context

Scope note: what the numbers mean for LIDM (ModernBERT-large, pass A = `[CLS] state [SEP] question [SEP]`, pass B = each option encoded separately, 2-block fusion head with option self-attention + option-to-pass-A cross-attention + GeGLU, optional looping) is given under "Inferences". Every number below lists backbone, task, split and metric where the source gave them. Dates are arXiv first-posting dates (from the arXiv ID) unless a venue was confirmed on the page.

A key distinction for reading the numbers: most MRC "matching heads" (DUMA, OCN, MAN/MMM) sit on top of a **joint** encoder (passage + question + option in one sequence) and re-split its output. DCMN+ is the one major head that the paper describes as encoding passage, question and option **separately**, which makes it the closest published analogue to LIDM.

## 1. Multiple-choice reading heads (DCMN+, DUMA, OCN, co-matching, MMM/MAN, option interaction): gains over a plain [CLS] head on the same backbone

### Takeaway
On BERT-class backbones, matching heads add about +0.7 to +4.8 RACE points over a plain [CLS] head. On strong backbones (ALBERT-xxlarge, ELECTRA-large) only DUMA (bidirectional co-attention, 2 shared layers, about +15% head params) still gives clear gains: +1.4 RACE, +1.6 to +2.0 DREAM. OCN, WAE and BiDAF give +0.2 to +0.9, and DCMN goes negative. Plain extra self-attention layers on top of the encoder recover about 60% of DUMA's gain. Option-to-option comparison on its own is worth under 1 point.

### Cited Findings
**DUMA** (Zhu, Zhao, Li; arXiv 2001.09415, Jan 2020; venue not stated on the fetched page) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Design: the encoder input is the joint P ⊕ Q ⊕ A. The output is split into E^P (passage) and E^QA (question+option). Two multi-head attentions run in opposite directions (MHA1: P queries QA; MHA2: QA queries P). Each branch is mean-pooled, then concatenated. k = 2 DUMA layers, with parameters shared across layers and across both branches. Each option is scored by a learned vector with a softmax over options. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- RACE test, ALBERT-xxlarge: 86.6 → 88.0 with DUMA (+1.4); middle 89.0 → 90.9, high 85.5 → 86.7. RACE dev: 87.4 → 88.1 (+0.7). Ensemble: 89.8 test. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- DREAM dev/test, ALBERT-xxlarge: 89.2/88.5 → 89.9/90.5 with DUMA (+0.7/+2.0). Adding multi-task learning gives 91.8 test. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Same-backbone head comparison, DREAM test (Table 7). Columns are ALBERT-base / ALBERT-xxlarge / ELECTRA-large:
  - Baseline: 64.4 / 88.5 / 88.2
  - Soft attention: 65.4 / 88.9 / 88.8
  - BiDAF: 65.6 / 89.3 / 89.1
  - OCN: 65.8 / 89.2 / 89.0
  - WAE: 66.5 / 89.9 / 89.5
  - DCMN: 63.3 / 87.8 / 87.7
  - **DUMA: 67.6 / 90.5 / 89.8**

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Same-backbone head comparison, RACE test (Table 1; prior heads, some numbers re-implemented):
  - BERT-large 71.0: OCN +0.7, WAE +2.1, DCMN +4.8
  - XLNet-large 80.1: OCN +0.8, WAE +1.7, DCMN +2.7
  - ALBERT-xxlarge 86.6: OCN +0.6, WAE +0.7, **DCMN −0.9**
  - ELECTRA-large 86.1: OCN +0.2, WAE +0.8, **DCMN −1.2**

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Control for "just add layers", ALBERT-base on DREAM dev/test (Table 6): baseline 64.51/64.43; adding vanilla self-attention gives 66.27/66.34; DUMA gives 67.06/67.56; TB-DUMA (with LayerNorm + FFN, i.e. full transformer blocks) gives 67.79/67.17. Full blocks are therefore no better than bare co-attention. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Direction ablation, ALBERT-base, DREAM, average of dev and test (Table 10): baseline 64.47. "P-to-Q" only gives 64.67 (+0.20), "Q-to-P" only gives 66.53 (+2.06), and both directions give 67.31 (+2.84). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Fusion ablation, ALBERT-base, DREAM dev/test (Table 8): element-wise product 65.29/64.58, sum 66.32/65.51, concatenation 67.06/67.56. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- On BERT-base, DREAM average: 61.36 → 64.43 (+3.07). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Head parameter counts on ALBERT-base (11.7M): DUMA 13.5M (+15.4%), BiDAF 12.0M, OCN 14.8M, DCMN 19.4M (+65.8%), WAE 23.4M (+100%). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)
- Replacing the encoder's own self-attention with co-attention collapses DREAM test to 40.08, against 64.43 for the baseline. Co-attention must be added on top of the encoder's self-attention, not used instead of it. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2001.09415)

**DCMN+** (Zhang et al.; arXiv 1908.11511, Aug 2019; AAAI 2020 per the brief, venue not shown on the page) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- Encoding is separate according to Eq. 1: H^p = Encode(P), H^q = Encode(Q), H^a = Encode(A). The paper does not compare separate with joint encoding. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- Components:
  - Bidirectional gated matching for P–Q, P–A and Q–A, with the three results concatenated.
  - Passage sentence selection (PSS): top-K sentences by cosine or bilinear score.
  - Answer option interaction (AOI): bilinear comparison between options, fused through a gate.

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- RACE test against the fine-tuned [CLS] baseline:
  - BERT-base: 65.0 → 67.0 with DCMN
  - BERT-large: 72.0 → 74.1 with DCMN → 75.8 with full DCMN+ (middle 79.3, high 74.4)
  - XLNet-large: 80.1 → 82.8 with DCMN+

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- Ablation, RACE dev (Table 5). Bidirectional matching is the largest contributor; PSS and AOI each add about 0.6 to 0.8 on top.
  - BERT-base: 64.6 → DCMN 66.0 → +PSS 66.6 / +AOI 66.8 → all 67.4
  - BERT-large: 71.8 → 73.8 → 74.6 / 74.4 → 75.4
  - XLNet-large: 80.1 → 81.5 → 82.1 / 82.2 → 82.6

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- Other datasets (Table 6), baseline → DCMN+:
  - With BERT: SemEval-2018 T11 90.5 → 91.8, ROCStories 90.8 → 92.4, MCTest MC160 73.8 → 85.0, MC500 80.4 → 86.5, COIN (dev) 84.3 → 88.8; average +4.9
  - With XLNet: average +2.8, including MC160 80.6 → 86.2

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)
- PSS choice of K: on RACE the best cosine-score K is 5 (66.5) and the best bilinear K is 4 (66.8); on COIN the best is K = 3. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1908.11511)

**OCN** (Ran et al., Tencent; arXiv 1903.03033, Mar 2019) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1903.03033)
- Each option is encoded jointly with the question and article ("for option O_k, it is concatenated with the question Q and article P"). Options are then compared at the word level with attention against every other option, fused through a gate, and the article is re-read with co-attention. Scores are max-pooled. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1903.03033)
- RACE: BERT-base 65.2 → OCN 66.8 (+1.6); BERT-large 71.0 → 71.7 (+0.7). Removing option comparison from OCN-base lowers 66.8 to 66.1, so option comparison is worth +0.7. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1903.03033)

**MMM / MAN head** (Jin et al.; arXiv 1910.00458, Oct 2019) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.00458)
- MAN design: a GRU state over passage memory attends over the (question, option) tokens for K steps. The logit is computed from [s; x; |s−x|; s·x]. Gains rise gradually from K = 1 to K = 5 and saturate after 5 (shown only in a figure, without numbers in the text). — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.00458)
- Ablation, BERT-base, dev (Table 5), with drops relative to the full model's DREAM 72.6 / MC160 86.7 / MC500 83.5:
  - Replacing MAN with a plain FCNN head costs −1.4 / −1.3 / −2.0.
  - Removing NLI coarse-tuning costs −3.1 / −5.9 / −1.7.
  - Removing the RACE multi-task stage costs −4.1 / −14.2 / −5.5.

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.00458)
- DREAM test with the full MMM recipe: BERT-base 63.2 → 72.2, BERT-large 66.9 → 76.0, RoBERTa-large 85.0 → 88.9. RACE test: RoBERTa-large 83.2 → 85.0, BERT-base 65.0 → 68.0. MCTest MC160 test, RoBERTa-large: 81.7 → 97.1. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.00458)

### Inferences
- DCMN+ is the closest published relative of LIDM: separate P/Q/A encodings plus a bidirectional matching head. On BERT/XLNet it beats the joint [CLS] baseline by 2.7 to 3.8 RACE points. Re-implemented on ALBERT-xxlarge or ELECTRA-large (DUMA Table 1), it is 0.9 to 1.2 points **below** the joint baseline. This matches LIDM's −5 on RACE against a joint ModernBERT-large: a separate-encoding head on a strong encoder does not, by itself, close the gap to joint encoding.
- LIDM's head has only one direction (option tokens query pass-A tokens). DUMA's ablation shows a single direction captures between 7% and 73% of the two-direction gain, depending on direction. The direction labels in DUMA's Table 10 are ambiguous about which is which. Adding the reverse direction is cheap: pass-A tokens query option tokens, pool, then concatenate with the option branch. DUMA's ablations support concatenation fusion, shared weights across layers and k = 2.
- DUMA's "vanilla self-attention" control suggests that part of any head's gain is just added depth. Ablations of LIDM's head should therefore include an "extra self-attention only" arm.
- MMM's ablations show that, for small MC sets (MCTest, DREAM), **transfer data** (NLI coarse-tuning, RACE multi-task) beats the head by 2 to 10x in effect size.

### Gaps
- No published DUMA, DCMN+ or OCN numbers on DeBERTa-v3, ModernBERT or other 2023–2026 encoders were found. The strongest backbones tested are ALBERT-xxlarge and ELECTRA-large.
- No fetched source reports these heads on CosmosQA.
- The original co-matching network (Wang et al. 2018) and later "option interaction" layers (2021–2026) were not fetched.
- DUMA's layer-count figure values are not in the text; the paper says only that 2 layers is best and more layers drop slightly.
- DUMA's venue could not be confirmed on the fetched page.

## 2. Claim–evidence fine-grained alignment for fact verification (KGAT, VitaminC, token-level alignment) and what it takes to catch single-word flips

### Takeaway
Token-level alignment heads help a little: KGAT's Gaussian-kernel token matching adds about +0.9 to +2.1 label accuracy (LA) on FEVER over same-backbone baselines, and replacing the kernels with dot-product attention costs about −0.65 to −1.0. Contrastive training data, by contrast, has a very large effect. On the same ALBERT cross-encoder, training on VitaminC's one-fact-changed pairs instead of FEVER lifts VitaminC-real accuracy by +30 points and adversarial FEVER by about +9 to +10. The FEVER-trained model's flip rate under contrastive evidence is only 55% (base) to 65% (xlarge); VitaminC training raises it to 84% and 90%. Catching single-word flips is mainly a training-signal problem, not a head problem.

### Cited Findings
**KGAT** (Liu et al.; arXiv 1910.09796, Oct 2019; ACL 2020 per the brief) — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.09796)
- Design: each claim–evidence pair is a node, encoded jointly by BERT as claim + title + evidence. Token-level cosine-similarity matrices between nodes are pooled with 21 Gaussian kernels. Edge kernels propagate evidence; node kernels (averaged over claim tokens) select evidence. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.09796)
- FEVER test, LA / FEVER score:
  - BERT-base: KGAT 72.81 / 69.40 vs a BERT-base baseline (Soleimani et al.) at 70.67 / 68.50
  - With ESIM retrieval: KGAT 72.48 / 68.16 vs GEAR 71.60 / 67.10
  - BERT-large: KGAT 73.61 / 70.24 vs BERT-large baseline 71.86 / 69.66
  - RoBERTa-large: KGAT 74.07 / 70.38

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.09796)
- Ablation, FEVER score on multi-evidence / single-evidence claims: full KGAT 39.23 / 78.07; all kernels replaced by dot-product attention (GAT) 38.21 / 77.42; node kernels only 38.52 / 77.73; edge kernels only 39.08 / 77.58. Single-evidence claims are 85.3% of the set. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1910.09796)

**VitaminC** (Schuster, Fisch, Barzilay, MIT; arXiv 2103.08541, Mar 2021; NAACL 2021 per the brief) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- Construction:
  - 304,671 revised Wikipedia sentences were annotated; 107,056 (35%) were factual changes.
  - For each factual revision, annotators wrote two symmetric claims, one supported by the old sentence and refuted by the new one, and the reverse. This gives 325,724 real pairs.
  - A further 163,180 synthetic pairs come from edited FEVER evidence, for over 400K pairs in total.

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- ALBERT-base test accuracy, FEVER-trained → VitaminC-trained:
  - VitC-real 54.78 → 86.16
  - VitC-synthetic 79.18 → 89.78
  - Adversarial FEVER 62.01 → 71.41
  - Symmetric FEVER 81.18 → 90.17
  - ANLI 32.94 → 37.25
  - "Contrast" (% of FEVER claims whose prediction flips correctly under contrastive evidence) 55.53 → 84.11
  - FEVER itself 95.07 → 74.56

  Training on VitaminC + FEVER keeps FEVER at 94.24 with VitC-real at 86.26 and ANLI at 38.50. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- ALBERT-xlarge, FEVER-trained → VitaminC-trained:
  - VitC-real 58.56 → 88.64
  - Adversarial 72.58 → 82.51
  - Symmetric 87.08 → 94.80
  - ANLI 36.03 → 42.66
  - Contrast 64.59 → 89.57

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- BERT-base, FEVER-trained → VitaminC-trained: VitC-real 60.55 → 85.80; Contrast 54.05 → 72.49, rising to 84.01 with VitC + FEVER. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- Larger backbones alone do not fix this. FEVER + MNLI training on ALBERT-xlarge reaches only 61.21 on VitC-real. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- Word-level rationales (300 annotated test examples): adding token labels derived from the revision diff raises edit-token F1 from 31.23 to 59.11 and rationale F1 from 37.33 to 47.36. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- Factual-revision flagging (deciding whether an edit changes a fact): ALBERT trained on VitC-full reaches AUC 91.97 / F1 83.18, against 71.34 / 63.56 for an edit-distance baseline. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)

### Inferences
- LIDM's −7 on VitaminC against the joint encoder was measured on the same training data. It is therefore likely an interaction-capacity effect. A one-token flip needs the claim token and the evidence token to meet in some attention layer, and in LIDM that happens only in the two fusion blocks.
- KGAT shows that explicit token-match features (a cosine-similarity matrix pooled by kernels) beat plain dot-product attention by about 1 point at the same encoder. This is a cheap addition to LIDM's cross-attention: a kernel-pooled feature vector from the option-token × pass-A-token cosine matrix. ESIM/MAN-style [a; b; |a−b|; a⊙b] difference features are a similar option.
- If LIDM trains with VitaminC pairs, its gap to the joint encoder should also be checked in the "contrast flip" form: the same claim with old and new evidence must flip.
- A diff-token auxiliary loss is a candidate cheap training signal. VitaminC's word-level supervision from revision diffs roughly doubles edit-token F1.

### Gaps
- No source compares a separately encoded claim and evidence with a fine-alignment head against a joint cross-encoder on VitaminC.
- KGAT was not evaluated on VitaminC in the fetched sources.
- No 2023–2026 FEVER/VitaminC head papers were fetched.

## 3. Are co-attention gains additive to a strong pretrained encoder, or do they vanish with scale?

### Takeaway
Gains shrink but do not vanish for the best head. DUMA goes from +3.1 to +3.2 (BERT-base and ALBERT-base, DREAM) to +2.0 (ALBERT-xxlarge, DREAM test), +1.6 (ELECTRA-large, DREAM test) and +1.4 (ALBERT-xxlarge, RACE test). Most other heads shrink to +0.2 to +0.9 or turn negative on strong encoders. Fact verification shows the same pattern: KGAT's gain over a BERT baseline is about +2.1 LA on base and +1.8 LA on large.

### Cited Findings
- DUMA gains (DREAM test unless noted):
  - BERT-base, DREAM average: +3.07
  - ALBERT-base: +3.2
  - ALBERT-xxlarge: +2.0, and +1.4 on RACE test
  - ELECTRA-large: +1.6

  — [ar5iv DUMA](https://ar5iv.labs.arxiv.org/html/2001.09415)
- DCMN on RACE: +4.8 on BERT-large and +2.7 on XLNet-large, but −0.9 on ALBERT-xxlarge and −1.2 on ELECTRA-large. OCN: +0.7 on BERT-large, +0.6 on ALBERT-xxlarge, +0.2 on ELECTRA-large. — [ar5iv DUMA](https://ar5iv.labs.arxiv.org/html/2001.09415)
- DCMN+ gain over the baseline: +3.8 RACE test on BERT-large, +2.7 on XLNet-large. Average gain on 5 other MC sets: +4.9 with BERT and +2.8 with XLNet. — [ar5iv DCMN+](https://ar5iv.labs.arxiv.org/html/1908.11511)
- MMM's full recipe adds +9.0 DREAM test on BERT-base but +3.9 on RoBERTa-large; on RACE test, +3.0 on BERT-base and +1.8 on RoBERTa-large. — [ar5iv MMM](https://ar5iv.labs.arxiv.org/html/1910.00458)
- KGAT test LA: +2.14 over the BERT-base baseline and +1.75 over BERT-large. — [ar5iv KGAT](https://ar5iv.labs.arxiv.org/html/1910.09796)

### Inferences
- On a ModernBERT-large-class encoder, expect at most about +1 to +2 points from a better head over a joint [CLS] baseline. A richer head on separately encoded options should not be expected to beat a joint encoder. It can plausibly narrow the gap, and DCMN+-style separate encoding still lost about 1 point to joint [CLS] on strong encoders.
- Extrapolating DUMA's trend, the part of LIDM's −5 (RACE) and −7 (VitaminC) gap that a head alone can recover is probably small (on the order of 1 to 2 points). Bigger levers are late joint layers (Section 4) and training data (Section 5). This is an inference, not a measured result.

### Gaps
- No measurement of DUMA-type heads on DeBERTa-v3-large or ModernBERT was found.
- The 2022+ RACE and DREAM leaderboards were not fetched, so later SOTA head designs are not covered.

## 4. How much interaction depth is needed (cross-attention layers, iterative or looped heads, partial joint encoding)?

### Takeaway
The evidence points to two to three joint layers being enough, if they are pretrained layers. DeFormer separates the lower 9 of 12 BERT-base layers and runs the top 3 jointly; with distillation it loses only 1.8 RACE points and 1.4 SQuAD F1. Fresh heads saturate quickly: DUMA is best at 2 shared layers, and MAN gains saturate around 5 steps. Looping a shared block recovers part of the depth gain on open-book (reading-style) QA but stays below a true deeper model.

### Cited Findings
**DeFormer** (Cao et al.; arXiv 2005.00697, May 2020) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)
- Lower k layers encode the question and the passage separately; the upper layers run jointly. Auxiliary losses: knowledge distillation (KD) from the joint model and layerwise representation similarity (LRS). BERT-base uses 9 separate + 3 joint layers; XLNet-base uses 8 + 4. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)
- RACE accuracy: BERT-base 66.3 → 64.5 (−1.8, 3.4x faster); XLNet-base 70.3 → 68.7 (−1.6). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)
- BoolQ: BERT-base 77.8 → 76.8; XLNet-base 80.4 → 78.8. MNLI (BERT-base): 84.4 → 82.6. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)
- SQuAD F1 ablation (base / large):
  - Joint model: 88.5 / 92.3
  - DeFormer with KD + LRS: 87.1 / 90.8
  - Without LRS: 86.2 / 88.9
  - Without KD and LRS: 85.8 / 87.5

  Distillation from the joint model recovers 1.3 (base) and 3.3 (large) F1 points. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)
- Effect of where the split is placed: the drop is negligible when only the lowest layers are separated (up to layer 3 of 12 for base, up to 13 of 24 for large). It grows beyond that and rises sharply when the last layers are separated too. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00697)

**Fresh-head depth**
- DUMA: 2 shared-weight layers chosen; performance "fluctuates" and drops slightly with more. — [ar5iv DUMA](https://ar5iv.labs.arxiv.org/html/2001.09415)
- MMM's MAN head: gains grow gradually from 1 to 5 reasoning steps, then saturate. — [ar5iv MMM](https://ar5iv.labs.arxiv.org/html/1910.00458)

**Looped transformers** (Saunshi et al., "Reasoning with Latent Thoughts", arXiv 2502.17416, Feb 2025, ICLR 2025)
- About 1B-scale models trained on the Pile. Open-book QA (reading-style) average accuracy:
  - 24-layer baseline: 33.9
  - 12-layer: 26.9; 12 layers looped twice (12⊗2): 30.8
  - 8-layer: 22.7; 8⊗3: 30.8
  - 4-layer: 13.8; 4⊗6: 26.2

  No looped model beats the 24-layer baseline on open-book QA. — [arXiv HTML](https://arxiv.org/html/2502.17416)
- Looping helps math and reasoning primitives more than knowledge recall. Math word problems: 12⊗2 scores 34.3 against 29.3 for the 24-layer baseline. Closed-book QA: 12⊗2 scores 9.3 against 11.2 for the baseline. — [arXiv HTML](https://arxiv.org/html/2502.17416)
- "Looping-inspired" regularization (a cosine-similarity penalty tying successive k-layer blocks of an unshared 24-layer model; k = 4, λ = 10) raises open-book QA from 33.9 to 36.2, math from 29.3 to 36.4 and reasoning primitives from 47.5 to 57.2, with perplexity essentially unchanged (7.40 → 7.38). — [arXiv HTML](https://arxiv.org/html/2502.17416)

### Inferences
- LIDM separates all 28 layers, which is exactly the "separate the last layers" regime where DeFormer reports the sharpest drop. The DeFormer evidence suggests a structural fix: run the **top 2 to 4 pretrained ModernBERT layers jointly** on [pass-A tokens ‖ option tokens], or on [pass-A ‖ all options], and cache layers 1 to 24 or 26 per segment. This is cheaper than a full joint encoder and starts from pretrained interaction weights, unlike a fresh fusion head. DeFormer recovered to within about 2 RACE points of joint encoding on BERT-base.
- LIDM already has a strong joint encoder. Distilling its logits (KD) into LIDM and matching upper-layer representations (LRS) is the most direct transfer of DeFormer's recovery result. It was worth 1.3 to 4.8 SQuAD F1 in DeFormer's ablation.
- Looping LIDM's fusion blocks is likely to give the "iso-param" gain (more effective depth at fixed parameters) but not to match real extra layers on reading tasks. The Saunshi regularization result suggests that tying weights softly is better than tying them hard.

### Gaps
- No source measured looped or recurrent heads (Universal Transformer, Huginn-style recurrent depth) specifically on RACE, DREAM, ANLI or VitaminC with encoders. Huginn-style results were not fetched.
- DeFormer's per-k curve is only in a figure (Fig. 4), so exact numbers per split layer are unavailable.
- Poly-encoder and ColBERT-style late-interaction comparisons are presumably covered by other researchers and were not fetched here.

## 5. Training signals for minimal-pair sensitivity (contrast sets, counterfactual augmentation, VitaminC pairs, generated pairs)

### Takeaway
Standard fine-tuning leaves large contrast gaps. Models lose 4.6 to 25.7 points on contrast sets, and the share of contrast groups answered fully correctly is 8% to 79%. Small amounts of minimal-pair training data close much of the gap:
- VitaminC: +30 on VitC-real and +29 on contrast flip rate (ALBERT-base).
- Counterfactually-augmented data (CAD) on SNLI: 8.3k mixed examples beat 500k original examples on revised-premise tests (64.6 vs 54.3).
- LLM-generated counterfactuals (DISCO): +6 on stress tests and +10 on pair consistency.

### Cited Findings
**Contrast sets** (Gardner et al., arXiv 2004.02709, Apr 2020). Original test → contrast set, with consistency (all items in a contrast group correct):
- BoolQ, RoBERTa: 86.1 → 71.1 accuracy; consistency 59.0
- DROP, MTMSN: 79.9 → 54.2 F1; 39.0
- Quoref, XLNet-QA: 70.5 → 55.4 F1; 29.9
- ROPES, RoBERTa: 47.7 → 32.5 EM; 17.6
- MC-TACO, RoBERTa: 38.0 → 14.0 EM; 8.0
- PERSPECTRUM, RoBERTa: 90.3 → 85.7; 78.8
- IMDb, BERT: 93.8 → 84.2; 77.8

Humans drop only 0.4 to 6.8 points. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2004.02709)

**Counterfactually-augmented data** (Kaushik, Hovy, Lipton; arXiv 1909.12434, Sep 2019; ICLR 2020 per common citation). BERT-base on SNLI, accuracy on Original / revised premise (RP) / revised hypothesis (RH):
- Trained on original 1.67k: 72.2 / 39.7 / 59.5
- Trained on RP & RH 6.6k: 64.7 / 64.6 / 67.8
- Trained on original + RP & RH, 8.3k: 73.5 / 64.6 / 69.6
- Trained on original 8.3k (size control): 77.8 / 44.6 / 66.1
- Trained on original 500k: 90.4 / 54.3 / 74.3

— [ar5iv](https://ar5iv.labs.arxiv.org/html/1909.12434)

- On IMDb, BERT trained on original + revised (3.4k) scores 88.5 on original and 95.1 on revised, against 90.2 and 86.1 for 3.4k original only. Out of domain, the same comparison gives Amazon 85.1 vs 80.0, SemEval Twitter 82.9 vs 79.3 and Yelp 89.4 vs 85.3. — [ar5iv](https://ar5iv.labs.arxiv.org/html/1909.12434)
- VitaminC contrastive pairs on ALBERT-base (from Section 2): contrast flip rate 55.53 → 84.11; ANLI 32.94 → 37.25 / 38.50. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2103.08541)
- DISCO (Chen et al., arXiv 2212.10534, Dec 2022; ACL 2023). An LLM generates phrasal perturbations, a teacher model filters them, and the result trains a small student. Reported gains: about +6% absolute on an NLI stress test, about +2% OOD, and about +10% counterfactual-pair consistency across 3 evaluation sets. — [arXiv](https://arxiv.org/abs/2212.10534)
- Petrov (arXiv 2501.02683, Jan 2025): ELECTRA-small on SNLI scores 89.8 on the test set and 74.9 on an automatically generated contrast set. After fine-tuning on contrast examples it reaches 90.7 on the contrast set and 89.3 on SNLI, with gains plateauing at about 1,500 examples. These figures come from a search summary, the work is a single small study, and the full text was not verified. — [arXiv](https://arxiv.org/html/2501.02683v1)

### Inferences
- In every case, the gain comes from *paired* data, where a minimal edit flips the label in the same training set. A random sample of more of the same original data does not do this: original 8.3k → 500k moved RP accuracy only from 44.6 to 54.3.
- For LIDM this argues for mixing VitaminC pairs, CAD/SNLI-revised pairs and CondaQA edit groups into stage training as grouped batches (both members of a pair in the same batch). This is cheap on one 12 GB GPU because the sets are 1.7k to 400k examples. A pairwise consistency loss is an untested extension; no fetched paper isolates the loss from the data.
- Report contrast consistency (group all-correct) next to accuracy. Gardner and CondaQA both show accuracy hides a 2 to 4x drop in consistency.

### Gaps
- "Nimble-style generated pairs" could not be identified as a specific published method. No source was found.
- No fetched paper ablates *pairwise / contrastive losses* against simply adding the pairs as ordinary examples.
- No source applies contrast-set training on RACE or DREAM-style multiple choice specifically.

## 6. Negation handling (CondaQA, NegBench etc.): what improved it for encoder models

### Takeaway
Fine-tuned large encoders sit near the CondaQA majority baseline (BERT-large 46.3, RoBERTa-large 54.08, DeBERTa-v3-large 57.09 against a 47.75 majority). Consistency is 2% to 18%. What helped:
- Scale and QA multitask pretraining: UnifiedQA-v2-3B fine-tuned reaches 73.26 / 42.18.
- Cheap negation-specific further pretraining: next-sentence polarity prediction (NSPP) or negation-reversed next-sentence prediction (NSP) on Wikipedia gives +2 to +9 accuracy and up to +15 consistency.
- Training on affirmative paraphrases of negated sentences.

LIDM's near-floor CondaQA score is consistent with an encoder problem in general, not a split-architecture problem.

### Cited Findings
**CondaQA** (Ravichander et al., arXiv 2211.00295, Nov 2022; EMNLP 2022 per the brief)
- 14,182 QA pairs; 219 negation cues, 75 of them unseen in training. Edit types are paraphrase, scope and affirmative. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.00295)
- Test accuracy / group consistency:
  - Majority baseline: 47.75 / 1.35
  - Negation-cue-only heuristic: 56.79 / 14.89
  - Fine-tuned BERT-large: 46.3 / 2.20
  - Fine-tuned RoBERTa-large: 54.08 / 13.64
  - Fine-tuned DeBERTa-v2-XLarge: 54.01 / 13.37
  - Fine-tuned DeBERTa-v3-Large: 57.09 / 18.02
  - Fine-tuned UnifiedQA-v2-Large: 66.72 / 30.20
  - Fine-tuned UnifiedQA-v2-3B: 73.26 / 42.18
  - Zero-shot UnifiedQA-v2-11B: 73.11 / 40.02
  - Zero-shot FLAN-T5-XXL: 67.53 / 31.61
  - Human: 91.94 / 81.58

  — [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.00295)
- Scope and affirmative edits are hardest; for UnifiedQA-v2-3B, consistency on those edits is 55.68 and 57.22, against 72.80 for paraphrase. When negation is undone, the model changes its prediction on only 43.52% of 1,080 pairs. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.00295)

**NSPP / negation-reversed NSP further pretraining** (Rezaei & Blanco, arXiv 2502.07717, Feb 2025)
- Training uses 500K to 1M Wikipedia sentence pairs at learning rate 1e-6; there is no inference-time cost. — [arXiv HTML](https://arxiv.org/html/2502.07717v1)
- CondaQA test accuracy / consistency:
  - RoBERTa-large: 64.9 / 29.6 → 68.3 / 34.8 (with NSPP + NSP)
  - RoBERTa-base: 48.2 / 2.4 → 57.3 / 17.6 (with NSP)
  - BERT-large: 48.5 / 3.6 → 51.9 / 11.1

  — [arXiv HTML](https://arxiv.org/html/2502.07717v1)
- Negated-NLI subsets: BERT-large RTE-negated 53.4 → 78.9 and RoBERTa-base 58.7 → 81.0, but RoBERTa-large *drops* with NSPP (84.7 → 81.1). Results on the original, non-negated dev sets change little. — [arXiv HTML](https://arxiv.org/html/2502.07717v1)
- Note: this paper's RoBERTa-large off-the-shelf CondaQA score (64.9) is far above the CondaQA paper's 54.08 for fine-tuned RoBERTa-large. The fine-tuning setup differs and the gap is not explained. — [arXiv HTML](https://arxiv.org/html/2502.07717v1); [ar5iv CondaQA](https://ar5iv.labs.arxiv.org/html/2211.00295)
- Affirmative interpretations (Rezaei & Blanco, ACL 2024 short): RoBERTa-large CondaQA rises from 54.1 to 67.1 when it is trained with generated paraphrases of negated sentences that contain no negation. This is from a search summary, not verified in the full text. — [ACL Anthology](https://aclanthology.org/2024.acl-short.55); [arXiv](https://arxiv.org/html/2406.07492v1)
- Affixal negation cues ("un-", "-less") in NSPP pretraining, arXiv 2609.13685 (Sep 2026): RoBERTa-large CondaQA is reported at 54.1 → 68.0. This also comes from a search snippet; the full text and venue were not verified. — [arXiv](https://arxiv.org/pdf/2609.13685)

### Inferences
- In CondaQA, fine-tuned encoders of LIDM's size (300 to 700M) are close to majority, while seq2seq QA models pretrained on many QA formats reach 66 to 73. For LIDM, multi-format QA pre-training data and negation-aware further pretraining matter more than the head. NSPP or NSP on Wikipedia is about 1M pairs and fits a 12 GB GPU easily.
- Because NSPP hurts RoBERTa-large on RTE-negated, any negation pretraining should be checked against ANLI/MNLI regressions.

### Gaps
- NegBench (2025, negation in vision–language models) and "This is not a dataset" (2023) were not fetched. Their results are about CLIP-style or LLM models, not text encoders.
- No source reports negation results for ModernBERT or for split-encoder (bi-encoder plus head) architectures.
- CondaQA zero-shot results for fine-tuned NLI encoders (the LIDM setting) are not in the CondaQA paper.
