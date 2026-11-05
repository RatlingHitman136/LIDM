# Handoff (2026-11-01): Jev/Laya-style decision model, late-interaction variant

(verified) = from model cards/papers. (my analysis) = unverified hypothesis.

## 1. Target
Jev (TypeSafe AI, early access 2026-09-15): state + typed questions -> typed calibrated answers, no text
generation. Proprietary, hosted only. Primitives: `choice` (up to 255 options), `score`, `noul`. (verified)

## 2. Open alternatives (verified, Sept 2026)
Laya (Convai, 421M en / 322M multilingual, Apache-2.0, encoder, closest to Jev); Kev (0.8/4/9B, Qwen3.5+LoRA+
pointer head, implements /v1/systemone); Nimble (Bespoke Labs, 9B); jevos (MIT, CPU, hosted Jev scored higher
on all 6 tasks); others unexamined. Open-Jev: CC0 ~79k training set.

## 3. Laya (verified unless marked)
Checkpoints: `convaiinnovations/laya` (ModernBERT-large, ctx 512), `laya-multilingual` (mmBERT-base, ctx 1024),
`laya-typed-decisions` (ModernBERT-large, ctx 1024).
Pipeline: router -> one packed sequence (state+question+options, one [MASK] per option; template unpublished)
-> bidirectional encoder (395M) -> from-scratch decision head (~26M: 2 transformer layers, option-marker scorer,
act/escalate head; wiring undocumented) -> softmax over options per question. Per-(type, option count)
temperature calibration (ECE 0.466 -> 0.081). Training: RLCD (noisy logits, proper scoring rules, GRPO-style).
Budgets: `max_len` 512, `head_max_len` 192 (256 multilingual); settings, not encoder limits (ModernBERT 8192).
Limits: >~20 options collapses (Banking77 0.425 vs Jev 0.870); base ckpts near chance zero-shot (0.362,
0.766 after fine-tune); ordinal score weak (SST-5 0.372); over-confident pre-calibration; English ckpt fails on
non-Latin scripts; ~33-40 ms/question on T4.
Independent Jev evals (arXiv 2609.37647, 2610.17820): well calibrated on 37 datasets; adversarial study found
27% benign flagged, 62% phishing detected.

## 4. Proposed design (my analysis)
A. Late interaction: same encoder, one pass for text+question, one batched pass for options; combine only in
head. Pros: per-option token budget, cacheable option vectors, slow cost growth. Risk: shallow text-option
interaction (negation/conditions) -> deeper head or cross-attention. Add role signal. Prior art: bi-encoder
GLiNER/GLiClass, ColBERT.
B. Looped head: 3 shared layers x2 (block 1,2,3,1,2,3 vs layer-wise 1,1,2,2,3,3); loop-step embedding.
~12.6M params/layer at d=1024 -> ~38M. Saves params, not latency. Prior art: ALBERT, Universal Transformer.
Shapes: text B x L x d; options N x L_opt x d; head: options as queries, text as K/V, self-attn among options,
looped; scorer N x d -> N logits -> softmax.

## 7. Head sketch (UNTESTED)
```python
import torch, torch.nn as nn

class LoopedCrossHead(nn.Module):
    def __init__(self, d=1024, n_layers=3, loops=2, heads=16):
        super().__init__()
        self.loops = loops
        self.cross = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(n_layers)])
        self.self_ = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(n_layers)])
        self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d)) for _ in range(n_layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(3 * n_layers)])
        self.loop_emb = nn.Embedding(loops, d)
        self.scorer = nn.Linear(d, 1)

    def forward(self, opt, text, text_pad_mask, opt_pad_mask):
        # opt: B x N x d (pooled option vectors), text: B x L x d
        for k in range(self.loops):
            opt = opt + self.loop_emb.weight[k]
            for i in range(len(self.cross)):
                opt = self.norm[3*i](opt + self.cross[i](opt, text, text, key_padding_mask=text_pad_mask)[0])
                opt = self.norm[3*i+1](opt + self.self_[i](opt, opt, opt, key_padding_mask=opt_pad_mask)[0])
                opt = self.norm[3*i+2](opt + self.ffn[i](opt))
        logits = self.scorer(opt).squeeze(-1)
        return logits.masked_fill(opt_pad_mask, float("-inf"))
```

## 8. Open questions
Laya's template/head wiring unpublished (read github.com/NandhaKishorM/laya); act/escalate output format
undocumented; all numbers marked "my analysis" are estimates.

## 9. Sources
typesafe.ai; huggingface.co/Emerald7664/laya, aaroncool9/laya, SandLogicTechnologies/laya-GGUF;
huggingface.co/blog/sora-2/laya-ai-model-how-it-works-run-it-locally-and-eval;
datacamp.com/blog/top-open-source-jev-alternatives; github.com/feder-cr/jev/wiki/open-source-alternative-to-jev;
arxiv 2609.37647, 2610.17820; GLiClass arxiv 2508.07662.
