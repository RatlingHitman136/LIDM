# Architecture (hypotheses, not yet validated)

## Baseline: Laya (single sequence)
`[state][question][opt1 [MASK]]...[optN [MASK]]` -> bidirectional encoder -> 2-layer decision head
(~26M, wiring undocumented) -> option-marker scorer at each `[MASK]` -> softmax over options.
Exact template unpublished; read github.com/NandhaKishorM/laya before copying details.

## Proposed: late interaction + looped cross-attention head

```
 text + question ──► [shared encoder, role=TEXT]  ──► T: B x L x d ─┐
                                                                    ├─► looped head ─► N logits ─► softmax
 option_1..N     ──► [shared encoder, role=OPTION] ─► O: N x k x d ─┘   (options = queries,
   (batched, each gets its own token budget)                            text tokens = keys/values)
```

Design decisions:
1. **One shared encoder, called twice**: pass A = text + question (question must stay with the text; needed
   for `noul`), pass B = all options batched, each with its own full budget (e.g. 64-128 tokens).
   A role prefix token / segment embedding tells the encoder which pass it is on.
2. **Option vectors**: start pooled (N x d); ablate k=4 vectors and full tokens.
3. **Head**: per layer: cross-attn (options -> text tokens), self-attn among options (lets them compete),
   FFN. Scorer maps d -> 1, softmax over the N options of that question.
4. **Looped head**: 3 shared layers run twice (6 layers compute, ~3 layers params, ~38M at d=1024).
   Two orderings to test: block loop (1,2,3,1,2,3) and layer-wise (1,1,2,2,3,3). Add loop-step embedding.
   Looping saves parameters, not latency.
5. **Option caching**: for a fixed label set, pass B is computed once (100 options x 1024 x fp16 ≈ 200 KB),
   so per-request cost is the text pass + head. This is the main route to keeping latency in Laya's range.
6. **Typed answers**: `choice` = softmax over options; `noul` = binary option pair (true/false) or sigmoid;
   `score` = ordinal levels with expected value (ranked probability score loss later).
7. **Calibration**: per-(question type, option count) temperature fitted on held-out data; Laya-style
   RLCD (noisy logits + proper scoring rule) only after the architecture works.

## Known risk
Text-option interaction happens only in the head, so nuance (negation, conditions) may suffer.
Mitigations: deeper/looped head, multiple option vectors, cross-attention over full text tokens.

## Sketch
A first untested head sketch (`LoopedCrossHead`) is in docs/HANDOFF.md section 7.

## Proposed code layout (to create)
```
src/cls/{data,encoder,head,model,train,calibrate,eval}.py
configs/*.yaml     # one per ablation
scripts/           # run_baseline.py, train.py, eval.py
tests/             # shape/mask tests, loop-order tests
```
