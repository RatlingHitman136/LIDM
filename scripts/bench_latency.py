"""Batch-1 latency of an LIDM checkpoint on Banking77 (N=77), as GOALS.md defines it: per question, fresh label set
vs cached options. Variants: (a) options encoded every call, LoRA unmerged (what eval_lidm.py measures);
(b) the same with the encoder LoRA merged into its weights; (c) merged, options encoded once and reused.
Also a per-stage breakdown and a check that (b) and (c) predict the same as (a).

    uv run python scripts/bench_latency.py checkpoints/stage1-Blora/last.pt
"""
import argparse
import time

import numpy as np
import torch

from cls.data.typed import banking77_items
from cls.train import load_model, make_batch

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--calls", type=int, default=100)
ap.add_argument("--check", type=int, default=300, help="questions for the same-prediction check")
a = ap.parse_args()

enc, m, prior, st = load_model(a.ckpt)
c = st["config"]
assert c.get("option_attn", "none") == "none", "bench_latency times encode_a/encode_options; LIDM-X/DJ go through encode_x"
assert prior is None, "external retrieval priors are not timed here"
items = banking77_items(None, n=max(a.calls + 10, a.check), seed=0)
labels = items[0]["criteria"]
index = {t: i for i, t in enumerate(labels)}


def ms(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t) * 1000


@torch.no_grad()
def call(item, cache=None):
    """One question. Returns (prediction, ms for pass A, ms for pass B, ms for head + scoring)."""
    ha, ta = ms(lambda: enc.encode_a([item["state"]], [item["instructions"]]))
    (opt_h, opt_m), tb = (cache, 0.0) if cache else ms(lambda: enc.encode_options(labels, enc.marker))
    oi, om, _, _ = make_batch([item], index, "cuda")

    def head():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(*ha, opt_h, opt_m, oi, om).argmax(-1).item()
    pred, th = ms(head)
    return pred, ta, tb, th


def bench(name, cache=None):
    rows = [call(it, cache) for it in items[: a.calls + 10]][10:]  # 10 warm-up calls
    tot = np.array([r[1] + r[2] + r[3] for r in rows])
    part = np.array([r[1:] for r in rows]).mean(0)
    print(f"{name:42s} p50 {np.percentile(tot, 50):6.1f} ms  p95 {np.percentile(tot, 95):6.1f} ms | "
          f"pass A {part[0]:5.1f}  pass B {part[1]:5.1f}  head {part[2]:5.1f} ms", flush=True)


def preds(cache=None):
    return [call(it, cache)[0] for it in items[: a.check]]


bench("(a) fresh options, LoRA unmerged")
ref = preds()
if enc.lora_rank:
    enc.model = enc.model.merge_and_unload()  # LoRA folded into the encoder weights: no extra matmuls per layer
    enc.lora_rank = 0
bench("(b) fresh options, LoRA merged")
same_b = np.mean(np.array(preds()) == np.array(ref))
cache = enc.encode_options(labels, enc.marker)  # 77 options encoded once
bench("(c) cached options, LoRA merged", cache)
same_c = np.mean(np.array(preds(cache)) == np.array(ref))
print(f"same prediction as (a) on {a.check} questions: (b) {same_b:.3f}, (c) {same_c:.3f}; "
      f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
