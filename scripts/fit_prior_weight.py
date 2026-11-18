"""Refit only the retrieval term's weight s (logit = head + s * cos) on zero-shot validation data, after training.
Training learns s on labels the head has seen, where the head beats retrieval, so it trusts the head too much
on unseen labels (RESULTS.md, stage 1, LoRA + bge). Picks s by mean accuracy over the `zs/` val sets and saves
<ckpt dir>/fit.pt with that s; nothing else changes.

    uv run python scripts/fit_prior_weight.py checkpoints/stage1-Bbge-from-L1/last.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from cls.tracking import ROOT
from cls.train import encode, load_model, read_jsonl

GRID = [0, 1, 2, 5, 10, 20, 40, 80, 160, 320, 1e4]  # 1e4: retrieval alone, the head only breaks ties

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--val", default=str(ROOT / "data" / "stage1_val.jsonl"))
a = ap.parse_args()

enc, m, prior, st = load_model(a.ckpt)
assert m.dual_scale is not None, "checkpoint has no retrieval term"
val = [v for v in read_jsonl(a.val) if v["task"].startswith("zs/")]
rows = []  # (task, head logits, cosine, gold)
with torch.no_grad():
    for s in range(0, len(val), 16):
        chunk = val[s:s + 16]
        inputs, _, _, cos = encode(enc, chunk, prior)
        if cos is None:  # in-model encoder cosine: recover it as (logits with s=1) - (logits with s=0)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                saved = m.dual_scale.item()
                m.dual_scale.fill_(0)
                head = m(*inputs).float()
                m.dual_scale.fill_(1)
                cos = m(*inputs).float() - head
                m.dual_scale.fill_(saved)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                head = m(*inputs, prior=torch.zeros_like(cos)).float()
        for i, it in enumerate(chunk):
            n = len(it["criteria"])
            rows.append((it["task"], head[i, :n].cpu().numpy(), cos[i, :n].float().cpu().numpy(), it["gold"]))

tasks = sorted({t for t, *_ in rows})
table = {g: {t: np.mean([(h + g * c).argmax() == y for tt, h, c, y in rows if tt == t]) for t in tasks} for g in GRID}
print(f"{'s':>8}" + "".join(f"{t[3:][:14]:>15}" for t in tasks) + f"{'zs mean':>9}")
for g in GRID:
    print(f"{g:8g}" + "".join(f"{table[g][t]:15.3f}" for t in tasks) + f"{np.mean(list(table[g].values())):9.3f}")
best = max(GRID, key=lambda g: np.mean(list(table[g].values())))
trained = m.dual_scale.item()
print(f"trained s = {trained:.2f}; best s on zero-shot val = {best:g}")
with torch.no_grad():
    m.dual_scale.fill_(best)
st["model"] = m.state_dict()
st["config"]["prior_weight_fit"] = {"s": best, "fit_on": "zs/ val sets", "trained_s": trained}
out = Path(a.ckpt).parent / "fit.pt"
torch.save(st, out)
print(f"saved {out}")
