"""Banking77 test with label descriptions as options (src/cls/data/banking77_descriptions.json, written from the
label names only). Option formats: `name` (as in M0 and every run so far), `desc`, and `name: desc`.
A separate benchmark condition: not comparable with names-only Banking77 numbers (M0 table, published results).

For LIDM checkpoints: head alone, retrieval term alone, combined at the trained weight and at the weight saved in
<run>/fit.pt (scripts/fit_prior_weight.py). For --biencoder models: plain cosine, as M0's B5b.

    uv run python scripts/eval_banking77_descriptions.py --ckpt checkpoints/stage1b-gte-lora/last.pt \\
        --biencoder Alibaba-NLP/gte-modernbert-base BAAI/bge-base-en-v1.5 [--n 3080]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cls.data.banking77 import load_banking77
from cls.data.typed import banking77_items
from cls.model import Encoder
from cls.train import encode, load_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", nargs="*", default=[])
ap.add_argument("--biencoder", nargs="*", default=[])
ap.add_argument("--n", type=int, default=None, help="test items (default all 3,080)")
a = ap.parse_args()

desc_by_id = json.loads((Path(__file__).parents[1] / "src/cls/data/banking77_descriptions.json").read_text())
b77 = load_banking77("test", n=5)
name_of = dict(zip(b77.descriptions, b77.label_ids))  # humanized name -> label id
FORMATS = {"name": lambda nm: nm, "desc": lambda nm: desc_by_id[name_of[nm]],
           "name: desc": lambda nm: f"{nm}: {desc_by_id[name_of[nm]]}"}
sets = {f"N{k or 77}": banking77_items(k, n=a.n, seed=0) for k in (None, 20)}


def with_format(items, fmt):
    return [dict(it, criteria=[FORMATS[fmt](c) for c in it["criteria"]]) for it in items]


@torch.no_grad()
def lidm_scores(enc, m, prior, items):
    """(head, retrieval) logits over the real options of each question."""
    s, rows = m.dual_scale.item(), []
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        inputs, _, _, cos = encode(enc, chunk, prior)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.dual_scale.fill_(0)
            head = m(*inputs, prior=cos).float()
            m.dual_scale.fill_(1)
            ret = m(*inputs, prior=cos).float() - head
            m.dual_scale.fill_(s)
        for k, it in enumerate(chunk):
            n = len(it["criteria"])  # real options only: padded slots are -inf, and -inf - -inf is NaN
            rows.append((head[k, :n].cpu().numpy(), ret[k, :n].cpu().numpy(), it["gold"]))
    return rows


def acc(rows, f):
    return round(float(np.mean([f(h, c).argmax() == g for h, c, g in rows])), 4)


for path in a.ckpt:
    enc, m, prior, st = load_model(path)
    s_tr = m.dual_scale.item()
    fit = Path(path).parent / "fit.pt"
    s_fit = torch.load(fit, map_location="cpu")["config"]["prior_weight_fit"]["s"] if fit.exists() else None
    for key, items in sets.items():
        for fmt in FORMATS:
            rows = lidm_scores(enc, m, prior, with_format(items, fmt))
            res = {"head": acc(rows, lambda h, c: h), "retrieval": acc(rows, lambda h, c: c),
                   f"combined s={s_tr:.1f}": acc(rows, lambda h, c, s=s_tr: h + s * c)}
            if s_fit is not None:
                res[f"combined s={s_fit:g} (fit)"] = acc(rows, lambda h, c, s=s_fit: h + s * c)
            print(json.dumps({"model": path, "set": key, "options": fmt, **res}), flush=True)
    del enc, m
    torch.cuda.empty_cache()

for name in a.biencoder:
    enc = Encoder(name)
    for key, items in sets.items():
        for fmt in FORMATS:
            its, ok, cache = with_format(items, fmt), 0, {}
            for i in range(0, len(its), 64):
                chunk = its[i:i + 64]
                q = enc.embed([it["state"] for it in chunk])
                for it, v in zip(chunk, q):
                    opts = tuple(it["criteria"])
                    if opts not in cache:
                        cache[opts] = enc.embed(list(opts))
                    ok += int((cache[opts] @ v).argmax() == it["gold"])
            print(json.dumps({"model": name, "set": key, "options": fmt, "biencoder": round(ok / len(its), 4)}), flush=True)
    del enc
    torch.cuda.empty_cache()
