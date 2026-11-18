"""Does the head read labels, or only correct retrieval on labels it has seen? Compares, on the Banking77 test
(unseen labels) and on a sample of training questions (seen labels): head alone, retrieval term alone, combined;
the head's confidence; re-ranking retrieval's top 5; and how much of the head's score is a fixed per-label offset.

    uv run python scripts/diagnose_head.py checkpoints/stage1-gte-lora/last.pt [--train data/stage1_train.jsonl]
"""
import argparse
import json
import random

import numpy as np
import torch

from cls.data.typed import banking77_items
from cls.train import encode, load_model

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--train", default="data/stage1_train.jsonl", help="training data the checkpoint saw")
ap.add_argument("--seen", type=int, default=1500, help="training questions to score")
a = ap.parse_args()

enc, m, prior, st = load_model(a.ckpt)
assert m.dual_scale is not None, "checkpoint has no retrieval term"
s_tr = m.dual_scale.item()


@torch.no_grad()  # the LoRA encoder must run without autograd here, or memory blows up
def collect(items):
    """Per question: head logits (retrieval weight 0) and the retrieval term (weight 1 minus weight 0)."""
    rows = []
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        inputs, _, _, cos = encode(enc, chunk, prior)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.dual_scale.fill_(0)
            head = m(*inputs, prior=cos).float()
            m.dual_scale.fill_(1)
            ret = m(*inputs, prior=cos).float() - head
            m.dual_scale.fill_(s_tr)
        for k, it in enumerate(chunk):
            n = len(it["criteria"])
            rows.append((head[k, :n].cpu().numpy(), ret[k, :n].cpu().numpy(), it["gold"]))
    return rows


def acc(rows, f):
    return float(np.mean([f(h, c).argmax() == g for h, c, g in rows]))


def rerank(rows, score):
    return float(np.mean([np.argsort(-c)[:5][np.argmax(score(h, c)[np.argsort(-c)[:5]])] == g for h, c, g in rows]))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


b77 = collect(banking77_items(None, n=None, seed=0))
with open(a.train) as f:
    train = [json.loads(line) for _, line in zip(range(40000), f)]
random.Random(1).shuffle(train)
seen = collect([t for t in train if t["type"] == "choice" and not t["task"].startswith("nli/")][:a.seen])
out = {"checkpoint": a.ckpt, "trained_s": s_tr}
for name, rows in (("banking77_unseen", b77), ("train_seen", seen)):
    out[name] = {
        "head_alone": acc(rows, lambda h, c: h), "retrieval_alone": acc(rows, lambda h, c: c),
        "combined_trained_s": acc(rows, lambda h, c: h + s_tr * c),
        "best_s_on_this_set": max((acc(rows, lambda h, c, g=g: h + g * c), g) for g in (0, 2, 5, 10, 20, 40, 80, 160, 320)),
        "head_top_prob": float(np.mean([softmax(h).max() for h, c, g in rows])),
        "rerank_top5_head": rerank(rows, lambda h, c: h), "rerank_top5_retrieval": rerank(rows, lambda h, c: c),
        "gold_in_retrieval_top5": float(np.mean([g in np.argsort(-c)[:5] for h, c, g in rows])),
    }
H = np.stack([h for h, c, g in b77])
out["banking77_unseen"]["head_per_label_offset_share"] = float(1 - ((H - H.mean(0)) ** 2).sum() / ((H - H.mean()) ** 2).sum())
print(json.dumps(out, indent=1))
