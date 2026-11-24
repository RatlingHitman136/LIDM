"""How well a model's probabilities match human vote distributions (stage 3a's target), on data/stage3/eval_*.jsonl:
held-out GoEmotions (P(emotion) vs share of raters), held-out DICES (P(unsafe) vs share), ChaosNLI SNLI/MNLI (3-way
vs 100 votes). Per set: accuracy vs the majority label, Brier vs the vote distribution, mean absolute error of the
positive probability (noul) and Jensen-Shannon divergence (choice). Results to artifacts/soft_labels/<name>.json.

    uv run python scripts/eval_soft_labels.py lidm --ckpt checkpoints/s3a-lidm/last.pt --name s3a-lidm [--temps ...]
    uv run python scripts/eval_soft_labels.py laya --ckpt checkpoints/s2-laya --name s2-laya
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cls import calibration
from cls.eval import harness
from cls.tracking import ROOT
from cls.train import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["lidm", "laya"])
ap.add_argument("--ckpt", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--temps")
a = ap.parse_args()
SETS = ["eval_goemotions", "eval_dices", "eval_chaosnli"]

if a.model == "lidm":
    from cls.train import encode, load_model

    enc, m, prior, _ = load_model(a.ckpt)

    @torch.no_grad()
    def predict_all(items):
        out = []
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            inputs, _, _, cos = encode(enc, chunk, prior)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = m(*inputs, prior=cos).float()
            out += [row[: 2 if it["type"] == "noul" else len(it["criteria"])].softmax(-1).cpu().numpy() for it, row in zip(chunk, logits)]
        return out
else:
    import laya

    agent = laya.load(str(Path(a.ckpt).resolve()) if Path(a.ckpt).exists() else a.ckpt)
    pred = harness.laya_predictor(agent)
    predict_all = lambda items: [pred(it) for it in items]

temps = calibration.load(a.temps) if a.temps else None


def jsd(p, q):
    mix = (p + q) / 2
    kl = lambda x, y: float(np.sum(np.where(x > 0, x * np.log(np.clip(x, 1e-12, 1) / np.clip(y, 1e-12, 1)), 0)))
    return (kl(p, mix) + kl(q, mix)) / 2


res = {}
for s in SETS:
    items = read_jsonl(ROOT / "data" / "stage3" / f"{s}.jsonl")
    P = [np.asarray(p, float) / np.sum(p) for p in predict_all(items)]
    if temps:
        P = [calibration.apply(p, it["type"], temps) for p, it in zip(P, items)]
    H = [np.asarray(it["soft"], float) for it in items]
    row = {"n": len(items), "acc_vs_majority": float(np.mean([p.argmax() == it["gold"] for p, it in zip(P, items)])),
           "brier_vs_votes": float(np.mean([((p - h) ** 2).sum() for p, h in zip(P, H)])),
           "jsd": float(np.mean([jsd(p, h) for p, h in zip(P, H)]))}
    if items[0]["type"] == "noul":
        row["mae_p_yes"] = float(np.mean([abs(p[1] - h[1]) for p, h in zip(P, H)]))
        row["corr_p_yes"] = float(np.corrcoef([p[1] for p in P], [h[1] for h in H])[0, 1])
    res[s] = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
    print(json.dumps({"name": a.name, "set": s, **res[s]}), flush=True)
out = ROOT / "artifacts" / "soft_labels"
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.name}.json").write_text(json.dumps(res, indent=2))
