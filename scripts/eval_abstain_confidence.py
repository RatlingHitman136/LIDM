"""Abstain by confidence instead of an abstain option (BACKLOG item 8). CondaQA, ANLI, VitaminC: score only the real
answers (yes / no; entailment / contradiction; supports / refutes) and abstain when the top probability is below a
threshold fitted on the task's own dev split (accuracy, abstaining counted right when the gold answer is the abstain
option). Compared on the same test questions as the as-evaluated, option-based scoring.

    uv run python scripts/eval_abstain_confidence.py --lidm checkpoints/s2x-lidm/last.pt --laya checkpoints/s2-laya
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from cls.data import context_suite as cs
from cls.data.typed import _load
from cls.eval import harness
from cls.tracking import ROOT

ABSTAIN = {"condaqa": 2, "anli": 1, "vitaminc": 2}  # index of the abstain option in each task's options

ap = argparse.ArgumentParser()
ap.add_argument("--lidm", nargs="*", default=[])
ap.add_argument("--laya", nargs="*", default=[])
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()

rng = random.Random(0)
test = cs.load_suite(n=a.n, tasks=list(ABSTAIN))
dev = {"condaqa": [it for it in map(cs.condaqa_item, _load("lasha-nlp/CONDAQA", None, "validation")) if it],
       "anli": [cs.nli_item("anli", r["premise"], r["hypothesis"], r["label"])
                for rd in ("dev_r1", "dev_r2", "dev_r3") for r in _load("facebook/anli", "plain_text", rd)],
       "vitaminc": [cs.vitaminc_item(r) for r in _load("tals/vitaminc", None, "validation")]}
dev = {t: rng.sample(v, min(a.n, len(v))) for t, v in dev.items()}


def drop(items, ab):  # the same questions without the abstain option
    return [dict(it, criteria=[c for i, c in enumerate(it["criteria"]) if i != ab]) for it in items]


def scores(run, items, ab):
    """Accuracy of: options as evaluated; real answers + abstain below a dev-fitted threshold."""
    full = run(items)
    part = run(drop(items, ab))
    gold = np.array([it["gold"] for it in items])
    real = [i for i in range(len(items[0]["criteria"])) if i != ab]
    pred_real = np.array([real[int(np.argmax(p))] for p in part])
    conf = np.array([float(np.max(p)) for p in part])
    return full, pred_real, conf, gold


def acc_at(pred_real, conf, gold, ab, t):
    pred = np.where(conf < t, ab, pred_real)
    return float(np.mean(pred == gold)), float(np.mean(pred == ab))


results = {}
for kind, ckpts in (("lidm", a.lidm), ("laya", a.laya)):
    for ckpt in ckpts:
        if kind == "lidm":
            from cls.train import encode, load_model

            enc, m, prior, _ = load_model(ckpt)

            @torch.no_grad()
            def run(items, enc=enc, m=m, prior=prior):
                out = []
                for i in range(0, len(items), 16):
                    inp, _, _, cos = encode(enc, items[i:i + 16], prior)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        out += [r.float().softmax(-1).cpu().numpy() for r in m(*inp, prior=cos)]
                return [p[: len(it["criteria"])] for p, it in zip(out, items)]
        else:
            import laya

            agent = laya.load(str(Path(ckpt).resolve()) if Path(ckpt).exists() else ckpt)
            pred = harness.laya_predictor(agent)
            run = lambda items, pred=pred: [pred(it) for it in items]
        name = ckpt.rstrip("/").split("/")[-2 if ckpt.endswith(".pt") else -1]
        results[name] = {}
        for task, ab in ABSTAIN.items():
            _, d_pred, d_conf, d_gold = scores(run, dev[task], ab)
            grid = np.unique(np.quantile(d_conf, np.linspace(0, 1, 101)))
            t = max(grid, key=lambda x: acc_at(d_pred, d_conf, d_gold, ab, x)[0])
            full, pred_real, conf, gold = scores(run, test[task], ab)
            opt_acc = float(np.mean(np.array([int(np.argmax(p)) for p in full]) == gold))
            c_acc, c_rate = acc_at(pred_real, conf, gold, ab, t)
            row = {"threshold": round(float(t), 3), "as_evaluated": round(opt_acc, 4), "by_confidence": round(c_acc, 4),
                   "abstain_rate": round(c_rate, 3), "gold_abstain_rate": round(float(np.mean(gold == ab)), 3),
                   "real_answers_only_acc": round(float(np.mean(pred_real[gold != ab] == gold[gold != ab])), 4)}
            results[name][task] = row
            print(json.dumps({"model": name, "task": task, **row}), flush=True)
        torch.cuda.empty_cache()
(ROOT / "artifacts" / "abstain_confidence.json").write_text(json.dumps(results, indent=2))
