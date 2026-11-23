"""Kill-criterion evaluation on the typed-decisions test (400 cases, 2,000 questions), for LIDM and Laya alike.
One call per case (all 5 questions of a state at once, each model's natural call), which is also the latency unit.

Metrics: accuracy (overall, per question type, per workflow; 95% CI by resampling whole cases), Brier score against
the benchmark's soft targets, ECE, and score MAE (expected level vs the gold expected level). Per-question
probabilities go to artifacts/typed_decisions/<name>.json for paired comparisons; metrics also to MLflow.

    uv run python scripts/eval_typed_decisions.py lidm --ckpt checkpoints/td-lidm-mbl-lr1e-4/best.pt --name td-lidm
    uv run python scripts/eval_typed_decisions.py laya --ckpt checkpoints/td-laya --name td-laya
"""
import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np
import torch

from cls.data.typed import _load
from cls.metrics import ece
from cls.tracking import ROOT, Tracker
from cls.train import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("model", choices=["lidm", "laya"])
ap.add_argument("--ckpt", required=True, help="lidm: .pt; laya: hub id or local dir")
ap.add_argument("--name", required=True)
ap.add_argument("--test", default=str(ROOT / "data" / "td" / "test.jsonl"))
ap.add_argument("--no-track", action="store_true")
a = ap.parse_args()

items = read_jsonl(a.test)
cases = collections.defaultdict(list)
for it in items:
    cases[it["case"]].append(it)

if a.model == "lidm":
    from cls.train import encode, load_model

    enc, m, prior, st = load_model(a.ckpt)

    @torch.no_grad()
    def predict_case(case_items):
        inputs, _, _, cos = encode(enc, case_items, prior)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(*inputs, prior=cos).float()
        n = [len(it["criteria"]) if it["type"] != "noul" else 2 for it in case_items]
        return [logits[i, :k].softmax(-1).cpu().numpy() for i, k in enumerate(n)]
else:
    import laya

    agent = laya.load(str(Path(a.ckpt).resolve()) if Path(a.ckpt).exists() else a.ckpt)  # relative path = hub id
    raw = {r["id"]: json.loads(r["questions"]) for sp in ("train", "test")
           for r in _load("LocalLLaMA/typed-decisions", "all", sp)}  # dev cases come from the train split

    def predict_case(case_items):
        ans = agent.system_one(case_items[0]["state"], raw[case_items[0]["case"]])["answers"]
        out = []
        for it in case_items:
            r = ans[it["qid"]]
            if it["type"] == "noul":
                out.append(np.array([1 - float(r["noul"]), float(r["noul"])]))
            else:
                keys = it["names"] if it["type"] == "choice" else [str(i) for i in range(len(it["criteria"]))]
                out.append(np.array([r["probabilities"].get(k, 0.0) for k in keys]))
        return out

probs, lat = {}, []
for i, (cid, cit) in enumerate(cases.items()):
    torch.cuda.synchronize()
    t = time.perf_counter()
    ps = predict_case(cit)
    torch.cuda.synchronize()
    if i >= 5:  # warm-up
        lat.append((time.perf_counter() - t) * 1000)
    for it, p in zip(cit, ps):
        probs[(cid, it["qid"])] = p / p.sum()

P = [probs[(it["case"], it["qid"])] for it in items]
correct = np.array([p.argmax() == it["gold"] for p, it in zip(P, items)], dtype=float)
conf = np.array([p.max() for p in P])
idx = {c: i for i, c in enumerate(cases)}
case_of = np.array([idx[it["case"]] for it in items])
per_case = np.bincount(case_of, correct) / np.bincount(case_of)
rng = np.random.default_rng(0)
boot = [per_case[rng.integers(0, len(per_case), len(per_case))].mean() for _ in range(2000)]
sc = [(p, it) for p, it in zip(P, items) if it["type"] == "score"]
met = {"accuracy": correct.mean(), "acc_ci_lo": np.percentile(boot, 2.5), "acc_ci_hi": np.percentile(boot, 97.5),
       "brier_soft": np.mean([((p - np.array(it["soft"])) ** 2).sum() for p, it in zip(P, items)]),
       "ece": ece(conf, correct),
       "score_mae": np.mean([abs((p * np.arange(len(p))).sum() - it["gold_score"]) for p, it in sc]),
       "latency_ms_per_case_p50": np.percentile(lat, 50), "latency_ms_per_case_p95": np.percentile(lat, 95)}
for key in ("type", "workflow"):
    for v in sorted({it[key] for it in items}):
        mask = np.array([it[key] == v for it in items])
        met[f"acc_{v}"] = correct[mask].mean()
met = {k: round(float(v), 4) for k, v in met.items()}

with Tracker("typed_decisions", a.name, tags={"model": a.model, "stage": "eval"}, enabled=not a.no_track) as tr:
    tr.log_params({"checkpoint": a.ckpt, "test": a.test, "n_questions": len(items), "n_cases": len(cases)})
    tr.log_metrics(met)
out = ROOT / "artifacts" / "typed_decisions"
out.mkdir(parents=True, exist_ok=True)
(out / f"{a.name}.json").write_text(json.dumps({"checkpoint": a.ckpt, "metrics": met, "predictions": [
    {"case": it["case"], "qid": it["qid"], "gold": it["gold"], "probs": [round(float(x), 5) for x in p]}
    for p, it in zip(P, items)]}))
print(json.dumps({"name": a.name, **met}), flush=True)
