"""Selective prediction on the typed-decisions test: answer only when confidence >= a threshold. Per run: coverage and
accuracy at fixed thresholds, accuracy at fixed coverage, AUROC of confidence for correctness, and a threshold chosen
on dev for a target accuracy then applied to test. `--calibrate` first applies one temperature per question type fitted
on that run's dev predictions (<run>-dev.json, from eval_typed_decisions.py --test data/td/dev.jsonl).

    uv run python scripts/selective_typed_decisions.py s2c-lidm s2c-laya --calibrate
"""
import argparse
import json

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import roc_auc_score

from cls.tracking import ROOT
from cls.train import read_jsonl

D = ROOT / "artifacts" / "typed_decisions"
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95])
ap.add_argument("--targets", type=float, nargs="+", default=[0.9, 0.95])
a = ap.parse_args()
TYPE = {(it["case"], it["qid"]): it["type"] for s in ("dev", "test") for it in read_jsonl(ROOT / "data" / "td" / f"{s}.jsonl")}


def load(name, temps=None):
    preds = json.loads((D / f"{name}.json").read_text())["predictions"]
    conf, corr, types = [], [], []
    for p in preds:
        t = TYPE[(p["case"], p["qid"])]
        lp = np.log(np.clip(p["probs"], 1e-9, 1)) / (temps[t] if temps else 1.0)
        q = np.exp(lp - lp.max())
        q /= q.sum()
        conf.append(q.max())
        corr.append(q.argmax() == p["gold"])
        types.append(t)
    return np.array(conf), np.array(corr, float), np.array(types), preds


def fit_temps(name):
    preds = json.loads((D / f"{name}-dev.json").read_text())["predictions"]
    temps = {}
    for t in ("choice", "noul", "score"):
        P = [(np.log(np.clip(p["probs"], 1e-9, 1)), p["gold"]) for p in preds if TYPE[(p["case"], p["qid"])] == t]
        temps[t] = minimize_scalar(lambda s, P=P: -np.mean([(lp / s)[g] - np.log(np.exp(lp / s).sum()) for lp, g in P]),
                                   bounds=(0.2, 5), method="bounded").x
    return temps


for run in a.runs:
    temps = fit_temps(run) if a.calibrate else None
    conf, corr, _, _ = load(run, temps)
    row = {"run": run + (" (calibrated)" if temps else ""), "acc": round(corr.mean(), 3),
           "auroc": round(roc_auc_score(corr, conf), 3)}
    for th in a.thresholds:
        m = conf >= th
        row[f">={th}"] = f"{m.mean():.0%} cov, {corr[m].mean():.3f} acc" if m.any() else "0% cov"
    order = np.argsort(-conf)
    for c in (0.5, 0.8):
        k = int(c * len(conf))
        row[f"acc@{c:.0%}cov"] = round(corr[order[:k]].mean(), 3)
    if (D / f"{run}-dev.json").exists():
        dconf, dcorr, _, _ = load(f"{run}-dev", temps)
        for tgt in a.targets:  # lowest dev threshold whose answered set reaches the target accuracy
            ths = sorted(set(np.round(dconf, 4)))
            th = next((t for t in ths if dcorr[dconf >= t].mean() >= tgt and (dconf >= t).sum() >= 30), None)
            if th is None:
                row[f"target {tgt}"] = "not reachable on dev"
                continue
            m = conf >= th
            row[f"target {tgt}"] = f"th {th:.3f}: test {m.mean():.0%} cov, {corr[m].mean():.3f} acc"
    print(json.dumps(row))
