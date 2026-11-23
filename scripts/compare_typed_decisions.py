"""Typed-decisions results from artifacts/typed_decisions/*.json (written by eval_typed_decisions.py).

    uv run python scripts/compare_typed_decisions.py A B     # paired: accuracy A - B, 95% CI by resampling cases, McNemar
    uv run python scripts/compare_typed_decisions.py --table # every run; leave-one-workflow-out runs by held-out workflow
"""
import argparse
import json

import numpy as np
from scipy.stats import binomtest

from cls.tracking import ROOT

D = ROOT / "artifacts" / "typed_decisions"
ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*")
ap.add_argument("--table", action="store_true")
a = ap.parse_args()


def load(name):
    return json.loads((D / f"{name}.json").read_text())


if a.table:
    cols = ["accuracy", "acc_ci_lo", "acc_ci_hi", "acc_choice", "acc_noul", "acc_score", "brier_soft", "ece",
            "score_mae", "latency_ms_per_case_p50"]
    runs = sorted(p.stem for p in D.glob("*.json"))
    print("| run | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for n in (r for r in runs if "-lowo-" not in r):
        m = load(n)["metrics"]
        print(f"| {n} | " + " | ".join(f"{m.get(c, float('nan')):.3f}" for c in cols) + " |")
    lowo = [r for r in runs if "-lowo-" in r]
    if lowo:
        print("\nLeave one workflow out: accuracy on the held-out workflow (test split)\n")
        print("| run | held-out workflow | accuracy | mean of the 3 trained workflows |\n|---|---|---|---|")
        for n in lowo:
            m, wf = load(n)["metrics"], n.split("-lowo-")[1]
            seen = [v for k, v in m.items() if k.startswith("acc_") and k[4:] not in (wf, "choice", "noul", "score", "ci_lo", "ci_hi")]
            print(f"| {n} | {wf} | {m[f'acc_{wf}']:.3f} | {np.mean(seen):.3f} |")
    raise SystemExit

A, B = (load(n)["predictions"] for n in a.runs)
key = {(p["case"], p["qid"]): p for p in B}
ca = np.array([np.argmax(p["probs"]) == p["gold"] for p in A], dtype=float)
cb = np.array([np.argmax(key[(p["case"], p["qid"])]["probs"]) == p["gold"] for p in A], dtype=float)
cases = sorted({p["case"] for p in A})
idx = np.array([cases.index(p["case"]) for p in A])
diff = np.bincount(idx, ca - cb) / np.bincount(idx)  # per-case accuracy difference
rng = np.random.default_rng(0)
boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(4000)]
only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
p = binomtest(only_a, only_a + only_b).pvalue if only_a + only_b else 1.0
print(json.dumps({"a": a.runs[0], "b": a.runs[1], "acc_a": round(ca.mean(), 4), "acc_b": round(cb.mean(), 4),
                  "diff": round(ca.mean() - cb.mean(), 4), "ci_lo": round(np.percentile(boot, 2.5), 4),
                  "ci_hi": round(np.percentile(boot, 97.5), 4), "only_a_right": only_a, "only_b_right": only_b,
                  "mcnemar_p": round(p, 4)}))
