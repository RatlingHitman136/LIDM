"""LIDM-X decision table (reports/Recovering interaction after late splits.md): stage-2 LIDM-X vs stage-2 LIDM and
Laya on the rows LIDM loses today, the guardrails and the rest of the context suite. Paired 95% CIs (resampling
questions) where per-question results exist.

    uv run python scripts/summarize_lidmx.py [--x s2x-lidm] [--lidm s2-lidm] [--laya s2-laya]
"""
import argparse
import json

import numpy as np

from cls.tracking import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--x", default="s2x-lidm")
ap.add_argument("--lidm", default="s2-lidm")
ap.add_argument("--laya", default="s2-laya")
ap.add_argument("--x2c", help="the 2c run of --x (fine-tuned on typed decisions), e.g. s2xc-lidm; guardrail skipped if absent")
a = ap.parse_args()
A = ROOT / "artifacts"


def get(kind, run, task):
    f = A / kind / f"{run}.json"
    if not f.exists():
        return None
    t = json.loads(f.read_text())["tasks"].get(task)
    return t


def nopt_slice(run):  # held-out label sets, questions with 13-40 options
    t = get("label_suite", run, "eval_heldout_sources")
    if not t:
        return None
    from cls.train import read_jsonl
    n = np.array([len(it.get("criteria") or [0, 0]) for it in read_jsonl(ROOT / "data/stage2/eval_heldout_sources.jsonl")])
    c = np.array(t["correct"])[(n >= 13) & (n <= 40)]
    return {"accuracy": float(c.mean()), "correct": c.tolist()}


ROWS = [("VitaminC", "context_suite", "vitaminc"), ("RACE", "context_suite", "race"),
        ("Banking77, 77 names", "label_suite", "eval_banking77_names"), ("13-40 options", None, None),
        ("Banking77, 77 descriptions", "label_suite", "eval_banking77_desc"),
        ("QuAIL", "context_suite", "quail"), ("When2Call", "context_suite", "when2call"), ("BoolQ", "context_suite", "boolq"),
        ("PubMedQA", "context_suite", "pubmedqa"), ("SQuAD 2.0", "context_suite", "squad2_answerable"),
        ("CondaQA", "context_suite", "condaqa"), ("ANLI", "context_suite", "anli"),
        ("held-out label sets", "label_suite", "eval_heldout_sources")]
rng = np.random.default_rng(0)


def diff(r1, r2):
    if not (r1 and r2):
        return "-"
    d = r1["accuracy"] - r2["accuracy"]
    if "correct" in r1 and "correct" in r2 and len(r1["correct"]) == len(r2["correct"]):
        x = np.array(r1["correct"]) - np.array(r2["correct"])
        b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        return f"{100 * d:+.1f} [{100 * np.percentile(b, 2.5):+.1f}, {100 * np.percentile(b, 97.5):+.1f}]"
    return f"{100 * d:+.1f}"


print(f"| Task | {a.laya} | {a.lidm} | {a.x} | X − Laya | X − LIDM |\n|---|---|---|---|---|---|")
gap_now, gap_x = [], []
for i, (label, kind, task) in enumerate(ROWS):
    r = [nopt_slice(m) if kind is None else get(kind, m, task) for m in (a.laya, a.lidm, a.x)]
    acc = [f"{x['accuracy']:.3f}" if x else "-" for x in r]
    print(f"| {label} | " + " | ".join(acc) + f" | {diff(r[2], r[0])} | {diff(r[2], r[1])} |")
    if i < 4 and all(r):
        gap_now.append(r[1]["accuracy"] - r[0]["accuracy"])
        gap_x.append(r[2]["accuracy"] - r[0]["accuracy"])
td = {m: json.loads(f.read_text())["metrics"]["accuracy"] for m in (a.laya, a.lidm, a.x, "s2c-laya", a.x2c or "")
      if (f := A / "typed_decisions" / f"{m}.json").exists()}
print("\ntyped decisions (unseen schemas, stage 2):", {k: round(v, 4) for k, v in td.items()})
if len(gap_x) == 4:
    print(f"\nprimary metric, mean X − Laya over VitaminC, RACE, Banking77 names, 13-40 options: {100 * np.mean(gap_x):+.1f} "
          f"(stage-2 LIDM: {100 * np.mean(gap_now):+.1f}); adopt LIDM-X if <= {50 * np.mean(gap_now):+.1f} (half the gap)")
b_desc = [get("label_suite", m, "eval_banking77_desc") for m in (a.laya, a.x)]
if all(b_desc):
    print(f"guardrail, Banking77 descriptions X − Laya >= +5: {100 * (b_desc[1]['accuracy'] - b_desc[0]['accuracy']):+.1f}")
xc, lc = A / "typed_decisions" / f"{a.x2c}.json", A / "typed_decisions" / "s2c-laya.json"
if a.x2c and xc.exists() and lc.exists():
    d = json.loads(xc.read_text())["metrics"]["accuracy"] - json.loads(lc.read_text())["metrics"]["accuracy"]
    print(f"guardrail, typed decisions after 2c X − Laya within 1.5 points: {100 * d:+.1f}")
