"""Does describing abstain-type options fix the abstain bias? CondaQA / ANLI / VitaminC (context-suite samples, n=1000
each) with the options as evaluated, other wordings, and `name: description` options. Per variant: accuracy, how often
the abstain option is predicted (gold rate alongside), and the paired difference to the as-evaluated variant.

    uv run python scripts/eval_abstain_descriptions.py --lidm checkpoints/s2x-lidm/last.pt checkpoints/s2xf-lidm/last.pt \\
        --laya checkpoints/s2-laya
"""
import argparse
import json

import numpy as np
import torch

from cls.data import context_suite as cs
from cls.eval import harness
from cls.tracking import ROOT

NEI = "the passage does not give enough information to answer the question"
VARIANTS = {  # task -> {variant: options}; the abstain option is always the last one
    "condaqa": {
        "as evaluated": ["yes", "no", "don't know"],
        "abstain described": ["yes", "no", f"don't know: {NEI}"],
        "all described": ["yes: the passage supports answering yes", "no: the passage supports answering no",
                          f"don't know: {NEI}"],
        "other wording": ["yes", "no", "the text does not say"],
        "other wording, described": ["yes", "no", f"the text does not say: {NEI}"],
    },
    "anli": {
        "as evaluated": ["entailment", "neutral", "contradiction"],
        "all described": ["entailment: the text shows the statement is true",
                          "neutral: the text neither shows the statement is true nor that it is false",
                          "contradiction: the text shows the statement is false"],
    },
    "vitaminc": {
        "as evaluated": ["supports the claim", "refutes the claim", "not enough information"],
        "abstain described": ["supports the claim", "refutes the claim",
                              "not enough information: the text neither supports nor refutes the claim"],
    },
}
ABSTAIN = {"condaqa": 2, "anli": 1, "vitaminc": 2}  # index of the abstain option in the original order

ap = argparse.ArgumentParser()
ap.add_argument("--lidm", nargs="*", default=[])
ap.add_argument("--laya", nargs="*", default=[])
ap.add_argument("--n", type=int, default=1000)
a = ap.parse_args()

suite = cs.load_suite(n=a.n, tasks=list(VARIANTS))


def lidm_predict(ckpt):
    from cls.train import encode, load_model

    enc, m, prior, _ = load_model(ckpt)

    @torch.no_grad()
    def run(items):
        out = []
        for i in range(0, len(items), 16):
            inp, _, _, cos = encode(enc, items[i:i + 16], prior)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out += m(*inp, prior=cos).argmax(-1).tolist()
        return np.array(out)

    return run, lambda: (enc, m)


def laya_predict(ckpt):
    from pathlib import Path

    import laya

    agent = laya.load(str(Path(ckpt).resolve()) if Path(ckpt).exists() else ckpt)
    pred = harness.laya_predictor(agent)
    return (lambda items: np.array([int(np.argmax(pred(it))) for it in items])), lambda: agent


results = {}
rng = np.random.default_rng(0)
for kind, ckpts in (("lidm", a.lidm), ("laya", a.laya)):
    for ckpt in ckpts:
        run, keep = (lidm_predict if kind == "lidm" else laya_predict)(ckpt)
        name = ckpt.rstrip("/").split("/")[-2 if ckpt.endswith(".pt") else -1]
        results[name] = {}
        for task, variants in VARIANTS.items():
            items, ab = suite[task], ABSTAIN[task]
            gold = np.array([it["gold"] for it in items])
            base = None
            for v, opts in variants.items():
                pred = run([dict(it, criteria=opts) for it in items])
                corr = (pred == gold).astype(float)
                row = {"acc": round(corr.mean(), 4), "abstain_pred": int((pred == ab).sum()), "abstain_gold": int((gold == ab).sum())}
                if base is None:
                    base = corr
                else:
                    d = corr - base
                    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
                    row["diff"] = f"{100 * d.mean():+.1f} [{100 * np.percentile(boot, 2.5):+.1f}, {100 * np.percentile(boot, 97.5):+.1f}]"
                results[name][f"{task} | {v}"] = row
                print(json.dumps({"model": name, "task": task, "variant": v, **row}), flush=True)
        del keep
        torch.cuda.empty_cache()

out = ROOT / "artifacts" / "abstain_descriptions.json"
out.write_text(json.dumps(results, indent=2))
names = list(results)
print("\n| Task, options | abstain gold | " + " | ".join(f"{n}: acc (abstains), vs as evaluated" for n in names) + " |")
print("|---|---|" + "---|" * len(names))
for key in results[names[0]]:
    g = results[names[0]][key]["abstain_gold"]
    cells = [f"{r[key]['acc']:.3f} ({r[key]['abstain_pred']})" + (f", {r[key]['diff']}" if "diff" in r[key] else "")
             for r in (results[n] for n in names)]
    print(f"| {key} | {g} | " + " | ".join(cells) + " |")
