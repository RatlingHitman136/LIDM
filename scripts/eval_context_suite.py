"""Zero-shot context-understanding suite (cls.data.context_suite): tasks where the answer depends on reading the
state. Per model: accuracy, macro-F1, ECE and NLL per task, to MLflow `context_suite` and
artifacts/context_suite/<name>.json; `--report` prints one table over every JSON written so far.

    uv run python scripts/eval_context_suite.py lidm --ckpt checkpoints/stage1b-gte-lora/last.pt --name lidm-1b-gte
    uv run python scripts/eval_context_suite.py lidm --ckpt ... --name lidm-1b-gte-optsA --options-in-question
    uv run python scripts/eval_context_suite.py biencoder --encoder BAAI/bge-base-en-v1.5 --name bge
    uv run python scripts/eval_context_suite.py laya [--ckpt convaiinnovations/laya-typed-decisions] --name laya
    uv run python scripts/eval_context_suite.py --report
    uv run python scripts/eval_context_suite.py lidm --ckpt ... --name x --items data/stage2/eval_*.jsonl  # label suite
    uv run python scripts/eval_context_suite.py --report-labels

Bi-encoders score cos(question + state, option) at a fixed temperature: accuracy is what they are compared on;
their ECE/NLL are not fitted and mean little. noul questions give a bi-encoder only "no"/"yes", i.e. chance.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cls.data import context_suite as cs
from cls.eval import harness
from cls.tracking import ROOT, Tracker

OUT = ROOT / "artifacts" / "context_suite"
NOPT = ((2, 4), (5, 12), (13, 40), (41, 10**6))
OPTLEN = ((0, 20), (21, 80), (81, 10**6))  # mean option length in characters


def _bucket(v, edges):
    return next(f"{lo}-{hi}" if hi < 10**6 else f"{lo}+" for lo, hi in edges if v <= hi)  # v may be fractional

ap = argparse.ArgumentParser()
ap.add_argument("model", nargs="?", choices=["lidm", "biencoder", "laya"])
ap.add_argument("--name")
ap.add_argument("--ckpt", help="lidm: checkpoint .pt; laya: hub id or local dir (default: shipped Laya)")
ap.add_argument("--encoder", default="BAAI/bge-base-en-v1.5", help="biencoder only")
ap.add_argument("--options-in-question", action="store_true",
                help="ablation: also write every option into the question, so pass A sees them with the state")
ap.add_argument("--n", type=int, default=1000, help="questions per task (typed-decisions: always all)")
ap.add_argument("--tasks", nargs="+", help="subset of " + " ".join(cs.TASKS))
ap.add_argument("--items", nargs="+", help="evaluate these item files instead (task = file stem), with accuracy by option "
                "count and option length, to artifacts/label_suite/")
ap.add_argument("--temps", help="stage-3b temperatures.json to apply (cls.calibration) to every prediction")
ap.add_argument("--report", action="store_true")
ap.add_argument("--report-labels", action="store_true")
ap.add_argument("--no-track", action="store_true")
a = ap.parse_args()


def report():
    runs = {p.stem: json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))}
    if not runs:
        raise SystemExit(f"no results in {OUT}")
    names = list(runs)
    print("| group | task | n | majority | " + " | ".join(names) + " |")
    print("|---|---|---|---|" + "---|" * len(names))
    for task, (group, _, _) in cs.TASKS.items():
        r = [runs[m]["tasks"].get(task) for m in names]
        ref = next((x for x in r if x), None)
        if ref:
            cells = [f"{x['accuracy']:.3f}" if x else "-" for x in r]
            print(f"| {group} | {task} | {ref['n']} | {ref['majority']:.3f} | " + " | ".join(cells) + " |")
    for label, keep in (("regime 1 mean", lambda t: cs.TASKS[t][1] == "1"), ("regime 2 mean", lambda t: t in cs.REGIME2)):
        cells = []
        for m in names:
            acc = [v["accuracy"] for t, v in runs[m]["tasks"].items() if keep(t)]
            cells.append(f"{np.mean(acc):.3f} ({len(acc)})" if acc else "-")
        print(f"| | **{label}** | | | " + " | ".join(cells) + " |")


def report_labels():
    d = ROOT / "artifacts" / "label_suite"
    runs = {p.stem: json.loads(p.read_text())["tasks"] for p in sorted(d.glob("*.json"))}
    names = list(runs)
    tasks = sorted({t for r in runs.values() for t in r})
    print("| task | n | " + " | ".join(names) + " |\n|---|---|" + "---|" * len(names))
    for t in tasks:
        ref = next(r[t] for r in runs.values() if t in r)
        print(f"| {t} | {ref['n']} | " + " | ".join(f"{runs[m][t]['accuracy']:.3f}" if t in runs[m] else "-" for m in names) + " |")
        for key in ("acc_by_nopt", "acc_by_optlen"):
            for b in ref.get(key, {}):
                cells = [f"{runs[m][t][key][b][0]:.3f}" if t in runs[m] and b in runs[m][t].get(key, {}) else "-" for m in names]
                print(f"| ↳ {key[7:]} {b} | {ref[key][b][1]} | " + " | ".join(cells) + " |")


if a.report:
    report()
    raise SystemExit
if a.report_labels:
    report_labels()
    raise SystemExit
if not (a.model and a.name):
    ap.error("model and --name are required unless --report")

if a.model == "lidm":
    from cls.train import encode, load_model

    enc, m, prior, st = load_model(a.ckpt)

    @torch.no_grad()
    def predict(item):
        inputs, _, _, cos = encode(enc, [item], prior)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(*inputs, prior=cos)
        return logits[0].float().softmax(-1).cpu().numpy()

    params = {"checkpoint": a.ckpt, "step": st["step"], "encoder": st["config"].get("encoder")}
elif a.model == "biencoder":
    from cls.model import Encoder, option_texts

    enc = Encoder(a.encoder)

    @torch.no_grad()
    def predict(item):
        q = enc.embed([f"{item['instructions']}\n\n{item['state']}"])[0]
        z = enc.embed(option_texts(item)) @ q / 0.05
        z = np.exp(z - z.max())
        return z / z.sum()

    params = {"encoder": a.encoder, "tau": 0.05}
else:
    import laya

    ckpt = a.ckpt or "convaiinnovations/laya"
    agent = laya.load(str(Path(ckpt).resolve()) if Path(ckpt).exists() else ckpt)  # a relative path is read as a hub id
    base = harness.laya_predictor(agent)

    def predict(item):
        if item["type"] == "choice" and len(set(item["criteria"])) < len(item["criteria"]):
            # Laya rejects repeated options (1 RACE + 1 QuAIL question): copies share one score and the first wins
            # the argmax, as in LIDM and the bi-encoders
            uniq = list(dict.fromkeys(item["criteria"]))
            p = base(dict(item, criteria=uniq))
            return np.array([p[uniq.index(c)] for c in item["criteria"]])
        if "criteria_dict" not in item:
            return base(item)
        # typed-decisions choice questions in Laya's native {name: description} form, as it was trained
        q = {"q": {"type": "choice", "instructions": item["instructions"], "criteria": item["criteria_dict"]}}
        p = agent.system_one(item["state"], q)["answers"]["q"]["probabilities"]
        return np.array([p.get(k, 0.0) for k in item["names"]])

    params = {"checkpoint": ckpt}

if a.temps:
    from cls import calibration

    _temps, _raw = calibration.load(a.temps), predict
    predict = lambda item: calibration.apply(_raw(item), item["type"], _temps)
if a.items:
    from cls.train import read_jsonl

    suite, OUT = {Path(f).stem: read_jsonl(f) for f in a.items}, ROOT / "artifacts" / "label_suite"
else:
    suite = cs.load_suite(n=a.n, tasks=a.tasks)
if a.options_in_question:
    suite = {t: [cs.with_options_in_question(it) for it in items] for t, items in suite.items() if t in cs.REGIME2}
results = {}
with Tracker("context_suite", a.name, tags={"model": a.model, "options_in_question": str(a.options_in_question)},
             enabled=not a.no_track) as tr:
    tr.log_params({**params, "n": a.n, "options_in_question": a.options_in_question})
    for task, items in suite.items():
        met, P, y = harness.evaluate_task(predict, items)
        met["n"], met["majority"] = len(items), float(np.bincount(y).max() / len(y))
        correct = (P.argmax(1) == y).astype(int)
        met["correct"] = correct.tolist()  # per question, for paired comparisons
        if a.items:
            for key, edges, val in (("acc_by_nopt", NOPT, lambda it: len(it.get("criteria") or [0, 0])),
                                    ("acc_by_optlen", OPTLEN, lambda it: np.mean([len(c) for c in it.get("criteria") or ["no", "yes"]]))):
                groups = {}
                for it, c in zip(items, correct):
                    groups.setdefault(_bucket(val(it), edges), []).append(c)
                met[key] = {b: (float(np.mean(v)), len(v)) for b, v in sorted(groups.items())}
        results[task] = met
        tr.log_metrics({f"{task}/{k}": v for k, v in met.items() if isinstance(v, float)})
        print(f"[{a.name}] {task}: acc={met['accuracy']:.4f} (majority {met['majority']:.3f}) ece={met['ece']:.3f} "
              f"n={len(items)} {met['wall_s']:.0f}s", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{a.name}.json").write_text(json.dumps({"params": params, "tasks": results}, indent=2))
print("saved", OUT / f"{a.name}.json")
