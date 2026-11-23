"""Stage-2 mix for the same-data LIDM vs Laya comparison (cls.data.stage2). Writes data/stage2/:
train.jsonl + dev.jsonl (LIDM items; dev = 1% of cases per source, at most 30, for monitoring only),
train_laya.jsonl (the same training cases as Laya rows), and evaluation item sets aimed at label variety and length:
eval_heldout_sources (tasksource sources never trained on), eval_banking77_names / eval_banking77_desc (N=77, names
and `name: description`), eval_clinc150 (N=150, in-scope test). Drops any case whose state prefix matches an
evaluation item (typed-decisions test, the context suite as evaluated, the sets above, Banking77 and SST-5 test).

    uv run python scripts/prepare_stage2.py
"""
import collections
import json
import random

from cls.data import stage2
from cls.data.banking77 import INSTRUCTION as B77
from cls.data.banking77 import load_banking77
from cls.data.context_suite import load_suite
from cls.data.typed import _choice, _load, humanize, write_jsonl
from cls.tracking import ROOT


def key(text) -> str:
    return " ".join(str(text).lower().split())[:200]


rng = random.Random(0)
cases = []
ts_train, ts_held = stage2.tasksource_cases(rng)
openjev = stage2.openjev_cases(rng)
for name, got in [("tdsynth", stage2.tdsynth_cases(rng)), ("openjev", rng.sample(openjev, min(20000, len(openjev)))),
                  ("tasksource", ts_train), ("helpsteer2", stage2.helpsteer_cases(rng)), ("civil", stage2.civil_cases(rng, n=5000)),
                  ("aegis2", stage2.aegis_cases(rng))]:
    print(f"{name}: {len(got)} cases, {sum(len(c['questions']) for c in got)} questions", flush=True)
    cases += got

out = ROOT / "data" / "stage2"
evals = {"eval_heldout_sources": [dict(it, task=f"heldout/{c['source'].split('/', 1)[1]}") for c in ts_held for it in stage2.lidm_items(c)]}
b77 = load_banking77("test", n=1000, seed=0)
desc = json.loads((ROOT / "src" / "cls" / "data" / "banking77_descriptions.json").read_text())
evals["eval_banking77_names"] = [_choice("banking77", t, B77, list(b77.descriptions), y) for t, y in zip(b77.texts, b77.labels)]
evals["eval_banking77_desc"] = [_choice("banking77_desc", t, B77, [f"{n}: {desc[i]}" for i, n in zip(b77.label_ids, b77.descriptions)], y)
                                for t, y in zip(b77.texts, b77.labels)]
cl = _load("clinc/clinc_oos", "plus", "test")
raw = cl.features["intent"].names
names, oos = [humanize(x) for x in raw], raw.index("oos")
inscope = [r for r in cl if r["intent"] != oos]
opts = [n for i, n in enumerate(names) if i != oos]
evals["eval_clinc150"] = [_choice("clinc150", r["text"], "What does the user want the assistant to do?", opts,
                                  opts.index(names[r["intent"]])) for r in rng.sample(inscope, 1000)]
for name, its in evals.items():
    write_jsonl(its, out / f"{name}.jsonl")
    print(f"{name}: {len(its)} questions, options per question {min(len(i.get('criteria', [0, 0])) for i in its)}-"
          f"{max(len(i.get('criteria', [0, 0])) for i in its)}")

banned = {key(it["state"]) for items in [*load_suite(n=1000).values(), *evals.values()] for it in items}
banned |= {key(r["text"]) for r in _load("PolyAI/banking77", None, "test")} | {key(r["text"]) for r in _load("SetFit/sst5", None, "test")}
before = len(cases)
cases = [c for c in cases if key(c["state"]) not in banned]
print(f"leak check: dropped {before - len(cases)} cases whose state matches an evaluation item")

by_src = collections.defaultdict(list)
for c in cases:
    by_src[c["source"]].append(c)
train, dev = [], []
for src in sorted(by_src):
    cs = by_src[src]
    rng.shuffle(cs)
    k = min(30, len(cs) // 100)
    dev += cs[:k]
    train += cs[k:]
rng.shuffle(train)
items = [it for c in train for it in stage2.lidm_items(c)]
write_jsonl(items, out / "train.jsonl")
write_jsonl([it for c in dev for it in stage2.lidm_items(c)], out / "dev.jsonl")
write_jsonl([{"state": c["state"], "questions": c["questions"], "expected": c["expected"]} for c in train], out / "train_laya.jsonl")
groups = collections.Counter(it["task"] for it in items)
print(f"train: {len(train)} cases, {len(items)} questions, {len({c['source'] for c in train})} sources; dev: {len(dev)} cases")
print("by group:", dict(groups), "| by type:", dict(collections.Counter(it["type"] for it in items)))
