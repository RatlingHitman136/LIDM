"""Kill-criterion test data: LocalLLaMA/typed-decisions (Apache-2.0). The train split (1,200 cases) is split *by
case* into train and dev (10% of the cases of every workflow, for checkpoint selection), so the 5 questions of a
case never sit on both sides; the official test split (400 cases) stays untouched. Hard labels for both models.

Writes data/td/:
  train.jsonl, dev.jsonl, test.jsonl   LIDM items, one per question, task = td/<workflow>
  train_laya.jsonl, dev_laya.jsonl     the same cases as native Laya rows {state, questions, expected}, one per case
  lowo_<workflow>/{train,dev,train_laya}.jsonl   leave-one-workflow-out: the same train/dev cases minus that
                                       workflow; test on the full test.jsonl and read that workflow's accuracy

    uv run python scripts/prepare_typed_decisions.py
"""
import collections
import json
import random

from cls.data.context_suite import typed_decisions_items
from cls.data.typed import _load, write_jsonl
from cls.tracking import ROOT

OUT = ROOT / "data" / "td"
rng = random.Random(0)
train_rows = list(_load("LocalLLaMA/typed-decisions", "all", "train"))
test_rows = list(_load("LocalLLaMA/typed-decisions", "all", "test"))

by_wf = collections.defaultdict(list)
for r in train_rows:
    by_wf[r["workflow"]].append(r)
dev_ids = set()
for wf in sorted(by_wf):
    ids = sorted(r["id"] for r in by_wf[wf])
    dev_ids |= set(rng.sample(ids, round(0.1 * len(ids))))
split = {"train": [r for r in train_rows if r["id"] not in dev_ids],
         "dev": [r for r in train_rows if r["id"] in dev_ids], "test": test_rows}


def items(rows):
    return [dict(it, task=f"td/{it['workflow']}") for r in rows for it in typed_decisions_items(r)]


def laya_row(r):
    """Native Laya row: the case's state with all its questions; expected = hard label in Laya's answer form."""
    qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
    exp = {}
    for qid, q in qs.items():
        lab = gold[qid]["label"]
        exp[qid] = {"noul": lab == "true", "score": int(lab) if q["type"] == "score" else lab}.get(q["type"], lab)
    return {"state": r["state"], "questions": qs, "expected": exp}


assert not {r["id"] for r in split["train"]} & {r["id"] for r in split["dev"]}
assert not {r["state"] for r in split["test"]} & {r["state"] for r in train_rows}, "test state also in train"
for name, rows in split.items():
    its = items(rows)
    write_jsonl(its, OUT / f"{name}.jsonl")
    if name != "test":
        write_jsonl([laya_row(r) for r in rows], OUT / f"{name}_laya.jsonl")
    print(f"{name}: {len(rows)} cases, {len(its)} questions, {dict(collections.Counter(it['type'] for it in its))}, "
          f"{dict(collections.Counter(r['workflow'] for r in rows))}")
for wf in sorted(by_wf):
    tr, dv = ([r for r in split[k] if r["workflow"] != wf] for k in ("train", "dev"))
    write_jsonl(items(tr), OUT / f"lowo_{wf}" / "train.jsonl")
    write_jsonl(items(dv), OUT / f"lowo_{wf}" / "dev.jsonl")
    write_jsonl([laya_row(r) for r in tr], OUT / f"lowo_{wf}" / "train_laya.jsonl")
    print(f"lowo_{wf}: {len(tr)} train / {len(dv)} dev cases")
print("->", OUT)
