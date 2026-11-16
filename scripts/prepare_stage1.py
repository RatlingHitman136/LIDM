"""Build the stage-1 label-reading mixture: data/stage1_train.jsonl + data/stage1_val.jsonl (zero-shot val).

    uv run python scripts/prepare_stage1.py          # full size (~1.5M questions)
    uv run python scripts/prepare_stage1.py 0.01     # 1% scale, for a quick smoke test

Prints counts per task and a leak check against Banking77 (label strings and verbatim texts).
"""
import collections
import json
import sys

from cls.data.banking77 import load_banking77
from cls.data.stage1 import build_stage1, leak_report
from cls.data.typed import write_jsonl
from cls.tracking import ROOT

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
train, val = build_stage1(seed=0, scale=scale)
d = ROOT / "data"
write_jsonl(train, d / "stage1_train.jsonl")
write_jsonl(val, d / "stage1_val.jsonl")
print("train", len(train), dict(sorted(collections.Counter(i["task"] for i in train).items())))
print("val", len(val), dict(sorted(collections.Counter(i["task"] for i in val).items())))
print("distinct train option strings", len({t for i in train for t in i.get("criteria", [])}))
leaks = leak_report(train, load_banking77("test").descriptions)
print("leak check vs Banking77:", json.dumps(leaks))
