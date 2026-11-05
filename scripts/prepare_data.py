"""Build the shared training mixture (Banking77 held out) and write it as JSONL for Laya + LIDM."""
import collections
import json
import sys

from cls.data.typed import build_train, to_laya_rows, write_jsonl
from cls.tracking import ROOT

scale = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
items = build_train(seed=0, scale=scale)
d = ROOT / "data"
write_jsonl(items, d / "train_items.jsonl")
write_jsonl(to_laya_rows(items), d / "train_laya.jsonl")
print(len(items), dict(collections.Counter(i["task"] for i in items)))
print(dict(collections.Counter(i["type"] for i in items)))
