"""Build data/stage1b_train.jsonl (label-variety mixture: refinedweb configs + subsampled stage-1 data) and
data/stage1b_val.jsonl (= stage1_val + held-out refinedweb questions with unseen answers, tasks `heldout/*`).
Needs data/stage1_train.jsonl and data/stage1_val.jsonl from scripts/prepare_stage1.py.

    uv run python scripts/prepare_stage1b.py
"""
import collections
import json

from cls.data.banking77 import load_banking77
from cls.data.stage1 import build_stage1b, leak_report
from cls.data.typed import write_jsonl
from cls.tracking import ROOT
from cls.train import read_jsonl

d = ROOT / "data"
val = read_jsonl(d / "stage1_val.jsonl")
items, heldout = build_stage1b(read_jsonl(d / "stage1_train.jsonl"), val)
write_jsonl(items, d / "stage1b_train.jsonl")
write_jsonl(val + heldout, d / "stage1b_val.jsonl")
print("held-out val", len(heldout), dict(sorted(collections.Counter(i["task"] for i in heldout).items())))
opts = collections.Counter(t for it in items for t in it.get("criteria", []))
print("train", len(items), dict(sorted(collections.Counter(i["task"].split("/")[0] for i in items).items())))
print(f"distinct option strings {len(opts)}; used once {sum(v == 1 for v in opts.values()) / len(opts):.2f}; "
      f"median uses {sorted(opts.values())[len(opts) // 2]}")
zs = {t.lower() for v in val if v["task"].startswith("zs/") for t in v["criteria"]}
shared = zs & {t.lower() for t in opts}
assert not shared, f"zero-shot val labels appear as training options: {sorted(shared)[:5]}"
print("zs val labels in training: 0 | leak check vs Banking77:", json.dumps(leak_report(items, load_banking77("test").descriptions)))
