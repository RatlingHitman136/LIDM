"""Few-shot Banking77 data: k examples per intent from the *train* split (several seeds, since few-shot results
depend on which examples are drawn), plus a small disjoint early-stopping set, also from train. The test split
stays untouched. Writes, per seed, data/fewshot/b77_k{k}_s{seed}_{train,val}.jsonl (LIDM items, all 77 labels as
options) and b77_k{k}_s{seed}_laya.jsonl (rows for cls.eval.laya_finetune).

    uv run python scripts/prepare_fewshot_banking77.py --k 8 --seeds 0 1 2
"""
import argparse
import collections
import random

from cls.data.banking77 import INSTRUCTION, load_banking77
from cls.data.typed import _choice, to_laya_rows, write_jsonl
from cls.tracking import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=8, help="training examples per intent")
ap.add_argument("--val-per-class", type=int, default=2)
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
a = ap.parse_args()

b = load_banking77("train")
by_label = collections.defaultdict(list)
for text, y in zip(b.texts, b.labels):
    by_label[y].append(text)
out = ROOT / "data" / "fewshot"
for seed in a.seeds:
    rng = random.Random(seed)
    train, val = [], []
    for y in sorted(by_label):
        texts = rng.sample(by_label[y], a.k + a.val_per_class)
        train += [_choice("banking77", t, INSTRUCTION, list(b.descriptions), y) for t in texts[:a.k]]
        val += [_choice("banking77", t, INSTRUCTION, list(b.descriptions), y) for t in texts[a.k:]]
    rng.shuffle(train)
    assert not {it["state"] for it in train} & {it["state"] for it in val}
    stem = f"b77_k{a.k}_s{seed}"
    write_jsonl(train, out / f"{stem}_train.jsonl")
    write_jsonl(val, out / f"{stem}_val.jsonl")
    write_jsonl(to_laya_rows(train), out / f"{stem}_laya.jsonl")
    print(f"seed {seed}: {len(train)} train, {len(val)} val -> {out}/{stem}_*.jsonl")
