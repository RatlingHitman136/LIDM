"""Stage-2 pass with stage-1 replay: data/stage2/train.jsonl plus a random sample of data/stage1_train.jsonl (the stage-1
data the stage-1 MBL checkpoint learned, under s2x-lidm) so that replay is `--share` of the result, shuffled together.
Writes data/stage2/train_replay.jsonl.

    uv run python scripts/prepare_stage2_replay.py [--share 0.2]
"""
import argparse
import collections
import json
import random

from cls.tracking import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--share", type=float, default=0.2)
a = ap.parse_args()

rng = random.Random(0)
with open(ROOT / "data" / "stage2" / "train.jsonl") as f:
    s2 = [json.loads(line) for line in f]
with open(ROOT / "data" / "stage1_train.jsonl") as f:
    s1 = f.readlines()
n = round(len(s2) * a.share / (1 - a.share))
replay = [json.loads(line) for line in rng.sample(s1, n)]
mix = s2 + replay
rng.shuffle(mix)
out = ROOT / "data" / "stage2" / "train_replay.jsonl"
with open(out, "w") as f:
    f.writelines(json.dumps(it) + "\n" for it in mix)
assert sum(it["task"].startswith("s2/") for it in mix) == len(s2)
print(f"{len(s2)} stage-2 + {n} stage-1 replay = {len(mix)} questions ({n / len(mix):.1%} replay) -> {out}")
print("replay by group:", dict(collections.Counter(it["task"].split("/")[0] for it in replay)))
