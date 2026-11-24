"""Stage 3a data (cls.data.stage3) -> data/stage3/: train.jsonl (GoEmotions + DICES x2 + 25% replay from the soft
stage-2 mix), dev.jsonl (a slice of the held-out sets, for monitoring), and evaluation sets eval_goemotions.jsonl,
eval_dices.jsonl (held-out comments / conversations) and eval_chaosnli.jsonl (evaluation only, CC BY-NC).

    uv run python scripts/prepare_stage3.py --dices <dir with the two DICES csv files>
"""
import argparse
import collections
import json
import random
from pathlib import Path

from cls.data import stage3
from cls.data.typed import write_jsonl
from cls.tracking import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--dices", required=True, help="folder with diverse_safety_adversarial_dialog_{350,990}.csv")
ap.add_argument("--replay-share", type=float, default=0.25)
a = ap.parse_args()

rng = random.Random(0)
go_train, go_held = stage3.goemotions(rng)
paths = sorted(str(p) for p in Path(a.dices).glob("diverse_safety_adversarial_dialog_*.csv"))
assert len(paths) == 2, paths
di_train, di_held = stage3.dices(paths, rng)
new = go_train + di_train * 2  # DICES is small (1,072 conversations); seen twice
with open(ROOT / "data" / "stage2_soft" / "train.jsonl") as f:
    s2 = f.readlines()
n_replay = round(len(new) * a.replay_share / (1 - a.replay_share))
train = new + [json.loads(line) for line in rng.sample(s2, n_replay)]
rng.shuffle(train)
out = ROOT / "data" / "stage3"
write_jsonl(train, out / "train.jsonl")
write_jsonl(rng.sample(go_held, 600) + di_held, out / "dev.jsonl")
write_jsonl(go_held, out / "eval_goemotions.jsonl")
write_jsonl(di_held, out / "eval_dices.jsonl")
chaos = stage3.chaosnli()
write_jsonl(chaos, out / "eval_chaosnli.jsonl")
held_states = {it["state"] for it in go_held + di_held}
assert not held_states & {it["state"] for it in new}, "held-out text in training"
soft = [it["soft"][1] for it in go_train]
print(f"train {len(train)}: GoEmotions {len(go_train)}, DICES {len(di_train)} x2, replay {n_replay} "
      f"({n_replay / len(train):.0%}) | groups {dict(collections.Counter(it['task'].split('/')[0] for it in train))}")
print(f"GoEmotions P(yes) share: =0 {sum(x == 0 for x in soft) / len(soft):.0%}, split {sum(0 < x < 1 for x in soft) / len(soft):.0%}, "
      f"=1 {sum(x == 1 for x in soft) / len(soft):.0%}")
print(f"eval: GoEmotions {len(go_held)}, DICES {len(di_held)}, ChaosNLI {len(chaos)} -> {out}")
