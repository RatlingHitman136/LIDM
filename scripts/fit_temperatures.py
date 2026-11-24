"""Stage 3b: fit LIDM temperatures (cls.calibration) on a held-out calibration set and save them next to the checkpoint
as temperatures.json. The calibration set (data/calibration/calib.jsonl) is built on first use from commercial-clean
held-out rows: Open-Jev's calibration split and tasksource-jev validation rows (the stage-2 converters and filters,
evaluation sources excluded), plus the stage-2 dev set.

    uv run python scripts/fit_temperatures.py checkpoints/s2x-lidm/last.pt [--max 6000]
"""
import argparse
import collections
import json
import random
from pathlib import Path

import torch

from cls import calibration
from cls.data import stage2
from cls.data.typed import _load, write_jsonl
from cls.tracking import ROOT
from cls.train import encode, load_model, read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--max", type=int, default=6000, help="calibration questions")
a = ap.parse_args()

CALIB = ROOT / "data" / "calibration" / "calib.jsonl"
if not CALIB.exists():
    rng = random.Random(0)
    cases = [c for r in _load("ZefanCai/Open-Jev", "release-v2-redistributable", "calibration")
             if not stage2.TD_DOMAINS.search(r["source"]) and (c := stage2.jev_case(f"openjev/{r['source']}", r))]
    ts = _load("tasksource/tasksource-jev", "default", "validation")
    keep = [i for i, (s, lic) in enumerate(zip(ts["source"], ts["license_use"]))
            if lic == "commercial" and not stage2.EVAL_SOURCES.search(s) and not stage2.TD_DOMAINS.search(s)]
    cases += [c for r in ts.select(rng.sample(keep, min(len(keep), 6000))) if (c := stage2.jev_case(f"tasksource/{r['source']}", r))]
    items = [it for c in cases for it in stage2.lidm_items(c)] + read_jsonl(ROOT / "data" / "stage2" / "dev.jsonl")
    rng.shuffle(items)
    write_jsonl(items, CALIB)
    print(f"calibration set: {len(items)} questions -> {CALIB}", dict(collections.Counter(it["type"] for it in items)))

items = read_jsonl(CALIB)[: a.max]
enc, m, prior, _ = load_model(a.ckpt)
preds = []
with torch.no_grad():
    for i in range(0, len(items), 16):
        chunk = items[i:i + 16]
        inputs, _, _, cos = encode(enc, chunk, prior)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(*inputs, prior=cos).float()
        for it, row in zip(chunk, logits):
            n = 2 if it["type"] == "noul" else len(it["criteria"])
            preds.append((it["type"], row[:n].softmax(-1).cpu().numpy(), it["gold"]))
temps = calibration.fit(preds)
out = Path(a.ckpt).parent / "temperatures.json"
out.write_text(json.dumps(temps, indent=2))
print(json.dumps({k: round(v["t"], 3) for k, v in sorted(temps.items())}), "->", out)
