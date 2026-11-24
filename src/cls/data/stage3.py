"""Stage 3a: soft labels from human vote distributions (docs/DESIGN.md, "Stage 3: calibration").

Training (commercial-clean): GoEmotions raw ratings (Apache-2.0), one noul question per emotion with P(true) = share of
raters who marked it; DICES (CC BY 4.0), one noul question per conversation, P(unsafe) = (Yes + Unsure / 2) / raters.
Evaluation only (CC BY-NC 4.0): ChaosNLI SNLI / MNLI, 100 votes per pair, as entailment / neutral / contradiction.
Items use the LIDM schema with `soft` = [P(no), P(yes)] (noul) or the vote distribution (choice).
"""
from __future__ import annotations

import ast
import collections
import csv
import random
import sys

from cls.data.context_suite import nli_item
from cls.data.typed import _load

EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
            "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
            "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
            "neutral"]


def _emotion_q(e: str) -> str:
    return "The text expresses no particular emotion." if e == "neutral" else f"The text expresses {e}."


def goemotions(rng: random.Random, per_comment: int = 3, heldout: int = 1000, min_raters: int = 3, n_train: int = 25000):
    """(train, held-out) noul items. Per comment: its most-marked emotion plus per_comment - 1 random others, so the
    targets span confident yes, split votes and confident no. Comments with fewer than min_raters clear ratings are
    skipped."""
    votes = collections.defaultdict(list)  # keyed by text: some comments recur under several ids
    for r in _load("google-research-datasets/go_emotions", "raw", "train"):
        if not r["example_very_unclear"]:
            votes[r["text"]].append([int(r[e]) for e in EMOTIONS])
    ids = sorted(t for t, v in votes.items() if len(v) >= min_raters)
    rng.shuffle(ids)
    ids = ids[: heldout + n_train]
    out = {"train": [], "heldout": []}
    for n, cid in enumerate(ids):
        v = votes[cid]
        share = [sum(col) / len(v) for col in zip(*v)]
        top = max(range(len(EMOTIONS)), key=share.__getitem__)
        picks = [top] + rng.sample([i for i in range(len(EMOTIONS)) if i != top], per_comment - 1)
        part = "heldout" if n < heldout else "train"
        for i in picks:
            out[part].append({"task": "s3/goemotions", "type": "noul", "state": cid,
                              "instructions": _emotion_q(EMOTIONS[i]), "gold": int(share[i] >= 0.5),
                              "soft": [round(1 - share[i], 4), round(share[i], 4)], "raters": len(v)})
    return out["train"], out["heldout"]


def dices(paths: list[str], rng: random.Random, heldout_share: float = 0.2):
    """(train, held-out) noul items, one per DICES conversation (350 and 990 sets)."""
    csv.field_size_limit(sys.maxsize)
    items = []
    for path in paths:
        by_item = collections.defaultdict(list)
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                by_item[r["item_id"]].append(r)
        for rows in by_item.values():
            ans = collections.Counter(r["Q_overall"] for r in rows)
            n = sum(ans.values())
            if not n:
                continue
            p = (ans["Yes"] + ans["Unsure"] / 2) / n
            r0 = rows[0]
            items.append({"task": "s3/dices", "type": "noul", "state": f"{r0['context']}\n\nAssistant: {r0['response']}",
                          "instructions": "The assistant's last response is unsafe or harmful.", "gold": int(p >= 0.5),
                          "soft": [round(1 - p, 4), round(p, 4)], "raters": n})
    rng.shuffle(items)
    k = round(heldout_share * len(items))
    return items[k:], items[:k]


def chaosnli() -> list[dict]:
    """ChaosNLI SNLI + MNLI-matched (evaluation only): soft = the 100-vote distribution over entailment / neutral /
    contradiction, gold = the majority label."""
    out = []
    for cfg in ("snli", "mnli_m"):
        for r in _load("earino/chaosnli", cfg, "validation"):
            ex = r["example"] if isinstance(r["example"], dict) else ast.literal_eval(r["example"])
            dist = [float(x) for x in (r["label_dist"] if isinstance(r["label_dist"], list) else ast.literal_eval(r["label_dist"]))]
            it = nli_item(f"chaosnli/{cfg}", ex["premise"], ex["hypothesis"], max(range(3), key=dist.__getitem__))
            out.append(dict(it, soft=dist))
    return out
