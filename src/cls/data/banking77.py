"""Banking77 loader for the preliminary Laya runs (`cls.eval.laya_baseline`) and the typed-item builder."""
from __future__ import annotations

import random
from dataclasses import dataclass

INSTRUCTION = "Which banking intent does the customer message express?"


@dataclass
class Banking77:
    texts: list[str]
    labels: list[int]
    label_ids: list[str]  # ClassLabel names, e.g. "card_arrival"
    descriptions: list[str]  # humanized ids, e.g. "card arrival"


def humanize(label_id: str) -> str:
    return label_id.replace("_", " ").lower()


def load_banking77(split: str = "test", n: int | None = None, seed: int = 0) -> Banking77:
    from datasets import load_dataset

    ds = load_dataset("PolyAI/banking77", split=split)
    ids = ds.features["label"].names
    idx = list(range(len(ds)))
    if n is not None and n < len(idx):
        idx = sorted(random.Random(seed).sample(idx, n))
    ds = ds.select(idx)
    return Banking77(list(ds["text"]), list(ds["label"]), ids, [humanize(i) for i in ids])


def make_question(label_ids: list[str], descriptions: list[str] | None = None) -> dict:
    """Laya question dict. Answers are keyed by label id either way; descriptions render as `id: desc`."""
    criteria = dict(zip(label_ids, descriptions)) if descriptions else list(label_ids)
    return {"intent": {"type": "choice", "instructions": INSTRUCTION, "criteria": criteria}}
