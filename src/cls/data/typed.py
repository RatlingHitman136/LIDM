"""Typed-decision items shared by every model (Laya baselines now, LIDM later).

Item: {"task", "type": choice|noul|score, "state", "instructions", "criteria", "gold"}
  choice: criteria = option names, gold = index into criteria
  noul:   no criteria, gold = 0 (false) or 1 (true)
  score:  criteria = level descriptions (index 0 first), gold = level index
Options are humanized label names (`card arrival`), no descriptions (docs/RESULTS.md, protocol).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from cls.data.banking77 import INSTRUCTION as B77_INSTRUCTION
from cls.data.banking77 import load_banking77

SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]  # glue/mnli label order
# CLINC150 banking + credit_cards domains: removed so no near-duplicate of Banking77 leaks into training
CLINC_HELD_OUT = {
    "transfer", "transactions", "balance", "freeze_account", "pay_bill", "bill_balance", "bill_due",
    "interest_rate", "routing", "min_payment", "order_checks", "pin_change", "report_fraud",
    "account_blocked", "spending_history", "credit_score", "report_lost_card", "credit_limit",
    "rewards_balance", "new_card", "application_status", "card_declined", "international_fees", "apr",
    "redeem_rewards", "credit_limit_change", "damaged_card", "replacement_card_duration",
    "improve_credit_score", "expiration_date",
}
# examples per task at scale=1.0 (12.9k total)
TRAIN_SIZES = {"ag_news": 2000, "emotion": 2000, "dbpedia": 2000, "clinc": 2400, "boolq": 1500,
               "mnli": 1500, "sst5": 1500}


def humanize(name: str) -> str:
    """`card_arrival` / `EducationalInstitution` / `oos` -> `card arrival` / `educational institution` / `out of scope`."""
    if name == "oos":
        return "out of scope"
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ").lower()


def _load(name: str, config: str | None, split: str):
    from datasets import load_dataset

    return load_dataset(name, config, split=split)


def _rows(ds, n: int | None, rng: random.Random) -> list[dict]:
    idx = list(range(len(ds)))
    if n is not None and n < len(idx):
        idx = sorted(rng.sample(idx, n))
    return [ds[i] for i in idx]


def _subsample_options(names: list[str], gold: int, k: int, rng: random.Random) -> tuple[list[str], int]:
    """Gold plus k-1 random distractors, shuffled. Returns (options, new gold index)."""
    others = rng.sample([i for i in range(len(names)) if i != gold], k - 1)
    picked = [gold] + others
    rng.shuffle(picked)
    return [names[i] for i in picked], picked.index(gold)


def _choice(task, state, instructions, criteria, gold):
    return {"task": task, "type": "choice", "state": state, "instructions": instructions,
            "criteria": criteria, "gold": gold}


# ---------------------------------------------------------------- eval sets
def banking77_items(n_opts: int | None = None, n: int | None = None, seed: int = 0) -> list[dict]:
    """Banking77 test. n_opts=None: all 77 labels in canonical order; else gold + random labels."""
    data = load_banking77("test", n=n, seed=seed)
    rng = random.Random(seed)
    out = []
    for text, y in zip(data.texts, data.labels):
        opts, g = (data.descriptions, y) if n_opts is None else _subsample_options(data.descriptions, y, n_opts, rng)
        out.append(_choice("banking77", text, B77_INSTRUCTION, list(opts), g))
    return out


def _boolq(row) -> dict:
    q = row["question"].strip()
    q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
    return {"task": "boolq", "type": "noul", "state": row["passage"], "instructions": q, "gold": int(row["answer"])}


def _sst5(row) -> dict:
    return {"task": "sst5", "type": "score", "state": row["text"],
            "instructions": "How positive is the sentiment of this text?", "criteria": SST5_LEVELS,
            "gold": int(row["label"])}


def boolq_items(n: int | None = None, seed: int = 0) -> list[dict]:
    return [_boolq(r) for r in _rows(_load("google/boolq", None, "validation"), n, random.Random(seed))]


def sst5_items(n: int | None = None, seed: int = 0) -> list[dict]:
    return [_sst5(r) for r in _rows(_load("SetFit/sst5", None, "test"), n, random.Random(seed))]


# ---------------------------------------------------------------- training mixture
def build_train(seed: int = 0, scale: float = 1.0) -> list[dict]:
    """Shared fine-tuning mixture; Banking77 and CLINC's banking/credit-card intents are held out."""
    rng = random.Random(seed)
    size = {k: max(1, round(v * scale)) for k, v in TRAIN_SIZES.items()}
    items: list[dict] = []

    for task, (name, cfg), instr in [
        ("ag_news", ("fancyzhx/ag_news", None), "What is the topic of this news article?"),
        ("emotion", ("dair-ai/emotion", "split"), "Which emotion does the text express?"),
    ]:
        ds = _load(name, cfg, "train")
        names = [humanize(x) for x in ds.features["label"].names]
        items += [_choice(task, r["text"], instr, names, r["label"]) for r in _rows(ds, size[task], rng)]

    ds = _load("fancyzhx/dbpedia_14", None, "train")
    names = [humanize(x) for x in ds.features["label"].names]
    items += [_choice("dbpedia", f"{r['title']}. {r['content'].strip()}",
                      "What kind of entity does this Wikipedia text describe?", names, r["label"])
              for r in _rows(ds, size["dbpedia"], rng)]

    # CLINC: 120 intents + out-of-scope is too many for Laya's head budget, so each item gets 2-24 options
    ds = _load("clinc/clinc_oos", "plus", "train")
    raw = ds.features["intent"].names
    keep = [i for i, x in enumerate(raw) if x not in CLINC_HELD_OUT]
    names = [humanize(raw[i]) for i in keep]
    remap = {old: new for new, old in enumerate(keep)}
    pool = ds.filter(lambda r: r["intent"] in remap)
    for r in _rows(pool, size["clinc"], rng):
        opts, g = _subsample_options(names, remap[r["intent"]], rng.randint(2, 24), rng)
        items.append(_choice("clinc", r["text"], "What does the user want the assistant to do?", opts, g))

    items += [_boolq(r) for r in _rows(_load("google/boolq", None, "train"), size["boolq"], rng)]

    for r in _rows(_load("nyu-mll/glue", "mnli", "train"), size["mnli"], rng):
        items.append(_choice("mnli", r["premise"],
                             f'How does the text relate to the statement: "{r["hypothesis"].strip()}"?',
                             NLI_LABELS, r["label"]))

    items += [_sst5(r) for r in _rows(_load("SetFit/sst5", None, "train"), size["sst5"], rng)]
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------- io
def to_laya_rows(items: list[dict]) -> list[dict]:
    """`{state, questions, expected}` rows for `laya.train.finetune`; the question id is the task name."""
    rows = []
    for it in items:
        q = {"type": it["type"], "instructions": it["instructions"]}
        if it["type"] == "noul":
            expected = bool(it["gold"])
        else:
            q["criteria"] = it["criteria"]
            expected = it["criteria"][it["gold"]] if it["type"] == "choice" else it["gold"]
        rows.append({"state": it["state"], "questions": {it["task"]: q}, "expected": {it["task"]: expected}})
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
