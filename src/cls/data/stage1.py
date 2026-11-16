"""Stage-1 label-reading mixture (docs/DESIGN.md, "Stage 1: label reading"): teach the model to score option
texts it has never seen. Same item schema as cls.data.typed.

Train (974k questions at scale 1; the design doc planned ~1.5M, but the synthetic set has only ~152k usable texts):
  synthetic  MoritzLaurer/synthetic_zeroshot_mixtral_v0.1, `mixtral_written_texts_for_tasks*` configs: choice
             over each task's hypothesis statements ("The text is ...")                    all usable, ~152k
  utcd/*     claritylab/utcd in-domain train sets: choice over descriptive labels          33k cap per set
  label_nli  tasksource/zero-shot-label-nli: noul "does this label statement fit the text?"    300k
  nli/*      MNLI, SNLI, WANLI, FEVER-NLI: choice over entailment / neutral / contradiction     300k
Val: UTCD's out-of-domain sets, all labels as options. "Out-of-domain" there means a domain shift, not unseen
labels (yelp's negative / neutral / positive are training options too), so each set is tagged after the
train mix is built: `zs/<set>` if none of its labels is a training option, else `ood/<set>`. Banking77 is
`zs/banking77`, built in the final test's format (humanized labels, Banking77 instruction) from its *train*
split, so the test split stays untouched for the final number.

Held out: Banking77 everywhere; CLINC's banking + credit-card intents (as in M0) and "exchange rate"; SGD's
Check Balance and Transfer Money; zero-shot-label-nli rows marked non-commercial and tasks named after
Banking77 / CLINC.
"""
from __future__ import annotations

import json
import random
import zipfile

from cls.data.typed import (
    CLINC_HELD_OUT,
    NLI_LABELS,
    _choice,
    _load,
    _subsample_options,
    humanize,
)

SIZES = {"synthetic": 600_000, "utcd": 300_000, "label_nli": 300_000, "nli": 300_000}
NLI_SHARE = {"mnli": 0.35, "snli": 0.25, "wanli": 0.2, "fever": 0.2}
MAX_OPTS = 24  # 2-24 options per question, as in Tev1 and the design doc
SYNTHETIC_CONFIGS = ["mixtral_written_texts_for_tasks", "mixtral_written_texts_for_tasks_v2",
                     "mixtral_written_texts_for_tasks_v3", "mixtral_written_texts_for_tasks_v4"]
UTCD_IN = ["go_emotion", "sentiment_tweets_2020", "emotion", "sgd", "clinc_150", "slurp", "ag_news", "dbpedia", "yahoo"]
UTCD_OUT = ["amazon_polarity", "finance_sentiment", "yelp", "snips", "nlu_evaluation", "multi_eurlex", "patent",
            "consumer_finance"]  # UTCD's banking77 is replaced by banking77_val (final-test format)
UTCD_HELD_OUT = {"clinc_150": {humanize(x) for x in CLINC_HELD_OUT} | {"exchange rate"},  # also a Banking77 label
                 "sgd": {"Check Balance", "Transfer Money"}}
UTCD_ASPECT = {**dict.fromkeys(["go_emotion", "sentiment_tweets_2020", "emotion", "amazon_polarity",
                                "finance_sentiment", "yelp"], "sentiment"),
               **dict.fromkeys(["sgd", "clinc_150", "slurp", "banking77", "snips", "nlu_evaluation"], "intent"),
               **dict.fromkeys(["ag_news", "dbpedia", "yahoo", "multi_eurlex", "patent", "consumer_finance"], "topic")}
# a few wordings per aspect: M1 showed accuracy moves 0.91 -> 0.59 when only the instruction changes
INSTRUCTIONS = {
    "sentiment": ["What sentiment or emotion does the text express?", "How does the writer feel?",
                  "Which feeling best describes this text?"],
    "intent": ["What does the user want to do?", "What is the intent of this message?",
               "Which request is the user making?"],
    "topic": ["What is the topic of this text?", "Which category does this text belong to?",
              "What is this text about?"],
}
NLI_Q = 'How does the text relate to the statement: "{}"?'


def _k(n_labels: int, rng: random.Random) -> int:
    return rng.randint(min(2, n_labels), min(MAX_OPTS, n_labels))


# ---------------------------------------------------------------- synthetic zero-shot tasks
def synthetic(n: int, rng: random.Random) -> list[dict]:
    """Rows are (text, hypothesis, labels) with labels 0 = entailment; keep texts with exactly one true hypothesis
    and use the task's full hypothesis list as the option pool."""
    gold: dict[str, tuple] = {}
    bad: set[str] = set()
    for cfg in SYNTHETIC_CONFIGS:
        ds = _load("MoritzLaurer/synthetic_zeroshot_mixtral_v0.1", cfg, "train")
        ds = ds.select_columns(["text", "hypothesis", "labels", "task_description", "task_hypotheses"])
        for r in ds.filter(lambda r: r["labels"] == 0):
            if r["text"] in gold:
                bad.add(r["text"])
            gold[r["text"]] = (r["hypothesis"], r["task_hypotheses"], r["task_description"])
    out = []
    for text, (hyp, pool, desc) in gold.items():
        pool = list(dict.fromkeys(pool))
        if text in bad or hyp not in pool or len(pool) < 2:
            continue
        opts, g = _subsample_options(pool, pool.index(hyp), _k(len(pool), rng), rng)
        instr = desc.split(". ")[0].rstrip(".") + "?"  # first sentence only: later ones list the classes
        out.append(_choice("synthetic", text.strip(), instr, opts, g))
    return rng.sample(out, min(n, len(out)))


# ---------------------------------------------------------------- UTCD
def _utcd_zip() -> zipfile.ZipFile:
    from huggingface_hub import hf_hub_download

    # utcd.py is a loading script, which current `datasets` no longer runs; read its data zip directly
    return zipfile.ZipFile(hf_hub_download("claritylab/utcd", "datasets.zip", repo_type="dataset"))


def _utcd_items(z, name: str, split: str, n: int, rng: random.Random, all_options: bool) -> list[dict]:
    domain = "in-domain_split" if name in UTCD_IN else "out-of-domain_split"
    data = json.loads(z.read(f"{domain}/{name}/{split}.json"))  # {text: [labels]}
    drop = UTCD_HELD_OUT.get(name, set())
    single = [(t, ls[0]) for t, ls in data.items() if len(ls) == 1 and ls[0] not in drop]
    labels = sorted({y for _, y in single})
    pick = rng.sample(single, min(n, len(single)))
    out = []
    for text, y in pick:
        if all_options:
            opts, g = labels, labels.index(y)
        else:
            opts, g = _subsample_options(labels, labels.index(y), _k(len(labels), rng), rng)
        out.append(_choice(f"utcd/{name}", text, rng.choice(INSTRUCTIONS[UTCD_ASPECT[name]]), list(opts), g))
    return out


def utcd_train(n: int, rng: random.Random, z=None) -> list[dict]:
    z = z or _utcd_zip()
    return [it for name in UTCD_IN for it in _utcd_items(z, name, "train", n // len(UTCD_IN), rng, False)]


def utcd_ood_val(per_set: int, rng: random.Random, z=None) -> list[dict]:
    """UTCD out-of-domain test sets, every label as an option; tasks `utcd/<set>` until build_stage1 tags them."""
    z = z or _utcd_zip()
    return [it for name in UTCD_OUT for it in _utcd_items(z, name, "test", per_set, rng, True)]


def banking77_val(n: int, seed: int) -> list[dict]:
    """Banking77 *train* slice in the exact format of scripts/eval_banking77.py (humanized labels, B77 instruction)."""
    from cls.data.banking77 import INSTRUCTION, load_banking77

    b = load_banking77("train", n=n, seed=seed)
    return [_choice("zs/banking77", t, INSTRUCTION, list(b.descriptions), y) for t, y in zip(b.texts, b.labels)]


def tag_zero_shot(val: list[dict], train: list[dict]) -> list[dict]:
    """utcd/<set> -> zs/<set> if no label of the set is a training option (case-insensitive), else ood/<set>.
    Sets already tagged zs/ (Banking77) must share no label with training."""
    seen = {t.lower() for it in train for t in it.get("criteria", [])}
    labels: dict[str, set] = {}
    for it in val:
        labels.setdefault(it["task"], set()).update(t.lower() for t in it["criteria"])
    for it in val:
        task = it["task"]
        shared = labels[task] & seen
        if task.startswith("zs/"):
            assert not shared, f"{task} shares labels with training: {sorted(shared)[:5]}"
        else:
            it["task"] = ("ood/" if shared else "zs/") + task.split("/", 1)[1]
    return val


# ---------------------------------------------------------------- NLI-style
def label_nli(n: int, rng: random.Random) -> list[dict]:
    ds = _load("tasksource/zero-shot-label-nli", None, "train")
    ds = ds.filter(lambda r: r["license_use"] != "non-commercial"
                   and not any(w in r["task"].lower() for w in ("banking", "clinc")))
    idx = rng.sample(range(len(ds)), min(n, len(ds)))
    return [{"task": "label_nli", "type": "noul", "state": r["premise"],
             "instructions": f'Does this statement fit the text: "{r["hypothesis"]}"?', "gold": int(r["labels"] == 0)}
            for r in ds.select(sorted(idx))]


def nli(n: int, rng: random.Random) -> list[dict]:
    """(state, statement, label) triples; FEVER's `premise` is the claim and `hypothesis` the evidence, so swap."""
    sources = {
        "mnli": (("nyu-mll/multi_nli", None), lambda r: (r["premise"], r["hypothesis"], r["label"])),
        "snli": (("stanfordnlp/snli", "plain_text"), lambda r: (r["premise"], r["hypothesis"], r["label"])),
        "wanli": (("alisawuffles/WANLI", None), lambda r: (r["premise"], r["hypothesis"], NLI_LABELS.index(r["gold"]))),
        "fever": (("pietrolesci/nli_fever", None), lambda r: (r["hypothesis"], r["premise"], r["label"])),
    }
    out = []
    for task, ((name, cfg), row) in sources.items():
        ds = _load(name, cfg, "train")
        idx = rng.sample(range(len(ds)), min(round(n * NLI_SHARE[task]), len(ds)))
        for r in ds.select(sorted(idx)):
            state, stmt, y = row(r)
            if y in (0, 1, 2):  # SNLI marks unlabelled pairs -1
                out.append(_choice(f"nli/{task}", state, NLI_Q.format(stmt.strip()), NLI_LABELS, y))
    return out


# ---------------------------------------------------------------- mixture
def build_stage1(seed: int = 0, scale: float = 1.0, val_per_set: int = 300) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    size = {k: max(1, round(v * scale)) for k, v in SIZES.items()}
    z = _utcd_zip()
    train = (synthetic(size["synthetic"], rng) + utcd_train(size["utcd"], rng, z)
             + label_nli(size["label_nli"], rng) + nli(size["nli"], rng))
    rng.shuffle(train)
    return train, tag_zero_shot(utcd_ood_val(val_per_set, rng, z) + banking77_val(val_per_set, seed), train)


def leak_report(train: list[dict], b77_labels: list[str]) -> dict:
    """Option strings shared with Banking77's labels, and train states that are Banking77 texts verbatim."""
    from cls.data.banking77 import load_banking77

    b77 = {x.lower() for x in b77_labels}
    opts = {t.lower() for it in train for t in it.get("criteria", [])}
    texts = {x.strip().lower() for split in ("train", "test") for x in load_banking77(split).texts}
    return {"shared_option_strings": sorted(opts & b77),
            "banking77_texts_in_train": sum(it["state"].strip().lower() in texts for it in train)}
