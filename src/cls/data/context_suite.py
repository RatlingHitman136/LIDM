"""Zero-shot context-understanding suite: tasks where the answer depends on reading the state, which plain
embedding retrieval cannot do. Same item schema as cls.data.typed ({task, type, state, instructions, criteria, gold}).

Regime 1 = the content to check is in the question (pass A sees it jointly with the state; options are trivial).
Regime 2 = the content is in the options (options see the state only through the head's cross-attention), the
case where the late split can lose reasoning that a joint encoder keeps.

Evaluation only. Several sources are non-commercial (RACE, QuAIL, ANLI) and must not be used for training.
"""
from __future__ import annotations

import ast
import json
import random

from cls.data.typed import NLI_LABELS, _boolq, _choice, _load

# task -> (group, regime, description)
TASKS = {
    "typed_decisions": ("1 system one", "mixed", "LocalLLaMA/typed-decisions test, 400 cases x 5 questions"),
    "race": ("2 content in options", "2", "RACE reading comprehension, 4 answer options"),
    "quail": ("2 content in options", "2", "QuAIL reading comprehension, 4 answer options"),
    "when2call": ("2 content in options", "2", "When2Call MCQ: pick the right response given tools and request"),
    "condaqa": ("3 negation / contrast", "1", "CondaQA: questions about negated statements, yes / no / don't know"),
    "vitaminc": ("3 negation / contrast", "1", "VitaminC: one changed fact flips supports / refutes"),
    "boolq": ("4 content in question", "1", "BoolQ validation, yes/no"),
    "pubmedqa": ("4 content in question", "1", "PubMedQA labelled, yes / no / maybe"),
    "squad2_answerable": ("4 content in question", "1", "SQuAD 2.0: can the question be answered from the passage"),
    "anli": ("4 content in question", "1", "ANLI rounds 1-3 test"),
    "wanli": ("4 content in question", "1", "WANLI test"),
}
REGIME2 = {t for t, (_, r, _) in TASKS.items() if r in ("2", "mixed")}


def typed_decisions_items(row) -> list[dict]:
    """One item per question of a case. choice: options `name: description`, as the benchmark ships them;
    noul: the statement as question; score: level descriptions in order. Gold = the label of the soft target;
    `soft` = the soft target in option order, `gold_score` = the expected level of a score question."""
    questions, gold = json.loads(row["questions"]), json.loads(row["gold"])
    out = []
    for qid, q in questions.items():
        g, kind = gold[qid], q["type"]
        base = {"task": "typed_decisions", "state": row["state"], "instructions": q["instructions"], "qid": qid,
                "case": row.get("id"), "workflow": row.get("workflow")}
        probs = g.get("probabilities", {})
        if kind == "choice":
            names = list(q["criteria"])
            out.append(dict(base, type="choice", criteria=[f"{n}: {q['criteria'][n]}" for n in names],
                            names=names, criteria_dict=q["criteria"], gold=names.index(g["label"]),
                            soft=[probs.get(n, 0.0) for n in names]))
        elif kind == "noul":
            out.append(dict(base, type="noul", gold=int(g["label"] == "true"),
                            soft=[probs.get("false", 0.0), probs.get("true", 0.0)]))
        elif kind == "score":
            k = len(q["criteria"])
            out.append(dict(base, type="score", criteria=list(q["criteria"]), gold=int(g["label"]),
                            soft=[probs.get(str(i), 0.0) for i in range(k)], gold_score=g.get("score")))
    return out


def _letter(answer: str) -> int:
    return "ABCD".index(answer.strip())


def race_item(row) -> dict:
    return _choice("race", row["article"], row["question"], list(row["options"]), _letter(row["answer"]))


def quail_item(row) -> dict:
    return _choice("quail", row["context"], row["question"], list(row["answers"]), int(row["correct_answer_id"]))


WHEN2CALL_KEYS = ["direct", "tool_call", "request_for_info", "cannot_answer"]


def when2call_item(row) -> dict:
    """State = the user request first (so truncation cuts the tool list, not the request), then the tools."""
    answers = row["answers"] if isinstance(row["answers"], dict) else ast.literal_eval(row["answers"])
    tools = row["tools"] if isinstance(row["tools"], list) else ast.literal_eval(row["tools"])
    names = []
    for t in tools:
        try:
            spec = json.loads(t)
            names.append(f"{spec.get('name')}: {spec.get('description', '')}")
        except (TypeError, ValueError):
            names.append(str(t)[:200])
    state = f"User request: {row['question']}\n\nAvailable tools:\n" + ("\n".join(names) or "(none)")
    keys = [k for k in WHEN2CALL_KEYS if answers.get(k)]
    return _choice("when2call", state, "Which response should the assistant give?",
                   [answers[k] for k in keys], keys.index(row["correct_answer"]))


CONDAQA_LABELS = {"YES": 0, "NO": 1, "DONT KNOW": 2}


def condaqa_item(row) -> dict | None:
    """None for span answers (CondaQA also has a few), which are not a choice."""
    y = CONDAQA_LABELS.get(row["label"].upper().replace("'", "").replace("\u2019", "").strip())
    return None if y is None else _choice("condaqa", row["sentence1"], row["sentence2"], ["yes", "no", "don't know"], y)


VITAMINC_LABELS = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


def vitaminc_item(row) -> dict:
    return _choice("vitaminc", row["evidence"], f'Does the text support or refute the claim: "{row["claim"].strip()}"?',
                   ["supports the claim", "refutes the claim", "not enough information"], VITAMINC_LABELS[row["label"]])


def pubmedqa_item(row) -> dict:
    ctx = row["context"] if isinstance(row["context"], dict) else ast.literal_eval(row["context"])
    return _choice("pubmedqa", " ".join(ctx["contexts"]), row["question"], ["yes", "no", "maybe"],
                   ["yes", "no", "maybe"].index(row["final_decision"]))


def squad2_item(row) -> dict:
    answers = row["answers"] if isinstance(row["answers"], dict) else ast.literal_eval(row["answers"])
    return {"task": "squad2_answerable", "type": "noul", "state": row["context"],
            "instructions": f'Can the question "{row["question"].strip()}" be answered from this passage?',
            "gold": int(bool(answers["text"]))}


def nli_item(task: str, premise: str, hypothesis: str, label: int) -> dict:
    return _choice(task, premise, f'How does the text relate to the statement: "{hypothesis.strip()}"?', NLI_LABELS, label)


def with_options_in_question(item: dict) -> dict:
    """Ablation: also write every option's text into the question, so pass A encodes state, question and options
    jointly (the information a Laya-style joint encoder has). Options in pass B are unchanged."""
    if item["type"] == "noul":
        return item
    listed = "\n".join(f"- {c}" for c in item["criteria"])
    return dict(item, instructions=f"{item['instructions']}\nOptions:\n{listed}")


def _sample(rows: list, n: int | None, rng: random.Random) -> list:
    return rows if n is None or n >= len(rows) else rng.sample(rows, n)


def load_suite(n: int | None = 1000, seed: int = 0, tasks=None) -> dict[str, list[dict]]:
    """Every task capped at n questions (typed-decisions: all 400 cases, ~2,000 questions)."""
    rng = random.Random(seed)
    want = set(tasks or TASKS)
    out: dict[str, list[dict]] = {}
    if "typed_decisions" in want:
        out["typed_decisions"] = [it for r in _load("LocalLLaMA/typed-decisions", "all", "test") for it in typed_decisions_items(r)]
    if "race" in want:
        out["race"] = [race_item(r) for r in _sample(list(_load("ehovy/race", "all", "test")), n, rng)]
    if "quail" in want:
        out["quail"] = [quail_item(r) for r in _sample(list(_load("textmachinelab/quail", "quail", "validation")), n, rng)]
    if "when2call" in want:
        out["when2call"] = [when2call_item(r) for r in _sample(list(_load("nvidia/When2Call", "test", "mcq")), n, rng)]
    if "condaqa" in want:
        rows = [it for it in map(condaqa_item, _load("lasha-nlp/CONDAQA", None, "test")) if it]
        out["condaqa"] = _sample(rows, n, rng)
    if "vitaminc" in want:
        out["vitaminc"] = [vitaminc_item(r) for r in _sample(list(_load("tals/vitaminc", None, "test")), n, rng)]
    if "boolq" in want:
        out["boolq"] = [_boolq(r) for r in _sample(list(_load("google/boolq", None, "validation")), n, rng)]
    if "pubmedqa" in want:
        out["pubmedqa"] = [pubmedqa_item(r) for r in _sample(list(_load("qiaojin/PubMedQA", "pqa_labeled", "train")), n, rng)]
    if "squad2_answerable" in want:
        out["squad2_answerable"] = [squad2_item(r) for r in _sample(list(_load("rajpurkar/squad_v2", None, "validation")), n, rng)]
    if "anli" in want:
        rows = [r for rd in ("test_r1", "test_r2", "test_r3") for r in _load("facebook/anli", "plain_text", rd)]
        out["anli"] = [nli_item("anli", r["premise"], r["hypothesis"], r["label"]) for r in _sample(rows, n, rng)]
    if "wanli" in want:
        rows = list(_load("alisawuffles/WANLI", None, "test"))
        out["wanli"] = [nli_item("wanli", r["premise"], r["hypothesis"], NLI_LABELS.index(r["gold"])) for r in _sample(rows, n, rng)]
    return out
