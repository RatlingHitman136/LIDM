"""Stage 2 (typed decisions), commercial-clean mix for the same-data LIDM vs Laya comparison.

Every source becomes canonical *cases* {source, state, questions: {qid: {type, instructions, criteria}}, expected:
{qid: hard label}} — Laya's native training row — and `lidm_items` turns a case into one LIDM item per question.
Hard labels for both models (argmax of the source target; ambiguous targets dropped).

Held out of training (evaluation sets): LocalLLaMA/typed-decisions and its four workflows' domains, every context-
suite task, Banking77, CLINC150, SST-5. Sources: typed-decisions-synth (MIT), Open-Jev release-v2 (CC0), tasksource-jev
rows with license_use == "commercial", HelpSteer2 (CC BY 4.0), Civil Comments (CC0), Aegis 2.0 (CC BY 4.0).
"""
from __future__ import annotations

import collections
import json
import random
import re

from cls.data.typed import _load

EVAL_SOURCES = re.compile(r"boolq|sst|banking|clinc|(^|[^a-z])race([^a-z]|$)|quail|cosmos|when2call|condaqa|vitaminc|"
                          r"pubmed|squad|anli|wanli|typed.decisions|massive|hwu|snips", re.IGNORECASE)
TD_DOMAINS = re.compile(r"agent.?trace|observab|invoice|security.?incident|customer.?service|customer.?support|"
                        r"support.?ticket|customer-control", re.IGNORECASE)
# many and long options are kept on purpose: that is where late interaction should beat a packed joint encoder
MAX_OPTS, MAX_OPT_CHARS, MAX_ALL_OPT_CHARS, MAX_STATE_CHARS = 40, 600, 6000, 3000


def _ok_options(opts) -> bool:
    opts = list(opts)
    return (2 <= len(opts) <= MAX_OPTS and len(set(opts)) == len(opts) and max(map(len, opts)) <= MAX_OPT_CHARS
            and sum(map(len, opts)) <= MAX_ALL_OPT_CHARS)


def _case(source, state, questions, expected) -> dict | None:
    keep = {}
    for qid, q in questions.items():
        if qid not in expected or len(q.get("instructions", "")) > 400:
            continue
        crit = q.get("criteria")
        if q["type"] in ("choice", "score") and not _ok_options(crit):
            continue
        if q["type"] == "choice" and expected[qid] not in crit:
            continue
        if q["type"] == "score" and not 0 <= int(expected[qid]) < len(crit):
            continue
        keep[qid] = q
    if not keep or not state:
        return None
    return {"source": source, "state": str(state)[:MAX_STATE_CHARS], "questions": keep,
            "expected": {k: expected[k] for k in keep}}


def jev_case(source: str, r: dict) -> dict | None:
    """One jev-format row (kind, options, target, state, question) -> a one-question case. noul comes as options
    [no, yes] with target [P(no), P(yes)] (Open-Jev) or no options and target [P(true)] (tasksource-jev)."""
    kind, opts, t = r["kind"], list(r["options"] or []), [float(x) for x in r["target"]]
    if kind == "noul":
        p = t[-1] if len(t) in (1, 2) else None
        if p is None or 0.35 < p < 0.65:
            return None
        q, exp = {"type": "noul", "instructions": r["question"]}, p >= 0.5
    elif kind in ("choice", "score") and len(t) == len(opts):
        top = sorted(t, reverse=True)
        if top[0] < 0.5 or top[0] == top[1]:
            return None
        i = t.index(top[0])
        q = {"type": kind, "instructions": r["question"], "criteria": opts}
        exp = opts[i] if kind == "choice" else i
    else:
        return None
    state = r.get("state") or r.get("state_json")
    if isinstance(state, str) and state.startswith('"'):
        try:
            state = json.loads(state)
        except ValueError:
            pass
    return _case(source, state, {"q": q}, {"q": exp})


def tdsynth_cases(rng: random.Random) -> list[dict]:
    out = []
    for r in _load("n4ze3m/typed-decisions-synth", None, "train"):
        if TD_DOMAINS.search(r["domain"]):
            continue
        qs, gold = json.loads(r["questions"]), json.loads(r["gold"])
        exp = {k: (str(v) if qs.get(k, {}).get("type") == "choice" else v) for k, v in gold.items()}
        c = _case("tdsynth", r["state"], qs, exp)
        if c:
            out.append(c)
    return out


def openjev_cases(rng: random.Random) -> list[dict]:
    rows = [r for r in _load("ZefanCai/Open-Jev", "release-v2-redistributable", "train") if not TD_DOMAINS.search(r["source"])]
    return [c for c in (jev_case(f"openjev/{r['source']}", r) for r in rows) if c]


def _rich(opts) -> bool:
    """Many options or long ones: the regime the comparison is about."""
    opts = opts or []
    return len(opts) > 8 or (bool(opts) and sum(map(len, opts)) / len(opts) > 80)


def tasksource_cases(rng: random.Random, per_source: int = 250, heldout_share: float = 0.1,
                     heldout_per_source: int = 100) -> tuple[list[dict], list[dict]]:
    """Commercial rows of every non-evaluation source, up to per_source each, half of them option-rich where the
    source has such rows. heldout_share of the sources are held out entirely (new label sets) and returned apart."""
    ds = _load("tasksource/tasksource-jev", "default", "train")
    meta = ds.select_columns(["source", "license_use", "options"])
    by_src = collections.defaultdict(lambda: ([], []))
    for i, (src, lic, opts) in enumerate(zip(meta["source"], meta["license_use"], meta["options"])):
        if lic == "commercial" and not EVAL_SOURCES.search(src) and not TD_DOMAINS.search(src):
            by_src[src][int(_rich(opts))].append(i)
    sources = sorted(by_src)
    held = set(rng.sample(sources, round(heldout_share * len(sources))))

    def pick(src, k):
        plain, rich = by_src[src]
        r = rng.sample(rich, min(len(rich), k // 2 * 3))  # oversample: some rows fail the filters
        return r + rng.sample(plain, min(len(plain), 3 * k - len(r)))

    idx = {j: src for src in sources for j in pick(src, heldout_per_source if src in held else per_source)}
    cases = collections.defaultdict(list)
    for j, r in zip(sorted(idx), ds.select(sorted(idx))):
        c = jev_case(f"tasksource/{r['source']}", r)
        if c:
            cases[r["source"]].append(c)
    take = lambda s: cases[s][:heldout_per_source if s in held else per_source]
    return ([c for s in sources if s not in held for c in take(s)], [c for s in sorted(held) for c in take(s)])


HELPSTEER = {
    "helpfulness": ("How helpful is the response to the prompt?",
                    ["Not helpful at all", "Slightly helpful", "Moderately helpful", "Very helpful", "Extremely helpful"]),
    "correctness": ("How factually correct and complete is the response?",
                    ["Mostly incorrect", "Several errors", "Some errors or omissions", "Minor issues", "Fully correct"]),
    "coherence": ("How clear and coherent is the response?",
                  ["Incoherent", "Mostly unclear", "Somewhat clear", "Mostly clear", "Perfectly clear and consistent"]),
    "complexity": ("How sophisticated is the language of the response?",
                   ["Basic, anyone could write it", "Simple", "Moderate vocabulary", "Advanced", "Expert-level"]),
    "verbosity": ("How verbose is the response relative to what was asked?",
                  ["Very terse", "Brief", "Moderate length", "Detailed", "Very verbose"]),
}


def helpsteer_cases(rng: random.Random, n: int = 3000) -> list[dict]:
    rows = list(_load("nvidia/HelpSteer2", None, "train"))
    out = []
    for r in rng.sample(rows, min(n, len(rows))):
        qs = {k: {"type": "score", "instructions": ins, "criteria": lv} for k, (ins, lv) in HELPSTEER.items()}
        c = _case("helpsteer2", f"Prompt: {r['prompt']}\n\nResponse: {r['response']}", qs, {k: int(r[k]) for k in HELPSTEER})
        if c:
            out.append(c)
    return out


def civil_cases(rng: random.Random, n: int = 8000, pos_share: float = 0.35) -> list[dict]:
    ds = _load("google/civil_comments", None, "train")
    tox = ds["toxicity"]
    pos = [i for i, t in enumerate(tox) if t >= 0.6]
    neg = [i for i, t in enumerate(tox) if t <= 0.3]
    idx = rng.sample(pos, int(n * pos_share)) + rng.sample(neg, n - int(n * pos_share))
    q = {"toxic": {"type": "noul", "instructions": "This comment is toxic: rude, disrespectful or unreasonable."}}
    return [c for i in idx if (c := _case("civil_comments", ds[i]["text"], q, {"toxic": tox[i] >= 0.6}))]


def aegis_cases(rng: random.Random, n: int = 7000) -> list[dict]:
    rows = list(_load("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "train"))
    out = []
    for r in rng.sample(rows, min(n, len(rows))):
        resp = r.get("response")
        has_resp = resp not in (None, "", "None") and r.get("response_label") in ("safe", "unsafe")
        state = f"User: {r['prompt']}" + (f"\n\nAssistant: {resp}" if has_resp else "")
        qs = {"prompt_unsafe": {"type": "noul", "instructions": "The user's message is unsafe or harmful."}}
        exp = {"prompt_unsafe": r["prompt_label"] == "unsafe"}
        if has_resp:
            qs["response_unsafe"] = {"type": "noul", "instructions": "The assistant's response is unsafe or harmful."}
            exp["response_unsafe"] = r["response_label"] == "unsafe"
        c = _case("aegis2", state, qs, exp)
        if c:
            out.append(c)
    return out


def lidm_items(case: dict) -> list[dict]:
    """One LIDM item per question; dict criteria become `name: description` options, as for typed-decisions."""
    out = []
    for qid, q in case["questions"].items():
        base = {"task": f"s2/{case['source'].split('/')[0]}", "type": q["type"], "state": case["state"],
                "instructions": q["instructions"], "qid": qid}
        exp, crit = case["expected"][qid], q.get("criteria")
        if q["type"] == "noul":
            out.append(dict(base, gold=int(bool(exp))))
        elif q["type"] == "score":
            out.append(dict(base, criteria=[str(c) for c in crit], gold=int(exp)))
        else:
            names = list(crit)
            opts = [f"{k}: {crit[k]}" for k in names] if isinstance(crit, dict) else [str(c) for c in names]
            out.append(dict(base, criteria=opts, gold=names.index(exp)))
    return out
