"""Stage-2 converters on rows shaped like the real sources (no downloads)."""
from cls.data import stage2


def test_jev_noul_conventions_and_ambiguity():
    oj = stage2.jev_case("openjev/x", {"kind": "noul", "options": ["no", "yes"], "target": [0.1, 0.9], "question": "Q?", "state": "S"})
    ts = stage2.jev_case("tasksource/y", {"kind": "noul", "options": [], "target": [0.0], "question": "Q?", "state": "S"})
    assert oj["expected"]["q"] is True and ts["expected"]["q"] is False
    assert stage2.jev_case("x", {"kind": "noul", "options": [], "target": [0.5], "question": "Q?", "state": "S"}) is None


def test_jev_choice_score_and_filters():
    c = stage2.jev_case("x", {"kind": "choice", "options": ["a", "b", "c"], "target": [0.1, 0.8, 0.1], "question": "Q?", "state": "S"})
    assert c["expected"]["q"] == "b" and stage2.lidm_items(c)[0]["gold"] == 1
    s = stage2.jev_case("x", {"kind": "score", "options": ["lo", "mid", "hi"], "target": [0, 0, 1], "question": "Q?", "state": "S"})
    assert s["expected"]["q"] == 2 and stage2.lidm_items(s)[0]["criteria"] == ["lo", "mid", "hi"]
    tie = {"kind": "choice", "options": ["a", "b"], "target": [0.5, 0.5], "question": "Q?", "state": "S"}
    long = {"kind": "choice", "options": ["a" * 700, "b"], "target": [1, 0], "question": "Q?", "state": "S"}
    assert stage2.jev_case("x", tie) is None and stage2.jev_case("x", long) is None


def test_lidm_items_dict_criteria_and_noul():
    case = {"source": "tdsynth", "state": "S", "questions": {
        "act": {"type": "choice", "instructions": "Do?", "criteria": {"go": "Proceed.", "stop": "Halt."}},
        "flag": {"type": "noul", "instructions": "Flag."}}, "expected": {"act": "stop", "flag": True}}
    act, flag = stage2.lidm_items(case)
    assert act["criteria"] == ["go: Proceed.", "stop: Halt."] and act["gold"] == 1 and flag["gold"] == 1
    assert act["task"] == "s2/tdsynth"


def test_eval_and_td_filters():
    assert stage2.EVAL_SOURCES.search("super_glue/boolq") and stage2.EVAL_SOURCES.search("race")
    assert not stage2.EVAL_SOURCES.search("embrace-dataset") and not stage2.EVAL_SOURCES.search("trace-rules")
    assert stage2.TD_DOMAINS.search("customer-control-v1") and stage2.TD_DOMAINS.search("Invoice processing")


def test_micro_batches_split_on_tokens_and_keep_every_item():
    from cls.train import micro_batches
    small = {"type": "noul", "state": "x" * 40, "instructions": "q"}
    big = {"type": "choice", "state": "y" * 4000, "instructions": "q", "criteria": [str(i) for i in range(40)]}
    chunk = [small] * 5 + [big] + [small] * 3
    plain = list(micro_batches(chunk, 4))
    assert [len(m) for m in plain] == [4, 4, 1]
    split = list(micro_batches(chunk, 4, max_tokens=10000, state_copies=True))
    assert sum(map(len, split)) == len(chunk) and [big] in split  # the 40-copy question gets its own micro-batch
    assert all(len(m) <= 4 for m in split)
