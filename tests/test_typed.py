import random

from cls.data.typed import _subsample_options, humanize, to_laya_rows

ITEMS = [
    {"task": "ag_news", "type": "choice", "state": "s", "instructions": "i", "criteria": ["world", "sports"], "gold": 1},
    {"task": "boolq", "type": "noul", "state": "s", "instructions": "i?", "gold": 0},
    {"task": "sst5", "type": "score", "state": "s", "instructions": "i", "criteria": ["bad", "ok", "good"], "gold": 2},
]


def test_subsample_keeps_gold():
    rng = random.Random(0)
    names = [str(i) for i in range(50)]
    for g in range(50):
        opts, ng = _subsample_options(names, g, 7, rng)
        assert len(opts) == len(set(opts)) == 7 and opts[ng] == names[g]


def test_humanize():
    assert humanize("card_arrival") == "card arrival"
    assert humanize("EducationalInstitution") == "educational institution"
    assert humanize("oos") == "out of scope"


def test_laya_rows_parse_to_gold_targets():
    from laya.train import target_from_expected, to_internal

    for it, row in zip(ITEMS, to_laya_rows(ITEMS)):
        (qid, q), = row["questions"].items()
        target = target_from_expected(to_internal(qid, q), row["expected"][qid])
        assert target.index(1.0) == it["gold"]
