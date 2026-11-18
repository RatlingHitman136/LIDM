import io
import json
import random
import zipfile

import pytest

from cls.data.stage1 import _utcd_items, tag_zero_shot


def fake_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("in-domain_split/clinc_150/train.json", json.dumps({
            "turn on the alarm": ["alarm"], "what is my apr": ["apr"],  # apr is held out (banking)
            "two intents": ["alarm", "timer"], "set a timer": ["timer"], "translate hi": ["translate"]}))
    return zipfile.ZipFile(buf)


def test_utcd_items_drop_held_out_and_multi_label_keep_gold():
    items = _utcd_items(fake_zip(), "clinc_150", "train", 10, random.Random(0), all_options=True)
    assert sorted(it["state"] for it in items) == ["set a timer", "translate hi", "turn on the alarm"]
    gold = {"set a timer": "timer", "translate hi": "translate", "turn on the alarm": "alarm"}
    for it in items:
        assert it["criteria"] == ["alarm", "timer", "translate"] and it["criteria"][it["gold"]] == gold[it["state"]]


def test_tag_zero_shot_splits_sets_by_label_overlap():
    train = [{"criteria": ["Positive", "alarm set"]}, {"type": "noul"}]
    val = [{"task": "utcd/yelp", "criteria": ["negative", "positive"]},
           {"task": "utcd/patent", "criteria": ["chemistry", "physics"]},
           {"task": "zs/banking77", "criteria": ["card arrival"]}]
    assert [it["task"] for it in tag_zero_shot(val, train)] == ["ood/yelp", "zs/patent", "zs/banking77"]
    with pytest.raises(AssertionError):
        tag_zero_shot([{"task": "zs/banking77", "criteria": ["alarm set"]}], train)


def test_refinedweb_items_use_own_negatives_and_skip_other_trues():
    import pandas as pd

    from cls.data.stage1 import refinedweb_items
    df = pd.DataFrame({"text": ["t1"] * 4 + ["t2"] * 3 + ["t3"],
                       "option": ["a", "b", "wrong1", "banned x", "c", "d", "wrong2", "lonely false"],
                       "labels": [0, 0, 1, 0, 0, 1, 1, 1]})
    for seed in range(20):
        items = refinedweb_items(df, "rw", ["Q?"], random.Random(seed), banned=lambda o: o == "banned x")
        assert [it["state"] for it in items] == ["t1", "t2"]  # t3 has no true option
        for it in items:
            gold, opts = it["criteria"][it["gold"]], it["criteria"]
            assert len(set(opts)) == len(opts) and "banned x" not in opts
            other_true = {"t1": {"a", "b"}, "t2": {"c"}}[it["state"]] - {gold}
            assert not other_true & set(opts)
            assert gold in {"a", "b", "c"}


def test_cross_task_distractors_keep_gold_and_draw_from_other_tasks():
    from cls.data.stage1 import cross_task_distractors
    items = [{"task": "x", "criteria": ["x1", "x2", "x3", "x4", "x5"], "gold": 2},
             {"task": "y", "criteria": ["y1", "y2", "y3", "y4", "y5"], "gold": 0}]
    cross_task_distractors(items, random.Random(0), p=1.0)
    assert items[0]["criteria"][2] == "x3" and items[1]["criteria"][0] == "y1"
    assert sum(o.startswith("y") for o in items[0]["criteria"]) == 2 and len(set(items[0]["criteria"])) == 5


def test_make_ban_catches_exact_and_word_superset_labels():
    from cls.data.stage1 import make_ban
    banned = make_ban(["request refund", "Credit card", "patent"])
    assert banned("Request Refund") and banned("request for refund") and banned("credit card payment security")
    assert banned("Patent") and not banned("patents and licensing") and not banned("refund policy")


def test_cross_task_distractors_terminate_with_a_single_task():
    """Regression: one task means no foreign labels; this used to loop forever (stage-1b build, 2026-11-17)."""
    from cls.data.stage1 import cross_task_distractors
    items = [{"task": "synthetic", "criteria": ["a", "b", "c", "d"], "gold": 1},
             {"task": "synthetic", "criteria": ["e", "f", "g"], "gold": 0}]
    cross_task_distractors(items, random.Random(0), p=1.0)
    assert items[0]["criteria"] == ["a", "b", "c", "d"] and items[1]["criteria"] == ["e", "f", "g"]
