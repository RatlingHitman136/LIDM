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
