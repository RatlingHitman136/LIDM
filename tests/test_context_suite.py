"""Item builders of the context suite on rows shaped like the real datasets (no downloads)."""
import json

from cls.data import context_suite as cs


def test_typed_decisions_items():
    row = {"state": '{"task": "x"}', "questions": json.dumps({
        "action": {"type": "choice", "instructions": "What now?", "criteria": {"go": "Proceed.", "stop": "Halt."}},
        "flag": {"type": "noul", "instructions": "Needs review.", "criteria": {"false": "No.", "true": "Yes."}},
        "risk": {"type": "score", "instructions": "How risky?", "criteria": ["Low.", "Mid.", "High."]}}),
        "gold": json.dumps({"action": {"label": "stop", "probabilities": {"go": 0.4, "stop": 0.6}},
                            "flag": {"label": "true", "probabilities": {"false": 0.3, "true": 0.7}},
                            "risk": {"label": "2", "probabilities": {"1": 0.2, "2": 0.8}, "score": 1.8}}),
        "id": "cs_1", "workflow": "customer_service"}
    ch, nl, sc = cs.typed_decisions_items(row)
    assert ch["criteria"] == ["go: Proceed.", "stop: Halt."] and ch["gold"] == 1 and ch["names"] == ["go", "stop"]
    assert nl["type"] == "noul" and nl["gold"] == 1
    assert sc["type"] == "score" and sc["gold"] == 2 and sc["criteria"] == ["Low.", "Mid.", "High."]
    assert ch["soft"] == [0.4, 0.6] and nl["soft"] == [0.3, 0.7] and sc["soft"] == [0.0, 0.2, 0.8]
    assert sc["gold_score"] == 1.8 and ch["case"] == "cs_1" and ch["workflow"] == "customer_service"


def test_reading_comprehension_and_tools():
    r = cs.race_item({"article": "A", "question": "Q?", "options": ["a", "b", "c", "d"], "answer": "C"})
    assert r["gold"] == 2 and r["criteria"][2] == "c"
    q = cs.quail_item({"context": "A", "question": "Q?", "answers": ["a", "b", "c", "d"], "correct_answer_id": "3"})
    assert q["gold"] == 3
    w = cs.when2call_item({"question": "Order a latte", "correct_answer": "request_for_info",
                           "answers": {"direct": "Done!", "tool_call": '{"name": "f"}', "request_for_info": "Which size?",
                                       "cannot_answer": "I can't."},
                           "tools": ['{"name": "order", "description": "Orders a drink."}']})
    assert w["criteria"][w["gold"]] == "Which size?" and w["state"].startswith("User request: Order a latte")
    assert "order: Orders a drink." in w["state"]


def test_label_mappings():
    assert cs.condaqa_item({"sentence1": "P", "sentence2": "Q?", "label": "DON'T KNOW"})["gold"] == 2
    assert cs.condaqa_item({"sentence1": "P", "sentence2": "Q?", "label": "the castle"}) is None
    assert cs.vitaminc_item({"evidence": "E", "claim": "C", "label": "REFUTES"})["gold"] == 1
    assert cs.pubmedqa_item({"context": {"contexts": ["a", "b"]}, "question": "Q?", "final_decision": "maybe"})["gold"] == 2
    assert cs.squad2_item({"context": "C", "question": "Q", "answers": {"text": [], "answer_start": []}})["gold"] == 0
    assert cs.nli_item("anli", "P", "H", 2)["criteria"][2] == "contradiction"


def test_options_in_question_ablation():
    it = cs.race_item({"article": "A", "question": "Q?", "options": ["red", "blue", "c", "d"], "answer": "A"})
    ab = cs.with_options_in_question(it)
    assert ab["criteria"] == it["criteria"] and "- red\n- blue" in ab["instructions"] and ab["state"] == "A"
    nl = cs.squad2_item({"context": "C", "question": "Q", "answers": {"text": ["x"]}})
    assert cs.with_options_in_question(nl) is nl
