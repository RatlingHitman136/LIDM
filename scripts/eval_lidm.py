"""Evaluate an LIDM checkpoint on the M0 task set through cls.eval.harness, plus AG News and Emotion test sets.
Logs to MLflow `baselines` as lidm-<name>, so scripts/report.py lists it next to the M0 baselines. Batch-1
predictions, so the latency it logs is per question with a fresh label set (options encoded every call).

    uv run python scripts/eval_lidm.py checkpoints/stage1-Blora/last.pt --name stage1-Blora [--quick]
"""
import argparse

import torch

from cls.data.typed import _choice, _load, humanize
from cls.eval import harness
from cls.train import encode, load_model

ap = argparse.ArgumentParser()
ap.add_argument("ckpt")
ap.add_argument("--name", required=True, help="run name becomes lidm-<name>")
ap.add_argument("--id", default="L1", help="baseline_id tag for the report table")
ap.add_argument("--quick", action="store_true", help="200 items per task")
ap.add_argument("--no-track", action="store_true")
a = ap.parse_args()

enc, m, prior, st = load_model(a.ckpt)
c = st["config"]


@torch.no_grad()
def predict(item):
    inputs, _, _, cos = encode(enc, [item], prior)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(*inputs, prior=cos)
    return logits[0].float().softmax(-1).cpu().numpy()


def test_items(task, name, cfg, instruction, n):
    """Test split with every label as an option; same instruction and label wording as the M0 mixture."""
    ds = _load(name, cfg, "test")
    names = [humanize(x) for x in ds.features["label"].names]
    rows = ds if n is None else ds.select(range(n))
    return [_choice(task, r["text"], instruction, names, r["label"]) for r in rows]


n = 200 if a.quick else None
tasks = harness.default_tasks(a.quick)
tasks["ag_news_test"] = test_items("ag_news", "fancyzhx/ag_news", None, "What is the topic of this news article?", n)
tasks["emotion_test"] = test_items("emotion", "dair-ai/emotion", "split", "Which emotion does the text express?", n)
params = {"checkpoint": a.ckpt, "step": st["step"], "encoder_lora": c.get("encoder_lora") or 0,
          "prior": c.get("prior", "encoder"), "run": c["run"], "loops": c["loops"],
          "encoder": c.get("encoder"), "option_vector": c.get("option_vector_resolved", "marker")}
tags = {"model": "lidm", "milestone": "stage1", "stage": "eval", "baseline_id": a.id}
n_params = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in enc.lora_params())
harness.run_eval(f"lidm-{a.name}" + ("-quick" if a.quick else ""), predict, params, tags, tasks,
                 track=not a.no_track, n_params=n_params)
