"""Model-agnostic evaluation over typed-decision items, logging to MLflow.

A predictor is `fn(item) -> np.ndarray` of option probabilities (length = #options of the item;
noul -> [P(false), P(true)]; score -> level distribution).
"""
from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np

from cls.data import typed
from cls.metrics import bootstrap_ci, summarize
from cls.tracking import ROOT, Tracker

Predictor = Callable[[dict], np.ndarray]


def n_options(item: dict) -> int:
    return 2 if item["type"] == "noul" else len(item["criteria"])


def laya_question(item: dict) -> dict:
    q = {"type": item["type"], "instructions": item["instructions"]}
    if item["type"] != "noul":
        q["criteria"] = item["criteria"]
    return {"q": q}


def laya_probs(result: dict, item: dict) -> np.ndarray:
    a = result["answers"]["q"]
    if item["type"] == "noul":
        p = float(a["noul"])
        return np.array([1 - p, p])
    probs = a["probabilities"]
    if item["type"] == "score":
        return np.array([probs.get(str(i), 0.0) for i in range(len(item["criteria"]))])
    return np.array([probs.get(k, 0.0) for k in item["criteria"]])


def laya_predictor(agent, **over) -> Predictor:
    return lambda item: laya_probs(agent.system_one(item["state"], laya_question(item), **over), item)


def laya_shortlist_predictor(agent, embed_fn, k=20) -> Predictor:
    import laya

    def fn(item):
        if item["type"] != "choice":
            return laya_probs(agent.system_one(item["state"], laya_question(item)), item)
        r = laya.predict_shortlist(agent, item["state"], laya_question(item), embed_fn, k=k)
        return laya_probs(r, item)

    return fn


def laya_tournament_predictor(agent, group=16) -> Predictor:
    import laya

    def fn(item):
        if item["type"] != "choice":
            return laya_probs(agent.system_one(item["state"], laya_question(item)), item)
        r = laya.predict_tournament(agent, item["state"], laya_question(item), group_size=group)
        return laya_probs(r, item)

    return fn


# ---------------------------------------------------------------- tasks
def default_tasks(quick: bool = False) -> dict[str, list[dict]]:
    n = 200 if quick else None
    tasks = {"banking77_N77": typed.banking77_items(n=n, seed=0)}
    for k in (5, 10, 20, 40):
        tasks[f"banking77_N{k}"] = typed.banking77_items(n_opts=k, n=200 if quick else 1000, seed=1)
    tasks["boolq_val"] = typed.boolq_items(n=n)
    tasks["sst5_test"] = typed.sst5_items(n=n)
    return tasks


def evaluate_task(pred: Predictor, items: list[dict]) -> tuple[dict, np.ndarray, np.ndarray]:
    t0 = time.perf_counter()
    probs = [pred(it) for it in items]
    wall = time.perf_counter() - t0
    k = max(len(p) for p in probs)
    P = np.zeros((len(probs), k))
    for i, p in enumerate(probs):
        P[i, : len(p)] = p
    y = np.array([it["gold"] for it in items])
    m = summarize(P, y)
    lo, hi = bootstrap_ci((P.argmax(1) == y).astype(float))
    m.update({"accuracy_ci_lo": lo, "accuracy_ci_hi": hi, "wall_s": wall})
    if items[0]["type"] == "score":
        ev = (P / P.sum(1, keepdims=True)) @ np.arange(k)
        m["mae_expected_level"] = float(np.abs(ev - y).mean())
        m["mae_argmax_level"] = float(np.abs(P.argmax(1) - y).mean())
    return m, P, y


def latency(pred: Predictor, items: list[dict], warmup=10, iters=100) -> dict:
    import torch

    lat = []
    for i, it in enumerate(items[: warmup + iters]):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        pred(it)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= warmup:
            lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "mean": float(lat.mean())}


def run_eval(run_name: str, pred: Predictor, params: dict, tags: dict, tasks: dict[str, list[dict]],
             experiment="baselines", track=True, n_params: int | None = None) -> dict:
    import torch

    flat: dict[str, float] = {}
    with Tracker(experiment, run_name, tags=tags, enabled=track) as tr:
        tr.log_params({**params, "tasks": list(tasks), "n_params": n_params})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        out = ROOT / "artifacts" / run_name
        out.mkdir(parents=True, exist_ok=True)
        arrs = {}
        for name, items in tasks.items():
            m, P, y = evaluate_task(pred, items)
            for k, v in m.items():
                flat[f"{name}/{k}"] = v
            arrs[f"{name}__probs"], arrs[f"{name}__y"] = P.astype(np.float32), y
            print(f"[{run_name}] {name}: acc={m['accuracy']:.4f} ece={m['ece']:.3f} nll={m['nll']:.3f}", flush=True)
        # accuracy-vs-N curve: logged with step=N so MLflow draws it
        for name in tasks:
            if name.startswith("banking77_N"):
                tr.log_metrics({"acc_vs_N": flat[f"{name}/accuracy"]}, step=int(name.split("N")[-1]))
        lat = latency(pred, tasks["banking77_N77"])
        for k, v in lat.items():
            flat[f"latency_ms_{k}"] = v
        if torch.cuda.is_available():
            flat["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        tr.log_metrics(flat)
        np.savez_compressed(out / "predictions.npz", **arrs)
        (out / "metrics.json").write_text(json.dumps(flat, indent=2))
        tr.log_artifact(out / "predictions.npz")
        tr.log_artifact(out / "metrics.json")
        return flat
