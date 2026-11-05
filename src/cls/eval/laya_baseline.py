"""M0: evaluate pretrained Laya (and its published workarounds) on Banking77, logging to MLflow.

Variants
  b0  laya as shipped                       (max_len=512,  head_max_len=192)
  b3  laya with larger budgets              (max_len=1024, head_max_len=512)
  b1  laya + embedding shortlist, k=20      (laya.predict_shortlist, embedder = laya's own encoder)
  b2  laya + tournament, groups of 16       (laya.predict_tournament)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cls.data.banking77 import load_banking77, make_question
from cls.metrics import bootstrap_ci, summarize
from cls.tracking import ROOT, Tracker

VARIANTS = {
    "b0": dict(desc="Laya as shipped", overrides={}),
    "b3": dict(desc="Laya max_len=1024 head_max_len=512", overrides={"max_len": 1024, "head_max_len": 512}),
    "b1": dict(desc="Laya + shortlist k=20", overrides={}),
    "b2": dict(desc="Laya + tournament group=16", overrides={}),
}


def probs_row(result: dict, label_ids: list[str]) -> np.ndarray:
    p = result["answers"]["intent"]["probabilities"]
    return np.array([p.get(l, 0.0) for l in label_ids], dtype=np.float64)


def run(args) -> dict:
    import laya
    import torch

    data = load_banking77("test", n=args.n_eval, seed=args.seed)
    label_ids = data.label_ids
    descs = data.descriptions if args.label_style == "desc" else None
    questions = make_question(label_ids, descs)
    y = np.array(data.labels)
    var = VARIANTS[args.variant]
    over = var["overrides"]

    run_name = f"laya-{args.variant}-banking77-{args.label_style}" + (f"-n{args.n_eval}" if args.n_eval else "")
    tags = {"baseline_id": args.variant.upper(), "model": "laya", "milestone": "M0", "stage": "baseline",
            "question": f"Reproduce Laya on Banking77 ({var['desc']}, labels as {args.label_style})"}
    with Tracker("baselines", run_name, tags=tags, enabled=not args.no_track) as tr:
        agent = laya.load(args.checkpoint)
        n_params = sum(p.numel() for p in agent.model.parameters())
        tr.log_params({
            "checkpoint": args.checkpoint, "variant": args.variant, "variant_desc": var["desc"],
            "dataset": "PolyAI/banking77", "split": "test", "n_eval": len(y), "n_options": len(label_ids),
            "label_style": args.label_style, "batch_size": args.batch_size,
            "cfg_max_len": agent.cfg.get("max_len"), "cfg_head_max_len": agent.cfg.get("head_max_len"),
            "override_max_len": over.get("max_len"), "override_head_max_len": over.get("head_max_len"),
            "shortlist_k": 20 if args.variant == "b1" else None,
            "tournament_group": 16 if args.variant == "b2" else None,
            "n_params": n_params, "seed": args.seed, "instruction": questions["intent"]["instructions"],
            "agent_temperatures": agent.cfg.get("temperature_by_options"),
        })
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- accuracy pass
        t0 = time.perf_counter()
        rows, trunc = [], 0
        if args.variant in ("b0", "b3"):
            for s in range(0, len(data.texts), args.batch_size):
                res = agent.predict_batch(data.texts[s:s + args.batch_size], questions, **over)
                rows += [probs_row(r, label_ids) for r in res]
        else:
            embed = laya.embed_fn_from_agent(agent)
            cache: dict[str, np.ndarray] = {}

            def cached(texts):  # label strings are identical for every example; embed them once
                out, todo = [None] * len(texts), []
                for i, t in enumerate(texts):
                    if i > 0 and t in cache:
                        out[i] = cache[t]
                    else:
                        todo.append(i)
                if todo:
                    vecs = np.asarray(embed([texts[i] for i in todo]))
                    for i, v in zip(todo, vecs):
                        out[i] = v
                        if i > 0:
                            cache[texts[i]] = v
                return np.stack(out)

            for text in data.texts:
                if args.variant == "b1":
                    r = laya.predict_shortlist(agent, text, questions, cached, k=20)
                else:
                    r = laya.predict_tournament(agent, text, questions, group_size=16)
                rows.append(probs_row(r, label_ids))
        wall = time.perf_counter() - t0
        probs = np.stack(rows)

        m = summarize(probs, y)
        lo, hi = bootstrap_ci((probs.argmax(1) == y).astype(float))
        m.update({"accuracy_ci_lo": lo, "accuracy_ci_hi": hi, "eval_wall_s": wall,
                  "eval_examples_per_s": len(y) / wall})

        # ---- batch-1 latency (the interactive path)
        lat = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        texts = data.texts[: args.lat_warmup + args.lat_iters]
        for i, text in enumerate(texts):
            t = time.perf_counter()
            if args.variant == "b1":
                laya.predict_shortlist(agent, text, questions, cached, k=20)
            elif args.variant == "b2":
                laya.predict_tournament(agent, text, questions, group_size=16)
            else:
                agent.system_one(text, questions, **over)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= args.lat_warmup:
                lat.append((time.perf_counter() - t) * 1000)
        lat = np.array(lat)
        m.update({"latency_ms_p50": float(np.percentile(lat, 50)), "latency_ms_p95": float(np.percentile(lat, 95)),
                  "latency_ms_mean": float(lat.mean())})
        if torch.cuda.is_available():
            m["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        tr.log_metrics(m)

        # ---- artifacts
        out = ROOT / "artifacts" / run_name
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "predictions.npz", probs=probs.astype(np.float32), y=y)
        (out / "metrics.json").write_text(json.dumps(m, indent=2))
        (out / "latency_ms.json").write_text(json.dumps(lat.tolist()))
        tr.log_artifact(out / "predictions.npz")
        tr.log_artifact(out / "metrics.json")
        tr.log_artifact(out / "latency_ms.json")
        tr.log_dict(agent.cfg, "laya_cfg.json")
        _plot(probs, y, tr, run_name)
        print(f"[{run_name}] " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
        return m


def _plot(probs, y, tr, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = probs / np.clip(probs.sum(1, keepdims=True), 1e-12, None)
    conf, correct = p.max(1), (p.argmax(1) == y)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 9)
    xs, ys = [], []
    for b in range(10):
        if (idx == b).any():
            xs.append(conf[idx == b].mean())
            ys.append(correct[idx == b].mean())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.plot(xs, ys, "o-")
    ax.set(xlabel="confidence", ylabel="accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
    tr.log_figure(fig, "reliability.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="b0")
    ap.add_argument("--checkpoint", default="convaiinnovations/laya")
    ap.add_argument("--label-style", choices=["id", "desc"], default="desc",
                    help="id: label id only (card_arrival); desc: 'card_arrival: card arrival'")
    ap.add_argument("--n-eval", type=int, default=None, help="subsample the 3080 test examples")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lat-warmup", type=int, default=10)
    ap.add_argument("--lat-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-track", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
