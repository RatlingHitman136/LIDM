"""B4: fine-tune Laya on the shared mixture (Banking77 held out) with Laya's own trainer + defaults."""
from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import time

from cls.tracking import ROOT, Tracker


class _Tee:
    def __init__(self, stream, tr):
        self.stream, self.tr, self.buf = stream, tr, ""

    def write(self, s):
        self.stream.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            m = re.search(r"epoch (\d+)/\d+ step (\d+) loss ([\d.]+)", line)
            if m:
                self.tr.log_metrics({"train_loss": float(m.group(3))}, step=(int(m.group(1)) - 1) * 100000 + int(m.group(2)))
            m = re.search(r"epoch (\d+)/\d+ mean loss ([\d.]+)", line)
            if m:
                self.tr.log_metrics({"epoch_mean_loss": float(m.group(2))}, step=int(m.group(1)))

    def flush(self):
        self.stream.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "train_laya.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "checkpoints" / "laya-b4"))
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--head-max-len", type=int, default=None)
    ap.add_argument("--loss", default="rlcd")
    ap.add_argument("--name", default="laya-b4")
    ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()

    import torch
    from laya.train import TrainConfig, finetune, resolve_checkpoint_dir

    cfg = TrainConfig(epochs=a.epochs, max_len=a.max_len, head_max_len=a.head_max_len, loss=a.loss)
    tags = {"baseline_id": "B4", "model": "laya-finetuned", "milestone": "M0", "stage": "train",
            "question": "Laya fine-tuned with its own trainer on the shared mixture (Banking77 held out)"}
    with Tracker("baselines", a.name + "-train", tags=tags, enabled=not a.no_track) as tr:
        tr.log_params({"base": a.base, "data": a.data, **{f"cfg_{k}": v for k, v in dataclasses.asdict(cfg).items()}})
        old = sys.stdout
        sys.stdout = _Tee(old, tr)
        t0 = time.time()
        try:
            model_dir = resolve_checkpoint_dir(a.base)
            summary = finetune(a.data, model_dir, a.out, cfg)
        finally:
            sys.stdout = old
        tr.log_metrics({"train_wall_min": (time.time() - t0) / 60, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
        tr.log_dict({k: v for k, v in summary.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}, "finetune_summary.json")
        print("saved", a.out)


if __name__ == "__main__":
    main()
