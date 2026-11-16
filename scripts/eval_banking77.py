"""Zero-shot Banking77 *test* accuracy of LIDM checkpoints at N=77 (all labels) and N=20 (gold + 19 random).
Final numbers only: never use this to pick checkpoints (stage 1 monitors a Banking77 *train* slice instead).

    uv run python scripts/eval_banking77.py checkpoints/stage1-B/best.pt [more.pt ...] [--n 1000]
"""
import argparse
import json

import torch

from cls.data.typed import banking77_items
from cls.model import LIDM, Encoder
from cls.train import validate

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--n", type=int, default=1000, help="test items (3080 = all)")
a = ap.parse_args()

sets = {"N77": banking77_items(None, n=a.n, seed=0), "N20": banking77_items(20, n=a.n, seed=0)}
enc = None
for path in a.ckpts:
    st = torch.load(path, map_location="cuda")
    c = st["config"]
    if enc is None or enc.lora_rank or c.get("encoder_lora"):  # a LoRA checkpoint changes the encoder itself
        enc = Encoder(lora_rank=c.get("encoder_lora") or 0)
    if c.get("encoder_lora"):
        enc.load_lora(st["encoder_lora"])
    m = LIDM(enc.d, enc.heads, enc.inter, enc.n_layers, c["n_blocks"], c["loops"], c["lora_rank"],
             dual_score=not c["no_dual_score"]).cuda()
    m.load_state_dict(st["model"])
    prior = None
    if c.get("prior") == "bge":
        from cls.eval.run_m0 import BiEncoder

        prior = BiEncoder("BAAI/bge-base-en-v1.5", "cls")
    loops = [L for L in (1, 2, 4, 8) if L <= c["loops"]]
    for key, items in sets.items():
        v = validate(m, enc, items, 16, loops, prior)
        accs = {f"L{L}": round(v.get(f"val_acc_L{L}", v["val_acc"]), 4) for L in loops}
        print(json.dumps({"ckpt": path, "set": key, "n": len(items), "acc": accs, "nll": round(v["val_loss"], 4)}))
