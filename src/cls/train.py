"""LIDM training, frozen encoder. Train + val only, no test eval.

    uv run python -m cls.train --run B          # 2 fusion blocks, 1 loop (M0 mixture, M1 grid)
    uv run python -m cls.train --run C          # 2 blocks looped 1-8 (sampled), per-loop FFN LoRA
    uv run python -m cls.train --run D          # 8 unshared blocks (iso-compute control for C)
    uv run python -m cls.train --run B --stage1 # stage 1 label reading (scripts/prepare_stage1.py)

Options are encoded per micro-batch (only that batch's distinct option strings), so label sets of any size fit.
Without --val, validation is a task-stratified split of --data; with it, a separate file (stage 1: UTCD
out-of-domain sets + a Banking77 *train* slice). Sets tagged `zs/` share no label with training;
`val_acc_zs` is their macro mean and what --stage1 selects best.pt on (`ood/` sets reuse training labels).

Live progress: tqdm bar in the terminal; MLflow experiment (default `m1`; scripts/mlflow_ui.sh) gets
train_loss, lr, epoch, progress every --log-every steps and val_* every --val-every steps (default: each
epoch). Checkpoints (trainable weights only): checkpoints/<name>/{best,last}.pt.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

# every micro-batch has a different shape (option count x option length); without this the allocator fragments
# and a stage-1 run died with "CUDA error: out of memory" at ~9 GB reserved on the 12 GB card
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import tqdm

from cls.model import ENCODER, LIDM, Encoder, decision_loss, option_texts
from cls.tracking import ROOT, Tracker

PRESETS = {"B": {"n_blocks": 2, "loops": 1}, "C": {"n_blocks": 2, "loops": 8}, "D": {"n_blocks": 8, "loops": 1}}
SCALARS = ("mix_a", "mix_b", "dual_scale")
# --stage1 = design doc stage 1 settings (docs/DESIGN.md, "Training plan"): one pass, 128 questions per step
STAGE1 = {"data": ROOT / "data" / "stage1_train.jsonl", "val": ROOT / "data" / "stage1_val.jsonl",
          "experiment": "stage1", "epochs": 1, "batch": 128, "micro_batch": 16, "val_every": 500, "select": "zs"}


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def split(items: list[dict], val_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified by task, so every task has val items."""
    by_task = collections.defaultdict(list)
    for it in items:
        by_task[it["task"]].append(it)
    rng = random.Random(seed)
    train, val = [], []
    for task in sorted(by_task):
        xs = by_task[task][:]
        rng.shuffle(xs)
        k = max(1, round(len(xs) * val_frac))
        val += xs[:k]
        train += xs[k:]
    return train, val


def length_sorted(items: list[dict]) -> list[dict]:
    """Order by pass-A length (characters as a cheap proxy for tokens). Applied inside each optimizer step, so a
    step is still a random sample but its micro-batches pad far less (stage 1 mixes ~370-token label_nli
    states with ~60-token NLI ones)."""
    return sorted(items, key=lambda it: len(it["state"]) + len(it["instructions"]))


def micro_batches(chunk: list[dict], size: int, max_tokens: int = 0, state_copies: bool = False):
    """Micro-batches of at most `size` questions; with max_tokens, also cut before the padded encoder tokens would pass
    it: (questions [+ option rows, with state_copies]) x longest state + option rows x longest option, with tokens ~
    characters / 3 (conservative: number-heavy states run below 4 characters per token). state_copies: LIDM-DJ holds a
    copy of the state per option, LIDM-X a copy of its keys/values per option (encode_x), so both count it."""
    cur, rows, s_max, o_max = [], 0, 0.0, 0.0
    for it in chunk:
        opts = option_texts(it)
        s = (len(it["state"]) + len(it["instructions"])) / 3
        o = max(map(len, opts)) / 3 + 2
        n_r, s_m, o_m = rows + len(opts), max(s_max, s), max(o_max, o)
        cost = (len(cur) + 1 + (n_r if state_copies else 0)) * s_m + n_r * o_m
        if cur and (len(cur) == size or (max_tokens and cost > max_tokens)):
            yield cur
            cur, n_r, s_m, o_m = [], len(opts), s, o
        cur.append(it)
        rows, s_max, o_max = n_r, s_m, o_m
    if cur:
        yield cur


def make_batch(items, index, device, rows=None):
    """rows: per item, its options' row indices (LIDM-X: one row per question and option); else via index."""
    opts = rows or [[index[t] for t in option_texts(it)] for it in items]
    n = max(map(len, opts))
    opt_idx = torch.zeros(len(items), n, dtype=torch.long)
    opt_mask = torch.zeros(len(items), n, dtype=torch.bool)
    for i, o in enumerate(opts):
        opt_idx[i, : len(o)] = torch.tensor(o)
        opt_mask[i, : len(o)] = True
    gold = torch.tensor([it["gold"] for it in items])
    is_score = torch.tensor([it["type"] == "score" for it in items])
    return [t.to(device) for t in (opt_idx, opt_mask, gold, is_score)]


def soft_target(items, n: int, device):
    """Q x N target distributions: an item's `soft` (normalised, in option order) where it has one, else one-hot."""
    t = torch.zeros(len(items), n)
    for i, it in enumerate(items):
        s = it.get("soft")
        if s and sum(s) > 0:
            t[i, : len(s)] = torch.tensor(s, dtype=torch.float) / sum(s)
        else:
            t[i, it["gold"]] = 1.0
    return t.to(device)


def make_prior(kind: str, enc):
    """External retrieval scorer for the dual_score term, or None for the encoder's in-model cosine. 'bge' =
    bge-base-en-v1.5 (M0 B5b); 'self' = this encoder's own retrieval embedding with LoRA off (Encoder.embed)."""
    if kind == "encoder":
        return None
    from cls.eval.run_m0 import BiEncoder

    if kind == "bge":
        return BiEncoder("BAAI/bge-base-en-v1.5", "cls")
    p = BiEncoder.__new__(BiEncoder)  # reuse its option cache, option_vecs and fit_tau around enc.embed
    p.torch, p.cache, p.prefix_q, p.tau, p.n_params, p.embed = torch, {}, "", None, 0, enc.embed
    return p


def load_model(path, device="cuda"):
    """Rebuild a checkpoint for evaluation: encoder (+ its LoRA), LIDM head and retrieval prior, in eval mode.
    Returns (enc, model, prior, checkpoint dict). Older checkpoints default to ModernBERT-large and the marker."""
    st = torch.load(path, map_location=device)
    c = st["config"]
    enc = Encoder(c.get("encoder", ENCODER), max_len=c.get("max_len", 512), max_opt_len=c.get("max_opt_len", 64),
                  lora_rank=c.get("encoder_lora") or 0, option_attn=c.get("option_attn", "none"),
                  joint_layers=c.get("joint_layers", 7))
    if c.get("encoder_lora"):
        enc.load_lora(st["encoder_lora"])
    enc.train(False)
    enc.marker = c.get("option_vector", "marker") == "marker"
    m = LIDM(enc.d, enc.heads, enc.inter, enc.n_layers, c["n_blocks"], c["loops"], c["lora_rank"],
             dual_score=not c["no_dual_score"], option_vector=c.get("option_vector_resolved", "marker")).to(device).eval()
    m.load_state_dict(st["model"])
    return enc, m, make_prior(c.get("prior", "encoder"), enc), st


def bge_prior(prior, items, n: int):
    """Q x N cosine between each state and its options from the external retrieval model; 0 for noul, whose
    "no" / "yes" options carry nothing to match (M0: bge alone scored 0.395 on BoolQ)."""
    states = prior.embed([it["state"] for it in items])
    out = torch.zeros(len(items), n)
    for i, it in enumerate(items):
        if it["type"] != "noul":
            out[i, : len(it["criteria"])] = torch.from_numpy(prior.option_vecs(it) @ states[i])
    return out


def encode(enc, items, prior=None):
    """Pass A for the items, pass B for their distinct options. Returns model inputs + gold + is_score + the
    external prior (None without one)."""
    states, questions = [it["state"] for it in items], [it["instructions"] for it in items]
    if enc.option_attn in ("state", "dj"):  # LIDM-X / -DJ: options depend on their question, no sharing
        opts = [option_texts(it) for it in items]
        ha, a_mask, opt_h, opt_tok_mask, _ = enc.encode_x(states, questions, opts)
        start = [0, *itertools.accumulate(map(len, opts))]
        rows = [list(range(start[i], start[i + 1])) for i in range(len(items))]
        opt_idx, opt_mask, gold, is_score = make_batch(items, None, enc.device, rows=rows)
    else:
        texts = sorted({t for it in items for t in option_texts(it)})
        index = {t: i for i, t in enumerate(texts)}
        ha, a_mask = enc.encode_a(states, questions)
        opt_h, opt_tok_mask = enc.encode_options(texts, enc.marker)
        opt_idx, opt_mask, gold, is_score = make_batch(items, index, enc.device)
    cos = bge_prior(prior, items, opt_idx.shape[1]).to(enc.device) if prior else None
    return (ha, a_mask, opt_h, opt_tok_mask, opt_idx, opt_mask), gold, is_score, cos


@torch.no_grad()
def validate(model, enc, val, bs, loops_list, prior=None, max_tokens=0):
    model.eval()
    enc.train(False)
    sums = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    for chunk in micro_batches(val, bs, max_tokens, enc.option_attn != "none"):
        inputs, gold, is_score, cos = encode(enc, chunk, prior)
        mask = inputs[-1]
        for loops in loops_list:  # encoder output reused across loop counts
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(*inputs, loops=loops, prior=cos)
            correct = (logits.argmax(-1) == gold).float()
            sums[f"L{loops}/loss"] += decision_loss(logits, gold, mask, is_score).item() * len(chunk)
            sums[f"L{loops}/acc"] += correct.sum().item()
            if loops == loops_list[-1]:
                for it, c in zip(chunk, correct.tolist()):
                    for key in (f"acc_{it['type']}", f"acc/{it['task']}"):
                        sums[key] += c
                        counts[key] += 1
                if cos is not None:  # the head alone (retrieval term zeroed): does it read labels by itself?
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        head = model(*inputs, loops=loops, prior=torch.zeros_like(cos))
                    for it, c in zip(chunk, (head.argmax(-1) == gold).float().tolist()):
                        sums[f"head_acc/{it['task']}"] += c
                        counts[f"head_acc/{it['task']}"] += 1
    model.train()
    enc.train(True)
    out = {"val_loss": sums[f"L{loops_list[-1]}/loss"] / len(val), "val_acc": sums[f"L{loops_list[-1]}/acc"] / len(val)}
    out.update({f"val_{k}": sums[k] / counts[k] for k in counts})
    for prefix, key in (("acc/zs/", "val_acc_zs"), ("head_acc/zs/", "val_head_acc_zs"),
                        ("acc/heldout/", "val_acc_heldout"), ("head_acc/heldout/", "val_head_acc_heldout")):
        vals = [sums[k] / counts[k] for k in counts if k.startswith(prefix)]
        if vals:  # macro means: zs/ = label-disjoint transfer sets, heldout/ = unseen labels from the training domain
            out[key] = sum(vals) / len(vals)
    if len(loops_list) > 1:
        out.update({f"val_acc_L{L}": sums[f"L{L}/acc"] / len(val) for L in loops_list})
    return out


def param_groups(model, wd, lr, scalar_lr):
    """Weight decay on head weight matrices only; none on norms, biases, mix scalars, dual scale.
    Mix weights and dual scale get their own (much larger) LR: at 1e-4 Adam moves them < 0.2 in a whole run."""
    named = list(model.named_parameters())
    scal = [p for n, p in named if n in SCALARS]
    decay = [p for n, p in named if n not in SCALARS and p.ndim >= 2]
    no_decay = [p for n, p in named if n not in SCALARS and p.ndim < 2]
    return [{"params": decay, "weight_decay": wd, "lr": lr}, {"params": no_decay, "weight_decay": 0.0, "lr": lr},
            {"params": scal, "weight_decay": 0.0, "lr": scalar_lr}]


def load_init(model, enc, path) -> dict:
    """Start from a checkpoint: head weights, plus the encoder LoRA if it has one. A non-looped checkpoint may
    initialise a looped head (B -> C): only the per-loop FFN LoRA is missing, and it starts at zero, so one
    loop reproduces the checkpoint. Anything else that does not match stops the run. Optimizer state is not
    restored (fresh AdamW and warm-up, as the design doc does between stages)."""
    st = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(st["model"], strict=False)
    bad = [k for k in missing if ".ffn.A" not in k and ".ffn.B" not in k]
    assert not unexpected and not bad, f"--init {path} does not fit this model: missing {bad[:3]}, unexpected {unexpected[:3]}"
    lora = st.get("encoder_lora")
    assert bool(lora) == bool(enc.lora_rank), f"--init {path} has encoder LoRA: {bool(lora)}; --encoder-lora set: {bool(enc.lora_rank)}"
    if lora:
        enc.load_lora(lora)
    return st["config"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", choices=PRESETS, default="B")
    ap.add_argument("--stage1", action="store_true", help="stage-1 defaults (data, val, 1 epoch, batch 128, ...)")
    ap.add_argument("--data", default=str(ROOT / "data" / "train_items.jsonl"))
    ap.add_argument("--val", default=None, help="separate val JSONL (default: split --data by --val-frac)")
    ap.add_argument("--experiment", default="m1", help="MLflow experiment")
    ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32, help="questions per optimizer step")
    ap.add_argument("--micro-batch", type=int, default=32, help="questions per forward (gradient accumulation)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scalar-lr", type=float, default=1e-2, help="LR for layer-mix weights and dual scale")
    ap.add_argument("--loops", type=int, default=None, help="override the preset's (max) loop count")
    ap.add_argument("--fixed-loops", action="store_true", help="always train at --loops instead of sampling 1..loops")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of steps")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--val-every", type=int, default=0, help="validate every N steps (0: once per epoch)")
    ap.add_argument("--select", choices=("loss", "acc", "zs"), default="loss",
                    help="best.pt by val loss, val acc, or val_acc_zs (mean over zs/ val sets)")
    ap.add_argument("--lora-rank", type=int, default=16, help="per-loop FFN LoRA (looped runs only)")
    ap.add_argument("--no-dual-score", action="store_true")
    ap.add_argument("--prior", choices=("encoder", "bge", "self"), default="encoder",
                    help="dual_score source: in-model encoder cosine; bge-base-en-v1.5 cosine (M0 B5b); or this "
                         "encoder's own retrieval embedding with LoRA off (for retrieval-trained encoders)")
    ap.add_argument("--prior-dropout", type=float, default=0.0,
                    help="share of micro-batches trained with the retrieval term off, so the head must solve them "
                         "alone instead of learning only a residual on top of retrieval")
    ap.add_argument("--option-vector", choices=("marker", "pooled"), default="marker",
                    help="score each option at a [MASK] marker, or with the encoder's own pooling (cls / mean)")
    ap.add_argument("--encoder-lora", type=int, default=0, help="LoRA rank on the encoder (0: frozen)")
    ap.add_argument("--option-attn", choices=("none", "state", "dj"), default="none",
                    help="'state' = LIDM-X: pass-B options attend to their question's pass A at every encoder layer; "
                         "'dj' = LIDM-DJ: the top --joint-layers run [pass A ; option] jointly, per option")
    ap.add_argument("--joint-layers", type=int, default=7, help="LIDM-DJ only: how many top encoder layers are joint")
    ap.add_argument("--soft-targets", action="store_true",
                    help="train on items' `soft` distributions where present (else one-hot); validation stays hard")
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="also split micro-batches above ~this many encoder tokens (0: off); for LIDM-DJ's state copies")
    ap.add_argument("--init", default=None, help="start from this checkpoint (head + encoder LoRA), e.g. B -> C")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-opt-len", type=int, default=64, help="token cap per option (pass B)")
    ap.add_argument("--limit", type=int, default=None, help="use only N train items (smoke test / LR slice)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default=None, help="run name (default m1-<run> / stage1-<run>)")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()
    if args.stage1:  # stage-1 defaults; flags given explicitly on the command line still win
        ap.set_defaults(**STAGE1)
        args = ap.parse_args()
    a = args

    from transformers import get_cosine_with_min_lr_schedule_with_warmup

    assert torch.cuda.is_available(), "LIDM training needs a CUDA GPU"
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    data_path = Path(a.data)
    if not data_path.exists():
        raise SystemExit(f"{data_path} missing: run scripts/prepare_data.py (M0) or scripts/prepare_stage1.py")
    items = read_jsonl(data_path)
    if a.val:
        train, val = items, read_jsonl(a.val)
    else:
        train, val = split(items, a.val_frac, a.seed)
    val = length_sorted(val)  # per-item metrics, so order only changes padding
    if a.limit:
        train = random.Random(a.seed).sample(train, min(a.limit, len(train)))
    preset = dict(PRESETS[a.run])
    if a.loops:
        preset["loops"] = a.loops
    prefix = "stage1" if a.stage1 else "m1"
    name = a.name or f"{prefix}-{a.run}" + ("-smoke" if a.limit else "")
    out_dir = ROOT / "checkpoints" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = Encoder(a.encoder, max_len=a.max_len, max_opt_len=a.max_opt_len, lora_rank=a.encoder_lora,
                  option_attn=a.option_attn, joint_layers=a.joint_layers)
    enc.train(True)
    model = LIDM(enc.d, enc.heads, enc.inter, enc.n_layers, preset["n_blocks"], preset["loops"], a.lora_rank,
                 dual_score=not a.no_dual_score,
                 option_vector="marker" if a.option_vector == "marker" else enc.pooling).cuda()
    enc.marker = a.option_vector == "marker"
    model.head.init_from_encoder(enc.model.layers)
    if a.init:
        load_init(model, enc, a.init)
    prior, tau = None, None
    if a.prior in ("bge", "self"):
        assert not a.no_dual_score, f"--prior {a.prior} needs the dual_score term"
        prior = make_prior(a.prior, enc)
        # prior scale = 1 / the retrieval model's fitted temperature (as in M0, 300 training choice items). From scratch the
        # scorer starts at zero, so training starts exactly at bge; with --init the loaded scorer is kept
        tau = prior.fit_tau([it for it in train if it["type"] == "choice"][:300])
        with torch.no_grad():
            model.dual_scale.fill_(1 / tau)
            if not a.init:
                model.scorer.weight.zero_()
                model.scorer.bias.zero_()
    n_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in enc.lora_params())

    steps_per_epoch = math.ceil(len(train) / a.batch)
    total = steps_per_epoch * a.epochs
    groups = param_groups(model, a.wd, a.lr, a.scalar_lr)
    if a.encoder_lora:  # design doc 2b: LoRA at the head's LR, no weight decay
        groups.append({"params": enc.lora_params(), "weight_decay": 0.0, "lr": a.lr})
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.98))
    sched = get_cosine_with_min_lr_schedule_with_warmup(opt, max(1, int(a.warmup * total)), total, min_lr_rate=0.1)
    max_loops = preset["loops"]
    val_loops = [L for L in (1, 2, 4, 8) if L <= max_loops] if max_loops > 1 else [1]
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, **preset,
           "n_train": len(train), "n_val": len(val),
           "n_options_unique": len({t for it in train for t in option_texts(it)}),
           "trainable_params": n_params, "steps": total, "out_dir": str(out_dir), "bge_tau": tau,
           "option_vector_resolved": model.option_vector}

    tags = {"milestone": "stage1" if a.stage1 else "M1", "stage": "train", "run": a.run, "model": "lidm"}
    with Tracker(a.experiment, name, tags=tags, enabled=not a.no_track) as tr:
        tr.log_params(cfg)
        (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"[{name}] {n_params / 1e6:.1f}M trainable, {len(train)} train / {len(val)} val, {total} steps")
        rng = random.Random(a.seed)
        best = -float("inf")  # best score; score = -val_loss, val_acc or val_acc_zs
        step, run_loss, t0 = 0, 0.0, time.time()
        pbar = tqdm(total=total, desc=name, dynamic_ncols=True)

        def evaluate(epoch: float):
            nonlocal best
            v = validate(model, enc, val, a.micro_batch, val_loops, prior, a.max_tokens)
            v["elapsed_min"] = (time.time() - t0) / 60
            tr.log_metrics(v, step=step)
            state = {"model": model.state_dict(), "config": cfg, "epoch": epoch, "step": step, "val": v}
            if a.encoder_lora:
                state["encoder_lora"] = enc.lora_state()
            score = {"acc": v["val_acc"], "zs": v.get("val_acc_zs", v["val_acc"]), "loss": -v["val_loss"]}[a.select]
            if score > best:
                best = score
                torch.save(state, out_dir / "best.pt")
            torch.save(state, out_dir / "last.pt")
            pbar.write(f"[{name}] step {step} (epoch {epoch:.2f}): "
                       + " ".join(f"{k}={x:.4f}" for k, x in v.items() if "/" not in k))

        for epoch in range(a.epochs):
            rng.shuffle(train)
            for s in range(0, len(train), a.batch):
                chunk = length_sorted(train[s:s + a.batch])
                # sampled depth per step (1 for unlooped runs), or always max_loops with --fixed-loops
                loops = max_loops if a.fixed_loops else rng.randint(1, max_loops)
                step_loss = 0.0
                for mb in micro_batches(chunk, a.micro_batch, a.max_tokens, a.option_attn != "none"):
                    drop = prior is not None and rng.random() < a.prior_dropout
                    inputs, gold, is_score, cos = encode(enc, mb, None if drop else prior)
                    if drop:  # zeros, not None: None would fall back to the in-model encoder cosine
                        cos = torch.zeros(inputs[4].shape, device=inputs[4].device)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(*inputs, loops=loops, prior=cos)
                    target = soft_target(mb, logits.shape[1], logits.device) if a.soft_targets else None
                    loss = decision_loss(logits, gold, inputs[-1], is_score, target=target) * len(mb) / len(chunk)
                    loss.backward()
                    step_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                run_loss += step_loss
                pbar.update(1)
                pbar.set_postfix(epoch=f"{epoch + 1}/{a.epochs}", loss=f"{step_loss:.3f}",
                                 lr=f"{sched.get_last_lr()[0]:.1e}")
                if step % a.log_every == 0 or step == total:
                    tr.log_metrics({"train_loss": run_loss / (step % a.log_every or a.log_every), "lr": sched.get_last_lr()[0],
                                    "epoch": step / steps_per_epoch, "progress": step / total}, step=step)
                    run_loss = 0.0
                if a.val_every and step % a.val_every == 0 and step != total:
                    evaluate(step / steps_per_epoch)
            if not a.val_every or epoch == a.epochs - 1:
                evaluate(epoch + 1)
        pbar.close()
        best_val = -best if a.select == "loss" else best
        tr.log_metrics({f"best_val_{a.select}": best_val, "train_wall_min": (time.time() - t0) / 60,
                        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9})
        print(f"[{name}] done: best val_{a.select} {best_val:.4f}, checkpoints in {out_dir}; "
              f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
