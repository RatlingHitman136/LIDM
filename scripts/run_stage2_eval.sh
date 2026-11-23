#!/usr/bin/env bash
# Stage-2 evaluation, started after scripts/run_stage2_train.sh: typed-decisions test (schemas never trained on),
# context suite, label suite (held-out label sets, Banking77 N=77 names / descriptions, CLINC150 N=150) with
# references, then stage 2c (each stage-2 model fine-tuned on typed-decisions train exactly like the direct runs).
cd "$(dirname "$0")/.." || exit 1
L=logs/stage2
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() {
    local name=$1 limit=$2
    shift 2
    echo "$(date '+%F %T') start $name" >> "$L/summary.log"
    timeout "$limit" "$@" > "$L/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $name (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT')) $(grep -E '^\{' "$L/$name.log" | tail -1 | cut -c1-300)" >> "$L/summary.log"
}
for _ in $(seq 1 480); do grep -q "end   s2-laya " "$L/summary.log" && break; sleep 60; done  # at most 8 h
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
ITEMS="data/stage2/eval_heldout_sources.jsonl data/stage2/eval_banking77_names.jsonl data/stage2/eval_banking77_desc.jsonl data/stage2/eval_clinc150.jsonl"

step s2-lidm-td 30m $EV lidm --ckpt checkpoints/s2-lidm/last.pt --name s2-lidm
step s2-laya-td 30m $EV laya --ckpt checkpoints/s2-laya --name s2-laya
step s2-cmp-td 5m uv run python scripts/compare_typed_decisions.py s2-lidm s2-laya
step s2-lidm-context 45m $CS lidm --ckpt checkpoints/s2-lidm/last.pt --name s2-lidm
step s2-laya-context 45m $CS laya --ckpt checkpoints/s2-laya --name s2-laya
step s2-lidm-labels 45m $CS lidm --ckpt checkpoints/s2-lidm/last.pt --name s2-lidm --items $ITEMS
step s2-laya-labels 45m $CS laya --ckpt checkpoints/s2-laya --name s2-laya --items $ITEMS
step ref-lidm1-labels 45m $CS lidm --ckpt checkpoints/stage1-Blora/last.pt --name ref-lidm-stage1 --items $ITEMS
step ref-laya-labels 45m $CS laya --name ref-laya-shipped --items $ITEMS
step ref-bge-labels 20m $CS biencoder --encoder BAAI/bge-base-en-v1.5 --name ref-bge --items $ITEMS
step ref-gte-labels 20m $CS biencoder --encoder Alibaba-NLP/gte-modernbert-base --name ref-gte --items $ITEMS

# stage 2c: same recipes as the direct typed-decisions runs (td-lidm: lr 1e-4, 6 passes, best on dev; td-laya:
# Laya defaults, 4 passes, final), starting from the stage-2 checkpoints
step s2c-lidm-train 2h uv run python -m cls.train --run B --experiment typed_decisions --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init checkpoints/s2-lidm/last.pt --data data/td/train.jsonl --val data/td/dev.jsonl \
    --epochs 6 --batch 32 --micro-batch 16 --lr 1e-4 --max-len 1024 --select acc --name s2c-lidm
step s2c-lidm-td 30m $EV lidm --ckpt checkpoints/s2c-lidm/best.pt --name s2c-lidm
step s2c-laya-train 1h uv run python -m cls.eval.laya_finetune --base "$PWD/checkpoints/s2-laya" --data data/td/train_laya.jsonl \
    --out checkpoints/s2c-laya --epochs 4 --max-len 1024 --encoder-lr 2.5e-5 --head-lr 1e-4 --no-track
step s2c-laya-td 30m $EV laya --ckpt checkpoints/s2c-laya --name s2c-laya
step s2c-cmp 5m uv run python scripts/compare_typed_decisions.py s2c-lidm s2c-laya
step s2c-cmp-lidm 5m uv run python scripts/compare_typed_decisions.py s2c-lidm td-lidm
step s2c-cmp-laya 5m uv run python scripts/compare_typed_decisions.py s2c-laya td-laya
echo "$(date '+%F %T') stage-2 evaluation finished" >> "$L/summary.log"
