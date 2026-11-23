#!/usr/bin/env bash
# Stage 2, same-data comparison: LIDM (stage-1 MBL LoRA init) and Laya (shipped) each take one pass over the same
# cases (data/stage2/), final checkpoints. Logs in logs/stage2/, summary in logs/stage2/summary.log.
cd "$(dirname "$0")/.." || exit 1
L=logs/stage2
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() {  # step <name> <time limit> <command...>
    local name=$1 limit=$2
    shift 2
    echo "$(date '+%F %T') start $name" >> "$L/summary.log"
    timeout "$limit" "$@" > "$L/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $name (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT'))" >> "$L/summary.log"
}
step s2-lidm 4h uv run python -m cls.train --run B --experiment stage2 --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init checkpoints/stage1-Blora/last.pt --data data/stage2/train.jsonl --val data/stage2/dev.jsonl \
    --epochs 1 --batch 32 --micro-batch 8 --lr 1e-4 --max-len 1024 --max-opt-len 128 --val-every 1000 --name s2-lidm
step s2-laya 4h uv run python -m cls.eval.laya_finetune --data data/stage2/train_laya.jsonl --out checkpoints/s2-laya \
    --epochs 1 --max-len 1024 --head-max-len 512 --no-track
