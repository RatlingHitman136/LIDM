#!/usr/bin/env bash
# Looped LIDM-X, stage 1 only. Head run C (2 blocks looped, loop count sampled 1-4 per step, per-loop FFN LoRA r=16,
# zero-initialised) with options reading pass A at every encoder layer (--option-attn state). One full pass over the
# label-variety mix (stage 1b: 522,691 questions, ~700k distinct option strings; Banking77 and zs/ labels banned),
# validated every 1,000 steps at 1, 2 and 4 loops on stage1b_val (zs/ 961, ood/ 1,500, heldout/ 1,936).
# No time limit: check logs/full_s1b_loop/summary.log and the training log.
#
#   nohup scripts/run_full_lidmx_loop.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
L=logs/full_s1b_loop
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') start s1bxl-train" >> "$L/summary.log"
uv run python -m cls.train --run C --loops 4 --stage1 --data data/stage1b_train.jsonl --val data/stage1b_val.jsonl \
    --encoder answerdotai/ModernBERT-large --encoder-lora 16 --option-attn state --lr 1e-4 --batch 128 --micro-batch 16 \
    --max-len 512 --max-opt-len 64 --val-every 1000 --name s1bxl-lidm > "$L/s1bxl-train.log" 2>&1
rc=$?  # before any other command resets $?
echo "$(date '+%F %T') end   s1bxl-train (exit $rc) $(tr '\r' '\n' < "$L/s1bxl-train.log" | grep -E 'done:|Error' | tail -1)" >> "$L/summary.log"
