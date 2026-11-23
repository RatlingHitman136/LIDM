#!/usr/bin/env bash
# LIDM-X stage 1 from scratch, capped at 2,000 steps x 128 (runs while LIDM-DJ is being implemented) (the previous stage 1 plateaued from ~3,500-5,000 steps).
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "$(date '+%F %T') start s1x-lidm" >> logs/full/summary.log
timeout 3h uv run python -m cls.train --run B --stage1 --encoder-lora 16 --option-attn state --limit 256000 \
    --name s1x-lidm > logs/full/s1x-lidm.log 2>&1
rc=$?
echo "$(date '+%F %T') end   s1x-lidm (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT'))" >> logs/full/summary.log
