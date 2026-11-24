#!/usr/bin/env bash
# Soft 2c directly on the adopted LIDM-X (s2x-lidm): fine-tune on typed-decisions train with its soft gold, exactly the
# s2xsc-lidm 2c recipe but without the soft stage-2 pass; then test, dev, comparisons, calibration / selective prediction.
cd "$(dirname "$0")/.." || exit 1
L=logs/stage3
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local n=$1; shift; echo "$(date '+%F %T') start $n" >> "$L/summary.log"; "$@" > "$L/$n.log" 2>&1; local rc=$?
        echo "$(date '+%F %T') end   $n (exit $rc) $(tr '\r' '\n' < "$L/$n.log" | grep -E '^\{|done:' | tail -1 | cut -c1-300)" >> "$L/summary.log"; }
EV="uv run python scripts/eval_typed_decisions.py"
run s2xcs-train uv run python -m cls.train --run B --experiment typed_decisions --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init checkpoints/s2x-lidm/last.pt --data data/td/train.jsonl --val data/td/dev.jsonl --epochs 6 \
    --batch 32 --micro-batch 16 --lr 1e-4 --max-len 1024 --select acc --option-attn state --soft-targets --name s2xcs-lidm
run s2xcs-td $EV lidm --ckpt checkpoints/s2xcs-lidm/best.pt --name s2xcs-lidm
run s2xcs-dev $EV lidm --ckpt checkpoints/s2xcs-lidm/best.pt --test data/td/dev.jsonl --name s2xcs-lidm-dev --no-track
for b in s2xsc-lidm s2xc-lidm s2c-laya; do run s2xcs-cmp-$b uv run python scripts/compare_typed_decisions.py s2xcs-lidm $b; done
run s2xcs-selective uv run python scripts/selective_typed_decisions.py s2xcs-lidm s2xsc-lidm s2xc-lidm s2c-laya --calibrate
echo "$(date '+%F %T') soft 2c finished" >> "$L/summary.log"
