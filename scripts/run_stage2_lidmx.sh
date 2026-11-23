#!/usr/bin/env bash
# LIDM-X on stage 2 (reports/Recovering interaction after late splits.md): same start (stage-1 MBL LoRA), data and
# recipe as s2-lidm, with pass-B options attending to pass A at every layer (--option-attn state). Then the same
# tests as stage 2: typed decisions (unseen schemas), context suite, label suite; 2c (fine-tune on typed-decisions
# train, as s2c-lidm); paired comparison with s2-lidm / s2-laya. No time limits: check the logs.
#
#   nohup scripts/run_stage2_lidmx.sh > /dev/null 2>&1 &
#   tail -f logs/stage2x/summary.log
cd "$(dirname "$0")/.." || exit 1
L=logs/stage2x
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <name> <command...>
    local n=$1
    shift
    echo "$(date '+%F %T') start $n" >> "$L/summary.log"
    "$@" > "$L/$n.log" 2>&1
    echo "$(date '+%F %T') end   $n (exit $?) $(tr '\r' '\n' < "$L/$n.log" | grep -E '^\{|done:' | tail -1 | cut -c1-300)" >> "$L/summary.log"
}
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
ITEMS="data/stage2/eval_heldout_sources.jsonl data/stage2/eval_banking77_names.jsonl data/stage2/eval_banking77_desc.jsonl data/stage2/eval_clinc150.jsonl"
X=checkpoints/s2x-lidm/last.pt

run s2x-train uv run python -m cls.train --run B --experiment stage2 --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init checkpoints/stage1-Blora/last.pt --data data/stage2/train.jsonl --val data/stage2/dev.jsonl \
    --epochs 1 --batch 32 --micro-batch 8 --lr 1e-4 --max-len 1024 --max-opt-len 128 --val-every 1000 \
    --option-attn state --name s2x-lidm
run s2x-td $EV lidm --ckpt $X --name s2x-lidm
run s2x-cmp-lidm uv run python scripts/compare_typed_decisions.py s2x-lidm s2-lidm
run s2x-cmp-laya uv run python scripts/compare_typed_decisions.py s2x-lidm s2-laya
run s2x-context $CS lidm --ckpt $X --name s2x-lidm
run s2x-labels $CS lidm --ckpt $X --name s2x-lidm --items $ITEMS
# re-score the two references so the context suite also has per-question results (paired CIs); same checkpoints
run s2-lidm-context $CS lidm --ckpt checkpoints/s2-lidm/last.pt --name s2-lidm
run s2-laya-context $CS laya --ckpt checkpoints/s2-laya --name s2-laya
# 2c, exactly as s2c-lidm
run s2xc-train uv run python -m cls.train --run B --experiment typed_decisions --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init $X --data data/td/train.jsonl --val data/td/dev.jsonl --epochs 6 --batch 32 --micro-batch 16 \
    --lr 1e-4 --max-len 1024 --select acc --option-attn state --name s2xc-lidm
run s2xc-td $EV lidm --ckpt checkpoints/s2xc-lidm/best.pt --name s2xc-lidm
run s2xc-cmp-laya uv run python scripts/compare_typed_decisions.py s2xc-lidm s2c-laya
run s2xc-cmp-lidm uv run python scripts/compare_typed_decisions.py s2xc-lidm s2c-lidm
uv run python scripts/summarize_lidmx.py --x2c s2xc-lidm >> "$L/summary.log" 2>&1
echo "$(date '+%F %T') LIDM-X chain finished" >> "$L/summary.log"
