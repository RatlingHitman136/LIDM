#!/usr/bin/env bash
# Label suite for stage 2 (rerun after the bucket fix), started once scripts/run_stage2_eval.sh has finished.
cd "$(dirname "$0")/.." || exit 1
L=logs/stage2
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for _ in $(seq 1 240); do grep -q "stage-2 evaluation finished" "$L/summary.log" && break; sleep 30; done
CS="uv run python scripts/eval_context_suite.py"
ITEMS="data/stage2/eval_heldout_sources.jsonl data/stage2/eval_banking77_names.jsonl data/stage2/eval_banking77_desc.jsonl data/stage2/eval_clinc150.jsonl"
run() { local n=$1; shift; echo "$(date '+%F %T') start $n" >> "$L/summary.log"; timeout 45m "$@" > "$L/$n.log" 2>&1
        echo "$(date '+%F %T') end   $n (exit $?)" >> "$L/summary.log"; }
run s2-lidm-labels $CS lidm --ckpt checkpoints/s2-lidm/last.pt --name s2-lidm --items $ITEMS
run s2-laya-labels $CS laya --ckpt checkpoints/s2-laya --name s2-laya --items $ITEMS
run ref-lidm1-labels $CS lidm --ckpt checkpoints/stage1-Blora/last.pt --name ref-lidm-stage1 --items $ITEMS
run ref-laya-labels $CS laya --name ref-laya-shipped --items $ITEMS
run ref-bge-labels $CS biencoder --encoder BAAI/bge-base-en-v1.5 --name ref-bge --items $ITEMS
run ref-gte-labels $CS biencoder --encoder Alibaba-NLP/gte-modernbert-base --name ref-gte --items $ITEMS
echo "$(date '+%F %T') label suite finished" >> "$L/summary.log"
