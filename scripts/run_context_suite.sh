#!/usr/bin/env bash
# Zero-shot context suite on existing checkpoints, no training: two LIDM checkpoints (plus the options-in-question
# ablation on regime-2 tasks), bge and gte alone, shipped Laya and laya-typed-decisions. Every step has a hard time
# limit and its own log in logs/context_suite/; summary.log gets start/end times, exit codes and the final table.
#
#   nohup scripts/run_context_suite.sh > /dev/null 2>&1 &
#   WITH_B4=1 nohup scripts/run_context_suite.sh > /dev/null 2>&1 &   # also retrain + evaluate Laya B4 (+~30 min)
cd "$(dirname "$0")/.." || exit 1
L=logs/context_suite
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E="uv run python scripts/eval_context_suite.py"

step() {  # step <name> <time limit> <command...>
    local name=$1 limit=$2
    shift 2
    echo "$(date '+%F %T') start $name" >> "$L/summary.log"
    timeout "$limit" "$@" > "$L/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $name (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT'))" >> "$L/summary.log"
}

echo "$(date '+%F %T') context suite started" >> "$L/summary.log"
step bge 30m $E biencoder --encoder BAAI/bge-base-en-v1.5 --name bge  # first: also downloads every dataset
step gte 30m $E biencoder --encoder Alibaba-NLP/gte-modernbert-base --name gte
step lidm-1b-gte 45m $E lidm --ckpt checkpoints/stage1b-gte-lora/last.pt --name lidm-1b-gte
step lidm-1b-gte-optsA 30m $E lidm --ckpt checkpoints/stage1b-gte-lora/last.pt --name lidm-1b-gte-optsA --options-in-question
step lidm-1-mbl 90m $E lidm --ckpt checkpoints/stage1-Blora/last.pt --name lidm-1-mbl
step lidm-1-mbl-optsA 45m $E lidm --ckpt checkpoints/stage1-Blora/last.pt --name lidm-1-mbl-optsA --options-in-question
step laya 45m $E laya --name laya
step laya-td 45m $E laya --ckpt convaiinnovations/laya-typed-decisions --name laya-td
if [ -n "$WITH_B4" ]; then
    step laya-b4-train 1h uv run python -m cls.eval.laya_finetune --no-track
    step laya-b4 45m $E laya --ckpt checkpoints/laya-b4 --name laya-b4
fi
$E --report >> "$L/summary.log" 2>&1
echo "$(date '+%F %T') context suite finished" >> "$L/summary.log"
