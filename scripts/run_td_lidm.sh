#!/usr/bin/env bash
# Kill-criterion test, LIDM side: fine-tune the stage-1 ModernBERT-large LIDM (encoder LoRA) on typed-decisions
# train at 3 learning rates, pick the run with the best dev accuracy (120 held-out cases), then evaluate only that
# checkpoint on the test split and on the context suite. No time limits: check progress in the logs.
#
#   nohup scripts/run_td_lidm.sh > /dev/null 2>&1 &
#   tail -f logs/td/summary.log                     # steps, dev results, final test metrics
#   tail -f logs/td/td-lidm-mbl-lr1e-4.log | tr '\r' '\n'   # one run's progress
cd "$(dirname "$0")/.." || exit 1
L=logs/td
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" >> "$L/summary.log"; }

[ -f data/td/train.jsonl ] || uv run python scripts/prepare_typed_decisions.py >> "$L/summary.log" 2>&1
log "LIDM typed-decisions sweep started"
for lr in 5e-5 1e-4 3e-4; do
    name=td-lidm-mbl-lr$lr
    log "start $name"
    uv run python -m cls.train --run B --data data/td/train.jsonl --val data/td/dev.jsonl --experiment typed_decisions \
        --encoder answerdotai/ModernBERT-large --encoder-lora 16 --init checkpoints/stage1-Blora/last.pt \
        --epochs 6 --batch 32 --micro-batch 16 --lr "$lr" --max-len 1024 --select acc --name "$name" > "$L/$name.log" 2>&1
    log "end   $name (exit $?): $(tr '\r' '\n' < "$L/$name.log" | grep -E 'done:|Error' | tail -1)"
done

best=$(for f in "$L"/td-lidm-mbl-lr*.log; do
    tr '\r' '\n' < "$f" | grep -oE 'done: best val_acc [0-9.]+' | awk -v n="$(basename "$f" .log)" '{print $NF, n}'
done | sort -rn | head -1 | cut -d' ' -f2)
[ -n "$best" ] || { log "no finished run, stopping"; exit 1; }
log "selected on dev: $best"
uv run python scripts/eval_typed_decisions.py lidm --ckpt "checkpoints/$best/best.pt" --name "$best" > "$L/$best-test.log" 2>&1
log "test: $(grep -E '^\{|Error' "$L/$best-test.log" | tail -1)"
uv run python scripts/eval_context_suite.py lidm --ckpt "checkpoints/$best/best.pt" --name "$best" > "$L/$best-context.log" 2>&1
log "context suite (exit $?):"
grep -E "^\[" "$L/$best-context.log" >> "$L/summary.log"
log "finished"
