#!/usr/bin/env bash
# Overnight stage 1b (label-variety data): LIDM on gte-modernbert-base, then LIDM on ModernBERT-large, each followed
# by the retrieval-weight fit, the Banking77 test (3,080) and the head diagnostics. Every step has a hard time limit
# and its own log in logs/overnight/; logs/overnight/summary.log has start/end times, exit codes and the results.
#
#   nohup scripts/run_stage1b_overnight.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
L=logs/overnight
mkdir -p "$L"
export PYTHONUNBUFFERED=1

step() {  # step <name> <time limit> <command...>
    local name=$1 limit=$2
    shift 2
    echo "$(date '+%F %T') start $name" >> "$L/summary.log"
    timeout "$limit" "$@" > "$L/$name.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $name (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT'))" >> "$L/summary.log"
}

evaluate() {  # evaluate <run name>
    local run=$1 ck=checkpoints/$1
    step "$run-fit" 1h uv run python scripts/fit_prior_weight.py "$ck/last.pt"
    step "$run-banking77" 1h uv run python scripts/eval_banking77.py "$ck/last.pt" "$ck/fit.pt" --n 3080
    step "$run-diagnose" 1h uv run python scripts/diagnose_head.py "$ck/last.pt" --train data/stage1b_train.jsonl
    {
        echo "== $run"
        grep -E "trained s =|best s on zero-shot" "$L/$run-fit.log"
        grep -E '^\{' "$L/$run-banking77.log"
        grep -A40 '^{' "$L/$run-diagnose.log" | tr -d '\n' | sed 's/  */ /g'
        echo
        tr '\r' '\n' < "$L/${run}-train.log" | grep -E "step [0-9]+ \(" | tail -2
    } >> "$L/summary.log"
}

D="--run B --stage1 --data data/stage1b_train.jsonl --val data/stage1b_val.jsonl --val-every 500 --encoder-lora 16"
echo "$(date '+%F %T') overnight stage 1b started" >> "$L/summary.log"
step stage1b-gte-lora-train 6h uv run python -m cls.train $D --encoder Alibaba-NLP/gte-modernbert-base \
    --option-vector pooled --prior self --prior-dropout 0.5 --name stage1b-gte-lora
evaluate stage1b-gte-lora
step stage1b-mbl-lora-train 7h uv run python -m cls.train $D --name stage1b-mbl-lora
evaluate stage1b-mbl-lora
echo "$(date '+%F %T') overnight stage 1b finished" >> "$L/summary.log"
