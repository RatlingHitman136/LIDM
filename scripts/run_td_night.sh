#!/usr/bin/env bash
# Kill-criterion test, rest of the chain (after scripts/run_td_lidm.sh): test evaluation of the dev-selected LIDM,
# references, the same-data Laya sweep (3 learning rates, dev-selected), paired comparison, then leave one workflow
# out for both models. No time limits: check the logs. Results: logs/td/night.log (summary) and
# `uv run python scripts/compare_typed_decisions.py --table`.
#
#   nohup scripts/run_td_night.sh > /dev/null 2>&1 &
#   tail -f logs/td/night.log
cd "$(dirname "$0")/.." || exit 1
L=logs/td
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "$(date '+%F %T') $*" >> "$L/night.log"; }
run() {  # run <log name> <command...>: log start, end, exit code and the result line
    local name=$1
    shift
    log "start $name"
    "$@" > "$L/$name.log" 2>&1
    log "end   $name (exit $?) $(tr '\r' '\n' < "$L/$name.log" | grep -E '^\{|done:|saved|Error' | tail -1 | cut -c1-400)"
}
acc() { uv run python -c "import json; print(json.load(open('artifacts/typed_decisions/$1.json'))['metrics']['accuracy'])"; }
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
LIDM="uv run python -m cls.train --run B --experiment typed_decisions --encoder answerdotai/ModernBERT-large
      --encoder-lora 16 --init checkpoints/stage1-Blora/last.pt --batch 32 --micro-batch 16 --max-len 1024 --select acc"
LIDM_CKPT=checkpoints/td-lidm-mbl-lr1e-4/best.pt  # dev-selected: 0.805 vs 0.788 (5e-5) and 0.795 (3e-4)

log "night chain started"
# 1. LIDM, dev-selected checkpoint: test split, then the context suite (what fine-tuning costs elsewhere)
run td-lidm-test $EV lidm --ckpt $LIDM_CKPT --name td-lidm
run td-lidm-context $CS lidm --ckpt $LIDM_CKPT --name td-lidm

# 2. references on the test split: published Laya-TD, shipped Laya and LIDM stage 1 (both zero-shot)
run ref-laya-td $EV laya --ckpt convaiinnovations/laya-typed-decisions --name ref-laya-td
run ref-laya $EV laya --ckpt convaiinnovations/laya --name ref-laya-zeroshot
run ref-lidm $EV lidm --ckpt checkpoints/stage1-Blora/last.pt --name ref-lidm-zeroshot

# 3. Laya fine-tuned by its own trainer on the same cases: 3 learning rates (encoder; head 4x, Laya's ratio),
#    Laya's default 4 passes, final checkpoint scored on the same 120 dev cases
for pair in 1.25e-5:5e-5 2.5e-5:1e-4 5e-5:2e-4; do
    elr=${pair%%:*} hlr=${pair#*:}
    run td-laya-elr$elr uv run python -m cls.eval.laya_finetune --data data/td/train_laya.jsonl \
        --out checkpoints/td-laya-elr$elr --epochs 4 --max-len 1024 --encoder-lr "$elr" --head-lr "$hlr" --no-track
    run td-laya-elr$elr-dev $EV laya --ckpt checkpoints/td-laya-elr$elr --test data/td/dev.jsonl \
        --name td-laya-elr$elr-dev --no-track
done
best=$(for pair in 1.25e-5:5e-5 2.5e-5:1e-4 5e-5:2e-4; do
    e=${pair%%:*}; [ -f artifacts/typed_decisions/td-laya-elr$e-dev.json ] && echo "$(acc td-laya-elr$e-dev) $pair"
done | sort -rn | head -1 | cut -d' ' -f2)
[ -n "$best" ] || { log "no Laya run finished, stopping"; exit 1; }
BEST_ELR=${best%%:*} BEST_HLR=${best#*:}
log "Laya selected on dev: encoder lr $BEST_ELR, head lr $BEST_HLR"
run td-laya-test $EV laya --ckpt checkpoints/td-laya-elr$BEST_ELR --name td-laya
run td-laya-context $CS laya --ckpt checkpoints/td-laya-elr$BEST_ELR --name td-laya

# 4. the kill-criterion comparison, paired on the same 2,000 test questions
run cmp-lidm-vs-laya uv run python scripts/compare_typed_decisions.py td-lidm td-laya
run cmp-lidm-vs-laya-td uv run python scripts/compare_typed_decisions.py td-lidm ref-laya-td

# 5. leave one workflow out: train on 3 workflows at each model's selected setting, test on all 4 and read the
#    held-out one (new decision schemas)
for wf in agent_trace_observability customer_service invoice_processing security_incidents; do
    D=data/td/lowo_$wf
    run td-lidm-lowo-$wf $LIDM --data $D/train.jsonl --val $D/dev.jsonl --epochs 4 --lr 1e-4 --name td-lidm-lowo-$wf
    run td-lidm-lowo-$wf-test $EV lidm --ckpt checkpoints/td-lidm-lowo-$wf/best.pt --name td-lidm-lowo-$wf
    run td-laya-lowo-$wf uv run python -m cls.eval.laya_finetune --data $D/train_laya.jsonl \
        --out checkpoints/td-laya-lowo-$wf --epochs 4 --max-len 1024 --encoder-lr "$BEST_ELR" --head-lr "$BEST_HLR" --no-track
    run td-laya-lowo-$wf-test $EV laya --ckpt checkpoints/td-laya-lowo-$wf --name td-laya-lowo-$wf
done

uv run python scripts/compare_typed_decisions.py --table >> "$L/night.log" 2>&1
log "night chain finished"
