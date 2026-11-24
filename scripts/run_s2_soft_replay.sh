#!/usr/bin/env bash
# Two next-stage runs on top of the adopted LIDM-X (checkpoints/s2x-lidm), one after the other, then their tests:
#   (1) s2xs-lidm: one more stage-2 pass with soft targets (data/stage2_soft: same questions as stage 2; 26k of them
#       carry teacher / annotator distributions), then 2c on typed decisions with its soft gold (s2xsc-lidm)
#   (2) s2xr-lidm: one more stage-2 pass with hard labels + 20% stage-1 replay (data/stage2/train_replay.jsonl)
# Both: lr 5e-5 (half the first pass), otherwise exactly the s2x-lidm recipe. No time limits: logs/s2_soft_replay/.
#
#   nohup scripts/run_s2_soft_replay.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
L=logs/s2_soft_replay
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {  # run <name> <command...>
    local n=$1
    shift
    echo "$(date '+%F %T') start $n" >> "$L/summary.log"
    "$@" > "$L/$n.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $n (exit $rc) $(tr '\r' '\n' < "$L/$n.log" | grep -E '^\{|done:' | tail -1 | cut -c1-300)" >> "$L/summary.log"
}
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
CMP="uv run python scripts/compare_typed_decisions.py"
ITEMS="data/stage2/eval_heldout_sources.jsonl data/stage2/eval_banking77_names.jsonl data/stage2/eval_banking77_desc.jsonl data/stage2/eval_clinc150.jsonl"
S2="--run B --experiment stage2 --encoder answerdotai/ModernBERT-large --encoder-lora 16 --option-attn state \
    --init checkpoints/s2x-lidm/last.pt --val data/stage2/dev.jsonl --epochs 1 --batch 32 --micro-batch 8 --lr 5e-5 \
    --max-len 1024 --max-opt-len 128 --val-every 1000"

# (1) soft targets
run s2xs-train uv run python -m cls.train $S2 --data data/stage2_soft/train.jsonl --soft-targets --name s2xs-lidm
S=checkpoints/s2xs-lidm/last.pt
run s2xs-td $EV lidm --ckpt $S --name s2xs-lidm
run s2xs-cmp $CMP s2xs-lidm s2x-lidm
run s2xs-context $CS lidm --ckpt $S --name s2xs-lidm
run s2xs-labels $CS lidm --ckpt $S --name s2xs-lidm --items $ITEMS
run s2xsc-train uv run python -m cls.train --run B --experiment typed_decisions --encoder answerdotai/ModernBERT-large \
    --encoder-lora 16 --init $S --data data/td/train.jsonl --val data/td/dev.jsonl --epochs 6 --batch 32 --micro-batch 16 \
    --lr 1e-4 --max-len 1024 --select acc --option-attn state --soft-targets --name s2xsc-lidm
run s2xsc-td $EV lidm --ckpt checkpoints/s2xsc-lidm/best.pt --name s2xsc-lidm
run s2xsc-dev $EV lidm --ckpt checkpoints/s2xsc-lidm/best.pt --test data/td/dev.jsonl --name s2xsc-lidm-dev --no-track
run s2xsc-cmp $CMP s2xsc-lidm s2xc-lidm
run s2xsc-cmp-laya $CMP s2xsc-lidm s2c-laya
run s2xsc-selective uv run python scripts/selective_typed_decisions.py s2xsc-lidm s2xc-lidm s2c-laya --calibrate
run s2xsc-selective-raw uv run python scripts/selective_typed_decisions.py s2xsc-lidm s2xc-lidm s2c-laya

# (2) second pass with stage-1 replay
run s2xr-train uv run python -m cls.train $S2 --data data/stage2/train_replay.jsonl --name s2xr-lidm
R=checkpoints/s2xr-lidm/last.pt
run s2xr-td $EV lidm --ckpt $R --name s2xr-lidm
run s2xr-cmp $CMP s2xr-lidm s2x-lidm
run s2xr-context $CS lidm --ckpt $R --name s2xr-lidm
run s2xr-labels $CS lidm --ckpt $R --name s2xr-lidm --items $ITEMS

{ echo "== soft targets vs s2x-lidm"; uv run python scripts/summarize_lidmx.py --x s2xs-lidm --lidm s2x-lidm --x2c s2xsc-lidm
  echo "== replay vs s2x-lidm"; uv run python scripts/summarize_lidmx.py --x s2xr-lidm --lidm s2x-lidm; } >> "$L/summary.log" 2>&1
echo "$(date '+%F %T') soft-target + replay chain finished" >> "$L/summary.log"
