#!/usr/bin/env bash
# Option A on top of the adopted LIDM-X, started after scripts/run_soft2c.sh:
#   3a  soft labels from human votes (data/stage3: GoEmotions + DICES + 25% soft stage-2 replay), lr 1e-5, one pass
#   3b  temperatures per question type x option-count bucket on a held-out calibration set (fit_temperatures.py)
#   abstain by confidence (eval_abstain_confidence.py)
# Then: vote-distribution match (GoEmotions / DICES / ChaosNLI), typed decisions and the context suite, with and
# without temperatures. No time limits; logs and summary in logs/stage3/.
cd "$(dirname "$0")/.." || exit 1
L=logs/stage3
mkdir -p "$L"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { local n=$1; shift; echo "$(date '+%F %T') start $n" >> "$L/summary.log"; "$@" > "$L/$n.log" 2>&1; local rc=$?
        echo "$(date '+%F %T') end   $n (exit $rc) $(tr '\r' '\n' < "$L/$n.log" | grep -E '^\{|done:|->' | tail -1 | cut -c1-300)" >> "$L/summary.log"; }
for _ in $(seq 1 240); do grep -q "soft 2c finished" "$L/summary.log" && break; sleep 30; done
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
SL="uv run python scripts/eval_soft_labels.py"
X=checkpoints/s2x-lidm/last.pt
A=checkpoints/s3a-lidm/last.pt

# 3a
run s3a-train uv run python -m cls.train --run B --experiment stage3 --encoder answerdotai/ModernBERT-large --encoder-lora 16 \
    --option-attn state --init $X --data data/stage3/train.jsonl --val data/stage3/dev.jsonl --epochs 1 --batch 32 \
    --micro-batch 8 --lr 1e-5 --max-len 1024 --max-opt-len 128 --val-every 1000 --soft-targets --name s3a-lidm
run s2x-votes $SL lidm --ckpt $X --name s2x-lidm
run s3a-votes $SL lidm --ckpt $A --name s3a-lidm
run laya-votes $SL laya --ckpt checkpoints/s2-laya --name s2-laya

# 3b
run s2x-temps uv run python scripts/fit_temperatures.py $X
run s3a-temps uv run python scripts/fit_temperatures.py $A
run s2x-votes-t $SL lidm --ckpt $X --name s2x-lidm-t --temps checkpoints/s2x-lidm/temperatures.json
run s3a-votes-t $SL lidm --ckpt $A --name s3a-lidm-t --temps checkpoints/s3a-lidm/temperatures.json

# typed decisions (unseen schemas) and the context suite: raw and with the general-purpose temperatures
run s3a-td $EV lidm --ckpt $A --name s3a-lidm
run s2x-td-t $EV lidm --ckpt $X --name s2x-lidm-t --temps checkpoints/s2x-lidm/temperatures.json --no-track
run s3a-td-t $EV lidm --ckpt $A --name s3a-lidm-t --temps checkpoints/s3a-lidm/temperatures.json --no-track
run s3a-cmp uv run python scripts/compare_typed_decisions.py s3a-lidm s2x-lidm
run s3a-context $CS lidm --ckpt $A --name s3a-lidm
run s2x-context-t $CS lidm --ckpt $X --name s2x-lidm-t --temps checkpoints/s2x-lidm/temperatures.json --no-track
run s3a-context-t $CS lidm --ckpt $A --name s3a-lidm-t --temps checkpoints/s3a-lidm/temperatures.json --no-track

# abstain by confidence
run abstain-conf uv run python scripts/eval_abstain_confidence.py --lidm $X $A --laya checkpoints/s2-laya
echo "$(date '+%F %T') stage 3 chain finished" >> "$L/summary.log"
