#!/usr/bin/env bash
# After scripts/run_s1x.sh: (1) LIDM-DJ on stage 2 (from the stage-1 MBL LoRA checkpoint, same recipe as s2-lidm /
# s2x-lidm) + evaluation; (2) LIDM-X stage 2 from its own stage 1 (s1x-lidm) + evaluation. Hard time limit per step;
# logs in logs/full/, summary in logs/full/summary.log.   MAXTOK=48000 nohup scripts/run_full_chain.sh > /dev/null 2>&1 &
cd "$(dirname "$0")/.." || exit 1
L=logs/full
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() {  # step <name> <limit> <command...>
    local n=$1 lim=$2
    shift 2
    echo "$(date '+%F %T') start $n" >> "$L/summary.log"
    timeout "$lim" "$@" > "$L/$n.log" 2>&1
    local rc=$?
    echo "$(date '+%F %T') end   $n (exit $rc$([ $rc -eq 124 ] && echo ', TIME LIMIT')) $(tr '\r' '\n' < "$L/$n.log" | grep -E '^\{|done:' | tail -1 | cut -c1-300)" >> "$L/summary.log"
}
EV="uv run python scripts/eval_typed_decisions.py"
CS="uv run python scripts/eval_context_suite.py"
CMP="uv run python scripts/compare_typed_decisions.py"
ITEMS="data/stage2/eval_heldout_sources.jsonl data/stage2/eval_banking77_names.jsonl data/stage2/eval_banking77_desc.jsonl data/stage2/eval_clinc150.jsonl"
S2="--run B --experiment stage2 --encoder answerdotai/ModernBERT-large --encoder-lora 16 --data data/stage2/train.jsonl \
    --val data/stage2/dev.jsonl --epochs 1 --batch 32 --micro-batch 8 --lr 1e-4 --max-len 1024 --max-opt-len 128 --val-every 1000"

step s1x-banking77 30m uv run python scripts/eval_banking77.py checkpoints/s1x-lidm/last.pt --n 3080
# (1) LIDM-DJ
step s2dj-train 6h uv run python -m cls.train $S2 --init checkpoints/stage1-Blora/last.pt --option-attn dj --joint-layers 7 \
    --max-tokens "${MAXTOK:-48000}" --name s2dj-lidm
D=checkpoints/s2dj-lidm/last.pt
step s2dj-td 45m $EV lidm --ckpt $D --name s2dj-lidm
step s2dj-cmp-x 5m $CMP s2dj-lidm s2x-lidm
step s2dj-cmp-laya 5m $CMP s2dj-lidm s2-laya
step s2dj-context 60m $CS lidm --ckpt $D --name s2dj-lidm
step s2dj-labels 60m $CS lidm --ckpt $D --name s2dj-lidm --items $ITEMS
uv run python scripts/summarize_lidmx.py --x s2dj-lidm >> "$L/summary.log" 2>&1
# (2) LIDM-X stage 2 from its own stage 1
X=checkpoints/s2xf-lidm/last.pt
step s2xf-train 4h uv run python -m cls.train $S2 --init checkpoints/s1x-lidm/last.pt --option-attn state --name s2xf-lidm
step s2xf-td 30m $EV lidm --ckpt $X --name s2xf-lidm
step s2xf-cmp-x 5m $CMP s2xf-lidm s2x-lidm
step s2xf-cmp-laya 5m $CMP s2xf-lidm s2-laya
step s2xf-context 45m $CS lidm --ckpt $X --name s2xf-lidm
step s2xf-labels 45m $CS lidm --ckpt $X --name s2xf-lidm --items $ITEMS
uv run python scripts/summarize_lidmx.py --x s2xf-lidm >> "$L/summary.log" 2>&1
echo "$(date '+%F %T') full chain finished" >> "$L/summary.log"
