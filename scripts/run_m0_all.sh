#!/usr/bin/env bash
cd "$(dirname "$0")/.."
for m in laya laya-b3 laya-td laya-b4 biencoder-mb biencoder-bge laya-short-bge laya-tourn; do
  echo "=== $m"; uv run python -m cls.eval.run_m0 --model $m 2>&1 | grep -v -i "warn\|Fetching\|CardData"
done
echo ALLDONE
