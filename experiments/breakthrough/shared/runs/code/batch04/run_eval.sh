#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../../.."
PY=".venv/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
declare -A CANDS=( [repeat_bag]=v6.6.0-scratchgene.rc4 [diverse_bag]=v6.6.0-scratchgene.rc5 )
for arm in repeat_bag diverse_bag; do
  for seed in 20261011 20261012 20261013; do
    echo "=== EVAL $arm $seed $(date '+%H:%M:%S') ==="
    $PY -X utf8 runs/breakthrough_20261002/code/batch04/eval_bag_v2.py \
      --arm "$arm" --seed "$seed" --lr 0.001 \
      --candidate "${CANDS[$arm]}" \
      --checkpoint "models/breakthrough_20261002/batch04/${arm}_s${seed}/final.pt"
  done
done
echo "ALL_EVAL_DONE"
