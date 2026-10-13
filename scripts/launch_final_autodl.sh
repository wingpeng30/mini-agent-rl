#!/usr/bin/env bash
set -Eeuo pipefail
task_root=/root/autodl-tmp/final-online-grpo-20261012
assets=/root/autodl-tmp/hybrid-adaptation-20261006
cd "$task_root/code"
export PYTHONPATH="$task_root/code/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FINAL_STARTED_EPOCH=1791737888
exec bash scripts/run_final_autodl.sh "$task_root/results" /root/miniconda3/bin/python \
  "$task_root/code/scripts/run_final_online_grpo.py" \
  --base-model /root/autodl-tmp/models/qwen_08b \
  --parent-adapter "$assets/code/runs/v133-goal-am-action-sft-v2/seed42-A/adapter-final" \
  --train-cases "$assets/experiment/selected-train.jsonl" \
  --train-inputs "$assets/experiment/training-inputs.jsonl" \
  --eval-cases "$assets/results/v133-hybrid-model-pair-20261011/cases.jsonl" \
  --eval-retrieval-config "$assets/experiment/retrieval-remote.json" \
  --train-corpus "$assets/experiment/corpus-train.json" \
  --train-index "$assets/experiment/training-index" \
  --batch-size 4 --eval-batch-size 1 --stage all
