#!/usr/bin/env bash
# Continuation supervisor: existing GPU experiment, then untouched-base128, then off.
set -Eeuo pipefail
task_root=/root/autodl-tmp/final-online-grpo-20261012
assets=/root/autodl-tmp/hybrid-adaptation-20261006
original_pid="${1:?original experiment PID is required}"
cd "$task_root/code"
export PYTHONPATH="$task_root/code/src"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FINAL_STARTED_EPOCH=1791737888
export FINAL_ADDITIONAL_PRODUCER_PID="$original_pid"
export FINAL_METRIC_CASES="$assets/results/v133-hybrid-model-pair-20261011/cases.jsonl"
export FINAL_METRIC_SOURCE_AUDIT="$assets/results/v133-hybrid-model-pair-20261011/source-audit.jsonl"
exec bash scripts/run_final_autodl.sh "$task_root/results" /root/miniconda3/bin/python \
  "$task_root/code/scripts/evaluate_final_bare.py" \
  --base-model /root/autodl-tmp/models/qwen_08b \
  --eval-cases "$FINAL_METRIC_CASES" \
  --eval-retrieval-config "$assets/experiment/retrieval-remote.json" \
  --wait-pid "$original_pid"
