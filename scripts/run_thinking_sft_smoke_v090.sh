#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mini-agent-rl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs
.venv/bin/mini-agent-rl train-sft \
  --model-path /models/qwen_08b \
  --adapter-path checkpoints/qwen35-08b-sft-v052 \
  --train-data data/thinking-sft-v090/train.jsonl \
  --validation-data data/thinking-sft-v090/validation.jsonl \
  --output-dir checkpoints/qwen35-08b-thinking-sft-v090-smoke \
  --max-steps 10 --epochs 2 --max-length 2560 --seed 42 \
  --batch-size 1 --gradient-accumulation-steps 8 \
  --no-gradient-checkpointing --enable-thinking --learning-rate 2e-5 \
  2>&1 | tee logs/thinking-sft-v090-smoke.log
