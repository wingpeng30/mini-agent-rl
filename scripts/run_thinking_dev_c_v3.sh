#!/usr/bin/env bash
# Fixed prompt-development sample.  Kept separate from formal A/B/C evaluation.
set -euo pipefail
cd /root/autodl-tmp/mini-agent-rl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p logs
.venv/bin/mini-agent-rl collect-rl \
  --model-path /models/qwen_08b \
  --adapter-path checkpoints/qwen35-08b-sft-v052 \
  --data data/hotpot-agent-v041/train.jsonl \
  --task-limit 10 --corpus-limit 100 --retriever-version bm25-v2 \
  --group-size 1 --max-steps 3 --max-input-tokens 2048 --max-new-tokens 512 \
  --enable-thinking --generation-timeout 600 \
  --db runs/thinking-dev-think-512-v4.db \
  --output runs/thinking-dev-think-512-v4.jsonl \
  --report reports/thinking-dev-think-512-v4.json \
  2>&1 | tee logs/thinking-dev-think-512-v4.log
