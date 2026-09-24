#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mini-agent-rl

for seed in 42 43 44; do
  .venv/bin/mini-agent-rl stop-probe \
    --data data/hotpot-agent-v041/validation.jsonl \
    --train-data data/hotpot-agent-v041/train.jsonl \
    --limit 30 --corpus-limit 100 --max-input-tokens 2048 \
    --observation-encoding compact-v3 \
    --adapter-path "checkpoints/v100-answer30-seed${seed}" \
    --model-path /root/autodl-tmp/models/qwen_08b \
    --output "reports/v100-probe120-seed${seed}" \
    --frozen-cases reports/v095-corrected-probe120-sft/cases.jsonl \
    --memory-limit-gb 21 --no-enable-thinking --max-new-tokens 128 \
    --generation-timeout 600 \
    > "reports/v100-probe120-seed${seed}.log" 2>&1
done
