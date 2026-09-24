#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mini-agent-rl

probe() {
  local label="$1"
  local adapter="$2"
  .venv/bin/mini-agent-rl stop-probe \
    --data data/hotpot-agent-v041/validation.jsonl \
    --train-data data/hotpot-agent-v041/train.jsonl \
    --limit 30 --corpus-limit 100 --max-input-tokens 2048 \
    --observation-encoding compact-v3 --adapter-path "$adapter" \
    --model-path /root/autodl-tmp/models/qwen_08b \
    --output "reports/v099-probe120-${label}" \
    --frozen-cases reports/v095-corrected-probe120-sft/cases.jsonl \
    --memory-limit-gb 21 --no-enable-thinking --max-new-tokens 128 \
    --generation-timeout 600 \
    > "reports/v099-probe120-${label}.log" 2>&1
}

probe sft checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200
probe v096-seed42 checkpoints/v099-v096-seed42
probe v096-seed43 checkpoints/v099-v096-seed43
probe v096-seed44 checkpoints/v099-v096-seed44
probe answer-seed42 checkpoints/v099-answer-seed42
probe answer-seed43 checkpoints/v099-answer-seed43
probe answer-seed44 checkpoints/v099-answer-seed44
