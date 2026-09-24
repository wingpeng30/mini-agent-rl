#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mini-agent-rl

for seed in 42 43 44; do
  .venv/bin/mini-agent-rl evaluate-adapter \
    --model-path /root/autodl-tmp/models/qwen_08b \
    --adapter-path "checkpoints/v100-answer30-seed${seed}" \
    --data data/hotpot-agent-v041/validation.jsonl --limit 100 --corpus-limit 100 \
    --retriever-version bm25-v2 --max-input-tokens 2048 --max-new-tokens 128 \
    --no-enable-thinking --generation-timeout 600 --seed 42 \
    --observation-encoding compact-v3 --no-anti-repeat-prompt \
    --output "reports/v100-eval100-seed${seed}.json" \
    --db "runs/v100-eval100-seed${seed}.db" \
    > "reports/v100-eval100-seed${seed}.log" 2>&1
done
