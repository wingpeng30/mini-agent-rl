#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mini-agent-rl

evaluate() {
  local label="$1"
  local adapter="$2"
  .venv/bin/mini-agent-rl evaluate-adapter \
    --model-path /root/autodl-tmp/models/qwen_08b --adapter-path "$adapter" \
    --data data/hotpot-agent-v041/validation.jsonl --limit 20 --corpus-limit 100 \
    --retriever-version bm25-v2 --max-input-tokens 2048 --max-new-tokens 128 \
    --no-enable-thinking --generation-timeout 600 --seed 42 \
    --observation-encoding compact-v3 --no-anti-repeat-prompt \
    --output "reports/v099-eval20-${label}.json" --db "runs/v099-eval20-${label}.db" \
    > "reports/v099-eval20-${label}.log" 2>&1
}

evaluate sft checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200
evaluate v096-seed42 checkpoints/v099-v096-seed42
evaluate v096-seed43 checkpoints/v099-v096-seed43
evaluate v096-seed44 checkpoints/v099-v096-seed44
evaluate answer-seed42 checkpoints/v099-answer-seed42
evaluate answer-seed43 checkpoints/v099-answer-seed43
evaluate answer-seed44 checkpoints/v099-answer-seed44
