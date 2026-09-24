#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/mini-agent-rl

for seed in 42 43 44; do
  .venv/bin/mini-agent-rl train-grpo \
    --model-path /root/autodl-tmp/models/qwen_08b \
    --adapter-path checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 \
    --data data/hotpot-agent-v041/train.jsonl \
    --task-limit 50 --corpus-limit 100 --retriever-version bm25-v2 \
    --group-size 4 --temperature 1.0 \
    --output-dir "checkpoints/v101-answer50-seed${seed}" --seed "$seed" --max-steps 3 \
    --stability-report reports/v099-logprob-stability.json \
    --db "runs/v101-answer50-seed${seed}.db" \
    --report "reports/v101-answer50-seed${seed}.json" \
    --no-load-in-4bit --max-input-tokens 2048 --observation-encoding compact-v3 \
    --exact-weight 1.0 --repeated-penalty -0.5 --evidence-stop-weight 0.2 \
    --incomplete-evidence-answer-penalty -0.15 --empty-evidence-answer-penalty -0.1 \
    --reward-version v098-answer-v1 --memory-limit-gb 21 --no-gradient-checkpointing \
    --max-new-tokens 128 --generation-timeout 600 --no-enable-thinking \
    --learning-rate 5e-6 --clip-epsilon 0.2 --kl-beta 0.02 --policy-epochs 2 \
    --max-grad-norm 1.0 --min-effective-groups 25 --no-anti-repeat-prompt \
    > "reports/v101-answer50-seed${seed}.log" 2>&1
done
