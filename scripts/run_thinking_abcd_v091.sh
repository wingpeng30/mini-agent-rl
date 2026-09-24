#!/usr/bin/env bash
# Frozen O/A/B/C comparison.  Run under an outer timeout to bound aggregate GPU time.
set -euo pipefail
cd /root/autodl-tmp/mini-agent-rl
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_VERSION="${RUN_VERSION:-v091}"
mkdir -p logs runs "reports/thinking-abcd-${RUN_VERSION}"

run_eval() {
  local group="$1" adapter="$2" tokens="$3" thinking="$4"
  local flags=()
  if [ "$thinking" = "true" ]; then flags+=(--enable-thinking); fi
  .venv/bin/mini-agent-rl evaluate-adapter \
    --model-path /models/qwen_08b --adapter-path "$adapter" \
    --data data/hotpot-agent-v041/test.jsonl --limit 100 --corpus-limit 100 \
    --retriever-version bm25-v2 --max-input-tokens 2048 --max-new-tokens "$tokens" \
    --observation-encoding compact-v3 --seed 42 --generation-timeout 600 \
    --db "runs/thinking-abcd-${RUN_VERSION}-${group}.db" --output "reports/thinking-abcd-${RUN_VERSION}/${group}-development.json" "${flags[@]}"
}

run_probe() {
  local group="$1" adapter="$2" tokens="$3" thinking="$4"
  local flags=()
  if [ "$thinking" = "true" ]; then flags+=(--enable-thinking); fi
  .venv/bin/mini-agent-rl stop-probe \
    --model-path /models/qwen_08b --adapter-path "$adapter" \
    --data data/hotpot-agent-v041/validation.jsonl --train-data data/hotpot-agent-v041/train.jsonl \
    --limit 30 --corpus-limit 100 --frozen-cases reports/v087-stop-2048/cases.jsonl \
    --max-input-tokens 2048 --max-new-tokens "$tokens" --observation-encoding compact-v3 \
    --memory-limit-gb 21 --generation-timeout 600 --output "reports/thinking-abcd-${RUN_VERSION}/${group}-probe" "${flags[@]}"
}

run_eval O checkpoints/qwen35-08b-sft-v052 128 false
run_eval A checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 128 false
run_eval B checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 512 false
run_eval C checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 512 true
run_probe O checkpoints/qwen35-08b-sft-v052 128 false
run_probe A checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 128 false
run_probe B checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 512 false
run_probe C checkpoints/qwen35-08b-thinking-sft-v090/checkpoint-200 512 true
