#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/mini-agent-rl
common=(--model-path /root/autodl-tmp/models/qwen_08b
  --gate-report reports/v112-frozen-data-gate.json
  --train-data data/evidence-sft-v112/train.jsonl
  --validation-data data/evidence-sft-v112/validation.jsonl)
for arm in baseline candidate; do
  if [[ "$arm" == baseline ]]; then
    adapter=checkpoints/v101-answer50-seed42
    expected=3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768
  else
    adapter=checkpoints/v112-evidence-sft-seed42/formal
    expected=31b625a9e3934012f9ba9b2f2054006ab968caeabb5238c5934988edccf423fd
  fi
  for mode in states tasks; do
    timeout --signal=TERM --kill-after=10s 1800 .venv/bin/python scripts/evaluate_v112.py run \
      --mode "$mode" "${common[@]}" --adapter-path "$adapter" \
      --expected-adapter-sha256 "$expected" --output "reports/v112-${arm}-${mode}.json" \
      > "reports/v112-${arm}-${mode}.log" 2>&1
  done
done
.venv/bin/python scripts/evaluate_v112.py compare \
  --base-tasks reports/v112-baseline-tasks.json \
  --candidate-tasks reports/v112-candidate-tasks.json \
  --candidate-states reports/v112-candidate-states.json \
  --output reports/v112-internal-comparison.json
