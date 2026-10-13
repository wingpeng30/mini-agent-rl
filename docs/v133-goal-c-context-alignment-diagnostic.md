# v133 Goal C context alignment diagnosis and one-run amendment

Recorded before any inference for this amendment. The original Goal C run and its result remain immutable. This is a post hoc diagnostic requested after its protocol failure, not a replacement for the precommitted 16-case result.

## Observed mismatch

All 25 frozen SFT rows have the role sequence `system,user,assistant(search),tool,assistant(search),tool,assistant(answer)`. All 16 frozen dev rows have `system,user,tool,assistant(answer)`. The Qwen chat template renders a `tool` message as a user `tool_response`, so all evidence is visible; the important difference is the missing assistant search history. The original 32 generations all selected `search`, producing zero protocol-valid answers. These observations identify a plausible cause, not a proven causal effect.

## Fixed transformation and scope

For each original dev row, use the unchanged system message, question, two evidence objects, and target answer. Construct the following prompt without consulting the target answer:

1. Assistant search query = original question.
2. First tool result = original first evidence object.
3. Assistant search query = first evidence object's text, which is visible at this point.
4. Second tool result = original second evidence object.
5. Generate the answer action.

This matches the train rows' role sequence and keeps all source evidence. It is a synthetic search history, not a claim that the agent actually retrieved those facts. Training tool results have three entries per turn while the derived dev turns each have one; the structural mismatch is addressed, but the content distribution is still different. No source data, labels, adapter weights, hyperparameters, or scoring code are changed. The existing two adapters each receive the same 16 derived prompts, greedy decoding, thinking disabled, maximum 128 new tokens. The original dev hash remains `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`.

Implementation: `scripts/evaluate_v133_goal_c_aligned_context.py`, SHA-256 `1308fb0c4aaa4b7dbccaefb7c583fb277c83f543ef941ab7331825a28d8c04a3`. CPU inspection of all 16 transformed prompts passed: both evidence JSON messages render in each prompt, role sequence matches training, and prompt lengths are 329–513 tokens, under the frozen 2048-token inference limit. The original evaluation script and frozen manifest are unchanged.

## Decision and stopping rules

Run at most one new paired evaluation (16 prompts × 2 adapters) on the user-selected AutoDL host, with a 60-minute hard cap and the original first-two-output speed projection. Save to a new output path; never overwrite or resume the original result. Record action validity, EM, F1, newly/lost/retained correct cases, raw generations, and elapsed time. If any arm is incomplete or continues to emit no valid answers, conclude technical failure of this diagnostic. If SFT gains fewer than two exact matches, stop the current adapter recipe. A net gain of at least two is exploratory until all changed outputs receive evidence-support review under the existing anonymous review rules; no positive claim follows from EM alone. Do not train again, alter prompts after seeing outputs, or expand the data pool as part of this amendment.
