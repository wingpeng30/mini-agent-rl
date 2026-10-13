# v133 Goal C context alignment diagnostic result

**Finding:** changing the original evaluation's prompt context restored answer-action validity. With a deterministic two-search/two-tool history matching the SFT turn structure, both fixed adapters produced 16/16 valid answer actions. The existing seven-step SFT adapter made no observable difference on these 16 derived prompts: every raw generation matched v118 exactly. The diagnostic changes the action history and how the two evidence items are grouped, so it does not isolate which detail caused the change. This is a post hoc diagnostic, not a replacement for the original frozen run or an end-to-end search score.

## Why this diagnostic was necessary

- All 25 SFT rows: `system,user,assistant(search),tool,assistant(search),tool,assistant(answer)`.
- All 16 original dev rows: `system,user,tool,assistant(answer)`.
- The Qwen chat template did include the original dev evidence, but the original evaluation generated `search` on all 32 attempts. The difference in assistant action history was a concrete context mismatch.
- The transformation and decision rules were recorded before this diagnostic inference in `docs/v133-goal-c-context-alignment-diagnostic.md`. It uses the original question and the two unchanged evidence objects to form an answer-blind synthetic search history. The gold answer is never read to form a query.

## Results

| Measure | v118 | Existing SFT adapter |
|---|---:|---:|
| Prompts | 16 | 16 |
| Valid answer actions | 16/16 | 16/16 |
| Strict normalized EM | 12/16 | 12/16 |
| Mean token F1 | 0.8417 | 0.8417 |

There were 0 newly correct, 0 lost correct, and 12 retained correct cases. All 16 paired raw outputs were identical. The first-two-output speed projection was 27.153 s; the complete paired evaluation took 33.87 s, within its 60-minute cap. The original 25/16 data, v118 and SFT adapters, tokenizer, scoring function, decoding settings, and old run files were unchanged. This diagnostic was run once without a retry; no further training or external review was performed.

The four strict EM errors, identical in both arms, are:

| Case | Gold | Both outputs | Observation |
|---:|---|---|---|
| 0 | New Orleans Gators | GMGB | The answer is the team, while GMGB is the league. |
| 2 | rock | folk rock | The first band is folk rock; the shared broader genre is rock. |
| 3 | Tad | The Sounds | The visible formation years are Tad 1988 and The Sounds 1998. |
| 15 | Mary Eileen McDonnell | Mary McDonnell | This may be an acceptable name variant, but strict EM does not credit it. No semantic adjudication was performed. |

The corrected protocol answers the user's immediate question: the old 0/16 values were caused by evaluation context and protocol failure, so they were not a measure of the reviewed data's usefulness. Under the aligned diagnostic, the existing v118 adapter already answers 12/16 exactly, and the small SFT produces the same outputs. The current SFT recipe has no measured gain on this development set. The synthetic history and 16-case train-derived dev set limit generalization; this result does not establish end-to-end performance or independent-test quality.

## Audit

- Diagnostic code SHA-256: `1308fb0c4aaa4b7dbccaefb7c583fb277c83f543ef941ab7331825a28d8c04a3`.
- Original dev data SHA-256: `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`.
- v118 adapter SHA-256: `ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5`.
- Existing SFT adapter SHA-256: `0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6`.
- Original paired result SHA-256: `86cc52318ccf07afb1839971585e1efbc7a497e6caf0995b115814b3b5fc120c`.
- This diagnostic's `paired-evaluation.json` SHA-256: `fe8ee1c6e0a6c4633a6dcf586115277c7b352aa99a0b428b9102b9413b49d203`. Local and AutoDL copies matched.
- CPU rendering check: all 16 derived prompts preserved both evidence messages, matched the seven-role train structure, and had 329–513 prompt tokens.
