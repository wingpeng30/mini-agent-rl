# Goal J protocol (v1)

## Research question
Does one fixed-budget SFT update from v118 on newly selected common-category evidence-answer cases improve answers to source- and entity-isolated cases under the same synthetic evidence context?

## Scope and budget
Use only the two previously acquired official HotpotQA distractor/train Parquet shards. Scan only ID, question, and type to locate candidates. Open at most the first 200 eligible full rows in frozen SHA256 order. Do not inspect final-test or the sealed independent 100 cases. Target 16 training and 16 evaluation cases; each side must have at least 8 fully approved and CPU-ready cases for GPU work. No automatic expansion.

## Capability and case acceptance
The task is an open-ended question asking for a category or attribute shared by two named entities. Keep the original question and official answer unchanged. Two distinct official supporting documents must each directly support the same answer at compatible granularity. Exclude yes/no, date/order, single-entity lookups, unsupported or contradictory evidence, ambiguous granularity, and any case needing a rewritten answer. Search 1 is the original question; Search 2 is the first visible evidence text; neither may use target answers or future evidence.

## Exposure, groups, and partition
Exclude known exposed source IDs and normalized questions, including all Goal G screened rows, Goal G frozen evaluation, Goal H fit probes, and prior training/development indexes. Exclude unknown exposure. A source/entity group may contribute at most one case. Group links use source ID, both evidence document IDs/titles, and named entities plus clear aliases after manual review; generic answer-category terms do not count as entities. Freeze the groups before partitioning. Assign each group by the parity of the first byte of `SHA256("v133-goal-j-v1" + sorted_group_source_ids_joined_by_newline)`: even to train, odd to evaluation. Within each side sort by the same digest and take up to 16. Never change a partition based on review or model results.

## Blind review and CPU acceptance
DeepSeek and an isolated GPT-6 Luna reviewer receive the same anonymous packet: original question, both complete visible documents, unchanged official answer, and this protocol. They independently assess capability fit, direct support on both sides, answer granularity, ambiguity, and leakage. Two valid approvals are required. One technical correction maximum per invalid ticket; no retries after unknown API outcomes and no re-review after a valid rejection. Encode with the existing assistant-only training function. Require exact generation/supervision prefix equality, correct final assistant and causal labels, correct padding/terminator, complete evidence, no target leakage, prompt <=2048, target <=128, total <=2176 tokens. Mark answer-value and format tokens separately.

## Frozen experiment
If both sides have >=8 approved and CPU-ready cases, train once from v118 with seed 42, BF16, thinking disabled, AdamW, LR 1e-4, weight decay 0, gradient clip 1, microbatch 1, accumulation equal to train count, 20 updates, linear schedule, 1 warmup update. No trainer evaluation set; use final adapter only. Greedy decode, max 128 tokens. GPU wall clock cap 7200 seconds. Paired evaluation: v118 and final adapter on the exact same frozen train, Goal G regression, and new evaluation inputs. The new evaluation file is never made available to the trainer or checkpoint selection.

## Reporting and stop rules
Report strict EM, F1, protocol validity, added/lost/retained correct items, train answer/format token NLL and accuracy, and review dispositions. Strict EM does not change for aliases. Blind-review every new strict hit and every strict loss; unsupported/contested results are not counted as evidence of benefit. Call results exploratory, not confirmatory. Stop after the 200-row full-text budget, if either eligible side has fewer than 8 cases, on review-channel failure, on CPU/data integrity failure, or when projected GPU work exceeds 7200 seconds. Preserve partial results and do not retry, tune, or expand.
