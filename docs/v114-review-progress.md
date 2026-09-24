# v114 evidence-SFT dataset package

This run completed dataset construction and review only. The model was not trained, no GPU was used, and no API fee bookkeeping was performed. `final-test` and the independent 100-question evaluation were not accessed; both remain unevaluated.

## Frozen datasets

The fail-closed join reviewed 975 candidate states. Both reviewers have complete coverage and there are no pending rows: 437 states have two valid approvals, 296 were rejected by both, 178 were disputed, and 64 were quarantined for invalid review records. The original 419-state work, 250 approved records, reserve waves, rejection history, and invalid records remain preserved. One Battlefield approval was repaired only by a hash-bound append-only rereview; its original decision is still present.

The training cohort is the deterministic minimum of 300 states across 182 questions. It contains 141 supported answers, 148 non-empty evidence searches, and 11 empty-evidence searches. Its risk counts are relation 244, time 128, scope 55, and measurement 30.

The internal cohort is the deterministic minimum of 100 states and covers all 50 questions in `internal-freeze-v3.json`. It contains 44 supported answers, 45 non-empty evidence searches, and 11 empty-evidence searches. Its risk counts are relation 84, time 28, scope 23, and measurement 9. The normalized-question overlap between train and internal candidates is zero, and the selected source IDs are disjoint.

The v3 freeze retained the prior Roosevelt Park replacement decision and replaced only three questions that both question-quality reviewers identified as intrinsically ambiguous or unsupported by their source context: Kokomo/WWI prosperity, the Cowboy Junkies versus Matt and Kim “more well-known” comparison, and Mac and Me/Caspary identity attribution. The first three mutually usable proposals in the pre-fixed blinded order became the replacements. Their question IDs are `5abe235655429976d4830a7c`, `5adff2b455429925eb1afb6f`, and `5ae7eed85542993210983fc4`. Both reviewers approved at least one valid state for each, giving 50/50 question coverage. Earlier freezes remain unchanged.

For the 61-state measurement supplement, the two reviewers produced 44 and 38 raw approvals; the strict join retained 17 mutually approved, citation-valid states, with 14 rejected by both, 11 disputed, and 19 quarantined. For the 18 internal replacement states, they produced 11 and 8 raw approvals; the join retained 3 mutually approved, citation-valid states, with 5 rejected by both, 5 disputed, and 5 quarantined. Risk labels and search missing-fact statements were added in separate, packet-hash-bound files. The originals were not rewritten.

The reviewers used slightly different field names for their supplemental records. A deterministic adapter maps those schemas into the strict validation schema, derives citation stages only from packet document IDs, verifies supplied titles and exact quote substrings, and requires hash-bound risk and search-gap addenda. The normalized results and original audit records are both retained. Invalid citations, approvals without a new top-three citation, and approvals without an explicit search gap remain quarantined.

Two question-only auditors examined four challenged questions and agreed that the Kokomo and Mac and Me source bundles are defective, the festival comparison is intrinsically ambiguous, and Battlefield is answerable from the supplied context. One reviewer reported that their parquet loader materialized an answer field with the row; they state they did not inspect or use it. That handling limitation is recorded and was not concealed.

## Verification and handoff

`reports/v114-review-gate.json` records `passed=true`, no gate errors, complete dual-review coverage, the action and risk counts, and the unchanged no-access flags. The fixed thresholds all pass: training 300–400 states and at least 150 questions; internal at least 100 states and all 50 frozen questions; answer and non-empty search each at least 30%; empty search at most 20%; each of relation, time, scope, and measurement at least 20 training and 5 internal states.

The SFT exports contain only visible messages, the approved assistant target, its index, and the thinking flag. `scripts/verify_v114_sft_exports.py` reconstructs the deterministic cohorts from the approved audit pool and verifies exact export equality, token preflight limits (2,048 input and 128 target tokens), unique states, question isolation, and absence of private/review fields. The hash manifest is `reports/v114-sft-export-manifest.json`.

The dataset gate now permits a future SFT experiment, but this task did not start it. Independent 100-question validation remains the next evaluation stage after a separately authorized, controlled SFT run; GRPO remains out of scope until that validation demonstrates a gain.
