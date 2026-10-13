# v110 accuracy-first validation

Frozen before paid execution, 2026-09-22. API ceiling: 60 CNY conservative upper accounting (8 CNY/M input, 24 CNY/M output), one durable SQLite ledger. Training cumulative ceiling20, evaluation48, review60; unused balances flow forward. Price snapshot is in configs/v110-hybrid.json. Instance remains running; no final-test access.

P0: verify original/replayed tokens, masks, prompt hashes and retained document provenance on training100 and legacy development100; rescore old outcomes with official HotpotQA EM/F1. Report whitespace-trimmed case-sensitive exact separately. Preserve legacy files.

P1: fixed training100 frozen v101 answers/evidence, fixed200-document BM25-v2 corpus. A original; B one candidate call plus anonymous selector, no substring answer compression; C independent review of B/original, support required for changed answer, at most2 missing-information retrieval rounds, each2 distinct queries and top5 results. No extra calls if no new documents. Candidate generation at most1 retry. First20 candidate order reversal diagnostics. Inputs to API contain no gold or support labels. API model deepseek-flash thinking enabled, temperature0, output8192/retry16384. Failed answers fall back and stay in denominator. Infrastructure errors stop the stage.

P1 eligibility: official EM and F1 >= A; baseline-correct preservation>=95%; whole-path protocol>=98%; no unresolved confirmed semantic degradation. Rank eligible arms by official EM, F1, correct preservation, then fewer calls. Changed-answer blind evidence review precedes P2. This is training development, not generalization.

P2: independently frozen100 official Hotpot distractor validation questions, SHA256(v110:ID) selection, excluding known v041 train/validation/test IDs and normalized questions. Corpus retains all selected context documents including distractors; task batch does not change corpus. Do not read sealed final data. Fixed v101 seed42 BF16 baseline,2048/128,compact-v3,3 actions; accuracy layer may use longer context. Source is a different official split; title-disjointness from sealed data is not asserted.

Run A, B and unique selected arm on first20; stop candidate if official EM/F1 declines or confirmed semantic regressions>1. Otherwise extend unchanged to100. Retain if EM/F1 nondecrease and EM net>=3 or F1>=.02, semantic regression<=1/no unresolved disputes, protocol>=.98. Bootstrap seed42,10000 paired question resamples. Positive lower bound required for statistically supported gain. Same-model evidence review remains auxiliary, not independent truth. No automatic default replacement or distillation.

Output all artifacts, ledger, tests and SHA256-matched backup. If any gate fails, preserve evidence and stop the downstream branch rather than relaxing the gate. New independent corpus is a harder within-run comparison, not comparable directly to old support-only corpus scores.
