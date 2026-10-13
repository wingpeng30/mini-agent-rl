# Goal AC — Failure-state repair result

## Decision

Goal AC stopped before training. The frozen evidence gate did not pass, so C42/C43 were not trained, the development candidates were not evaluated, and the sealed confirmation set was not accessed. The recommended product remains Goal Z full-trajectory SFT seed 42.

## Frozen starting point and checks

- Remote workspace: `/root/autodl-tmp/mini-agent-rl` on `connect.cqa1.seetacloud.com:25320`.
- Existing AB baseline was reused without regenerating model outputs. Its frozen development result was 36/64 semantic successes, 19 combined factual-error/unsupported answers, and 0 invalid actions. The other frozen baseline counts were 15, 12, 40, and 32 for old-main, old-regression, Goal Y, and AA-regression respectively.
- The confirmation set remained sealed (`confirmation_accessed: false`).
- The original 128 training traces had 52 previously identified stop-decision errors: 26 premature answers and 17 wrong-bridge searches, with 5 ambiguous and 4 lacking recoverable evidence. Audits covered all 52 stop-error traces.

## Material construction and gate

Exact replay of the original saved input tokens produced 31 original repair paths, including 20 paths targeting stop-decision errors. The remaining fixed-candidate shortfall triggered the already frozen supplemental pool: 128 candidates (64 MuSiQue and 64 2Wiki) were each evaluated once. Their conservative dual-review merge marked 28 as repairable and 6 as repairable stop errors; reviewer agreement was 78/128.

After real search actions were executed, 17 supplemental repair paths were built. Combined with the 31 original paths, 48 paths entered blind evidence review. Each of two reviewers independently checked the target answer against the final actual visible passages, and verified bridge entities against the question or evidence visible before each query. Reviewers cited direct support for 15 and 17 paths respectively. The frozen consensus and citation checks approved 14 paths and rejected 34.

The 14 approved pairs contained 14 search states and 14 answer states, but only 4 corrected a stop-decision error. The fixed training gate required 32–64 approved pairs, at least 16 stop-error corrections, and at least 32 examples of each state. It therefore failed on both pair count and stop-error coverage. No thresholds were relaxed.

## Implementation and audit record

The runner, supplement utilities, replay-fix utility, local tests, and review artifacts are preserved in this repository and the remote run directory. The replay utility restores the original tool-result field order before verifying exact prompt tokens. A pre-training loss aggregation issue was corrected and recorded as a code revision before any optimizer update or checkpoint was created.

The supplemental candidate rollouts were sampled before the exact-replay defect was identified. This sequencing deviation is recorded in `remote-final/state.json`; the fixed supplemental pool was not enlarged or rerun. One supplemental repair construction whose query executed but whose output could not be serialized is preserved in the attempt ledger and excluded without retry.

Detailed records are in `remote-final/`, `audit-a.jsonl`, `audit-b.jsonl`, `extra-audit-a.jsonl`, `extra-audit-b.jsonl`, `repair-evidence-a.jsonl`, and `repair-evidence-b.jsonl`. The remote originals remain under `runs/v133-goal-ac-failure-state-repair/` on AutoDL.

## Interpretation

The experiment provides no evidence about whether failure-state repair training improves search accuracy because no model was trained or evaluated. It does show that the fixed pool yielded too few repair demonstrations that both correct a stop error and have a directly supported answer with a verifiable bridge. The product checkpoint and confirmation data are unchanged.
