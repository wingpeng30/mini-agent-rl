# v133 Goal D result

**Decision: stop this SFT recipe; the extra two passes produced no measurable improvement.** The incremental run completed successfully, changed the adapter weights, and finished a context-aligned paired evaluation. All 16 v118 outputs and all 16 Goal D outputs were protocol-valid, but the two arms produced exactly the same raw answer on every case.

## Why Goal D was run

The original one-pass adapter was not a no-op: compared with v118, 3,194,829 of 3,194,880 scalar weights changed, with relative L2 delta 0.0010614. Its seven-step run had one zero-learning-rate warmup step, leaving six optimizer updates. All 25 approved targets were literally supported in their visible evidence. This justified a single dosage check using those same examples; it did not justify adding data or training on dev labels.

The corrected Goal C evaluation showed why the first run emitted all `search` actions: training rows had two assistant-search/tool-response turns before the answer, while its original evaluation rows had none. Under the deterministic aligned diagnostic, v118 already scored 12/16 EM and the one-pass adapter matched every output.

## Execution and result

The existing SFT adapter was used as the initialization. It received 14 more optimizer steps over two passes of the same 25 training rows, with seed 42, batch size 1, gradient accumulation 4, BF16, LR 5e-6, one warmup step, maximum length 2176, and no gradient checkpointing. It trained for 23.11 seconds; the two-step estimate was 41.49 seconds, under the 60-minute cap. Peak GPU memory was 12.185 GiB. The paired evaluation took 23.048 seconds; its first-two-output estimate was 27.383 seconds, also under the 60-minute cap.

| Measure | v118 | Goal D cumulative adapter |
|---|---:|---:|
| Valid answer actions | 16/16 | 16/16 |
| Strict normalized EM | 12/16 | 12/16 |
| Mean token F1 | 0.8417 | 0.8417 |
| Newly correct |  | 0 |
| Lost correct |  | 0 |
| Retained correct |  | 12 |

All 16 raw output pairs were identical. The adapter nevertheless changed relative to its parent: 3,194,850 of 3,194,880 scalar weights changed, max absolute delta 3.4896e-5, relative L2 delta 0.0015929. The training run and saved checkpoint therefore both occurred as configured; the lack of gain is not a missing-run or unchanged-weight artifact.

The four strict EM errors remained the same as the prior aligned diagnostic: case 0 answered the league instead of the team; case 2 answered “folk rock” rather than the requested shared “rock” category; case 3 reversed the two formation dates; case 15 used “Mary McDonnell” while the gold string includes the middle name. No answer changed, so the conditional blind review gate did not trigger.

## Interpretation and limits

The context mismatch explains the original 0/16 protocol result. With the corrected synthetic history, both adapters answer coherently and v118 is already correct on 12/16 cases. Continuing the existing SFT on the same 25 examples for two more passes changed millions of weights but changed no greedy answer on this fixed 16-case set. The current data and SFT recipe have shown no evaluation gain; further repetition of this recipe is not supported by these results.

This remains a small, train-derived development diagnostic with a synthetic search history. It is not an end-to-end search result or an independent generalization estimate. The 25 training rows were not changed, the 16 dev labels were not used for training, and no new sources or review calls were used.

## Traceability

- Execution record: `docs/v133-goal-d-execution-record.md`.
- Incremental SFT config SHA-256: `dcb7886c8a92511cbab3405ee75412f3890105b43130f11fe65bbe0e411b638b`.
- Training runner SHA-256: `94389099b184af5e92dd55c9727841049d19fc8890b91b0a7a2443fec4999bee`.
- Paired evaluator SHA-256: `f492560fa4a6b0a126e4a386a91141c83d7c296aa083f282edad4eecf2b43f48`.
- Train SHA-256: `b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df`.
- Dev SHA-256: `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`.
- Parent adapter SHA-256: `0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6`.
- Goal D adapter SHA-256: `b8782a1a36ddaa44e2a2d7446763b7e50ae40865aae326366d3f0e50b9fc94af`.
- Training metrics SHA-256: `37e99c77970304d4f2486e83dece13231564a2fe96afbd132c499c82156f49dc`.
- Training-start record SHA-256: `a7059f0999a0af93ad7b5a012d6be886712537fc110bf9cb78213973972b9d34`.
- Paired output SHA-256: `448cbdee7b0561d51c35fae0cceb91f8cc0be7853d802e6a644cf2f5abff4920`.

Local and AutoDL hashes matched for the result, adapter, metrics, and training-start record. The first post-training CPU-preflight call stopped on a guard that incorrectly expected the new adapter not to exist; it loaded no weights and generated no outputs. That guard was removed, the evaluator was re-hashed, CPU preflight passed, and only then did the paired inference run.
