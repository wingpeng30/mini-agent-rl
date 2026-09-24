# v108 evidence-constrained discriminative answer selector

## Decision

Stop before selector training. The frozen candidate generator provided no
valid selectable repair candidates, and the deduplicated v107 data also failed
the pre-registered internal-validation size gate. Training a classifier under
these conditions could only learn to preserve or abstain; it could not repair
the observed wrong answers.

## P0 audit

The v107 train split has 213 rows but only 158 independent states: 88 keep, 68
replace, and 2 insufficient. The validation split has 37 rows but only 23
independent states: 15 keep and 8 replace. The required validation thresholds
were 30 independent states with at least 10 keep and 10 replace. Both
supplemental replacement records were duplicates of existing validation
states. No additional DeepSeek calls were made in v108 because the remaining
v107 budget could not be reconstructed precisely and the plan forbade spending
when the balance was unverifiable.

All 100 frozen source trajectories had prompt token IDs and equally sized
attention masks. The installed processor rendered the same prefix when
`enable_thinking=False` was explicit or omitted; neither rendering contained a
thinking marker. The implementation now passes the flag explicitly to prevent
future model-template drift, but this was not the cause of v107's failure.

## Frozen candidate coverage

The v101 seed42 adapter was evaluated at temperature 0 with a 128-token limit.
Inputs contained only the question, actual model-visible evidence, and original
answer. Candidate strings were required to be unique contiguous evidence spans.

| metric | result |
|---|---:|
| source tasks / auditable inputs | 100 / 100 |
| infrastructure errors | 0 |
| strict-correct original answers | 68 |
| strict-wrong original answers | 32 |
| valid candidate-protocol outputs | 0 |
| accepted evidence-span alternatives | 0 |
| wrong answers with a strict-correct alternative | 0 / 32 |
| validation wrong answers covered | 0 / 5 |
| generation time | 121.32 s |
| peak allocated / device-process memory | 1.766 / 2.377 GiB |

The frozen agent continued to emit variants of its normal action protocol
(`answer`, `speak`, or `think`) instead of the candidate-list protocol. Strict
schema validation therefore rejects every output. The initial permissive
parser treated extra fields as an empty candidate list; this reporting defect
was fixed by forbidding extra schema fields and the saved raw outputs were
re-audited without another GPU run. Either interpretation yields zero accepted
alternatives and zero repair coverage.

## Interpretation

The demonstrated bottleneck was the frozen v101 adapter's inability to follow
the new candidate-list protocol. It was not evidence that correct candidates
could not be generated from the visible documents. v109 replaced that component
with an independent strong model and obtained substantial candidate recall and
a positive development-set accuracy result. Gold answers were not injected into
runtime candidates, final-test was not accessed, and v101 remains unchanged.

A future branch should not retry this prompt on the same policy. A defensible
next experiment is a separately trained evidence-span extractor (or a
deterministic entity/date span extractor) whose candidate recall is measured
before any ranking model is trained. The minimum useful gate remains at least
50% correct-candidate coverage among evidence-sufficient wrong-answer states.
