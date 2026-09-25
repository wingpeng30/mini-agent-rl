# v122 unified-action development experiment

## Frozen setting and protocol preflight

The v118 adapter remained fixed. All four arms emitted the existing SFT action
protocol (`{"action":"search","query":"..."}` or
`{"action":"answer","answer":"..."}`), without nested JSON in the answer.
The setting hash was
`956c11437814a10b9b4b24b45b373d71ac1672912db89941f212ce96894582ab`.
On 50 training-source states per arm (25 search, 25 answer), raw protocol
compliance was 50/50 for A, B, C, and D. The preflight was completed before
development evaluation.

## Fixed-first-search development comparison

All arms continued from the same first search on the same 50 repeatedly used
development questions and 1,986-document corpus. Scores below describe this
fixed-state diagnostic, not end-to-end accuracy or an independent test.
The v111 78% figure remains a training-side v101 plus external-API mixed
workflow result, not a local-model score.

| Arm | Policy | EM | Mean F1 | v118 correct retained | v120 correct retained | Protocol | Searches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | repeated-query repair | 29/50 | 0.6133 | 13/14 | 26/26 | 100% | 94 |
| B | A + query constraint Q | 24/50 | 0.5133 | 11/14 | 21/26 | 100% | 89 |
| C | A + stopping prompt S | 31/50 | 0.6533 | 14/14 | 25/26 | 100% | 98 |
| D | A + Q + S | 25/50 | 0.5333 | 11/14 | 21/26 | 100% | 92 |

C corrected three A errors and introduced one regression, giving a net gain of
two. It failed the registered requirement of at least three additional correct
answers over contemporaneous A. B and D lost accuracy. There is **no selected
arm**, so the programmatic gate blocked the full 50-question end-to-end run.
No independent 100-question or `final-test` set was opened.

The three C improvements were a show identification, Atlanta's 2016
population instead of the same-name Rathbun, Iowa population, and the
defending champion's nationality rather than the champion's name. Its one
regression changed the bridge entity for Wormsley Park from Mark Getty to
Leonard Ingrams; the latter appears in a distractor about an opera that moved
to Wormsley Park. This is an entity-chain error, not evidence that searching
longer is always beneficial. Evidence support for these changes remains
subject to the independent blind audit.

The current Q surface-form gate uses a capitalized-span regex. It rejects at
least one seemingly grounded query for `3OH!3` because the band name begins
with a numeral and contains punctuation. It also blocks some proposed
entities not found as exact surface forms in the question or visible excerpts.
Thus B/D failure mixes query policy, controller validation and retrieval; it
must not be attributed to model ability or SFT alone. Any repair to this gate
changes the setting hash and requires a new isolated evaluation. We will not
retune Q on these 50 development questions and then claim a clean result.

## Audit and P3 preparation

The offline DeepSeek audit received question, actual post-truncation evidence,
anonymous actions, and retrieval snippets, without gold, score, arm, or task
ID. It covered all 200 fixed-state traces. Two reviews outside C contain
illegal verbatim quotes and remain invalid. All 50 C reviews passed structural
and quote checks: DeepSeek marked 31 final answers supported, 15 unsupported,
two unanswered and two uncertain. One independent `gpt-6-luna` low agent
separately audited the 50 anonymous C traces, and all its quotes and coverage
validated. Both reviewers agreed on 30 supported, nine unsupported, and two
unanswered cases; nine others have a support disagreement or uncertainty.
Both rejected the Wormsley Park regression. One of the three score improvements
has a support disagreement. These remain unresolved. Neither incorrect
answers nor model-produced searches are promoted to SFT labels.

A new internal candidate pool was prepared from previously unexposed
HotpotQA distractor training questions, in fixed
`SHA256("v122-internal:" + question_id)` order. The exposure scan completed
without failures; 200 candidate questions were selected, their fixed corpus
contains 1,991 documents, and 163 questions produced 326 structurally valid
search/answer states. Candidate question IDs and normalized question text have
zero overlap with the historical exposure inventory. Token replay passed for
all 326 final inputs. The DeepSeek reviewer completed all 326 and approved 86
complete pairs on its own. These are **single-review candidates**, not a new
frozen 50-question internal set. They cannot authorize training until the
independent reviewer also approves the same inputs, quote binding and risk
coverage pass, and the training data gate is rerun. A separate fixed-order
training-side supplement candidate pool contains 120 relation questions, of
which 106 yielded 212 structurally valid paired states. It is isolated from
both the old training set and new internal pool, but likewise cannot be
used without dual approval.

The numerical result alone does not justify another SFT. The blind review
confirmed one semantic regression in C, while nine C cases still have a
support disagreement or uncertainty. The query constraint failure also
includes a controller defect. These conclusions motivate cleaner data and
controller design, but do not authorize training from disputed outputs.

## P3 data gate and stopping decision

The new internal pool was reviewed independently through a fixed 12-batch
prefix: 240/240 states have a saved `gpt-6-luna` low judgment, with all packet
hashes and exact quotes validated. DeepSeek reviewed all 326 states. At this
prefix, 45 questions have both search and answer states unanimously approved.
The target was 50 complete questions with each risk category represented by
at least five semantically confirmed states.

The scope category cannot reach five within the frozen candidate pool. An
exhaustive upper-bound check over all 326 DeepSeek votes, including the 86
complete pairs it approved on its own, found four scope states in complete
dual-approved pairs and **zero possible additional scope states** from
unreviewed pairs. One additional scope-marked state is orphaned because its
paired action did not pass double review. Thus reviewing the remaining states
cannot satisfy the category gate. The certificate is
`reports/v122-internal-scope-impossibility.json`; the local internal gate is
`reports/v122-internal-preflight.json`. No new internal 50-question set was
frozen, and the training-side supplement remains single-reviewed. The formal
SFT, smoke, new internal evaluation and independent 100-question evaluation
were not run.

A bounded feasibility scan found 81 previously unexposed training questions
whose two official support titles are already represented in the fixed
1,991-document corpus; only 16 produced structurally valid pairs under the
same retriever. That scan did not yield a reviewed scope replacement. Adding
new corpus documents now would change the audited search results and require
a new freeze and re-review; we did not relabel disputed examples to force
the quota. The current experiment ends at the registered data gate.
