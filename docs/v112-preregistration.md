# v112 evidence-constrained SFT

## User-authorized budget amendment

After the first bounded audit batch (255 states, 96 settled calls, new11.559504 CNY),
the user explicitly instructed: "取消预算限制，继续任务". `configs/v112-budget-policy.json`
therefore supersedes the monetary caps below, including the original60 CNY ceiling.
Keep all prior ledger rows and account every new call; unknown outcomes still stop
dispatch. Data, isolation, evidence and downstream evaluation gates are unchanged.
Work proceeds in bounded state batches and stops once qualified material is sufficient.

Pre-training compatibility correction: existing local SearchTool returns top3.
Candidate audit packets retain top5 diagnostic retrieval for provenance, but final
acceptance requires every future-evidence quote to be within actual top3 (current
evidence quotes remain allowed). Search usefulness evaluation also uses top3.
The local runtime/baseline protocol is unchanged. Uniform initial-question searches
were added for all fixed50 internal questions; only those needed to represent a
question lacking accepted nonempty states are selected, under the original20% cap.

Approved objective: improve local-agent accuracy by answering only with sufficient
visible evidence and searching for an identifiable missing fact otherwise.
v111's 78% is a training-side API hybrid result, not local-model accuracy.

## Input replay incident (before any internal score was accepted)

The first internal state run stopped on a token replay mismatch. Frozen original
rows contain a single-batch token array, while inference returns a flat array.
The second attempt accepts only a strict flat or single-batch array of nonnegative
integer token IDs, normalizes its shape, and still compares every token. Frozen
data, targets, weights and acceptance thresholds were not changed. The failed
attempt and its log remain preserved; replay2 uses separate output paths.

The same batching issue made the original builder's `prompt_tokens` metadata and
its preliminary input-length check inaccurate. An additional full read-only audit
on both local and remote processors verified all 513 selected states: saved,
direct-rendered and actual runtime-budgeted tokens are identical after shape
normalization; maximum input lengths are 787 for train and 794 for internal.
No selected state needed runtime truncation. The training launch independently
checked the actual flattened prompt length, full sequence length and target mask.
Original erroneous metadata is retained for provenance and is not used as an
actual length measurement. This audit establishes input consistency, not semantic
correctness or accuracy benefit.

## Frozen data policy

Use only the existing 600-question training source. Freeze 50 internal questions
from the 104 outside v090 exported train/validation and v111 diagnostic IDs using
SHA256(`v112-internal:` + task ID). No claim of historical novelty. All normalized
question groups and their variants remain in one split. v111's 100 are diagnostic
only. The corpus is training-side background, not document-disjoint evaluation.

Original trajectory actions and removed-document variants are unapproved
candidates, including possible oracle-title search queries. Re-encode actual
visible prompts using direct-v090, compact-v3, 2048 input tokens. Audit AFTER
compaction. Targets must fit 128 supervised tokens; full sequence at most 2176.
Do not train rejected answers as negative targets. Gold stays in metadata only.

API blind review and separate astra-low task-agent review must both approve exact
question/entity/relation/object/unit/time/scope support. Search needs a concrete
gap, a query justified by current evidence/question, no repetition, and retrieved
new evidence. Both reviews retain verbatim citations. Disagreements and uncertain
states are quarantined, not rewritten toward gold. These are auxiliary audits,
not human or independent ground truth. Risk counts use the reviewers' intersection.

Training gate: 300–400 unique states / at least 150 unique questions; internal
gate: at least 100 unique states covering all fixed 50 questions. Each split has
at least 30% answer and 30% nonempty-evidence search, at most 20% empty-evidence
initial search. Relation, measurement, time, scope each require 20 training and
5 internal states. No relaxation after inspection. Missing coverage leads to
additional training-source construction within budget, or no training.

## Budget and model protection

Authoritative ledger: local `runs/v110-api.sqlite`; never initialize a replacement
empty ledger. Opening 921 settled calls / 27.600696 CNY; maximum new API spend
12 CNY, cumulative cap 39.600696. Reserve maximal 8/24 CNY per million input/output
tokens before dispatch. Unknown calls stop dispatch and retain reservations.
Prior call rows must stay unchanged. Remaining conservative budget is not an
account balance. AutoDL remains running; no final-test access at any stage.

Frozen source: v101-answer50-seed42, weights SHA256
`3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768`.
New independent smoke and formal copies; source/output overlap forbidden. Recheck
the actual data gate, hashes and token masks before training. Five-step smoke,
then restart formal from source: BF16, no thinking/checkpointing, LR5e-6, batch1,
accumulation8, seed42, one epoch and at most 50 updates. Use final checkpoint only.
Loss/gradients must be finite, output must change, source must remain identical,
and saved adapter must reload. No default model replacement.

## Downstream gates

Internal: protocol >=98%, supported-answer correctness >=90%, insufficient-evidence
search >=90%, baseline-correct preservation >=95%; full 50-question normalized EM
net gain >=3, F1 nondecrease, zero unresolved support disputes or confirmed semantic
regressions. Loss cannot clear behavioral gates.

Only after all internal gates freeze the entire run and evaluate the untouched
independent100 paired local v101/new-adapter comparison on the existing frozen
986-document environment. First20 is operational-only, no score-based stopping or
tuning. Independent gates: normalized EM net >=3, F1 nondecrease, preservation
>=95%, protocol >=98%, no unresolved support disputes/confirmed semantic regressions.
Question-paired bootstrap seed42/10000 EM interval must have positive lower bound
before proposing GRPO. Failure does not authorize holdout tuning. GRPO is a separate
later plan, not automatic execution in this run.
