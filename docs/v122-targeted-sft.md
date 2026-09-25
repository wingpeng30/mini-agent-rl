# v122 targeted SFT: sequential decision and data gate

The v118 adapter remained fixed while the v122 controller comparison ran.
Protocol preflight passed 50/50 in each arm. On the reused 50-question
fixed-first-search development set, A/B/C/D obtained 29/24/31/25 exact
answers. C gained three A errors and lost one answer, so its **net gain of two**
did not meet the registered threshold of three. It also introduced a confirmed
entity-chain regression on Wormsley Park. B/D are further confounded by a
surface-form query gate that rejects a grounded `3OH!3` query. No arm was
selected for full end-to-end evaluation; these are not independent results.

The next experiment tests one targeted SFT only if both new data sets pass
independent evidence review. Candidate actions are formed from official
HotpotQA training sources, never from the model's erroneous output. Search
states must have a missing relation, a query grounded in the pre-action input,
and a top-three result that adds the missing fact. Answer states must have
complete support in the actual truncated input. Both reviewers must approve
the same final state and exact quotes; a disagreement or invalid quote removes
the state. Questions are isolated by ID and normalized text from historical
training, diagnostics, and evaluation.

The target training export is 395 states across 211 questions: the frozen v117
345 states plus 25 newly reviewed search/answer pairs, including at least 20
additional relation-risk states. A new 50-question, 100-state internal set is
selected only from a fixed, previously unexposed 200-question candidate pool
and a fixed 1,991-document corpus. No model score is used to pick those
questions. The new internal set is not used in gradient updates.

The code path is `scripts/finalize_v122_internal.py`,
`scripts/finalize_v122_train_supplement.py`, `scripts/audit_v122_tokens.py`,
`scripts/preflight_v122.py`, `scripts/train_v122.py`, and
`scripts/evaluate_v122_internal.py`. The frozen configuration is
`configs/v122-targeted-sft.json`. The remote preflight checks source/reference
adapter hashes, BF16 GPU, disk, frozen data, and per-record equality of token
IDs and target masks across local and remote. Only after that may a fresh v101
copy train for five smoke updates; a separate fresh v101 copy then trains for
exactly 50 updates with the last checkpoint. No v118 continuation, parameter
search, or validation-driven checkpoint choice is permitted.

The new internal gate compares v101, v118, and the new adapter under the same
50 questions, corpus, retriever, and decoding configuration. Required results
are protocol at least 98%, at least 45/50 supported answers and 45/50
insufficient-evidence searches, v101 correct-answer retention at least 95%,
at least three net additional exact answers, nondecreasing F1, and no
unresolved support dispute or confirmed semantic degradation. Any failed gate
ends this run. The independent 100 questions remain sealed until all internal
criteria pass; `final-test` is never opened. The v111 78% score remains a
training-side v101 plus external-API mixed workflow result.

The v122 run stopped at the internal data gate. The first 12 fixed-order
batches yielded 45 complete dual-approved questions. Exhaustive review of
the other auditor's full 326-state record proves the frozen pool can provide
at most four complete-pair scope-risk states, below the required five. The
precise certificate is `reports/v122-internal-scope-impossibility.json`.
The new internal set and supplemental train set were not frozen, and the
remote GPU was not used for SFT. The frozen training configuration and
runner remain available for a future, genuinely new data freeze; this run
does not treat them as a passed preflight.
