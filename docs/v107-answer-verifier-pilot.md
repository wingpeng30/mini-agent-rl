# v107 evidence-constrained answer verifier pilot

## Decision

Stop the verifier branch before development evaluation. None of the three
saved checkpoints passed the frozen internal eligibility gate. Keep v101
seed42 unchanged as the current agent.

## Data and training

The source was a new greedy run of frozen v101 seed42 on the first 100 training
questions. Each verifier input contained only the question, the documents in
the final transition's actual visible messages, and the candidate answer.
DeepSeek produced and independently reviewed keep/replace supervision under a
10 CNY ceiling. Actual usage was approximately 0.84 CNY plus 6,415 supplemental
review tokens.

After citation and second-review rejection, the dataset contained:

| split | distinct tasks | rows | keep | replace | insufficient |
|---|---:|---:|---:|---:|---:|
| train | 77 | 213 | 139 | 72 | 2 |
| internal validation | 15 | 37 | 27 | 10 | 0 |

Fifty-two candidates were rejected. The source SQLite hash was unchanged.
The verifier continued from a copy of v101 seed42 for 54 optimizer steps,
exactly two data epochs, at LR 1e-5. Training took 159.9 seconds; mean training
loss was 0.1406, final validation loss 0.0675, and peak allocated GPU memory was
10.937 GiB. All logged loss and gradient values were finite.

The source v101 adapter remained byte-identical at SHA-256
`3b759be69a0c082fd5274eb99b696bf6a60662b991c2234aac59d0646ff08768`.
The final verifier adapter had a distinct hash, confirming an independent
update rather than an overwrite.

## Frozen internal gate

| checkpoint | protocol legal | preserve correct candidate | repair wrong candidate |
|---|---:|---:|---:|
| step 25 | 94.59% | 100% | 0% |
| step 50 | 94.59% | 100% | 0% |
| step 54/final | 89.19% | 100% | 0% |

The required thresholds were 98%, 95%, and 20%. The model mostly emitted
`keep`; occasional `replace` outputs either preserved the wrong candidate or
failed the citation protocol. Lower validation loss therefore measured target
token prediction without producing usable correction behavior.

The fixed 20-question and 100-question development evaluations were not run,
because doing so after a failed internal gate would use development data to
iterate on an ineligible model. No development answer pairs were sent for the
authorized DeepSeek blind review because no development answers changed.

## Interpretation and next step

This result shows that this particular generative three-way verifier did not
pass its internal gate. It does **not** establish that a 0.8B model cannot
verify answers. A later v108 audit found substantial duplicated states: the
213/37 raw train/validation rows reduce to 158/23 independent states, and the
validation split contains only 8 independent replacement states. The two
supplemental replacement rows duplicated existing states. Consequently, the
original validation-data gate was not actually met and capacity-level claims
are unsupported. The installed processor's default template was byte-identical
to explicit `enable_thinking=False`, so template mismatch was ruled out for
this model. Repeating the same SFT remains unsupported because the evaluated
checkpoint failed the frozen behavior gate.

A future attempt should first test a simpler discriminative selector on the
same frozen data: score `keep` versus a small set of evidence spans, with the
span copied deterministically after selection. This removes free-form citation
generation and makes repair signal measurable. That is a new model interface
and requires a separate preregistration; it is not a continuation of v107.
