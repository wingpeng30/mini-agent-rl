# v106 error calibration, few-shot screen, and oracle-evidence diagnosis

## Decision

Keep v101 as the current model. Reject the tested three-example prompt. The
next model experiment should target evidence-grounded entity and relation
selection, while retrieval remains a secondary source of error.

## Semantic calibration

All 17 preselected v101 seed42 answer-boundary candidates were reviewed by an
independent `deepseek-flash` call at temperature zero. The reviewer received
the question, reference answer, candidate answer, and only the evidence that
the policy had actually seen.

| verdict | count |
|---|---:|
| equivalent | 8 |
| acceptable short form | 6 |
| incomplete | 2 |
| wrong | 1 |

Counting the 14 accepted answers gives a preliminary audit-adjusted estimate of
74% rather than the 60% strict EM. This estimate is not a calibrated accuracy:
only 17 preselected boundary candidates were audited, the reviewer saw at most
the first 1,200 characters of reconstructed evidence, and one accepted
classification ("popular science magazine" for "magazine") is debatable. It
does not replace strict EM or token F1 and is not a final-test result.

## Fixed three-example prompt

The prompt used three compact demonstrations selected only from the training
split. They covered comparison/entity selection, full date precision, and an
alias requested by a relation question. No development answer or support label
was included. No prompt truncation, lossy evidence trimming, or context-budget
error occurred in this run. The prompt reduced mean searches from 1.05 to 0.60,
so the observed regression is associated with changed search behavior; it
cannot be attributed to the examples physically displacing evidence.

On the frozen first 20 development questions, v101 seed42 changed as follows:

| metric | direct-v090 | fewshot-v106 | delta |
|---|---:|---:|---:|
| strict EM | 70% | 55% | -15 pp |
| token F1 | 85.83% | 65.83% | -20 pp |

This prompt is rejected and was not expanded to 100 questions. The result does
not prove that every possible demonstration is harmful, but it shows that
adding generic demonstrations to this 0.8B policy is not a promising default.

## Complete-evidence diagnostic

For the same 100 development questions, the model received the original SFT
search actions and tool documents with the final answer removed. All 100 cases
contained both support documents. Gold answers and support-title labels stayed
outside the model messages.

| metric | autonomous retrieval | complete evidence | paired delta |
|---|---:|---:|---:|
| strict EM | 60% | 67% | +7 pp, 95% CI [0,+14] |
| token F1 | 74.77% | 83.04% | +8.27 pp, 95% CI [+1.88,+15.25] |

The paired transitions were 10 wrong-to-correct, 3 correct-to-wrong, 57
correct-to-correct, and 30 wrong-to-wrong. Complete evidence therefore provides
a real benefit, especially in partial-credit answer quality, but it does not
remove the dominant failure mode. This is an evidence-condition comparison,
not a pure retrieval-recall intervention: the complete-evidence arm also uses
teacher search history and a different distractor mix. It establishes that the
model can benefit from a cleaner complete evidence state, but does not assign
the whole delta causally to retrieval. Most remaining errors occur after both
support documents are already visible.

## Next priority

The highest-value next experiment is a small evidence-grounded answer selector
or verifier. It should receive the question, existing visible evidence, and the
candidate answer, then either preserve the answer or emit one corrected JSON
answer. Its training examples must come from the training split and should
include explicit hard alternatives for entity, relation, date, and answer
boundary mistakes. A fixed 20-question screen should gate a 100-question
evaluation. The main safety metric is correct-to-wrong migration, because v106
shows that an intervention can improve some cases while damaging already
correct answers.

The project should continue reporting strict EM and F1 together with a frozen
semantic audit. Development-set semantic adjudications must never enter model
inputs, training rewards, or final-test decisions.
