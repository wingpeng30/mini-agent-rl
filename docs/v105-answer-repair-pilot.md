# v105 answer-repair SFT pilot

## Decision

Stop this branch and keep `v101-answer50-seed42` as the seed-42 model. Do not
replicate v105 to seeds 43 and 44.

The 50-step final-answer-only continuation was numerically healthy, but it did
not improve the fixed 100-question development evaluation:

| metric | v101 seed42 | v105 | delta |
|---|---:|---:|---:|
| strict EM | 60% | 59% | -1 pp |
| normalized EM | 63% | 63% | 0 pp |
| token F1 | 75% | 74% | -1 pp |
| invalid actions | 0% | 0% | 0 pp |

The paired transitions were 2 wrong-to-correct and 3 correct-to-wrong. The
paired 95% intervals were `[-6,+3]` pp for strict EM and approximately
`[-4.48,+3.53]` pp for token F1. This is neither a useful point estimate nor
evidence of improvement.

## What the pilot taught us

The failure is informative. Merely increasing supervision on final answers
does not solve the dominant error modes. Of the 40 v101 seed42 strict errors:

- 17 were answer-boundary or answer-variant candidates;
- 17 were entity/relation selection errors even though the gold string was
  present in model-visible evidence;
- 5 stopped after only partial supporting evidence;
- 1 had both supporting titles but still failed inference.

Many strict failures are plausible aliases or boundary variants, for example
`Jungle Book` vs `The Jungle Book`, `Colin Hay` vs `Colin James Hay`, and
`American Chemical Society` vs `the American Chemical Society`. Strict EM is
therefore useful as a reproducible conservative metric, but it is not a clean
estimate of semantic answer correctness.

## Highest-value next experiment

Before further model updates, perform a blinded semantic-equivalence audit of
the answer-boundary candidates and freeze an adjudication protocol. Then test
one evidence-grounded answer verification/refinement pass on a small fixed
development subset. The pass must see only the question and evidence already
visible to the agent, never the gold answer or supporting-title labels. It is
worth scaling only if it converts entity/boundary errors without increasing
correct-to-wrong transitions or latency beyond the predefined limit.

Further generic final-answer SFT, prompt wording variants, and removal of the
over-search penalty are rejected by the existing evidence and should not be
repeated unchanged.

## Reproducibility

The paired audit reads both SQLite databases in read-only mode and verifies
their SHA-256 hashes before and after analysis. The detailed result is in
`reports/v105-answer-repair-analysis.json`; the analysis entry point is
`scripts/analyze_v105_answer_repair.py`.
