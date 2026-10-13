# v109 hybrid answer generation and evidence selection

## v110 audit correction (2026-09-22)

The original outcome below is exploratory and does not constitute a fully frozen,
independently replicated pipeline. v110's official HotpotQA rescore retains strict
EM60→65 and normalized EM63→69, but corrects hybrid F1 from82.53% to82.33%
(baseline74.77%). Direct-answer F1 is67.32%, not67.71%. All100 outcomes and
explicit retry precedence were checked; failures were retained in the audit.

The v109 recommended one-pass configuration was not consumed by its runner,
which still ran two views and required a separate retry script. Its98% candidate
schema rate is not an end-to-end supported-answer rate. The substring boundary
guard could suppress real entity/date corrections. v110 removes that rule and
performs independent evidence review before treating replacement as supported.

v110 explicitly verified actual/replayed prompt IDs, masks, saved prompt hashes
and retained document provenance for both100-question databases. E0=E1 shows
that compaction did not change those extracted documents; it does not establish
retrieval sufficiency. The previous reason for skipping supplemental retrieval
was therefore not adequate as a causal conclusion.

See `../reports/v110-legacy-rescore.json` and `v110-preregistration.md`. Original
v109 reports remain unchanged for traceability. Historical final-set files are
not read by v110; prior resealing does not imply they were never evaluated.

## Result

Keep the hybrid mechanism as an experimental, default-off accuracy layer. On
the fixed 100-question development set it raised strict EM from 60% to 65%,
normalized EM from 63% to 69%, and token F1 from 74.77% to 82.53%. The paired
95% interval was 0–10 points for strict EM, 1–12 points for normalized EM, and
3.01–13.11 points for F1. Thus normalized EM and F1 have positive paired
development evidence; strict EM improved but its interval touches zero.

The one-shot strong-model answer baseline was worse: 50% strict EM, 54%
normalized EM, and 67.71% F1. The gain therefore came from generating multiple
evidence-grounded candidates, selecting among them, and preserving concise
answer boundaries rather than replacing the local Agent with one direct API
answer.

## Input and mechanism audit

Both the training-side and development-side databases achieved exact token and
attention-mask replay for 100/100 prompts. Strict and normalized EM are now
reported separately. The original training-side result is 65% strict and 68%
normalized EM, correcting v108's earlier mixed label.

E0 and E1 were identical on every task: these runs had no evidence trimming or
document deletion. The complete-document comparison therefore provided no new
condition and was not charged twice. Supplemental retrieval was not activated.

The second adversarial candidate view consistently returned a different field
shape and contributed no accepted candidate. It is removed from the recommended
configuration. The first view was valid on 96/100 development tasks; one generic
retry recovered two failures, reaching the pre-registered 98% protocol gate.
The remaining two failures preserved the original answer. Final answer
completion was 100/100.

The fixed boundary guard first maps explanatory selections back to the shortest
matching candidate, then preserves the original when the change is only an
alias or containment expansion. This eliminated training-side answer-boundary
regressions without using gold labels at inference.

## Error movement and review

Strict scoring found six wrong-to-right changes and one right-to-wrong change.
The apparent regression was `U.S. Route 51` to `US 51`; the blinded auxiliary
evidence review marked both as correct aliases. Across all 14 changed answers,
the same-model auxiliary review marked 13 as baseline-wrong/hybrid-correct and
one as both correct, with no semantic regression. This review is supportive,
not independent human evidence.

Candidate sets contained an exact gold answer for 18 of 40 strict-wrong cases
and a normalized match for 23 of 40. Candidate best-achievable F1 across all
questions was 90.35%, leaving further headroom in candidate recall and final
selection.

## Cost and recommendation

The run used `deepseek-flash`, which the model endpoint exposed alongside
`deepseek-v4-pro`. The current official catalog identifies it as
DeepSeek-V4.1-Flash. All calls, including truncated attempts and retries, are
recorded. Using deliberately conservative ceilings of 8 CNY per million input
tokens and 24 CNY per million output tokens, total estimated cost was 28.08128
CNY, below the authorized 30 CNY. These assumptions exceed the official Flash
peak token prices at execution time, so this is an upper bound rather than the
provider's exact account deduction.

The hybrid layer is suitable for another independent evaluation, but should
remain disabled by default because this development set has informed several
iterations and the final-test remains sealed. The next high-value step is an
independent held-out validation of the frozen one-pass mechanism. If it
replicates the gain, distill its candidate generation and selection traces into
a local model.
