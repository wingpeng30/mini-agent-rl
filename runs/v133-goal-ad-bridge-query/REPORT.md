# Goal AD — Bridge query training

Status: `complete_diagnostic_gate_failed`

Published: `False`
Selected adapter: `/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final`

The experiment uses the Goal Z trajectory seed 42 checkpoint and the frozen Goal Q search path.
It only adds search-action supervision at audited failure states; it does not supervise the corrected answer.

## Diagnostic gate

```json
{
  "diagnosed_failures": 34,
  "passed": false,
  "path_review": {
    "ambiguous": 1,
    "cases": 48,
    "supported": 14,
    "unsupported": 33
  },
  "query_remediable_cases": 0
}
```

## AD0 review and baseline

The frozen baseline is **36/64** semantically sufficient successes on Goal Z development, with **19** combined factual/unsupported errors and **0** invalid actions. The reused comparison scores are 15/16 main, 12/16 regression, Goal Y 40/48, and AA regression 32/64.

The AB audit packet already stored each token-decoded actual prompt and verified visible document IDs against that prompt; it did not require the complete source document to be present. No old AB anonymous review needed re-review, and its original outputs were reused.

The frozen confirmation file received one schema-only deserialization before preparation; only its record count and field names were emitted. No question or answer values were used, and no confirmation outputs were generated. This protocol deviation is preserved in the config.

Details: [AD0 review impact](AD0-review-impact.json).

