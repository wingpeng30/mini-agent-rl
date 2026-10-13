# Goal AR: constrained evidence selection

Research parent: AM seed42-A, adapter SHA-256 3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c. Published Goal Z remains d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52. Both actual remote hashes verified before preparation.

A: direct-v090, legacy-json, first-json-v1, no grammar. Reuse the 32 AP v2 AM A probes only after exact production input reconstruction and default-path compatibility checks.

C: direct-v090, evidence-addresses-v1, first-json-v1, json-action-v1 grammar. B: same original history and evidence as C, direct-v090-evidence-address-v1, evidence-action-v1 grammar. The B prompt is registered verbatim in prompts.py and hashed before any evaluation.

Document boundaries use deterministic splits at newlines and punctuation followed by whitespace and an uppercase character. All original characters and metadata, ranks and duplicates are retained. Address format: m<message-index>-r<rank>-s<span-number>; mapping stores document identity, body hash and original character span. No answer-based filtering or highlighting. B may select zero to four visible addresses before its action; empty evidence is allowed. Grammar validity is not semantic correctness. No forced answer, rewritten action, retry, second JSON selection, or generated-evidence history reinjection.

Common: remote AutoDL .venv/bin/python, one RTX4090D, BF16, no quantization, greedy, thinking=false, 16384 input / 2048 output, legacy budget, BM25-v2/top-3, Goal Q 4 calls / 3 searches. Zero training and new confirmation. Defaults remain unchanged. Constrained inference must not report unmasked logprobs as its behaviour probability.

Dependencies: xgrammar==0.2.8, apache-tvm-ffi==0.1.14.post1. Preserve wheels and hashes; do not upgrade Torch/Transformers. CPU checks compile all grammars and mechanically verify complete original bodies, exact A input IDs, C/B same evidence/history, token maps, no clipping, default path compatibility, evaluation isolation, address restrictions and one-JSON parsing.

Stages: prepare; engineer (first two complete and first two partial probes, C/B once each, 8 GPU calls, zero search); probe C then B (32 each, zero search); one-pass main-agent adjudication; gate; eligible real-search arms A then C then B, 64 each. Maximum 840 new GPU calls, 576 searches, 192 real-search trajectories. No quota filling. GPU/numeric/identity/infrastructure failure preserves scene and stops; semantic failure is scored without rerun.

Probe promotion: complete >=22/24, >=2 net improvement over A, at least one new entity/role/relation repair, at most one lost A success; partial legal bridge search >=7/8 and >=A, premature answers/invalid/repeated searches do not increase. B also requires >=2 complete net improvement over C and >=2 gains with independently verified correct selected evidence. A/C vs C/B comparisons have different purposes; do not claim every difference is due to evidence selection.

Real-search promotion: support success net >=2/29 against current parent; fact-error/unsupported total and invalid/repeated/out-of-budget actions do not increase; preserve Rapana and duke; lose at most one of the seven frozen continuous-search protections. Every gain uses exact actual-input relationship evidence. C may be promoted alone; B must independently beat C. Unconfirmed records cannot satisfy thresholds. Reuse 29-case existing source admission; other 35 remain unconfirmed. No external facts or source audit expansion.

Execution entry: .venv/bin/python scripts/run_v133_goal_ar.py <prepare|engineer|probe|gate|evaluate> [--arm A|C|B]. All later stages verify the frozen manifest. Human semantic review lives in probe-adjudications.jsonl, keyed by arm+probe_id; summary does not infer correctness from gold text or case ordinal.

Subsequent sequence: AS existing failure-recovery inventory gate (known 1/24 and 10 pairs are insufficient; skip without expanding); AT 16 training-only states x4 bounded sampling, matched positive-SFT vs group-relative updates only if >=4 independent mixed groups and reliable rewards; AU fixed-corpus BM25 top20 cross-encoder ranking to top3; AV accepted cumulative version regression. Only >=2 real-search support gain and protection passing promotes research parent; +1 archives. Final model publication and unseen confirmation are separate. No automatic rollout of a failed variant.
