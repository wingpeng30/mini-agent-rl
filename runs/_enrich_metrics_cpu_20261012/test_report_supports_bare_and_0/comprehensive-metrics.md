# Comprehensive metrics from existing ledgers

仅复用现有答案别名、trace 和既有来源审计；无模型调用、无新人工或语义审查。未知来源题的证据指标保留 NA。

配对 bootstrap 复用 [summary.json](summary.json) 与 [REPORT.md](REPORT.md)，本脚本不重复抽样。完整逐题及分布见 [comprehensive-metrics.json](comprehensive-metrics.json)。

| Branch | n / expected | answered coverage | accuracy among answered | normalized EM | best-alias P / R / F1 | wrong / invalid / repeated / budget |
|---|---:|---:|---:|---:|---|---|
| am-serial | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |
| am-batch | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |
| sft42 | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |
| grpo42 | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |
| grpo43 | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |
| base-untrained | 1 / 1 | 1.0 | 1.0 | 1.0 | 1.0 / 1.0 / 1.0 | 0 / 0 / 0 / 0 |

## Sources and final input visibility

Doc recall uses existing required document IDs. Quote recall is exact substring coverage of existing quoted spans, not a new semantic correctness judgment. Last input excludes tool results returned by the final model call. Macro recall denominators include only confirmed source annotations, and last-visible metrics additionally require saved input metadata.

| Branch | confirmed / unknown | retrieved doc recall | all required docs | last-visible doc recall | all quotes last visible |
|---|---:|---:|---:|---:|---:|
| am-serial | 0 / 1 | NA | NA | NA | NA |
| am-batch | 0 / 1 | NA | NA | NA | NA |
| sft42 | 0 / 1 | NA | NA | NA | NA |
| grpo42 | 0 / 1 | NA | NA | NA | NA |
| grpo43 | 0 / 1 | NA | NA | NA | NA |
| base-untrained | 0 / 1 | NA | NA | NA | NA |

## am-serial: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": 2.5, "model_calls_per_second": 0.5, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## am-batch: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": 2.5, "model_calls_per_second": 0.5, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## sft42: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": 2.5, "model_calls_per_second": 0.5, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## grpo42: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "latest_process", "resumed": true, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": null, "model_calls_per_second": null, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## grpo43: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": 2.5, "model_calls_per_second": 0.5, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## base-untrained: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 1 / 0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| per_case.searches | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_case.input_tokens | 1 / 0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| per_case.output_tokens | 1 / 0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| per_case.model_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.tool_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.elapsed_seconds | 0 / 1 | NA | NA | NA | NA | NA | NA | NA |
| per_case.unique_documents | 1 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| per_action.input_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.output_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.json_action_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.raw_action_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.model_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.query_chars | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.left_padding_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |
| per_action.dropped_observation_tokens | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| development | 1 | 1.0 | 1.0 | 0 | NA |

Termination reasons: `{"answer": 1}`.
Search/query diagnostics: `{"query_count": 0, "duplicate_queries": 0, "duplicate_query_fraction": null, "document_results": 0, "identified_document_results": 0, "duplicate_document_results": 0, "duplicate_document_fraction": null, "successful_searches": 0, "search_ok_measurable": 0, "search_success_rate": null, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 2.0, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 1, "questions_per_second": 0.5, "output_tokens_per_second": 2.5, "model_calls_per_second": 0.5, "model_loading_seconds": null, "gpu_memory": null, "gpu_hours": 0.0005555555555555556, "monetary_cost": null}`.

## Training

| Branch | groups | mixed / all correct / all wrong | effective groups | unique updates | training seconds | completed generation seconds |
|---|---:|---|---:|---:|---:|---:|

## Measurement boundaries

- Best-alias precision/recall use one existing reference chosen by highest normalized token F1; existing normalization removes English articles, punctuation and extra spaces.
- Required source coverage is measured only on previously confirmed source annotations; unresolved/missing cases are NA, including the existing 35 unresolved cases.
- Document ID retrieval/visibility and exact quoted-span coverage do not certify answer entailment or relation correctness. No new semantic success rate is created.
- Last-visible evidence uses explicit document_changes segments or certified legacy untruncated saved-input metadata; the current call's returned tool observation is excluded.
- Per-row model_seconds in GPU batches are allocated batch time / active rows. batch_seconds repeats per row and its sum is not GPU wall time; generation-attempt ledger completed times are physical batch wall times.
- Per-case elapsed time overlaps across wavefront rows; use saved evaluation_wall_seconds for throughput. Resumed evaluation summaries describe the latest process and cannot certify all-process wall time.
- Training ledgers retain physical attempts, including recovery replay. Last group record per index summarizes logical groups; summary.json defines final checkpoint state. Update interval includes rollout/scoring overhead, not pure optimizer latency.
- Outcome entropy describes the binary group reward distribution; token policy entropy is NA because full probability distributions were not logged. Ratio percentiles summarize recorded per-update summaries, not a reconstructed global token distribution.
- Missing measurements are null/NA; monetary cost, energy, GPU utilization, and new semantic correctness are unavailable. No estimates from hardware pricing or extra detectors are used.

## Warnings

- Existing source audit unavailable; source coverage remains NA (no new review).
