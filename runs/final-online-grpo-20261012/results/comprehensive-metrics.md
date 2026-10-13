# Comprehensive metrics from existing ledgers

仅复用现有答案别名、trace 和既有来源审计；无模型调用、无新人工或语义审查。未知来源题的证据指标保留 NA。

配对 bootstrap 复用 [summary.json](summary.json) 与 [REPORT.md](REPORT.md)，本脚本不重复抽样。完整逐题及分布见 [comprehensive-metrics.json](comprehensive-metrics.json)。

| Branch | n / expected | answered coverage | accuracy among answered | normalized EM | best-alias P / R / F1 | wrong / invalid / repeated / budget |
|---|---:|---:|---:|---:|---|---|
| am-serial | 128 / 128 | 0.9766 | 0.536 | 0.5234 | 0.6512 / 0.6724 / 0.6466 | 58 / 0 / 3 / 0 |
| am-batch | 128 / 128 | 0.9766 | 0.512 | 0.5 | 0.6291 / 0.6451 / 0.6232 | 61 / 0 / 3 / 0 |
| sft42 | 128 / 128 | 0.9844 | 0.5397 | 0.5312 | 0.6497 / 0.6685 / 0.6447 | 58 / 0 / 2 / 0 |
| grpo42 | 128 / 128 | 1.0 | 0.4766 | 0.4766 | 0.5898 / 0.6095 / 0.5868 | 67 / 0 / 0 / 0 |
| grpo43 | 128 / 128 | 0.9922 | 0.5197 | 0.5156 | 0.6656 / 0.6712 / 0.648 | 61 / 0 / 1 / 0 |
| base-untrained | 128 / 128 | 0.0 | NA | 0.0 | 0.0 / 0.0 / 0.0 | 0 / 14 / 114 / 1 |

## Sources and final input visibility

Doc recall uses existing required document IDs. Quote recall is exact substring coverage of existing quoted spans, not a new semantic correctness judgment. Last input excludes tool results returned by the final model call. Macro recall denominators include only confirmed source annotations, and last-visible metrics additionally require saved input metadata.

| Branch | confirmed / unknown | retrieved doc recall | all required docs | last-visible doc recall | all quotes last visible |
|---|---:|---:|---:|---:|---:|
| am-serial | 93 / 35 | 0.9247 | 0.871 | 0.9247 | 0.871 |
| am-batch | 93 / 35 | 0.9247 | 0.871 | 0.9247 | 0.871 |
| sft42 | 93 / 35 | 0.9301 | 0.8817 | 0.9301 | 0.8817 |
| grpo42 | 93 / 35 | 0.8763 | 0.7742 | 0.8763 | 0.7742 |
| grpo43 | 93 / 35 | 0.9086 | 0.828 | 0.9086 | 0.828 |
| base-untrained | 93 / 35 | 0.629 | 0.3763 | 0.629 | 0.3763 |

## am-serial: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 2.8984 | 2.0 | 3.0 | 3.0 | 3.0 | 3.73 | 4.0 |
| per_case.searches | 128 / 0 | 1.8984 | 1.0 | 2.0 | 2.0 | 2.0 | 2.73 | 3.0 |
| per_case.input_tokens | 128 / 0 | 1722.8906 | 541.0 | 1682.0 | 2430.6 | 2660.4 | 3450.19 | 3663.0 |
| per_case.output_tokens | 128 / 0 | 50.7891 | 30.0 | 50.0 | 62.0 | 67.0 | 70.73 | 76.0 |
| per_case.model_seconds | 128 / 0 | 4.6143 | 2.2123 | 4.731 | 5.8207 | 5.9687 | 6.5776 | 7.2968 |
| per_case.tool_seconds | 128 / 0 | 0.0469 | 0.0157 | 0.0409 | 0.0605 | 0.1032 | 0.1387 | 0.2734 |
| per_case.elapsed_seconds | 128 / 0 | 4.6617 | 2.231 | 4.7788 | 5.878 | 6.0292 | 6.7912 | 7.4117 |
| per_case.unique_documents | 128 / 0 | 4.8281 | 3.0 | 5.0 | 6.0 | 6.0 | 6.73 | 7.0 |
| per_action.input_tokens | 371 / 0 | 594.4205 | 45.0 | 537.0 | 1282.0 | 1440.5 | 1860.0 | 2194.0 |
| per_action.output_tokens | 371 / 0 | 17.5229 | 12.0 | 17.0 | 22.0 | 24.0 | 29.3 | 34.0 |
| per_action.json_action_tokens | 371 / 0 | 17.5175 | 12.0 | 17.0 | 22.0 | 24.0 | 29.3 | 34.0 |
| per_action.raw_action_chars | 371 / 0 | 59.159 | 36.0 | 56.0 | 78.0 | 87.0 | 109.9 | 122.0 |
| per_action.model_seconds | 371 / 0 | 1.592 | 1.003 | 1.5596 | 2.0785 | 2.2734 | 2.5875 | 3.0932 |
| per_action.batch_seconds | 0 / 371 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 371 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 243 / 128 | 0.0247 | 0.0129 | 0.0208 | 0.0308 | 0.048 | 0.0959 | 0.2512 |
| per_action.query_chars | 246 / 125 | 31.6504 | 12.0 | 28.0 | 50.5 | 58.75 | 79.55 | 89.0 |
| per_action.left_padding_tokens | 0 / 371 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 371 / 0 | 594.4205 | 45.0 | 537.0 | 1282.0 | 1440.5 | 1860.0 | 2194.0 |
| per_action.dropped_observation_tokens | 371 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.8333 | 0.8889 | 24 | 1.0 |
| development | 64 | 0.4375 | 0.5524 | 29 | 0.8621 |
| natural_supplement | 40 | 0.475 | 0.6521 | 40 | 0.8 |

Termination reasons: `{"answer": 125, "repeated_action": 3}`.
Search/query diagnostics: `{"query_count": 246, "duplicate_queries": 3, "duplicate_query_fraction": 0.012195121951219513, "document_results": 729, "identified_document_results": 729, "duplicate_document_results": 111, "duplicate_document_fraction": 0.1522633744855967, "successful_searches": 243, "search_ok_measurable": 243, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 604.1108012646437, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.2118816610000106, "output_tokens_per_second": 10.76127092313335, "model_calls_per_second": 0.6141257518047182, "model_loading_seconds": 2.5388003028929234, "gpu_memory": {"peak_allocated_gb": 1.916, "peak_reserved_gb": 2.035, "device_process_gb": 2.469, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.16780855590684546, "monetary_cost": null}`.

## am-batch: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 2.9062 | 2.0 | 3.0 | 3.0 | 3.0 | 3.73 | 4.0 |
| per_case.searches | 128 / 0 | 1.9062 | 1.0 | 2.0 | 2.0 | 2.0 | 2.73 | 3.0 |
| per_case.input_tokens | 128 / 0 | 1732.5 | 541.0 | 1682.0 | 2430.6 | 2684.9 | 3450.19 | 3663.0 |
| per_case.output_tokens | 128 / 0 | 50.8125 | 31.0 | 50.0 | 62.0 | 66.3 | 70.73 | 76.0 |
| per_case.model_seconds | 128 / 0 | 1.4841 | 0.7627 | 1.4361 | 1.911 | 1.9203 | 2.9801 | 3.4071 |
| per_case.tool_seconds | 128 / 0 | 0.293 | 0.0155 | 0.2674 | 0.6128 | 0.694 | 0.774 | 0.8134 |
| per_case.elapsed_seconds | 128 / 0 | 6.2263 | 3.3112 | 6.1786 | 8.326 | 8.3376 | 8.3697 | 9.0723 |
| per_case.unique_documents | 128 / 0 | 4.8281 | 3.0 | 5.0 | 6.0 | 6.0 | 6.73 | 7.0 |
| per_action.input_tokens | 372 / 0 | 596.129 | 45.0 | 532.5 | 1280.3 | 1442.8 | 1857.52 | 2194.0 |
| per_action.output_tokens | 372 / 0 | 17.4839 | 12.0 | 17.0 | 22.0 | 24.0 | 29.29 | 34.0 |
| per_action.json_action_tokens | 372 / 0 | 17.4785 | 12.0 | 17.0 | 22.0 | 24.0 | 29.29 | 34.0 |
| per_action.raw_action_chars | 372 / 0 | 58.8468 | 36.0 | 56.0 | 77.0 | 86.35 | 109.87 | 122.0 |
| per_action.model_seconds | 372 / 0 | 0.5107 | 0.2731 | 0.4902 | 0.6996 | 0.7254 | 0.8797 | 1.7435 |
| per_action.batch_seconds | 372 / 0 | 1.9499 | 1.0926 | 1.8666 | 2.6746 | 2.8835 | 3.1224 | 3.5186 |
| per_action.batch_active_rows | 372 / 0 | 3.8763 | 1.0 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 |
| per_action.tool_seconds | 244 / 128 | 0.1537 | 0.0155 | 0.1284 | 0.3131 | 0.3598 | 0.4234 | 0.4984 |
| per_action.query_chars | 247 / 125 | 31.2591 | 12.0 | 28.0 | 49.0 | 58.7 | 79.54 | 89.0 |
| per_action.left_padding_tokens | 372 / 0 | 151.8306 | 0.0 | 10.0 | 453.7 | 652.45 | 956.57 | 1207.0 |
| per_action.original_prompt_tokens | 372 / 0 | 596.129 | 45.0 | 532.5 | 1280.3 | 1442.8 | 1857.52 | 2194.0 |
| per_action.dropped_observation_tokens | 372 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.8333 | 0.8889 | 24 | 1.0 |
| development | 64 | 0.4062 | 0.5211 | 29 | 0.8621 |
| natural_supplement | 40 | 0.45 | 0.6271 | 40 | 0.8 |

Termination reasons: `{"answer": 125, "repeated_action": 3}`.
Search/query diagnostics: `{"query_count": 247, "duplicate_queries": 3, "duplicate_query_fraction": 0.012145748987854251, "document_results": 732, "identified_document_results": 732, "duplicate_document_results": 114, "duplicate_document_fraction": 0.1557377049180328, "successful_searches": 244, "search_ok_measurable": 244, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 217.34119565412402, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.5889357496850194, "output_tokens_per_second": 29.925297780870046, "model_calls_per_second": 1.7115945225220874, "model_loading_seconds": 1.9410679042339325, "gpu_memory": {"peak_allocated_gb": 2.875, "peak_reserved_gb": 4.305, "device_process_gb": 4.801, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.06037255434836778, "monetary_cost": null}`.

## sft42: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 2.9297 | 2.0 | 3.0 | 3.0 | 3.0 | 3.73 | 4.0 |
| per_case.searches | 128 / 0 | 1.9297 | 1.0 | 2.0 | 2.0 | 2.0 | 2.73 | 3.0 |
| per_case.input_tokens | 128 / 0 | 1784.5625 | 541.0 | 1740.5 | 2456.0 | 2751.8 | 3137.76 | 4329.0 |
| per_case.output_tokens | 128 / 0 | 51.3672 | 31.0 | 50.0 | 62.0 | 65.0 | 72.46 | 90.0 |
| per_case.model_seconds | 128 / 0 | 4.7013 | 2.319 | 4.7916 | 5.9505 | 6.1829 | 6.3963 | 7.6362 |
| per_case.tool_seconds | 128 / 0 | 0.0461 | 0.0155 | 0.0424 | 0.0593 | 0.0862 | 0.1232 | 0.1362 |
| per_case.elapsed_seconds | 128 / 0 | 4.7479 | 2.3347 | 4.8347 | 6.0129 | 6.2353 | 6.506 | 7.7016 |
| per_case.unique_documents | 128 / 0 | 4.9219 | 3.0 | 5.0 | 6.0 | 6.0 | 7.46 | 8.0 |
| per_action.input_tokens | 375 / 0 | 609.1307 | 45.0 | 553.0 | 1282.6 | 1495.8 | 2016.02 | 2083.0 |
| per_action.output_tokens | 375 / 0 | 17.5333 | 12.0 | 17.0 | 22.0 | 25.0 | 30.0 | 36.0 |
| per_action.json_action_tokens | 375 / 0 | 17.5333 | 12.0 | 17.0 | 22.0 | 25.0 | 30.0 | 36.0 |
| per_action.raw_action_chars | 375 / 0 | 59.1707 | 35.0 | 56.0 | 76.0 | 86.6 | 114.56 | 157.0 |
| per_action.model_seconds | 375 / 0 | 1.6047 | 0.9718 | 1.5671 | 2.0636 | 2.2363 | 2.6866 | 3.434 |
| per_action.batch_seconds | 0 / 375 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 375 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 247 / 128 | 0.0239 | 0.0138 | 0.0209 | 0.0331 | 0.0354 | 0.081 | 0.1089 |
| per_action.query_chars | 249 / 126 | 31.9558 | 13.0 | 28.0 | 50.0 | 59.8 | 86.52 | 124.0 |
| per_action.left_padding_tokens | 0 / 375 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 375 / 0 | 609.1307 | 45.0 | 553.0 | 1282.6 | 1495.8 | 2016.02 | 2083.0 |
| per_action.dropped_observation_tokens | 375 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.7917 | 0.9014 | 24 | 1.0 |
| development | 64 | 0.4844 | 0.5595 | 29 | 0.8966 |
| natural_supplement | 40 | 0.45 | 0.6271 | 40 | 0.8 |

Termination reasons: `{"answer": 126, "repeated_action": 2}`.
Search/query diagnostics: `{"query_count": 249, "duplicate_queries": 2, "duplicate_query_fraction": 0.008032128514056224, "document_results": 741, "identified_document_results": 741, "duplicate_document_results": 111, "duplicate_document_fraction": 0.14979757085020243, "successful_searches": 247, "search_ok_measurable": 247, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 616.3340141139925, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.20767959753771775, "output_tokens_per_second": 10.667916826644486, "model_calls_per_second": 0.6084363209112824, "model_loading_seconds": 1.9803319871425629, "gpu_memory": {"peak_allocated_gb": 1.966, "peak_reserved_gb": 2.15, "device_process_gb": 2.648, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.17120389280944234, "monetary_cost": null}`.

## grpo42: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 2.6484 | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| per_case.searches | 128 / 0 | 1.6484 | 1.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| per_case.input_tokens | 128 / 0 | 1410.5156 | 436.0 | 1485.5 | 2171.1 | 2331.6 | 2970.73 | 3139.0 |
| per_case.output_tokens | 128 / 0 | 41.7969 | 23.0 | 44.0 | 55.0 | 57.0 | 62.73 | 70.0 |
| per_case.model_seconds | 128 / 0 | 3.9191 | 1.8889 | 3.9793 | 5.1253 | 5.4119 | 5.9758 | 6.1785 |
| per_case.tool_seconds | 128 / 0 | 0.0407 | 0.014 | 0.0401 | 0.0542 | 0.0764 | 0.1154 | 0.123 |
| per_case.elapsed_seconds | 128 / 0 | 3.9602 | 1.9033 | 4.025 | 5.1715 | 5.4514 | 6.0291 | 6.2431 |
| per_case.unique_documents | 128 / 0 | 4.3672 | 3.0 | 5.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| per_action.input_tokens | 339 / 0 | 532.5841 | 45.0 | 497.0 | 1178.4 | 1299.4 | 1748.44 | 2066.0 |
| per_action.output_tokens | 339 / 0 | 15.7817 | 10.0 | 15.0 | 20.0 | 22.0 | 24.0 | 29.0 |
| per_action.json_action_tokens | 339 / 0 | 15.7817 | 10.0 | 15.0 | 20.0 | 22.0 | 24.0 | 29.0 |
| per_action.raw_action_chars | 339 / 0 | 55.4012 | 35.0 | 52.0 | 72.2 | 81.1 | 96.48 | 113.0 |
| per_action.model_seconds | 339 / 0 | 1.4798 | 0.7277 | 1.4791 | 1.8985 | 2.0226 | 2.3714 | 2.4429 |
| per_action.batch_seconds | 0 / 339 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 339 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 211 / 128 | 0.0247 | 0.014 | 0.0214 | 0.0327 | 0.0403 | 0.099 | 0.1161 |
| per_action.query_chars | 211 / 128 | 29.0569 | 5.0 | 26.0 | 46.0 | 51.5 | 76.1 | 83.0 |
| per_action.left_padding_tokens | 0 / 339 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 339 / 0 | 532.5841 | 45.0 | 497.0 | 1178.4 | 1299.4 | 1748.44 | 2066.0 |
| per_action.dropped_observation_tokens | 339 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.875 | 0.9306 | 24 | 1.0 |
| development | 64 | 0.4062 | 0.5037 | 29 | 0.6897 |
| natural_supplement | 40 | 0.35 | 0.5133 | 40 | 0.7 |

Termination reasons: `{"answer": 128}`.
Search/query diagnostics: `{"query_count": 211, "duplicate_queries": 0, "duplicate_query_fraction": 0.0, "document_results": 633, "identified_document_results": 633, "duplicate_document_results": 74, "duplicate_document_fraction": 0.11690363349131122, "successful_searches": 211, "search_ok_measurable": 211, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 515.1494100056589, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.24847160360446482, "output_tokens_per_second": 10.385336556905367, "model_calls_per_second": 0.6580615126711998, "model_loading_seconds": 1.9884395375847816, "gpu_memory": {"peak_allocated_gb": 2.06, "peak_reserved_gb": 2.988, "device_process_gb": 3.58, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.14309705833490524, "monetary_cost": null}`.

## grpo43: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 2.7969 | 2.0 | 3.0 | 3.0 | 3.0 | 3.0 | 3.0 |
| per_case.searches | 128 / 0 | 1.7969 | 1.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| per_case.input_tokens | 128 / 0 | 1591.5391 | 436.0 | 1580.5 | 2321.7 | 2533.95 | 2970.3 | 3139.0 |
| per_case.output_tokens | 128 / 0 | 48.0234 | 26.0 | 49.0 | 60.0 | 62.65 | 66.46 | 70.0 |
| per_case.model_seconds | 128 / 0 | 4.2452 | 2.1654 | 4.3708 | 5.4694 | 5.9375 | 6.4295 | 7.0691 |
| per_case.tool_seconds | 128 / 0 | 0.0382 | 0.0151 | 0.0395 | 0.0479 | 0.0527 | 0.0732 | 0.0843 |
| per_case.elapsed_seconds | 128 / 0 | 4.284 | 2.186 | 4.4112 | 5.5136 | 5.9879 | 6.4884 | 7.1262 |
| per_case.unique_documents | 128 / 0 | 4.5859 | 3.0 | 5.0 | 6.0 | 6.0 | 6.0 | 6.0 |
| per_action.input_tokens | 358 / 0 | 569.0419 | 45.0 | 513.0 | 1223.4 | 1423.65 | 1817.2 | 2063.0 |
| per_action.output_tokens | 358 / 0 | 17.1704 | 12.0 | 17.0 | 21.0 | 24.0 | 27.0 | 29.0 |
| per_action.json_action_tokens | 358 / 0 | 17.1676 | 12.0 | 17.0 | 21.0 | 24.0 | 27.0 | 29.0 |
| per_action.raw_action_chars | 358 / 0 | 58.5112 | 35.0 | 56.0 | 77.3 | 84.0 | 104.43 | 113.0 |
| per_action.model_seconds | 358 / 0 | 1.5178 | 0.8705 | 1.4718 | 1.9216 | 2.1146 | 2.6762 | 2.9093 |
| per_action.batch_seconds | 0 / 358 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 358 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 230 / 128 | 0.0212 | 0.0142 | 0.021 | 0.0251 | 0.0283 | 0.0345 | 0.0635 |
| per_action.query_chars | 231 / 127 | 31.0996 | 13.0 | 28.0 | 47.0 | 53.5 | 74.6 | 80.0 |
| per_action.left_padding_tokens | 0 / 358 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 358 / 0 | 569.0419 | 45.0 | 513.0 | 1223.4 | 1423.65 | 1817.2 | 2063.0 |
| per_action.dropped_observation_tokens | 358 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.8333 | 0.9222 | 24 | 1.0 |
| development | 64 | 0.4531 | 0.5752 | 29 | 0.7931 |
| natural_supplement | 40 | 0.425 | 0.6 | 40 | 0.75 |

Termination reasons: `{"answer": 127, "repeated_action": 1}`.
Search/query diagnostics: `{"query_count": 231, "duplicate_queries": 1, "duplicate_query_fraction": 0.004329004329004329, "document_results": 690, "identified_document_results": 690, "duplicate_document_results": 103, "duplicate_document_fraction": 0.1492753623188406, "successful_searches": 230, "search_ok_measurable": 230, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 557.3989589177072, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.2296380320633099, "output_tokens_per_second": 11.028007680415358, "model_calls_per_second": 0.6422688709270699, "model_loading_seconds": 2.1343238949775696, "gpu_memory": {"peak_allocated_gb": 2.061, "peak_reserved_gb": 2.988, "device_process_gb": 3.58, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.15483304414380755, "monetary_cost": null}`.

## base-untrained: distributions and source strata

| Metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| per_case.model_calls | 128 / 0 | 1.9453 | 1.0 | 2.0 | 2.0 | 2.0 | 3.0 | 4.0 |
| per_case.searches | 128 / 0 | 0.9453 | 0.0 | 1.0 | 1.0 | 1.0 | 2.0 | 3.0 |
| per_case.input_tokens | 128 / 0 | 683.4688 | 49.0 | 632.5 | 945.0 | 1189.2 | 2774.34 | 3147.0 |
| per_case.output_tokens | 128 / 0 | 229.1016 | 41.0 | 100.0 | 199.6 | 2048.0 | 2048.0 | 2048.0 |
| per_case.model_seconds | 128 / 0 | 9.9689 | 1.965 | 5.1354 | 9.6575 | 79.2159 | 81.3145 | 81.7455 |
| per_case.tool_seconds | 128 / 0 | 0.0266 | 0.0 | 0.0231 | 0.0381 | 0.0555 | 0.0984 | 0.1653 |
| per_case.elapsed_seconds | 128 / 0 | 9.9958 | 1.9887 | 5.1689 | 9.704 | 79.216 | 81.3145 | 81.7455 |
| per_case.unique_documents | 128 / 0 | 2.7266 | 0.0 | 3.0 | 3.0 | 3.0 | 4.0 | 5.0 |
| per_action.input_tokens | 249 / 0 | 351.3414 | 45.0 | 63.0 | 795.6 | 911.0 | 1318.96 | 2056.0 |
| per_action.output_tokens | 249 / 0 | 117.7711 | 15.0 | 29.0 | 129.2 | 175.8 | 2048.0 | 2048.0 |
| per_action.json_action_tokens | 249 / 0 | 50.4378 | 0.0 | 24.0 | 110.6 | 136.0 | 200.88 | 331.0 |
| per_action.raw_action_chars | 249 / 0 | 426.004 | 52.0 | 102.0 | 427.0 | 691.4 | 8236.6 | 9825.0 |
| per_action.model_seconds | 249 / 0 | 5.1246 | 0.7696 | 1.5114 | 5.9291 | 8.0858 | 80.8793 | 81.7455 |
| per_action.batch_seconds | 0 / 249 | NA | NA | NA | NA | NA | NA | NA |
| per_action.batch_active_rows | 0 / 249 | NA | NA | NA | NA | NA | NA | NA |
| per_action.tool_seconds | 121 / 128 | 0.0282 | 0.0159 | 0.0232 | 0.0375 | 0.0553 | 0.0986 | 0.1653 |
| per_action.query_chars | 235 / 14 | 43.3915 | 19.0 | 41.0 | 63.6 | 68.0 | 81.0 | 81.0 |
| per_action.left_padding_tokens | 0 / 249 | NA | NA | NA | NA | NA | NA | NA |
| per_action.original_prompt_tokens | 249 / 0 | 351.3414 | 45.0 | 63.0 | 795.6 | 911.0 | 1318.96 | 2056.0 |
| per_action.dropped_observation_tokens | 249 / 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

| Source stratum | n | normalized EM | best-alias F1 | source-known | all required docs retrieved |
|---|---:|---:|---:|---:|---:|
| constructed_singlehop | 24 | 0.0 | 0.0 | 24 | 0.875 |
| development | 64 | 0.0 | 0.0 | 29 | 0.1724 |
| natural_supplement | 40 | 0.0 | 0.0 | 40 | 0.225 |

Termination reasons: `{"repeated_action": 113, "invalid_action": 14, "max_steps": 1}`.
Search/query diagnostics: `{"query_count": 235, "duplicate_queries": 114, "duplicate_query_fraction": 0.4851063829787234, "document_results": 363, "identified_document_results": 363, "duplicate_document_results": 14, "duplicate_document_fraction": 0.03856749311294766, "successful_searches": 121, "search_ok_measurable": 121, "search_success_rate": 1.0, "empty_search_results": 0}`.
Measured cost/throughput: `{"evaluation_wall_seconds": 1280.0377118960023, "wall_scope": "complete_branch", "resumed": false, "current_process_cases": 128, "questions_per_second": 0.09999705384492567, "output_tokens_per_second": 22.909481281269105, "model_calls_per_second": 0.19452551880770697, "model_loading_seconds": 2.150679960846901, "gpu_memory": {"peak_allocated_gb": 1.889, "peak_reserved_gb": 2.0, "device_process_gb": 2.434, "device_memory_source": "single_compute_process_fallback"}, "gpu_hours": 0.35556603108222284, "monetary_cost": null}`.

## Training

| Branch | groups | mixed / all correct / all wrong | effective groups | unique updates | training seconds | completed generation seconds |
|---|---:|---|---:|---:|---:|---:|
| sft42 | 0 | 0 / 0 / 0 | 0 | 128 | 233.4252 | NA |
| grpo42 | 128 | 54 / 16 / 58 | 54 | 108 | 1882.7115 | 694.4491 |
| grpo43 | 128 | 55 / 17 / 56 | 55 | 110 | 1909.8552 | 745.642 |

sft42 reward histogram: `{}`; outcome entropy (bits) mean `NA`; effective fraction `NA`.
Group diversity means: `{"candidate_count": null, "unique_normalized_answers_including_empty": null, "unique_nonempty_normalized_answers": null, "empty_candidate_count": null, "unique_query_sequences": null, "unique_action_sequences": null, "unique_action_type_sequences": null, "equal_normalized_answer_pair_fraction": null}`.
SFT process NLL (not final checkpoint PPL): `{"record_count": 128, "supervised_tokens": 6479, "token_weighted_nll": 0.3215638162220206, "exp_process_nll": 1.3792830238492877, "interpretation": "Token-weighted mean of recorded SFT forward NLL during training (physical records, including replay); exp is a process statistic, not final-checkpoint perplexity. RL surrogate loss is not NLL/PPL."}`.

| Update metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 128 / 0 | 0.2851 | 0.0026 | 0.2003 | 0.6678 | 0.749 | 0.8682 | 0.9486 |
| kl | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| clip_fraction | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| grad | 128 / 0 | 7.1551 | 0.2859 | 6.9707 | 13.342 | 14.6159 | 16.8337 | 37.6121 |
| supervised_tokens | 128 / 0 | 50.6172 | 28.0 | 54.5 | 63.0 | 65.0 | 69.19 | 79.0 |
| ratio.min | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| ratio.max | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| ratio.p95 | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| ratio.count | 0 / 128 | NA | NA | NA | NA | NA | NA | NA |
| group.reward_std | 0 / 0 | NA | NA | NA | NA | NA | NA | NA |

grpo42 reward histogram: `{"0.0": 338, "1.0": 174}`; outcome entropy (bits) mean `0.3629`; effective fraction `0.4219`.
Group diversity means: `{"candidate_count": 4.0, "unique_normalized_answers_including_empty": 2.6171875, "unique_nonempty_normalized_answers": 2.421875, "empty_candidate_count": 0.2734375, "unique_query_sequences": 3.3984375, "unique_action_sequences": 3.671875, "unique_action_type_sequences": 1.5546875, "equal_normalized_answer_pair_fraction": 0.3580729166666667}`.
SFT process NLL (not final checkpoint PPL): `{"record_count": 0, "supervised_tokens": 0, "token_weighted_nll": null, "exp_process_nll": null, "interpretation": "Token-weighted mean of recorded SFT forward NLL during training (physical records, including replay); exp is a process statistic, not final-checkpoint perplexity. RL surrogate loss is not NLL/PPL."}`.

| Update metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 108 / 0 | -0.0 | -0.02 | 0.0005 | 0.0028 | 0.0057 | 0.0138 | 0.0176 |
| kl | 108 / 0 | 0.0724 | 0.0 | 0.0449 | 0.0876 | 0.1528 | 0.8481 | 1.2432 |
| clip_fraction | 108 / 0 | 0.0031 | 0.0 | 0.0 | 0.0115 | 0.0147 | 0.0403 | 0.0466 |
| grad | 108 / 0 | 7.1356 | 1.3506 | 4.0182 | 9.1305 | 21.6746 | 47.1005 | 106.4153 |
| supervised_tokens | 0 / 108 | NA | NA | NA | NA | NA | NA | NA |
| ratio.min | 108 / 0 | 0.8963 | 0.4899 | 0.9789 | 1.0 | 1.0 | 1.0 | 1.0 |
| ratio.max | 108 / 0 | 1.0822 | 1.0 | 1.0092 | 1.2005 | 1.3448 | 1.5349 | 1.5957 |
| ratio.p95 | 108 / 0 | 1.009 | 1.0 | 1.0008 | 1.0274 | 1.0409 | 1.069 | 1.0923 |
| ratio.count | 108 / 0 | 176.8519 | 107.0 | 178.5 | 214.0 | 222.6 | 226.79 | 227.0 |
| group.reward_std | 128 / 0 | 0.19 | 0.0 | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 |

grpo43 reward histogram: `{"0.0": 334, "1.0": 178}`; outcome entropy (bits) mean `0.3678`; effective fraction `0.4297`.
Group diversity means: `{"candidate_count": 4.0, "unique_normalized_answers_including_empty": 2.640625, "unique_nonempty_normalized_answers": 2.4296875, "empty_candidate_count": 0.2578125, "unique_query_sequences": 3.5, "unique_action_sequences": 3.6953125, "unique_action_type_sequences": 1.4765625, "equal_normalized_answer_pair_fraction": 0.34765625}`.
SFT process NLL (not final checkpoint PPL): `{"record_count": 0, "supervised_tokens": 0, "token_weighted_nll": null, "exp_process_nll": null, "interpretation": "Token-weighted mean of recorded SFT forward NLL during training (physical records, including replay); exp is a process statistic, not final-checkpoint perplexity. RL surrogate loss is not NLL/PPL."}`.

| Update metric | count / missing | mean | min | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| loss | 110 / 0 | -0.0007 | -0.0057 | 0.0001 | 0.0006 | 0.0011 | 0.0017 | 0.0021 |
| kl | 110 / 0 | 0.0165 | 0.0 | 0.0114 | 0.031 | 0.0657 | 0.0936 | 0.0966 |
| clip_fraction | 110 / 0 | 0.0028 | 0.0 | 0.0 | 0.0104 | 0.0145 | 0.0274 | 0.0483 |
| grad | 110 / 0 | 4.0631 | 1.0218 | 3.7319 | 6.0134 | 7.0875 | 12.5031 | 17.8246 |
| supervised_tokens | 0 / 110 | NA | NA | NA | NA | NA | NA | NA |
| ratio.min | 110 / 0 | 0.9054 | 0.3371 | 0.9674 | 1.0 | 1.0 | 1.0 | 1.0 |
| ratio.max | 110 / 0 | 1.0934 | 1.0 | 1.0027 | 1.2415 | 1.3924 | 1.5217 | 1.6749 |
| ratio.p95 | 110 / 0 | 1.0106 | 1.0 | 1.0002 | 1.0333 | 1.0436 | 1.0526 | 1.0725 |
| ratio.count | 110 / 0 | 193.9455 | 126.0 | 196.0 | 226.0 | 246.2 | 268.56 | 270.0 |
| group.reward_std | 128 / 0 | 0.1929 | 0.0 | 0.0 | 0.4531 | 0.5 | 0.5 | 0.5 |

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

