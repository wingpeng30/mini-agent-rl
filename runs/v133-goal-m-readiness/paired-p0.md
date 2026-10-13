# P0 逐题配对（开发集，事后诊断）

| 集合 | ID | v118 动作/EM/F1 | Goal J 动作/EM/F1 | 变化类 |
|---|---|---|---|---|
| new_eval | 5a8673585542991e77181662 | search/0/0.000 | answer/0/0.667 | answer_timing_with_qualifier_omission |
| new_eval | 5ac00c72554299012d1db589 | search/0/0.000 | answer/1/1.000 | answer_timing_and_shared_attribute |
| new_eval | 5add23bd5542994734353870 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5ae30de055429928c42395cf | answer/0/0.250 | answer/0/0.500 | answer_granularity_under_specified |
| new_eval | 5a8d92a3554299068b959d40 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5ae3be675542992e3233c46e | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5add27645542992ae4cec4b5 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5adde5005542990dbb2f7eee | answer/0/0.500 | answer/1/1.000 | requested_attribute_selection |
| new_eval | 5a7789215542997042120a99 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5a73583e55429901807dafd0 | answer/0/0.000 | answer/0/0.000 | serialization_only |
| new_eval | 5a8c717f5542995e66a475ef | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5ac259bc5542992f1f2b386a | search/0/0.000 | answer/0/0.667 | answer_timing_with_qualifier_omission |
| new_eval | 5ae0e4a55542990adbacf6b6 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| new_eval | 5a78cc3f554299029c4b5e9b | answer/0/0.000 | answer/0/0.000 | wrong_attribute_and_under_specification |
| new_eval | 5ac169c25542994ab5c67d1a | answer/1/1.000 | answer/0/0.667 | answer_granularity_loss |
| new_eval | 5a842f375542990548d0b25c | answer/1/1.000 | answer/1/1.000 | serialization_only |
| goal_G_regression | g000 | search/0/0.000 | answer/1/1.000 | answer_timing_and_shared_attribute |
| goal_G_regression | g001 | search/0/0.000 | answer/0/0.400 | answer_timing_with_missing_attributes |
| goal_G_regression | g003 | answer/0/0.667 | answer/1/1.000 | answer_granularity |
| goal_G_regression | g004 | answer/0/0.000 | answer/0/0.667 | answer_granularity_under_specified |
| goal_G_regression | g005 | answer/0/0.000 | answer/1/1.000 | shared_attribute_selection |
| goal_G_regression | g006 | answer/0/0.667 | answer/0/0.667 | answer_granularity_under_specified |
| goal_G_regression | g008 | answer/0/0.000 | answer/1/1.000 | answer_granularity |
| goal_G_regression | g009 | answer/1/1.000 | answer/0/0.000 | semantic_equivalent_strict_em_loss |
| goal_G_regression | g010 | answer/0/0.000 | answer/0/0.000 | wrong_shared_attribute |
| goal_G_regression | g011 | answer/0/0.051 | answer/1/1.000 | answer_granularity |
| goal_G_regression | g012 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| goal_G_regression | g013 | answer/1/1.000 | answer/1/1.000 | serialization_only |
| goal_G_regression | g014 | answer/0/0.800 | answer/0/0.667 | answer_granularity_under_specified |
| goal_G_regression | g015 | answer/0/0.000 | answer/1/1.000 | requested_taxonomic_level_selection |
| goal_G_regression | g017 | answer/0/0.125 | answer/1/1.000 | answer_granularity |
| goal_G_regression | g018 | answer/0/0.286 | answer/0/0.000 | unsupported_shared_attribute |

new_eval 双方有效 answer：13 题；严格 EM 9→9；ID：5add23bd5542994734353870, 5ae30de055429928c42395cf, 5a8d92a3554299068b959d40, 5ae3be675542992e3233c46e, 5add27645542992ae4cec4b5, 5adde5005542990dbb2f7eee, 5a7789215542997042120a99, 5a73583e55429901807dafd0, 5a8c717f5542995e66a475ef, 5ae0e4a55542990adbacf6b6, 5a78cc3f554299029c4b5e9b, 5ac169c25542994ab5c67d1a, 5a842f375542990548d0b25c。

goal_G_regression 双方有效 answer：14 题；严格 EM 3→8；ID：g003, g004, g005, g006, g008, g009, g010, g011, g012, g013, g014, g015, g017, g018。
