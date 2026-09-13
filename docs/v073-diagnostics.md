# v0.7.3 诊断与 SFT/GRPO 对照

本版本只修正评测可观测性，不修改 v0.6/v0.7 的 reward 权重，也不读取已封存的 `data/hotpot-agent-v072/final-test.jsonl`。

## 指标

- `exact_match_rate` / `strict_exact_match_rate`：历史规则，只做小写和空白归一化。
- `normalized_exact_match_rate`：进一步移除英文冠词和标点，作为表达差异诊断。
- `token_f1`：归一化 token 的多重集合 F1。
- `answer_completion_rate`：终止原因为 `answer` 的比例，不代表答案正确。
- `repeated_rate`、`invalid_rate`、`truncation_rate`：分别表示策略重复、非法动作和 prompt 截断比例。

## 命令

```powershell
mini-agent-rl analyze-rollouts --db runs-v072-dev-seed44.db --data data/hotpot-agent-v041/test.jsonl --output reports/dev-v073-seed44-diagnostics.json
mini-agent-rl evaluate-matrix --data data/hotpot-agent-v041/test.jsonl --limit 100 --output reports/v073-matrix.json --db-dir runs-v073
```

矩阵命令固定顺序运行 `qwen35-08b-sft-v052`、`qwen35-08b-grpo-v071-audit-smoke`、`qwen35-08b-grpo-v071-seed43` 和 `qwen35-08b-grpo-v071-seed44`。每个模型使用单独的 SQLite、逐模型报告和诊断明细，并保存本地策略指纹。四个模型都使用同一批开发数据、同一离线语料、greedy 解码、最多 3 步和单 GPU 串行执行。

## 解释边界

规范化 EM 和 token F1 只用于诊断，不能替换历史严格 EM，也不修改 reward。开发集对照可以说明 SFT 与 GRPO 在当前离线环境中的行为差异，但不能覆盖 v0.7.2 final 的结论，也不能代表开放检索或完整 HotpotQA 泛化。

## 100 条开发集结果（2026-09-14）

| checkpoint | EM | normalized EM | token F1 | answer completion | repeated | invalid | mean reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT v0.5.2 | 13% | 13% | 0.135 | 20% | 61% | 15% | -0.1746 |
| GRPO seed42 | 37% | 37% | 0.432 | 68% | 19% | 13% | 0.3916 |
| GRPO seed43 | 45% | 45% | 0.493 | 66% | 32% | 2% | 0.4851 |
| GRPO seed44 | 44% | 44% | 0.494 | 71% | 24% | 5% | 0.5015 |

四组基础设施失败率均为 0，峰值显存约 `2.85 GB`。严格 EM 与规范化 EM 完全相同，说明低分不是冠词或标点造成的评分假象。相同题目的成对比较中，三个 GRPO checkpoint 相比 SFT 分别新增答对 `25/32/31` 题，仅丢失 `1/0/0` 个 SFT 原本正确的题，支持 GRPO 确实改善了当前开发环境中的策略。

主要作用路径是停止行为：SFT 只有 20% 的轨迹进入 answer，61% 因重复搜索终止；GRPO 把 answer completion 提高到 `66%–71%`，重复率降到 `19%–32%`。在已经回答的轨迹中，SFT/seed42/seed43/seed44 的严格正确率分别为 `65.0%/54.4%/68.2%/62.0%`，因此不能把全部收益归因于答案生成能力。

正确轨迹的平均 supporting-document coverage 明显高于错误轨迹：三个 GRPO 分别为 `0.892 vs 0.381`、`0.933 vs 0.482`、`0.909 vs 0.411`。空检索结果仍有 `20/32/24` 条；prompt 截断仅 `2%/3%/2%`，当前不是主要失败来源。

首次矩阵运行因截断诊断元数据被错误加入 Transformers `model_kwargs`，四组均基础设施失败。修复后元数据只挂在 `BatchEncoding` 对象属性上；失败产物保留在 `runs-v073/`，有效复跑保存在 `runs-v073-rerun/`，避免结果混合。

机器可读总报告为本地忽略文件 `reports/v073-matrix-rerun.json`，四份逐模型诊断报告使用同名前缀。结果只用于解释既有模型，不据此重新选择 checkpoint。
