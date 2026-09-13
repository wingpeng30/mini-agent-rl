# v0.7.2 最终评测分析

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-14
- Verification Status: ANALYZED
- Version Label: final_eval_v072
- Experiment ID: mini-agent-rl-v072-final
- Frozen Policy: `checkpoints/qwen35-08b-grpo-v071-seed44`
- Adapter SHA256: `76f61ebd2313b6939075d7767e3b424a04f2007ff1003e08b41d5334c062d90f`
- Data SHA256: `85cd82152fbce1a567311b27587111ce98a3d3a51f1a5ae088851430c3bc628c`
- External Transfer: none

## 验证结论

300 条最终测试全部完成并写入 SQLite；基础设施失败为 0，数据库包含 300 个 rollout、726 个 transition 和 2053 个 event。报告和数据库的汇总一致。整体置信等级为 **CAUTION**：样本规模足以说明当前固定环境中的表现，但只有一个最终 checkpoint、一个确定性解码设置和一个离线检索环境，不能外推到完整 HotpotQA 或开放式 Agent。

| 指标 | 最终结果 | 95% Wilson CI |
| --- | ---: | ---: |
| Exact match | 27.3%（82/300） | 22.6%–32.6% |
| Answer completion | 67.0%（201/300） | 61.5%–72.1% |
| Repeated-action termination | 26.0%（78/300） | 21.4%–31.2% |
| Invalid-action termination | 6.7%（20/300） | 4.4%–10.1% |

平均 reward 为 `0.2847`，总体标准差为 `0.7119`，范围为 `[-0.60, 1.37]`；196 条为正奖励、104 条为负奖励。平均搜索 `1.683` 次，平均执行 `2.42` 步，搜索动作成功率 `84.6%`，峰值显存 `2.852 GB`。

## 终止原因和奖励

| 终止原因 | 数量 | 比例 | 平均 reward | Exact match |
| --- | ---: | ---: | ---: | ---: |
| answer | 201 | 67.0% | 0.6816 | 82 |
| repeated_action | 78 | 26.0% | -0.5144 | 0 |
| invalid_action | 20 | 6.7% | -0.5570 | 0 |
| max_steps | 1 | 0.3% | -0.3400 | 0 |

Reward 分项总和为：exact match `+82.0`、evidence support `+36.8`、support coverage `+32.1`、search cost `-15.15`、repeated search `-39.0`、invalid action `-10.0`、max steps `-0.25`、ungrounded answer `-1.1`。重复检索是最大的可学习负项。

回答完成率为 67%，但全体 exact match 只有 27.3%；在已回答轨迹内部，exact match 为 `82/201 = 40.8%`。因此下一阶段不能只继续强化“及时停止”：还需要提升检索后的答案抽取、答案规范化和证据整合。

## 按原始 difficulty 分层

| difficulty | n | Exact match | repeated | invalid | 平均 reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| easy | 68 | 39.7% | 16.2% | 5.9% | 0.5018 |
| medium | 184 | 22.8% | 28.8% | 8.2% | 0.2010 |
| hard | 48 | 27.1% | 29.2% | 2.1% | 0.2977 |

原始 difficulty 与本 Agent 的实际难度并非单调关系；它不是专门针对当前检索工具和动作协议定义的标签，不应据此声称 hard 优于 medium。

## 与开发集的差异

被选中的 seed 44 在 30 条开发集上达到 EM `53.3%`、mean reward `0.6143`、repeated `23.3%`、invalid `3.3%`。最终集分别为 `27.3% / 0.2847 / 26.0% / 6.7%`。EM 下降 26.0 个百分点、mean reward 下降 0.3296。

这说明开发集 30 题的结果偏乐观，可能同时包含小样本波动、在同一开发集上多次选择 checkpoint 的选择偏差，以及最终样本组成差异。由于没有对多个最终 checkpoint 进行评测，无法分解三种因素；最终集也不应再次用于调参验证。

## 11 项谬误扫描

- Coverage: 11/11 checked。
- Simpson's paradox：NOTE；已按 difficulty 检查，未出现所有子组方向与总体相反，但标签分布不均。
- Ecological fallacy：NOTE；仅报告轨迹/数据集层面结果，不推断真实用户行为。
- Berkson's paradox：CAUTION；最终样本经过 supporting-document 隔离筛选，并非完整 HotpotQA 随机样本。
- Collider bias：NOTE；未进行控制变量回归，不适用。
- Base-rate neglect：NOTE；报告了全部终止原因基率。
- Regression to the mean：CAUTION；checkpoint 从三个候选中按 30 题开发分数选出，开发分数可能自然回落。
- Survivorship bias：NOTE；300 条全部完成，没有只分析成功轨迹。
- Look-elsewhere effect：CAUTION；比较了三个 seed 后择优，开发集结果存在多重选择效应。
- Garden of forking paths：CAUTION；最终集协议已预注册，但此前训练与开发实验经历多轮迭代。
- Correlation ≠ causation：NOTE；结果只描述关联，不声称 GRPO 单独造成提升。
- Reverse causality：NOTE；不适用于该离线基准设计。

## 下一步建议

保持本最终集封存，不再据此修改后重新报告同一 final 指标。v0.8.0 应回到训练/验证/开发数据，优先实现三项：针对重复检索的困难负轨迹训练；对 answer-only token 增加答案正确性学习信号；增加规范化 EM 与 token-level F1 作为诊断指标。完成新版本后应重新构造另一份隔离 final set，而不是复用本次 300 题。

v0.7.3 已在未读取 final 的前提下补充规范化 EM、token F1 和 100 条开发集 SFT/GRPO 对照。新结果解释了既有行为，但不追溯改写本页的 v0.7.2 最终结果。
