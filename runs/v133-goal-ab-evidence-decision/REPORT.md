# Goal AB：保留连续搜索，训练证据充分性判断

## 结论与产品决定

**本轮完成了诊断、54 对真实上下文决策材料、四次 128 更新训练和全部开发／回归评价；没有路线通过开发门槛。** 新确认集的 64 题及独立 2048 文档库已冻结，但未生成任何确认输出。产品继续使用 Goal AA 发布的 Goal Z 完整轨迹 SFT seed 42 adapter：`runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final`。

本轮是对“追加决策监督”这套训练方案的检验。B 在两个 seed 上各比匹配的 A 多 1 道语义成功题，但未达到相对起点每 seed 至少 +4/64 的预设产品门槛；开发集错误合计上升，停止决策错误未在两个 seed 上同时相对 A 下降。**没有证据支持发布 B，也不能据此断言证据充分性判断本身无价值。** 两 seed 的 B 都有增益但小，且 54 对中仅 8 对的起点轨迹含已审核的停止决策错误，直接针对错误状态的监督比例有限。

## 固定材料与工程验收

- 起点、模型与环境：Goal Z 完整轨迹 seed 42 adapter，原 0.8B 基础权重，Goal Q 搜索入口、BM25-v2/top-3、最多 3 次搜索与 4 次模型调用、2048 输入／128 输出 token；运行采用远程仓库 `.venv/bin/python`。资产、语料与代码哈希见 [config.json](config.json) 和 [execution-code.json](execution-code.json)。
- 诊断：Goal Z 开发 64 题中有 17 道停止决策错误，超过预设的 8 题启动门槛。128 道原训练题的新 greedy 轨迹有 52 道停止决策错误。64 个定额提案中 54 个获准，超过 32 对门槛；每对均有证据不足时的 search 状态和证据充分时的 answer 状态。审核详情见 [pair-review-summary.json](pair-review-summary.json)、[decision-pairs.jsonl](decision-pairs.jsonl) 和 [diagnostic-review.jsonl](diagnostic-review.jsonl)。
- A 和 B 各以 seed 42、43 从同一起点独立训练一次，每次 128 次更新。B 对 54 道合格题采用 50% 原轨迹损失加 50% 决策损失，其余题与 A 相同。两臂题序、优化器和原轨迹监督相同；B 额外监督 2017 个决策动作 token。四个 checkpoint 的 LoRA 均变化，基础权重保持不变，独立重载一致；摘要与逐步损失在 `seed42-A`、`seed42-B`、`seed43-A`、`seed43-B` 目录。峰值 PyTorch allocated 显存约 11.85 GB，训练加开发／回归评价约 2589 秒。没有设置 GPU 时间上限。
- 起点和四候选各评价 Goal Z 开发 64、旧题 32、Goal Y 48、AA 已暴露确认 64，共 1040 条产品轨迹。匿名去重后审核 394 个“问题—实际输入—答案”组合。所有成功判断均有最终实际可见文档 ID 引用。两个 reviewer 辅助，主 agent 裁定 5 个分类争议，记录见 [development-review-adjudications.jsonl](development-review-adjudications.jsonl)；这是代理辅助审核，不是独立人工金标准。

## 主要结果

表内为语义成功题数。`事实错＋无证据` 是 Goal AB 预先约定的合计；无效动作在全部集合及模型中都是 0。严格 EM、F1、搜索次数和延迟的逐题数据见 [per-question-results.csv](per-question-results.csv) 与 [summary.json](summary.json)。

| 集合 | 起点 | A42 | B42 | A43 | B43 |
|---|---:|---:|---:|---:|---:|
| Goal Z 新开发 /64 | 36 | 38 | 39 | 36 | 37 |
| 旧主要开发 /16 | 15 | 15 | 14 | 14 | 14 |
| 旧回归开发 /16 | 12 | 12 | 12 | 12 | 12 |
| Goal Y /48 | 40 | 43 | 43 | 43 | 43 |
| AA 已暴露确认，现作回归 /64 | 32 | 36 | 38 | 38 | 39 |

| Goal Z 新开发指标 /64 | 起点 | A42 | B42 | A43 | B43 |
|---|---:|---:|---:|---:|---:|
| 严格 EM | 24 | 27 | 26 | 26 | 27 |
| 事实错＋无证据 | 19 | 21 | 20 | 24 | 24 |
| 停止决策错误题 | 15 | 20 | 19 | 19 | 19 |
| 搜索总次数 | 130 | 129 | 128 | 126 | 126 |

起点需达到 **40/64**，两 seed 同时满足，才可进入新确认。A 为 38、36，B 为 39、37；两路线均不合格。开发集错误合计也均超过起点 19。逐题对照：A42 新增 3、失去 1；B42 新增 4、失去 1；A43 新增 2、失去 2；B43 新增 3、失去 2。完整题号、答案、审核依据与轨迹指标在 [per-question-results.csv](per-question-results.csv) 和 [per-question-analysis.json](per-question-analysis.json)。

机制判据亦未通过：相对 A，B 在 seed 42 将停止错误题从 20 降到 19，seed 43 都是 19；两 seed 的语义成功虽各 +1，停止错误并未双 seed 严格下降。开发集起点 14 道证据不足时提前回答，候选 A42/B42/A43/B43 分别 18/17/19/19；当前方案未减少这类核心失误。AA 回归中，按本轮同口径起点已成功的必要连续搜索题比较，A42 丢失 2，B42/A43/B43 未丢失；这说明部分候选保留了该能力，但不能弥补新开发门槛未通过。

## 审核口径与更正

Goal AA 原报告不改写。AB 预先冻结“事实错误需明确错误关系或反证；单纯缺少支持归证据不支持”，并把两者合并约束。重新审核 AA 已暴露确认输出后，本轮同口径起点为 32/64；这不是新模型使历史 35/64 变差。题目不更换、分母不缩小。

原门槛程序在计算“起点已成功的必要连续搜索题丢失”时，直接使用 AA 历史成功清单 31 题；按 AB 同口径，其中 4 题在起点已不再算成功，故原程序把这些也计为候选丢失。保留原 [development-decision.json](development-decision.json) 以供追溯；[gate-correction.json](gate-correction.json) 给出同口径保护集 27 题以及更正后丢失数 A42=2、B42=0、A43=0、B43=0。**修正不改变“无路线合格、确认集关闭”的决定**：四候选的新开发收益与错误合计均未过线。后续复用门槛代码时须修复该保护集定义。

此外，主 agent 将“FDA”回答“该机构的食品安全体系”的同一输出判为答错属性；对三个原问题含不可解释的 `free crops` 的输出判为无法判定并保留在分母；一条 Mara Region 回答因缺少水体命名关系改判为证据不支持。这些更正对起点与候选相同输入同步应用。

## 新确认集与复现

新确认一次性冻结 MuSiQue 32、2Wiki 32 题和独立 2048 篇自然文档库。由于开发门槛失败，`state.json` 中 `confirmation_accessed=false`，`eval-fresh_confirm-*.jsonl` 不存在。此处“没有确认结果”表示**按预设规则没有运行确认评价**，不是确认集得 0 分。

在 AutoDL 仓库中复算和查看：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_ab_evidence_decision.py summarize
PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v133_goal_ab_evidence_decision.py -q --basetemp /tmp/pytest-goal-ab-final -p no:cacheprovider
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_z_selected.py --corpus /path/to/matching-corpus.json --question '你的问题'
```

`prepare` 与 `run` 已在冻结的独立输出目录执行；目录拒绝覆盖已开始阶段，不应对现有结果再次运行。演示入口会打印实际语料 SHA-256，未指定 `--corpus` 时使用旧 96 篇产品文档库。产品选模函数已核实仍返回 Goal AA 发布的 seed 42 adapter。聚焦测试 **5 项通过**。

本轮结果限于已暴露开发／回归集。当前唯一推荐模型仍为 Goal Z 完整轨迹 SFT seed 42；下一步若继续研究停止判断，应先改进训练题中**真正发生停止错误且可形成合格状态对**的覆盖，再预先冻结新实验，而不是依据这次未打开的确认集选模。
