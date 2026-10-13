# Goal R：双侧证据与文档顺序对照

状态：`COMPLETED_NO_NEW_MODEL_PROMOTED`。两组匹配的答案 SFT 均执行完成，独立顺序诊断完成。最终产品继续使用 Goal P RL-only adapter。两个新模型均从该 adapter 出发，但在原产品检索顺序下由 19/32 降至 15/32，回归开发由 7/16 降至 6/16，没有达到预定升级门槛，故不启动 seed 43 复现。

## 同条件真实搜索结果

环境沿用 Goal Q：96 篇共享文档、BM25-v2 top-3、`direct-v090`、通用 `search.text → query` 兼容、每题最多 4 次模型调用/3 次搜索、128 新 token、2048 输入 token。每个模型在两个已暴露开发集各 16 题上 greedy 运行一次，所有失败计入。

| 模型 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 双侧证据召回 | 无效动作 | 搜索数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前最佳 Goal P RL-only | **12/16** | **7/16** | **19/32** | **.7346** | 32/32 | 0 | 32 |
| A：原顺序答案 SFT | 9/16 | 6/16 | 15/32 | .6938 | 32/32 | 0 | 32 |
| B：加入反转顺序的答案 SFT | 9/16 | 6/16 | 15/32 | .6938 | 32/32 | 0 | 32 |

基线在新运行中完全复现 Goal Q 的 19/32、回归 7/16。A/B 都保留了 32/32 双侧检索和每题一次搜索，退步发生在最终回答。相对基线，两臂都只新答对 `g008`，同时丢失主要开发 3 题和回归开发 2 题；B 与 A 只有 `g018` 的答案文字不同，严格分数相同。

丢失题中，`film director → director` 两题、`documentary film → documentary`、`Democratic Party → Democratic` 显示训练后更常给出较粗粒度的标签；`lemon juice → honey` 是内容差异。训练题的冻结目标含 `director`、`documentary` 等短标签，这是值得重视的迁移风险，但不能仅凭这一次运行证明全部退步都由标签粒度造成。我们没有修改金标或事后加别名。

## 训练和顺序诊断

当前最佳模型对 16 道训练题各运行一次真实搜索。16 题均在回答前检索到两侧支持证据且输入未截断。原顺序重建后的输入 token 与产品生成时完全相同；反转版本仅改变实际工具返回的文档数组顺序。问题、查询、文档内容/ID/分数及目标答案保持一致。

A/B 均从相同的历史 adapter 独立加载，使用相同 seed 42 题目顺序、两遍训练、每题每遍两次更新，共各 64 次。A 用两份原顺序输入，B 各用一份原顺序和反转顺序输入；两者都只监督最终 answer 动作及终止符。A/B 分别有 192 个 LoRA tensor 变化，base 未变化，独立重载的 logprob 最大差为 0。平均 token CE 第一遍约 .306/.298，第二遍两臂均约 .055。损失下降证明训练拟合，产品评价显示并未获得增益。

统一产品评价结束后，使用**基线实际保存的回答前上下文**对三个模型分别生成原顺序与反转顺序回答，共 192 条诊断输出。该干预不参与选模：

| 模型 | 原顺序 EM | 反转顺序 EM | 答案改变题数 |
|---|---:|---:|---:|
| 当前最佳 | 19/32 | 18/32 | 8/32 |
| A | 15/32 | 19/32 | 7/32 |
| B | 15/32 | 19/32 | 9/32 |

顺序确实会改变部分回答，但 B 未比 A 更稳定。A/B 在反转条件下同样升至 19/32，说明这次反转顺序训练没有提供可分辨的额外优势。反转分数是固定上下文下的诊断，产品运行仍保留检索器原顺序，不将 19/32 的反转结果当作产品成绩。

## 运行核查与故障处理

首次诊断进程因回放轨迹的文档 JSON 字段顺序被保存器排序而停止：原产品输入中的单篇文档按 `id,title,text,score` 编码，持久化记录按字母顺序排列。训练输入在内存中构造并通过逐 token 对照，不受此问题影响；A/B 产品评价也已完成。独立修复脚本恢复原工具字段顺序，先核对原顺序提示逐 token 相同，再完成全部 192 条诊断。失败和修复记录均保留，不重跑训练。

原阶段台账记录 GPU 进程 **387.208 秒**。另有定位错误和交互演示的手工 GPU 进程约 32 秒，含入后的估计总占用 **约 419.206 秒**，低于 7200 秒上限。详见 `state.json` 与 `manual-gpu-audit.json`。CPU 审计重新计算三模型共 96 条产品输出及 192 条顺序诊断的 EM/F1，并核对支持文档和原顺序一致性，全部通过。新增四项针对性测试通过。

## 复现与产品入口

AutoDL 工作目录：`/root/autodl-tmp/mini-agent-rl`。现有运行目录拒绝覆盖，重新完整运行须使用新的隔离目录。查看结果与运行当前选中模型：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_r_evidence_order.py summarize
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_r_selected.py
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_r_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

完整复现请使用修复了持久化回放字段顺序的 v2 入口，并指定新的独立输出目录；本次原始训练仍由 v1 脚本及其冻结哈希界定：

```bash
export GOAL_R_OUT=/root/autodl-tmp/mini-agent-rl/runs/v133-goal-r-reproduction-01
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_r_evidence_order_v2.py prepare
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_r_evidence_order_v2.py run
```

本地 CPU 审计和曲线：

```bash
PYTHONPATH=src:scripts python scripts/analyze_v133_goal_r_evidence_order.py
```

`config.json` 冻结权重、输入、代码哈希、seed 42/43 的题目顺序和胜出规则；`train-inputs.jsonl`、`eval-*.jsonl`、`order-diagnostic.jsonl`、`audit.json` 和 `training-curves.png` 保留逐条证据。当前 96 篇文档来自这批已知题目的冻结证据，两个开发集已在多轮决策中使用；本结果不能证明新问题上的泛化表现。
