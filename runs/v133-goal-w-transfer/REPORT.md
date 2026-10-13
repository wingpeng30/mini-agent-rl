# Goal W：扩大双侧证据训练覆盖的前置验收

## 决策

**按 W1 的预设数据门槛停止，未启动 128 对训练或新开发／确认评价。** 当前唯一产品模型继续为 Goal T seed 42、第 32 次更新的 adapter。本轮没有产生有关“大规模偏好训练是否有效”的模型对照结论；发现的是固定候选池不足以支撑已约定的四类训练覆盖。

AutoDL 实例仍为 `ssh -p 25320 root@connect.cqa1.seetacloud.com`。W0 在该实例重新运行完整旧开发集：总严格 EM **20/32**、Goal G 回归 **7/16**、无效动作 **0**，与冻结起点一致。实际 adapter、基础权重文件及 tokenizer 的 SHA-256 分别重算为 `e1680c2b2a6a60ec89b4b54249e3132e2005ee6eb94bb04e1acdbd4bf76e064e`、`04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`、`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`，均与既有冻结配置相符。原始轨迹见 [`w0-old-baseline.jsonl`](w0-old-baseline.jsonl)，正式停止决定见 [`decision.json`](decision.json)。

## 一次性候选池与缺口

使用本地已有、校验哈希通过的 HotpotQA `distractor/train` 两个原始 Parquet 分片（共 90,447 行）。从原题问法识别共同属性问题，再按固定哈希顺序和类别配额形成训练 512、开发 128、确认 128 道**未审查候选**。每个 split 的独立语料均为 2048 篇原始自然段落，候选来源 ID、规范化问题和语料标题跨 split 不重叠。训练语料平均每篇约 75.5 个英文词；比 Goal V 的短摘录更接近真实检索输入。文档和候选哈希见 [`data/manifest.json`](data/manifest.json)。

旧封存集没有被打开。其 300 个来源 ID 和 600 个支持文档标题由公开训练源及既有选择算法重建，扫描数 648，与封存 manifest 的 ID 集 SHA-256 `c8f575d8d55cb32937d1b649213f81c20dbaaa007e87d291ef8c9d8c11e89432` 完全匹配。另按已有历史暴露索引排除相关来源、问题和标题；方法及输入哈希均在数据 manifest 中。

最初准备版本将稀少的隶属关系候选全部顺序分配给训练，开发与确认均为零，并错误计入一个未计划的第五类别。当时没有新模型输出。完整保留在 `data-attempt01/`，修正和两次 manifest 哈希见 [`pre-model-correction.json`](pre-model-correction.json)。正式版本先为四类在三个 split 预留名额，再填充余量；没有根据模型成绩改题或补采。

| 阶段 | 训练候选 | 开发候选 | 确认候选 |
|---|---:|---:|---:|
| 固定问题池 | 512 | 128 | 128 |
| 原题被规则归入“共同隶属关系” | 28 | 8 | 8 |
| 原答案字面完整出现于两篇支持文档 | 15 | 3 | 6 |
| 逐题宽松计入后，**可能**符合训练题义 | **至多 12** | 未逐题复核 | 未逐题复核 |
| 预设最低合格数 | **16** | 8 | 8 |

字面出现只是诊断，不能替代语义证据：合理改写可能不包含原答案的完整字面。决定性的是训练池 28 道隶属关系候选的**原题问法审查**：其中 16 道明确在问某个人、日期、职业、年龄、作品或其他关系；即使宽松地保留全部其余 12 道，它们仍未经过真实检索及负答案审查，数量也已经低于 16。逐题问题与排除理由见 [`affiliation-question-audit.jsonl`](affiliation-question-audit.jsonl)。各 split 的聚合字面检查见 [`data-only-feasibility-audit.json`](data-only-feasibility-audit.json)，确认集个别题目未进入训练或模型评价流程。

因此不进行候选模型采样、128 对构造、四臂训练或确认集解封。按既定规则不降低类别配额，也不在看过缺口后追加候选池。基础设施和 GPU 可用；停止原因是**原题问法与计划中的训练目标不匹配**，不是训练数值或显存故障。

## 复算与后续判断

独立运行入口提供 `prepare`、`run`、`summarize`，已在远程依次执行，最终状态为 `complete_data_shortfall`，`gpu_training_started=false`。本地 CPU 检查通过 2 项，覆盖三个 split 的标题／问题／来源隔离及停止判据；数据聚合脚本另保存自身和输出哈希。复算命令：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_w_transfer.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_uv_selected.py
```

要继续追求更高的产品搜索成功率，下一轮须**重新立项**并在读取任何新模型输出前解决数据定义：可以从明确问“双主体共同属性”的问题池构造新问题，或换用确实能达到覆盖要求的四类属性。需要重新冻结来源、问题和盲审规则，再评估真实检索能否找齐证据。本轮不能据此判断 128 对训练会提升或损害模型，也不支持直接扩大到千条级训练。
