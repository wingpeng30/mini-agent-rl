# Goal G — 共同类别评价题 CPU 准备结果

**状态：完成数据准备；16 条合格评价题已冻结。** 目标为 20 条，但固定的 100 条全文检查预算已用尽，按预注册规则停止。没有加载模型权重、生成答案、训练或使用 GPU。结果用于后续固定证据输入下的小规模探索，不是端到端检索成绩，也尚未进行任何模型比较。

## 处理路径与数量

| 阶段 | 结果 |
|---|---:|
| 官方 HotpotQA distractor/train 元数据（仅 ID、题面、类型） | 90,447 条 |
| 冻结关键词定位 | 10,004 条 |
| 已有项目暴露排除后进入哈希排序 | 8,334 条 |
| 按固定顺序打开答案与支持证据 | 100 条，达到上限 |
| 结构筛查后待语义判断 | 50 条 |
| 逐题能力与证据筛查后待审 | 19 条 |
| 实际训练编码与 tokenizer CPU 预检 | 19/19 通过 |
| DeepSeek 有效批准 | 19/19 |
| 隔离 Luna 有效批准 | 18/19 |
| 祖先训练／开发题源实体重叠 | 2 条 |
| **双审、隔离及 CPU 均通过** | **16 条** |

50 条结构候选中的 31 条因任务不符、证据不足或答案粒度错误排除。Luna 否决的 1 条是运动类别的证据粒度不清；另 2 条虽获两方批准，其支持文档实体已出现于 v118 祖先模型使用的 v041 题源，按预定隔离规则排除。没有重审有效否决，也没有因不足 20 条而扩池。

## 审核与 CPU 检查

两方看的是相同的匿名包，没有来源 ID、历史成绩和对方投票。DeepSeek 使用已配置的官方端点，返回模型身份与请求的 `deepseek-flash` 相同；四个一次性小批次全部获得可机械校验的票。AutoDL 未配置审核密钥，因此 DeepSeek 请求只在本地通过既有受限配置发出；密钥没有显示或上传。Luna 在无历史上下文的 GPT-6 Luna 子会话完成独立审核。

Luna 原票有 3 条批准票的引用文档 ID 错误，其中 1 条还存在引文字符差异。原票保留；同一审核员对这 3 条各作一次**仅限引用的技术更正**，语义决定及理由未变。更正后，所有批准票均有两份可见文档中的逐字引文。DeepSeek 的 19 条批准票也均分别引用两份文档。

首次 CPU 预检的 19 条全部触发检查器边界错误：错误地把答案两侧的 JSON 引号当作答案值。修复检查器后只重检这 19 条，全部通过。原失败记录保留；实际训练编码、生成前缀、目标 assistant、因果标签、padding、结束符、证据可见性和 2048／128／2176 token 限制均已检查。7 个针对正常输入、错误目标、证据缺失、超长、答案泄漏、JSON 转义与跨边界的回归场景通过。

## 隔离范围与产物

从 v118 实际训练清单及 Goal C/D 训练开发索引核对题面与实体；从 v117、v122、v123、v130 和 Goal C 的既有索引核对来源 ID 与已筛查问题；补核 v118 继承的 v041 train／validation 题源。两条祖先实体冲突已排除。`evaluation_isolation_verified=true` 的含义是**这些已核实的项目内训练、开发和筛查范围**；不声称基础模型预训练从未见过这些题。评价输入采用 Goal E 对齐的两轮合成搜索历史，不能解释为真实检索性能。

主要交付文件：

- `PROTOCOL.md`：读取答案前冻结的单能力规范和停止规则。
- `locator-summary.json`、`screen-ledger.jsonl`、`semantic-ledger.jsonl`：元数据定位、100 条全文检查及逐题筛查依据。
- `ancestry-audit.json`：祖先题源的逐题隔离核查。
- `review-packets.jsonl`、`deepseek/`、`luna-votes.json`、`luna-votes-technical-correction.json`、`vote-disposition.jsonl`：匿名双审原票、更正及最终处置。
- `cpu-preflight.jsonl`、`cpu-preflight-v2.jsonl`：修复前后逐题 CPU 记录。
- `evaluation-frozen.jsonl`：16 条正式评价题，保留原题、官方答案、可见证据及合成上下文。
- `freeze-manifest.json`：白名单输入、代码、tokenizer、票据和产物 SHA-256 及状态位。清单 SHA-256 为 `76605092f8582111093138ba14c31204b3303ffe8025d87a3dce2dd3fa661c36`。

产物主目录为 AutoDL 的 `/root/autodl-tmp/mini-agent-rl/runs/v133-goal-g-cpu/`，并镜像必要文件到本地同名目录；镜像的正式评价集与冻结清单哈希已核对一致。实际修复后 CPU 预检时的运行器 SHA-256 是 `d6389d68f57fa61a3cd24ce423bec944350a89ac6f347998637aa8179d95dc4d`；随后只增加审核包构建功能的最终运行器哈希由冻结清单记录。CPU 回归脚本 SHA-256 是 `64d90c9adab1e256b2dcc99ec1bbe5dace401d1c23a405023d68cbd36aa3019a`。

在具有相同白名单输入和 tokenizer、且产物目录尚不存在的 AutoDL 工作副本中，CPU 阶段命令依次为：

```bash
cd /root/autodl-tmp/mini-agent-rl
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/run_v133_goal_g_cpu.py locate
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/run_v133_goal_g_cpu.py screen
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/run_v133_goal_g_cpu.py adjudicate
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/run_v133_goal_g_cpu.py preflight
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q tests/test_v133_goal_g_cpu.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python scripts/run_v133_goal_g_cpu.py review-packets
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m scripts.audit_v133_goal_g_ancestry
```

双审后运行 `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m scripts.finalize_v133_goal_g_cpu`。这些是一次性冻结命令；现有产物目录故意拒绝覆盖。下一步如启动模型比较，须在同一批 16 题上固定模型、证据与解码配置，预先说明它是小样本探索；本轮未启动该阶段。
