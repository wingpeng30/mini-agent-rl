# v0.7.2：最终测试集与 checkpoint 选择协议

## 固定数据

2026-09-14 已生成 `data/hotpot-agent-v072/final-test.jsonl`。该文件为 300 条 HotpotQA `distractor` 样本；它与 `data/hotpot-agent-v041` 的训练、验证和开发评测数据没有 task ID 或 supporting-document 标题重叠。

- 最终文件 SHA256：`85cd82152fbce1a567311b27587111ce98a3d3a51f1a5ae088851430c3bc628c`
- 任务顺序 SHA256：`c8f575d8d55cb32937d1b649213f81c20dbaaa007e87d291ef8c9d8c11e89432`
- 审计结果：300 条、重复 task ID 为 0、task 重叠为 0、supporting-document 重叠为 0、manifest 哈希匹配。

源文件哈希、许可证和扫描数量记录在本地忽略文件 `data/hotpot-agent-v072/manifest.json`；完整审计在 `reports/final-test-audit-v072.json`。这些产物不提交 Git，以免最终题目随仓库公开。

## 模型选择

`data/hotpot-agent-v041/test.jsonl` 自本版本起称为开发评测集。候选 adapter 可以在它上面多次比较；最终集绝不用于决定超参数、seed 或 checkpoint。

候选 checkpoint 的预注册选择规则为：在相同的前 30 个开发任务、greedy 解码、最大 3 步下，优先选择平均 reward 最大者；若相同，依次比较 exact match、较低 invalid action rate、较低 repeated search rate。选择决定写入本文件和任务成果后，冻结该 checkpoint 的目录及 adapter SHA256。

运行命令：

```powershell
mini-agent-rl evaluate-adapter --adapter-path checkpoints/qwen35-08b-grpo-v071-seed42 --data data/hotpot-agent-v041/test.jsonl --limit 30 --db runs-v072-dev-seed42.db --output reports/dev-seed42.json
```

完成三组 seed 后，只对被选中的 checkpoint 运行一次：

```powershell
mini-agent-rl evaluate-final --adapter-path <已冻结的目录> --confirm-final --db runs-v072-final.db --output reports/final-v072.json
```

`evaluate-final` 会再次执行隔离审计。无 `--confirm-final`、审计失败或把最终集误传给 `evaluate-adapter` 均会拒绝执行。

## 已冻结的选择（2026-09-14）

三个 v0.7.1 候选在相同 30 条开发任务上的结果如下。它们均没有基础设施失败。

| checkpoint | 平均 reward | exact match | invalid | repeated search |
| --- | ---: | ---: | ---: | ---: |
| `qwen35-08b-grpo-v071-audit-smoke`（seed 42） | 0.5620 | 46.7% | 10.0% | 13.3% |
| `qwen35-08b-grpo-v071-seed43` | 0.6007 | 50.0% | 0.0% | 26.7% |
| `qwen35-08b-grpo-v071-seed44` | **0.6143** | **53.3%** | 3.3% | 23.3% |

因此依照预注册的“平均 reward 优先”规则，冻结 `checkpoints/qwen35-08b-grpo-v071-seed44`。其 `adapter_model.safetensors` SHA256 为 `76f61ebd2313b6939075d7767e3b424a04f2007ff1003e08b41d5334c062d90f`。开发报告分别保存在本地忽略目录 `reports/dev-v072-seed42.json`、`dev-v072-seed43.json` 和 `dev-v072-seed44.json`。

## 最终评测状态

唯一一次 300 题最终评测已完成：exact match `27.3%`、answer completion `67.0%`、mean reward `0.2847`、repeated action `26.0%`、invalid action `6.7%`、基础设施失败 `0`。完整解释见 [v0.7.2 最终评测分析](final-evaluation-v072.md)。该 final set 自此封存，不用于后续调参。

## 解释边界

最终集只评估当前固定的离线 SearchTool、动作 JSON 协议和最多三步的运行时。它能衡量该实验设置下的泛化，不证明模型在完整 HotpotQA、开放网络检索或真实生产代理场景中的表现。
