# v0.7.1 训练正确性审计

## Material Passport

- 审计 ID：`mini-agent-rl-v071-training-audit`
- 日期：2026-09-13
- 审计对象：`checkpoints/qwen35-08b-grpo-v070-formal`
- 基础模型：`D:\qwen_08b`（Qwen3.5-0.8B）
- 外部传输：无；所有检查都在本地模型、SQLite、checkpoint 与 JSON 报告上完成。
- 结论状态：`AUDITED_WITH_CORRECTIONS_REQUIRED`

## 已证实的事实

1. 正式 checkpoint 存在，且 `grpo_metrics.json` 记录 10 个 group、192 个发生变化的 LoRA 参数；其中 8 个 group 更新，2 个因 reward 零方差跳过。
2. 重新针对正式 adapter 执行 `validate-logprob --repeats 10`，同实例、BF16 跨加载与 FP32 跨加载均通过当前命令门槛。报告中的正式 adapter 权重 SHA256 为 `56e47a2059d5efe3e8464d3d044a660edbddc3d8e04d9186d844dd1750c3873b`。
3. `grpo-v070-smoke.db` 保留了 20 个 group、80 条 rollout、178 条 transition 和 521 条 event；不存在 `model_error` 或 `tool_error` 终止。该数据库包含多次训练运行，不能把其整体终止比例归因于某一次正式训练。
4. 正式训练的 group metrics 中 KL 处于约 `2.51` 到 `222.80`；最大 KL group 的最后 epoch loss 为约 `4.51`，而 `0.02 × 222.80 ≈ 4.46`，说明该 group 的总 loss 基本由 KL 项决定。
5. 已存在的 `adapter-comparison-v070-30.json` 与 `adapter-comparison-v070-100.json` 文件实际仍写有 `version: v0.5.2`，且没有 `adapter_path` 或策略指纹。它们产生于 CLI 修正报告字段之前，不能单靠文件内容证明所使用的 adapter。
6. `base-comparison.db` 与 `adapter-comparison.db` 是固定文件名，已累计超过一次评测的 rollout。因此数据库是调试历史，不能作为某一个报告的唯一轨迹快照。

## Dropout 诊断实验

对正式 adapter 的同一条 17-token action，使用相同消息和权重进行只读前向：

| 比较 | 结果 |
|---|---:|
| `eval()` old 与 `train()` current 最大 logprob 差 | `0.06139` |
| 对应 policy ratio 最小值 | `0.94045` |
| 对应 policy ratio 最大值 | `1.04205` |
| 五次 `train()` 前向之间最大差 | `0.07449` |

这是 LoRA `lora_dropout=0.05` 在训练模式启用的直接结果。它不表示权重损坏，但表示第一次 optimizer step 之前的 current/old ratio 已经不是严格的 `1`，会给 clipped objective 注入随机变化。

## 解释与限制

当前实现把 reference logprob 设为“临时禁用 adapter 后的 base model”，而非冻结的 SFT 起点 adapter。因此 KL 衡量的是当前策略与基础模型之间的距离；SFT adapter 本身已经偏离基础模型时，第一步 KL 就可能很大。这与代码设计一致，但它使 `kl_beta=0.02` 对小样本更新的约束强度难以解释。

`RolloutStatus.SUCCEEDED` 表示运行时正常收束，包含 `invalid_action`、`repeated_action` 和 `max_steps` 等策略终止。它不能被解释为回答完成或答案正确。正式评测应以 `termination_reason`、exact match、reward 与基础设施失败率分别报告。

此前 30/100 条结果仍可作为探索性观察，但由于报告缺少 adapter 指纹、评测数据库复用、且训练目标受 dropout 影响，它们不能作为严格可复现的性能结论。

## v0.7.1 修正清单

下一次训练前应完成以下修改并重新从 `qwen35-08b-sft-v052` 开始：

1. 在带梯度的 current logprob 前向中保持模型 `eval()`；eval 模式不禁止反向传播，因此可关闭 dropout 而保留 LoRA 梯度。
2. 将 optimizer 生命周期移到整个训练 run；不得在每个 group 重新创建 AdamW，以保留动量与二阶矩。
3. 将 reference policy 设为冻结的起点 SFT adapter，或显式提供 `reference_mode=base|initial_adapter` 并写入报告。实验默认应使用 `initial_adapter`。
4. 为每次 run 创建唯一目录，例如 `runs/v071/<timestamp>-seed42/`，写入 manifest：起始/最终策略指纹、数据文件 SHA256、任务 ID 列表、seed、配置、命令、退出状态、SQLite 路径与报告路径。
5. 每个评测报告写入 adapter path、adapter weight SHA256、data SHA256、任务 ID 哈希与创建时间；每次评测使用独立 SQLite 文件。
6. 把当前 100 条集合固定为开发评测集，建立一个未参与调参的最终测试集合；至少运行 seed `42/43/44`，报告均值、标准差和逐题终止原因。

在这些修正完成并通过 5×4 冒烟前，不扩大 GRPO 训练预算。

## 修正后验证

已从 `checkpoints/qwen35-08b-sft-v052` 重新运行 5×4，使用独立数据库 `runs-v071-audit-smoke.db` 和报告 `reports/grpo-v071-audit-smoke.json`。结果为：1 个零方差 group 跳过，4 个 group 完成更新，192 个 LoRA 参数发生变化；各 group 最后 epoch 的 KL 为 `0.00/0.01/0.02/0.03`，clip fraction 约 `0–3%`，loss 与梯度均为有限值，峰值显存最高 `7.88 GB`。

这次结果支持两项修正：关闭 dropout 后更新前 ratio 不再被训练模式随机扰动；使用起点 adapter 快照作为 reference 后，KL 不再被基础模型与 SFT adapter 的固有差异放大。该冒烟仍不是多 seed 性能结论。

## 三个 seed 的复现检查

在相同 5×4 预算下补跑 seed `42/43/44`，每次从相同 v0.5.2 adapter 起点开始，并使用独立数据库与报告：

| seed | 有效 group | 平均最后 epoch KL | 最大 KL | 最大 clip fraction | 峰值显存 | 用时 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4 | 0.0170 | 0.0282 | 3.18% | 7.88 GB | 238 s |
| 43 | 5 | 0.0046 | 0.0082 | 2.10% | 7.89 GB | 1060 s |
| 44 | 5 | 0.0088 | 0.0218 | 2.19% | 7.89 GB | 838 s |

三个 seed 都完成了有限值更新，峰值显存稳定在约 7.9GB，KL 和 clip fraction 没有出现前一版的数十到数百级别。seed 之间用时差异较大，当前实现仍然是逐 transition、逐 token 的低吞吐审计实现；这组结果验证训练数值边界，不代表三 seed 的任务质量已经相同。下一步应在同一独立评测集上比较三个 checkpoint 的回答完成率、exact match 和终止原因。
