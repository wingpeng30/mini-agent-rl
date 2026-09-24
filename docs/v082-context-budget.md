# v0.8.2 上下文预算与 8GB 显存实验

## Material Passport

- 实验 ID：`mini-agent-rl-v082-context-budget`
- 状态：`STOPPED_BY_GATE`
- 起点：`checkpoints/qwen35-08b-sft-v052`
- 硬件：RTX 4060 Laptop，8188 MiB
- 数据：本地 HotpotQA validation/train 子集
- 外部传输：无

## 实现成果

本轮将“保留最后 N 个 token”改为结构化预算。system 动作协议和 user 问题始终完整保留；超限时先按检索相关度顺序缩短 observation 文本和文档数，再从最旧的 assistant/tool 组开始删除，最近一组环境状态仍保留。裁剪逻辑不读取 gold answer、support titles 或 reward。

模型响应新增 context budget、各类消息 token 数、删除的 observation token 数、裁剪阶段和 prompt hash。采样、old/current/reference logprob 都调用相同预算函数，训练报告按阶段分别统计截断。

GRPO loss 同时修正为：先按同一 rollout 内所有 assistant action token 聚合，再对 rollout 等权平均。报告新增 allocated/reserved 显存、action length 相关性和 reward component 覆盖。RL 信号审计会重算 reward 总和、group mean/std、advantage 均值及即时奖励标签。

## 实验结果

固定 10 条 validation 任务、同一 SFT checkpoint 和 greedy 解码：

| 输入预算 | 采样截断率 | EM | Token F1 | 重复搜索 | 峰值显存 | 门槛 |
|---:|---:|---:|---:|---:|---:|---|
| 384 | 45% | 20% | 0.267 | 0% | 1.928 GB | 通过推理门槛 |
| 768 | 10% | 40% | 0.467 | 0% | 3.301 GB | 通过推理门槛 |
| 1024 | 0% | 40% | 0.467 | 0% | 3.688 GB | 通过推理门槛 |
| 2048 | 0% | 40% | 0.467 | 0% | 3.688 GB | 通过推理门槛 |

这 10 条样本中，768 已达到 1024/2048 相同的离散结果；该观察只用于长度选择，不代表泛化性能。

4-bit 单 transition backward 探测均通过：384/768/1024 的峰值 reserved 分别为 `1.289/1.438/1.438 GB`。768 和 1024 使用的探测 prompt 实际均为 503 token，因此该探测不能代表完整多 rollout 更新的最坏显存。

768-token 5×4 冒烟通过：采样与三类训练 logprob 截断率均为 0%，3/5 group 有非零方差，192 个 LoRA 参数更新，峰值 reserved `7.123 GB`，checkpoint 可重新加载。

768-token 10×4 正式实验完成 40/40 rollout，但触发两个停止条件：

- rollout sampling 截断率 `10.39%`，高于 `5%` 门槛；old/reference/current logprob 截断率约 `10.81%`。
- 峰值 reserved `7.703 GB`，高于本轮 `7.4 GB` 安全门槛。

其余审计通过：5/10 group 有非零方差，192 个 LoRA 参数更新，基础设施失败为 0，reward、group statistics、advantage 和 action/logprob/mask 长度均一致。正式 checkpoint 仅作为失败实验产物保留，不进入 100 条开发集评测。

## 修订结论与边界（v0.8.3）

本机适合 768-token 的小规模冒烟；10×4 正式运行完成但未通过预先定义的采样截断门槛。原报告的 reserved 峰值 `7.703 GB` 超过 `7.4 GB` 保守线，但 reserved 是 PyTorch 缓存统计，不能直接解释成活动张量实际占用超过 8GB，也不能证明高显存设备没有价值。

因此暂缓租用设备的理由是先排除可修正的系统问题：768/1024 的单步 backward 探测实际输入都只是 503 token；训练循环还存在 rollout 内计算图累计空间；推理扫长使用 BF16、训练使用 4-bit，且历史版本还改动了动作历史表达。上述因素使得“长度、reward 或显存导致效果变化”的因果归因尚不成立。

本轮事实包括：正式运行约 294 秒、冒烟约 121 秒，并非因耗时停止；正式训练未进行独立 checkpoint 重载，未启动 100 条开发集对照；原始正式报告也未把显存作为实时中止 gate。它们均不支持“训练提高模型能力”或“已经实时安全拦截”的说法。

按实验解释风险检查：样本量只有 10 条，存在小样本不稳定与选择偏差；只报告完成 rollout 会有幸存者偏差；已探索多个长度和训练条件，不能把单一结果当确认性结论；本实验是系统比较而非证明上下文长度或显存造成性能变化。下一版会先验证重放和梯度等价，再以统一量化配置做近满预算压力测试。

本轮不读取封存 final-test，不创建 `v0.8.2` 标签。下一步是 v0.8.3 的可信 harness：固定预算后 token 重放、流式 rollout 等权反传、真实 reward 审计、逐 group gate 和可迁移运行配置；不通过这些工程验证前，不扩大训练或租用设备。

## 复现命令

```powershell
conda activate mini-agent-rl
mini-agent-rl context-length-sweep --lengths 384,768,1024,2048
mini-agent-rl probe-context-backward --max-input-tokens 768
mini-agent-rl train-grpo --load-in-4bit --max-input-tokens 768 --task-limit 5 --group-size 4
```
