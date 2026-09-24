# v0.8.3：可信、可迁移的 RL Harness

## Material Passport

- 实验 ID：`mini-agent-rl-v083-harness`
- 状态：`LOCAL_SMOKE_VERIFIED_FORMAL_EXPANSION_STOPPED`
- 代码版本：`0.8.3`
- 目标：先验证训练信号与运行边界，再迁移到单卡 24GB Linux GPU。
- 最终测试集：封存；本版本不读取 `data/hotpot-agent-v072/final-test.jsonl`。

## 本轮实现

- 采样响应保存预算后 `prompt_token_ids` 与 attention mask；old/current/reference logprob 直接重放该输入并核验哈希。
- GRPO 改为逐 transition 加权反传：同一 rollout 内仍按 action token 聚合、不同 rollout 等权，但不再同时保留多个计算图。
- KL 修正为 current 相对 reference 的非负近似项；此项属于训练语义修正，不与历史指标混比。
- reward 配置随新 rollout 持久化。审计对缺少该配置或 prompt token 的旧数据库显示 `unverifiable`，而不是误报通过；审计数据库以 SQLite read-only 模式打开。
- 新增 `mini-agent-rl experiment preflight/run/inspect`。TOML 冻结命令和输入，独立目录写入 manifest、状态和 stdout/stderr；仅硬超时自动终止，不自动重试。

## 迁移边界

`configs/v083-local-8gb.toml` 是 8GB 冒烟配置；`configs/v083-single-gpu-24gb.toml` 是 Linux 单卡 parity 模板。云端第一轮保持 4-bit、768 token、相同 adapter、数据和 seed；只有 parity 通过后才复制配置并单独测试 1024/2048 token。

Linux 配置尚未执行，属于未验证能力。模型目录可以映射为 `/models/qwen_08b`，但必须核对模型与 adapter 内容哈希；跨 kernel 仅比较带容差的 logprob、逐题结果和完整 manifest，不要求字节级生成一致。

## 本地验收顺序

1. 对既有 v0.8.2 SQLite 做只读审计；历史缺失 provenance 应被明确标为不可验证。
2. 运行纯函数/CPU 测试，验证流式反传与聚合目标的梯度等价。
3. 使用 4-bit 和真实三步轨迹进行 384、768 近满预算压力探测，包含 old/reference/current、两轮 policy epoch 和 optimizer 状态。
4. 仅当输入重放、数值、LoRA-only、重载、显存和 group variance 都通过时，运行 5×4 冒烟；旧口径采样截断率不高于 5% 才能扩大为 10×4。

本版本不自动开始 100 条开发集评测，不创建 `v0.8.3` tag。训练效能是后续受控实验问题，不是本 harness 交付的验收条件。

## 本次验证记录

- 已完成 Python AST 解析、模块导入、CLI 子命令帮助与本地 8GB TOML 预检；预检确认不会读取封存 final-test，且目标运行目录尚不存在。
- 已对 `runs-v082-context768-formal.db` 执行只读审计：40 条 rollout、10 个 group、5 个非零方差 group、无基础设施失败；全部 40 条缺少 `reward_config`，因此 `reward_recompute_consistent=false`、总 gate 为不通过。这是历史证据边界的报告，不代表数据被损坏。
- `tests/test_logprob_grpo.py` 的 7 项纯函数测试已通过。完整 pytest 在此机器的自动权限审批阶段两次超时，且 Windows pytest 临时目录 ACL 返回 `WinError 5`；未把它解释为测试失败。

## v0.8.3 本机实验记录（2026-09-16）

本节是本轮真实运行记录；它不追溯改写 v0.8.2 的原始报告，也不读取封存最终集。

### 修正前的可观测性问题

第一次 768-token 正式扩展发现：`SearchTool` 的执行耗时 `latency_ms` 被拼入下一轮模型上下文。该值与机器负载有关，虽然不携带任务语义，却会改变 token 边界并使临界样本的截断状态随运行波动。现已修正为：完整 observation（含耗时）只保存在 transition 供审计；模型可见 observation 删除 `latency_ms`。该修正不改变工具结果、奖励或 gold 边界。

### 已完成的门槛试验

| 配置 | 结果 | 事实解释 |
|---|---|---|
| 384 token，1 task × 4 rollout | `GATE_FAILED`，采样截断率 20%，reserved 峰值 1.303 GiB | 不是显存问题；384 对该任务的输入预算不足。 |
| 768 token，5 task × 4 rollout | `COMPLETED`，采样截断率 0%，4/5 group 有非零 reward 方差，reserved 峰值 3.504 GiB | 流式 GRPO、精确 prompt 重放、LoRA-only 更新和数值门槛均通过。 |
| 768-token smoke checkpoint 重载 | 成功；1 条 validation rollout 得到 EM 100%、reward 1.370 | 只证明产物可加载与单条链路可执行，不证明质量提升。 |
| 768 token，目标 10 task × 4 rollout | 在第 2 个 group 停止，`GATE_FAILED`；当组采样截断率 20%，reserved 峰值 1.363 GiB | 未满足旧口径采样截断率不高于 5% 的扩大条件；只保留 metrics 审计目录，不保存可评测 adapter、未运行 100 条开发集评测。 |

该正式扩展的只读审计已通过：4 条 rollout、1 个 group，token/logprob/action-mask 长度一致；即时奖励与 rollout 奖励、group advantage 均可核验；无 NaN/Inf、无基础设施失败、无 observation token 进入 policy loss。它是截断 gate 的失败样本，而不是训练基础设施失败样本。

### 当前决定与边界

- 本机 8GB 已足以验证 harness 的训练正确性、显存 gate、checkpoint 重载和失败停止语义；当前不因“算不动”而必须租用设备。
- 但 768-token 的 10-task 输入分布仍出现超预算样本。下一步应先分析并改进 observation 的预算策略/数据长度分层，再在固定配置重跑；不能靠放宽 gate 或静默删除证据来取得通过。
- 本轮没有运行 100 条开发集质量对照，不创建 `v0.8.3` tag。Linux 24GB 配置仍是未验证的迁移模板。
