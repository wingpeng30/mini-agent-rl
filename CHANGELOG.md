# 更新日志

本项目使用[语义化版本](https://semver.org/lang/zh-CN/)：

- **主版本**：出现不兼容 API 改动时递增；
- **次版本**：新增向后兼容的功能时递增；
- **修订版本**：修复向后兼容的问题时递增。

每个提交到 GitHub 的可用阶段都必须：更新 `pyproject.toml` 版本、补充本文件，并创建对应的 Git tag `vX.Y.Z`。

## [0.7.3] - 诊断修正与 SFT/GRPO 开发集对照

- 保留历史严格 exact match，新增规范化 exact match、token F1、指标分母和逐 rollout 失败分类。
- 本地模型记录原始/实际 prompt token 数及是否发生上下文截断。
- 新增 `analyze-rollouts` 与 `evaluate-matrix`；矩阵评测固定 100 条开发集，顺序比较 v0.5.2 SFT 和三个 v0.7.1 GRPO adapter。
- 不修改 reward 权重、不读取 v0.7.2 的 300 条 final-test、不根据本轮开发集结果再次选择 checkpoint。
- 完成四模型 100 题开发集复跑：SFT EM `13%`、重复搜索 `61%`；三个 GRPO EM 为 `37%/45%/44%`，重复搜索为 `19%/32%/24%`，全部基础设施失败率为 0。
- 诊断表明规范化 EM 与严格 EM 完全一致，prompt 截断仅 `2%–3%`；GRPO 的主要收益来自提高 answer completion，空检索和证据覆盖仍是后续瓶颈。

## [0.7.2] - 最终测试集隔离与评测协议

- 新增 `prepare-final-test`：以流式 HotpotQA 构建与 v0.4.1 train/validation/test 的 task ID 和 supporting-document 标题均无重叠的 300 条最终集。
- 新增 manifest（来源 split、内容与任务顺序 SHA256）和 `audit-final-test`，用于验证最终集没有泄漏。
- 新增必须传入 `--confirm-final` 的 `evaluate-final`，在运行前再次执行隔离审计，降低将最终集用于调参的风险。
- 将 `data/hotpot-agent-v041/test.jsonl` 明确定义为开发评测集；最终集仅在 checkpoint 冻结后运行一次。
- 三个 seed 的 30 题开发评测完成并按预注册规则选择 seed 44；唯一一次 300 题 final 得到 EM `27.3%`、answer completion `67.0%`、mean reward `0.2847`，基础设施失败为 0。

## [0.7.0] - 最小 LoRA-GRPO

- 新增同一模型实例内 sampled rollout、old/current/reference logprob 重算、clipped GRPO loss、KL penalty 和 LoRA-only 更新。
- 训练前强制校验策略指纹；零方差 group 与基础设施失败轨迹跳过更新，策略负轨迹仍参与训练。
- 新增 `train-grpo` 5×4 单卡冒烟命令，保存 adapter、每组 loss/KL/clip fraction/grad norm 与显存。
- 真实 5×4 冒烟：4 个非零方差 group 完成、192 个 LoRA 参数变化；最后一组峰值分配显存 `8.54 GB`，故按停止规则不扩大训练或运行 30 条 A/B。
- 增加 gradient checkpointing、输入梯度钩子和 1024 token 训练截断；优化复跑峰值降至 `2.73 GB`，满足 RTX 4060 8GB 的显存门槛。
- 30 条 holdout greedy 对照通过最低门槛：GRPO adapter repeated search `23.3%`（v0.5.2 为 `56.7%`）、invalid action `13.3%`、mean reward `0.469`、execution success `100%`；结果仅作为小样本观察。
- 完成 10×4 正式采集与 100 条 holdout：v0.7.0 exact match `34.0%`、mean reward `0.5147`、repeated search `0%`、execution success `100%`；v0.5.2 对照分别为 `14.0%`、`-0.1528`、`59.0%`、`100%`。结果固定单 seed，需后续独立复现。

## [0.7.1] - 训练正确性修正与审计冒烟

- current logprob 在关闭 dropout 的 `eval()` 状态下保留梯度计算，避免预更新 ratio 受到 LoRA dropout 随机扰动。
- AdamW 跨 group 持续复用，reference 改为训练起点 LoRA 快照，避免将基础模型 KL 误当作起点策略约束。
- train-grpo 支持独立 SQLite、报告路径；报告记录 reference mode 和正式运行产物。
- v0.7.1 5×4 审计冒烟：reference KL `0.00–0.03`、clip fraction `0–3%`、峰值显存 `7.88 GB`，192 个 LoRA 参数发生变化。
- 增加 seed `43/44` 复现检查；三个 seed 的最大 KL 均低于 `0.03`，最大 clip fraction 均低于 `3.2%`，峰值显存约 `7.9 GB`。

## [0.6.1] - Logprob 稳定性诊断

- 新增 `validate-logprob`：比较同实例 BF16、跨加载 BF16 与跨加载 FP32 的 token 级差异和 policy ratio。
- 新增模型、adapter、tokenizer、软件与硬件策略指纹，写入 rollout group 和导出的轨迹。
- 训练前从未更新策略重算 old logprob，SQLite 数值只用于审计。

## [0.6.0] - Anti-Repeat RL 轨迹闭环

- 策略失败不再丢弃：非法 JSON、重复搜索和最大步数均保存为可学习的负轨迹。
- 新增 teacher-forced action token logprob、action mask、termination reason、eligibility 和 group-level advantage。
- SQLite schema 升级至 v2，JSONL 导出包含 group、rollout 与完整 transition。
- 新增 `collect-rl` CLI，支持 sampled rollout、硬超时和本地报告。
- 真实 1×4 冒烟：离散动作可复现、logprob 长度完整；独立 GPU 加载之间仍存在最大 `0.0745` 的 BF16 logprob 差异，因此按停止规则未扩大到 5×4 或 30 条实验。

## [0.5.2] - 完整 LoRA-SFT 基线

- `train-sft` 增加固定 `--seed`，训练指标记录 seed、loss、验证 loss、耗时和显存。
- 完成 600 条 Hotpot 训练样本的 1 epoch、75 step LoRA-SFT：训练 loss `0.2489`、验证 loss `0.1961`、峰值显存 `4.116 GB`。
- 30 条快速 holdout 评测发现 adapter exact match `10.0%`，高于基础模型 `3.3%`，但重复搜索率达到 `56.7%`，执行成功率降至 `16.7%`。
- 因快速评测出现明显策略退化，暂不运行完整 100 条评测；该结果作为下一轮奖励和 anti-repeat 设计的失败基线。

## [0.5.1] - LoRA Adapter 对比评测

- 新增 `compare-adapter` CLI：在相同的离线 Hotpot 检索环境中顺序评测基础模型与 LoRA adapter。
- 支持从 SFT 测试 JSONL 恢复 task 与 supporting-document 语料；gold answer 仅用于 reward，不传入模型或 SearchTool。
- 本地 8 题 holdout 结果：exact match 从 `0.0%` 提升至 `12.5%`，平均 reward 从 `0.006` 提升至 `0.178`。
- 新增 adapter 加载路径和原始模型输出记录，便于分析 JSON 格式失败与重复检索。

## [0.4.0] - 离线 SFT 数据生成

- 新增 `generate-sft` CLI，从离线语料确定性生成 train/validation/test JSONL。
- 样本包含 search、tool observation、answer 完整动作轨迹。
- 增加消息角色、动作 JSON、gold metadata 的质量校验。
- 生成过程不调用 API、不读取密钥、不下载模型。

## [0.4.1] - 公共多跳检索数据导入

- 新增 `download-hotpot`，通过流式 HotpotQA 导入构造多跳 search→observation→answer 轨迹。
- 仅保留 supporting documents，并按文档标题的稳定哈希进行 train/validation/test 切分，避免支持文档泄漏。
- 输出 manifest，记录来源、许可证、扫描量、样本数和文档数。

## [0.5.0] - Qwen3.5 LoRA-SFT

- 新增 Qwen chat-template assistant-only loss mask，排除 system/user/tool observation token。
- 新增文本侧 LoRA 配置，冻结视觉塔并训练 Qwen3.5 语言模块。
- 新增 `train-sft` CLI，保存 adapter、训练状态和显存指标。
- 已完成 RTX 4060 8GB 的 10 step 冒烟验收：训练 loss `0.4313`、验证 loss `0.2847`、峰值显存 `4.116 GB`。

## [0.3.1] - Qwen3.5-0.8B 本地推理基线

- 支持本地 Qwen3.5 多模态架构的 `AutoProcessor` 与 `AutoModelForMultimodalLM` 加载路径。
- 固定 Transformers Qwen3.5 兼容 commit `5474a55e920f358d8382f3ecd3377edca979baa1`。
- 增加本地文件模式、非思考模式、GPU 推理锁、动作原文记录和 `local-smoke` 命令。
- 扩展 Base 评测指标：合法动作率、工具成功率、重复搜索率、平均步数、加载耗时和峰值显存。

## v0.3.0 - 本地模型基础框架

- 新增懒加载 `TransformersModelClient`，在首次推理前不加载、不下载模型。
- 新增 Qwen2.5-0.5B 默认配置、依赖/显卡预检和 `local-check` 命令。
- 定义严格的 `search`/`answer` JSON 策略动作协议。
- 保存本地模型 prompt/action token IDs 与 usage；工具 observation 继续排除在策略动作之外。
- 新增 SFT 消息数据导出边界及本地模型单元测试。
- 忽略模型权重和 checkpoint，避免误提交大文件。

## [0.2.0] - 2026-09-12

### 新增

- 离线 benchmark 与 `mini-agent-rl evaluate` 命令。
- DeepSeek/OpenAI-compatible 可选后端与 Tool Calling 消息协议。
- 模型请求、奖励和生命周期事件；rollout 并发上限。
- 工具参数基本 schema 校验和结构化 JSON observation。

### 验证

- 离线 benchmark：执行成功率 100%，答案准确率 87.5%，平均 reward 1.045。

## [0.1.0] - 2026-09-12

### 新增

- 初始 RL-ready Agent 运行时、离线搜索工具、FakeModel、SQLite 轨迹存储与 JSONL 导出。
# v0.8.0

- 新增 RL 学习信号只读审计命令 `audit-rl-signal`。
- 增加受控 reward 配置和 rollout-equal GRPO loss 接口。
- 补充 v0.8.0 审计与受控实验文档。
