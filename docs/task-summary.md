# 本次任务成果

## 最新成果（2026-10-12）

以[当前项目状态](../PROJECT_STATUS.md)为准。固定0.8B的GPU批量生成与异步工具执行、继续SFT、在线GRPO双seed和裸模型评价全部完成，结果已取回校验并触发远端关机流程。

自然104题EM/F1：AM47/104、0.5907；继续SFT49/104、0.5855；GRPO42为40/104、0.5074，GRPO43为46/104、0.5847。全128题EM依次67/68/61/66。batch4吞吐2.78倍（12.71→35.34题/分钟），EM67→64。两RL seed各512条真实轨迹、108/110次更新，没有稳定质量提升。裸模型未输出最终答案，协议下EM0/128。

指标、adapter和简历说明见[最终结果解释](final-online-grpo-20261012-results.md)与[原始报告](../runs/final-online-grpo-20261012/results/REPORT.md)。下方为历史日志，“未实现”“下一步”等不表示当前状态，不据此重启旧实验。

## 历史任务日志

## v0.8.3：可信与可迁移 Harness（未发布）

本阶段先修订 v0.8.2 的实验解释边界，再实现 token 重放、流式 rollout 等权反传、reward provenance、只读审计和受监督实验接口。没有读取封存最终集、执行云端实验或宣称模型质量提高。

历史 v0.8.2 正式轨迹缺少完整 reward provenance，会被新审计标为 `unverifiable`；这不删除或改写历史轨迹。下一步是先做重放、梯度等价和近满预算压力探测，再决定是否启动新的 5×4 冒烟。

已完成独立的 Mini Agent RL 基础框架：离线 SearchTool、FakeModel、OpenAI-compatible 模型接口、单 Agent 多步循环、并发 rollout group、组合式 reward、SQLite 事件存储、CLI 查询和 RL-ready JSONL 导出。

未实现：本地模型下载、GPU 训练、GRPO/PPO 更新、Kubernetes、多 Agent、Web UI。

后续路线：增加真实 tokenizer/logprob、实现 group-level GRPO、加入 evidence/process reward、增加轨迹回放和成本约束实验。

## 第二阶段：可靠运行时与离线评测

本次任务新增 OpenAI-compatible Tool Calling 消息格式、DeepSeek 可选后端、JSON 工具观察、工具参数基本类型校验、模型/奖励/生命周期事件、并发上限和 `evaluate` CLI。新增 8 条离线检索 benchmark，并输出准确率、平均奖励、平均搜索次数与失败数量。

本阶段不调用或保存任何 API key；DeepSeek 密钥只能通过 `DEEPSEEK_API_KEY` 环境变量提供。后续接入可训练模型时，再以同一 rollout group 数据计算 group-level advantage 并实现 GRPO 更新。

### 本地验证结果

- `python -m compileall -q src` 通过。
- 1 条离线任务、1 个 rollout 的 smoke run 成功，最终 reward 为 `1.170`。
- 8 条 benchmark、每题 2 个 rollout 的离线评测：执行成功率 `100%`、答案准确率 `87.5%`、平均 reward `1.045`、平均搜索次数 `1.0`。
- 当前 Codex 沙箱对 pytest 临时目录存在 Windows 权限限制；常规本地环境可使用 `pytest` 运行全部测试。

## 版本记录

本阶段版本为 `v0.2.0`。项目从此遵循语义化版本：每次完成一个可用阶段并推送 GitHub 时，同步更新 `pyproject.toml`、`CHANGELOG.md`，并推送对应 Git tag。
# v0.3.1：Qwen3.5-0.8B 本地推理基线

本次完成 Qwen3.5-0.8B 的本地推理兼容骨架：使用 `AutoProcessor` 和 `AutoModelForMultimodalLM`，固定 Transformers 兼容 commit，启用本地文件模式和非思考模式，增加 GPU 推理锁、local-smoke 命令及 Base 评测指标。未实现 SFT、LoRA 或 GRPO。

验证结果：固定 Transformers `5.18.0.dev0` 能识别本地 `qwen3_5` 配置；RTX 4060 Laptop 8GB 成功加载本地权重并完成一次 rollout，状态为 `SUCCEEDED`，但答案奖励为 0.000，说明当前模型需要进一步的 prompt/SFT 才能稳定遵守检索问答任务。

随后新增 v0.4.0 SFT 数据基础：通过 `generate-sft` 从离线语料确定性生成 train/validation/test JSONL，生成前检查消息角色、动作 JSON 和 metadata，暂不调用教师模型 API。

v0.4.1 新增 HotpotQA 流式导入与文档级切分：用于提供真实的英文多跳检索监督样本，默认只保存小规模子集，不下载完整语料。

v0.5.0 新增 Qwen3.5 LoRA-SFT 训练能力：基于模型 chat template 计算 assistant-only loss，工具 observation 不参与监督；LoRA 仅注入语言模块，视觉塔保持冻结。为兼容 Qwen3.5 模板对消息首尾空白的规范化，训练编码器按完整渲染 token 序列定位内容，只标注 assistant 的 JSON 动作 token。

### v0.5.0 本地验收结果

- HotpotQA 导入子集（训练 600 条、验证 100 条）已完成全量编码预检；最长样本 871 token，`max_length=1024` 下无样本丢失，最少有 39 个 assistant 监督 token。
- RTX 4060 8GB 上完成 10 step LoRA-SFT 冒烟训练，训练耗时 87.14 秒，平均训练 loss 为 `0.4313`，验证 loss 为 `0.2847`，GPU 峰值显存 `4.116 GB`。
- 训练产物保存为本地忽略目录 `checkpoints/qwen35-08b-sft-smoke/`：包含约 12.8 MB 的 LoRA adapter、训练状态和 `training_metrics.json`；基础模型权重未被改写、不会提交 Git。
- 使用 `pytest --basetemp .test-tmp -p no:cacheprovider` 完成 18 项测试。此参数只规避当前 Windows 用户临时目录与 pytest cache 的权限问题，不影响测试语义。

### v0.5.1 Adapter 对比评测

本阶段新增固定环境的基础模型/LoRA adapter 对比。评测从 `data/hotpot-agent-v041/test.jsonl` 恢复问题及其 supporting documents；标准答案只给 RewardFunction 使用，不会出现在模型上下文或 SearchTool 语料中。

- 使用 8 条未参与训练的 holdout 题、greedy 解码、最多 3 步、同一离线语料。
- 基础模型：exact match `0.0%`、平均 reward `0.006`、执行成功率 `62.5%`、动作合法率 `62.5%`、非法 JSON 率 `37.5%`。
- LoRA adapter：exact match `12.5%`、平均 reward `0.178`、执行成功率 `62.5%`、动作合法率 `100%`、重复搜索率 `37.5%`。
- 结论：小规模 10 step adapter 已带来可测的格式遵循、回答与 reward 提升，但统计量非常小，且三条失败轨迹为重复搜索；不能据此宣称泛化能力。下一阶段应增加训练步数、独立验证集评测和失败类型分析。

完整机器可读报告为本地忽略文件 `reports/adapter-comparison-v051.json`；基础/adapter rollout 时间线分别保存于 `base-comparison.db` 与 `adapter-comparison.db`。

### v0.5.2 完整 SFT 基线与退化分析

- 增加 `--seed` 参数，默认 `42`，并将 seed 写入训练 metrics。
- 使用 600 条 Hotpot 训练样本完成 1 epoch（75 个 optimizer step）：训练 loss `0.2489`、验证 loss `0.1961`、训练耗时 `570.09 秒`、峰值显存 `4.116 GB`。
- 30 条快速 holdout 对比：基础模型 exact match `3.3%`、执行成功率 `86.7%`；1 epoch adapter exact match `10.0%`、执行成功率 `16.7%`。
- Adapter 的重复搜索率达到 `56.7%`，说明仅扩大 SFT 训练并不能自动学会停止检索；当前应优先增加 anti-repeat 过程奖励、截断重复轨迹，并实现 sampled rollout 后再进入 GRPO。
- 由于 30 条快速评测已显示退化，本版本没有继续运行 100 条正式评测，避免把明显失败的策略扩大为无价值的长时间实验。

### v0.6.0 Anti-Repeat RL 轨迹闭环

- 新增策略终止原因、RL eligibility、action mask、teacher-forced logprob 和 group advantage；非法 JSON 与重复搜索不再丢失为普通异常。
- 真实 1 个任务 × 4 条 sampled rollout 冒烟成功：2 条 answer、2 条 invalid action，group reward mean `0.370`、std `0.970`；导出的 token/logprob/mask 长度一致且全部为有限值。
- 同 seed 重放的离散动作、termination reason 和 reward 完全一致。但两次独立加载下 BF16 logprob 最大绝对差为 `0.0745`，未达到严格字节级复现要求。
- 因此按实验停止条件不运行 5×4、30 条 A/B 或 10×4 正式采集；该问题需在后续尝试 FP32 logprob 复算或确定性兼容 kernel 后再继续。

### v0.6.1 → v0.7.0 Logprob 稳定化与最小 GRPO

- 新增同实例严格稳定性、BF16/FP32 跨加载诊断与策略指纹；训练不用跨进程 SQLite 数值作为 old policy，而是在未更新策略快照内重算。
- 新增最小 LoRA-GRPO：对每条 assistant action token 广播 rollout advantage，以 clipped policy loss 和 reference KL 更新 LoRA；基础模型与视觉塔保持冻结，tool observation 永不参与 loss。
- 新增 `validate-logprob` 与 `train-grpo`。本次代码验证不等同真实 GPU 验收：需先运行稳定性命令；仅当同实例 gate 通过后才运行 5×4 冒烟，随后按文档做 30 条 greedy 对照。模型、checkpoint、SQLite、轨迹和报告继续不提交 Git。

#### 实际 v0.6.1 / v0.7.0 验收（2026-09-13）

- `validate-logprob --repeats 10` 已完成。同实例 10 次的 max logprob delta 为 `0`；本次 BF16、FP32 跨加载的 P95/max delta 也为 `0`，所有 ratio 为 `1.0`。策略指纹记录了 Qwen 配置、adapter hash、tokenizer hash、Torch `2.14.0+cu126`、Transformers `5.18.0.dev0`、PEFT `0.20.0` 与 CUDA `12.6`。
- 5 任务 × 4 rollout 的 GRPO 冒烟完成，1 个 group 因 reward 零方差被正确跳过，余下 4 个 group 完成更新；共 192 个 LoRA 参数发生变化，adapter 与 metrics 已保存至本地忽略目录 `checkpoints/qwen35-08b-grpo-v070-smoke/`。
- 所有记录的 loss、KL、梯度范数和 logprob 均有限，未出现 OOM 或基础设施失败；但最后一个 group 的 PyTorch 峰值分配显存为 `8.54 GB`，超出 RTX 4060 8GB 的保守预算。因此不执行计划中的 30 条 A/B 或更大训练。下一步应先降低 `max_input_tokens`/动作上限、启用梯度累积或检查峰值统计，再重新运行 5×4 作为显存门槛验证。
- 已完成显存优化复跑：gradient checkpointing、输入梯度钩子和 1024 token 训练截断后，4 个实际更新 group 峰值为 `2.49/2.44/2.69/2.73 GB`；192 个 LoRA 参数变化，loss/KL/grad norm 有限，优化 checkpoint 保存在 `checkpoints/qwen35-08b-grpo-v070-smoke-optimized/`。显存门槛通过，下一步才进入 30 条 greedy A/B 对照。
- 30 条同一 holdout greedy 对照已完成。v0.5.2 adapter：exact match `10.0%`、mean reward `0.123`、invalid action `16.7%`、repeated search `56.7%`、execution success `16.7%`；优化后的 v0.7.0 GRPO adapter：exact match `46.7%`、mean reward `0.469`、invalid action `13.3%`、repeated search `23.3%`、execution success `100%`。基础模型同批为 exact match `6.7%`、mean reward `0.052`、invalid `20.0%`、repeated `0%`、execution success `100%`。
- 按计划门槛，GRPO adapter 的 repeated action 已低于 v0.5.2 的 `56.7%`，invalid action 不高于 `16.7%`，基础设施失败为 `0`，mean reward 高于 v0.5.2；该结果支持进入更大规模采集，但仍只是单次 30 题观察，不能表述为泛化能力证明。机器报告为 `reports/adapter-comparison-v052-30.json` 与 `reports/adapter-comparison-v070-30.json`。
- 随后完成 10 个训练任务 × 4 rollout 的正式采集：8 个 group 更新、2 个零方差 group 跳过，正式 adapter 为 `checkpoints/qwen35-08b-grpo-v070-formal/`。在 100 条同一 holdout 上，正式 GRPO adapter：exact match `34.0%`、mean reward `0.5147`、invalid action `8.0%`、repeated search `0%`、execution success `100%`；v0.5.2 对照：`14.0% / -0.1528 / 15.0% / 59.0% / 100%`。机器报告为 `reports/adapter-comparison-v070-100.json` 与 `reports/adapter-comparison-v052-100.json`。
- 100 条结果同时满足预设门槛并显示明显改善，但仍只来自一个本地 holdout 子集、一个 seed 和固定离线 SearchTool；不能据此宣称真实 HotpotQA 泛化，后续应增加独立 seed、完整验证集和统计置信区间。

### v0.7.1 训练正确性审计

- 已完成正式 checkpoint、SQLite、报告、策略指纹与训练目标的本地审计，详见 [v0.7.1 训练正确性审计](v071-training-audit.md)。
- 审计确认正式 checkpoint 可加载、单实例及跨加载 logprob 验证通过；同时发现 LoRA dropout 使 `eval old` 与 `train current` 的预更新 ratio 出现 `0.94045–1.04205` 波动，且历史评测报告缺少 adapter 指纹并复用 SQLite 文件。
- 因此此前质量指标保留为探索性观察。下一阶段先完成可复现性修正，再从同一 SFT 起点进行多 seed 训练与独立评测。
- v0.7.1 修正已完成并验证：current logprob 使用 eval 模式保留梯度，AdamW 跨 group 持续复用，reference 改为起点 adapter 快照，训练产物可指定独立 SQLite/报告路径。重新 5×4 后 KL 降至 `0.00–0.03`、clip fraction 为 `0–3%`、峰值显存 `7.88 GB`，192 个 LoRA 参数发生变化。报告见 `reports/grpo-v071-audit-smoke.json`。
- 已补跑 seed `43/44`。三个 seed 的有效 group 数为 `4/5/5`，最大 KL 为 `0.0282/0.0082/0.0218`，最大 clip fraction 为 `3.18%/2.10%/2.19%`，峰值显存均约 `7.9 GB`。这验证了数值边界在不同 seed 下稳定，但尚未完成三 seed 的独立任务质量评测。

### v0.7.2 最终测试集隔离与评测协议

- 新增流式 `prepare-final-test` 和 `audit-final-test`，用于构造并验证 300 条与既有 Hotpot 训练、验证和开发评测集隔离的最终样本。
- 最终集使用 task ID、supporting-document 标题和内容 SHA256 三重审计；manifest 固定来源数据哈希与任务顺序，保证未来结果可追溯。
- 旧 `data/hotpot-agent-v041/test.jsonl` 更名为“开发评测集”的语义，不再用于最终结论。新增 `evaluate-final --confirm-final`，要求选择并冻结 checkpoint 后才允许评测。
- 本版本的任务成果是数据/协议与可执行审计；尚未读取最终集结果，因此没有产生新的模型质量结论。
- 增加 `evaluate-adapter` 作为可重复运行的开发集单 checkpoint 评测入口；其拒绝接收 `final-test.jsonl`。最终模型选择规则与一次性最终评测命令记录在 [最终测试与 checkpoint 选择协议](final-evaluation-protocol.md)。

### v0.7.3 诊断与可信对照

- 保留严格 EM 以兼容历史结果，新增规范化 EM、token F1、prompt 截断标记和 rollout 失败分类。
- 新增 `analyze-rollouts` 与固定 100 条开发集的 `evaluate-matrix`，比较 v0.5.2 SFT 与三个 v0.7.1 GRPO checkpoint；本版本不修改 reward，也不触碰 v0.7.2 final。
- 有效矩阵复跑完成：SFT 与三个 GRPO 的 EM 为 `13%/37%/45%/44%`，token F1 为 `0.135/0.432/0.493/0.494`，answer completion 为 `20%/68%/66%/71%`。四组均无基础设施失败。
- 首轮评测发现截断元数据误入模型参数的 bug，已修复并使用新数据库完整复跑；首轮失败产物保留用于审计。诊断确认主要收益来自减少重复搜索并完成回答，答案质量和空检索仍需在 v0.8.0 处理。
- 三个 v0.7.1 adapter 已在相同的 30 条开发任务上完成对照：seed 42/43/44 的平均 reward 分别为 `0.5620/0.6007/0.6143`，均无基础设施失败。依据预注册规则冻结 seed 44；最终集结果将在唯一一次评测完成后另行记录。
- 唯一一次 300 条最终评测已完成：300 个 rollout 均成功持久化，exact match `27.3%`、answer completion `67.0%`、mean reward `0.2847`、repeated action `26.0%`、invalid action `6.7%`。结果明显低于 30 条开发集，表明此前开发结果偏乐观；详见 [v0.7.2 最终评测分析](final-evaluation-v072.md)。

### v0.8.1 停止策略诊断实验（未发布）

- 基于 v0.8 A/B/C 的失败诊断，新增仅在“严格正确答案 + supporting documents 完整覆盖”时触发的 `EvidenceStopReward`，以及“证据已齐仍搜索”的即时 `-0.2` 轨迹标签。support titles 从未进入模型 prompt 或 SearchTool。
- 4-bit QLoRA 下完成 5×4 冒烟、checkpoint 重载和 10×4 正式运行：40/40 rollout 成功、6 个有效 group、192 个 LoRA 参数更新、峰值显存 `3.661 GB`，无 OOM/NaN/Inf。
- 固定 100 条开发集的实际结果为 EM `13%`、F1 `0.143`、回答完成 `20%`、重复搜索 `71%`、非法动作 `6%`。相对 v0.8 arm A（`14% / 0.168 / 20% / 70% / 8%`）未形成整体改善。
- 结论：证据停止奖励在此规模下过于稀疏，不能作为 v0.8 发布依据；停止继续调参，不重跑 final-test，也不创建 `v0.8.0` 标签。详见 [v0.8.1 停止策略诊断](v081-stop-policy-diagnosis.md)。

### v0.8.2 上下文预算与显存门控（未发布）

- 训练数据库复核发现旧 384-token 运行的模型调用截断率达 `32.9%`；新版本改为保护 system/user 的结构化 observation 压缩，并补齐各阶段上下文与显存记录。
- 10 条 validation 长度诊断中，384/768/1024/2048 的截断率为 `45%/10%/0%/0%`；768、1024、2048 的 EM 均为 `40%`，因此更长输入没有显示额外收益。
- 768-token 5×4 冒烟达到 0% 训练截断、3/5 有效 group 和 `7.123 GB` reserved 峰值；checkpoint 重载成功。
- 10×4 正式运行完成 40/40 rollout，但截断率升至 `10.39%`、reserved 峰值升至 `7.703 GB`，超过预注册门槛。实验停止，不运行开发集对照、不租服务器、不读取 final-test、不创建 v0.8.2 标签。
- 完整实现、观察与解释边界见 [v0.8.2 上下文预算实验](v082-context-budget.md)。

### v0.8.3 RL Harness 本机门槛实验（候选，未发布）

- 保存预算后的 prompt token 与 attention mask，old/current/reference 三类 logprob 以同一不可变输入重放；训练使用逐 transition 流式、rollout 等权反传，并以只读审计核验 token、mask、奖励与 advantage。
- 发现工具 `latency_ms` 曾作为非语义文本进入下一轮模型上下文，可能改变临界截断；已移出模型可见 observation，完整耗时继续留在 SQLite transition 中用于审计。
- 384-token 1×4 因采样截断率 `20%` 按 gate 停止。修正后 768-token 5×4 冒烟通过：截断率 `0%`、有效 group `4/5`、无基础设施失败、reserved 峰值 `3.504 GiB`、192 个 LoRA 参数更新，checkpoint 已成功重载。
- 768-token 10×4 在第 2 个 group 因当组截断率 `20%` 停止；显存仅 `1.363 GiB`、无 NaN/Inf 或基础设施失败。只读审计验证这 4 条已采集轨迹完整；仅保留 metrics 审计目录，未保存可评测 adapter。正式训练未达到扩大门槛，故不运行 100 条开发集评测、不读取封存 final-test、不创建 `v0.8.3` 标签。

### v0.8.4 固定检索环境与可信 GRPO 对照（未发布）

- 解耦 `task_limit` 与 `corpus_limit`，固定训练前 100 题产生的 200 篇文档；5×4 与 10×4 的 corpus hash 均为 `29f33888358a10661c69447bf880ba0a9bb08ceb9c74e494a90bf394256bc2f7`。
- 新增 `bm25-v2`：过滤英文单字符和 stopword、标题加权、精确标题 boost，标准 query 诊断在 100 条 validation 上 Support Hit@1/Hit@3/MRR 均为 100%。
- 30 条 validation 的检索 A/B 中，v1→v2 的 EM 为 `20.0%→33.3%`，F1 为 `0.283→0.433`，support coverage 为 `0.400→0.633`，基础设施失败为 0；v2 的采样截断率为 `10.2%`，低于 v1 的 `34.5%`。
- v2 固定环境下 5×4 smoke 通过：3/5 有效 group、采样截断 0%、reserved 峰值 `2.414 GiB`、192 个 LoRA 参数更新；checkpoint 重载成功。
- v2 固定环境下 10×4 formal 通过：6/10 有效 group、采样截断 0%、reserved 峰值 `3.141 GiB`、无基础设施失败；formal checkpoint 重载成功。
- 100 条开发集对照（SFT / 历史 seed44 / v0.8.4 formal）：strict EM `33%/37%/38%`，F1 `0.441/0.467/0.492`，answer completion `90%/96%/96%`，invalid action `10%/4%/4%`，support coverage `0.595/0.620/0.645`。v0.8.4 相对 SFT 的 paired bootstrap EM 差值为 `+5pp`（95% CI `+1pp,+10pp`），F1 差值为 `+5.1pp`（95% CI `+1.2pp,+9.4pp`）。
- 重复搜索在该固定开发环境中 SFT、历史 seed44 和 v0.8.4 均为 `0%`，因此预注册的“下降 5 个百分点”条件出现地板效应，不能作为已达标证据。代码与实验报告完成，但按规则不创建 `v0.8.4` 标签；结果只说明本固定离线环境下的观察，不证明 HotpotQA 泛化。
- 编译检查和 19 项无临时目录依赖的测试通过；完整 pytest 为 `29 passed, 15 errors`，15 项均在 Windows pytest 临时目录 ACL 的 fixture setup 阶段报 `WinError 5`，不是代码断言失败。v0.8.4 暂保留为未发布候选。

### v0.8.5 困难停止决策诊断（未发布）

- 新增固定历史的一步停止探针：一次搜索、两次不同搜索、空检索和重复历史；预置搜索不计入模型重复行为、policy token 或训练 reward。训练/验证前 100 条 ID 与规范化问题审计无重叠，gold answer/support titles 仅写入离线审计字段。
- 30 条 validation × 4 条件的真实推理已完成。768/1024 token 的 prompt 截断率分别为 `45.8%/33.3%`，均超过 `5%` gate；历史删除与预算错误均为 0，reserved 峰值仅 `2.787/2.410 GiB`，因此停止原因是输入证据压缩而非 OOM。
- 两档都几乎总是直接回答：空检索直接回答率 `96.7%`、新搜索和新重复 query 均为 0。两次不同搜索且完整支持文档覆盖时，1024 的 EM/F1 为 `43.3%/0.551`，但该观察仍受截断影响，不能作为长度或 RL 效果结论。
- 768 与 1024 的动作类型仅有 `2/120` 差异。按预注册规则不启动三 seed GRPO、不读取封存 final-test、不创建 `v0.8.5` 标签。完整 pytest 已在用户可写临时目录通过 `51 passed`；Windows Git 输出 GBK 问题已修复为显式 UTF-8 解码。

### v0.8.6 证据表达压缩与真实输入审计（未发布）

- 新增 `compact-v1` observation encoding：移除模型上下文的 score、重复文档使用稳定 ID 引用、正文仅在完整句子边界裁剪；原始 observation、可见 messages、最终 token IDs/attention mask 与逐文档变化同时保存。
- 固定 120-case tokenizer-only 审计与真实推理完成。768/1024 的模型可见 support 保留率为 `98.3%/100.0%`，无历史删除、预算错误、OOM、NaN 或基础设施失败；reserved 峰值为 `3.447/3.539 GiB`。
- 但有损正文裁剪率为 `31.7%/5.8%`，768 和 1024 都未低于预注册 `5%` 门槛。因 768 训练前 gate 失败，本轮未运行 sampled rollout、GRPO 或多 seed 训练，也未读取 final-test。
- 新增长期下载约束：同一下载或安装任务预计累计超过 `1GB` 必须先得到用户确认。完整结果与比较边界见 [v0.8.6 证据表达审计](v086-observation-encoding.md)。


本次任务在不下载模型的前提下完成 Qwen2.5-0.5B 接入骨架：增加懒加载本地客户端、环境预检、动作 JSON 协议、本地模型配置、可选依赖、token 数据记录与 SFT 数据导出边界。模型权重、缓存和 checkpoint 不进入 Git。

测试覆盖合法/非法 search 与 answer 动作、本地配置校验，并继续执行原有 FakeModel 端到端、SQLite 和导出测试。共 11 项测试全部通过；FakeModel 冒烟运行成功，单条 rollout 奖励为 1.170。真实 GPU 推理留待模型下载后单独验收。
# v0.8.7 候选：证据审计与输入重放

本轮恢复了可容纳输入的 legacy 表达，并新增 `compact-v2` 的内容哈希引用、句级可追溯压缩和采样 token 精确重放。冻结 120-case 审计表明 768 token 仍有 30 个明确预算失败，故训练与多 seed GRPO 仍按门槛停止；未读取最终集，未下载依赖或数据，未创建版本标签。完整说明见 `v087-input-replay.md`。

后续收口新增 compact-v3 低排名优先裁剪、共享 gate、完整失败分母和 CLI 冻结输入统一。57 项测试通过；768/1024/2048 三档均完成 120-case，只有 2048 纯推理通过。已冻结 24GB 单卡迁移配置，本机训练保持停止。
