# Mini Agent RL

当前候选版本为 v0.8.7。它新增可审计的 `compact-v2` 证据表达与精确输入重放；768-token 的训练门槛尚未通过，因此不会把该版本表述为已验证的 GRPO 改善。参见 `docs/v087-input-replay.md`。

收口实验进一步加入 `compact-v3` 的低排名优先裁剪。完整 120-case 结果显示 768/1024 仍有预算失败，2048 纯推理通过；下一步使用已冻结的单卡 24GB 配置验证 2048-token GRPO 冒烟。

一个可离线运行、并为本地小模型训练保留强化学习数据边界的最小工具调用 Agent 框架。FakeModel 路径仍不需要 GPU；本地模型适配器采用懒加载，不会在安装基础依赖或执行测试时下载权重。

当前开发版本：**v0.8.7（未发布）**。本版补充 compact-v3 的证据保留和精确输入重放，并记录受控训练、评测和停止门槛；版本说明见 [CHANGELOG.md](CHANGELOG.md)。

v115 的受控 SFT 试验已完成一次50步训练，但内部评测未达到证据充分回答和证据不足搜索门槛；因此未打开独立100题，未启动 GRPO。完整方法、限制与结果见 [v115 实验记录](docs/v115-sft-experiment.md)。

```powershell
mini-agent-rl experiment preflight --config configs/v083-local-8gb.toml
mini-agent-rl experiment inspect runs/v083-local-8gb-smoke
```

## 快速运行

```bash
pip install -e ".[test]"
mini-agent-rl run --group-size 4
mini-agent-rl evaluate --task examples/benchmark.jsonl --output reports/baseline.json
mini-agent-rl export
```

默认使用 FakeModel 和内置离线 SearchTool。真实模型可通过 `OpenAICompatibleClient` 接入兼容 Chat Completions 的服务。

## 本地模型（下载前准备）

当前版本提供 Qwen3.5-0.8B 的本地适配框架，但不会自动下载模型。先执行只读预检：

```powershell
mini-agent-rl local-check
```

准备进行真实推理时再安装可选依赖并启动；第一次启动才可能从 Hugging Face 下载权重：

```powershell
pip install -e ".[local-model]"
mini-agent-rl local-smoke --model-path D:\qwen_08b
```

RTX 4060 8GB 首次推理使用 BF16、`group-size 1`、最大 2048 输入 token。Qwen3.5 需要项目固定的 Transformers 兼容 commit；模型权重、checkpoint 和 `*.safetensors` 已加入 `.gitignore`。

## 生成离线 SFT 数据

```powershell
mini-agent-rl generate-sft --corpus examples/corpus.json --output-dir data/sft-v040
```

默认生成 800 条 train、100 条 validation 和 100 条 test 数据，不调用 API。标准答案只写入 metadata，模型 messages 中只包含问题、search 动作、工具 observation 和 answer 动作。

## 导入多跳检索数据

```powershell
pip install -e ".[data]"
mini-agent-rl download-hotpot --output-dir data/hotpot-agent-v041
```

该命令从 HotpotQA 流式导入 800 条英文多跳检索样本，保留 supporting documents 并按文档级别切分。详见 [HotpotQA 数据说明](docs/hotpot-data.md)。

```powershell
mini-agent-rl audit-sft --data-dir data/hotpot-agent-v041
```

`data/hotpot-agent-v041/test.jsonl` 是开发评测集：可用于排错、选 checkpoint 和多次对照，不能再称为最终测试集。

## 隔离的最终测试集

当训练与开发阶段结束后，先生成一份从未参与训练、调参或 checkpoint 选择的最终集。该命令仍是流式读取，不会下载完整 HotpotQA：

```powershell
mini-agent-rl prepare-final-test --source-dir data/hotpot-agent-v041 --output-dir data/hotpot-agent-v072 --count 300
mini-agent-rl audit-final-test --source-dir data/hotpot-agent-v041 --final-dir data/hotpot-agent-v072
```

生成的 manifest 固定来源 split 与最终文件的 SHA256，并审计 task ID 与 supporting-document 标题均无重叠。选定且冻结唯一 checkpoint 后，才执行一次最终评测；该命令要求显式确认，防止误把最终集用于调参：

```powershell
mini-agent-rl evaluate-final --adapter-path checkpoints/qwen35-08b-grpo-v071-seed43 --confirm-final
```

## LoRA-SFT

```powershell
mini-agent-rl train-sft --model-path D:\qwen_08b --max-steps 10 --output-dir checkpoints/qwen35-08b-sft-smoke
```

训练仅更新 LoRA adapter，且只计算 assistant action token 的 loss。详见 [SFT 训练说明](docs/sft-training.md)。

## LoRA Adapter 对比评测

```powershell
mini-agent-rl compare-adapter --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-smoke --limit 8 --output reports/adapter-comparison-v051.json
```

该命令让基础模型与 LoRA adapter 在同一批完全离线的 Hotpot 测试任务、语料、解码参数和步数限制下运行，报告准确率、动作合法率、检索成功率、奖励、步数、加载时间和显存。详见 [Adapter 对比评测](docs/adapter-evaluation.md)。

## RL 轨迹采集

```powershell
mini-agent-rl collect-rl --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-v052 --task-limit 10 --group-size 4 --temperature 0.7
```

该命令只采集 sampled rollout、逐 action token logprob 和 group advantage，不更新任何权重。重复搜索、非法 JSON 和最大步数均作为带负奖励的策略轨迹保存。详见 [RL 轨迹说明](docs/rl-rollout.md)。

## DeepSeek（可选）

不需要部署本地模型。设置环境变量后可切换到 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="在控制台创建的新密钥"
mini-agent-rl evaluate --backend deepseek --task examples/benchmark.jsonl
```

默认使用 `https://api.deepseek.com` 与 `deepseek-v4-flash`；密钥不会写入 SQLite、报告或 Git。

## 当前范围

当前版本包含最小 LoRA-GRPO 冒烟训练：训练前必须通过同实例 logprob 稳定性验证；工具 observation 只作为上下文，不参与 policy loss。v0.8 的 reward-ablation 与停止策略实验均保留为本地审计产物，尚未形成可发布的泛化改进，故最新发布标签仍为 `v0.7.3`。

本地 8GB 显存的上下文诊断可使用：

```powershell
mini-agent-rl context-length-sweep --lengths 384,768,1024,2048
mini-agent-rl probe-context-backward --max-input-tokens 768
```

结构化预算会保护 system/user，按文档边界压缩 observation，并分别记录采样和 logprob 重算的截断率。当前 768-token 只通过 5×4 冒烟；10×4 的截断率与 reserved 显存超过门槛，因此仍属于未发布实验。

## 文档

- [架构](docs/architecture.md)
- [数据模型](docs/data-model.md)
- [使用方式](docs/usage.md)
- [任务成果](docs/task-summary.md)
- [本地模型基础框架](docs/local-model.md)
- [Adapter 对比评测](docs/adapter-evaluation.md)
- [Logprob 稳定性](docs/logprob-stability.md)
- [最小 GRPO 训练](docs/grpo-training.md)
- [v0.7.1 训练正确性审计](docs/v071-training-audit.md)
- [HotpotQA 数据与最终评测协议](docs/hotpot-data.md)
- [最终测试与 checkpoint 选择协议](docs/final-evaluation-protocol.md)
- [v0.7.2 最终评测分析](docs/final-evaluation-v072.md)
- [v0.7.3 诊断与对照](docs/v073-diagnostics.md)
- [v0.8.1 停止策略诊断（未发布实验）](docs/v081-stop-policy-diagnosis.md)
- [v0.8.2 上下文预算与 8GB 显存实验](docs/v082-context-budget.md)
- [v0.8.4 固定检索环境与可信 GRPO 对照](docs/v084-fixed-retrieval.md)
- [v0.8.5 困难停止决策诊断](docs/v085-stop-decisions.md)
- [v0.8.6 证据表达压缩与真实输入审计](docs/v086-observation-encoding.md)
