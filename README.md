# Mini Agent RL

一个可离线运行、并为本地小模型训练保留强化学习数据边界的最小工具调用 Agent 框架。FakeModel 路径仍不需要 GPU；本地模型适配器采用懒加载，不会在安装基础依赖或执行测试时下载权重。

当前版本：**v0.7.3**。完整变更记录见 [CHANGELOG.md](CHANGELOG.md)。

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

当前版本包含最小 LoRA-GRPO 冒烟训练：训练前必须通过同实例 logprob 稳定性验证；工具 observation 只作为上下文，不参与 policy loss。

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
