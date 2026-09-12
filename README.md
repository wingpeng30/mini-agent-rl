# Mini Agent RL

一个不下载本地模型、无需 GPU 的最小工具调用 Agent 框架。它保留了强化学习所需的 rollout group、transition、reward、token/logprob 字段和 JSONL 导出边界。

## 快速运行

```bash
pip install -e ".[test]"
mini-agent-rl run --group-size 4
mini-agent-rl export
```

默认使用 FakeModel 和内置离线 SearchTool。真实模型可通过 `OpenAICompatibleClient` 接入兼容 Chat Completions 的服务。

## 当前范围

当前版本是 RL-ready 数据采集框架，不执行 GRPO/PPO 权重更新。工具 observation 只作为上下文，不参与 policy loss；后续训练应在 rollout group 层计算 advantage。

## 文档

- [架构](docs/architecture.md)
- [数据模型](docs/data-model.md)
- [使用方式](docs/usage.md)
- [任务成果](docs/task-summary.md)
