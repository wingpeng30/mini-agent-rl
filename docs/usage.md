# 使用方式

```bash
mini-agent-rl init --db mini_agent.db
mini-agent-rl run --task examples/tasks.jsonl --group-size 4
mini-agent-rl list
mini-agent-rl show <rollout-id>
mini-agent-rl export --output trajectories.jsonl
```

API 模型适配器需要在 Python 中构造 `OpenAICompatibleClient(base_url, model)`；API key 从 `OPENAI_API_KEY` 读取。首版演示不需要联网或密钥。
