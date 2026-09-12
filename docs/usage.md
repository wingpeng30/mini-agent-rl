# 使用方式

```bash
mini-agent-rl init --db mini_agent.db
mini-agent-rl run --task examples/tasks.jsonl --group-size 4
mini-agent-rl list
mini-agent-rl show <rollout-id>
mini-agent-rl export --output trajectories.jsonl
```

批量离线评测：

```bash
mini-agent-rl evaluate --task examples/benchmark.jsonl --group-size 4 --output reports/baseline.json
```

DeepSeek 可选后端：

```powershell
$env:DEEPSEEK_API_KEY="在 DeepSeek 控制台创建的新密钥"
mini-agent-rl evaluate --backend deepseek --model deepseek-v4-flash
```

该后端使用 OpenAI-compatible Chat Completions 与 Tool Calls；密钥只从环境变量读取。

API 模型适配器需要在 Python 中构造 `OpenAICompatibleClient(base_url, model)`；API key 从 `OPENAI_API_KEY` 读取。首版演示不需要联网或密钥。
