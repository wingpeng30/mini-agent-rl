# 架构

系统由 ModelClient、AgentRuntime、ToolRegistry、Reward、RolloutRunner 和 SQLiteStore 组成。AgentRuntime 负责真实的多步交互，RolloutRunner 负责为同一任务并发产生多个独立 rollout，SQLiteStore 保存可审计事件和 RL-ready transition。

数据流为：任务 → group → rollout → 模型 action → 工具 observation → transition → reward → SQLite → JSONL。

评测层可批量运行任务集，输出执行成功率、答案准确率、平均奖励与平均搜索次数。DeepSeek 通过 OpenAI-compatible Chat Completions 接入：assistant 的 `tool_calls` 与 tool message 的 `tool_call_id` 会被保留，因此真实云端模型和离线 FakeModel 使用同一条 Agent 轨迹链路。

模型生成的 action 是未来策略优化对象；工具 observation 是环境反馈，不应计入 policy loss。这与 Agent Lightning 的“执行与训练解耦”思想一致，但本项目保持独立、轻量和离线可运行。
