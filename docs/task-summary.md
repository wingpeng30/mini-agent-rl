# 本次任务成果

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
