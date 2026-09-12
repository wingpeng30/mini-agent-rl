# 更新日志

本项目使用[语义化版本](https://semver.org/lang/zh-CN/)：

- **主版本**：出现不兼容 API 改动时递增；
- **次版本**：新增向后兼容的功能时递增；
- **修订版本**：修复向后兼容的问题时递增。

每个提交到 GitHub 的可用阶段都必须：更新 `pyproject.toml` 版本、补充本文件，并创建对应的 Git tag `vX.Y.Z`。

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
