# 数据模型

`Task` 保存问题和标准答案；`RolloutGroup` 保存同一问题的多次采样；`Rollout` 保存状态和终局 reward；`Transition` 保存每一步 prompt、模型响应、工具 observation 和终止信息；`Event` 以 append-only 方式记录审计事件。

token IDs 和 logprobs 当前允许为空，因为 FakeModel 和多数普通 API 不返回它们。未来接入 GRPO 时，应使用同组 rollout 的 reward 计算 group-level advantage，并避免把工具 observation 作为 policy action。
# v0.8.2 上下文使用字段

`ModelResponse.usage` 除 token usage 外保存：`context_budget`、`original_prompt_tokens`、`actual_prompt_tokens`、`system_tokens`、`question_tokens`、`history_tokens`、`observation_tokens`、`dropped_observation_tokens`、`prompt_truncated`、`truncation_stage`、`phase` 和 `prompt_sha256`。这些字段只描述模型实际输入，不包含 gold answer 或 reward。
