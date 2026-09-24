# 数据模型

## v0.8.4 补充

`RolloutGroup.environment_fingerprint` 保存 task 顺序、固定 corpus hash、corpus 文档数和 retriever 版本。改变 task_limit 不得隐式改变 corpus；训练和回放必须核对该指纹。

`SearchTool` 的 `keyword-v1` 与 `bm25-v2` 共用工具 schema，但检索器版本属于实验环境字段，不能在同一对照中混用。

## v0.8.3 补充

`ModelResponse.prompt_attention_mask` 与 `prompt_token_ids` 共同表示预算后、模型实际看到的输入，GRPO 训练必须原样重放。`Rollout.reward_config` 保存生成 reward 的公开权重；旧轨迹缺失它时审计状态为不可验证，而不是默认通过。

`Task` 保存问题和标准答案；`RolloutGroup` 保存同一问题的多次采样；`Rollout` 保存状态和终局 reward；`Transition` 保存每一步 prompt、模型响应、工具 observation 和终止信息；`Event` 以 append-only 方式记录审计事件。

token IDs 和 logprobs 当前允许为空，因为 FakeModel 和多数普通 API 不返回它们。未来接入 GRPO 时，应使用同组 rollout 的 reward 计算 group-level advantage，并避免把工具 observation 作为 policy action。
# v0.8.2 上下文使用字段

`ModelResponse.usage` 除 token usage 外保存：`context_budget`、`original_prompt_tokens`、`actual_prompt_tokens`、`system_tokens`、`question_tokens`、`history_tokens`、`observation_tokens`、`dropped_observation_tokens`、`prompt_truncated`、`truncation_stage`、`phase` 和 `prompt_sha256`。`observation_encoding` 标识 `legacy` 或 `compact-v1`；后者还保存 `lossless_expression_changed`、`lossy_body_trimmed`、`documents_deleted`、`reused_document_references` 与逐文档 `document_changes`。这些字段只描述模型实际输入，不包含 gold answer 或 reward。
# v0.8.7 上下文审计补充

`observation_encoding` 支持 `legacy`、`compact-v1`、`compact-v2` 与 `compact-v3`。compact-v3 的每个文档变更记录包含 rank、tool index、正文哈希、原文长度、可见长度、原文片段偏移、引用目标和内容冲突。探针报告显式保存 expected、attempted、completed、failed、not_run 及 gate 分母；缺少必需字段时状态为 `UNVERIFIABLE`。
