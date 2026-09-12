# 数据模型

`Task` 保存问题和标准答案；`RolloutGroup` 保存同一问题的多次采样；`Rollout` 保存状态和终局 reward；`Transition` 保存每一步 prompt、模型响应、工具 observation 和终止信息；`Event` 以 append-only 方式记录审计事件。

token IDs 和 logprobs 当前允许为空，因为 FakeModel 和多数普通 API 不返回它们。未来接入 GRPO 时，应使用同组 rollout 的 reward 计算 group-level advantage，并避免把工具 observation 作为 policy action。
