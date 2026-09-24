# v0.8.9 思考模式协议

思考模式使用 Qwen 模板的 `enable_thinking` 直接参数。模型完整生成序列（思考段和最终 JSON）仍写入 `action_token_ids` 和 old logprob，供精确重放与后续 GRPO 使用。

运行时只解析闭合 `</think>` 之后的唯一 JSON 动作；思考段中的 JSON 不是工具调用。多轮上下文只回填已执行 JSON 和工具观察，完整思考保留在 `ModelResponse.reasoning_text` 与事件记录中。

对照冻结：A 关闭思考/128输出，B 关闭思考/512输出，C 开启思考/512输出；其余模型、adapter、BM25-v2、题集、输入预算相同。C 不通过协议合法率、预算或基础设施门槛时不启动GRPO。
