# v0.9.0 思考模式 SFT 协议

本轮只验证 Qwen3.5-0.8B 是否能用短思考可靠地提交检索动作；它不是 GRPO 实验，也不使用封存的 `hotpot-agent-v072`。

教师数据来自 `hotpot-agent-v041/train.jsonl`。教师只看问题、已执行动作和真实 BM25-v2 返回的可见证据；gold answer 与 supporting titles 不进入教师请求。教师生成一两句策略说明和下一步 JSON；回答随后才与 gold answer 匹配，搜索随后才在 BM25-v2 上回放。每个通过状态同时导出思考与直接 JSON 两个目标，按 task ID 固定划分 train/validation。

训练从 `qwen35-08b-sft-v052` 继续加载 LoRA，输出到独立 checkpoint。Qwen 的实际思考模板会把 `</think>` 与 JSON 分隔为一个空行；训练标签按渲染后的 token 序列定位，当前状态的思考、闭合标签和动作均参与 loss，历史 assistant/tool 消息不参与 loss。

正式训练前要求：至少 300 个训练状态、50 个验证状态，每种动作各至少 20%；实际 Qwen tokenizer 审计无超长样本；10-step 冒烟的 loss、梯度、LoRA-only 参数变化和重载均通过。正式 SFT 后，思考模式必须在内部验证达到合法率 98% 及以上、输出耗尽不超过 5%，才进入开发集 A/B/C 对照。

思考模式仅在 C 组的 EM 相对等输出预算的 B 组提升至少 3 个百分点，或部分证据继续搜索率提升至少 10 个百分点且 F1 不下降，并且配对置信区间支持提升时保留。否则默认弃用思考模式；非思考目标如有独立提升可以保留，但不归因为思考。
