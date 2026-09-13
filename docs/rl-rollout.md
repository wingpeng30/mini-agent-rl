# v0.6.0：RL 轨迹采集

本版本只构造强化学习数据边界，不更新模型权重。每个 rollout 保存模型 action token、teacher-forced old logprob、action mask、工具 observation、过程奖励、终止原因和 group advantage。tool observation 只作为上下文，其 mask 恒为 0。

策略终止包括 `answer`、`invalid_action`、`repeated_action` 和 `max_steps`；它们仍是可学习样本。模型、CUDA 或工具内部错误则为基础设施失败，标记为 `eligible_for_rl=false`，不参与 group advantage。

默认奖励为正确答案 `+1.0`、支持文档覆盖最多 `+0.2`、答案证据 `+0.2`、每次搜索 `-0.03`、重复搜索 `-0.5`、非法动作 `-0.5`、最大步数 `-0.25`、无检索错误回答 `-0.1`。

## 复现命令

```powershell
mini-agent-rl collect-rl --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-v052 --task-limit 1 --group-size 4 --temperature 0.7 --seed 42
```

已验证同 seed 下离散动作完全一致。但 RTX 4060 上 Qwen3.5 BF16 teacher-forced logprob 在两次独立模型加载之间仍有最大 `0.0745` 的差异；在修复此问题前，不应将当前 logprob 用于严格的 GRPO 比率计算。
