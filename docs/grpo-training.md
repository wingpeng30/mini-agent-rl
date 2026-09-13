# v0.7.0：最小 LoRA-GRPO

本实现只更新现有 LoRA adapter，不实现全参训练、PPO value model 或多机训练。每题采样 4 条轨迹，按 group reward 计算 advantage；非法动作、重复搜索和 max-step 轨迹保留并参与训练，基础设施失败与零方差 group 跳过更新。

```powershell
mini-agent-rl train-grpo --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-v052 --data data/hotpot-agent-v041/train.jsonl --task-limit 5 --group-size 4 --output-dir checkpoints/qwen35-08b-grpo-v070-smoke --seed 42
```

默认参数：temperature `0.7`、max steps `3`、learning rate `5e-6`、clip epsilon `0.2`、KL beta `0.02`、policy epochs `2`、max grad norm `1.0`。训练前校验 base config、adapter config/weight 与 tokenizer 哈希；不匹配会终止。每个 group 在同一实例中依次重算 old logprob、禁用 adapter 重算 reference logprob、计算 current logprob 并反传；system/user/tool observation token 永不进入 policy loss。

输出 adapter 与 `grpo_metrics.json`，其中包含 loss、KL、clip fraction、grad norm、奖励和显存。首次仅做 5×4 冒烟；完成后应以同一 holdout 的 greedy 30 条对照检查重复率、非法率、平均奖励和基础设施失败率，再决定是否扩大采样。

本机实际 5×4 冒烟记录到最后一组峰值分配显存 `8.54 GB`。虽然无 OOM，仍超过 RTX 4060 8GB 的保守预算；请先完成显存优化并重跑 5×4，再做 30 条对照。

随后启用 gradient checkpointing、输入梯度钩子、训练上下文 1024 token 截断并按 group 重置显存统计；优化版 5×4 的五组峰值分别为 `2.49/2.44/2.69/2.73 GB`（第一组零方差跳过），192 个 LoRA 参数发生变化，checkpoint 为 `checkpoints/qwen35-08b-grpo-v070-smoke-optimized/`。该结果满足单卡显存门槛，但仍需独立 30 条 A/B 评测后才能判断策略质量。

30 条 greedy holdout 对照已完成：v0.5.2 adapter 的 exact/reward/repeated-search/execution-success 为 `10.0% / 0.123 / 56.7% / 16.7%`；优化 v0.7.0 adapter 为 `46.7% / 0.469 / 23.3% / 100%`。这些是固定小样本的实验观察，不等价于泛化结论；正式扩大训练前仍应固定测试集并重复 seed。

正式 10×4 采集后，100 条同一 holdout 的最终对照为：v0.5.2 `14.0% / -0.1528 / 59.0% / 15.0%`，v0.7.0 `34.0% / 0.5147 / 0% / 100%`（依次为 exact match / mean reward / repeated search / execution success）。报告保存在 `reports/adapter-comparison-v052-100.json` 与 `reports/adapter-comparison-v070-100.json`。这是固定单 seed 的实验观察，后续仍需独立 seed 与验证集复现。
