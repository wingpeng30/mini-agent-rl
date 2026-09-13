# v0.6.1：Logprob 稳定性

此版本把两类问题分开：同一已加载模型实例必须严格稳定；BF16 独立加载允许小幅数值差，但以 policy ratio 阈值审计。采集到 SQLite 的 logprob 仅供追溯，训练开始前会从未更新的同一策略实例重算 old logprob。

```powershell
mini-agent-rl validate-logprob --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-v052 --repeats 10 --output reports/logprob-stability-v061.json
```

报告保存模型/adapter/tokenizer 哈希、Torch/Transformers/PEFT/CUDA、GPU、dtype 及生成参数。门槛为：同实例 max delta 不超过 `1e-6`；BF16 跨加载 P95 delta 不超过 `1e-3`，P95 ratio 在 `[0.99, 1.01]`，全部 ratio 在 `[0.90, 1.10]`。FP32 跨加载 max delta `<=1e-5` 是诊断目标，不改变 BF16 采样配置。出现 NaN、Inf、token 数变化或离散动作变化均视为失败。

该命令不下载模型、不上传数据、不更新权重。报告属于本地实验产物，已被 Git 忽略。
