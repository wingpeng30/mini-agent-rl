# v0.5.0：Qwen3.5 LoRA-SFT

训练命令：

```powershell
mini-agent-rl train-sft --model-path D:\qwen_08b --max-steps 10 --output-dir checkpoints/qwen35-08b-sft-smoke
```

默认使用 BF16、batch size 1、gradient accumulation 8、LoRA rank 8、learning rate 1e-4、最大1024 token。LoRA 仅注入文本语言模块的 attention 与 MLP 投影层，视觉塔不在 target modules 中。

训练标签只覆盖 assistant 生成的 search/answer JSON；system、user 和 tool observation 的 label 为 `-100`，不会参与交叉熵损失。先使用 `--max-steps 10` 冒烟验证，再移除该参数进行完整训练。

## 已验证的本地冒烟结果

在 RTX 4060 8GB、`D:\qwen_08b` 和 HotpotQA 600/100 子集上，10 step 冒烟训练已成功完成：平均训练 loss 为 `0.4313`、验证 loss 为 `0.2847`、峰值显存为 `4.116 GB`、耗时 `87.14` 秒。产物位于本地 `checkpoints/qwen35-08b-sft-smoke/`，其中只有 LoRA adapter 和训练状态，不包含基础模型。

Qwen3.5 chat template 会移除消息首尾空白，因此编码器在完整渲染后的 token 序列里顺序定位每条消息内容；仅 assistant 内容被写入 label。这样 tool 的 JSON observation 始终为 `-100`，不会作为策略监督目标。

## v0.5.2 一轮完整训练结果

600 条训练样本完成 1 epoch、75 个优化步骤后，训练 loss 为 `0.2489`，验证 loss 为 `0.1961`，耗时 `570.09` 秒，峰值显存 `4.116 GB`。在 30 条 holdout 上，adapter 的 exact match 为 `10.0%`，高于基础模型 `3.3%`；但重复搜索率达到 `56.7%`，执行成功率降为 `16.7%`。这表明当前 SFT 数据虽然改善了动作格式和少量回答能力，却没有形成可靠的停止策略，后续需要显式 anti-repeat reward/数据过滤。
