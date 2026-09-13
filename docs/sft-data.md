# v0.4.0：离线 SFT 数据

## 生成方式

```powershell
mini-agent-rl generate-sft --corpus examples/corpus.json --output-dir data/sft-v040
```

默认生成 800 条 train、100 条 validation、100 条 test。相同语料和 seed 会产生相同 JSONL，便于实验复现。

每条样本包含五条消息：system、user、assistant search、tool observation、assistant answer。标准答案只放在 `metadata.gold_answer`，不会暴露给 Agent 的上下文。

这批数据首先训练动作格式和检索流程，不负责扩大模型知识。当前生成器不调用网络、不读取 API key、不下载模型，适合作为 LoRA-SFT 前的可审查基线。
