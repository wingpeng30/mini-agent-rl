# v0.5.1：LoRA Adapter 对比评测

## 目标

确认 LoRA-SFT 产生的 adapter 是否相较相同基础模型改善工具 Agent 行为。评测不调用网络、不访问训练集，也不将 gold answer 暴露给模型。

## 数据与公平性边界

`compare-adapter` 从 Hotpot SFT 测试 JSONL 中读取题目和 tool observation 里的 supporting documents，构造运行时 SearchTool。`metadata.gold_answer` 仅进入 `Task.answer`，供 rollout 结束后的 reward 计算；它不会写入 system prompt、user prompt 或检索语料。

基础模型与 adapter 采用完全相同的：测试题、离线语料、`temperature=0`、`max_steps=3`、最大生成 token 和 reward 配置。由于单张 8GB 显卡，二者顺序运行，避免显存竞争。

## 运行方式

```powershell
mini-agent-rl compare-adapter `
  --model-path D:\qwen_08b `
  --adapter-path checkpoints/qwen35-08b-sft-smoke `
  --data data/hotpot-agent-v041/test.jsonl `
  --limit 8 `
  --output reports/adapter-comparison-v051.json
```

输出报告包括 exact match、平均 reward、动作合法率、SearchTool 成功率、重复搜索率、平均步数、模型加载时间与显存峰值。`reports/` 和比较产生的 SQLite 数据库均被 Git 忽略。

## 已验证结果

在 RTX 4060 8GB 上，使用 v0.5.0 的 10 step LoRA adapter 对 8 条 holdout 题运行：基础模型的 exact match 为 `0.0%`、平均 reward `0.006`、动作 JSON 合法率为 `62.5%`（3 条 JSON 解析失败）；adapter 的 exact match 为 `12.5%`、平均 reward `0.178`、动作合法率为 `100%`，但有 `37.5%` 的轨迹因重复搜索失败。

这是功能与数据边界的冒烟结论，不是统计显著的模型能力评估：样本只有 8 条，且 adapter 仍有三条重复检索导致的失败。扩大测试集、固定随机种子并报告置信区间后，才可作为正式实验结论。
