# v0.8.1 停止策略诊断

v0.8.0 的 A/B 开发集结果显示，回答一旦产生通常正确，但大多数轨迹因重复搜索终止。当前奖励只惩罚重复搜索，未在支持文档齐全且答案正确时明确奖励停止。

本轮新增 `EvidenceStopReward`。它只在以下条件同时满足时给出奖励：最终答案严格正确、任务声明了支持文档、已检索结果覆盖全部支持文档。支持文档仍只用于环境后的 reward 计算，不进入模型上下文。

此外，运行时会在**本步搜索前**检查之前已获得的 observation：若 supporting documents 已覆盖完整而策略仍选择 `search`，该 transition 标记为 `over_search_after_evidence` 并写入即时 `-0.2`。这不是基础设施失败，动作及其 token/logprob 仍可用于训练；总奖励亦会单独记录该分项。

## 已执行的受控实验

- 固定起点：`checkpoints/qwen35-08b-sft-v052`。
- 训练：4-bit QLoRA、384 token、temperature `0.7`、group size `4`、最多 3 步、seed `42`；只改变 evidence-stop 权重 `+0.3`。
- 冒烟 5×4：3 个非零方差 group，192 个 LoRA 参数变化，峰值显存 `3.633 GB`，所有 loss 有限。
- checkpoint 重载：独立 `evaluate-adapter --limit 1` 成功完成，没有加载错误。
- 正式 10×4：40/40 rollout 成功；6/10 group 有非零方差，4 个按零方差规则跳过更新；192 个 LoRA 参数变化，峰值显存 `3.661 GB`，未出现 OOM、NaN、Inf 或基础设施失败。

## 固定开发集结果与解释边界

在 `data/hotpot-agent-v041/test.jsonl` 前 100 条上，以 greedy、最多 3 步、独立 SQLite 运行：严格/规范化 EM 均为 `13%`，token F1 为 `0.143`，mean reward 为 `-0.1957`，answer completion 为 `20%`，重复搜索为 `71%`，非法动作为 `6%`，prompt 截断为 `2%`。

与同协议的 v0.8 arm A 相比（EM `14%`、F1 `0.168`、完成率 `20%`、重复 `70%`），本轮没有显示出停止策略改善；它只降低了非法动作（`8%` 到 `6%`），且该差异不足以支持泛化结论。可能原因是“严格正确且完整 supporting coverage”过于稀疏：10 个训练任务中仍只有 6 个有效 reward-variance group。以上是实验观察，不是模型能力结论。

因此停止扩大样本、停止使用开发集继续调参，并且不触碰已封存的 final-test。下一轮应先在训练前增加 reward-signal coverage 审计和更密集但不泄漏答案的过程奖励，再以新的预注册协议重新训练。
