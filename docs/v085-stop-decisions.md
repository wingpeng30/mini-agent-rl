# v0.8.5：困难停止决策诊断与有条件多 seed GRPO

## 目的与边界

本轮研究的是小模型在给定检索历史后，是否会继续搜索或停止回答；它不把人工注入的历史搜索动作当作 Agent 自己的行为，也不将这些轨迹用于 GRPO 训练。固定四种条件是一次搜索、两次不同搜索、空检索和已重复执行同一 query。题目顺序和条件顺序在运行前冻结，不能根据模型失败样本替换题目。

最初 10 题预探针后，47 项完整测试已通过，并修复了 Windows 下 Git binary diff 的默认 GBK 解码问题；本轮增加任务构造、交叉审计、输出比较和编码回归测试后，完整测试结果为 51 passed。

gold answer 与 support titles 只写入离线审计字段。它们不进入模型消息、SearchTool 排序、工具结果裁剪或 reward 以外的策略上下文。文档覆盖仅代表检索到 supporting document，不代表模型必然能从证据推出答案。

## 已完成的 10 题预探针事实

`reports/v085-stop-probe-20260917/` 的冻结 SFT、4-bit、768-token、greedy 预探针完成 40 个 case。四种条件都 100% 直接回答，新增重复 query、搜索动作和非法动作均为 0%。空检索条件严格 EM 为 10%，一次与两次前置搜索条件均为 30%。这说明“重复搜索为零”不能证明停止策略正确；在空检索后仍直接回答是更值得诊断的行为。

该预探针的 prompt 截断率为 45%，故不能解释为完整证据历史下的停止能力，也不能启动 GRPO。显存峰值 reserved 为 2.012 GiB，未触发显存门槛。该结论只描述该探针与该 checkpoint，不代表 HotpotQA 泛化表现。

## 本轮 30 题预算对照结果（2026-09-17）

训练/诊断前 100 条前缀的 ID 与规范化问题交叉审计通过；训练、验证源文件 SHA256 分别为 `7af11132…c7c042` 与 `9b703d6a…b9ca0a`。每个预算完成 30 题 × 4 条件 = 120 个 case，均为 4-bit、greedy、seed 42 与同一 SFT adapter。

| 预算 | 截断/observation 压缩 | 历史删除/预算错误 | reserved 峰值 | 两预算动作差异 |
|---|---:|---:|---:|---:|
| 768 | 45.8% | 0% / 0 | 2.787 GiB | 基准 |
| 1024 | 33.3% | 0% / 0 | 2.410 GiB | 2/120（1.7%） |

两档均超过 5% 截断门槛，因此都只作为预算压力诊断，不能作为完整历史停止能力的训练前证据。未触发 OOM，也没有上下文预算错误；问题是 observation 需要压缩，而不是最小 system/user 无法放入。

| 条件 | 768 严格 EM / F1 | 1024 严格 EM / F1 | 1024 回答率 | 解释边界 |
|---|---:|---:|---:|---|
| 一次搜索 | 33.3% / 0.383 | 33.3% / 0.383 | 100% | 仅 8/30 达到完整文档覆盖 |
| 两次不同搜索 | 30.0% / 0.396 | 43.3% / 0.551 | 100% | 30/30 完整文档覆盖，但不等于可推出答案 |
| 空检索 | 6.7% / 0.102 | 6.7% / 0.102 | 96.7% | 仍直接回答，不代表该动作必然错误 |
| 重复历史 | 23.3% / 0.278 | 23.3% / 0.334 | 100% | 前置重复不计为模型重复 |

模型在两档均没有生成新的搜索或重复 query；768 有 3 个非法 JSON，1024 有 1 个。完整证据的两次搜索条件在 1024 下有更高分数，但只有 1.7% 的动作类型发生改变，且仍受 33.3% 压缩影响。它是值得后续验证的观察，不能当作上下文长度的因果结论。

## 本轮可复现命令

```powershell
conda activate mini-agent-rl
python -m mini_agent_rl.stop_probe --data data/hotpot-agent-v041/validation.jsonl --train-data data/hotpot-agent-v041/train.jsonl --limit 30 --corpus-limit 100 --max-input-tokens 768 --output reports/v085-stop-768
python -m mini_agent_rl.stop_probe --data data/hotpot-agent-v041/validation.jsonl --train-data data/hotpot-agent-v041/train.jsonl --limit 30 --corpus-limit 100 --max-input-tokens 1024 --output reports/v085-stop-1024
python -m mini_agent_rl.stop_probe --compare-left reports/v085-stop-768 --compare-right reports/v085-stop-1024 --output reports/v085-stop-budget-compare.json
```

每个目录保存原始检索 observation、模型可见 prompt token、压缩/历史删除统计、逐题动作和独立 split 审计。输出目录已存在时命令拒绝覆盖。

## GRPO 决策门槛

训练候选固定为 train 前 20 题，诊断固定为 validation 前 30 题；两侧均从各自前 100 题构造固定语料。先检查 ID 和规范化问题无交叉，随后验证输入重放、奖励独立重算、action mask、固定 reference、LoRA-only 更新及至少 50% 的非零方差 group。

仅全部通过后，才能用同一 SFT 起点为 seed 42、43、44 分别运行 5×4；每个通过后再从原始 SFT 独立运行 10×4。训练保持 BM25-v2、768 token、既有 reward 与超参数，不改 reward 或把前置历史探针放进训练。OOM、非有限值、基础设施失败、显存超过 7.4 GiB、采样截断超过 5%、重放不一致或 checkpoint 不能重载时，停止该 seed 及后续扩大。

最终只在 100 条开发集和本探针上比较 SFT/三个 GRPO seed，并完整报告每个 seed。封存 final-test 不读取；重复搜索继续为零时，只报告地板效应，不能主张反重复收益。

## 本轮决定

本轮不运行 5×4 或 10×4 多 seed GRPO，也不创建版本标签。直接原因是两个无梯度停止探针均未通过 5% 输入截断 gate；另外，当前 reward 没有显式区分“空检索后依据不足仍回答”和“证据充分后有效停止”，不能把这个探针直接转换为训练信号。

下一阶段应保持模型、检索器、数据和 reward 不变，先以单一变量改进 tool observation 的证据压缩表达（不是删除困难 case），重新运行相同 120-case probe。只有预算输入审计通过、训练正确性审计通过且训练候选具有足够 reward 方差后，才执行三个 seed 的独立 GRPO 对照。
