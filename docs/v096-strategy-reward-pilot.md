# v096：合法但答错的策略奖励诊断与预注册 pilot

## 诊断目标

v095 的总体收益主要来自非法 JSON 消失；三个 seed 在共同合法轨迹上的 EM/F1 区间仍跨越零。本轮不扩大任务数，先验证合法但错误的回答是否主要发生在证据未完整或检索为空后停止。

## 冻结候选 reward

除 v095 原有 reward 外，唯一新增项如下：

- `evidence_stop = +0.20`：答对且 supporting evidence 完整后停止；
- `incomplete_evidence_answer = -0.15`：已经搜索、却在 supporting evidence 不完整时最终回答；
- `empty_evidence_answer = -0.10`：所有已执行搜索均为空时最终回答。

supporting titles 只在轨迹结束后使用；不会放入 prompt、工具结果、检索、裁剪或评测输入。该版本不改变模型、LoRA 起点、温度、prompt、语料、动作协议或评测题。

## 进入 pilot 的门槛

对 v095 corrected 三个训练数据库执行只读反事实重算。仅当三组都可读、累计至少 10 条合法错误轨迹会被候选 reward 改变，且至少 6 个 rollout group 受影响，才启动 pilot。

pilot 固定为 checkpoint-200、前 10 个训练题 × 4 rollout、语料前 100 题、seed 42/43/44；BF16、2048/128、BM25-v2、compact-v3、temperature .7、LR 5e-6、clip .2、KL .02、policy epochs 2、max steps 3、21GiB。每个 seed 至少 5 个有效更新组、保存/独立重载通过才进入固定 20 题快速回归。

任何 seed 的 EM 相对 SFT 降低超过 10pp 或非法动作增加超过 5pp，停止，不扩量。三 seed 全部通过快速回归后才运行 100 题开发评测与 120-case 探针；final-test 不访问。

## 已执行结果

只读反事实诊断满足进入门槛：seed42/43/44 分别有 10/9/9 条合法错误 rollout 的 reward 被候选改变，影响 24/22/25 个 group。空结果项在这些旧轨迹中没有触发；它保留在冻结协议中作为安全行为约束，而不是事后移除。

新代码下同实例、BF16 跨加载与 FP32 诊断均通过，随后完成三 seed 的 10×4 pilot。seed42 和 seed43 各只有 4 个有效（非零方差）更新组，低于预注册下限 5；seed44 恰有 5 个。因此整轮为 **GATE_FAILED**：无基础设施、有限值、重载或 21GiB 显存失败，但没有完整三 seed 可比较训练结果。没有执行 20 题或 100 题效果评测，也没有挑选 seed44。

下一步不是扩量或放宽门槛。应先用新的独立采样诊断确定为何 10 题中 5–6 组仍零方差；若要测试采样多样性，必须预注册单变量（例如 group size 或 temperature）并重新进行三 seed pilot。
