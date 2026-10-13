# Goal U→V：双侧证据偏好训练与新题验证

## 决策

**维持 Goal T seed 42、第 32 次更新的 adapter 为产品默认模型。** 针对单侧证据的偏好训练在已暴露旧开发集上，两 seed 均带来语义改善；但冻结的新题集上严格 EM 为 20/32，低于起点的 21/32，语义充分性也未提高。因此未通过预先冻结的产品发布门槛。新 checkpoint 全部保留为实验产物，未切换产品入口。

本轮在 AutoDL `ssh -p 25320 root@connect.cqa1.seetacloud.com` 执行，实验目录为 `/root/autodl-tmp/mini-agent-rl/runs/v133-goal-u-v-evidence-preference/`。完整原始记录、冻结哈希、逐题输出和决策分别见 [`config.json`](config.json)、[`summary.json`](summary.json)、[`decision.json`](decision.json)。累计 GPU 进程时间 714.3 秒；此值仅作资源记录，没有时间停止门槛。

## 训练材料与对照

16 道原训练题中，8 道能够从 Goal R 保存的实际可见双侧证据构造同一属性的有效 chosen/rejected 对；其余 8 道保留排除理由。每对的证据原文、支持文档 ID、chosen、rejected、原始输入 token 和 mask 见 [`pair-audit.json`](pair-audit.json) 与 [`pairs.json`](pairs.json)。没有使用旧开发题或新题构造训练对，也没有将答案或支持映射注入模型输入。

两臂都从同一个 Goal T 产品 adapter 独立起步，seed 42/43 各 32 次更新，使用相同的冻结题目顺序、学习率 5e-6 和 AdamW 参数。A 仅对 chosen 最终回答 token 做平均交叉熵；B 用 chosen/rejected 的**序列 logprob 差**做 DPO，beta=0.1，冻结 reference 始终为原 Goal T adapter。两个回答的长度不同属于该干预的一部分；不能把两臂的 loss 数值直接当成同一指标比较。四个终点均通过 LoRA 变化、base 不变、独立重载一致性核查；各 32 次更新、192 个 LoRA 张量变化，重载最大差值为 0。训练日志见 `seed42-a/updates.jsonl` 等四份文件。

| 模型 | 旧集主要/16 | 旧集回归/16 | 旧集总 EM/32 | 旧集语义充分/32 | 单侧证据错误/32 | 新集 EM/32 | 新集语义充分/32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Goal T 起点 | 13 | 7 | 20 | 23 | 6 | 21 | 29 |
| seed 42 A：chosen SFT | 13 | 8 | 21 | 23 | 6 | — | — |
| seed 43 A：chosen SFT | 13 | 8 | 21 | 23 | 6 | — | — |
| seed 42 B：偏好训练 | 13 | 8 | 21 | 25 | 5 | 20 | 29 |
| seed 43 B：偏好训练 | 13 | 8 | 21 | 25 | 5 | 20 | 29 |

旧集全部 32 题、各模型均实际召回两侧支持文档，无效动作均为 0。新集三模型也均为 32/32 双侧召回、无效动作 0；因此本轮主要差异发生在最终回答。旧集的完整 F1 和每题轨迹见 `eval-*.jsonl`，新集见 `fresh-*.jsonl`。审查只用实际工具返回的可见文本，审查理由及文档引用见 `old-blind-votes.jsonl` 和 `fresh-blind-votes.jsonl`；这是自动代理辅助的证据审查，并非独立人工金标准。

## 逐题变化与机制判断

旧集严格 EM 的 +1 有不同含义：A 的唯一新正确题是 `g009` 的 `Democratic`→`Democratic Party`，主要是金标措辞；B 的新正确题是 `g001` 的 `film director`→`film director and screenwriter`，补全了两侧证据支持的职业。B 另在 The Rev 与 Mike Sparks 一题把单侧 `drummer` 改为共同的 `musician`；严格 EM 仍为 0，但语义审查转为充分。B 两 seed 均呈现这两项改善，同时 `g010`、`g018` 等单侧误判仍在。A 没有减少旧集单侧错误。固定上下文回放使用起点回答前的**同一实际输入 token**，也观察到这些对应答案变化，支持“回答策略改变”而非检索顺序改变的解释；原始回放见 `fixed-*.jsonl`。

冻结新题集含 96 篇与旧库标题不重叠的文档、32 道从未参与训练的新题，按共同职业、地点属性、对象类别、艺术/电影类别四组各 8 题；one-sided/ordinary 各 16 题。构造、来源及冻结清单见 [`fresh-cases.jsonl`](fresh-cases.jsonl)、[`fresh-sources.jsonl`](fresh-sources.jsonl)、[`fresh-set-freeze.json`](fresh-set-freeze.json)。页面来源记录中 76 项有 revision，20 项只保存规范 URL 与可见摘录，降低了逐字来源复核的强度。

新题两 seed 输出完全相同，只有两题相对起点改变：第 1 题从仅 Sally Ride 一侧支持的 `physicist` 变为仅 Mae Jemison 一侧支持的 `engineer`，仍是单侧错误；第 6 题从金标 `actor` 变为双侧都支持的 `actor and comedian`，严格 EM 从 1 变 0，但语义仍正确。其余 30 题未变化。严格 EM 为起点 21、候选 20；语义分类三者均为 29 充分、1 缺必要类别、2 单侧错误。新题集的基线语义充分率已达 29/32，留给这类干预的可测空间较小；问题也偏短、实体较知名。结论是**在旧集修复了具体单侧错误，但没有显示出新题上的产品收益**，不能由此断言方法在更广分布上无效。

## 训练曲线与工程核查

| 臂 | 前 8 次平均 loss | 后 8 次平均 loss | 前 8 次平均 chosen−rejected logprob | 后 8 次平均 | 最大裁剪前梯度范数 |
|---|---:|---:|---:|---:|---:|
| 42 A | 0.0580 | 0.0127 | — | — | 7.535 |
| 43 A | 0.0568 | 0.0149 | — | — | 7.646 |
| 42 B | 0.6842 | 0.5023 | 5.008 | 9.143 | 12.920 |
| 43 B | 0.6885 | 0.5038 | 4.921 | 9.081 | 12.845 |

曲线说明优化目标确实发生变化，不能据此推出产品收益。训练、评价、固定回放、新集验证和 demo 均完成；最大记录的 CUDA reserved 显存约 6.64 GiB。PyTorch 提示当前 CuBLAS 算子非严格确定，两个 seed 的一致输出也不等于形式上的位级确定性保证。实际基础权重、tokenizer、语料、两套题目、训练输入、代码与起点 adapter 的哈希都保存在 `config.json`；起点 adapter 权重 SHA-256 为 `e1680c2b2a6a60ec89b4b54249e3132e2005ee6eb94bb04e1acdbd4bf76e064e`。

## 复现与产品入口

在远程仓库运行（首次执行需准备相同资产；`run` 后按冻结协议审查旧集再 `advance`，审查新集再 `finish`）：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/validate_v133_goal_v_fresh_set.py
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py prepare
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py run
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py continue
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py advance
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py finish
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_uv_selected.py
```

`prepare`/`run` 拒绝覆盖已开始的目录；复现时使用新的空目录或归档当前结果，不能在当前目录直接重跑。demo 已实测读取 `decision.json`，加载仍为 Goal T 起点的 adapter，完成一次真实搜索与回答。

下一次实验最有信息价值的是改进**新题集的难度与语义判据的独立审查**，再测试更直接的双侧比较监督；不应仅因旧集 +1 追加偏好更新或直接发布。本轮开发集已反复暴露，新题集虽预先冻结，规模仍只有 32，不能视为开放域泛化证明。
