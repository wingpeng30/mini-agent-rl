# Goal AB：证据充分性决策监督

状态：`complete_development_gate_failed`。

## 门槛与结果

Goal Z 开发题中停止决策错误：17/64；预设门槛 8。

合格决策对：54/64；预设门槛 32。

128 条训练轨迹的停止决策错误：52；预设门槛 16。

产品发布：False；推荐 adapter：`/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final`。

本轮语义审核由代理辅助、主 agent 裁定；不是独立人工金标准。新确认题只有开发门槛通过后才运行。

## 开发与回归分数

| 集合 / 模型 | 语义成功 | EM | 事实错＋无证据 | 无效动作 | 停止错误 |
|---|---:|---:|---:|---:|---:|
| aa_regression-42-A | 36 | 26 | 22 | 0 | 23 |
| aa_regression-42-B | 38 | 27 | 20 | 0 | 23 |
| aa_regression-43-A | 38 | 29 | 20 | 0 | 20 |
| aa_regression-43-B | 39 | 29 | 19 | 0 | 19 |
| aa_regression-baseline | 32 | 26 | 23 | 0 | 24 |
| dev-42-A | 38 | 27 | 21 | 0 | 20 |
| dev-42-B | 39 | 26 | 20 | 0 | 19 |
| dev-43-A | 36 | 26 | 24 | 0 | 19 |
| dev-43-B | 37 | 27 | 24 | 0 | 19 |
| dev-baseline | 36 | 24 | 19 | 0 | 15 |
| old-42-A | 27 | 21 | 5 | 0 | 6 |
| old-42-B | 26 | 20 | 6 | 0 | 6 |
| old-43-A | 26 | 20 | 6 | 0 | 6 |
| old-43-B | 26 | 20 | 6 | 0 | 6 |
| old-baseline | 27 | 21 | 5 | 0 | 8 |
| y-42-A | 43 | 37 | 5 | 0 | 6 |
| y-42-B | 43 | 36 | 5 | 0 | 7 |
| y-43-A | 43 | 37 | 5 | 0 | 5 |
| y-43-B | 43 | 37 | 5 | 0 | 5 |
| y-baseline | 40 | 36 | 8 | 0 | 11 |
