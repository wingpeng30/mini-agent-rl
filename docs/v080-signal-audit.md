# v0.8.0 RL 学习信号审计

本版本首先以只读方式检查已有 SQLite 轨迹：reward、group advantage、action token、logprob 和 mask。基础设施失败、非有限值、长度不一致会阻止训练。审计不会读取封存的 final-test 数据，也不会修改历史轨迹。

命令：

```powershell
mini-agent-rl audit-rl-signal --db runs-v071-seed44.db --output reports/v080-signal-audit.json
```

`infrastructure_failures` 必须为 0，非零方差 group 至少占一半，且所有数值有限后才允许进入 GRPO。报告区分事实检查和训练门槛，不把小样本审计解释为泛化结论。
