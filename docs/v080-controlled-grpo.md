# v0.8.0 受控 GRPO

v0.8.0 将每条 rollout 的 action-token loss 先聚合，再对 rollout 等权平均，避免长回答在 batch 中获得过高权重。reference policy 使用同一基础模型临时禁用 adapter；system、user 和 tool observation mask 始终为 0，只有 LoRA 参数更新。

三臂 reward 消融（A 当前基线、B 加强重复搜索惩罚、C 加强 exact-match 奖励）必须从同一 SFT checkpoint 独立启动，使用独立 SQLite 和 checkpoint 目录。训练前先通过 `audit-rl-signal`，训练中记录 loss、KL、clip fraction、grad norm、显存和策略指纹。未通过门槛时只保存报告，不扩大数据规模。

本次执行在当前运行环境被阻断：PyTorch 为 CPU-only（`Torch not compiled with CUDA enabled`），无法满足 RTX 4060 训练验收条件。因此未伪造 A/B/C 训练结果，也未创建 `v0.8.0` 标签；需在用户的 CUDA 环境重新执行。

后续已在 `mini-agent-rl` Conda CUDA 环境重试。A 臂以 4-bit QLoRA、512 token 上下文完成 5 个任务 × 4 rollout：3 个有效更新 group、2 个零方差跳过 group，192 个 LoRA 参数发生变化，loss/KL/grad norm 均为有限值，且无 OOM。训练记录的 PyTorch 峰值分配值最高为 9.811GB，超过 8GB 显存门槛；因此按预注册门槛，不继续 B/C、10×4 或开发集对照，也不创建 v0.8.0 标签。
