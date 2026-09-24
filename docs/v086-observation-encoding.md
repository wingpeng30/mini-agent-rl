# v0.8.6：证据表达压缩与真实输入审计

本版本保持 Qwen3.5-0.8B、SFT 起点、BM25-v2、原始 SearchTool 结果和 reward 不变，只改变工具 observation 进入模型上下文的表达。

`legacy` 保留历史编码，供轨迹重放和对照；`compact-v1` 先使用紧凑 JSON、移除模型上下文中的 score，并将同一文档的后续出现编码为稳定 ID 引用。原始 observation 永远保留在审计记录中。预算仍不足时，正文只在完整句子边界缩短，并按原始文档排名处理；system、问题和动作历史保持完整。

每个 case 保存原始 observation、模型可见 messages、最终 prompt token IDs/attention mask 和逐文档 change record。审计分别报告原始 support 覆盖、模型可见 support 覆盖、无损表达变化、有损正文裁剪、文档删除、历史删除和预算错误。历史 `prompt_truncated` 保留，用于兼容旧报告，不把无损表达缩短写成证据丢失。

本轮运行前先执行 tokenizer-only 审计：

```powershell
mini-agent-rl audit-stop-probe --max-input-tokens 768 --observation-encoding compact-v1 --output reports/v086-budget-audit-768.json
mini-agent-rl stop-probe --max-input-tokens 768 --observation-encoding compact-v1 --output reports/v086-stop-768
mini-agent-rl stop-probe --max-input-tokens 1024 --observation-encoding compact-v1 --output reports/v086-stop-1024
mini-agent-rl compare-stop-probe reports/v086-stop-768 reports/v086-stop-1024 --output reports/v086-budget-compare.json
```

768-token 只有在无基础设施失败、reserved 不超过 7.4 GiB、历史删除与预算错误均为 0、有损正文裁剪不超过 5%、模型可见 support 保留率至少 95% 时，才能进入既有训练正确性检查和后续多 seed GRPO。封存 final-test 不读取。

项目的长期下载约束：任何单次下载或同一安装任务预计累计超过 1GB，均必须先获得用户明确确认；大小未知时先查明，不能拆分下载绕过该约束。

## 实际运行结果（2026-09-17）

本轮没有下载数据、模型、解释器或依赖。tokenizer-only 审计和两个 120-case 真实探针均使用本地 `D:\qwen_08b` 与已有 SFT adapter。

| 预算 | 有损正文裁剪 | 文档删除 | 可见 support 保留率 | 历史删除/预算错误 | reserved 峰值 |
|---|---:|---:|---:|---:|---:|
| 768 | 31.7% | 5 | 98.3% | 0% / 0 | 3.447 GiB |
| 1024 | 5.8% | 1 | 100.0% | 0% / 0 | 3.539 GiB |

两档均未触发 OOM、NaN 或基础设施失败。`prompt_truncated` 在 compact-v1 中为 100%，因为该历史兼容字段把无损紧凑 JSON 与重复引用造成的 token 减少也统计在内；它不能单独解释为证据损失。新口径表明 768 仍有 31.7% case 实际裁剪正文，1024 也以 5.8% 略高于预注册 5% 门槛。

768 与 1024 的动作类别/query 没有变化，但 9/120（7.5%）答案文本变化；1024 相对 768 的聚合 strict EM/F1 差值为 `+5.0pp/+0.037`。compact-v1 相对 v0.8.5 legacy 的 768 对照有 40/120 答案文本变化，F1 增加 `0.031`、strict EM 无变化。这些均来自单次冻结推理与跨加载比较，不能说明泛化提升或训练收益。

v0.8.6 未通过 768-token 训练门槛，因此未运行 sampled rollout、GRPO、三 seed 冒烟、10×4 正式训练或开发集矩阵，不读取封存 final-test，也不创建版本标签。下一轮应在不改变模型、检索器、reward 或题目集的条件下，继续降低 observation 的有损正文裁剪率，并用同一 120-case 协议复验。
