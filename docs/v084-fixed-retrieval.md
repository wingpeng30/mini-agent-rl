# v0.8.4：固定检索环境与可信 GRPO 对照

## 实施边界

本版本把 `task_limit` 与检索语料范围解耦。实验固定使用训练集前 100 题产生的 corpus，5×4 和 10×4 只改变任务前缀，不改变 corpus。封存的 `data/hotpot-agent-v072/final-test.jsonl` 不读取。

检索器保留历史 `keyword-v1`，新增无外部依赖的确定性 `bm25-v2`。v2 过滤英文单字符和 stopword，提升标题权重，并用文档 ID 作为稳定 tie-break。reward、模型起点和 GRPO 超参数保持不变。

## 验收顺序

1. `verify-environment` 验证 task prefix、corpus hash 和 retriever 版本。
2. `evaluate-retriever` 在固定 validation 子集上做 v1/v2 greedy A/B。
3. 通过检索与上下文门槛后，使用 v2 运行 5×4 smoke。
4. smoke 通过后，从同一个 SFT checkpoint 运行 10×4 formal。
5. formal 通过后，才运行固定 100 条开发集对照。

任一硬门槛失败都只保留报告、SQLite 和审计轨迹，不创建可评测 checkpoint 标签。

## 当前已知基线

在前 100 条任务构建的固定语料上，历史 keyword-v1 的 canonical search query 诊断为：训练集 Support Hit@3 `89%`、验证集 `84%`；验证 Top-1 `74.5%`。这些数字只用于检索器诊断，不等同于 Agent 最终答案准确率。

## 复现命令

```powershell
conda activate mini-agent-rl
mini-agent-rl verify-environment --task-limit-small 5 --task-limit-large 10 --corpus-limit 100
mini-agent-rl evaluate-retriever --limit 30 --corpus-limit 100
mini-agent-rl train-grpo --task-limit 5 --corpus-limit 100 --retriever-version bm25-v2 --group-size 4 --load-in-4bit --max-input-tokens 768
```

## 解释边界

本版本只允许在相同环境指纹、数据顺序、语料哈希和 reward 配置下比较 SFT/GRPO。开发集差异需要报告配对置信区间；小样本结果不能作为 HotpotQA 泛化证明。模型、SQLite、轨迹、checkpoint 和机器报告继续被 Git 忽略。

## 实际运行结果（2026-09-16）

阶段 A 环境一致性通过。阶段 B 的 BM25-v2 标准 query Hit@1/Hit@3/MRR 均为 100%；30 条 validation greedy A/B 中 v1/v2 的 EM 为 `20.0%/33.3%`、F1 为 `0.283/0.433`、support coverage 为 `0.400/0.633`，基础设施失败均为 0。

阶段 C 的 5×4 smoke 和阶段 D 的 10×4 formal 均通过：采样截断率均为 0%，reserved 峰值分别为 `2.414/3.141 GiB`，有效 group 分别为 `3/5` 和 `6/10`；两个 checkpoint 均可重载。

阶段 E 的 100 条开发集结果为：

| 模型 | strict EM | normalized EM | F1 | answer completion | invalid action | support coverage | mean reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT v0.5.2 | 33% | 34% | 0.441 | 90% | 10% | 0.595 | 0.515 |
| v0.7.1 seed44 | 37% | 38% | 0.467 | 96% | 4% | 0.620 | 0.605 |
| v0.8.4 formal | 38% | 39% | 0.492 | 96% | 4% | 0.645 | 0.623 |

v0.8.4 相对 SFT 的 paired bootstrap EM 差值为 `+5pp`（95% CI `+1pp,+10pp`），F1 差值为 `+5.1pp`（95% CI `+1.2pp,+9.4pp`）。三个模型的重复搜索率都是 0%，所以“重复搜索下降 5 个百分点”的预注册条件处于地板效应，不能宣称通过。该版本不创建 Git tag；后续应在更丰富的重复/困难任务集上重新注册停止策略指标。

## 验证状态

编译检查和不依赖 `tmp_path` 的 19 项测试通过。完整 pytest 收集到 44 项，其中 29 项通过、15 项因当前 Windows 用户临时目录 `C:\Users\PengZixuan\AppData\Local\Temp\pytest-of-PengZixuan` 返回 `WinError 5` 而在 fixture setup 阶段停止；这些是环境权限错误，不是断言失败。未因该环境问题创建版本标签。
