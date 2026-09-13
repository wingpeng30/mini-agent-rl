# v0.4.1：HotpotQA 多跳检索数据

本项目使用 HotpotQA `distractor` 配置构造英文多跳检索轨迹。HotpotQA 提供问题、答案、上下文和 supporting facts，适合作为两次搜索的监督数据。数据集采用 CC BY-SA 4.0，原始出处为 Yang et al., EMNLP 2018。

```powershell
pip install -e ".[data]"
mini-agent-rl download-hotpot --output-dir data/hotpot-agent-v041
```

默认通过 streaming 读取，不下载完整数据集；保存 600 条 train、100 条 validation、100 条 test。每个样本只保留答案所需的支持文档，并按支持文档标题进行稳定 split，防止同一支持文档跨 split 出现。

这里的 `test.jsonl` 是**开发评测集**，可用于训练期间的多次模型选择和失败分析，不能作为无偏最终结果。v0.7.2 为最终结果单独建立不可见集合：

```powershell
mini-agent-rl prepare-final-test --source-dir data/hotpot-agent-v041 --output-dir data/hotpot-agent-v072 --count 300
mini-agent-rl audit-final-test --source-dir data/hotpot-agent-v041 --final-dir data/hotpot-agent-v072
```

构建器从同一 HotpotQA `distractor/train` 流中顺序筛选，但会排除已有三个开发 split 的 task ID 和全部 supporting-document 标题，同时确保最终集内部标题不重复。`manifest.json` 保存三个来源文件、最终文件和 task 顺序的 SHA256。审计通过必须同时满足：无重复 final task ID、无 task ID 重叠、无 supporting-document 重叠、manifest 哈希匹配。

最终集只能在 checkpoint 冻结后运行一次：

```powershell
mini-agent-rl evaluate-final --adapter-path checkpoints/qwen35-08b-grpo-v071-seed43 --confirm-final
```

`--confirm-final` 是有意设计的人工确认点；如需再次调参，应回到 validation/development 集，不应反复查看 final 指标。

生成结果不提交 Git；`manifest.json` 记录数据来源、许可证、扫描数量、样本数量和文档数量。该数据用于训练通用的多跳检索动作，不替代中文离线语料。

使用 `mini-agent-rl audit-sft --data-dir data/hotpot-agent-v041` 可生成审计报告，检查每个 split 的样本数、问题重复、搜索步数分布、支持文档数及跨 split 文档重叠。
