# 当前项目状态（2026-10-12）

这是新会话了解个人项目时优先读取的入口。工作目录为`D:/AgentLightning/mini-agent-rl`；父目录README介绍上游Agent Lightning，其成果不能计入个人项目。

## 当前成果与收尾状态

实习面试项目，固定Qwen3.5-0.8B。已实现真实多步检索Agent、LoRA轨迹SFT、BM25/MiniLM向量＋RRF混合检索、GPU真实batch生成、逐轨迹停止、异步工具执行、在线GRPO及完整训练/评价账本。

2026-10-12最终实验全部完成：AM基线、batch4效率、继续SFT seed42、在线GRPO seed42/43及裸0.8B，六条路径各评价同一128题。继续SFT完成128次更新；GRPO两个seed各采样512条真实搜索轨迹、实际更新108/110次。最终adapter、恢复点、配置、账本、曲线和综合指标已取回，归档SHA256校验通过并发送本地确认。远端关机流程已触发，原SSH被远端关闭、新连接拒绝；命令返回回执未取得。

没有待执行训练或评价。当前优先整理面试、演示和简历；新实验按用户新要求开展，不自动恢复旧计划的探针、灰度、门槛、新数据或Goal。

## 最新质量与效率结果

| 分支 | 自然104题归一化EM | 自然F1 | 全128题归一化EM | 全F1 |
|---|---:|---:|---:|---:|
| AM seed42-A＋混合检索 | 47/104 | 0.5907 | 67/128 | 0.6466 |
| AM＋继续SFT seed42 | 49/104 | 0.5855 | 68/128 | 0.6447 |
| AM＋在线GRPO seed42 | 40/104 | 0.5074 | 61/128 | 0.5868 |
| AM＋在线GRPO seed43 | 46/104 | 0.5847 | 66/128 | 0.6480 |
| 原始裸0.8B，同Agent协议 | 0/104 | 0 | 0/128 | 0 |

batch4吞吐12.71→35.34题/分钟（2.78倍），128题耗时604.11→217.34秒，输出token/s为10.76→29.93，峰值reserved显存2.04→4.31GiB。同题EM67→64，答案文本变化6题；不声称与串行逐token相同。

GRPO没有稳定提升。继续SFT自然题EM增加2题但F1略降，不能宣布稳定收益。裸模型128题未输出最终答案：113题重复查询、14题无效JSON、1题调用预算用尽。零分衡量工具协议表现，不是直接知识问答能力。全部128题均已暴露，EM/F1不是新确认或新增语义支持成功率。

## 模型身份与历史口径

本轮实验起点是AM seed42-A＋混合检索，三个新训练分支均独立从相同AM初始化；两个RL seed是追加优化的随机性对照。没有按评价分挑中间checkpoint，也没有自动切换旧演示默认adapter。

AA历史发布模型及部分旧演示仍为Goal Z seed42。AA的64题证据支持评分、BD的208题回归和本轮128题EM/F1分开报告。旧文档所写“正式模型Goal Z”“最后一轮BD”“尚未实现训练”只适用于旧阶段，不表示本轮未完成。

## 结果入口

- [结果解释、负结果与两条简历表述](docs/final-online-grpo-20261012-results.md)。
- [最终原始报告](runs/final-online-grpo-20261012/results/REPORT.md)与[配对逐题变化](runs/final-online-grpo-20261012/results/summary.json)。
- [综合指标](runs/final-online-grpo-20261012/results/comprehensive-metrics.md)及[JSON](runs/final-online-grpo-20261012/results/comprehensive-metrics.json)：答案P/R/F1、完成率、终止原因、检索/可见覆盖、重复、调用/token/延迟分布、吞吐显存、奖励/KL/clip/梯度/ratio、候选多样性与SFT过程NLL。
- [实现与复现配置](docs/final-online-grpo-20261012.md)、[执行取回记录](runs/final-online-grpo-20261012/execution-context.json)、[关机观测](runs/final-online-grpo-20261012/shutdown-evidence.json)。
- 最终adapter：`runs/final-online-grpo-20261012/results/branches/{sft42,grpo42,grpo43}/adapter-final`。

历史结果保留在原文件。新增成果时同步README、任务摘要、新会话交接、实验索引和面试指南。
