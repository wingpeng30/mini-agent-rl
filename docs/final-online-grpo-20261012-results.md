# 0.8B最终实验结果与面试说明

本轮所有预定实验和追加裸模型评价均已完成，使用4090D、固定原有训练材料及已暴露128题。远端归档约150MB，已下载、SHA256校验并解包；本地取回确认后关机包装器进入`/usr/bin/shutdown`流程，随后原SSH连接被远端关闭，新连接被拒绝。未取得关机命令返回回执，保留这一观测边界。

原始不可变归档：[results-results.tar.gz](../runs/final-online-grpo-20261012/results-results.tar.gz)，SHA256为`dd01ac20e0b46ac371552155ed246ee22d0d2386d190b5a4011a93db7be4307a`。本说明在取回后补写，不改动原始归档。

## 最终结果

| 分支 | 自然104题EM | 自然F1 | 全128题EM | 全F1 | 单跳24题EM |
|---|---:|---:|---:|---:|---:|
| AM＋混合检索 | 47 | 0.5907 | 67 | 0.6466 | 20 |
| AM＋继续SFT seed42 | 49 | 0.5855 | 68 | 0.6447 | 19 |
| AM＋在线GRPO seed42 | 40 | 0.5074 | 61 | 0.5868 | 21 |
| AM＋在线GRPO seed43 | 46 | 0.5847 | 66 | 0.6480 | 20 |
| 原始裸0.8B，同Agent协议 | 0 | 0 | 0 | 0 | 0 |

EM均为现有别名规则下的归一化EM。主要比较自然104题；构造单跳24题单列。继续SFT自然题新增7、丢失5，EM增加2题，但F1略降，EM差值的配对bootstrap95%区间为[-4.81,8.65]个百分点。不能据此宣布稳定提升。

在线GRPO seed42自然题新增4、丢失11；seed43新增6、丢失7。两seed分别有54/55个混合奖励有效组、74/73个零方差组，共采样1024条真实搜索轨迹，实际执行108/110次更新；训练耗时31.38/31.83分钟，峰值reserved显存17.99/15.86GiB。继续SFT执行128次更新，耗时233.43秒，峰值reserved显存13.48GiB，二者不是等算力比较。

GRPO seed42回答完成率从125/128到128/128，但EM下降，搜索次数243→211；seed43为127/128和230次搜索。既有93道来源已确认题的全部支持文档召回覆盖从AM的81/93到GRPO的72/93、77/93。更少搜索、覆盖下降与自然题质量退步同时出现，是可讨论的行为变化；本实验没有证明因果关系，也没有新增答案蕴含或语义证据审核。35道来源未确认题保留NA。

裸模型没有输出任何最终答案：113题因重复查询停止、14题无效JSON、1题四次调用均为搜索而耗尽预算。它实际调用模型249次、搜索121次，评价耗时1280.04秒。典型错误包括重复同一查询，以及JSON内扩写reasoning/evidence到2048token后字符串未闭合。零分衡量的是固定工具协议下的端到端行为，不能等同于裸模型的知识或直接问答能力。

## 异步批量执行

| 指标 | 串行 | batch4 |
|---|---:|---:|
| 128题wall秒 | 604.11 | 217.34 |
| 题/分钟 | 12.71 | 35.34 |
| 输出token/s | 10.76 | 29.93 |
| 峰值reserved GiB | 2.04 | 4.31 |
| 归一化EM | 67/128 | 64/128 |

吞吐2.78倍，同题答案文本变化6/128。GPU生成真实batch化，轨迹独立停止并逐轮退出；工具任务通过asyncio并发执行，MiniLM编码前向保留现有线程锁。这是整体执行路径的实测收益，没有把全部提速归因于工具并发，也没有声称与串行逐token相同。

## 简历可直接使用

- 构建0.8B多跳检索Agent，结合LoRA轨迹SFT与BM25/MiniLM-RRF混合检索，在已暴露128题回归集取得52.34%归一化EM；实现在线GRPO，完成双seed共1024条真实搜索轨迹及继续SFT对照，记录奖励零方差、KL与逐题正负变化。
- 实现GPU batch4逐轨迹停止与异步工具执行，将128题实测吞吐从12.71提升至35.34题/分钟（2.78倍），峰值reserved显存2.04→4.31GiB；建立答案质量、检索覆盖、调用/token消耗、延迟分布及训练指标完整账本，保留batch答案变化与准确率代价。

面试时可以具体解释：为什么必须等整组结束后更新；为什么全对/全错组没有相对优势；padding/position_ids与logprob重放如何对应；为什么完成率上升但多跳准确率下降；批量提速为什么会改变greedy输出。当前结果支持展示工程实现及对照实验能力，不支持声称在线GRPO稳定提高准确率。

## 完整交付入口

- [原始实验报告](../runs/final-online-grpo-20261012/results/REPORT.md)：分层EM/F1、配对变化、资源与曲线。
- [综合指标Markdown](../runs/final-online-grpo-20261012/results/comprehensive-metrics.md)与[JSON](../runs/final-online-grpo-20261012/results/comprehensive-metrics.json)：P/R/F1、完成率、终止原因、检索/最终可见覆盖、重复查询和文档、token/延迟p50/p90/p95/p99、吞吐显存、训练奖励/KL/clip/梯度/ratio、候选多样性、SFT过程NLL。
- [训练曲线](../runs/final-online-grpo-20261012/results/training-curves.png)、[曲线CSV](../runs/final-online-grpo-20261012/results/training-curves.csv)、[配对明细](../runs/final-online-grpo-20261012/results/summary.json)。
- `runs/final-online-grpo-20261012/results/branches/{sft42,grpo42,grpo43}/adapter-final`：最终adapter；同目录包含恢复点、训练及评价完整账本。
- [实现与复现说明](final-online-grpo-20261012.md)。

无法从账本得到的策略token熵、能耗和新增语义成功率留空。SFT过程token加权NLL/expNLL不能当作最终checkpoint的独立PPL；GRPO surrogate loss也不作为PPL。
