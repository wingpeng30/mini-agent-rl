# 最终面试版：0.8B多跳搜索Agent

## 最新面试口径（2026-10-12）

先读[当前项目状态](../PROJECT_STATUS.md)。最新成果包括混合检索、真实GPU batch4与异步工具执行、继续SFT和在线GRPO双seed。AM全128题EM67/128、自然47/104；SFT68/128、49/104；GRPO61/128、66/128，自然40/104、46/104，没有稳定提升。batch4吞吐12.71→35.34题/分钟（2.78倍），同时EM67→64。裸模型未完成固定Agent协议答案输出，EM0/128。

当前两条简历表述、面试追问、指标和adapter位置见[最终结果与面试说明](final-online-grpo-20261012-results.md)。本轮EM/F1与AA/BD历史证据评分分别叙述；AM实验起点与旧演示默认Goal Z分别说明。

## 历史面试指南：Goal BD（2026-10-06）

下方为BD当时的成果、演示和冻结状态，“最后一次”“当前”仅指该阶段，不覆盖10月12日已完成实验。

## 可以证明的成果

正式模型为AA发布的Goal Z完整搜索轨迹SFT seed42，基础模型Qwen3.5-0.8B。AA当时冻结的64题新确认：Goal T起点12/64，轨迹SFT seed42为35/64、seed43为36/64；严格EM为9/64、26/64、25/64。有限封闭语料、代理辅助审核与主审裁定，不冒称开放域或独立人工金标准。两个seed各31题的第二侧支持正文由后续搜索首次取得。平均搜索次数与延迟随之增加，收益不是免费。

AM普通状态动作SFT在已有开发准入29题上19→21，固定64题动作问题8→1；完整证据探针19→20/24未过预设门槛，未运行seed43或发布。新增案例Rapana通过新查询取得公告关系，公爵在证据完整后及时回答。收益属于整个“原128轨迹＋34状态”配方，不能把全部增益单独归因于34状态。

BD是最后一次固定权重回归，结果见`runs/v133-goal-bd-final-regression-v1/REPORT.md`与`decision.json`。这208题全已暴露，只叫回归；开发29题证据评分，其他179题只计EM/F1与机械指标。正式模型始终Goal Z。

最终BD复核：支持成功19→21/29，错误7→7，全208题严格EM106→111、动作问题12→3；AM取得局部收益及机械保护标记。开发／旧主要／旧回归／Goal Y／AA的严格EM分别23→26、13→13、8→8、36→37、26→27。旧两组F1略降；实体关系错配标签7→8，不能描述为全面绑定能力改善。416轨迹、1141生成、725搜索完成，17项CPU测试、1141个输入及108处引用核验通过。项目不再追加实验。

## 三个固定演示

默认离线回放，不加载权重、不冒充实时推理。代码仓库根目录执行：

```bash
PYTHONPATH=src:scripts python scripts/demo_v133_final.py
PYTHONPATH=src:scripts python scripts/demo_v133_final.py --model am-research
PYTHONPATH=src:scripts python scripts/demo_v133_final.py --case-id 2wiki:f299dd600bdd11eba7f7acde48001122 --model am-research
PYTHONPATH=src:scripts python scripts/demo_v133_final.py --case-id musique:2hop__96194_78303 --model am-research
```

三个案例分别是Rapana、Medinaceli公爵、Hero and the Terror／Walker。保持本轮原始表现，不按结果挑换案例。显示每步query、top3正文、输入输出长度、最终动作、模型/语料哈希和历史延迟。

实时模式在AutoDL使用仓库`.venv/bin/python`，调用同一Goal Q循环。它使用明确标注的BD宽窗研究preset，不改旧正式入口的预算：

```bash
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_final.py --mode live --model goal-z
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_final.py --mode live --question '你的问题' --corpus /absolute/path/to/matching-corpus.json
```

手动live不属于BD评分及冻结额度；不要在正式实验中追加演示生成。问题不在冻结语料时检索可能为空，不能用模型凭记忆答中代替证据支持。

## 面试叙述

项目：0.8B多跳检索Agent后训练与证据驱动评测。

- 构建本地小模型、BM25检索、JSON search/answer协议、有限调用预算和完整轨迹记录的Agent。
- 完整轨迹LoRA-SFT在64题冻结确认上将证据支持成功12→35，另一训练seed复现36；保留来源、评分和模型身份记录。
- 实现助手token mask、轨迹/状态损失归一、LoRA重载、有限值检查以及实际输入token级证据定位；通过匹配对照分开取证、关系绑定、停止和措辞变化。

按实际参与程度写“开发”或“主导并使用AI辅助实现”；应能解释主要实现和结果。不要声称所有RL有效、教师蒸馏已完成、AM已发布、最终208题是新确认或全面语义保护通过。

## 常见追问

1. 为什么不是只训练答案？完整搜索轨迹监督提供了真实下一步查询与继续搜索行为；Goal Z答案SFT未取得同等收益。
2. 怎样避免评价泄漏？训练和评价来源隔离；运行时只传题面和实际工具正文；金标用于事后机械评分，不进入模型输入。
3. 为什么EM不够？命中金标可能缺必要关系；从actual token与正文逐边判断主体、角色、关系、对象和限定。
4. 目前仍失败在哪里？完整证据下错选人物或角色、混淆出生/死亡、缺限定；也存在查询未补到必要边。不能都归咎于上下文长度。
5. 负结果有什么作用？明确淘汰已冻结配方：关系提示、难例/字段加权、配对监督、固定池GRPO、当前重排及CAD没有稳定新版本；没有把门槛未触发的AX/AY/AZ写成已训练失败。

## 冻结与复现

当前BD-1至BD-4已完成，六个离线回放通过；BD-5因脚本要求不存在的.gitignore而停止。下面描述是待完成的快照结构，尚未生成freeze-manifest.json，不把已同步结果冒称完整冻结。参见`runs/v133-goal-bd-final-regression-v1/FREEZE_STATUS.md`。

BD的`freeze-manifest.json`记录代码哈希、Git HEAD、依赖、资产位置和快照SHA256；`remote-assets.json`记录模型/语料/完整轨迹。小型回放和代码快照同步本地。已有Git修改保留，不批量提交/清理；密钥、`.env`、私密API日志、缓存和模型权重本体不进入代码快照。

只有文件哈希、配置、软件与模型身份一致时复现才有意义；GPU数值不承诺跨机器逐bit一致。既有GPU产物拒绝覆盖或补跑。最终验收后不再追加实验。
