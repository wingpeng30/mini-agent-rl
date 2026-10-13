# Goal AO：材料准入停止记录

**决定：按冻结计划停止，不启动训练或GPU评价。** 8组固定材料中有1组存在历史动作答案泄漏；计划明文规定任一组不合格即停止，不换组、不缩组、不扩候选。运行目录、历史AM/AN档案和该次审查均保留。

## 核验结果

远程AutoDL上只进行了CPU材料核验，RTX 4090 D未用于生成或训练。证据账本见[`evidence-audit.json`](evidence-audit.json)，汇总见[`preparation-status.json`](preparation-status.json)。

| 检查 | 结果 |
|---|---:|
| 固定材料组 | 8/8 |
| 原始生产输入用真实Qwen chat template精确重建 | 8/8 |
| 原题在首个用户消息中唯一定位 | 8/8 |
| 新答案出现在该题实际工具正文 | 8/8 |
| 历史assistant动作不含新答案 | **7/8** |
| 可进入冻结训练材料 | **0组**（整体门槛未通过） |

Believe Tour组 (`musique:2hop__407314_80178`) 的原问题是“Who sings Never Say Never with the performer who gave the Believe Tour?”，新问题询问“Who gave the Believe Tour?”，答案为 `Justin Bieber`。新答案确实由实际返回正文支持，`Believe Tour`文档token 131–146的原句是“The Believe Tour was the second concert tour by Canadian pop star Justin Bieber.”；但原轨迹的一次历史助手搜索动作已是 `{"action": "search", "query": "Justin Bieber"}`。因此衍生问答可以通过复制历史查询直接答出，无法区分模型是否利用当前问题绑定关系，属于预先禁止的答案泄漏。

剩余7组的新答案也各自出现在相同真实正文中，并且未在其历史assistant动作中发现答案字符串。例如：Patrick Donner对应“The British Conservative MP Patrick Donner was Ossian's son.”（token 396–408）；Burt Kennedy正文同时列出生卒日期（token 595–635）；Louis King正文同时列出生日期和地点（token 523–591）；John Cale正文列出父母（token 1043–1085）；Richilde正文列出其死亡日期（token 140–175）；Sri Lanka正文列出共和国日期（token 1312–1352）；Carpenter正文列出ZIP code（token 1042–1065）。原文、文档ID、正文哈希和token位置都保存在证据账本中。

## 执行边界与现场

冻结的8组、问题、来源与标准没有改动，也没有替换失败案例。输入重建与token位置使用AutoDL上的实际tokenizer/模板完成。准备器调试过程中出现的6次结构/API检查失败记录在[`preparation-attempts.jsonl`](preparation-attempts.jsonl)；每次均发生在生成材料冻结前，GPU生成、搜索、优化器更新始终为0。

因全组准入未通过，没有创建正式AO执行manifest/训练授权，也没有运行CPU训练预检、AM状态基线、seed42 A/B、开发搜索或seed43。故无AO模型收益结论；这是材料识别性问题，不是模型机制的负结果。AM和AN既有结果不变。

## 唯一后续含义

本轮严格设计不能在不改题或历史的情况下完成。按原计划不将7组合格项临时缩成新实验。若之后继续研究同正文关系绑定，应先重新规划一个带固定规模的新方案，明确如何消除历史搜索答案泄漏且保持A/B完全匹配；这需要单独冻结，不能悄悄改写本轮失败材料。
