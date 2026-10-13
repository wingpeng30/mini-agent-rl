# Goal AQ：单次生成中核对关系，再决定搜索或回答

## 冻结目标

先检验同一0.8B模型在同一次生成中列出实体—关系—对象和原句，再输出动作，是否改善证据使用和真实搜索。AP文档块呈现已停止：AM完整探针20→19/24，修复1个绑定错误、丢失2个正确回答。本轮恢复legacy-json，不训练、不增数据、不补检索、不使用新确认集。

主试验AM seed42-A，哈希3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c；Goal Z仅描述性探针复查，发布身份由AA记录核对，哈希d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52。

## 唯一系统差异

A为direct-v090；B为direct-v090-grounded-action-v1，完整提示：

```text
You are an evidence-driven retrieval agent. Output exactly one JSON object.
Before the final action, list the necessary links supported by visible document bodies. For each link, give subject, relation, object, document_id, and an exact quote. Keep entity roles, relation direction, time, and scope. Related people or matching names do not prove a link.
Use "links" for zero to two supported links and "missing_relation" for what remains unproved.
Then output "action":"search" with "query", or "action":"answer" with "answer".
Answer only when the necessary links are supported; otherwise search for a known entity and the missing relation. Use visible evidence only and at most three searches.
```

B一个顶层JSON：links数组0至2条，每条subject、relation、object、document_id、quote均为字符串；missing_relation字符串；最后action与answer/query。首个完整JSON停止。核对字段只诊断，不干预派发、不强制回答、不补写、不挑后续JSON、不重试。沿用原动作解析及search别名兼容；历史仍保存canonical动作，不注入生成核对内容。引用限当前实际输入正文；标题、历史回答与离线解释不得作为证据。机械引用命中不等于语义蕴含。

共同条件：远程AutoDL单卡BF16、不量化、greedy、thinking=false、Goal Q、legacy预算、legacy-json、BM25-v2/top-3、16384输入/2048输出、每题4调用/3搜索。系统提示和输出协议联合变化，只能称系统行为收益，不能称权重能力改善。

## 准备、阶段与额度

独立目录；冻结代码、配置、模型、tokenizer、题目、2048篇开发语料和评分依据。生产_inputs CPU stub仅设备为CPU，不加载权重；逐条验证A等于原档案token、B仅系统变化、无裁剪、正文哈希/顺序一致、原句字符/token位置可追踪。32状态=24完整+8部分；采用已有修订关系依据，不继承错误旧引用。离线目标不进入输入。

第一阶段AM/Z各A/B32决策，共128有效探针、零搜索。身份/输入/生成条件核对后复用AP64条A，新增64次B。复用失败停止，不补跑。AM门槛：B完整正确≥22/24、比A≥2；新增正确且生成关系/引用正确≥2，至少1个实体/角色/关系对象修复；原成功丢失≤1；部分合法补链≥7/8且≥A，提前回答/无效/重复不增。Goal Z不得用未裁定记录计算正确率或选模型。

第一阶段通过才AM先A后B各原64题真实搜索。总新增上限576次生成、384次真实搜索、128轨迹；0训练、0新确认。搜索门槛：支持成功≥23/29且比A净增≥2；错误合计≤A且≤7；动作问题题数和各计数不增；保留Rapana/公爵，7保护题丢失≤1；至少2新增支持成功有正确生成关系及引用。

## 评分、停止与交付

最多128探针和58准入答案组合一次主agent裁定，重复组合复用；仅实际输入和已有依据。固定64分母EM/F1、动作、搜索、长度、延迟；29准入支持成功/完整链/错配/错误合计，35未确认不强判。分别评分引用存在但错绑、缺链猜测、核对正确但动作错、同证据绑定修复、新取证、及时回答与措辞变化。新增/丢失保存首次分歧、必要边返回和引用位置。失败探针报告无需正式评价文件。

身份/哈希/GPU/数值/基础设施失败保存现场停止，不重跑/降预算/改参；语义失败正常计分。测试JSON嵌套/转义/长度、角色/亲属方向/出生来自/发行签约/日期范围、默认入口和核对不干预动作。

探针及搜索均通过才准备训练题真实失败前缀的原提示动作蒸馏，另冻材料、匹配对照、更新和双seed；评价题不进入训练。探针过搜索失败定位闭环；探针失败结束本格式，不扩大数据或调协议凑收益。不发布。沿用已授予GPU权限，不再启动确认。

执行入口：`PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_aq.py prepare`；随后`probe --model AM`、`probe --model Z`、一次裁定、`gate`；仅passed=true后`evaluate --arm A`再`evaluate --arm B`。
