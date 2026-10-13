# v133 实验索引

## 最新已完成实验（2026-10-12）

先读[当前项目状态](../PROJECT_STATUS.md)。AM seed42-A＋混合检索起点的继续SFT、在线GRPO双seed、真实GPU batch4与异步执行，以及裸0.8B评价全部完成；六条路径各128题。全128题归一化EM：AM67、SFT68、GRPO42为61、GRPO43为66、裸模型0；batch4为64，吞吐2.78倍。GRPO无稳定提升，裸模型0分是未完成协议答案输出。见[原始报告](../runs/final-online-grpo-20261012/results/REPORT.md)与[结果解释](final-online-grpo-20261012-results.md)。

## 历史实验索引

下方“最终完成”“当前执行路线”和阶段内模型身份均是历史记录，不是当前待办。

## 最终完成：Goal BD（2026-10-06）

[独立计划](../runs/v133-goal-bd-final-regression-v1/PLAN.md)、[最终报告](../runs/v133-goal-bd-final-regression-v1/REPORT.md)：固定Goal Z/AM42-A，416轨迹、1141生成、725搜索、0训练/教师/新确认。开发支持19→21/29、错误7→7，全208题EM106→111、动作问题12→3；局部收益及机械保护门槛通过。旧两组F1略降，完整证据下角色错绑仍存在。仅开发29题证据评分；旧144题只检查EM/F1。面试版冻结，正式模型不变，不追加实验。

## 当前路线（2026-10-05）

后续路线更新：Goal AP v2已完成同证据探针，但未达冻结门槛，停止文档块呈现路线；未运行真实搜索阶段。参见[AP v2报告](../runs/v133-goal-ap-evidence-presentation-v2/REPORT.md)。v1因工具结果解码包装未还原，A/B实际输入相同，详见[v1有效性修订](../runs/v133-goal-ap-evidence-presentation-v1/INVALID_RUN.md)。

| Goal | 资产 | 当前结论 |
|---|---|---|
| AJ–AK | `runs/v133-goal-aj-*`、`runs/v133-goal-ak-*` | 关系提示最初受多 JSON 协议问题阻断；修复为 first-json 后重新检验，结论以各自报告为准，不将提示行为变化称为权重学习 |
| AL | `runs/v133-goal-al-wide-context-v1/REPORT.md` | 输入/输出预算联合扩大到16384/2048；真实正文裁剪0→0，准入支持成功19→19，未证明扩大预算带来支持成功收益 |
| AM | `runs/v133-goal-am-action-sft-v2/REPORT.md` | 普通状态动作SFT seed42支持成功19→21/29，动作问题8→1；完整探针仅19→20/24，未过继续门槛，未跑seed43 |
| AN | `runs/v133-goal-an-hard-state-sft-v2/REPORT.md` | 集中难例加权seed42支持成功20/29、完整探针20/24，低于门槛并丢失AM Rapana收益；停止seed43 |
| **AO** | `runs/v133-goal-ao-relation-contrast-sft-v2/REPORT.md`；v1审计：`runs/v133-goal-ao-relation-contrast-sft-v1/PREPARATION_STOP.md` | 修复Believe Tour组答案泄漏后完成seed42匹配训练：A支持成功19/29，B20/29，B比A仅+1；完整证据探针两臂均20/24，B错误合计8，未过冻结门槛，因此不跑seed43、不发布。v1停止账本保留 |
| **AP v1** | `runs/v133-goal-ap-evidence-presentation-v1/INVALID_RUN.md` | 128次生成保留，但工具包装还原缺陷导致A/B输入相同，不用于推断呈现收益；A输出有条件复用，B输出作废 |
| **AP v2** | `runs/v133-goal-ap-evidence-presentation-v2/REPORT.md` | 修复工具角色还原后完成128条有效同证据探针（复用64条A、新跑64条B）：AM完整正确20→19/24，B修复1个错绑但丢失2个原正确回答；部分合法补链均7/8。未过门槛，未运行真实搜索/训练/确认，不发布 |

后续路线：有界的真实失败状态蒸馏 → 独立证据呈现或工具辅助实验。AO的同正文关系对照已经完成，不能重做来凑门槛；优先复用训练题上的真实失败状态，保持原搜索轨迹、普通继续训练对照和评价隔离。提示及窗口建议已执行，不作为待办。AO不发布模型；未通过seed42门槛，不运行seed43。

下表为早期 Goal C–T 的历史索引，保留当时记录，不代表当前模型身份或路线优先级。当前唯一发布产品模型仍以 AA 发布记录为准。

---

# 早期 v133 固定证据训练实验索引

当前执行路线：Goal AQ单次生成关系核对，[计划](../runs/v133-goal-aq-grounded-action-v1/PLAN.md)。AM固定权重主试验，Goal Z仅复查；先探针、通过才真实搜索，0训练/0新确认。AP文档块呈现已停止，不作为待办。AQ结论完成后链接独立报告，不修改历史分数。

Goal AQ已经完成并停止，[报告](../runs/v133-goal-aq-grounded-action-v1/REPORT.md)：新增64次B生成，64次A复用；B0/64生成核对字段，AM完整动作20→18/24、部分合法查询7→5/8，没有新增成功。不运行真实搜索阶段、训练或新确认，不发布。新协议未被执行，不能将此结果直接解释为显式关系核对机制无效；下一项实验需先证明协议可执行，不能修改AQ提示追加采样。

| 阶段 | 资产 | 结论 |
|---|---|---|
| Goal C | `docs/v133-goal-c-context-alignment-diagnostic.md` | 两轮 search/tool 历史需与训练对齐；合成上下文有适用范围 |
| Goal I | `runs/v133-goal-i-microfit/attempt-02/` | 旧4题微拟合的有效尝试 |
| Goal J | `runs/v133-goal-j-transfer/REPORT.md` | v118→Goal J，16条训练题、20次 LoRA-SFT 更新；训练 4→16/16、开发 9→10/16、回归 3→9/16 |
| Goal K | `runs/v133-goal-k-attribution/REPORT.md` | 复核数据流和 P0 归因；开发评价未进入 Trainer 或 checkpoint 选择 |
| Goal L | `runs/v133-goal-l-answer-policy/REPORT.md` | P1/P2 未给 Goal J 带来新严格收益；提示干预未可靠控制 v118 回答时机 |
| Goal M | `runs/v133-goal-m-readiness/REPORT.md` | 48条 CPU 前缀/标签验收通过；64条训练题采样未达到内容奖励差异门槛，停止 Goal J 起点 RL |
| Goal N | `runs/v133-goal-n-v118-grpo/REPORT.md` | 独立从v118出发；64条探针通过，9次有效LoRA-GRPO更新；同期主要开发集v118 9/16、Goal J 10/16、RL 11/16，回归集3/16、9/16、5/16 |
| Goal O | `runs/v133-goal-o-budget/REPORT.md` | 固定16题、3 seed×4遍；105次有效更新及384条统一评价。主要开发第1→4遍变化+1/0/−1，预定扩数据门槛未通过，停止预算扩展 |
| Goal P | `runs/v133-goal-p-system/REPORT.md` | Goal J真实检索系统完成，32题EM 13/32；协议兼容使无效动作8→3、F1 .518→.580。三臂训练完成，联合臂未达到后续复现门槛，保留Goal J为默认模型 |
| Goal Q | `runs/v133-goal-q-search-success/REPORT.md` | 从Goal J训练真实搜索轨迹：直接RL 12/32、轨迹SFT 16/32；历史Goal P RL-only在同入口19/32、回归7/16，成为当前产品演示模型。SFT后RL因混合组3<4跳过；无新模型升级 |
| Goal R | `runs/v133-goal-r-evidence-order/REPORT.md` | 从当前最佳Goal P RL-only训练两组匹配的答案SFT；原顺序/加入反转顺序两臂均为15/32、回归6/16，未超过19/32、7/16基线。固定状态顺序诊断完成，无新模型升级 |
| Goal S | `runs/v133-goal-s-label-granularity/REPORT.md` | 两处双侧证据支持的训练标签粒度修订。原标签／修订标签：seed42 15→18/32，seed43 16→20/32；seed42仍低于当前最佳19/32且回归6/16，保留Goal P RL-only产品模型 |
| Goal T | `runs/v133-goal-t-update-length/REPORT.md` | 两 seed 连续训练并预存第32/64次 checkpoint。短预算分别20/32、21/32且回归7/16、8/16，按预设门槛选择seed42第32次模型；严格EM增益主要来自表达格式 |

所有指标须标明集合身份：训练集是拟合，Goal J 与 Goal G 的16题均是已暴露开发集；本轮没有新的独立确认集。严格 EM/F1 保持冻结规则。Goal J与Goal N从共同v118起点分叉，训练预算不同。Goal Q/R/S/T采用真实搜索入口，其成绩不可与固定证据成绩混算。命令见项目 README。

## 当前顺序研究（2026-10-05）

- [AR→AV总计划](v133-sequential-research-plan.md)：依次验证，负结果舍弃，达标版本累积，最终统一保护验证。
- [AR报告](../runs/v133-goal-ar-constrained-evidence-v1/REPORT.md)：约束协议可执行但语义退步，舍弃，72次新增生成、0搜索。
- AS恢复库存核验：门槛未通过，跳过，无新增数据或计算。
- [AT计划](../runs/v133-goal-at-supported-action-learning-v1/PLAN.md)：16状态×4固定采样，50/64支持、6混合组；匹配普通SFT/GRPO执行中，结果待评价。

产品仍为Goal Z；AM及后续候选均为实验模型。历史分数、旧账本和停止决定保留。


## AR→AV最终收尾（2026-10-05）

此条替代此前“AT执行中/结果待评价”的进度文字，不修改历史分数。AR约束协议未改善语义，AS库存不足跳过，AT普通SFT/GRPO支持21→20/19，AU排序支持21→21且EM26→24，均未接受新版本。AV因无新累计候选不触发，不能声称完整回归保护通过。

- [总报告](../runs/v133-sequential-method-checks-v1/REPORT.md)
- [AT结果](../runs/v133-goal-at-supported-action-learning-v1/REPORT.md)
- [AU结果与网络修复](../runs/v133-goal-au-frozen-reranking-v3/REPORT.md)
- [AV条件收尾](../runs/v133-goal-av-cumulative-validation-v1/REPORT.md)

当前研究参考保持AM seed42-A（21/29支持），产品保持Goal Z。已完成方法不再作为待办，不增加数据、采样或微调参数来补门槛。本序列实际968次新增生成、512次搜索、256次更新、新确认0。负结果只限定已冻结配方，不代表0.8B能力上限。

- [AW→BB计划](v133-aw-bb-sequential-plan.md)：逐跳读取→独立模块→真实状态蒸馏→刷新动作RL；已授权，结果待执行。

AW已完成并关闭（完整证据A/C/B为20/1/0，172次生成）；AX因RN-94第一跳未确认停止材料阶段。AY脚本已实现但未启动，远程同步被审批模型容量故障阻断。详见[执行进度](../runs/v133-aw-bb-sequential-checks-v1/REPORT.md)。父版本保持AM，AZ/BA/BB资格待后续结果。

**AW→BB最终更新**：上述中间进度已收尾。审批恢复后，用户授权修复AY零生成初始化错误；独立v2完成32题、97生成／65搜索，2个合格候选不足8，未运行教师或训练。AZ/BA/BB条件未满足。全序列269生成／65搜索／0更新；研究仍AM，产品仍Goal Z。参见[最终报告](../runs/v133-aw-bb-sequential-checks-v1/FINAL_REPORT.md)和[AY报告](../runs/v133-goal-ay-real-state-distillation-v2/REPORT.md)。不将材料不足写成训练负结果。

- [BC报告](../runs/v133-goal-bc-context-aware-decoding-v1/REPORT.md)：AM固定权重整条动作CAD，完整证据20→16/24、部分补链7→5/8；修复1题丢失5题，2非法JSON。74主生成+36参考序列，零搜索/训练/新确认；门槛失败，停止，不准备蒸馏。A档案token32/32一致、CPU34测试与工程10次通过。研究AM及发布Goal Z不变。
