# 固定Goal Z：必要关系覆盖与搜索决策诊断

日期：2026-10-03。全部实验计算在远程AutoDL完成。本轮冻结24题、53旧替代分支及三份各2048篇语料；未使用新确认集，未训练、未更新adapter。

## 结论

唯一后续路线：**优先核验证据覆盖与题目依据**。22/24题仍有必要关系未确认或关键歧义，超过预定12题门槛，不能用本轮结果声称主要瓶颈已经是模型关系理解。未确认不等于整份语料绝对没有关系；审核完整读取已有定位及针对性的相关正文，其余候选仅核查标题和主体匹配句，未做2048篇全量否定证明。

| 指标 | 固定分母或有效分母 | 结果 |
|---|---|---|
| 必要关系直接支持 | 48条边 | 28 |
| 两边均有源文支持 | 24题 | 4（其中2题队伍指向歧义） |
| 覆盖未确认或关键歧义 | 24题 | 22 |
| 合法查询新增至少一条必要边 | 24题 | 5 |
| 无关键歧义、预算内完整链返回 | 24题 | 2 |
| 无关键歧义、预算后完整链保留 | 24题 | 1 |
| GPU条件决策充分且正确 | 1个合格状态 | 1 |
| 新查询 | 上限48 | 36，另12分支复用缓存 |
| GPU调用 | 上限24 | 1 |
| 权重更新、后续GPU输出search执行 | 0 | 0 |

有依据的机制案例：查询生成1题、上下文保留1题、检索0题、回答决策0题。0表示没有满足本轮因果判据的独立案例，不表示机制绝对不存在。原查询另有明显角色/关系/实体错误，保存在original-query-review，但不凭查询文本单独宣称可恢复。

## 模型、生产条件和身份

基模`/root/autodl-tmp/qwen_08b`，实际加载`runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final`；AA发布身份核验沿用并验证AI/AH原guard，adapter实际权重SHA-256为`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。GPU前后相同。RTX4090D单卡BF16、不量化、greedy。Goal Q/direct-v090/thinking=false/legacy/BM25-v2/top3，4调用/3搜索，2048输入/128生成token均未改变。

## 冻结与分支隔离

先冻结query-freeze，再生成覆盖候选。每题固定锚点、A/B两查询，主体必须在题目或原实际正文中有正确绑定；不按新结果移动锚点。48冻结分支中36新查询、12缓存。原52次查询和53旧替代查询取得相同BM25-v2完整排名；137个缓存键的前三项均与生产检索一致，深层文档未进入模型输入。

每个分支只使用自己的原前缀和结果，预留最终answer调用。旧53分支重新核验后52合法，1条Juan第五公爵查询在早期前缀中嵌入尚不可见的第四公爵，拒绝计修复。旧分支未进入GPU名单；GPU仅从冻结A/B中按A优先选唯一合格状态。

## 关键原文和上下文结果

- Juan第五公爵原文明确父亲是第四公爵；第四公爵生命日期支持死亡年1575。A/B返回完整链，但legacy压缩从2155 token降为936 token，父亲关系句消失。实际保留标题和首句不能补回父子边。详见branch-audit的context_usage.document_changes。
- Philipp母亲明确是Margarete Sophie，母亲正文给出13 May 1870。A分支实际输入1187 token、完整链可见，GPU回答同一日期。原查询把父亲Albrecht当母亲搜索，是可修复查询/角色错误；此次教师补证后正确回答不等于自主搜索成功。
- Chelsea额外工程哨兵复现245 token、正文全部丢失，原经理关系存在。它不进入24题分母，不追加GPU调用，不计为第二个诊断案例。
- Witold在Łódź毕业不证明在那里出生；Arthur Rupert Dickey出生Amherst不证明Charles Tupper出生Amherst；区域12校区和7校区运营联盟不证明整个CSU数量；Tegh Bahadur与Gobind Singh相遇不证明父子。
- Hagar是Sarah的女仆、Ishmael母亲，不自动证明婚姻；Tony Parker历史效力、拥有ASVEL不固定当前工作地；Rosaline任职2003–2011不能无时间限定回答当前首席法官；Julie Dawn Cole的1971选角不能直接用于2005版。
- 关系边界同时检查编剧/导演教育、发行/签约、来自/出生、区域/系统数量、相遇/父子。前两项是审核拒绝准则，本轮24题中无对应实测案例，不宣称新测出了这些能力。等价表达例如son/father、生命日期/死亡年和written/composed可以通过；缺证不自动判事实错误。

## 逐题最早阻断

| case_id | 锚点调用（0起） | 最早阻断 | GPU结果 |
|---|---:|---|---|
| musique:2hop__862994_69048 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__44522_58556 | 0 | coverage_or_question_basis | 未调用 |
| musique:2hop__442175_56873 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__337223_56873 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__468778_88165 | 0 | coverage_or_question_basis | 未调用 |
| 2wiki:f299dd600bdd11eba7f7acde48001122 | 1 | context_retention | 未调用 |
| 2wiki:d94d1d720bdd11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__71034_343058 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__864101_135373 | 0 | coverage_or_question_basis | 未调用 |
| 2wiki:015a72a60bde11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__475070_532404 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:66616e2a0bde11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:b592bb5c0bd911eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:f480bf400bdd11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:ae714dea0bdc11eba7f7acde48001122 | 1 | conditional_repaired | sufficient_correct_answer |
| 2wiki:085ace840bde11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:d438a1020bd911eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__65576_80026 | 0 | coverage_or_question_basis | 未调用 |
| 2wiki:66c00bf60bde11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:43a949ae0bdc11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__495316_660579 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__455392_40315 | 1 | coverage_or_question_basis | 未调用 |
| musique:2hop__124932_278133 | 1 | coverage_or_question_basis | 未调用 |
| 2wiki:13e2f86c0bde11eba7f7acde48001122 | 1 | coverage_or_question_basis | 未调用 |

## 可核验交付

- freeze.json、query-freeze.jsonl：模型、语料、生产代码、题目、原轨迹与查询哈希；原76次输入token精确重放。
- coverage-review-packets / coverage-review / coverage-reviewed：每边候选上限12，主体、方向、对象、正文引用、源字符位置及人工裁定；未确认保留理由。
- support-ranks：原52查询、旧53替代查询和新48分支支持文档的完整排名/分数；返回相关人物不等于证明关系。
- branches / branch-audit / exact-input-quotes：实际输入IDs、工具JSON、预算变化、引用精确token/字符位置、独立分支充分性。
- historical-branch-audit：53旧分支重新按逐边原文审核，不继承按题号写死的充分性。
- original-call-edge-audit：76原调用实际输入、原输出、解析动作、工具JSON及逐边可见引用位置。
- gpu-list / gpu-next-actions / gpu-action-review：冻结名单、真实输入、原始输出/解析协议、耗时及源文判断。唯一调用生成耗时约4.96秒；不包含加载时间。
- verification、delivery-manifest-v2：身份、前三排名一致、原输入token、位置、隔离、额度和文件哈希。

旧审核及分数没有覆盖。本轮另存裁定，source_chain支持与题目无歧义分开。不存在完整产品评测，不能报告产品EM/F1、事实错误率、搜索成功率或收益变化；条件检查有效分母仅1，源文答案正确1/1。原产品分数仍继承各自审核版本。

## 下一步有限方案

本轮到此停止。下一步只准备证据覆盖与题目依据审计：沿当前24题已有候选，先明确完整正文审核范围、桥接别名与时间/版本限定，再冻结有限审计数量与停止门槛；不提前扩池、换题或训练。需进一步查看语料的范围和数量另行冻结。保留Juan/Chelsea作为后续独立上下文实验工程依据。

动作恢复SFT仍须另行满足8/24、16对合格材料、8个真实错误前缀及三类各4独立案例。本轮实际保留完整链仅1/24，不满足恢复门槛。GPU条件检查不替代训练材料资格，不允许评价题进训练。以后若训练达标，先展示真实数据、损失/混合比例/更新数、两seed对照、评价数量及发布条件，再按用户要求确认具体GPU训练计划。

## 实现修复与局限

修复了manifest对外部基模路径使用relative_to的序列化错误，保留旧脚本和failure.manifest-path-v1；只恢复manifest，不重复查询或GPU。补充原调用引用账本发现旧schema缺派生正文位置，改为解码已有token IDs，记录supplemental-implementation-note，不重做实验。四项CPU回归测试通过。训练、GPU、数值、基础设施及模型身份没有失败；GPU库使用正确的参考实现，既有deterministic warn_only警告已保留原终端记录，未改参或重跑。
