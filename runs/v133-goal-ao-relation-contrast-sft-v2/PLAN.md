# Goal AO v2：同一证据、不同关系的动作SFT匹配对照

## 目的与历史结论

检验在同一证据正文下改变问题所问关系，是否能训练模型选择正确答案对象，并迁移到未参与训练的开发搜索。AO v1按预先冻结的无泄漏条件在材料阶段停止：8组中Believe Tour组的旧历史查询已经包含反事实答案。v1账本和停止决定保留；本v2仅修复这一已定位的材料可识别性问题，不改题、不扩池、不按模型结果挑材料。

唯一发布起点仍为AA确认的Goal Z完整轨迹seed42。AM状态SFT开发层有局部收益但未过继续门槛；AN难例加权未取得额外收益。AM、AN结果原样保留。

## 唯一材料修复

8个固定问题组、原题/答案、新题/答案及顺序沿用[v1计划](../v133-goal-ao-relation-contrast-sft-v1/PLAN.md)。继续复用AM已有answer状态，不产生新题或搜索数据。

Believe Tour组改用其AM状态中**第二次搜索之前、第一轮真实搜索返回之后**的真实历史前缀。第一轮查询是原始问题；真实top-3正文已经包含：

- `Believe Tour`文档：该巡演是Justin Bieber的演唱会巡演；
- `Never Say Never`文档：歌曲由Justin Bieber演唱，并features rap interludes from Jaden Smith。

这两份原文和原始干扰文档保持不变，首轮工具返回直接取自已存AM输入。选择该前缀是为排除后续历史动作 `search("Justin Bieber")` 的答案泄漏；不删改或伪造工具历史，不从工具结果拼接文档。两种目标都能由此真实前缀支持。其他7组仍使用冻结AM回答状态。训练状态明确标记为“派生的充分证据状态”，不声称为自主搜索轨迹、不监督衍生search动作。

所有组必须通过一次性CPU准入：原状态真实tokenizer/template重建精确；首个用户问题唯一；A/B成对输入除初始问题外逐消息相同；两答案有同一实际可见正文支撑；没有历史assistant动作提前泄露各自目标；引用、字符/ token位置可追溯；历史动作和工具正文合法；排除所有评价ID与规范化题面。任何组不合格即停止，不换组、不减为7组、不继续补候选。

## 固定8组

| case_id | 原答案 | 对照问题 | 对照答案 |
|---|---|---|---|
| `musique:2hop__407314_80178` | Jaden Smith | Who gave the Believe Tour? | Justin Bieber |
| `2wiki:b2addc160bdc11eba7f7acde48001122` | University of Helsinki | Who was the son of Ossian Donner identified as a British Conservative MP? | Patrick Donner |
| `2wiki:fa9654640bdb11eba7f7acde48001122` | September 3, 1922 | When did the director of film Return Of The Seven die? | February 15, 2001 |
| `2wiki:a874fa560bda11eba7f7acde48001122` | Christiansburg, Virginia | When was the director of film Terror Mountain born? | June 28, 1898 |
| `musique:2hop__829831_135304` | Margaret Davies | Who was the father of the performer of Animal Justice? | Will Cale |
| `2wiki:67adecd80bdc11eba7f7acde48001122` | 6 October 877 | What is the date of death of Richilde Of Provence herself? | 2 June 910 |
| `musique:2hop__418766_73460` | February 4, 1948 | On what date did the country containing Kavisigamuwa become a republic? | May 22, 1972 |
| `musique:2hop__135820_578368` | Clark County | What is the ZIP code of Carpenter in the state that SoDak Con was named after? | 57322 |

## 训练两臂

两臂独立从AA发布Goal Z seed42 adapter载入（预期SHA-256 `d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`），各保留Z原128完整轨迹及AM原34状态。

|  | A：等量重复监督 | B：关系对照监督 |
|---|---|---|
| 原128轨迹、AM34状态、更新顺序 | 相同 | 相同 |
| 8组每次出现的两个答案状态 | 原题、原题 | 原题、对照题 |
| 权重 | token与状态均等权 | 相同 |

每臂128更新。更新2、4、…、68加入AM的34状态；更新72–103按8组顺序循环4次，每组两个动作损失取均值。A也计算两次相同原问题实例，匹配B的监督实例数。目标：`L = L轨迹 + 0.25×L_AM状态 + 0.25×L关系组`。轨迹损失按该轨迹助手监督token总数归一；状态损失仅为当前JSON动作及终止符token均值NLL；不训练prompt、正文或历史动作。

LR `5e-6`；AdamW betas `(0.9,0.999)`、eps `1e-8`、weight_decay `0`；梯度裁剪1；仅LoRA；单卡BF16、不量化。无字段/难例权重、RL、DPO、参数扫描或附加训练遍数。

## 评价、预算与继续门槛

所有评价固定Goal Q、direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top-3、greedy、输入16384/输出2048、每题最多4调用/3搜索，各集合使用冻结语料。

执行次序：材料CPU准入与工程测试 → AM普通SFT对16状态各生成一次 → seed42 A、B各训练128次并独立重载 → 各运行开发64题、24完整证据探针、8 AN冻结缺链探针、16拟合状态检查。探针不继续搜索。结果一次收尾；seed42全门槛通过才运行seed43两臂，二者仍各从Goal Z seed42独立载入。

冻结门槛：B支持成功至少23/29且比A至少多2；完整探针至少22/24且比A至少多2，至少1例修复实体/角色/关系对象错绑；错误合计不高于A且≤7，动作问题题数不高于A且≤1，各项动作计数不增；保留AM赢回Rapana和公爵两题；旧7题连续搜索保护最多丢1；部分证据提前答、无效、重复不增，合法补链搜索不低于A且至少7/8。未确认结果不能计入通过。只有两个seed都满足冻结规则才建议做独立回归；不发布或选择单seed。

总上限：4次训练、512更新、256条真实搜索轨迹、1232次GPU生成、768次搜索；seed42阶段最多2训练、256更新、128条轨迹、624次生成、384次搜索；新确认0。GPU/数值/身份/哈希/基础设施失败保存现场并停止，不自动重跑或改参。用户已授权GPU执行，无需再次确认。

## 执行记录

材料泄漏修正后，八组原定关系问题与答案全部保留；Believe Tour组使用其AM真实首轮搜索返回后的原始前缀。v1停止档案未覆盖。Manifest SHA-256为`e28ce625fdd04497413f7d1035c0f7301e28a23ab8790f4b456b1f524de0aaf7`，Goal Z起点adapter SHA-256为`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。

seed42 A、B均完成128更新及独立重载。固定64题均完成真实搜索；A支持成功19/29、B20/29，B比A多1题；完整证据探针两臂均20/24，绑定修复0；错误合计A9、B8，B超过≤7门槛。动作问题A2题、B1题；部分证据合法补链查询均7/8，无提前回答。冻结继续门槛未通过，**seed43不执行**。当前发布模型仍为Goal Z。

本阶段实际消耗：2次训练/256更新、128条真实轨迹、495次GPU生成、255次搜索；seed43与新确认调用为0。候选A SHA-256 `621bf9e3b03441fb4ffe89f91b26092b33c1e31b6d8697eb926946d745d35797`；候选B SHA-256 `ea9179c803d2b7a74dcf51e60238cc2ade0d3a080238bc7bda683093310bcb57`。完整指标、逐题胜负、引用、探针、额度和门槛见[报告](REPORT.md)与`seed42-gate.json`。以上候选不发布，评价材料禁止训练使用。
