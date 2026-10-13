# Goal AO：同一证据、不同关系的动作 SFT 对照

## 目的和假设

检验在证据正文及既有搜索历史不变时，把问题改为询问不同实体关系，是否能通过动作 SFT 改善未参与训练的真实搜索。AO 是行为学习实验，不是新数据收集、来源全面审核或模型发布。

历史约束：Goal Z 完整轨迹 seed42 是唯一发布起点；AM seed42 普通状态 SFT取得局部收益但未过探针门槛；AN 集中难例 seed42 未达到继续门槛。AM、AN 的结果、文件和停止决定原样保留。AM 普通 SFT 只作为历史参考；AO 的 A 必须重新训练。

## 固定材料

从 AM 的 24 个 answer 状态中按 `case_id + state_id` 固定以下 8 组，每组保留原问题与原答案，并添加一个答案由相同可见正文支持的不同关系问题：

| case_id | 原问题答案 | 新问题 | 新答案 |
|---|---|---|---|
| `musique:2hop__407314_80178` | Jaden Smith | Who gave the Believe Tour? | Justin Bieber |
| `2wiki:b2addc160bdc11eba7f7acde48001122` | University of Helsinki | Who was the son of Ossian Donner identified as a British Conservative MP? | Patrick Donner |
| `2wiki:fa9654640bdb11eba7f7acde48001122` | September 3, 1922 | When did the director of film Return Of The Seven die? | February 15, 2001 |
| `2wiki:a874fa560bda11eba7f7acde48001122` | Christiansburg, Virginia | When was the director of film Terror Mountain born? | June 28, 1898 |
| `musique:2hop__829831_135304` | Margaret Davies | Who was the father of the performer of Animal Justice? | Will Cale |
| `2wiki:67adecd80bdc11eba7f7acde48001122` | 6 October 877 | What is the date of death of Richilde Of Provence herself? | 2 June 910 |
| `musique:2hop__418766_73460` | February 4, 1948 | On what date did the country containing Kavisigamuwa become a republic? | May 22, 1972 |
| `musique:2hop__135820_578368` | Clark County | What is the ZIP code of Carpenter in the state that SoDak Con was named after? | 57322 |

总计 8 组、16 个问答状态、8 个派生题面。派生状态只改初始用户问题和目标动作；历史搜索、工具正文、文档顺序、干扰文本保持不变。用真实 tokenizer 和生产模板重编码。原状态须与 AM 保存 token 完全一致；衍生状态须只有用户问题与目标动作不同。要求实际可见正文逐项支持目标、历史动作合法、无答案泄漏/矛盾动作，并排除所有评价 ID 与规范化题面。派生充分证据状态不代表自主搜索轨迹，也不监督派生搜索动作。任何组不合格即停止，不换组、不补充候选。

## 训练对照

两臂各自从 AA 发布的 Goal Z seed42 adapter 加载；预期 SHA-256：`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。两臂同用 128 条完整轨迹、AM 34 个状态、每步顺序和超参数。

| 条件 | A 等量重复 | B 关系对照 |
|---|---|---|
| 每关系组监督 | 原问题、原问题 | 原问题、新问题 |
| 轨迹、AM状态、监督实例数 | 相同 | 相同 |
| token/状态权重 | 等权 | 等权 |

每臂 128 更新；更新 2、4、…、68 使用 34 个 AM 状态；更新 72–103 依表顺序循环四次，每更新一组，两问当前动作损失取平均。A 的重复问答也按两次实例计入。损失为 `L轨迹 + 0.25 × L_AM状态 + 0.25 × L关系组`，无对应项时为零。轨迹损失按该轨迹所有助手监督 token 数归一；状态损失仅覆盖当前 JSON 动作与终止符并按目标 token 数归一；prompt、工具正文和历史动作不计当前状态损失。

优化器：LR `5e-6`；AdamW betas `(0.9, 0.999)`、eps `1e-8`、weight_decay `0`；梯度裁剪 `1`；仅更新 LoRA；单卡 BF16、不量化。无字段/难例加权、RL、DPO、参数扫描或附加训练遍数。

## 产品评价固定项与额度

Goal Q；direct-v090；first-json-v1；thinking=false；legacy；BM25-v2/top-3；greedy；输入 16384 / 输出 2048；每题最多4次模型调用、3次搜索。评价用冻结开发64题各自语料。A/B各做64题真实搜索、24完整证据探针、8个 AN 冻结缺链探针及16个训练状态检查；AM 普通 SFT 先对16个问题状态各生成一次拟合参考。探针只生成一条动作，不执行后续搜索。AM 普通模型的历史搜索/完整探针输出复用 AM 原档案；AO 仅增加其16个状态检查。

最多 4 次训练、512更新、256条真实搜索轨迹、1232次GPU生成、768次真实搜索；第一阶段最多 624次生成、384次搜索、256更新；新确认调用为0。seed42门槛不通过时立即停止且不执行seed43。GPU、模型身份/哈希、数值或基础设施错误保存现场并停止；不自动重跑、降预算或改参数。已有用户授权允许直接执行 GPU，不重复确认。

## seed43冻结门槛

只有 seed42 B 同时满足以下条件才启动 seed43 A/B；seed43规则不变：

1. B 支持成功至少23/29，且比本轮A净增至少2题。
2. B完整证据探针至少22/24，且比A多至少2题；至少一个新增正确例修复实体、角色或关系对象绑定。
3. 事实错误与证据不支持合计不高于A且不超过7；动作问题不高于A且不超过1，各动作类别计数不增加。
4. 保留AM赢回的 Rapana、公爵两题；旧7题连续搜索保护集最多丢失1题。
5. 部分证据探针提前回答、无效动作、重复查询不增加；合法补链搜索不少于A且至少7/8。

seed43 两臂都独立从 Goal Z seed42 重新加载。两个seed均过门槛才建议独立回归验证，不选择单seed、不发布候选。训练拟合提升但真实搜索与冻结探针无收益，结论为局部拟合。

## 固定顺序和交付

材料与实现一次冻结 → CPU预检 → AM 16状态基线 → seed42 A训练/重载 → seed42 B训练/重载 → 两臂64题及固定探针 → 一次评分收尾 → 按门槛决定是否执行seed43 → 生成结果报告与完整哈希/额度校验。

逐题报告支持成功新增/丢失、首次动作分歧、必要关系首次返回、回答实际引用位置；将收益拆成搜索取证、关系对象绑定、限定保留、停止重复搜索或仅措辞变化。16训练状态双问题正确数只算拟合。留存每次实际输入/生成 token、原文、JSON解析动作、停止原因、延迟、loss/mask/更新日志、adapter哈希及独立重载检查。

未来实验次序固定：AO关系对照训练 → 有界真实失败状态蒸馏 → 独立证据呈现或工具辅助实验。MCP核验/读取需单列调用成本；RL只在正确动作可达且奖励依据可靠时再考虑。不可把已完成的提示/窗口建议重新列为待办；每轮只改一个因素，避免循环扩数据证明数据。

## 执行记录

准备/预检/训练/评价结束后，在此追加冻结 manifest、实际材料准入、阶段计数、seed42门槛、是否执行seed43与结果报告链接。此文件的固定方案不得按观察到的模型输出追改。

## 执行记录（2026-10-05）

固定8组一次性CPU核验已完成。原始输入精确重建8/8，目标新答案出现在实际工具正文8/8，但无历史assistant动作答案泄漏仅7/8。Believe Tour组历史搜索查询已经包含新关系题答案 `Justin Bieber`，违反材料准入条件。依据本计划“任一组语义不合格即停止，不换题、不扩池”，本轮停止于材料准入，未生成正式冻结manifest或训练材料，seed42/43均未启动。GPU生成/训练更新/生产搜索均为0。完整依据见[停止记录](PREPARATION_STOP.md)和[evidence audit](evidence-audit.json)。
