# Goal AE：离线语料恢复与检索对照

**更正状态（2026-09-30）：** 416 条评价实际使用 Goal T seed 42/update 32，而非已发布 Goal Z 完整轨迹 seed 42。本轮没有完成当前产品模型的检索对照。原报告保存在 `REPORT-before-model-identity-correction.md`，原始输出、配置和决策保持不变；机器可读更正见 `model-identity-correction.json`。不进入新确认集，不训练，不发布检索器变更。

远程文件 SHA-256 核查确认：评价配置中的 adapter 哈希 `e1680c2b2a6a60ec89b4b54249e3132e2005ee6eb94bb04e1acdbd4bf76e064e` 与 Goal T checkpoint 完全相同；已发布 Goal Z adapter 的实际哈希为 `d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。原因是 AE 运行器使用 `SOURCE_ADAPTER = z.SOURCE`，而 `z.SOURCE` 是 Goal Z 训练前的 Goal T 起点。配置中的模型名称写错，原报告沿用了该名称。这是实验接线和报告错误，不能将本轮下降解释为当前产品模型退步。

## 是否需要配置搜索 API

不需要。已通过普通文件下载取得公开的 Wikimedia 2023-11 英文静态 Parquet 快照，并按此前冻结的标题清单在本地筛选、切分段落。数据集固定在提交 `e6057dc557255a03c9c3c47ceab0eb44353b1bc5`，记录了来源标题、URL、分片和 SHA-256。下载和运行产品检索均没有调用搜索引擎 API，也没有使用 API key。数据说明见 [Wikimedia Wikipedia 数据集卡](https://huggingface.co/datasets/wikimedia/wikipedia)。

静态快照覆盖 Goal Z 训练 1792/2048、Goal Z 开发 1802/2048、Goal Z/AA 回归 1798/2048、Goal Y 回归 1730/2023、旧回归 87/96 个标题。未命中的标题继续使用各自冻结语料中的原始段落，因此每套文档仍有完整标题覆盖。输出共约 184,336 个自然段落；完整段落索引及逐篇哈希见 `static-data/manifest.json`。

独立覆盖审计进一步确认五套索引的所有 8,263 个冻结标题均有至少一个非空自然段落，且没有缺失标题；审计记录在 `source-and-baseline-audit.json`。Goal Z 训练、开发和 AA 回归的静态页缺口分别为 256、246、250 篇；其中 256/256 和 246/246 个训练/开发缺页段落可在对应官方候选源中逐字匹配，AA 回归有 229/250 个可逐字匹配，其余 21 个使用原冻结自然段落。

准备阶段记录了 AutoDL RTX 4090D、实际基础权重、tokenizer、四个评价集和语料哈希，但错误地将 Goal T 起点 adapter 标记为已发布 Goal Z。新确认集未被读取。配对评价使用同一个 Goal T 模型、提示、问题和生成预算，分别运行原 BM25-v2 与 BM25-v3 段落检索，各 208 条，共 416 条轨迹。

BM25-v2 在四组题上的 208 个答案和 case ID 均与历史 incumbent（Goal T）逐题一致；这验证了旧起点的复现，不能验证已发布 Goal Z 的基线。

## 结果

| 数据组 | 题数 | BM25-v2 EM | 段落检索 EM | BM25-v2 支持标题检出 | 段落检索支持标题检出 |
|---|---:|---:|---:|---:|---:|
| Goal Z 新开发 | 64 | 6 | 3 | 9 | 25 |
| 旧回归 | 32 | 20 | 5 | 32 | 31 |
| Goal Y 回归 | 48 | 34 | 20 | 47 | 46 |
| AA 已暴露回归 | 64 | 9 | 5 | 9 | 24 |
| **合计** | **208** | **69** | **33** | **97** | **126** |

段落方案的平均 F1 从 0.425 降到 0.229。两方案均为 0 次无效动作、平均每题 1.04 次搜索。Goal Z 开发严格 EM 从 6 降到 3；旧回归从 20 降到 5。段落索引检出了更多支持标题，但这只是检索代理指标，不代表回答时的片段包含充分证据。

逐题检查发现了直接失败机制：对“Rómulo Gallegos 与 Tobias Wolff 的共同职业”，段落方案返回 Tobias Wolff 的 `Bibliography` 段落，模型回答了仅能由另一侧支持的 `politician`；原方案返回 Tobias 的简介并答对 `novelist`。对 Paul Young/David Draiman，段落方案返回了 David 的非职业段落和 Paul 的唱片引用，丢失了原方案中的 `songwriter` 证据。说明按段落独立排序会把正确页面的无关小节排到前面，支持标题命中增加却损害了实际证据组合。

本轮没有对新输出做完整语义盲审，因此不把 EM/F1 宣称为语义成功率。下降仅描述 Goal T 在此次语料与检索组合变化下的表现；既未评价当前产品模型，也未完成原定语义门槛，不能据此判定 Goal Z 的开发门槛失败或通过。新确认集保持关闭。

## Goal AD 路由更正

Goal AD 的旧失败诊断按 `source` 字段选择语料，正确规则应按 case ID 是否属于冻结的 `selected-train.jsonl` 选择。48 条修正路径中，17 条路由不符；6 条受影响路径属于原来诊断的失败案例。六条原始查询已在错误路由和正确的 Goal Z 训练语料上重放，并按正确可见证据重新裁定。受影响案例中有 1 道能用 `Glenn Miller nationality` 取得答案证据；其余 5 道属于答案关系或来源标签不支持。加上未受路由影响的 28 道失败，修正后可修复估计为 1/34，仍未达到 AD 的 8 题训练门槛。逐题记录见 `ad-corrected-failure-adjudication.jsonl`，摘要见 `ad-corrected-failure-adjudication-summary.json`。原 Goal AD 报告、评分和停止决策保持不变；修正账本及路由重放分别见 `ad-corpus-route-correction.jsonl`、`ad-corpus-route-correction-summary.json`、`ad-failure-route-replay-corrected.jsonl`。

后续 Goal AD 路由函数已改为按冻结 case ID 成员关系选语料，并增加回归测试。这一更正不重写历史评分，也不证明查询训练会改善产品表现。

## 决策

依据此前 AA 发布结果，当前唯一推荐仍为 Goal Z 完整轨迹 seed 42 adapter + 原 BM25-v2 产品语料。本轮没有提供替换当前产品的证据，也没有证明该产品在段落方案下下降；暂不切换、不训练，Goal AB 新确认集保持关闭。

本次无 API 的数据获取路径已经成立。Goal T 对照揭示了段落方案存在返回正确标题但无关正文的问题；对当前产品是否有效仍待验证。下一步先修正模型选择并冻结实际 Goal Z 权重哈希，再通过 CPU 查询回放检查标题加权与片段选择，预选一个修复方案后进行当前产品的受控配对评价。语料、切分和排序同时变化，现有结果不能把全部下降归因于单一因素，也不支持自动追加训练。
