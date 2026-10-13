# Goal Z：真实连续搜索轨迹 SFT

## 结论与产品决定

**完整轨迹监督在新开发题上产生了明显的真实搜索收益，但没有通过预先约定的全部回归门槛；确认集没有打开，产品继续使用 Goal T seed 42、update 32。**

64 道新开发题中，“回答正确、信息充分且实际可见证据支持”的题数，起点为 **6**，仅监督最终答案的两个 seed 均为 **6**，监督搜索和回答完整轨迹的两个 seed 为 **34、31**。完整轨迹模型的严格 EM 为 24、23，起点为 6。两个完整轨迹模型分别在 61、58 题上进行了至少两次搜索；起点只有 4 题。这表明监督动作链改变了搜索行为。对起点的相同回答前输入回放时，这两个模型严格 EM 只有 2、4，因此新开发收益不能解释成仅凭最终答案措辞改善。

开发集选模规则要求两个 seed 在旧主要开发、旧回归和 Goal Y 已暴露题上语义成功数都不下降。完整轨迹 seed 42 在 Goal Y 48 题上从 **40 降为 39**，seed 43 为 41；因此整条路线不合格。seed 42 的四道旧题从成功变失败，其中一题回答虽仍是 `Italian`，但新搜索轨迹未展示另一侧国籍证据。复核未发现这些失分是盲审或证据引用错误。按冻结规则停止，未评价确认集、未发布任一新 adapter。

该结果支持“完整搜索动作监督比仅监督最终答案更可能改善当前多跳搜索”的开发集假设；尚不支持产品升级或跨新题的独立确认。所选新开发题现已暴露，后续不能把它当独立测试集。

## 冻结材料与执行

材料来自 [MuSiQue](https://github.com/StonyBrookNLP/musique) 和 [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) 官方有标签数据。CoRAG 仅作为轨迹组织参考；所有搜索结果均由本项目 BM25-v2 真实返回。官方压缩包 SHA256 记录在 [source-audit.json](source-audit.json)，本轮一次性候选池和逐题排除见 `candidates-*.jsonl`、`source-audit-rejections.jsonl`。题义审核发现部分原始 MuSiQue 标注中的关系与段落不符，使用原冻结候选池中下一道合格题替换，历次记录和 57 个最终排除 ID 见 `replacement-review*.json`、`train-semantic-rejections.jsonl`、`eval-semantic-rejections.jsonl`。训练、新开发、确认分别冻结为 **128、64、64** 题；三个 2048 篇自然文档库标题隔离。旧封存资料只通过既有排除键使用。

训练由 64 道 MuSiQue、32 道 2Wiki、32 道旧共同属性题构成。新来源的 96 条示范各两次真实搜索：第二条查询中的桥接实体必须在第一次实际工具返回中可见，第二侧支持文档须经第二次搜索返回；没有手工替换检索结果。32 道旧题沿用此前真实上下文。128 条 materialized 输入中，98 条含三个 assistant 动作（两次搜索、一次回答），其余 30 条含两个动作。所有提示都不超过 2048 token，动作不超过 128 token。题目、语料、审核状态和资产哈希见 [config.json](config.json)、[data-review.json](data-review.json)、[curation.json](curation.json) 和 `training-inputs.jsonl`。

起点是 Goal T seed 42、update 32；基础权重**实际权重文件** SHA256 为 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`，起点 adapter 为 `e1680c2b2a6a60ec89b4b54249e3132e2005ee6eb94bb04e1acdbd4bf76e064e`，tokenizer 为 `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。旧 32 题基线复现 **20/32 EM、回归 7/16、无效动作 0**；旧题和 Goal Y 题的实际输入 token 与历史输出逐条一致。

两臂各 seed 42、43，均独立从起点加载、一遍 128 次更新，题序相同。A 只监督最终 answer 和终止符；B 在每一步实际呈现的输入上分别监督搜索动作、answer 和终止符，按整条轨迹所有 assistant token 归一。learning rate `5e-6`，AdamW `(0.9, 0.999)`、`eps=1e-8`、`weight_decay=0`，梯度裁剪 1，仅更新 LoRA。四个阶段末 checkpoint 的基础权重不变、LoRA 变化，独立重载 logprob 最大绝对差均为 0。四次训练耗时及其他阶段资源见 [state.json](state.json)；累计 GPU 进程时间约 **1978 秒**，没有时间预算截断。CUDA 提示 CuBLAS 在当前环境下不保证严格逐位确定性，两个 seed 是独立重复而非逐位复制。

## 完整开发集结果

语义列来自冻结的 [evidence-rubric.json](evidence-rubric.json) 及匿名、去重的 145 条新增审核；相同“问题—实际证据—答案”组合只审一次。720 条评价记录中另有 144 个唯一组合沿用已冻结审核。代理辅助审核及主 agent 争议核对均保留在 `blind-review-*` 文件。语义成功要求可引用实际可见证据，不以严格 EM 或官方答案文本直接代替证据判断。

| 模型 | 新开发语义 /64 | MuSiQue /32 | 2Wiki /32 | 新开发 EM /64 | 事实错误 | 证据不支持 | ≥2 次搜索 /64 | 双支持召回 /64 | 旧主要语义 /16 | 旧回归语义 /16 | 旧 EM /32 | Goal Y 语义 /48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 起点 | 6 | 3 | 3 | 6 | 34 | 17 | 4 | 9 | 14 | 9 | 20 | 40 |
| 答案 SFT 42 | 6 | 3 | 3 | 5 | 34 | 17 | 2 | 8 | 14 | 9 | 22 | 38 |
| 答案 SFT 43 | 6 | 3 | 3 | 5 | 35 | 16 | 2 | 8 | 14 | 9 | 22 | 39 |
| 完整轨迹 SFT 42 | **34** | 12 | 22 | 24 | 16 | 5 | 61 | 47 | 15 | 9 | 21 | **39** |
| 完整轨迹 SFT 43 | **31** | 9 | 22 | 23 | 18 | 10 | 58 | 47 | 14 | 9 | 20 | 41 |

所有模型新开发无效动作为 0。平均搜索次数从起点 1.06 升到完整轨迹模型 2.03、1.92；平均每题延迟从 1.37 秒升到 2.56、2.47 秒。严格 EM 与语义成功的差距包含答案粒度及表达差异，历史评分未改变。完整数据及评分复算见 [development-decision.json](development-decision.json)、[analysis.json](analysis.json) 和 [development-per-case.jsonl](development-per-case.jsonl)。

相对起点，完整轨迹 seed 42 新增 28 道语义成功，seed 43 新增 25 道，均未丢失起点的 6 道新开发成功；两 seed 共同新增 25 道。两臂答案 SFT 均无净新增。完整轨迹成功题中，两个 seed 各有 **13 道**的第二侧支持文档在首次搜索中尚未出现、经后续搜索才呈现。这是更直接的连续搜索证据；其余成功题不能仅凭“两次搜索”认定第二次搜索是必要的，因为有些题第一次已检出双侧文档。

按冻结支持标题做的失败定位，起点 23 题首次未检出第一支持、33 题已检出第一支持却停止继续搜索；完整轨迹 seed 42 分别降至 9 和 0，seed 43 为 12 和 1。完整轨迹仍有两侧证据已呈现而回答失败的题（seed 42 为 6，seed 43 为 9），以及无最终回答的题（7、2）；搜索行为仍需改善。一个真实链例子是问 Johann Nicolaus Bach 父亲的出生地，模型先查父亲，观察到 Johann Christoph Bach，再查询其出生地并回答 Arnstadt。也出现了查询桥接实体选错、却偶然检出正确支持段落的情况，故不能把所有收益解释为严格按正确链规划。

固定上下文诊断对起点实际回答前输入逐 token 回放，64 题严格 EM：起点 6，答案 SFT seed 42/43 各 6，完整轨迹 seed 42/43 为 **2、4**。该诊断与产品分数分开保存于 `fixed-dev-*.jsonl`，不参与选模。训练 loss 曲线见 [training-curves.png](training-curves.png)，原始每更新 loss、梯度、显存与监督 token 数见 `seed*/updates.jsonl`；loss 下降本身不作为成功标准。

## 门槛、限制与下一步

答案 SFT 两 seed 在新开发均未达到起点 +4，且 Goal Y 语义下降。完整轨迹两 seed 均超过新开发 +4，事实错误和证据不支持数下降，旧主要与旧回归语义没有下降；但 seed 42 的 Goal Y 语义 **39 < 40**。失去的四道 Goal Y 成功题分别出现错误媒体类型、单侧 genre 扩大、把 raga 答成 sign language，以及继续搜索后丢失 Matteo Renzi 国籍证据。三道新增成功不足以抵消四道丢失。详见匿名审核及 `eval-y-*` 原始轨迹。此回归不是单纯严格 EM 波动。

MuSiQue 新开发中有一题的原题要求 ATS-6 的“前身”关系，但可见段落只证明与 ISRO 的合作，无法证明该关系。它保留在冻结 64 题分母中，并按“证据不支持”计；没有在模型输出后换题。该源标注问题和代理辅助语义审核均限制结论强度。

**唯一产品推荐：继续使用 Goal T seed 42、update 32。** Goal Z 的完整轨迹 checkpoint 是有价值的研究候选，但现有预设发布规则未通过，确认集保持未访问。最值得单独检验的后续问题是：怎样保留连续搜索的收益，同时避免在已有双侧证据充分时无谓追加搜索、导致旧题证据丢失或回答漂移。需要在新的冻结方案中解决该问题，再走全套回归与确认；不能用本轮未打开的确认集来改选当前路线。

## 复现与文件

远程工作目录 `/root/autodl-tmp/mini-agent-rl`，官方源压缩包保存在 `/root/autodl-tmp/goal-z-sources/`（MuSiQue 272049578 字节、2Wiki 258968175 字节），压缩包哈希见 `source-audit.json`。数据源下载、一次性审查与修订的完整脚本分别为 `scripts/download_v133_goal_z_sources.py`、`scripts/prepare_v133_goal_z_multihop_data.py`、`scripts/curate_v133_goal_z_multihop.py`、`scripts/review_v133_goal_z_data.py`、`scripts/finalize_v133_goal_z_data.py`。运行器 `scripts/run_v133_goal_z_multihop_sft.py` 提供 `prepare`、`run`、`summarize`；已开始的目录拒绝覆盖。冻结输入可直接做 CPU 复算：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_z_multihop_sft.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/analyze_v133_goal_z.py  # 在新的输出目录执行；当前结果已生成
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_z_selected.py --question '你的问题'
```

独立新运行须在新的工作副本保留冻结源文件、已批准审核记录及原资产，然后依次执行 `prepare`、`run`。当前状态为 `complete_development_gate_failed`，再次调用 `run` 不会跳过门槛打开确认集。`summarize` 不加载模型。新增测试：`PYTHONPATH=src:scripts .venv/bin/python -m pytest tests/test_v133_goal_z_multihop_sft.py -q`。


