# 本地混合检索

现有2048篇英文开发文档提供BM25、MiniLM向量和等权RRF混合检索。无需外部API、数据库服务或API Key，不训练模型。

模型、索引、检索及问答运行记录位于 `E:\AgentLightningAssets`。源码位于当前项目。共享配置为 `configs/retrieval-local.json`。使用已缓存的MiniLM模型，CPU编码；现有Agent沿用CUDA/BF16。索引包含2181个分块，文档返回保留原始全文。

## 使用

在PowerShell执行：

```powershell
& 'D:\AgentLightning\mini-agent-rl\scripts\start_demo.ps1' -NoBrowser
```

浏览器打开 http://127.0.0.1:8765 。检索方式默认“混合检索”；“仅检索文档”无需Agent模型就绪。“运行搜索”使用当前选中模型并执行真实工具调用。想只运行检索服务，可在启动时加 `-ReplayOnly`；之后切换模型选项可以加载Agent。自定义上传语料在首版使用BM25。

命令行使用绝对路径（该Conda环境的Scripts目录未加入系统PATH）：

```powershell
& 'D:\anaconda3\envs\mini-agent-rl\Scripts\mini-agent-search.exe' index
& 'D:\anaconda3\envs\mini-agent-rl\Scripts\mini-agent-search.exe' query 'Who owned Rapana?' --retriever hybrid-rrf-v1
& 'D:\anaconda3\envs\mini-agent-rl\Scripts\mini-agent-search.exe' query 'Who owned Rapana?' --retriever bm25-v2
& 'D:\anaconda3\envs\mini-agent-rl\Scripts\mini-agent-search.exe' evaluate
```

亦可用项目Python执行 `python -m mini_agent_rl.retrieval.cli`。可选 `--config` 放在子命令之前。已有索引经哈希检查后复用，身份不匹配时明确失败，不覆盖历史索引。

## 接口

- `POST /api/search`：`{"query":"Who owned Rapana?","corpus_id":"goal-z-dev","retriever":"hybrid-rrf-v1"}`。返回文档、模式、索引身份和耗时，独立于Agent加载状态。
- `POST /api/runs`：原参数可增加 `retriever`，支持 `bm25-v2`、`dense-v1`、`hybrid-rrf-v1`。省略字段的旧请求继续使用BM25。
- `/api/status` 的 `retrieval.ready` 表示索引通过校验。历史回放明确标注原BM25，不声称使用新检索器。

每路召回20篇，RRF参数60，最终返回3篇。同一文档的多个向量块取最高余弦分数；融合按文档ID去重。三种模式分数含义不同，不应横向比较score的数值。标题最多32 token，正文220 token、重叠32 token，全部正文token均覆盖；查询超过256 token会截断，首版以短英文查询为目标。

## 本轮实测与边界

28项检索、索引、网页和旧BM25兼容性测试通过，包含索引缺失时明确报错且不回退、重复文档ID和语料哈希漂移检查。运行中的服务额外通过12次真实检索（6个并发线程、三种模式各4次），重复查询文档及分数一致，12份记录均在E盘保存并核对。补验报告：`E:\AgentLightningAssets\runs\retrieval-v1\service-check-11194453e6ee4dcc8542379df9a7221f.json`。这只是小规模并发功能检查，不是负载容量评估。

64道已暴露开发题只使用题面查询，在固定语料下运行三路检索，结果是检索诊断，不是端到端问答成功率。

| 模式 | Recall@3 | Recall@10 | 双侧支持@3 | 双侧支持@10 |
|---|---:|---:|---:|---:|
| BM25 | 56.25% | 64.84% | 9/64 | 20/64 |
| 向量 | 58.59% | 72.66% | 13/64 | 29/64 |
| 混合 | 58.59% | 73.44% | 12/64 | 30/64 |

报告：`E:\AgentLightningAssets\runs\retrieval-v1\evaluate-1c6e6fe8c99945bf9ca1eefc08ce72be\report.json`，同目录ledger保存192条逐题结果。计时排除编码模型初次加载；CPU编码模型预热后执行。

固定Goal Z的三个已有案例分别运行BM25和混合，共六次本机问答完成；其中Medinaceli案例BM25回答1575，混合因重复查询停止。Rapana两模式都重复查询退出，Hero/Walker两模式均回答Alex Cahill。说明工具集成可用，但检索指标改善不能直接解释为Agent问答提升，当前没有新的语义能力发布结论。

问答记录：`E:\AgentLightningAssets\runs\retrieval-v1\agent-smoke-cec73b1ff2b44361ad3a84d35e7ba6ba\summary.json`。原实验记录、模型权重及调用预算未改写。

## 完整64题端到端配对复验（2026-10-06）

复用上述64道开发题，固定Goal Z权重、seed=42、现有网页推理配置及预算，每题分别执行BM25与混合，共128次真实问答；交替两种模式执行顺序，不训练、不调参。以项目既有答案评分函数对参考答案及已有别名取最高分，保留完整工具调用。全部运行完成，评测后网页恢复AM模型。

| 指标 | BM25 | 混合 |
|---|---:|---:|
| 答案准确率（严格及归一化EM） | 23/64，35.94% | 25/64，39.06% |
| 平均token F1 | 0.4802 | 0.4906 |
| 全过程读到两侧支持文档 | 47/64，73.44% | 49/64，76.56% |
| 平均检索次数 | 2.016 | 2.016 |
| 平均单题时间（模型已加载） | 3.984秒 | 3.978秒 |
| 重复查询退出 | 4/64 | 5/64 |
| 调用预算耗尽 | 2/64 | 3/64 |

逐题准确率比较：4题改善、2题退步、58题不变，净提高3.125个百分点。配对bootstrap（2000次，seed=42）95%差值区间为[-4.69，10.94]个百分点，包含零；这是小样本开发集上的观察收益，不能描述为已证实的稳定泛化提升。题面检索Recall@10提高8.59个百分点与端到端准确率提高3.125个百分点是不同指标。

例如The Blind Girl作者去世日期、Swayze Field所属学校首位黑人学生的问题得到改善；Medinaceli父亲去世时间及唱片厂牌问题退步。单次推理未增加模型重复采样，不据结果挑题或调融合参数。

本轮无需另造题：现有题集已能验证接入前后差异。若后续验证新题表现，应预先冻结题集、核对训练/开发问题重合和2048篇库的证据覆盖；旧题改写或从已知语料构造的题只能标注为补充诊断，不当作独立泛化证据。无需为开展这些评测重新训练。

完整报告和128条逐次记录：`E:\AgentLightningAssets\runs\retrieval-v1\agent-paired-dev-e24475adc24549e1b19c7d12fce7380c\report.json`，同目录`ledger.jsonl`、`protocol.json`保留答案、证据、参数和资产身份。复验脚本为`scripts/evaluate_local_hybrid_agent.py`，使用运行中的网页API、暂时切换Goal Z并在结束后恢复原模型；运行时请保持网页无其他推理任务。

可用于简历的表述：实现本地BM25与MiniLM向量召回及RRF融合，在2048篇语料、64道开发题上，支持文档Recall@10由64.84%升至73.44%；固定Goal Z权重端到端配对评测中，答案EM由35.94%升至39.06%。面试中应说明这是开发集观察结果及样本量。

## 新增题集审核与实验（2026-10-06）

初筛发现92道未在Goal Z已选训练、开发、确认列表内且支持正文完整覆盖的候选题。排除10道已有质量拒绝记录后，候选池为82道（MuSiQue 50、2Wiki 32）。进一步进行本轮单人语义复核，检查答案关系、同名实体混淆、时间限定和证据充分性，再剔除42道，保留40道自然题（MuSiQue 26、2Wiki 14）。例如将电影导演与同名解剖学教授混淆、把在某地工作当作出生地、以共同出演推断婚姻关系的题目被拒绝。

合格自然题不足64道，因此未强行维持原计划32/32来源配额。另从现有文档构建24道有明确答案片段的单跳诊断题，总计64道，但两组分别报告，诊断题不算自然多跳题、独立未见题或新的泛化证据。题目问题文本经标准化与原已选题比较无完全重复，但该检查不能证明语义或事实不重合。

题集、分组、答案和别名在本次推理前冻结，顺序seed=20261006，SHA-256为`9e9ce1413b521bfa084a8bfc8d4a1e87a9ea7fa7f97e93605836bd747cbfcbc0`。选择和语义审查未使用本轮模型或检索表现。部分自然题出现在旧候选池中，且使用同一已知语料；历史扫描仅覆盖本地runs目录的JSON/JSONL完整case ID，不能证明没有其他形式的使用或暴露。这轮称为同语料补充评测，不称为严格独立留出集。

冻结题集及逐题审核位于：`E:\AgentLightningAssets\runs\retrieval-v1\supplement-reviewed-64-7011e1801e974f0eb055256abddbeab9`。其中`manifest.json`记录身份与历史引用，`semantic-review.jsonl`保留82道题的接受/拒绝理由，`natural.jsonl`和`constructed.jsonl`分别保存两组题目，`selected.jsonl`为随机混排后的执行顺序。

### 40道自然多跳补充题

| 指标 | BM25 | 向量 | 混合 |
|---|---:|---:|---:|
| 题面支持文档Recall@3 | 53.75% | 61.25% | 57.50% |
| 题面支持文档Recall@10 | 63.75% | 71.25% | 72.50% |
| 题面双侧支持命中@3 | 5/40 | 10/40 | 8/40 |
| 题面双侧支持命中@10 | 13/40 | 17/40 | 19/40 |

| 固定Goal Z端到端指标 | BM25 | 混合 |
|---|---:|---:|
| 答案EM（严格及归一化） | 18/40，45.00% | 15/40，37.50% |
| 平均token F1 | 0.6042 | 0.5420 |
| 全过程读到两侧支持文档 | 32/40，80.00% | 30/40，75.00% |
| 平均检索次数 | 2.000 | 1.975 |
| 平均单题时间（模型已加载） | 4.903秒 | 4.900秒 |
| 重复查询退出 | 1/40 | 2/40 |

1题改善、4题退步、35题不变，净EM下降7.5个百分点；配对bootstrap（2000次、seed=42）95%差值区间[-17.5，2.5]个百分点，包含零。题面召回改善8.75个百分点与原开发集8.59个百分点方向一致，但在当前Goal Z搜索循环中没有转化为稳定的问答收益。

改善题是Stewart Patridge所在大学首位黑人学生：BM25未读到两侧证据且答错，混合读到两侧证据并答对James Howard Meredith。退步题中，Barney演员在Batman中的角色、Princess Diaries人物、歌曲作曲者父亲三个案例，两路都读到支持文档，但混合分别回答Neil Patrick Harris、Julie Andrews、Felix Körling，而标准答案分别是Nightwing / Dick Grayson、Queen dowager Clarisse Renaldi、August Körling；这三例是中间实体/关系推理错误。Taifa of Ronda结束年份一题混合未读全支持文档并回答1066，而BM25回答1065。该分类是事后轨迹诊断，未修改已冻结题目或评分。

### 24道人工单跳诊断题

三种检索模式的题面Recall@3和Recall@10均为100%。固定Goal Z端到端两种模式的归一化EM均为18/24（75%），严格EM均为17/24，平均F1均为0.8389；逐题1题改善、1题退步、22题不变。这组说明简单有证据题上的功能表现，不用于证明混合检索收益，不与40道自然题混合计算主要结论。

### 记录、结论与复现

128次真实问答及192次题面检索均完成，核对每题完整配对、实际检索模式、固定adapter及语料身份。报告：`E:\AgentLightningAssets\runs\retrieval-v1\agent-paired-dev-29c9988ce7ca4dac882a3dbb73dd5c9f\report.json`；同目录`ledger.jsonl`保存128条问答与工具轨迹，`retrieval-ledger.jsonl`保存192条检索，`protocol.json`保存冻结题集清单、配置及资产身份。评测结束后网页恢复AM模型；未训练、未改权重、未修改历史实验。

复现脚本：`scripts/prepare_hybrid_supplement.py`生成待复核候选池，`scripts/finalize_hybrid_supplement.py`应用本轮证据审核并单独标注人工诊断题，`scripts/evaluate_local_hybrid_agent.py --cases <冻结题集的selected.jsonl绝对路径>`执行实验。准备脚本会生成新目录，不覆盖原题集。主实验固定Goal Z，不能直接外推为AM模型的问答表现。

最新简历建议：写本地BM25+MiniLM+RRF工程实现及两组自然题上的Recall@10收益，说明固定权重配对评测和失败轨迹分析。原64题的EM提高3.125个百分点仍是该开发集上的观察值；新增40题出现反向变化后，不宜将端到端问答提升作为整体卖点。后续若优化，先诊断不同召回排序和干扰文档导致的关系推理错误，不需要立即重新训练，也不根据这批题反复调参数后继续把它称为验证集。

后续已完成[最终关系复核实现与诊断](local-relation-review.md)：49 次生成，原混合错误修复 0/3，未达到阶段门槛，停止后续大规模评测。功能保留为默认关闭的 API 实验选项，不能宣称关系推理修复成功。

随后完成了[扩大到128题的 Goal Z 重测](local-hybrid-128-rerun.md)：两份历史账本覆盖的256条问答全部复现；104道自然多跳题上，混合检索EM略低于BM25，题面Recall@10较高但没有带来答案收益。
