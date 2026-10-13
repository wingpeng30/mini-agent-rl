# Mini Agent RL

新会话请先读[当前项目状态](PROJECT_STATUS.md)。下方历史阶段中的“当前/最终”按其记录日期理解。

GitHub代码同步于2026-10-13：包含最新源码、脚本、配置、测试、文档和汇总实验结果。原始数据、逐条训练/评价轨迹、模型权重、优化器恢复点和本地归档未随本次提交上传；下文提及这些文件的路径是本地实验记录，复现实验需要另行准备相应资产。

## 最新收尾实验（2026-10-12）

已在AutoDL 4090D完成固定0.8B的真实GPU批量生成、异步工具执行、继续SFT、在线GRPO双seed及原始裸模型评价。六条执行路径均完成同一128题；三个训练分支均保存最终adapter、恢复点和完整账本。结果归档已取回并校验，远端关机流程已触发，SSH已关闭。详见[结果解释与两条简历表述](docs/final-online-grpo-20261012-results.md)、[完整实验报告](runs/final-online-grpo-20261012/results/REPORT.md)、[综合指标](runs/final-online-grpo-20261012/results/comprehensive-metrics.md)及[实现与复现说明](docs/final-online-grpo-20261012.md)。

| 分支 | 自然104题归一化EM / F1 | 全128题归一化EM / F1 |
|---|---|---|
| AM＋混合检索 | 47/104 / 0.5907 | 67/128 / 0.6466 |
| AM＋继续SFT | 49/104 / 0.5855 | 68/128 / 0.6447 |
| AM＋在线GRPO seed42 | 40/104 / 0.5074 | 61/128 / 0.5868 |
| AM＋在线GRPO seed43 | 46/104 / 0.5847 | 66/128 / 0.6480 |
| 裸0.8B，同Agent协议 | 0/104 / 0 | 0/128 / 0 |

batch4吞吐由12.71提升到35.34题/分钟（2.78倍），峰值reserved显存2.04→4.31GiB；同128题EM67→64、答案文本变化6题。GRPO两seed各采样512条轨迹，实际更新108/110次，未带来一致质量提升；继续SFT仅有小幅EM变化且自然题F1略降。裸模型没有完成最终答案输出，113题重复查询、14题无效JSON、1题耗尽调用预算；其零分表示当前工具协议下的失败，不能解释为通用问答能力为零。本轮题集已暴露，指标不是新增语义证据支持成功率。

下方Goal BD/AG/AA及更早内容保留各阶段当时的结果和产品选择；本轮实验的EM/F1与历史证据支持评分分别报告。

本地混合检索已接入网页和命令行：BM25、MiniLM向量、RRF混合三种模式，复用2048篇开发语料，模型及索引存于E盘，无需API或训练。使用、接口与实测边界见[本地混合检索说明](docs/local-hybrid-retrieval.md)。

## 历史面试版（Goal BD，2026-10-06）

最后一轮固定Goal Z与AM普通SFT进行208题统一已暴露回归；0训练、0教师、0新确认。**正式模型始终是AA发布的Goal Z seed42**，AM仅为研究候选，原停止决定不撤销。

最终結果入口：[BD报告](runs/v133-goal-bd-final-regression-v1/REPORT.md)、[判定账本](runs/v133-goal-bd-final-regression-v1/decision.json)、[面试与复现说明](docs/v133-final-interview-guide.md)。416条轨迹完成：开发支持成功19→21/29，全208题EM106→111、动作问题12→3；AM通过本轮局部收益及机械保护门槛。旧主要／旧回归F1略降，实体关系错配尚未解决，不宣布全面能力提升。AA历史确认收益12/64→35/64、36/64保留原口径；BD的EM/F1及29题证据评分不能与其混算，也不能称旧144题全面语义保护通过。

默认离线回放（无GPU；明确显示档案与历史延迟）：

```bash
PYTHONPATH=src:scripts python scripts/demo_v133_final.py
PYTHONPATH=src:scripts python scripts/demo_v133_final.py --model am-research
```

实时模式必须显式`--mode live`；自定义问题必须`--corpus`，避免原96篇默认库导致库外空检索。新入口明确标注BD宽窗研究配置，不改旧正式入口默认值。代码及小型回放快照、环境与远程资产哈希由BD冻结清单记录，不包含`.env`或模型权重。

交付状态：评价、评分和六个离线回放已完成。最终快照打包因文件清单错误停止，尚未生成冻结清单；现场和剩余工作见[FREEZE_STATUS](runs/v133-goal-bd-final-regression-v1/FREEZE_STATUS.md)。不自动重跑实验或归档。

## 历史可复现结果（v133各阶段）

**Goal AG 已完成新题确认，未发布新模型；当前产品仍为 Goal Z 完整轨迹 seed 42。** 修正真实失败状态路线 B 在 Goal Z 开发集从 34/64 提高到 39/64、38/64，并通过旧题回归门槛；但冻结新确认集上起点为 30/64，B 两 seed 为 29/64、32/64，均未达到预设的至少 +4/64。错误合计为 31、32、29，无效动作均为 0。开发集收益未在本次确认中复现为足够稳定的产品增益，因此不切换 adapter、不追加训练。109 个匿名组合的盲审及主审修订、192 条确认轨迹和逐项结果见 [Goal AG 报告](runs/v133-goal-ag-failure-repair/REPORT.md)。该确认集现已暴露，后续不得再称作独立确认集。

**Goal AB 已完成证据充分性决策监督的四次训练与开发验收，产品保持 Goal AA 发布的 Goal Z 完整轨迹 seed 42。** 54 对真实上下文决策材料用于匹配的 A（继续完整轨迹 SFT）与 B（完整轨迹加决策监督）对照，各两 seed、每次 128 更新。Goal Z 开发集同口径语义成功为起点 **36/64**、A **38/64、36/64**、B **39/64、37/64**；两路线都未达到两个 seed 各至少 40/64、且错误不增加的预设门槛。B 相对 A 各多 1 题，但停止错误未在两个 seed 上同时下降。新确认集已冻结且保持关闭，没有确认输出。完整结果、逐题账本、保护集口径更正及复现命令见 [Goal AB 报告](runs/v133-goal-ab-evidence-decision/REPORT.md)。演示入口现支持 `--corpus /path/to/matching-corpus.json` 并打印语料哈希；默认仍用旧 96 篇库。

**Goal AA 已复核 Goal Z 并完成冻结新题确认，当前产品选用完整轨迹 SFT seed 42。** 720 条旧输出同口径重审后，新开发语义成功为 Goal T 起点 **10/64**、完整轨迹两 seed **36/64、32/64**，均通过原开发门槛。用户授权受控恢复一次启动解释器错误后，冻结确认集的“答案正确、信息充分且实际可见证据支持”为起点 **12/64**、完整轨迹 seed 42 **35/64**、seed 43 **36/64**；两个 seed 均达到预设提升与错误不增加门槛。严格 EM 为 9/64、26/64、25/64。按预设发布 seed 42，而非确认分更高的 seed 43。确认语料是有限封闭库，审核由代理辅助，不能解释为开放域泛化或独立人工金标准。详见 [Goal AA 报告](runs/v133-goal-aa-audit-confirmation/REPORT.md)；[Goal Z 原报告](runs/v133-goal-z-multihop-sft/REPORT.md)保留更正前记录。

当前产品真实搜索入口优先读取 Goal AA 的发布决定，加载完整轨迹 SFT seed 42 adapter：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_z_selected.py --question '你的问题'
```

演示入口默认使用旧产品的 96 篇文档库；问题若不在库中，检索可能为空。Goal AA 的新题确认分数来自独立的 2048 篇确认语料，不能用一个库外演示回答代替证据检验。

**Goal Y 已完成三类真实证据缺陷对的小规模训练与清洁新题开发检验，产品未切换。** 原定三类各 16 对仅审核出职业 12、实体类型 8、来源 5，故将原 48 对目标判为数据不足；另行冻结这 25 对做两 seed 的 SFT/DPO 对照。清洁新开发 48 题的“正确充分且有证据”从 Goal T 起点 40/48 到 SFT 42/48、41/48 或 DPO 41/48、41/48，均未达到两 seed 各至少 +3 的确认门槛。确认集未运行；当前产品仍用 Goal T seed 42 update 32。完整证据、逐题变化、checkpoint、复现命令见 [Goal Y 报告](runs/v133-goal-y-three-class-transfer/REPORT.md)。

**Goal X 完成了 Goal W 候选池的完整题义与真实检索验收，按原四类门槛停止。** 512 道训练候选逐题审核后，342 道进入一次真实搜索，335 道在最终回答前完整看到双方证据；但“共同隶属关系”类最多只有 4 道可用，低于最低 16 道。没有启动新训练或确认集模型评价，产品仍选 Goal T seed 42、第 32 次更新。Goal W 遗漏的后续历史筛查、训练、评价与标题重叠已分层补核，详情及逐题账本见 [Goal X 报告](runs/v133-goal-x-audit-and-transfer/REPORT.md)。

**Goal W 已完成扩大训练前的数据验收，并按预设门槛停止。** 旧开发基线重新得到 20/32、回归 7/16、无效动作 0；公开来源中的封存排除键也经 ID 哈希复核。冻结候选池和三个独立的 2048 篇自然段落语料已保存，但训练池中“共同隶属关系”类按原题问法宽松计入也至多 12 道，低于要求的 16 道，因此没有启动新训练、开发选模或确认集模型评价。产品仍选用 Goal T seed 42、第 32 次更新的 adapter。[Goal W 报告](runs/v133-goal-w-transfer/REPORT.md)保存候选池、逐题排除、停止决定和复算命令。

**Goal U→V 已完成双侧证据训练与冻结新题验证，产品模型保持 Goal T seed 42、第 32 次更新。** 在旧开发集，偏好训练两 seed 均从 20/32 到 21/32，语义充分答案从 23/32 到 25/32，单侧证据错误从 6 降至 5；普通 chosen SFT 的严格 EM 也是 21/32，但未减少单侧错误。新 32 题上，起点为 21/32，两个偏好模型均为 20/32；三者语义充分均为 29/32。因此没有发布新 adapter。训练对、四臂曲线、逐题盲审、来源冻结、代码和完整决策见 [Goal U→V 报告](runs/v133-goal-u-v-evidence-preference/REPORT.md)。

Goal U→V 阶段的产品入口读取该阶段最终选模决定：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_uv_selected.py
```

CPU 离线复算：`PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_u_v_evidence_preference.py summarize`。以下 Goal T/S/R/Q/P 段落记录各阶段历史结果；现行选模与入口以上方 Goal U→V 决策为准。

**Goal T 已按预设门槛更新真实搜索产品模型。** 从 Goal P RL-only 出发，沿用 Goal S 的两处修订标签；两个 seed 的第 32 次更新模型分别达到 **20/32（回归 7/16）** 和 **21/32（回归 8/16）**，均超过原产品的 19/32，且无效动作仍为 0。因此当前入口选用 **seed 42、第 32 次更新**的 adapter。第 64 次更新模型分别为 18/32 和 21/32。新增严格正确主要是答案格式更贴近金标，尚不能说明新事实问题的语义成功率提高。逐题变化、训练曲线和历史差异见 [Goal T 报告](runs/v133-goal-t-update-length/REPORT.md)。

Goal T 阶段的实时搜索入口：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_t_selected.py
```

CPU 离线复算：`PYTHONPATH=src:scripts python scripts/run_v133_goal_t_update_length_v2.py summarize`。以下 Goal S/R/Q/P 段落记录各阶段历史结果；其中“当前模型”指该阶段当时的选择。

**Goal S 已完成训练标签粒度的配对实验。** 仅把 16 道训练题中两处有双侧证据支持的标签改为 `rock band`、`film director`。相同起点和预算下，原标签／修订标签的真实搜索 EM：seed 42 为 15/32→18/32，seed 43 为 16/32→20/32；修订使部分过粗回答恢复，但 seed 42 的回归仅 6/16，未通过双 seed 产品发布门槛。当前推荐仍为 Goal P RL-only（19/32、回归 7/16）。完整逐题、语义审查和预算见 [Goal S 报告](runs/v133-goal-s-label-granularity/REPORT.md)。

AutoDL 当前选中模型的实时入口：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_s_selected.py
```

CPU 离线复算：`PYTHONPATH=src:scripts python scripts/analyze_v133_goal_s_label_granularity.py`。以下 Goal R/Q/P 段落记录历史结果。

**Goal R 已完成从当前最佳搜索模型出发的答案训练对照。** 普通答案 SFT 与证据顺序变化 SFT 在原产品入口均为 **15/32**，低于当前最佳 Goal P RL-only 的 **19/32**；回归开发为6/16，对照模型为7/16，因此不升级。两臂训练时每题均有真实检索到的双侧证据，产品评价中也仍是32/32双侧召回；退步主要出现在最终答案。文档顺序反转会改变部分回答，但本轮顺序变化训练未比普通答案 SFT 提供额外收益。完整轨迹、逐题变化及诊断故障修复见 [Goal R 报告](runs/v133-goal-r-evidence-order/REPORT.md)。

AutoDL 当前选中模型的实时交互入口：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_r_selected.py
```

CPU 离线复算：`PYTHONPATH=src:scripts python scripts/analyze_v133_goal_r_evidence_order.py`。以下 Goal Q/P 段落保留各自阶段的历史设置和结果。

**Goal Q 已从 Goal J 出发完成真实搜索轨迹 SFT 与在线 RL。** 在同一 96 篇文档、32 道已暴露开发题的真实工具调用评价中，Goal J 为 13/32；从 Goal J 训练的轨迹 SFT 为 16/32、直接 RL 为 12/32；已有 Goal P RL-only 为 **19/32**，且回归开发与 Goal J 同为 7/16。因此当前产品演示选择 Goal P RL-only adapter，未将本轮新模型升级。它在该小型已知文档库上 32/32 题检索到双侧支持；这不代表开放域性能。完整逐题结果、训练曲线、预算和选择规则见 [Goal Q 报告](runs/v133-goal-q-search-success/REPORT.md)。

AutoDL 实时交互入口（GPU、单进程常驻选中模型）：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_q_selected.py
```

CPU 离线复算：`PYTHONPATH=src:scripts python scripts/analyze_v133_goal_q_search.py`。下方 Goal P 段落记录其历史阶段结果与原演示入口。

**Goal P 已交付真实检索问答入口。** 在AutoDL中，Goal J模型可常驻运行，实际调用本地检索工具并展示查询、文档、最终动作和轨迹。96篇共享冻结证据文档上的32题开发基线为13/32严格EM、8次协议无效；兼容通用`search.text`字段后，无效降至3次，EM仍为13/32，F1从.518升至.580。后续一次从v118起步的SFT-only、RL-only、SFT+RL三臂比较未通过预设的继续复现门槛；Goal J继续作为演示默认模型。范围、全部对照、曲线和准确表述见[Goal P 报告](runs/v133-goal-p-system/REPORT.md)。

AutoDL实时交互（需要GPU；一进程常驻模型，空行退出）：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_p_system.py
```

该入口检索96篇已知文档库，运行时不读取本题金标或支持映射。展示的是检索到的证据，不冒称模型明确引用；不代表开放域能力。冻结输出的CPU离线审计入口为`PYTHONPATH=src .venv/bin/python scripts/analyze_v133_goal_p.py`。

Goal O 已在相同16条训练题上完成 seed 42/43/44 各四遍固定证据 LoRA-GRPO，六个预定checkpoint及384条统一评价均通过。第1→4遍主要开发严格EM分别为10→11、11→11、11→10（每格16题）；回归开发为3→4、3→3、4→5。训练拟合均值6→7/16，但主要开发均值10.67→10.67/16，未达到预先规定的三seed稳定改善门槛。**当前不启动大规模数据实验或第5遍训练。**完整表格、逐题得失、曲线、冻结配置及命令见 [Goal O 报告](runs/v133-goal-o-budget/REPORT.md)。

Qwen3.5-0.8B 已完成两条从 v118 SFT adapter 出发的真实训练分支：Goal J 用16条审核合格题完成20次追加 LoRA-SFT 更新；Goal N 在通过一次64条训练题信号探针后，完成9次固定证据 LoRA-GRPO 更新。同期主要开发集严格EM为v118 **9/16**、Goal J **10/16**、Goal N **11/16**；Goal G回归开发集为**3/16、9/16、5/16**。训练集分别为**4/16、16/16、6/16**。这些都是小样本探索结果，未使用独立确认集；两条分支预算不同。Goal M 已停止从训练集饱和的Goal J继续RL，Goal N则明确从v118独立起步。方法、完整F1、动作类别和限制见 [Goal N 报告](runs/v133-goal-n-v118-grpo/REPORT.md)及 [v133 实验索引](docs/v133-experiment-index.md)。

无需 GPU 的固定证据回放演示（显示问题、两条证据和三套adapter的冻结动作，**不实时生成**）：

```bash
python scripts/demo_v133_fixed_evidence_replay.py --split new_eval --case-id 5ac00c72554299012d1db589
```

回放中的两轮 search 历史是按冻结证据合成的上下文，不代表自主检索轨迹。`--split` 可选 `train`、`new_eval`、`goal_G_regression`；不传 `--case-id` 时显示该组第一题。

AutoDL GPU上实时加载三套adapter生成同一题：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_fixed_evidence_live.py --split new_eval --case-id 5ac00c72554299012d1db589
```

实时入口会计入 Goal N 的GPU预算，回放命令不加载模型。两种演示都只评估已冻结的固定证据回答状态。

当前候选版本为 v0.8.7。它新增可审计的 `compact-v2` 证据表达与精确输入重放；768-token 的训练门槛尚未通过，因此不会把该版本表述为已验证的 GRPO 改善。参见 `docs/v087-input-replay.md`。

早期收口实验还加入 `compact-v3` 的低排名优先裁剪；对应 GRPO 冒烟计划是独立于当前 v133 固定证据 Goal M/N 的旧路线。

一个可离线运行、并为本地小模型训练保留强化学习数据边界的最小工具调用 Agent 框架。FakeModel 路径仍不需要 GPU；本地模型适配器采用懒加载，不会在安装基础依赖或执行测试时下载权重。

当前开发版本：**v0.8.7（未发布）**。本版补充 compact-v3 的证据保留和精确输入重放，并记录受控训练、评测和停止门槛；版本说明见 [CHANGELOG.md](CHANGELOG.md)。

v115 的受控 SFT 试验已完成一次50步训练，但内部评测未达到证据充分回答和证据不足搜索门槛；因此未打开独立100题，未启动 GRPO。完整方法、限制与结果见 [v115 实验记录](docs/v115-sft-experiment.md)。

```powershell
mini-agent-rl experiment preflight --config configs/v083-local-8gb.toml
mini-agent-rl experiment inspect runs/v083-local-8gb-smoke
```

## 快速运行

```bash
pip install -e ".[test]"
mini-agent-rl run --group-size 4
mini-agent-rl evaluate --task examples/benchmark.jsonl --output reports/baseline.json
mini-agent-rl export
```

默认使用 FakeModel 和内置离线 SearchTool。真实模型可通过 `OpenAICompatibleClient` 接入兼容 Chat Completions 的服务。

## 本地模型（下载前准备）

当前版本提供 Qwen3.5-0.8B 的本地适配框架，但不会自动下载模型。先执行只读预检：

```powershell
mini-agent-rl local-check
```

准备进行真实推理时再安装可选依赖并启动；第一次启动才可能从 Hugging Face 下载权重：

```powershell
pip install -e ".[local-model]"
mini-agent-rl local-smoke --model-path D:\qwen_08b
```

RTX 4060 8GB 首次推理使用 BF16、`group-size 1`、最大 2048 输入 token。Qwen3.5 需要项目固定的 Transformers 兼容 commit；模型权重、checkpoint 和 `*.safetensors` 已加入 `.gitignore`。

## 生成离线 SFT 数据

```powershell
mini-agent-rl generate-sft --corpus examples/corpus.json --output-dir data/sft-v040
```

默认生成 800 条 train、100 条 validation 和 100 条 test 数据，不调用 API。标准答案只写入 metadata，模型 messages 中只包含问题、search 动作、工具 observation 和 answer 动作。

## 导入多跳检索数据

```powershell
pip install -e ".[data]"
mini-agent-rl download-hotpot --output-dir data/hotpot-agent-v041
```

该命令从 HotpotQA 流式导入 800 条英文多跳检索样本，保留 supporting documents 并按文档级别切分。详见 [HotpotQA 数据说明](docs/hotpot-data.md)。

```powershell
mini-agent-rl audit-sft --data-dir data/hotpot-agent-v041
```

`data/hotpot-agent-v041/test.jsonl` 是开发评测集：可用于排错、选 checkpoint 和多次对照，不能再称为最终测试集。

## 隔离的最终测试集

当训练与开发阶段结束后，先生成一份从未参与训练、调参或 checkpoint 选择的最终集。该命令仍是流式读取，不会下载完整 HotpotQA：

```powershell
mini-agent-rl prepare-final-test --source-dir data/hotpot-agent-v041 --output-dir data/hotpot-agent-v072 --count 300
mini-agent-rl audit-final-test --source-dir data/hotpot-agent-v041 --final-dir data/hotpot-agent-v072
```

生成的 manifest 固定来源 split 与最终文件的 SHA256，并审计 task ID 与 supporting-document 标题均无重叠。选定且冻结唯一 checkpoint 后，才执行一次最终评测；该命令要求显式确认，防止误把最终集用于调参：

```powershell
mini-agent-rl evaluate-final --adapter-path checkpoints/qwen35-08b-grpo-v071-seed43 --confirm-final
```

## LoRA-SFT

```powershell
mini-agent-rl train-sft --model-path D:\qwen_08b --max-steps 10 --output-dir checkpoints/qwen35-08b-sft-smoke
```

训练仅更新 LoRA adapter，且只计算 assistant action token 的 loss。详见 [SFT 训练说明](docs/sft-training.md)。

## LoRA Adapter 对比评测

```powershell
mini-agent-rl compare-adapter --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-smoke --limit 8 --output reports/adapter-comparison-v051.json
```

该命令让基础模型与 LoRA adapter 在同一批完全离线的 Hotpot 测试任务、语料、解码参数和步数限制下运行，报告准确率、动作合法率、检索成功率、奖励、步数、加载时间和显存。详见 [Adapter 对比评测](docs/adapter-evaluation.md)。

## RL 轨迹采集

```powershell
mini-agent-rl collect-rl --model-path D:\qwen_08b --adapter-path checkpoints/qwen35-08b-sft-v052 --task-limit 10 --group-size 4 --temperature 0.7
```

该命令只采集 sampled rollout、逐 action token logprob 和 group advantage，不更新任何权重。重复搜索、非法 JSON 和最大步数均作为带负奖励的策略轨迹保存。详见 [RL 轨迹说明](docs/rl-rollout.md)。

## DeepSeek（可选）

不需要部署本地模型。设置环境变量后可切换到 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="在控制台创建的新密钥"
mini-agent-rl evaluate --backend deepseek --task examples/benchmark.jsonl
```

默认使用 `https://api.deepseek.com` 与 `deepseek-v4-flash`；密钥不会写入 SQLite、报告或 Git。

## 当前范围

当前版本包含最小 LoRA-GRPO 冒烟训练：训练前必须通过同实例 logprob 稳定性验证；工具 observation 只作为上下文，不参与 policy loss。v0.8 的 reward-ablation 与停止策略实验均保留为本地审计产物，尚未形成可发布的泛化改进，故最新发布标签仍为 `v0.7.3`。

### Goal AD：失败状态桥接查询审计

Goal AD 复核了 Goal AC 的 48 条真实路径：14 条有完整可见证据，33 条证据不足，1 条仍有歧义。对 AC 原先拒绝的 34 条路径逐条诊断后，25 条缺少必要语料关系，8 条属于标签或审核问题，1 条的支持句被截断在模型实际输入之外。没有样本同时满足“语料中有必要证据、仍有调用预算、可由修正查询补齐”，低于预设的 8 题启动门槛，因此本轮在制作新查询和训练前停止；没有消耗 GPU，也没有更改产品 adapter。

审核使用保存的输入 token 重建模型实际看到的片段。封存确认集曾被做过一次仅查看记录数量和字段名的 schema 检查；问题、答案和模型输出未显示或用于决策，确认集未生成评价结果。完整账本与运行配置见 [Goal AD 报告](runs/v133-goal-ad-bridge-query/REPORT.md)、[逐题诊断](runs/v133-goal-ad-bridge-query/failure-diagnosis.jsonl) 和 [审核更正](runs/v133-goal-ad-bridge-query/root-path-adjudication.jsonl)。

### Goal AF：真实产品模型下的检索对照

Goal AF 用已发布的 Goal Z 连续轨迹 seed 42 模型完成了原 BM25-v2（A）、旧段落检索（B）和页面优先片段选择（C）的 624 条真实搜索评价。对照 A 复现了 Goal Z 的原始轨迹；严格 EM 合计为 A 107/208、B 36/208、C 88/208。2026-09-30 主 agent 复核发现至少三条 C 输出在最终实际输入包含完整答案关系时仍被错误标为证据缺失，因此此前 B/C 的语义总分和门槛决策需要重新审核，不能当作最终结论。原报告和原审核保留，问题依据见 [审核问题记录](runs/v133-goal-af-retrieval-validation/audit-issues-20260930.json)。当前继续使用 Goal Z seed 42＋BM25-v2；新确认集未打开。完整结果、逐题记录和 CPU 检索回放见 [Goal AF 报告](runs/v133-goal-af-retrieval-validation/REPORT.md)。

本地 8GB 显存的上下文诊断可使用：

```powershell
mini-agent-rl context-length-sweep --lengths 384,768,1024,2048
mini-agent-rl probe-context-backward --max-input-tokens 768
```

结构化预算会保护 system/user，按文档边界压缩 observation，并分别记录采样和 logprob 重算的截断率。当前 768-token 只通过 5×4 冒烟；10×4 的截断率与 reserved 显存超过门槛，因此仍属于未发布实验。

## 文档

- [最终关系复核实现与失败诊断](docs/local-relation-review.md)
- [Goal Z 128 题混合检索重测](docs/local-hybrid-128-rerun.md)
- [架构](docs/architecture.md)
- [数据模型](docs/data-model.md)
- [使用方式](docs/usage.md)
- [任务成果](docs/task-summary.md)
- [本地模型基础框架](docs/local-model.md)
- [Adapter 对比评测](docs/adapter-evaluation.md)
- [Logprob 稳定性](docs/logprob-stability.md)
- [最小 GRPO 训练](docs/grpo-training.md)
- [v0.7.1 训练正确性审计](docs/v071-training-audit.md)
- [HotpotQA 数据与最终评测协议](docs/hotpot-data.md)
- [最终测试与 checkpoint 选择协议](docs/final-evaluation-protocol.md)
- [v0.7.2 最终评测分析](docs/final-evaluation-v072.md)
- [v0.7.3 诊断与对照](docs/v073-diagnostics.md)
- [v0.8.1 停止策略诊断（未发布实验）](docs/v081-stop-policy-diagnosis.md)
- [v0.8.2 上下文预算与 8GB 显存实验](docs/v082-context-budget.md)
- [v0.8.4 固定检索环境与可信 GRPO 对照](docs/v084-fixed-retrieval.md)
- [v0.8.5 困难停止决策诊断](docs/v085-stop-decisions.md)
- [v0.8.6 证据表达压缩与真实输入审计](docs/v086-observation-encoding.md)
