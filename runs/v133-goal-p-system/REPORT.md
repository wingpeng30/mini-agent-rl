# Goal P：真实检索问答交付与有界失败处理

状态：`COMPLETED_WITH_NEGATIVE_TRAINING_GATE`。P0至P4已执行；系统改进候选1已保留，三臂训练和240条统一评价已完成。三臂结果未达到预先规定的后续复现门槛。默认演示模型仍为 Goal J SFT adapter。

## 实际交付的系统

交互入口是`scripts/demo_v133_goal_p_system.py`：一个进程常驻加载 Goal J，用户输入问题，模型产生JSON搜索或回答动作，`SearchTool`在共享文档索引中实际检索，结果进入下一轮模型上下文；输出原始动作、检索查询、文档ID/标题/片段、token、时延与结束原因。检索到的文档展示为“检索证据”；模型没有显式引用时，界面不声称它引用了文档。最多4次模型调用、3次实际搜索，重复查询或无效动作会记录退出。

演示索引由冻结48题的两篇支持证据各取一份，去重得到96篇文档。所有问题搜索同一索引，运行模型只接收问题和工具返回，不接收本题金标或支持ID。答案/支持映射保存在独立的`eval-only-cases.jsonl`，仅用于事后评分。这是**小型已知文档库上的真实工具调用**，不代表开放域检索。固定证据的合成两轮历史实验仍在Goal J/N/O原有目录，两种设置不可混称。

成功演示问题“Paul Young and David Draiman, share which mutual occupation?”：模型先搜索，工具返回包含David Draiman和Paul Young的文档及一个干扰文档，随后回答` songwriter`（实际字符串`"songwriter"`）；2次模型调用、1次真实检索、8.469秒，未截断上下文。完整轨迹见`demo-live-example.json`。

## P1与P2：同32题真实检索对照

Goal J、原`direct-v090`系统提示、greedy、thinking=false、128新token、2048输入token、同96篇共享索引、BM25-v2。两组各评价Goal J和Goal G已暴露开发集16题，所有32题均计入。候选1在看过P1失败类型后**预先写明**，只兼容模型输出`{"action":"search","text":"..."}`到原`query`字段；其它坏JSON继续无效。未更换模型、检索器或提示。

| 设置 | 严格EM | F1 | 无效动作 | 找到双侧支持证据 | 模型调用/搜索 | 进程内评价秒数 |
|---|---:|---:|---:|---:|---:|---:|
| P1原协议 | 13/32 | .5177 | 8/32 | 15/32 | 56/24 | 42.029 |
| P2通用字段别名 | 13/32 | .5802 | **3/32** | 17/32 | 61/29 | 44.489 |

P1的8个首次无效动作中，5个是合法JSON搜索但使用`text`字段，3个是坏JSON。候选1使这5题能够继续搜索并回答；五题严格EM仍全错，但两个题新检索到双侧证据，因此F1上升。其它27题结果未改变。依预设规则：无效动作下降、严格EM/F1均不下降，**保留候选1**，不运行候选2。改进是协议完成率与可观察证据，不是新增严格正确答案。

P2后17题找齐双侧证据，其中9题仍未严格答对；另有5题未找齐两侧支持却严格答对。这提示检索质量和答案判断都仍有独立缺口，也提醒“答对”不能自动证明使用了两侧证据。首次动作仍有3个坏JSON；本轮没有把坏JSON自动解释成策略动作。

两个开发集已参与多轮研究决策，P2是开发干预结果。小文档库由相同48题证据形成，双侧召回仅反映这个封闭范围。此次真实检索的13/32与先前固定证据Goal J的19/32具有不同输入和检索条件，不能把6题差额直接归因于单一组件。

## P3-F0/F1：失败机制核查

Goal O既有768个候选/192个组复算：106组无严格正确候选，52组全正确，34组既有正确又有错误；105个非零奖励方差组中，71组的四个候选全部严格错误。6道训练题在三seed各48次采样中从未严格答对：`band`、`director`、`writer`、`city`、`flowering plant`、`genus`。其中既有词面/粒度差异，也有共同属性或分类层级错误。逐题分类见`p3-f0-signal.json`；这只是对既有训练输出的事后诊断，没有改金标或补采。

新配方使用temperature1、top-p1、top-k0，采样与原始teacher-forced logprob约定一致；沿用v118起点、16条冻结训练题、LoRA、奖励`EM+0.25×F1`、组大小4、KL beta.02和原长度限制。CPU可解析目标测试核对组内优势、正确/错误动作的梯度方向、KL回拉方向及零方差跳过。16题监督目标的assistant-only mask再次验证，前缀/tool没有监督标签。最多2道训练题/8候选的GPU核查用了一次可丢弃SFT更新：首题loss .7636→.6761，192个LoRA tensor变化、base不变、有限梯度；核查模型没有作为正式训练起点。见`p3-f1-cpu-audit.json`与`p3-training/probe.json`。

该核查证明新监督路径可以反传，没有单独证明完整RL目标在真实模型上的每一步都朝严格EM上升。单次每组更新前ratio约为1仍只是重放检查，不能把clip当作更新后策略限制；新配方与Goal O还同时改变温度，跨Goal差异不能单独归因于温度修正。

## P3-F2：一次预定三臂实验

三个臂均从初始v118独立加载，seed42、冻结16题顺序、4遍、每题一次训练机会。A仅标准答案assistant-only交叉熵；B仅GRPO；C为GRPO加固定`0.1×`监督loss，一个组最多一步，零方差组仅监督更新。A/B/C训练预算和实际更新数不同，结果不是等算力比较。最终checkpoint在评价前预定，未依据开发集挑选。三个臂均通过LoRA/base/reference和独立重载检查；B/C各256条候选，合计512条。

| 模型 | 训练EM/F1 | 主要开发EM/F1 | 回归开发EM/F1 | 更新数 | 训练秒数 |
|---|---:|---:|---:|---:|---:|
| v118 | 4/16 / .3800 | 9/16 / .6094 | 3/16 / .3497 | — | — |
| Goal J SFT | **16/16 / 1.0000** | 10/16 / .7813 | **9/16 / .7125** | 历史20次 | 历史结果 |
| A：SFT-only | 14/16 / .9063 | 10/16 / .8000 | 6/16 / .5250 | 64 | 32.478 |
| B：RL-only | 6/16 / .6389 | **12/16 / .8385** | 7/16 / .6250 | 42 | 228.798 |
| C：SFT+RL | 11/16 / .8438 | **12/16 / .8333** | 6/16 / .5833 | 64，其中46组有RL项 | 257.294 |

三个新臂在48题冻结输出均为有效answer。五模型×48题共240条，逐条复算EM/F1与运行记录一致。C相对A在主要开发净增2题、回归没有变化，**但训练净丢3题**。C的主要开发高于Goal J，回归6/16低于Goal J的9/16；没有通过预设的“两个开发集均不低于Goal J”条件。B主要开发也为12/16，回归7/16高于C。因此这次单seed数据不能证明联合目标优于单独RL，更不能声称稳定优于SFT。

原先6道训练难题：v118答对0、Goal J答对6、A答对4、B答对0、C答对2。C相对B显示监督项有助于部分目标答案可达性；A更多，说明联合RL项没有全面保留监督的训练拟合。所有模型都是单seed及已暴露开发评测，差异不构成独立泛化证据。逐题新增/丢失见`offline-summary.json`，训练曲线见`p3-training-curves.png`。

## P3-F3：按六类问题结案

| 问题 | 本次证据与状态 |
|---|---|
| 起点选择 | Goal J在历史两开发集合计19/32，是本系统默认模型；P3三臂从v118重新起步以回答新配方问题，不更换产品默认模型。已处理。 |
| 采样无正例 | Goal O的106/192组无正例。C在原6道题中答对2、B为0，支持监督信号能帮助部分题；A答对4，联合效果不足。部分处理，机制仍不确定。 |
| 词面奖励 | `flowering plants`等存在严格EM粒度差异；冻结评分保持。未尝试事后别名或开发题定制奖励，词面与语义提升仍未分离。 |
| 动作偏好 | 固定证据下各新臂均为answer；真实系统原有8次协议无效、别名兼容后为3次。完成率有工程提升，正确性未同步提高。 |
| 真实检索 | 已完成真实工具闭环；32题中17题找齐双侧支持，9题仍答错。系统和检索各有可见缺口；不声称自主检索能力已经稳定提升。 |
| 算法约定 | 温度1与原始logprob统一，CPU方向/掩码及GPU监督更新验收通过。Goal O温度不一致已避免，但没有完成因果消融；单次更新的clip保护仍有局限。 |

结论：保留Goal J+候选1协议兼容作为项目可用系统；三臂结果作为可信探索记录，不追加seed、第五遍、权重搜索或扩训练数据。后续若研究答案质量，应先定义独立确认性测量和真实检索目标；这不是本Goal交付的前置条件。

## 复现与演示

AutoDL工作目录：`/root/autodl-tmp/mini-agent-rl`。首次复现的`prepare`和`run`需要**新隔离输出目录**；现有目录拒绝覆盖。查看本次结果不加载模型。真实交互需要GPU，模型常驻在进程中；空行退出。

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_p_system.py
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_p_system.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

固定证据的历史回放与离线审计：

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_v133_fixed_evidence_replay.py --split new_eval --case-id 5ac00c72554299012d1db589
PYTHONPATH=src .venv/bin/python scripts/analyze_v133_goal_p.py
```

本次首次运行命令（复现时须换用新隔离目录，避免覆盖）：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_p_system.py prepare
timeout -k 10s 1200s env PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_p_system.py eval
timeout -k 10s 900s env PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_p_variant.py
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_p_training.py prepare
timeout -k 10s 6900s env PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_p_training.py run
```

哈希、配置、原始逐题轨迹、三臂采样、适配器和资源台账均保存在本目录。P3阶段GPU实际执行时间702.592秒；P1/P2/P4模型内执行约95秒，加启动开销后在总10800秒上限以内。平台有确定性kernel警告与旧PEFT配置兼容警告；固定输出均逐条保留，本次没有借警告改依赖或重跑。

演示建议：先运行上述一道真实检索例，展示问题→搜索动作→3篇工具文档→最终答案→可追溯轨迹；随后展示P1/P2失败对照和三臂表格，明确说明“协议完成率提高、模型严格答案能力没有在真实系统基准上增长”。简历/面试可以准确描述：在0.8B LoRA模型上完成SFT、固定证据GRPO、三seed预算验证、SFT/RL/联合目标对照及真实检索系统；建立冻结数据、重载与逐题评测，并根据负结果停止无收益的扩训练。
