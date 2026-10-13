# Goal O：固定数据下的训练预算扩展与三 seed 验证

状态：`EXPERIMENT_COMPLETE`，离线评分审计 `AUDITED`。三个 seed 的四遍训练、六个预定 checkpoint、八模型统一评价均完成。**预先约定的扩训练数据门槛未通过；结束本轮预算扩展，不启动大规模数据实验。**

## 固定设计与执行

从同一 v118 LoRA adapter 分别以 seed 42、43、44 初始化；每次按冻结顺序处理训练16题四遍，每题采四个候选。每组奖励有方差时仅更新一次；零方差跳过，不补采。第一遍和第四遍 checkpoint 在评价前预定。reference 恒为各次训练初始 v118 LoRA 的独立快照。原 Goal N 运行器、奖励、AdamW 学习率5e-6、温度0.7、top-p1、top-k0、组大小4、clip0.2、KL beta0.02、梯度裁剪1.0及128新token上限保持一致。采样分布使用温度0.7，old/current/reference 的 teacher-forced logprob 使用原始未调温策略；首次 ratio=1 仅验证该次重放一致，不能证明全部算法正确性。

每seed上限64组、256候选、64更新、1800秒；统一评价最多1800秒，总GPU上限7200秒。本次实际三次训练分别225.076、241.538、227.117秒，评价214.660秒，GPU阶段含加载和进程开销共 **952.772秒**。未执行独立探针、重跑、调参或第5遍。冻结代码、输入、软件版本及生成配置见 [config-frozen.json](config-frozen.json)；AutoDL为单张4090 D 24GB。

## 完整统一评价

同一P0、tokenizer、greedy、thinking=false及终止规则下，8个模型各评价训练、Goal J开发、Goal G回归开发三组16题，合计384条。表格数值为严格EM/16；括号内为F1。两个开发集均已经暴露，训练集仅表示拟合。

| 模型 | 训练16题 | 主要开发16题 | 回归开发16题 |
|---|---:|---:|---:|
| v118 | 4 (.3800) | 9 (.6094) | 3 (.3497) |
| Goal J SFT | 16 (1.0000) | 10 (.7813) | 9 (.7125) |
| seed 42，第1遍 | 6 (.5556) | 10 (.7708) | 3 (.3914) |
| seed 42，第4遍 | 8 (.6806) | 11 (.8177) | 4 (.4872) |
| seed 43，第1遍 | 6 (.5050) | 11 (.7796) | 3 (.3735) |
| seed 43，第4遍 | 6 (.5913) | 11 (.8073) | 3 (.3978) |
| seed 44，第1遍 | 6 (.5868) | 11 (.8073) | 4 (.4649) |
| seed 44，第4遍 | 7 (.6493) | 10 (.7865) | 5 (.5310) |

所有六个RL checkpoint在三组评价均输出合法answer；v118在训练、主要开发、回归开发分别有3、3、2条search。因此相对v118的净增包含回答时机变化，不能直接归因为证据属性识别改善。Goal J在训练与回归开发明显高于本轮RL；主要开发的RL长预算三seed分别为11、11、10，与Goal J的10相比并不构成稳定RL优越证据。

## 训练过程与配对变化

| seed | 各遍有效更新 | 总更新/零方差组 | 训练EM 第1→4遍 | 主要开发EM 第1→4遍 | 回归开发EM 第1→4遍 | 峰值reserved GiB |
|---|---|---:|---:|---:|---:|---:|
| 42 | 11, 7, 9, 6 | 33/31 | 6→8 | 10→11 | 3→4 | 7.359 |
| 43 | 11, 8, 11, 7 | 37/27 | 6→6 | 11→11 | 3→3 | 7.258 |
| 44 | 12, 8, 9, 6 | 35/29 | 6→7 | 11→10 | 4→5 | 7.238 |

三seed第4遍较第1遍的主要开发差值为 **+1、0、−1**，均值0，范围−1至+1题；回归开发差值为+1、0、+1，均值+0.67题。训练拟合差值为+2、0、+1，均值+1题。四遍主要开发均值10.67/16、范围10至11；回归均值4/16、范围3至5。相对v118的均值增量分别为+1.67及+1题，但没有独立确认集，不能解释为统计显著或独立泛化。逐遍采样平均奖励曲线、有效更新与更新组KL见 [training-curves.png](training-curves.png)，源数据见各seed的`groups.jsonl`、`passes.jsonl`及[offline-summary.json](offline-summary.json)。采样奖励是训练中随策略变化的观测量，不是冻结评价指标。

三seed均完成64组/256候选，共192组/768候选/105次更新。每组梯度有限；按seed的最大裁剪前梯度范数为20.198、42.824、11.803。每遍平均更新组KL见离线汇总；seed42第1至4遍为.0092/.0122/.0257/.0400，seed43为.0043/.0049/.0144/.0249，seed44为.0195/.0117/.0226/.0388。采样最长action为72、90、63 token，最大prompt均505 token，未触发长度门槛。六个checkpoint均有LoRA变化，base/reference哈希不变，独立重载后固定动作logprob最大差为0。GPU显存峰值低于21 GiB预设上限。

## 逐题新增与丢失

第1→4遍的主要开发变化：seed42新增`5adde5005542990dbb2f7eee`（`Major League Baseball`→`baseball`），seed43无严格EM变化，seed44丢失`5a8673585542991e77181662`（`documentary film`→`documentary`）。回归开发seed42与seed44均新增`g014`（`documentary film series`→`documentary film`），seed43无变化。

相对v118，三个长预算模型的主要开发都新增两条由search转为正确answer的题：`5ac00c72554299012d1db589`和`5ac259bc5542992f1f2b386a`。seed42额外新增`5adde5005542990dbb2f7eee`，但丢失`5ac169c25542994ab5c67d1a`（`film director`→`director`）；seed43无额外得失；seed44丢失后者。回归集三个长预算模型都在`g001`由search转正确answer；seed42/44还在`g014`新增，seed42/43丢失`g009`（`Democratic Party`→`Democrat`）。这些词粒度差异按冻结严格EM计分；`g009`历史争议未在本轮重新裁决。所有模型、集合、题ID的完整得失及前后预测见[offline-summary.json](offline-summary.json)，逐条原始输出见[evaluation.jsonl](evaluation.jsonl)。

## 预定决策与范围

| 扩数据工程门槛 | 结果 |
|---|---|
| 三seed训练与重载均通过 | 通过 |
| 四遍相对一遍，主要开发三个seed均不下降 | **未通过**：seed44下降1题 |
| 主要开发至少两个seed上升 | **未通过**：仅seed42上升 |
| 回归开发EM均值不下降 | 通过：3.33→4 |
| 四遍均值在两个开发集都高于v118 | 通过：主要10.67>9，回归4>3 |

因此本轮只证明固定数据和预算下的工程可训练性及有限的拟合提升；更多遍数没有稳定提高主要开发严格EM。按预先约定结束预算扩展，保留所有结果并完善项目展示。此决策不访问封存数据，不启动扩训练数据Goal，也不追加第5遍。要研究真实自主检索，须另设环境和评价；本轮只包含冻结证据的合成两轮search/tool上下文。

## 复现与核查

远端路径`/root/autodl-tmp/mini-agent-rl`，实例连接`ssh -p 44249 root@connect.cqa1.seetacloud.com`。以下`prepare`/`run`仅用于**新隔离目录**的复现；本次运行目录拒绝覆盖。原始运行器和结果保持原位。`summarize`不加载模型；离线审计重算384条评分并生成训练图。

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_o_budget.py prepare
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_o_budget.py run
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_o_budget.py summarize
```

```powershell
python scripts/analyze_v133_goal_o_budget.py
python -m pytest tests/test_v133_goal_o_budget.py -q
```

注：远端PEFT对旧adapter中的`kasa_config`给出兼容警告，保存时还提示旧Windows基座路径；实际加载使用显式Linux基座，六次重载检查通过。训练实现沿用Goal N的温度采样/原始logprob约定，后续若研究更严格的策略目标，需要单独评估这一偏差。
