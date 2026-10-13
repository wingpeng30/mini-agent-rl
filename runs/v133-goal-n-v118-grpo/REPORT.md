# Goal N：v118 起点固定证据 LoRA-GRPO

状态：`COMPLETED_EXPERIMENT`。单 seed、一次探针、一次正式训练、最终 checkpoint 和同期三模型评价均完成。不是端到端检索策略训练；两轮 search/tool 历史是按冻结证据合成的固定上下文。

## 结论

从相同 v118 SFT adapter 出发，本次 GRPO 完成 **9 次有效 LoRA 更新**。相对 v118，冻结训练16题严格 EM 4→6，Goal J 开发16题9→11，Goal G回归开发16题3→5。Goal J 追加 SFT 的对应结果是16/16、10/16、9/16，因此两条分支的强弱随集合而异。Goal J 开发集上的 9→11 来自已参与多轮决策的16题，属于探索信号，不能称独立泛化或大样本稳定提升。训练方法与预算不同，不能单凭本表断言 GRPO 优于 SFT。

## 同条件结果

三套adapter在当前同一AutoDL实例、原P0、greedy、thinking=false、128新token上各评价三组冻结题，每格16题。严格EM沿用项目冻结的归一化规则，F1保持原规则。

| 集合与身份 | v118 EM / F1 / answer-search-invalid | Goal J SFT | Goal N GRPO |
|---|---|---|---|
| 训练16题；拟合 | 4/16 / 0.3800 / 13-3-0 | 16/16 / 1.0000 / 16-0-0 | 6/16 / 0.5556 / 15-1-0 |
| Goal J评价16题；已暴露开发集 | 9/16 / 0.6094 / 13-3-0 | 10/16 / 0.7813 / 16-0-0 | 11/16 / 0.8073 / 16-0-0 |
| Goal G回归16题；已暴露开发集 | 3/16 / 0.3497 / 14-2-0 | 9/16 / 0.7125 / 16-0-0 | 5/16 / 0.4703 / 16-0-0 |

相对v118，RL在训练/Goal J开发/Goal G开发集分别新增严格正确2/2/2题，丢失0/0/0题。主要开发集新增是 `5ac00c72554299012d1db589` 与 `5ac259bc5542992f1f2b386a`，两题的v118输出均为合法search，RL转为严格正确answer；因此不能把主要开发净增2题单独归因于共同属性选择能力。相对Goal J，RL主要开发集新增2题、丢失1题；回归集新增3题、丢失7题。逐题ID见 `offline-summary.json`，原始动作见 `evaluation.jsonl`。g009等历史争议仍按冻结严格指标计分，语义审核未在本轮补齐。

## 训练前门槛与更新

N0复核训练、两个开发集和v118/Goal J adapter的SHA256均匹配冻结清单，实际GPU为4090 D 24GB，软件为torch 2.5.1+cu124、transformers 5.17.0、PEFT 0.20.0、tokenizers 0.23.2。N1当前tokenizer对48条完整前缀/标签检查通过；新增动作解析与EM/F1对Goal J历史96条逐条复算一致。配置、代码与输入哈希见 `config-frozen.json`。

N2预冻采样为训练16题×4、seed42、temperature0.7、top-p1、top-k0、最大128新token，内容奖励仅合法answer的 `EM+0.25×F1`。64条输出中39 answer、22 search、3 invalid；12/16组奖励不同，6组在有效answer间奖励不同，2组同时含严格正确/错误的有效answer，达到预先规定的4组和2组门槛。探针只检验工程可训练性，不是显著性检验，也不复用其logprob做更新。

N3在线重新采样16组、每组4候选，9组奖励不同并各完成1次optimizer更新，7组零方差跳过；9组中8组有有效answer之间的奖励差异，2组含严格正确/错误有效answer。AdamW lr 5e-6、betas(0.9,0.999)、eps1e-8、weight_decay0；policy_epochs1、clip0.2、KL beta0.02、梯度裁剪1.0。old/current/reference对相同prompt和完整action token重算；reference为v118 LoRA快照，prompt与tool token不进入policy loss。首次ratio最大1.0；更新前梯度有限，最大裁剪前范数11.308。192个LoRA参数发生变化，base和reference哈希不变。显存峰值reserved 7.189 GiB、allocated 6.867 GiB、设备进程7.607 GiB，低于21 GiB上限。最终adapter独立重载后固定动作logprob最大差0。

最终adapter SHA256：`b7202f08f64dbd7d6ddd5db3cfb38f937110c6aac26583f60cf75fdc0864b61a`。探针、训练、统一评价耗GPU执行时间218.754秒；三模型实时演示后累计225.774秒，低于7200秒总上限。N3训练64.427秒，N4评价83.866秒。没有按开发表现选checkpoint或调整奖励、seed和预算。

## 复现与演示

AutoDL当前有效地址：`ssh -p 44249 root@connect.cqa1.seetacloud.com`；目录 `/root/autodl-tmp/mini-agent-rl`。`prepare`创建配置和CPU检查；`run`依次执行一次probe、条件训练和统一评价，已有运行目录拒绝重跑或覆盖。命令用于新隔离复现目录时必须先保存本次原始结果；**不要对本次输出直接重跑**。

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_n_v118_grpo.py prepare
timeout -k 10s 7200s env PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_n_v118_grpo.py run --config runs/v133-goal-n-v118-grpo/config-frozen.json
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_n_v118_grpo.py summarize
```

无需GPU的三模型冻结输出回放：

```bash
python scripts/demo_v133_fixed_evidence_replay.py --split new_eval --case-id 5ac00c72554299012d1db589
```

AutoDL GPU上的三模型实时生成演示：

```bash
PYTHONPATH=src .venv/bin/python scripts/demo_v133_fixed_evidence_live.py --split new_eval --case-id 5ac00c72554299012d1db589
```

该例展示相同362-token上下文下：v118发出search，Goal J与Goal N均发出`novelist`最终answer。回放使用历史冻结输出；实时入口实际加载三套adapter生成，示例结果见`demo-live-example.json`。两者都明确展示合成历史的范围。

## 限制与工程事实

本轮只有一个seed、16条训练题与两组已经暴露的16题开发集，未使用独立确认集。主要开发2题新增主要体现回答时机变化；不能把它写成属性推理能力的独立证明。Goal J相对v118的SFT与本次GRPO训练方式/预算不同，比较是结果对照，不是等算力实验。若要声称自主检索能力提升，需另行定义含真实检索环境与多步奖励的实验。

加载v118时PEFT 0.20.0提示忽略adapter配置中的`kasa_config`，与历史运行环境一致；实际adapter激活、LoRA变化及重载均通过，未升级依赖。保存adapter时出现旧配置中的Windows基座路径警告；本次实际加载使用明确的Linux本地基座，重载logprob核查通过。复现仍需显式传入当前基座路径。

一次运行的状态、逐条探针、在线rollout、组指标、逐题评价、CPU检查和分数复算均在本目录。未访问final-test、封存独立100题或新数据来源；未调用外部API逐条评分。
