# v133 Goal C：CPU 数据与小规模对照准备

状态：数据已冻结，运行方案已准备；训练和模型推理未获授权、未启动。本交付不代表实验已完成或有效果。

## 可证伪假设

依据同期 A2/A3 的6个新增/丢失正确差异，探索“当作答时可见证据已经明确包含题目所问属性/数值，模型能否更精确地抽取该值，而不是输出相关实体或部分值”。这只是一个可测试的答案抽取假设，样本量小，不预测端到端搜索收益。

例证包括：证据含“21 August 1765”时A2只答“1765”；证据同时列出两部电影导演时A2选了邻近但无关的导演；以及Valve/Valve Corporation表述变体可能受严格EM影响。A3新增正确中只有一个案例明确利用第三次搜索，因此不把搜索预算效果误归为该抽取假设。

## 有限来源筛查记录

从固定的90,447条官方 HotpotQA distractor train 行号中，确定性抽样300条。42条因历史暴露索引命中而跳过，只打开258道题；252条通过源记录支持标题/句号解析、答案在所选支持句中出现、未命中暴露索引等检查，按冻结SHA顺序构造64条候选。未扩池。

原始64候选及审阅包已做来源重开、token与匿名字段审计。其48条新候选未送审，审阅包已作废。随后发现与假设直接匹配的既有 v122 answer-state 样本：25条各自均有有效 DeepSeek 批准、隔离 Luna 批准、主账本保留、严格配对关联，且目标答案逐条出现在可见证据中。复用这些25条作训练，不以旧的395状态/211题/100配对总量门槛作准入条件，也不宣称旧门槛已经通过。

冻结结果：

- 训练：25条既有有效双审记录；保留原始输入与目标。
- 开发：16条从本次预先固定的官方训练集候选中选定的证据充分单动作样例。官方答案与支持句可追溯；这16条未双审，只用于开发评价，不得训练。
- 训练/开发题目和来源ID交集均为0。
- 64候选中的其余32条训练候选和16条reserve不使用。无有效双审、争议、否决或单审标签进入训练。
- v118 tokenizer实际输入长度：训练记录已用冻结processor复核；新候选的prompt为225–364 tokens、目标最多37、完整最多377。SFT `max_length=2176`。
- 新来源筛查/数据构造没有调用模型或审核API；没有访问封存独立100题或final-test。

冻结文件与逐条索引：

- 训练：[train-candidate.jsonl](../data/evidence-sft-v133-goal-c-private/frozen/train-candidate.jsonl)，25条，SHA256 `b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df`
- 开发：[dev-candidate.jsonl](../data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl)，16条，SHA256 `23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`
- 训练票据/来源索引：[train-review-index-private.jsonl](../data/evidence-sft-v133-goal-c-private/frozen/train-review-index-private.jsonl)
- 开发来源索引：[dev-source-index-private.jsonl](../data/evidence-sft-v133-goal-c-private/frozen/dev-source-index-private.jsonl)
- 完整冻结清单：[freeze-manifest-private.json](../data/evidence-sft-v133-goal-c-private/frozen/freeze-manifest-private.json)
- 原始有限筛查及独立审计：[manifest-private.json](../data/evidence-sft-v133-goal-c-private/manifest-private.json)、[audit-private.json](../data/evidence-sft-v133-goal-c-private/audit-private.json)

索引含私有源ID和审票关联，不要向盲审者或外部服务发送。独立候选数据审计通过；但开发集是官方训练拆分中的16条小样本，不能称为独立测试集。

## 冻结的一次性SFT计划

配置：[v133-goal-c-small-sft.json](../configs/v133-goal-c-small-sft.json)。唯一训练变化是基于固定 v118 adapter 做LoRA SFT；seed 42、1 epoch上限、7 optimizer steps、batch 1、梯度累积4、LR `5e-6`、warmup 1/7、BF16、max length 2176、无梯度检查点。数据共25 train/16 dev。禁止自动重试、换seed、提示调参或扩池。

需要可用的CUDA GPU，显存至少24 GiB，单次训练硬上限60分钟。本机8 GiB RTX 4060 Laptop不满足资源预算。训练前由用户另行授权；本次没有运行训练。

训练命令（PowerShell；只有你单独明确授权训练后，才可在仓库根目录执行）：

```powershell
& 'D:\anaconda3\envs\mini-agent-rl\python.exe' scripts/run_v133_goal_c_sft_once.py --authorize-training
```

该单次runner在启动子进程前校验配置/数据/基线权重哈希、CUDA、24 GiB显存和BF16，拒绝覆盖既有输出，并在60分钟杀停；步数异常不重试。

AutoDL执行时，须用CR0确认的Linux模型目录覆盖Windows默认位置：

```bash
/root/autodl-tmp/mini-agent-rl/.venv/bin/python scripts/run_v133_goal_c_sft_once.py \
  --authorize-training --model-path /root/autodl-tmp/qwen_08b
```

训练结束后，只有在用户再明确授权模型推理时，才在同一16条冻结证据输入上分别运行v118与SFT adapter。评估器：[evaluate_v133_goal_c_sft.py](../scripts/evaluate_v133_goal_c_sft.py)会核验adapter/data/config/训练指标和24 GiB显存门槛；[一次性限时runner](../scripts/run_v133_goal_c_eval_once.py)执行60分钟硬停止并拒绝覆盖或重跑。它不会搜索，也不能用历史29/50替代同期比较。授权后的调用形状：

```powershell
& 'D:\anaconda3\envs\mini-agent-rl\python.exe' scripts/run_v133_goal_c_eval_once.py `
  --model-path D:/qwen_08b `
  --base-adapter backups/v118/extracted/runs/v118/formal/adapter `
  --sft-adapter runs/v133-goal-c-sft-once/adapter `
  --dev-data data/evidence-sft-v133-goal-c-private/frozen/dev-candidate.jsonl `
  --training-report runs/v133-goal-c-sft-once/adapter/training_metrics.json `
  --output runs/v133-goal-c-sft-once/paired-dev-results.json `
  --max-minutes 60 --authorize-model-inference
```

AutoDL评估命令把解释器和模型目录改为：`/root/autodl-tmp/mini-agent-rl/.venv/bin/python scripts/run_v133_goal_c_eval_once.py --model-path /root/autodl-tmp/qwen_08b`，其余相对路径仍从 `/root/autodl-tmp/mini-agent-rl` 仓库根目录解析。

评估记录SQuAD归一化EM（大小写、标点和冠词归一）、token F1、协议有效性、新增正确、丢失正确、正确保持与逐题变更。若结果至少+2/16 EM，再由独立匿名审核检查变更答案的证据支持后，才可讨论是否继续；未达门槛或有确认支持回退则停止。该小开发集只能给出探索性继续/停止信号，不支撑泛化结论。一次性SFT及其后模型推理均需分别授权。
