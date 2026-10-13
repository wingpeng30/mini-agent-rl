# Goal H：共同类别同期评价

## 执行与输入

- AutoDL：`ssh -p 25320 root@connect.cqa1.seetacloud.com`；RTX 4090 D；GPU阶段实际运行 44.363 秒，完成 52/52 条，未触发7200秒上限。
- 固定 `seed=42`、BF16、关闭thinking、贪心解码、最多生成128 tokens。串行加载 v118、首次SFT、Goal D adapter；每份均为激活的 `default` adapter、96个LoRA模块。PEFT恢复到历史使用的0.20.0；`torch=2.5.1+cu124`、`transformers=5.17.0`。
- 新16题直接使用Goal G冻结消息。每题在三个模型下的生成输入token哈希完全相同，并与Goal G预检哈希一致。旧4题从v118实际训练记录第46、80、129、217条构建两轮证据历史，仅作拟合诊断。两类题均是**固定合成搜索历史下的证据回答**，不是端到端检索成绩。
- 冻结主评价文件 SHA256 `f689f94b598a1c0d89f1b87f0b008c8a03158a7ab479028368285564244aad69`；H0清单 SHA256 `b160fc622362632f60f1a9b5103dc0b37ea8e63ffee4a56095330c08845232d8`；轨迹 SHA256 `df247a8b26cdca01aaee4d7d141491fed774ec9840c0e742835ee3721425edab`。完整权重与输入哈希见 `h0-manifest.json`。
- CPU门槛检查通过：输入或adapter哈希变化会拒绝；答案目标不进入生成前缀；运行器只读取显式冻结路径；单次运行禁止覆盖；7200秒由父进程硬超时控制。前两题实际平均生成2.063秒，预计总计113.879秒，满足继续条件。

## 同期结果

| 模型 | 新16题EM | 平均F1 | JSON回答协议有效 |
|---|---:|---:|---:|
| v118 | 3/16 | 0.3497 | 14/16 |
| 首次SFT | 4/16 | 0.4539 | 16/16 |
| Goal D追加SFT | 5/16 | 0.5164 | 16/16 |

相对v118：首次SFT新增正确 `g001`，追加SFT新增 `g001`、`g015`；两者均没有丢失v118已正确的 `g009`、`g012`、`g013`。这些是16题上的探索性净增，不是统计显著性证明。

`g001`：两份证据分别列出James Young和Michael Jones的职业，均直接支持“film director and screenwriter”。`g015`：两份证据均直接称Bletilla和Vernonia为“genus”。两道新增正确的输出动作与Goal G原匿名DeepSeek及隔离Luna双审的**同一答案、同一可见上下文**完全一致；两票均有效批准。绑定依据见 `review-binding.json`，本轮未重复调用审核。`g000`虽从继续搜索变为有效回答，但“voice actress”只适用于其中一人，共同职业应为“actress”，故仍计错。

旧4道训练暴露题在**改变后的两轮上下文**中，v118严格EM为0/4：`core:46`偏离问题，`core:80`答“rock”而目标为“rock band”，`core:129`未给出合规答案，`core:217`答“film director”而目标为“director”。其中有粒度接近的答案，0/4不应解释为毫无语义能力。这四条新上下文没有沿用旧审核成为新训练合格样本。原核心记录不含独立来源ID；H0清单里的 `core:<index>` 是诊断ID，不证明与新题的真实来源ID隔离。已核对题面、证据标题及显见实体无重叠；主评价的项目内隔离结论仍依据Goal G自身的祖先审计。

## 判断与下一步

现有adapter在冻结的共同类别题上出现**有限的迁移信号**：Goal D净增2题、没有确认的证据支持退步，达到预注册的探索性继续研究条件。严格成绩仍低（5/16），不能据此宣布训练配方有效。旧4题提示有可学习空间，但这些题的上下文变化使旧审核不覆盖新输入。

下一步优先提出一个独立微型拟合Goal：先审核旧4题的新上下文与答案粒度，合格后固定v118起点、一次训练和一个种子，并预留新的未用共同类别评价题。Goal G的16题已经用于本次研究决策，此后只能称探索评价，不能再称未参与设计的独立验证。**Goal H没有执行新增SFT。**

## 复现入口

在远端仓库目录 `/root/autodl-tmp/mini-agent-rl`：

```bash
.venv/bin/python scripts/check_v133_goal_h_preflight.py
.venv/bin/python scripts/run_v133_goal_h_comparison.py prepare
.venv/bin/python scripts/run_v133_goal_h_comparison.py run
.venv/bin/python scripts/run_v133_goal_h_comparison.py summarize
```

本轮 `prepare` 和 `run` 是单次冻结入口，现有目录不能覆盖重跑。运行器、CPU门槛与结果审核源码分别为 `scripts/run_v133_goal_h_comparison.py`、`scripts/check_v133_goal_h_preflight.py`、`scripts/audit_v133_goal_h_results.py`。
