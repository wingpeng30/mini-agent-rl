# Goal AA：Goal Z 评分复核、冻结确认与产品选模

## 当前结论

Goal Z 的原报告、原盲审和原开发决策保留不变。本目录按原规则重新审核了全部 **720 条**已有产品输出，依据最终回答前的实际输入 token 去重为 **334 组**，由两名模型身份隐藏的辅助 reviewer 分别审核，再由主 agent 裁定分歧与共同误判。代理辅助审核不是独立人工金标准。修订产生 **136 条输出级分类变更**；逐条旧值、新值、理由见 `development-corrections.jsonl`，全部输出及证据引用见 `development-revised-results.jsonl`。

修订后，**完整轨迹 SFT 的两个 seed 都通过 Goal Z 原开发门槛**，答案 SFT 没有通过。用户授权从一次启动解释器错误中受控恢复后，冻结确认集也通过原门槛：起点 **12/64**，完整轨迹 seed 42 **35/64**、seed 43 **36/64**。两个 seed 的事实错误、证据不支持及无效动作均未高于起点。按预设规则，**发布 Goal Z 完整轨迹 SFT seed 42 adapter** 为本轮唯一产品推荐；seed 43 只用于复现验证。这是在有限封闭语料和代理辅助审核下的新题收益，不是开放域泛化或独立人工金标准。

## 开发评分与门槛

数字为修订后的“回答正确、信息充分、且最终输入中的可见证据支持”；原严格 EM 未改。旧题的 16 道主要开发与 16 道回归开发分别列出。

| 模型 | 新开发 /64 | 旧主要 /16 | 旧回归 /16 | Goal Y /48 | 新开发事实错 | 新开发证据不支持 | 原门槛 |
|---|---:|---:|---:|---:|---:|---:|---|
| Goal T 起点 | 10 | 14 | 12 | 40 | 27 | 27 | 基准 |
| 答案 SFT 42 | 9 | 14 | 12 | 38 | 27 | 28 | 未过 |
| 答案 SFT 43 | 9 | 14 | 12 | 39 | 27 | 28 | 未过 |
| 完整轨迹 SFT 42 | 36 | 15 | 12 | 40 | 10 | 11 | 通过 |
| 完整轨迹 SFT 43 | 32 | 14 | 12 | 40 | 11 | 19 | 通过 |

两条完整轨迹路线的新开发分别较同口径起点增加 26、22 题，事实错误和证据不支持均未增加，无效动作均为 0；三个旧题分组均未下降。两 seed 独立重载、权重和输入验收沿用 Goal Z 原记录并再次核验 checkpoint 文件哈希。原排序选中完整轨迹 SFT。这里的旧题语义数与 Goal Z 原报告不同，因为此次同时复核了**起点和全部候选**，不能把 14→12 等变化解读为新训练结果。原始 EM 仍可在逐条结果表中复算。

此前 Renzi/Delrio 国籍一题：完整轨迹 seed 42 的最终实际输入同时包含双方的 Italian 证据，答案 `Italian` 应判正确；原判“证据不支持”是漏读。四道原称 Goal Y 回归的题已逐步重查（`goal-y-four-case-diagnosis.json`）：两题多搜后追随无关结果而答错，一题在双侧证据可见时答出不受支持的共同类型，Renzi 一题是评分错误。四题最终输入分别为 1160、1114、1253、1130 token，没有实际截断；因此“追加搜索导致证据丢失”不能作为这四题的统一解释。

## 受控恢复与冻结确认

开发复核通过后，首次确认调用误用了 `/root/miniconda3/bin/python`（PEFT `0.21.0`），与冻结的 `0.20.0` 不符，检查在读取确认题、加载模型前停止。仓库 `.venv/bin/python` 的 Python `3.12.3`、PyTorch `2.5.1+cu124`、Transformers `5.17.0`、PEFT `0.20.0`、CUDA `12.4` 与 RTX 4090 D 均匹配冻结配置。用户随后明确授权恢复。`resume-confirmation` 在验证原状态、零确认输出、资产与软件哈希及唯一选定路线后，记录 `confirmation-recovery.json`，只执行一次起点和完整轨迹 seed 42、43 的 `3 × 64 = 192` 条 greedy 真实搜索。未新增训练或模型候选。

192 条原始轨迹完整落盘，按实际最终输入去重为 150 组匿名审核。两名辅助 reviewer 各审全部 150 组，主 agent 裁定 55 个分类分歧，并复核双方共同漏判的关系：首次演唱《You Belong to Me》的 Joni James 对应 Sharp Records；《The Hero of Color City》作曲者 Zoë Poledouris 的父亲是 Basil Poledouris。另一个双方误判的答案把 CeCe Winans 的 Grammy 错接给 Meghan Trainor。Hampartsoum Limondjian 一题的可见段落明确是 **18 世纪土耳其**音乐体系，与题问的 **14 世纪巴尔干**体系冲突，分类为事实错误而非“仅缺证据”。这些裁定及逐条依据均保存在本目录。代理辅助审核仍不是独立人工金标准。

| 模型 | 语义成功 /64 | 严格 EM /64 | 事实错 | 证据不支持 | 无效动作 | 平均搜索次数 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Goal T 起点 | 12 | 9 | 40 | 9 | 0 | 1.05 | 1.66 秒 |
| 完整轨迹 seed 42 | 35 | 26 | 15 | 9 | 0 | 2.03 | 2.72 秒 |
| 完整轨迹 seed 43 | 36 | 25 | 16 | 8 | 0 | 1.98 | 2.67 秒 |

seed 42 相对起点新增 26 题、丢失 3 题；seed 43 新增 26 题、丢失 2 题。所有 35/36 道成功题均至少搜索两次，其中两个 seed 各有 31 题的第二侧支持文档由后续搜索首次取到；起点对应为 2 题。这是连续检索确实参与成功的证据。代价是平均搜索次数和延迟上升约一倍。逐题结果、证据、检索步骤及新增/丢失见 `confirmation-revised-results.jsonl` 和 `confirmation-paired-changes.json`。

两个 seed 的确认成功数均比同环境起点增加至少 4，事实错误、证据不支持、无效动作均不增加。`product-decision.json` 因而发布预先指定的 **seed 42**，没有按 seed 43 的较高确认分数挑选模型。确认题此后已暴露，本轮不再用它们调参或选择其他路线。

产品入口 `scripts/demo_v133_goal_z_selected.py` 已改为优先读取 Goal AA 的发布决定，并加载 seed 42 adapter。发布后的唯一演示使用 Delrio/Renzi 问题，模型正常加载并回答 `Italian`；但该问题不在产品入口默认的旧 96 篇语料中，两次检索均为空，因此**这条演示只验证入口与 adapter 接线，不计有证据的搜索成功**。确认结论来自冻结的独立 2048 篇确认语料和完整 192 条轨迹。产品入口的检索库覆盖范围仍由其配置决定。

## 审核与复现记录

- `config.json`：Goal Z 原配置、原数据、实际模型文件及四个 checkpoint 的 SHA-256，原评分规则和门槛；确认文件仅哈希核查。
- `development-packet.jsonl` 与 `development-map.json`：实际输入 token 解码的匿名审核包和 720 条输出映射。
- `development-review-a.jsonl`、`development-review-b.jsonl`：两份 334 组独立审核；`development-adjudication.jsonl`：56 个分类分歧的主 agent 裁定。
- `development-main-overrides.jsonl`：双方一致却误读了具体关系的三处显式复核。它们保留原始 reviewer 意见；`code-amendment.json` 记录为此增加覆盖接口后的运行器哈希。
- `development-revised-results.jsonl`：720 条旧/新类别、EM、F1、证据 ID 和理由；`development-corrections.jsonl`：136 条发生改变的输出。
- `development-decision.json`：按原门槛重算的各 seed 分数、完整错误类别和唯一选路。
- `confirmation-recovery.json`：用户授权后的受控恢复前状态与环境；`eval-confirm-*.jsonl`：全部 192 条原始轨迹。
- `confirmation-packet.jsonl`、两份 `confirmation-review-*.jsonl`、`confirmation-adjudication.jsonl` 与 `confirmation-main-overrides.jsonl`：匿名审核和主审裁定；初版 reviewer B 意见单独保存。
- `confirmation-revised-results.jsonl`、`confirmation-paired-changes.json`、`confirmation-diagnostics.json`：逐题评分、配对变化和搜索/证据指标。
- `product-decision.json`：冻结确认门槛与唯一发布结果；`state.json`：完成状态。

执行入口（远程仓库根目录）：

```bash
.venv/bin/python scripts/run_v133_goal_aa_audit_confirmation.py summarize
```

`prepare` 拒绝覆盖，`run` 按状态推进；`resume-confirmation` 仅适用于此次无输出的启动失败，已使用一次。产品演示命令见 README。Goal Z 原报告与原决策保持原样，Goal AA 的发布决定作为独立更正记录。
