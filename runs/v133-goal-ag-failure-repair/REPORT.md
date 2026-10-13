# Goal AG：真实失败修正训练与新题确认

## 决策

**不发布新 adapter；当前产品仍为 Goal Z 完整轨迹 SFT seed 42。** 修正轨迹路线 B 在开发集通过预设门槛，但一次性新确认集没有达到发布条件：同口径起点为 30/64 语义成功，B42 为 29/64（-1），B43 为 32/64（+2）；要求两个 seed 都至少增加 4 题。B42、B43 均未通过，因此不按 seed 43 的单次更高分切换模型，也不追加训练。

确认集上，起点、B42、B43 的事实错误加证据不支持分别为 31、32、29。B42 比起点增加 1 题，未通过错误不增加条件；无效动作均为 0。核对原冻结支持映射、每步返回及终审可见证据后，起点连续搜索保护组为 21 题；B42 丢失 1 题，B43 丢失 0 题，均满足最多丢失 1 题的条件。两 seed 仍因成功增益不足而不能发布。

## 本轮结果

起点为已发布的 Goal Z 完整轨迹 seed 42。两臂各 seed 42、43，四次训练各 128 次更新，使用冻结的 16 条真实修正轨迹；权重完整性与独立重载验收通过。开发集上只有修正路线 B 通过：起点 34/64，B42 39/64，B43 38/64；开发错误合计由 23 降至两 seed 各 21。普通继续训练 A 为 37/64、37/64，未通过预设增益门槛。完整开发逐题账本及历史保护组结果保存在本目录的冻结文件中。

| 模型 | 开发成功 /64 | 旧主要 /16 | 旧回归 /16 | Goal Y /48 | AA 回归 /64 |
|---|---:|---:|---:|---:|---:|
| 起点 | 34 | 13 | 8 | 34 | 27 |
| A42 | 37 | 14 | 11 | 44 | 43 |
| A43 | 37 | 14 | 11 | 44 | 45 |
| B42 | 39 | 15 | 11 | 45 | 42 |
| B43 | 38 | 15 | 11 | 45 | 44 |

实际 B 臂保留原轨迹 6479 个监督 token，另加入 490 个修正监督 token，包含 12 个修正 search 和 16 个修正 answer 动作。人读前提曾误写“修正答案不作为新增答案监督”，已在收尾核查中更正并保留原稿；实际代码与材料执行的是 Goal AG 预设的完整修正轨迹监督。四个日志均为连续的 1—128 次更新，loss 和梯度全部有限；每个 checkpoint 有 192 个 LoRA 张量发生变化，base 不变，独立重载差值为 0。指标汇总见 `training-verification.json`。

![四次训练的损失与梯度](training-curves.png)

冻结确认集共评价起点、B42、B43 三个模型，各 64 题，共 192 条真实搜索轨迹。匿名审阅覆盖 109 个唯一“问题—最终可见证据—答案”组合；审核 A 覆盖 55 条、审核 B 覆盖 54 条。主审复核每个被标为成功的关系链和引用，并对 6 个标签作了版本化修正。最终 109 个组合分类为：53 正确充分、11 事实错误、40 实际可见证据不支持、5 缺少必要信息；无无法判定项。审阅原件保留在 `confirmation-review-a.jsonl`、`confirmation-review-b.jsonl`，主审更正见 `confirmation-review-adjudication-ledger.jsonl`。

| 模型 | 正确充分 | 事实错误 | 证据不支持 | 两类错误合计 | 缺必要信息 | 严格 EM | 平均 F1 | 平均搜索 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Goal Z seed42 起点 | 30/64 | 7 | 24 | 31 | 3 | 21/64 | 0.471 | 2.05 |
| Goal AG B seed42 | 29/64 | 6 | 26 | 32 | 3 | 22/64 | 0.479 | 2.03 |
| Goal AG B seed43 | 32/64 | 5 | 24 | 29 | 3 | 23/64 | 0.497 | 2.00 |

两 seed 的 EM 和 F1 都略有提高，但 B42 的错误合计增加，语义成功减少；B43 的错误合计减少、语义成功增加 2 题。两 seed 均未达到强制增益门槛，方向也不一致。最稳妥的结论是：**修正监督在已暴露开发题上收益明显，但这次新题确认没有证明其带来足够稳定的产品增益。** 本轮结果不支持发布新模型，也不证明修正路径完全无效。

逐题对照进一步显示：B42 新增成功 1 题、丢失原成功 2 题，净增为 -1；B43 新增成功 2 题、丢失 0 题，净增为 2。完整 64 题三模型对照见 `confirmation-paired-results.jsonl`；保护题、逐 seed 新增和丢失 ID 均保存在 `confirmation-decision-local.json`。保护组按冻结支持页面在不同搜索步骤首次返回、且必要证据在已审核最终输入中可见构成；这是封闭数据集内的能力保护判据，不能据此推广为所有开放域查询都需要连续搜索。

## 评分复核

复核依据是模型回答当时实际输入中可见的工具正文，不以完整搜索历史替代最终上下文。审核证据 ID 与匿名包里的 `visible_documents[].document_id` 逐条核对；所有引用均对应实际可见文档。

主审调整的主要边界包括：`Capitol Records` 只被明确证明是某张专辑的发行标签，不能据此断言是艺人的签约标签；`Universal Records` 虽与专辑共同发行有关，但没有证明问题所问的签约关系；`Saint Paul` 给出了大致城市，却遗漏了证据明确指出的 Minnesota History Center 总部位置。另外统一了同一省份边界问题的审核：保护区分布在两省，不能直接代替共享边界关系；此前两个包对同类证据作出不同判定，终审同步更正。原始两位 reviewer 文件未改写，六项改判完整记录在修订账本。

严格 EM 保持原始金标和口径。语义成功仍是代理辅助审阅结果，并非独立人工金标准。该确认语料已执行评价，今后必须视为已暴露，不能再称为独立确认集。

## 复现与运行状态

- 三组原始评价：`eval-confirmation-baseline.jsonl`、`eval-confirmation-42-B.jsonl`、`eval-confirmation-43-B.jsonl`。
- 匿名包与映射：`confirmation-review-packet.jsonl`、`confirmation-review-map.jsonl`。
- 冻结终审：`confirmation-review-final.jsonl`、`confirmation-decision-local.json`。
- 最终裁定脚本：`scripts/finalize_v133_goal_ag_confirmation_review.py`。
- 训练曲线与验收：`training-curves.png`、`training-curves.svg`、`training-verification.json`。
- 原实验运行器：`scripts/run_v133_goal_ag_failure_repair.py`。

本地无需加载模型的复算命令：

```powershell
python scripts/finalize_v133_goal_ag_confirmation_review.py
python scripts/plot_v133_goal_ag_training.py
```

Goal AG CPU 验收测试共 11 项通过，脚本语法检查通过。pytest 的未知 asyncio 配置和缓存目录权限警告不影响测试结果。

2026-10-02 用户明确授权回传后，终审文件已同步至同一 AutoDL 项目，远端最终决定为 `complete_confirmation_failed`；本地与远端的分数及保护题损失一致。报告生成时发现缺失 `render_report` 函数，已修复并增加两项回归测试。全程复用已有 192 条确认输出，产品仍使用 Goal Z seed42。
