# Goal L：共同属性能力与回答策略对照

## 结论

提示变化没有提升Goal J adapter的严格成绩或修正其错误。新评价16题中，Goal J在P0、P1、P2下均为10/16 EM、0.7813 F1、16条有效answer；Goal G回归集三条件均为9/16、0.7125 F1，逐题答案也相同。P0到P1仅有一道新评价题的答案从“drummer”变成“band”，两者都错；P1到P2没有改变答案。当前没有证据支持继续调提示。

v118对提示更敏感，而且P1结果变差：新评价严格EM由P0的9/16降到P1的6/16，P2回到7/16；有效answer为13、9、11，合法search为3、5、4，坏/不合协议输出为0、2、1。Goal G上v118是3/16、0/16、1/16。模型多次忽略P1“不要search”的明确要求，转而search或输出无效内容。因此，本对照显示Goal J adapter已经稳定采取回答行为，但不能把P1/P2当成成功强制回答的干预，也不能仅凭这种条件把所有模型差异归因为共同属性判断能力。

在主要新评价集的同条件模型比较中，Goal J分别比v118多4题(P1)和3题(P2)严格正确，F1分别高0.3646和0.2552，且没有严格丢分。两方盲审对入选的4个独立新增答案（novelist、songwriter、singer、baseball）均批准；其中P1/P2重复的同一题按各自真实可见指令分别审核。Goal J与v118的P0比较则只有2条新增正确、1条丢分，净增1题，沿用Goal K的主指标结果。上述成绩来自已参与研究决策的16题，属于探索证据，不是独立确认。

严格丢分“director”对应的官方答案更具体，为“film director”；严格EM仍按冻结规则记错。新审核在P1完整上下文下，DeepSeek和Luna均认为“director”得到双侧证据支持、足以回答问题。旧P0条件有一张Goal K Luna批准票，Goal K DeepSeek请求结果仍未知；本报告不把P1双审冒称为P0完整双审。Goal G的g009在本轮8条审核预算之外，保持未决；历史Luna曾否决“Democrat”，Goal K DeepSeek票缺失。

## 逐条件指标

| 集合 / 模型 | 提示 | 严格EM | F1 | 有效answer | 合法search | 坏/无效输出 |
|---|---:|---:|---:|---:|---:|---:|
| 新评价 / v118 | P0 | 9/16 | 0.6094 | 13 | 3 | 0 |
| 新评价 / v118 | P1 | 6/16 | 0.4167 | 9 | 5 | 2 |
| 新评价 / v118 | P2 | 7/16 | 0.5260 | 11 | 4 | 1 |
| 新评价 / Goal J | P0 | 10/16 | 0.7813 | 16 | 0 | 0 |
| 新评价 / Goal J | P1 | 10/16 | 0.7813 | 16 | 0 | 0 |
| 新评价 / Goal J | P2 | 10/16 | 0.7813 | 16 | 0 | 0 |
| Goal G回归 / v118 | P0 | 3/16 | 0.3497 | 14 | 2 | 0 |
| Goal G回归 / v118 | P1 | 0/16 | 0.0495 | 8 | 6 | 2 |
| Goal G回归 / v118 | P2 | 1/16 | 0.1620 | 9 | 5 | 2 |
| Goal G回归 / Goal J | P0/P1/P2 | 9/16 | 0.7125 | 16 | 0 | 0 |

合法search与坏JSON分开统计。完整逐题六条件表在 `paired-differences.jsonl`；逐条原始输出在 `trajectories-new.jsonl`，历史P0在 `P0-history.jsonl`。

## 运行、输入与审核

- AutoDL为NVIDIA GeForce RTX 4090 D；128条新轨迹于约93秒完成。两套adapter均激活96个LoRA模块，BF16，adapter及输入哈希与冻结清单一致。
- 复用64条Goal J历史P0输出；新生成数量为128。未训练、未提取新题、未读取final-test或封存100题。
- PEFT加载日志提示配置中的`kasa_config`参数被当前已冻结的PEFT 0.20.0忽略；相同版本、adapter和加载路径与Goal J历史运行一致，运行器确认adapter激活及模块数为96。记录该兼容警告，不据此单独推断影响大小。
- DeepSeek批次`goal-l-answer-policy-01`和全新无历史GPT-6 Luna各审核8条；16张票均通过v130机械校验，且均批准。Luna所读匿名包SHA256与冻结文件一致。绑定和逐文引用见 `review-bindings.json`；匿名包、私有映射和DeepSeek请求日志分开保存。
- 固定8条优先级先覆盖主要评价集丢分和新增正确，因此Goal G g009未被本轮双审覆盖，不能把这轮报告理解为该题已解决。

## 下一步建议

不要继续对Goal J adapter追加提示变体，也不要立刻用相同16题追加SFT。优先准备一次独立、来源和实体隔离的共同属性评价，并采用单一、无冲突的生成协议，对v118和Goal J作同期原提示比较；在此之前把当前结果称为有支持的探索性迁移信号。若独立评价复现多题优势，再决定是否为缺失属性维度补训练案例。当前结果说明模型更新和回答时机/粒度都参与了表现，但现有已暴露小样本不足以将二者的贡献完全分开。

## 复现与哈希

```bash
cd /root/autodl-tmp/mini-agent-rl
.venv/bin/python scripts/run_v133_goal_l_answer_policy.py summarize
```

冻结配置、提示、输入哈希和CPU检查见 `freeze-manifest.json`、`runtime-config.json` 与 `cpu-checks.jsonl`。新增轨迹SHA256：`f920d26c5bb8c82821ec45879879cc04ee5372b40403c6b2d5d65f665693f8f6`。逐题对照SHA256：`0a93fbb4d28a892cbaf14e0dee6afdcaa249a1360c2280c383b1c4c8d516221a`。
