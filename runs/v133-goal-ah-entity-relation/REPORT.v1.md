# Goal AH CPU准备结果

**训练材料已就绪；完整实验因新确认集仅36/64合格而停止在GPU之前，缺28题。没有训练、评价或发布任何新模型。** 产品仍是Goal Z完整轨迹seed42。

实际远端起点权重与AA发布记录核对通过。保留原128条完整轨迹、6479监督token；一次性生成87条判断候选，主审后按冻结规则选64条，四类各16，新增官方训练题0。辅助目标2274token，最大输入1888、输出含终止符58。训练设计为A普通继续SFT、B完整原损失加0.25证据判断损失，两个seed各128更新。

## 诊断边界

`diagnostic-ledger-v1.jsonl`记录已有开发64题和AG确认64题的实际调用输入；全部输入完成token解码/编码一致性核查。13题完成针对性原文语义复核，115题保留AG判断并明确标注继承，不能把本账本当作全量重新人工标注的金标准。

- 取证不足：最高收入题没有取到目标链，却使用最高收入女演员材料。
- 实体错配：Coffee Swamp的Wisconsin岛与Michigan同名岛混用；Hagar题把儿子当后娶配偶。
- 关系方向/范围：ALARM前身与后身；DSC七校运营联盟与CSU系统总数。
- 链缺一环：专辑发行不能证明艺人签约；保护区覆盖两省不能证明共享边界。
- 必要限定：出版社Press与母机构Society、总部位置精度未明确绑定。
- 充分仍答错：Darley→Pedro Leopoldo→Minas Gerais证据齐全却答Minsk。
- 歧义：A Stitch问导演，不能沿用“另一部电影不能获奖”的旧理由；个人/影片获奖归属仍保守保留原类别。

四项明确修订单独保存：Home Fries导演传记明确NYU就读，改为成功；Sir James旧理由错引Tony Parker；Hagar最终输入没有旧理由声称的Keturah段落；DSC的12是地区子集不是CSU总数。只有Home Fries改变类别。AH旧输出修订快照可得到开发35/64，**不能与原AG候选37/39等旧评分直接混比，也不是本轮同时重测的起点结果**。所有正式门槛以之后统一重测/审核的272题起点为准。

没有补齐人工证据，没有把诊断或固定上下文计入真实搜索成功。

## 新确认集门槛失败

一次性官方候选池各256，初步隔离固定选择32+32；源文审核经主审修正后40/64合格。主审认可“直接支流→汇入河流”和明示县级出生地的等价/粒度表达；也新增剔除乐队成立地当出生地、迁居地当死亡地、双导演题义不唯一等问题。

本地此前缺少实际AG完整确认语料。取回远端原冻结语料后，额外核验发现6题支持页面已曝光，4题在源文合格组内。因此同时满足源文质量和全部现有语料隔离的仅36题（MuSiQue20、2Wiki16），分别缺12、16题，总缺28。保留原筛选包及补核账本，未扩池、替换或重采。

现有候选准备包不能宣称为可发布验收的新64题确认集。`confirmation-final-cpu-gate.json`和`execution-freeze-addendum-v1.json`为最终判据；早期隔离manifest的32+32不是最终通过数。

## 交付与验证

- 人读完整前提：`PRECONDITIONS.md`。
- 诊断与修订：`diagnostic-ledger-v1.jsonl`、`adjudication-corrections-v1.jsonl`、`cpu-eval-contexts.jsonl`。
- 可审训练材料：`judgments-reviewed.jsonl`、`training-material-examples.jsonl`、`material-citation-positions.jsonl`、`judgment-training-inputs.jsonl`。
- 审核原件/主审：`material-review-helper.jsonl`、`material-review-main.jsonl`、`material-adjudication-v1.jsonl`。
- 冻结参数与评价272题：`execution-freeze.json`，最终缺口补记`execution-freeze-addendum-v1.json`；中间`training-freeze.json`保留，不作为最后GPU前提。
- 新确认质量/隔离：`sealed-confirmation-v1/`、`exposure-registry.json`、`additional-ag-corpus-isolation-audit.json`、`confirmation-final-cpu-gate.json`。
- 执行与CPU门槛：`scripts/run_v133_goal_ah_entity_relation.py`、`scripts/assess_v133_goal_ah.py`。

CPU验收12项通过，覆盖伪造引用拒绝、跨类别凑数拒绝、材料不足、评价隔离、预算/正文保真、无效动作并集、产品与机制资格区分、两seed发布约束。GPU数值、权重变化及独立重载只有实现，未执行，不能称为已验收。

```powershell
python -m unittest discover -s tests -p test_goal_ah_cpu.py -v
```

本轮停止符合冻结边界。下一步先决定确认准备缺口的处理边界；完整实验前提满足后，再由用户确认具体GPU实验。不能把AA/AG确认集重新当新题、挑seed、扩大搜索次数或增加训练遍数来绕过本次缺口。
