# Goal AT：固定采样普通SFT/GRPO对照

两臂均未达到继续门槛，舍弃；实验父版本仍AM42-A，产品仍GoalZ。不追加采样、更新或seed43。

| 指标 | AM父版本 | A正确采样SFT | B固定组GRPO |
|---|---:|---:|---:|
| 支持成功/29 |21|20|19|
| 完整证据探针/24 |20|20|19|
| 部分证据合法补链/8 |7|7|6|
| 事实错误+证据不支持 |7|7|9|
| 动作问题题数/64 |1|2|1|
| EM/64 |26|26|27|

A赢回Bach父亲出生地：实际两边可见，回答Arnstadt。B仍答儿子Eisenach，不能归因奖励驱动绑定改善。两臂Crossley真实搜索新返回Forest正文明确赢杯1898和1959，回答1898按原When接受一个正文支持年份；单列答案集合不完整，不声称两年等价。不要求Crossley本人在夺冠时效力，该额外限定不来自题面。若要求完整年份集合，该题不计成功，A/B各再少1；路线失败不变。

两臂都丢失Rapana、BillMaher和Abercrombie父亲死亡地。A的Rapana公告链完整但答service stations，选错关系对象；B仅搜船只后答merchant aircraft carrier，缺公告边且提前回答。BillMaher未检索2018复播关系而答2017。Abercrombie查询转向本人CharingCross/Edinburgh，没有补到父亲死亡边。B同证据探针另把Millais死亡日答为出生日期。A真实Bach收益没有迁移到固定完整证据探针；没有稳定同证据绑定提升。

Westminster来自实际父亲死亡句HydePark,Westminster，允许Where的较粗地点，单列精度下降，不称公园级等价。29准入层之外35题保持未确认；EM不能替代支持成功。旧账本及评分未修改。

固定16训练状态×4采样，50/64支持、6混合组。JadenSmith组4次均答JustinBieber，零方差不能提供相对优势。两臂各128更新，原128轨迹+34状态相同，6混合组A监督首个正确样本，B四样本组标准化奖励；其他组零新增损失。B使用.7温度全softmax的old/current/reference，parent冻结后共同SFT会使策略偏移，clipped比率和KL如实保存；这是有限离线池对照，不称持续在线强化学习。loss/optimizer/历史mask测试3项通过。base未变、192LoRA更新、两次独立重载max差0。

实际额度：511新增生成、255真实搜索、256更新；无新确认。两臂均无正文裁剪或长度停止。每题/调用延迟与动作分类见mechanical-metrics.json；引用、token位置、首次分歧见semantic-review.jsonl/wins-losses.jsonl。失败属于语义与门槛失败，未触发GPU/数值/身份/基础设施停止。

按顺序进入AU独立reranker可行性核验。不得把本轮B的EM比A多1解释为模型升级，也不以新增数据重跑本配方。


## 固定池策略更新限制

B有效混合组更新6次，覆盖24个采样输出；平均token裁剪比例0.0435，最大0.1538，平均冻结父策略KL估计0.0280。共同轨迹SFT发生在组监督之前，采样池没有刷新，old/ref始终来自AM父模型。该结果只淘汰本轮固定离线池配方，不能概括为所有RL策略无效；裁剪统计也不单独证明语义退步的原因。未因此追加采样或改变clip/KL。
