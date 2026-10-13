# Goal AH CPU准备结果 v2

**数据缺口已补齐，训练材料和新确认集CPU门槛通过；GPU尚未启动，等待本版具体计划确认。产品仍为Goal Z完整轨迹seed42。**

唯一训练设计、参数和实验数量保持：原128条轨迹加64条训练专用判断，A/B各seed42、43，每次128更新，共4次训练512更新。训练数据没有扩大，辅助材料及原轨迹哈希未变。

## 有界补齐结果

用户授权后，从本地已验来源的MuSiQue、2Wiki、HotpotQA官方数据筛查。机械隔离合格余量分别110、1151、52665条；各只冻结64候选，合计192。前两者128条原文初审通过32、47；主agent检查53条阳性并选28条，原36条合格题保留，最终MuSiQue32+2Wiki32。Hotpot备用64未使用。

不修改原问题或金标：出生地/from错配、多导演/多角色歧义、仅有厂牌发行或间接县界关系、生日缺日期、Palestinian与Lebanon标签冲突等候选排除。初审阳性不是最终可用数；主审25条排除包含同链去重。

隔离包括全部记录的历史训练、评价与候选池、Goal G外部历史、AA/AG回归和实际AG完整语料。28条新增还排除旧512候选；最终64题实体链互不重复。新2048篇冻结搜索语料逐字保存128段必要支持原文，不按模型或oracle检索表现挑题。仅检查语料含证据，真实取证是否成功留给产品评价。

## 验证与边界

原12项CPU测试与新增5项确认检查通过；原冻结文件哈希全部一致；新确认问题/金标/链未改、原文引用、来源配比、实际语料和冻结文件哈希通过。修复Windows冻结路径表示，保证远端POSIX使用。

GPU身份、有限数值、LoRA变化、基础权重不变和独立重载尚未执行。没有训练、模型推理或新确认得分。辅助agent初审及主agent裁定不冒充独立人类金标准。

诊断结论和13题复核/115题继承的边界保持，见REPORT.v1.md；未重新全量改评。AA与AG仍仅作回归。候选审核原文的接触不进入训练、提示示范或能力保留损失。

## 可审阅文件

- PRECONDITIONS.md：完整人读GPU前提。
- execution-freeze.json：v2实际冻结；execution-freeze.v1.json保留旧冻结。
- confirmation-repair-v2/final-manifest.json：64题、2048篇语料及全部输入哈希。
- confirmation-repair-v2/source-review-main.jsonl：53条主审原文关系/引用/理由。
- confirmation-repair-v2/final-isolation.json：完整曝光检查及128段语料逐字匹配。
- diagnostic-ledger-v1.jsonl、training-material-examples.jsonl：原诊断与四类训练示例。

已执行CPU命令：`python -m unittest discover -s tests -p test_goal_ah_confirmation_v2.py -v`。未生成gpu-approval.json；运行器必须获得绑定本版冻结哈希的具体用户确认才可启动。
