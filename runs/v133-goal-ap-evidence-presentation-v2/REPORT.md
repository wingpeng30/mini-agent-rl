# Goal AP v2：证据呈现探针结果

## 结论

无损文档块呈现没有通过预先冻结的同证据探针门槛，按计划停止；未运行真实搜索A/B、未训练、未发布，也没有使用新确认集。结果显示呈现会改变少数动作，但没有形成稳定的证据绑定收益：AM完整证据正确动作由A的20/24降至B的19/24（净变化−1），B虽修复1个关系绑定错误，却让2个A原本正确的完整证据状态改为继续搜索。冻结条件要求B至少22/24、比A增加至少2且最多丢失1题。

部分证据状态A、B均有7/8个合法补链查询；提前回答、无效动作与重复查询均为0。满足部分证据子条件不能抵消完整证据门槛失败。Goal Z仅作描述性复查，不决定是否继续。

## v1有效性修订

v1的128次生成保留，但不能用作呈现A/B对照：冻结输入解码后，搜索观察位于`user`消息的`<tool_response>…</tool_response>`包装文本中；v1只转换`tool`角色，故B序列化器未处理这些正文，A/B输入实际相同。输出与账本原样保留，详见[v1修订记录](../v133-goal-ap-evidence-presentation-v1/INVALID_RUN.md)。

v2恢复工具消息角色后，通过远程CPU真实tokenizer预检：64条A探针逐条重现原token；64条B输入确实变化；工具载荷解析值、正文哈希、文档顺序、重复项和元数据一致；326个B正文出现均保存字符/token映射；最大输入2584 token，低于16384。v1中AM-A与Goal Z-A共64个状态输出在逐条prompt-token完全相同时复用；v1 B输出全部排除。v2新增64次GPU生成，0次搜索；v2 A/B合计128条有效条件探针。

## AM证据变化

- `Where was the father of Johann Nicolaus Bach born?`：A回答Eisenach，B回答Arnstadt。B是正确关系绑定修复。实际B输入分别保留“Johann Nicolaus was the eldest son of Johann Christoph Bach”和“He was born at Arnstadt”：文档ID `4daf04bbbc3ecccdf9a4fd0f`、`37f133d1958d1a8bc640b31d`，对应文档token跨度`[184,439]`与`[555,1045]`。
- `Where was the place of death of Sir James Abercrombie, 1St Baronet's father?`：A依据James Hamilton的死亡正文回答Hyde Park, Westminster；B在同一完整关系链可见时又搜索。目标人物关系与死亡地点正文仍在输入中，token跨度`[214,692]`和`[1180,1360]`。
- `What is the date of death of Juan De La Cerda, 5Th Duke Of Medinaceli's father?`：A回答1575；B又搜索。父子关系和4th Duke的日期范围在输入中，token跨度`[224,435]`和`[1604,2102]`。
- Las Piñas Medical Center的回答省略了“in Manila”，但仍指向来源中的同一医院，计为等价正确，不算实体绑定修复。Apple Corps答案仅多句点，也仍正确。

AM 32个状态中，9个动作/答案/查询有变化；其中完整证据状态有两次answer转search。其余变化主要是部分证据查询改写，查询差异不等同于已取得必要关系，因为本阶段不执行搜索。

## 部分证据复核与账本修订

初次门控账本把Mark Crossley查询中的“2015”判为可见，因此部分B显示8/8合法。复核发现该年份只出现在无关的2015–16 UEFA Europa League规则正文，不支持题目所问的球队FA Cup夺冠日期；新增窄化仍判不合法。修订后B为7/8，与A持平。初次`probe-gate.json`和初次adjudication均保留，修订版及理由见`probe-gate-revision.json`与`probe-adjudication-amendment.json`。

## 配置与执行记录

两臂固定同一权重、提示、正文、排名、检索、预算和动作协议；唯一输入处理差异为`legacy-json`与`document-blocks-v1`。AM A/B平均输入分别1029.59/1315.78 token，最大输入2245/2584。AM延迟均值约0.855/0.770秒；Goal Z约0.770/0.790秒。时间仅作描述，样本为32次短探针。

Goal Z有3/32个状态的动作或答案不同，其中2个是答案表述变化；这些样本未纳入AM门槛或正确率。GPU日志含64次v2 B生成、0次搜索、0次训练更新。推理启用了greedy，但运行时提示cuBLAS和部分算子不保证逐位确定性，并回退到正确但较慢的参考实现；未重跑或调整环境。结论因此限于这次冻结执行观察，不声称位级可重复。

## 后续决定

结束当前文档块呈现路线，不执行原计划中的64题真实搜索阶段。这个结果只能说明固定权重在同一完整证据下对布局敏感且响应混合；没有证明真实搜索产品收益，也没有证明模型权重能力变化。按照路线图，下一步应改为先评估显式证据引用/关系核验动作的协议成本和匹配对照，或根据现有训练题真实失败状态准备有界蒸馏；二者择一并先冻结调用预算、材料隔离和停止条件，不为AP失败扩充评价材料。

逐条输入、原始输出、正文token映射、修订判分及GPU账本均在本目录。训练使用被禁止。
