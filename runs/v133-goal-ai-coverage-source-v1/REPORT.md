# 证据覆盖与题目依据：有限来源核验

日期：2026-10-03。独立CPU审计，固定既有24题；新增搜索0、GPU0、训练0。旧题、旧语料及旧分数不修改。

## 主要结论

24题全部定位到官方dev原始样本，48份官方支持正文在各自冻结语料中**逐字完整保留**。本轮未发现原文抽取、分段或建池丢段，不能把关系缺口归因为材料组装丢文。

对上一轮20条未确认边，每边完整读完既有12候选，240个边—文档组合、224份去重正文，共146915字符；没有新增直接支持。这仍只是有限候选审核，不证明2048篇语料绝对缺证。官方支持标签、答案词出现和正文支持关系分别记录。

| 来源裁定 | 24题固定分母 |
|---|---:|
| 原题完整链有直接支持 | 2 |
| 需要时间、版本、角色或关系等价限定 | 8 |
| 官方支持正文仍不足以证明必要关系 | 14 |
| 原文到冻结语料转换丢失 | 0 |

这24题来自既有失败深诊断，不是随机样本，上述比例不能推广到全部数据集或全部272题。

## 具体处理

两条可直接支持的原题是Juan父亲死亡年和Philipp母亲生日。保留此前上下文丢边、合法补证后正确回答的诊断，不新增模型调用。

14条依据不足题保留原题、标签和来源记录，从未来“充分回答”目标的准入中拒绝；这不是宣布标签在现实世界必然错误。例如Amherst出生句说的是Arthur Rupert Dickey而非Charles Tupper；学校Łódź不是出生地；Kimi来自组合名而非婚姻；12校区是区域范围而非整个CSU；相遇不是父子。Brian的官方出生地点答案English还存在答案类别与关系类型不一致。

8条限定类逐题保留理由：首席法官继任有历史证据，但不能无条件作当前办公室断言；Ward最近列出的2015俱乐部可作来源一致解释，不能自动断言所有球队都被唯一排除；Crossley有多个历史俱乐部；Tony的Spurs是历史效力；FSMA法案与food-safety system需处理等价措辞；Bering Sea行政单元到地理位置属间接推断；中世纪王位/国籍/公民身份需固定关系；Chocolate Factory官方分解明确1971 Willy Wonka，合成题却使用Charlie题名，不能自动把Julie Dawn Cole判事实错误或强制改成2005版本。

所有8条目前仅提出限定需求，未改写题目、评分规则或生产提示。若以后制作变体，另存variant_id、限定理由和独立分母；不能替换旧题后报告产品收益。

## 唯一推荐下一步

准备一套有原文依据的评价与材料准入规则，优先检查现有、已暴露开发材料的必要关系。原官方support标记只作定位线索；每边须有正确实体/角色、关系方向、对象及时间/地域/数量/版本的正文依据。训练材料和评价材料继续分开，当前24题及其评价来源不能导入训练。

这一轮不建议修生产语料构建代码：48/48已完整保留，缺乏该改动依据。也不建议开始四臂训练：可靠的原题诊断基础和材料门槛尚未满足。后续开发材料审计应另行冻结有限数量、只做一次来源审核、保留不合格题并报告缺口；不按模型成败挑题，不访问未暴露新确认题，不通过互联网补事实来凑可答率。

首个交付应是源文准入规范与隔离清单，以及一个有限审计方案，随后才决定上下文、检索或动作恢复训练。Juan/Chelsea上下文证据继续保留，但不能用这两个案例代替完整机制实验。

## 交付与边界

freeze记录24个目标ID、官方档案哈希和仅dev成员；流式解析只导出这24题，未将其他题面、答案或新确认题送入模型或人工报告。扫描为定位既有ID，不进行候选扩池或新题筛选。source-corpus-mapping记录正文精确匹配；official-source-packets保存官方链/标签，标记为诊断，不能当作关系证据。

fulltext-reading-log记录224份正文完整阅读及六个批次；candidate-edge-review保存240组全文引用、字符位置和拒绝理由；edge-review-v2继承已有正例并更新审核范围；case-source-review列出24题原文、关系缺口、分类与处理；review-revisions保留改判理由，不覆盖旧账本。

核验原题逐字相同、每题2份官方支持、48份正文逐字一致、20条未确认边各12候选、全部旧运行冻结哈希未变。模型未加载，不触发GPU实验或重新评价。

## 逐题结果

| case_id | 类别 | 理由 |
|---|---|---|
| musique:2hop__862994_69048 | qualification_required | Source proves Bozimo retired in 2011 and Abiodun succeeded her; supports a qualified historical successor answer, not an unrestricted current-office assertion. |
| musique:2hop__44522_58556 | qualification_required | FSMA grants FDA food-safety powers; food safety system versus named legislation is an unresolved wording equivalence, not proven false. |
| musique:2hop__442175_56873 | qualification_required | Forest is latest dated club listed (2015), with two FA Cup years; question does not specify date or distinguish national team. Source-aligned club interpretation is plausible. |
| musique:2hop__337223_56873 | qualification_required | Multiple previous clubs; notably Forest does not uniquely resolve Crossley team. Forest years are supported conditionally. |
| musique:2hop__468778_88165 | source_insufficient | Hagar handmaid and mother of Abraham son does not prove spouse; Keturah second edge is explicit. |
| 2wiki:f299dd600bdd11eba7f7acde48001122 | original_chain_supported | Named fourth-duke father and death year1575 explicit; year granularity is adequate for the original when question. |
| 2wiki:d94d1d720bdd11eba7f7acde48001122 | qualification_required | San Antonio Spurs is explicitly a historical team, while source says former player and final Charlotte season. Present work-at requires a time interpretation. |
| musique:2hop__71034_343058 | source_insufficient | Film singer Jennifer Saunders is explicit; Mirrorball writer/director credits do not establish marriage with Adrian Edmondson. |
| musique:2hop__864101_135373 | qualification_required | Island administrative membership of Bering Sea refuge unit is indirect geographic evidence; sea-naming edge is explicit. Exact location inference remains unresolved. |
| 2wiki:015a72a60bde11eba7f7acde48001122 | source_insufficient | Bernard VII father explicit; Paris occurs in government invasion/death narrative, not birth. |
| musique:2hop__475070_532404 | source_insufficient | Charles Tupper husband explicit; Amherst birth belongs to Arthur Rupert Dickey. |
| 2wiki:66616e2a0bde11eba7f7acde48001122 | source_insufficient | Henry VIII father explicit; Szprotawa is ruled territory, not death location. |
| 2wiki:b592bb5c0bd911eba7f7acde48001122 | source_insufficient | Witold director explicit; Lodz is graduation school location, not birth. |
| 2wiki:f480bf400bdd11eba7f7acde48001122 | source_insufficient | Thomas second baron father explicit; Mary Mann of Lilford Hall does not establish Thomas birth there. |
| 2wiki:ae714dea0bdc11eba7f7acde48001122 | original_chain_supported | Mother Margarete and birth date13May1870 explicit in original body. |
| 2wiki:085ace840bde11eba7f7acde48001122 | source_insufficient | Annapurna mother explicit; being named by Maihar ruler does not prove birth in Maihar. |
| 2wiki:d438a1020bd911eba7f7acde48001122 | qualification_required | Question nationality, official chain country-of-citizenship and source kingship are distinct; medieval interpretation requires explicit scope. |
| musique:2hop__65576_80026 | qualification_required | Original decomposition explicitly asks 1971 Willy Wonka actor Julie Dawn Cole, but composed question uses Charlie title; report version ambiguity rather than automatically marking label false. |
| 2wiki:66c00bf60bde11eba7f7acde48001122 | source_insufficient | Kimi appears in pop duo Kimi and Ritz; no marriage edge. Song-writing/composition evidence alone cannot supply spouse. |
| 2wiki:43a949ae0bdc11eba7f7acde48001122 | source_insufficient | Brian director explicit; English-Australian is nationality descriptor, not a birth place. Official answer English has category mismatch. |
| musique:2hop__495316_660579 | source_insufficient | Oktibbeha county explicit; merging county schools into Starkville district does not prove county seat. |
| musique:2hop__455392_40315 | source_insufficient | CSU institution explicit; 12 campuses in Southern California is regional scope, not whole system. |
| musique:2hop__124932_278133 | source_insufficient | Gobind father explicit; meeting Tegh Bahadur does not establish fatherhood. |
| 2wiki:13e2f86c0bde11eba7f7acde48001122 | source_insufficient | Adalbert I father explicit; realm/office and Kingdom Italy political narrative do not establish birthplace. |
