# Goal AI第一阶段收尾：证据核对到动作恢复

2026-10-03。结论：本轮固定审核已完成，**未通过动作恢复SFT入口，不启动训练**。推荐下一步先做固定模型的必要关系覆盖与检索可行性实验。唯一产品仍为Goal Z完整轨迹SFT seed42；本轮没有产品升级或泛化收益结论。

## 实际执行与可靠性

全部实验计算在远程AutoDL仓库，用`.venv/bin/python`完成。本次无GPU采样、无模型调用、无权重更新。使用此前32条GPU轨迹、94次调用和62次搜索；94次原输入token IDs逐次重建一致。继承AH终审分数，不改旧评分。

24题复用已有53次独立替代查询；每个分支分别重建实际产品输入和调用/搜索预算。32条训练轨迹只冻结并执行16次CPU补证查询，每题一次，失败不换查询。教师检索累计22+16=38次，低于原96次额度。不重扫144候选池、不追加数据、不访问封存新确认题。

保留Goal Q、direct-v090、thinking=false、legacy、BM25-v2/top3、4调用/3搜索、2048输入/128新token。加载身份核验使用AA发布的Goal Z adapter权重SHA-256：`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`。本次只加载tokenizer/processor重建输入，没有加载模型做推理。

机械验证通过：34个动作样例、70处引用位置；最大实际输入1713 token、最大动作目标24 token；prompt与工具观察loss mask为0，正确assistant动作和结束符为1；272题评价ID及规范化题面均与32训练题隔离。语义裁定由主agent逐边作出，机械验证不冒充独立语义审查。

## 资格决定

| 条件 | 核验结果 | 决定 |
|---|---:|---|
| 原预算内恢复完整链至少8/24 | 1/24 | 缺7题 |
| 合格完整search→answer恢复对至少16 | 10 | 缺6对 |
| 至少8个真实Goal Z失败前缀 | 10 | 通过；旧池8、新训练池2 |
| 实体/角色错配至少4独立例 | 5 | 通过 |
| 关系类型/方向/范围错配至少4独立例 | 3 | 缺1例 |
| 缺失关系至少4独立例 | 10 | 通过 |

覆盖标签可重叠，只计算完整恢复对。4条回答修正和10条已成功正例单独导出，不能凑完整恢复对；8条拒绝/歧义案例保留原因。旧池8不等于原Z训练池8：其中7来自Z训练源、1来自AC；它们和新2条的实际失败状态均由发布Goal Z产生。

`gate-decision-v4.json`是最终统计。v3把来源枚举误写成无后缀的`old_real_failure`，造成旧池计数0；v4按真实枚举和“发布模型实际失败”的门槛更正。原文件保留，语义裁定及不训练结论均未改变。

## 固定24题：实际发现了什么

| 案例简名 | 缺失或误用的必要关系 | 冻结分支结论 |
|---|---|---|
| Gloria Kemasuode | Delta State首席法官；取到Rivers State官员 | 未补齐；不能跨州套用 |
| OTC/FDA | 药物营销权限→机构→食品安全系统 | 未补齐；机构名不是系统名 |
| Jamie Ward | 相关球队→FA Cup胜利年 | 未补齐；多历史球队限定待明确 |
| Mark Crossley | 球队→胜利年 | 未补齐；参赛资格规则年份不是胜利年 |
| Hagar | 配偶关系→Sarah死后婚姻 | 未补齐；共同生育不独立证明婚姻 |
| Juan第五公爵 | 父亲→父亲死亡日期 | 一个查询提前夹入第四公爵姓名/序号，剔除；合法分支缺边 |
| Tony Parker | 所问时间的工作机构 | 所有者/退役球员不能自动转为当前雇主 |
| Shrek 2演唱者 | Jennifer Saunders→配偶 | Mirrorball共同创作/导演不证明婚姻 |
| St. Matthew Island | 海域位置→海域命名对象 | 未补齐；保护区单元不足以独立证明所在海域 |
| Bonne of Armagnac | Bernard VII→出生地 | 人物正文无出生地 |
| Frances Tupper | Charles Tupper→出生地 | Dickey/Anselm Tupper出生地不能套用 |
| Wenceslaus of Krosno | Henry VIII Sparrow→死亡地 | 统治地和其他同名人物不足 |
| Axiliad | Witold Leszczyński→出生地 | 国籍/就读Łódź不证明出生地 |
| Horatio Powys | Thomas第二男爵→出生地 | 婚礼、墓地、死亡地不能转出生地 |
| Philipp Albrecht | 母亲Margarete→出生日期 | **唯一合法完整分支：13 May 1870** |
| Shubhendra Shankar | Annapurna Devi→出生地 | moved to Bombay不证明出生于Bombay |
| Stefan Vladislav | 父亲→国籍 | 统治Serbia不独立证明国籍，保留歧义 |
| Charlie巧克力工厂 | spoilt角色→演员 | 主角演员命中不足以证明指定角色 |
| Rose Tint My World | Richard O'Brien→配偶 | 作曲者本人不是配偶；正文无婚姻 |
| Day of the Assassin | Brian Trenchard-Smith→出生地 | English-Australian不证明出生地 |
| Hickory Grove Estates | Oktibbeha County→县治 | 学区并入Starkville不证明县治 |
| Desert Studies Center | CSU全系统校区数 | 联盟7、南加州12都不能转成全系统总数 |
| Jujhar Singh | Guru Gobind Singh→父亲 | 两人会面不证明父子 |
| Adalbert II | 父亲→出生地 | 统治地及Bologna同名人不能补边 |

唯一通过的分支从真实call1前缀搜索母亲生日，再在call2回答，合计3次模型调用/2次搜索；原输入中父亲生日被误套到母亲。这里只证明教师选择的合法查询能提供可回答上下文，**没有执行模型最终回答，不计真实搜索成功率**。

1/24是本次冻结分支的可恢复数，不等于其余23题在所有可能查询下都不可恢复。已有源文定位记录中也有明确缺边或实体/关系不符；这支持先核查必要关系覆盖，不能证明整个冻结语料绝对没有这些关系。未重开全量评分，也未将全部缺证归因为模型理解失败。

进度分开统计：合法诊断分支中4题（6分支）新取得至少一条必要边，1题取得完整链、1题在实际输入保留完整链。16次训练补证中11题的原前缀加原始返回能组成完整链，但只有10题在实际下一输入保留完整链；差额是Chelsea。详见`evidence-progress-v4.json`与逐分支进度账本，隐藏实体分支不计入这些数字。

## 32条训练轨迹与材料

10对通过：Carpenter县、Titik国家统一年、Pawnee所在州入联邦、Cabriola导演配偶、Foggy Pass大陆面积、Scalvini出生城市市长、Takoma殖民地欧洲创建者、Kavisigamuwa国家独立日、Animal Justice艺人母亲、Damian Green被捕新闻的议会特权。每对保留原错误前缀、合法查询、真实返回、预算后的输入、逐边引用和目标动作。

4条回答修正：Jaden Smith（不是Justin Bieber）、University of Helsinki（不是professor）、September 3, 1922（补全年份）、Christiansburg, Virginia（不是出生日期）。完整链已在原输入中；无需伪造一次搜索。

10条已成功正例：The Path She Chose、Claude Whatham学校、Bluebottle OS、Murder in a Blue World、Rhodes of Africa、Sam Fishburn春训、Delamere Way、Richilde配偶死亡日、Marvin Cohen剧本朗读者、Dopey Dicks。Chicago, Illinois与Chicago等价；直接“English county”足够证明England，不要求取回每篇官方支持文档。

8条拒绝：Vilsandi返回俄国岸区但major-city限定未充分建立；Joe McLaughlin补证后上下文丢证；Virgin Islands Pipeline命名关系歧义；Something For题名歧义；David Friesen县域缺边；Elbert邻县缺边及同名县疑点；Joshiy出生地缺边；Maria Nowakowska出生地缺边。

特别改判：`from Varkala`与`student from Legnica`仅支持“来自”，不支持“出生于”；第二条还缺明确battle end关系。PCCNG在Tacoma附近发行不能证明Tacoma属于Pierce County。Elbert案目前只证明West Virginia的McDowell County，不断言金标Avery已确定错误。

旧6条待审全部有主agent结论：学校案例完整通过；Maryland案例按“欧洲永久殖民地创建者”语义通过；Pipeline和Something For因歧义拒绝；Never Say Never与Ossian Donner为回答修正。保留旧A/B及lead记录，新结论不是新增独立双审。

## 可审核材料例子

**Carpenter恢复**：原输入SoDak Con说明名称来自South Dakota，模型却回答Utah。修正查询`Carpenter South Dakota county`全部实体来自题目/可见正文。真实返回“Carpenter is an unincorporated community in Clark County, South Dakota, United States.” 与SoDak命名边同时可见，答案`Clark County`。补证后实际输入1346 token；新增动作不超128；总4调用/3搜索。

**John Cale恢复**：Animal Justice明确由John Cale表演，模型取了虚构角色母亲的演员。查询`John Cale mother`不夹入答案Margaret Davies。真实John Cale正文给出父母Will Cale与Margaret Davies及母亲叙述。补证后1447 token，两边均可见，答案`Margaret Davies`。

**Chelsea拒绝**：查询结果的经理表包含“Roberto Di Matteo Italy 2012 2012 FA Cup 2012 UEFA Champions League”。但实际下一输入只有245 token、0篇可解析文档，完整工具消息因超预算被丢弃。检索返回完整不代表模型看到了完整。这是上下文保留的可复现实证，未在本轮修改产品实现。

## 唯一后续路线与停止边界

优先准备**固定Goal Z模型的必要关系覆盖与检索可行性实验**：先在同一24题的冻结语料内逐边验证关系是否存在，再限定实体来自可见证据的查询，区分“语料有边但没搜到”和“已审核原文不能支持”。下一轮数量、源语料、可改变量及停止条件见`docs/v133-goal-ai-next-stage-plan.md`更新。当前不启动、不扩池，不同时改模型、提示、检索与上下文。

上下文丢证单独登记为后续工程实验候选；不能因一个案例就归因所有失败。关系判断和回答错误确实存在，但本次材料不足以验证训练泛化；不再增加孤立判断辅助任务，也不追加遍数。

动作恢复SFT的A/B seed42、43设计保留为条件方案，本轮没有实际训练材料规模达到16，因此不冻结并启动128更新×4训练。未来达标后仍须先展示并确认实际数据、参数、mask/loss归一、混合权重、更新数、评价及发布前提。

本次真实搜索成功、产品EM/F1、事实错误加证据不支持、无效动作、搜索次数及产品延迟没有新的比较测量；不能用10对教师修复替代产品结果。未来产品验收仍要求两个seed各+4/64、坏错与无效动作不增、各回归最多降1题、86连续搜索保护最多丢1题、工程与重载通过；机制归因比较B与A。AA/AG确认集只作回归，发布用真正未暴露新确认集且不按结果挑seed。

## 文件与复现

- `diagnostic-ledger-v4.jsonl`：24题必要三元组、七类轴、实际引用/token位置、分支及缺口；评价诊断不可训练。
- `branch-audit-v3.jsonl`、`budget-reconstruction.jsonl`：53分支逐一隔离，实际输入及预算。
- `train-audit-ledger-v4.jsonl`：32题完整审计；`material-review-revisions-v3.jsonl`保留改判理由。
- `repair-query-freeze.jsonl`、`repair-query-results.jsonl`：16次事前冻结CPU查询和实际观察。
- `recovery-pairs-v3.jsonl`：10对真实恢复；另外两个example文件分别保存4条回答修正与10条正例。
- `gate-decision-v4.json`、`verification-v4.json`：最终门槛及机械验证。
- `freeze.json`、`closure-manifest-v3.json`与最终交付manifest：身份、输入及产物哈希。

脚本：`close_v133_goal_ai.py prepare/search`、`freeze_v133_goal_ai_closure_queries.py`、`finalize_v133_goal_ai_closure.py`、`verify_v133_goal_ai_closure.py`、`annotate_v133_goal_ai_closure.py`。均在远程执行；已有产物用独占写入，不能原目录重跑。v3→v4统计修正由独立验证脚本记录，保留旧版。任何后续运行应建新冻结目录而非删除已有产物。
