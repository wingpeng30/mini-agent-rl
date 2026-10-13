# 新会话交接：最终实验已完成（2026-10-12）

先读[当前项目状态](../PROJECT_STATUS.md)和[最终结果与面试说明](final-online-grpo-20261012-results.md)。批量异步执行、继续SFT、在线GRPO双seed与裸模型评价全部完成，128题结果、最终adapter、恢复点和综合指标已取回，远端关机流程已触发。当前是面试收尾，没有待执行GPU实验。

最新全128题归一化EM：AM67、继续SFT68、GRPO42为61、GRPO43为66。batch4为64，吞吐2.78倍；GRPO无稳定提升。裸模型未输出最终答案，协议下EM0。自然104题、F1与其余指标见当前状态。

下方2026-09-27交接保留为历史，已被后续实际工作取代。“新会话首先执行”“仅CPU准备”“尚未授权推理”等均不构成当前任务或权限状态，不自动创建旧Goal、运行探针或重启实例；以用户当前请求为准。

## 历史交接：2026-09-27搜索预算实验

项目：`D:/AgentLightning/mini-agent-rl`。交接日期：2026-09-27。

用户已经授权重写研究目标并放宽探索阶段的数量/配额前置门槛；用户尚未授权本轮实际模型推理、SFT、GRPO、GPU实例操作或新候选池。本会话仅编写交接，不运行实验。下面的来源与历史连接信息均为定位线索，不代表新会话已验证可用。

## 新会话首先执行的任务

阅读 `docs/v133-staged-research-goal.md`，按该文件创建并执行限定为S0–S1的原生Goal A。目标是完成一次A2/A3开发对照的CPU准备与预注册，不是把全部历史数据修成理想训练集。v128原生Goal仍blocked，保留原状；不要运行其next/accept调度来决定v133的工作，也不要标记它已完成。

Goal A建议目标文本：

> 在D:/AgentLightning/mini-agent-rl完成v133阶段S0–S1：按具名白名单核对旧开发50题、对应1,986篇语料、v118权重/tokenizer和A运行器，准备从问题起点执行的A2最多两次搜索与A3最多三次搜索对照；修正比较所必需的运行器问题，用离线fixture验证，冻结指标、审核范围、停止规则与运行配置，交付输入哈希、测试证据及具体的推理授权清单。只做CPU准备，不调用模型生成、训练或GPU入口，不新建候选池，不访问final-test或独立100题。数量395/211/100和新内部50题不是该Goal前置条件；数据真实性、来源/隔离和证据时点不放宽。真正完成上述CPU交付才完成Goal A。模型推理须待用户明确授权后另启动Goal B；不把准备完成称为实验完成。

## 最小必要上下文

1. 当前已在数据审计上耗时过多。用户要求以发现模型/检索/控制器实际问题为中心，减少重复全量审核与为配额扩池。新路线是CPU准备 → 一次固定权重A2/A3推理 → 配对诊断和必要双审 → 决定是否值得继续A3、修控制器或训练。负结果可以结束研究分支。
2. v122的固定首轮续跑A/B/C/D分别29/24/31/25（共50题）；C净增2、引入1个回退且有语义争议，未晋级。不是端到端、不是未见集。v111的78%是训练侧API混合流程成绩。
3. v128当前G0–G3通过，G4阻断，后续等待。G3已恢复DeepSeek审核通道；阻断不再是缺API key。父工作区`.env`有过可用配置，禁止显示/复制密钥。Python环境代理曾拒绝TCP连接；显式直连、关闭thinking及v130引用协议的小包调用得到两张有效否决票，只证明当时通道可用。
4. v130唯一获准的新池筛查5,000题并冻结100题/200状态，额度已耗尽。有效Luna票：搜索否决85、答案否决93、答案批准2；引用无效20，无完整批准配对。原不足证书stop_reason里“All 100 ... valid ...”过强，使用上述细分统计，不能把无效票当有效否决。
5. linked-title CPU构造方法把可见文本标题当作后续查询，并把官方答案放到末尾，没有保证缺失关系相关性或最终证据支持，沿用旧语料也不保证新题覆盖。因此停止该构造器。失败针对所造动作/证据，不等于官方问题无用；不是模型成绩。
6. 已有训练候选预检为278状态/175题/37严格配对（228核心＋50补充）。另有19组旧内部配对可按当前机械复核复用旧双审。它们不是自动可训练资产或新未暴露验收集；本轮A2/A3不依赖完成它们的全量再审。
7. 旧开发50题已反复暴露，只作开发诊断。新A2/A3应各跑同期端到端50题，不能与历史29/50直接比较。只改变搜索预算，不同时加入Q、S、C、重排序或训练。
8. 旧授权源池额度不得重置；目前不新建池。正式端到端评估以后需要问题/答案/语料与隔离，不要求事先造出每题成功轨迹。人工参考答案或支持事实可用于评分/事后诊断，不能泄漏给动作生成过程。

## 优先读取的具名证据

以下均相对项目根目录。按实际问题读取，不一次性重审所有内容；不以旧报告布尔值代替需要的原始证据。

- 新目标：`docs/v133-staged-research-goal.md`。
- 旧实验：`docs/v122-unified-actions-results.md`、`docs/v122-unified-actions-progress.md`。
- A运行器定位：`scripts/evaluate_v122_unified.py`、`scripts/evaluate_v118.py`；设置定位：`configs/v118-sft.json`、`configs/v122-targeted-sft.json`。阅读调用链后才能执行，特别防止训练/封存评估分支。
- 当前旧Goal摘要：`docs/v128-data-readiness-goal.md`、`data/evidence-sft-v124-forensic/goal-v128/status.json`。文档同时保留旧快照，“当前”标题可能过时，以具名原始票和最新记录为依据。
- 通道：`data/evidence-sft-v124-forensic/goal-v128/g3-channel-diagnostic-v132-private.json`、`scripts/v130_deepseek_review.py`、`scripts/v130_review_protocol.py`。
- 新池失败：`data/evidence-sft-v124-forensic/goal-v128/g4-insufficiency-certificate-v132-private.json`、`g4-blind-review-disposition-v130-private.json`、`g4-new-pool-review-diagnosis-v130-private.md`（后两项同目录）。
- 构造代码：`scripts/v130_build_new_internal_pool.py`；实际否决示例：`data/evidence-sft-v124-forensic/goal-v128/new-internal-audit/deepseek/g4-small-02/validated-review-private.json`。该示例题目是1919年西部短片制片人的出生年份；不要沿用把它误写成电视台问题的旧摘要。
- 现有训练数量：`data/evidence-sft-v124-forensic/train-core-candidate-preflight-private.json`、`data/evidence-sft-v124-forensic/double-reviewed-v122-supplement-preflight-private.json`。
- 19组旧票范围：`data/evidence-sft-v124-forensic/goal-v128/g4-v123-reuse-v129-audit-v2-private.json`，只覆盖42个已记录等价映射，不能外推全体候选。

未知API请求g4-batch-01与g4-batch-03不得自动重发；g4-batch-02及g4-small-01的技术更正机会已耗尽。不得为了连通性再发送这些包。服务暂不可用时继续独立CPU工作，只阻断依赖审核的结论。

## 保留的操作边界

- 封存final-test与独立100题保持不扫描、不读取、不上传。禁止通过压缩包或别名绕过。只读与当前任务相关的具名文件；不把整个项目递归扫描当作定位方法。
- 本轮不启动模型推理、smoke、SFT、GRPO或GPU/电源操作。Goal A完成后请求具体的一次推理授权；后续如果用户已明确授权相同操作，就不要重复询问。SFT永远需要独立明确授权。
- 继续DeepSeek＋隔离Luna审核。必要时最多两个GPT-6 Luna agent，建议low；审查者无历史上下文、只读匿名包与规则、不看另一票，不参与开发选题或调参，不派生更多agent。Goal A无需为整理文档启动审核agent。
- 无效、争议、单审和有效否决样本不进入合格训练集合；不能重审求通过。旧输入和引用完全等价才复用票，不反复全量审核。
- 不恢复费用账本、不打印凭据、不自动上传私有数据。远端连接仅沿用已明确授权范围并先验证当前状态，历史地址/端口不等于存活会话；不要把密码复制到新对话或交接文件。

## 首轮交付和验收

只在实际完成S0/S1后交付：确认过的资产清单与哈希、必要代码变更与离线测试、两组固定配置、预注册指标/审核范围/停止条件、推理命令与资源范围。不要仅写“可执行方案”就结束Goal A，也不要以“新内部集未达50题”阻止开发对照准备。

Goal B必须在明确授权后运行100条端到端轨迹并完成判读。其完整交付可支持继续或停止，不能因结果不理想自动再做第二轮。当前没有模型提升、训练通过或正式验收通过的新结论。
