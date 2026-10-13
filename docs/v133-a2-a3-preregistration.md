# v133 A2/A3 开发对照预注册（CPU准备）

日期：2026-09-27。状态：CPU准备通过；模型推理未授权、未启动。该文件仅冻结一次旧开发集诊断，不是实验结果或确认性评估。

## S0 输入核验

| 资产 | 实际文件与结果 | SHA256 / 说明 |
|---|---|---|
| 旧开发题 | `data/evidence-sft-v117/frozen/internal-tasks-private.jsonl`；50题、50个唯一ID、50个唯一题面 | `19e6328fa008fb7520c443c293bb4acb6e8f307b569d5a49cb48e17ee56e58d1`，与manifest一致 |
| 固定语料 | `data/evidence-sft-v117/frozen/internal-corpus.json`；1,986篇、1,986个唯一文档ID | `989d7ea08187dfdbf2a07d69cbf6a8d7dd1127ac6d971c9f62b365475f46cea9`，与manifest一致 |
| 冻结manifest | `data/evidence-sft-v117/frozen/internal-environment-manifest.json` | `ccd334605167aaf928d84ecad0d9f53c698a106e2ac93f2d90b78a4c411f154a`；声明50题、100状态、1,986篇、official train only、BM25-v2 top3 |
| 问题时点状态与冻结导出 | `internal-states-private.jsonl`、`internal.jsonl` | `5f7f073a97319dba36f7e867a0a38849e1646dee47069df51c5cd88fcb0d09e1`、`1a33ec08d5b23a365f0b679034806cfb89bdb48c59711db72b98dc253a4ab4f0`；不作为模型输入 |
| v118旧开发记录 | `backups/v118/extracted/runs/v118/eval/v118-tasks.json` | `a388f586bb12ce79b1f1314981ae35876470b7d94c54d9443c504cd83d0ef5bb`；50条完整覆盖、题ID与本轮50题集合完全一致，记录语料哈希相同。仅核实配对身份，不把历史得分当同期A2/A3结果 |
| 基础模型 | `D:\qwen_08b` | 与v118旧评测记录中的13个文件哈希全部一致；1,746,942,600字节基础权重SHA256 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。模型文件未加载 |
| Tokenizer | 同一模型目录的`tokenizer.json`、`tokenizer_config.json` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`、`49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| v118 adapter | `backups/v118/extracted/runs/v118/formal/adapter` | `adapter_model.safetensors` 为 `ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5`，与v118冻结值一致；adapter config SHA256 `c24403ef11e19fc3761452e6389b643331d99d502b11683952a0cd38dd8b8330`，基础模型路径为`D:\qwen_08b` |

v118评测记录的基础模型文件树与本地树逐文件一致。历史记录的adapter目录哈希还包含训练器侧文件；本轮使用清洁的推理adapter目录，冻结其adapter权重、配置和README哈希，且单独匹配权重冻结值。v123候选语料没有被加入本次输入。

## S1 冻结设置

- 运行器：`scripts/evaluate_v133_search_budget.py`。不走`run_v122_remote.py`，不调用训练入口。
- 每组都从相同50题的原始问题开始：system为`direct-v090`，user仅含题面。没有预填首轮搜索、历史首轮结果或gold；动作生成不能读取参考答案。
- 两组唯一的实验变量是成功检索调用的总上限（包含首搜）：A2=2、A3=3。共同A策略识别证据缺失关系，搜索已知实体与缺失关系，不重复查询，证据完整则回答。A2/A3策略正文除“最多2次/最多3次”这一句外相同。它是本次冻结的新端到端设置，不能与v122的29/50固定首轮续跑成绩作同期比较。
- 两组固定：v118权重和tokenizer、1,986篇原语料、BM25-v2 top3、direct-v090、compact-v3、输入上限2048 token、输出上限128 token、BF16、temperature=0、seed42及相同题序。推理顺序为A2后A3，逐题顺序运行；不做并发或参数搜索。
- 重复查询按原A策略拒绝执行，至多一次相同修复调用；修复调用不计搜索。仅成功执行的检索调用计预算。耗尽预算后如仍请求搜索则终止并记`search_limit_insufficient`。协议格式/截断/工具错误分别记录。
- 逐题结果写入`A2.json`、`A3.json`，每次动作另写入append-only事件账本；包括模型实际输入消息、原始输出、prompt/action token记录、截断后可见证据、检索查询与完整top3结果、错误类型及题目开始/结束事件。已完整写入的题可以跳过续跑；发现有部分调用记录而题目结果未完整落盘时，运行器拒绝静默重发并保留账本，须先处理该题。
- 确定性环境须在Python进程启动前设`CUBLAS_WORKSPACE_CONFIG=:4096:8`与`PYTHONHASHSEED=42`。后端仍可能有非确定性，实际设置与输入哈希写入`settings.json`。

### 指标和决策门槛

- 主报告：严格EM、平均token F1、A3相对A2新增正确/丢失正确/净增、A2正确保留率。A2正确数为零时保留率为不适用，不填100%。
- 附报：每题协议合规、弃答/终止原因、实际搜索调用、修复调用、重复查询、截断后证据可见性和检索文档ID。
- A3探索性晋级门槛：EM至少净增3/50、F1不下降、A2正确保留率至少95%、题级协议合规率至少98%。新增正确须经双审证实答案有可见证据支持；影响结论的语义争议保持未解决即不晋级。
- 未同时满足门槛即停止A3候选晋级并交付诊断；不能改prompt、换seed、挑题或重跑以求通过。满足门槛也只可提出独立验证计划，不得声称泛化提升。门槛是工程筛选，不是显著性检验。

### 审核范围

匿名审核覆盖：A2/A3最终答案文本变化的题对；A3实际执行第三次成功检索的题对；以及运行器检测到非法动作、工具失败、输出截断或检索证据为空/不可见的题对。另从其余题对按`SHA256("v133-sentinel:" + task_id)`排序取最多5题。题对先去重。两位审核者分别只见匿名题面、两组对照答案、各自动作前实际可见证据和必要检索原文，不见组名、gold、分数、任务ID或对方意见；不同审核者不看彼此票。双审为DeepSeek及隔离GPT-6 Luna low；S3按交接文件验证通道/身份和调用结果后再做。未审核的共同答案不宣称全量语义支持；不为报告全量支持率而临时扩大范围。

## CPU fixture 证据

运行：`python -m pytest -q -p no:cacheprovider tests/test_v133_search_budget.py`，结果 **6 passed**。覆盖A2/A3策略差异仅为上限、实际成功搜索计数、重复查询的一次修复、预算耗尽终止、截断后可见证据片段、逐题落盘、完整题跳过续跑，以及部分题调用留痕后拒绝自动重发。测试仅使用fixture；未加载模型或调用API。

CPU输入预检：`runs/v133-search-budget-20260927/cpu-preflight.json`，settings SHA256为`dea52bc59e1f5822e5ed0f4679fa85c25eba54b4348ea35f7460a926e6261d14`；`models_loaded=false`、`generation_calls=0`、`inference_authorized=false`、`training_allowed=false`、`confirmatory_ready=false`。完全冻结设置副本为`configs/v133-search-budget.json`。

## Goal A交付状态

Goal A的S0–S1 CPU准备已完成。当前仅能说明输入与运行器已冻结并通过fixture；没有任何模型推理结果。Goal B需用户另行明确授权后，运行两组50题、至多250次成功搜索及至多650次模型生成调用（含每题一次修复上限），串行使用一台已经可用的本地CUDA设备；不启动远端实例、不做电源操作。
