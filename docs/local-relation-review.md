# 最终关系复核：实现与失败诊断

2026-10-06：工程功能已实现，实验候选未通过第一阶段门槛，不能宣称多条关系推理错误已修复。没有训练、下载模型或调整检索参数。网页服务已恢复 AM，复核默认关闭。

## 接口与预算

`POST /api/runs` 接受可选 `reasoning_mode`：`off`、`generic-review-v1`、`relation-review-v1`，省略为 `off`。独立模块位于 `src/mini_agent_rl/model/final_review.py`，复用现有搜索循环，不修改冻结研究模块的预算。

首次准备回答且已有可见工具证据时，最多额外生成一次。普通决策最多 4 次，搜索最多 3 次，实际模型调用最多 5 次。复核只读取原问题、未经验证的候选、已有查询及实际输入中可见的文档正文，不读取标准答案或支持标签。有效复核替换候选；JSON、字段或引用无效则保留候选，记录失败且不重试。缺失关系需要搜索但预算不足时，以 `insufficient_evidence` 结束。

关系复核检查一至两条 `subject/relation/object/document_id/quote`，引用必须来自实际可见正文，答案必须与最后一条关系的对象一致。这些机械检查不能证明关系在语义上成立。运行记录保留原候选、复核原文、校验、最终动作、调用数和耗时。

37 项相关测试通过，覆盖一次性触发、预算、无证据跳过、无效 JSON、伪造与截断引用、回退、缺证据搜索、旧请求兼容和网页 API 工具循环。实际服务 OpenAPI 已确认三个模式和默认 `off`。由于候选未通过验收，未添加网页关系复核开关。

## 精确状态诊断

从保存的轨迹重建三个案例的 BM25/混合最终回答输入，六份实际 token 数组均与原记录完全一致；GPU 重放也复现了原 BM25 三题全对、原混合三题全错。固定 Goal Z、seed=42、BF16 和原输入/输出上限。

每题准备原 BM25、原混合、按 ID 排序、逆序、仅支持文档、替换干扰文档六种状态。排序只改每轮顺序；替换使用原查询的 BM25 非支持结果。Körling 的替换条件不满足，标为不可比较。仅支持文档是使用离线标签的理想诊断，不能用于产品或算作真实检索收益。

| 案例 | 原混合 A | 排序/逆序观察 | 仅支持文档 A |
|---|---|---|---|
| Barney → Batman 角色 | Neil Patrick Harris，错误 | ID 排序仍错；逆序为 Jason Isaacs，仍错 | 仍返回 Neil Patrick Harris |
| Julie Andrews → Princess Diaries 角色 | Julie Andrews，错误 | 两种顺序均返回 Ernest Thompson，错误 | 输出搜索动作，未回答 |
| Felix → 父亲 | John “Felix” August Körling，错误 | 两种顺序均返回 John，错误 | August Körling，正确 |

Princess Diaries 原混合第二次查询已经选择 `Ernest Thompson cast member`，原 BM25 则查询 `Julie Andrews character in Princess Diaries`，说明错误在最后一步前已出现。文档顺序会影响行为，但这三例没有支持一种通用的排序修复；移除干扰只在父亲案例中直接修复了答案。不能据此认定所有错误都由排序或干扰造成。

## 复核结果与停止决定

17 个可比较状态执行 A，其中一个 A 输出搜索，未触发 B/C；剩余 16 个状态各执行一次 B/C，共 49 次生成，没有执行新的工具搜索或完整问答。

**C 修复原混合错误 0/3，低于至少 2/3 的进入门槛。** 16 次 C 均未通过关系协议校验：模型输出普通 `search` JSON，缺少要求的关系链与缺失关系字段。系统正确回退原候选。少数 C 记录中的正确答案来自原候选回退，不能算作关系复核收益。B 的 16 次输出均为搜索动作，本阶段不执行新搜索，不能据此推断完整流程 B 的效果。

| 实验臂 | 生成次数 | 输入 token | 输出 token | 生成耗时 |
|---|---:|---:|---:|---:|
| A 原流程 | 17 | 18,648 | 250 | 26.562 秒 |
| B 普通复核 | 16 | 14,561 | 239 | 22.917 秒 |
| C 关系复核 | 16 | 16,401 | 242 | 23.184 秒 |

耗时不含模型加载和 CPU 准备；B/C 调用次数匹配，但输入量不同，不能称为计算量完全相同。按冻结计划停止，未运行后续 40 题 A/B/C 和 64+24 题回归，因而没有这几组的新 EM/F1 或配对置信区间。没有继续扫描提示词。

本轮暴露了关系输出协议未被遵守的问题，不能证明必须重新训练，也不能判断结构化输出成功后是否能改善关系推理。后续若研究格式约束或关系监督，应另立候选与验证协议；本次没有自动启动训练。全部数据属于已暴露诊断/回归，不能称作独立泛化验证。

## 记录与复现

正式目录：`E:\AgentLightningAssets\runs\retrieval-v1\relation-review-7f3e7c56e59c45118bd3ca2c8bbc1bdc`。

- `experiment-manifest.json`：冻结代码、权重、配置、题集身份与预算。
- `preflight.json`、`states.json`：精确输入验证、状态变体、不可比较条件。
- `generation-ledger.jsonl`、`probe-ledger.jsonl`：49 次原始生成及逐状态校验。
- `report.json`：`complete=true`、`accepted=false`、`probe_repairs=0` 与停止原因。

在项目根目录设置 `PYTHONPATH` 为本项目的 `src` 和 `scripts`，使用 `D:\anaconda3\envs\mini-agent-rl\python.exe -B scripts/run_local_relation_review.py` 可生成新的 CPU 预检目录。`--run <新目录>` 才进行 GPU 实验；须先确认网页空闲并释放其模型，避免同时加载两份 GPU 权重。不要覆盖正式目录或重复执行已有生成账本。本轮仅停止和恢复了本地服务以释放显存，没有重启 GPU、CUDA、电脑或远程设备；结束后恢复 AM。
