> 历史文档提示（2026-10-12更新）：最新成果与执行状态见[当前项目状态](../PROJECT_STATUS.md)及[最终结果](final-online-grpo-20261012-results.md)。本文件保留旧阶段记录，其中“当前/下一步/尚未实现/最终”按原日期理解，不构成当前待办或新的实验授权。批量异步、继续SFT、双seed在线GRPO及裸模型128题评价现已全部完成。

# 4090D 24GB：实验进度与下一阶段执行提纲

日期：2026-09-17。用户已确认项目部署到 4090D 24GB；本文件依据本地代码与报告编写，尚未读取新机器硬件、依赖或运行结果。Linux 为迁移配置默认假设，实际系统与资源路径须由新机器预检确认。

## 1. 已有证据与当前目标

系统已有离线多步检索 Agent、SQLite 轨迹、SFT 起点、LoRA-GRPO、组内 advantage、固定 SFT reference、逐 transition 流式反传和精确 token 重放。最近一次完整回归为 57 项通过，不能代替新机器回归。

历史 v0.7.3 开发集 SFT/GRPO EM 为 13% 与 37%/45%/44%；封存的 300 题历史最终评测 EM 为 27.3%。这些结果来自旧实现，后续 KL、训练模式和上下文逻辑有变化，不可作为当前版本效果证明；最终集不重新读取。

v0.8.7 compact-v3 的固定 30 题 × 4 条件一步探针：

| 输入预算 | 预算失败 | 有损 case 比例 | reserved 峰值 | 可支持的结论 |
| --- | ---: | ---: | ---: | --- |
| 768 | 34/120 | 22.5% | 2.473 GiB | 该实现与预算组合不满足门槛 |
| 1024 | 8/120 | 20.0% | 3.248 GiB | 仍不满足门槛 |
| 2048 | 0/120 | 0% | 3.223 GiB | 纯推理能容纳固定探针 |

原始输入 P50/P95/max 为 721/1436/1883 token。成功生成的动作均为回答，未观察到新搜索。2048 下单次历史/双次历史/空检索/重复历史的 EM 为 33.3%/43.3%/6.7%/30.0%。预置搜索次数不等同于证据充分程度；此结果不代表自主 Agent 性能，也不证明训练显存安全。正文保留与 support 标题覆盖的有效样本分母须单列。

当前研究目标：在完整且可审计的上下文下，验证 GRPO 能否改善答案质量、有效搜索与停止决策。重复率为零时报告地板效应，不要求人为制造重复。

## 2. 开始训练前必须修改的部分（P0）

以下是核查到的真实代码缺口，既有迁移 TOML 仅是启动模板，不能视为全部验收已经实现。

| 部分 | 现状 | 新机器需要的修改与验收 |
| --- | --- | --- |
| 显存配置 | GRPOConfig 默认 7.4 GiB，CLI 直接构造默认配置；探针 gate 也写死 7.4 | 统一暴露 memory limit，所有命令从配置读取。24GB 默认 `min(21 GiB, 实测总显存×0.9)`；记录实测总显存，缺失或非有限数据不可通过 |
| 上下文 gate | trainer 使用旧 prompt_truncated，compact 无损缩短也会触发 >5% 停止 | 保留旧指标用于历史比较；新协议以有损 case≤5%、预算失败=0、历史删除=0、引用有效为 gate。单独统计无损变化和正文保留，不改写历史结论 |
| 检查顺序 | run_group 后先 update_group，再检查失败/上下文/显存 | 采样后先核验训练资格、输入与上下文；每轮更新前后检查数值、显存；失败立即持久化并停止 |
| 成功与保存 | checkpoint 在最终全部 gates 确认前保存，CLI 主要检查 status | status、退出码、gates.passed 一致；所有 gate 通过后才发布可评测 checkpoint，失败只留诊断产物；单独进程重载验收 |
| 方差与计数 | 有效 group 门槛使用已处理记录数；失败行可能被记为非法动作 | 所需 group 数按预注册任务数计算：5 题至少3组、10题至少5组。分开成功生成、预算错误、模型非法动作、基础设施错误 |
| 稳定性资格 | 启动仅检查历史报告 same_instance_passed 布尔值 | 新机器重做同实例重算，报告绑定模型/adapter/tokenizer、量化、dtype、kernel、代码与依赖；不盲用旧 Windows 报告 |
| 精确重放 | 保存 prompt IDs/mask，但 mask 与 current 长度校验需补齐 | 对采样/old/current/reference 核对完整 IDs、mask、hash、长度、有限值；非策略 token 不参与 loss；初次 ratio 接近1 |
| 历史编码与裁剪 | 当前 v3 仍使用预置 rank-cap 组合，并非穷尽逐句搜索；引用 source 未包含标题身份 | 固定历史实现与哈希；新修复用独立实现标识，source 身份包含 ID/标题/正文。失败只称“当前算法未装入”，不能称物理不可能；本轮不再探索摘要算法 |
| 运行护照 | requirements 较宽，部分模型路径写死；比较只验证部分 manifest | 新机器导出实际依赖与源码 commit，统一 model/adapter/data/cases 路径；核对权重内容哈希、代码及完整实验配置。比较只允许预注册字段不同 |

修改后运行全部回归，增加失败发生在更新前、无损变化不误触发 gate、缺失显存、mask 错误、checkpoint 门槛与超时测试。按新语义重新汇总旧轨迹时输出新报告，保留原报告。

## 3. 按门槛执行实验（P1）

### A. 新机器预检与推理复现

1. 保存 nvidia-smi、驱动/CUDA runtime、PyTorch/Transformers/PEFT/bitsandbytes、OS、RAM、GPU 实测显存与当前其他进程占用；确认 PyTorch CUDA 正常、模型与 adapter 哈希一致。
2. 检查本地忽略产物是否已迁移：原始 SFT、train/validation、固定 corpus、120-case 和必要参考报告。缺文件停止，不临时重建或换题；无需迁移封存最终集。
3. 首轮保持 4-bit、compact-v3 的冻结实现、2048 输入、128 输出、3步、BM25-v2、相同语料与 seed42，运行固定120-case；新机器结果和旧机器逐题比较，跨平台允许数值容差，动作差异逐条列出。
4. 检查模型错误/OOM/非有限值=0、预算错误/历史删除=0、有损≤5%、有效引用与可见支持文档保留≥95%。各条件 EM/F1 不低于同预算历史基线超过5个百分点；失败停止后续训练。

### B. 训练资格与代表性显存探测

1. 固定 train 前20题、corpus前100题，每题4条 sampled rollout，temperature0.7，参数不更新。按完整20题计算至少50%非零 reward 方差；独立重算 reward/advantage，检查输入、logprob、mask。探针预置历史不混入训练。
2. 使用采集轨迹验证固定 SFT reference、流式/聚合梯度与一次更新等价、冻结参数不变、LoRA-only 更新、checkpoint 重载。将训练最大输入长度与输出长度共同计入显存压力。
3. 以接近2048输入、多个transition、2轮policy epoch和完整optimizer状态探测训练峰值；如真实轨迹不足以覆盖上限，独立构造压力测试并标明不用于效果评估。无OOM、NaN/Inf且不越显存线才进入训练。

### C. 固定协议三 seed GRPO

冻结：SFT v0.5.2起点、固定train前缀/corpus、reward原权重、2048输入、128输出、max_steps3、group_size4、temperature0.7、learning_rate5e-6、clip0.2、KL0.02、policy_epochs2、grad_norm1.0，seed42/43/44。

三个seed串行各跑5×4；全部通过后，从原始SFT分别重启10×4，不接续smoke。每次独立数据库、报告与checkpoint。门槛沿用P0和资格验收，有效group分别至少3/5与5/10；任何失败停止扩展，不替换seed或降低门槛。

### D. 同协议质量评测

SFT与三个GRPO使用同一2048编码、精度、检索环境，运行固定100题开发集自主Agent和120-case历史探针。报告EM/normalized EM/F1、回答完成、搜索成本、非法/重复/max-step、证据覆盖、上下文损失、耗时与显存。报告全部seed及均值范围；置信区间按题配对bootstrap，四条件按题聚类。重复率仍为零时不宣称反重复收益，不挑最好seed作为结论。

## 4. 提高4090D吞吐（P2，正确性基线后）

性能目标为减少完成同一实验的时间；显存占满或瞬时GPU利用率不是验收目标。首轮保持串行、4-bit以控制变量。记录load/prefill/decode、old/reference/current forward、backward、optimizer和SQLite耗时，以及tokens/s、rollouts/min、有效updates/min、峰值allocated/reserved。

优化顺序，每次只改变一项：

1. **BF16对照4-bit**：0.8B可以作为测试候选，但完整训练峰值必须实测。相同固定轨迹测forward/backward耗时与数值；通过后重新冻结精度并重跑SFT基线，不能直接混合不同量化下的质量结论。
2. **scorer微批处理**：由batch1到2再到4，长度分桶、正确padding/mask，保持每条rollout等权、固定old/reference与相同optimizer步数。不能简单平均不同长度transition。用梯度等价测试确认后再测吞吐。
3. **真实批量采样**：当前模型锁会串行化generate，提高asyncio并发不会自动提升GPU吞吐。需要显式batch生成、每条seed/停止长度与padding正确记录；先保持group_size4，禁止跨参数更新混用旧策略轨迹。
4. **kernel/checkpointing**：确认新机器实际走的kernel；逐项测checkpointing开关，或安装与固定依赖兼容的优化kernel。Qwen混合架构并非全部耗时都能由FlashAttention解决；不默认安装flash-attn/causal-conv1d/flash-linear-attention。torch.compile最后考虑，并将编译时间单列。

每项优化须通过有限值、输入重放、loss/mask、初始ratio与固定轨迹梯度容差检查；保留开关与参考实现。稳定收益后再开展新训练，不在三seed中途换后端。

## 5. 交付、停止条件与下一步

新机器第一项交付是P0补丁与完整回归，其次是A/B资格报告，然后才是C/D的训练与质量结果。预计交付路径：独立run目录中的manifest、环境锁定文件、逐case/逐group指标、SQLite、重载结果，以及可版本化的中文摘要。

当前24GB配置仅作为待修订模板：其CLI尚未暴露显存线，直接启动仍沿用7.4GiB和旧截断gate。不要用删除gate的方式适配设备。预算/质量失败、OOM、非有限值、基础设施失败或必需审计字段未知都停止后续阶段；小样本失败同样记录成果，不自动扩大规模。

本提纲未执行远程操作或新增训练。超过1GB的单次下载或同一安装任务累计下载必须先说明用途、大小和位置并获用户确认；大小未知先查询。保留最终集封存、中文任务记录、独立输出目录与无自动重试规则。
