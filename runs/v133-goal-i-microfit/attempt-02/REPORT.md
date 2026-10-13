# Goal I — 微型拟合诊断报告

日期：2026-09-28（AutoDL 实例 SSH 端口 25320）

## 结论

Goal I 的一次固定预算微型 SFT 已完成。4 条经本轮新双审、且均已在 v118 训练中暴露的共同类别案例，通过 CPU 编码检查后，使用 v118 adapter 做 20 次更新。训练题严格 EM 从 0/4 提高到 4/4，说明这批具体案例能被当前监督与更新流程拟合；这不是泛化证据，也不能证明扩大训练后会提升独立评价成绩。

Goal G 的 16 题是已参与研究决策的探索评价集。同期 v118→microfit 的严格 EM 为 3/16→5/16，均值 F1 为 0.3497→0.5125，协议有效率为 14/16→16/16。严格新增正确 4 题（g003、g005、g015、g017），严格丢分 2 题（g009、g013），保留正确 1 题（g012）。新增正确题的官方答案及上下文与 Goal G 已双审包完全一致，故可复用其两张合格票。g009 的新答为“Democrat”，g013 为“director”；DeepSeek 与隔离 Luna 对完全相同上下文中的这两个新答均判为证据支持。原有 EM 标准不变，所以这两题仍计作严格 EM 丢分，但目前没有证据支持把它们解释为语义错误。

## 执行与冻结记录

首次运行在完成20条 v118 基线生成后、训练开始前，因运行器向 `train_sft()` 传入不支持的 `validation_not_run` 参数而退出；没有启动训练。用户明确允许修复后重跑。修复后在独立 `attempt-02/` 重跑，保留失败现场，最终成功完成。未重试训练阶段或追加训练。

- 远端输出：`/root/autodl-tmp/mini-agent-rl/runs/v133-goal-i-microfit/attempt-02/`
- 4 条训练样本和16条探索评价题的冻结上下文、审核包、输入哈希及 CPU 预检记录均保留在该目录。
- GPU：NVIDIA GeForce RTX 4090 D；seed 42；BF16；贪心解码；最多生成128 tokens；无 thinking。
- v118 起点路径显式指定；训练 20 步，microbatch 1、梯度累积4、AdamW、LR 1e-4、weight decay 0、grad clip 1.0、线性衰减、warmup 1步。
- 基线、训练和复评总用时 56.574 秒，低于7200秒硬上限；实际训练段约29.2秒。总计正式生成40条（20条v118基线＋20条最终adapter复评），并完成20次更新。
- 训练日志 loss 从第1步1.0108降至第20步0.0001289。训练样本答案内容 token 的汇总准确率由0.333变为1.0，NLL由2.9528降至0.000713；格式/结束 token 准确率由0.8125变为1.0，NLL由0.7606降至0.000052。目标共54 tokens，其中48个为格式/结束 token；答案内容部分也有明显改善，不能把下降归因于格式 token 单独驱动。
- 最终 adapter 共192个张量全部变化；权重 SHA256：`9e470ce04f22bade3dcb47bd1150d953559a2eec395d5ffed2e91a77fc72905d`，相对参数差 L2 0.901306，基准范数16.1513，相对变化5.5804%。
- 环境版本与冻结记录一致：torch 2.5.1+cu124、transformers 5.17.0、PEFT 0.20.0、tokenizers 0.23.2、safetensors 0.8.0、datasets 5.0.1、accelerate 1.15.0。
- Goal I 四题审核均由 DeepSeek 与隔离 Luna 独立批准。新增正确题复用 Goal G 对完全相同题面、答案及上下文的合格双审；g009/g013 新预测各有一组匿名独立双审支持。详细票据和绑定索引见本目录审核产物。
- 最终模型加载成功，adapter 激活且匹配96个 LoRA 模块。PEFT 提示忽略 adapter 配置中的未知 `kasa_config` 字段；运行仍成功。保存配置内记录了旧 Windows 基座路径，但复评通过显式 AutoDL 基座路径加载，未触发默认路径回退。

## 解释限制

Trainer 日志在结束时出现 `eval_loss`，因为运行接口的 evaluation path 指向同一个4题训练文件；该值是训练样本上的重复诊断，不是留出验证。它未用于 checkpoint 选择。本轮没有独立验证集，Goal G 16题也已用于研究决策，不能称为独立确认实验。其余发生变化但未影响决策的预测没有逐项语义审核，因此不对它们作语义正确性声明。

本轮不读取 final-test 或封存独立100题，不扩池，不运行第二轮训练、GRPO 或超参数搜索。

## 决策建议

当前结果足以排除“这4条监督样本完全无法拟合”这一解释；不支持在同样4题上继续追加训练。若继续研究，应先围绕更广的目标能力准备合格且来源/实体隔离的训练案例，并保留真正未用于训练选择的评价题。Goal G 的小样本变化是探索性信号，尚不能证明迁移收益。扩池或新训练仍需单独授权。

## 关键复现产物

- `summary.json`：主要数值结果与冻结哈希。
- `run-state.json`：GPU、时限、计划轨迹数、计时及完成状态。
- `training-config-effective.json`、`training-metrics.json`：实际训练配置及逐步日志。
- `trajectories.jsonl`：逐题原始输出、协议状态、EM/F1及输入哈希。
- `target-token-scores-v118.json`、`target-token-scores-microfit.json`：答案和格式 token 分项概率统计。
- `freeze-manifest.json`、`execution-preflight.json`：输入/代码/环境冻结与启动门禁。
- `review-packets.jsonl`、`luna-votes.json`、`deepseek/`：匿名双审记录。
- `post-eval-review-packets.jsonl`、`post-eval-luna-votes.json`、`deepseek/i-regression-01/`、`result-review-binding.json`：结果变化审核及其逐题绑定。
- `trained-adapter-hashes.json`：最终 adapter 哈希和变化量。

审计复核命令（本地仓库根目录执行）：

```powershell
python scripts/audit_v133_goal_i_results.py --run-dir runs/v133-goal-i-microfit/attempt-02
python scripts/check_v133_goal_i_preflight.py
```

修复后的 summary 汇总读取器在 GPU 计算结束后更新；这项更新不改变训练或推理。执行期冻结代码哈希以 `execution-preflight.json` 为准。
