# v133 Goal E — SFT 无收益诊断

**状态：完成。决定：停止重复现有 25 题 SFT 配方。** 未发现需要修复的 adapter 加载、监督前缀或生成评估缺陷。现有权重确实改变计算，也提高了训练题拟合；在同一套对齐开发题上没有可测回答收益。该结论只适用于本轮小规模、合成搜索历史的诊断。

## 冻结输入与执行

- 训练 25 条、开发 16 条，分别 SHA-256 `b4d38e1e99886e3a254a6e881a28b7a36194f7f185b686a64dc7dbdf449ee9df`、`23cde9d8f25376d469c91128742ca647fc4d84afe04b2009e38e571198780e58`。
- v118、Goal C、Goal D adapter 权重 SHA-256 分别为 `ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5`、`0a80d82918e301cac05f0cc2892a7ccd007138996e121549b95dbcbf7f8919c6`、`b8782a1a36ddaa44e2a2d7446763b7e50ae40865aae326366d3f0e50b9fc94af`。
- 模型及 tokenizer 文件逐一与 Goal C 冻结清单核对；AutoDL 使用 torch `2.5.1+cu124`、Transformers `5.17.0`、PEFT `0.20.0`。生成使用 seed 42、thinking 关闭、贪心、最多 128 个新 token。
- 训练题沿用原始训练消息；开发题使用此前冻结的合成搜索历史。没有访问封存测试集，没有新训练或扩池。
- AutoDL 两个 GPU 阶段分别耗时 57.223 秒、30.430 秒，合计 87.653 秒，低于两小时硬上限；首两条投影分别为 259.06 秒、152.12 秒。两次均正常完成，无运行中诊断进程。

## E1：上下文和标签

全部 41 条样本的生成输入 token 与监督目标之前的 token 前缀一致；目标是最后一条 assistant answer，标签连续，终止符被监督，问题及工具证据可见，目标内容未进入生成输入。长度在 2176 token 限制内，未发现裁剪。唯一的模板差异是一条训练问题末尾空白被 `strip`；CPU 证据检查按实际模板内容比较，训练编码未修改。

被监督的 521 个 token 中，答案值 151 个，JSON 格式及结束标记 370 个。后者占 71.0%，且三个模型在格式 token 上都已达到 100% token 准确率。因此总 loss 下降有相当部分来自格式概率提高，不能代表答案能力同幅提高。历史 validation loss 使用原开发上下文；下表另行使用与生成相同的对齐开发上下文。

## E2：adapter 实际作用

三份权重均加载为活动 `default` adapter，处于 eval 模式，96 个 LoRA 模块匹配。固定训练锚点 0、8、16、24：启用与禁用 adapter 的末位置 logits 最大绝对差为 10.83–14.77；相同输入重复前向差为 0。v118 相比首次 SFT 及 Goal D，在共同 top-32 logits 上已有可测变化。禁用 adapter 的对照是原始基座，研究主对照仍是 v118。

按 `alpha/r × B@A` 计算全部 96 个模块的有效 LoRA 权重，首次 SFT 与 Goal D 相对 v118 的有效权重 Frobenius 差分别为 v118 有效权重范数的 1.07% 和 2.57%。这与生成训练题答案变化一致。BF16 下差异清晰，高精度补测的触发条件未出现。PEFT 对旧配置的空 `kasa_config` 字段给出兼容警告，但已加载 adapter 的前向作用和结果可测；本轮没有据此改动依赖。

## E3：训练拟合与开发迁移

| 模型 | 训练有效 / EM / F1 | 开发有效 / EM / F1 | 训练答案 token NLL | 开发答案 token NLL | 训练格式 token NLL |
|---|---:|---:|---:|---:|---:|
| v118 | 25/25 · 20/25 · 0.8133 | 16/16 · 12/16 · 0.8417 | 0.2291 | 0.1404 | 0.0152 |
| 首次 SFT | 25/25 · 21/25 · 0.8533 | 16/16 · 12/16 · 0.8417 | 0.2079 | 0.1378 | 0.0070 |
| Goal D | 25/25 · 23/25 · 0.9200 | 16/16 · 12/16 · 0.8417 | 0.1701 | 0.1385 | 0.0047 |

NLL 为相关 token 的加权均值；答案值与格式 token 分开计算。开发答案 token 的准确率三组相同，按题平均约 0.878；开发格式 token 准确率均为 1.0。Goal D 相比 v118 的训练题新增正确为索引 3、17、22，没有丢失，保留 20 题；首次 SFT 仅新增索引 17。训练题索引 6、20 在 Goal D 下仍严格错误。**v118 已答对 20/25，现有训练集的严格 EM 改善空间最多 5 题**，且这一上限只描述这 25 题。

既有开发结果未保存精确 prompt token 哈希，因此重新生成 16 × 3＝48 条。三组在全部 16 题的原始回答完全相同：新增正确 0、丢失正确 0、保留正确 12。训练拟合变化未迁移到本组开发题。

四个开发错误仍为：索引 0 把球队答成联盟；索引 2 把共同类别 `rock` 答成 `folk rock`；索引 3 把成立年份先后反过来；索引 15 省略人名中间名。逐 token 候选概率记录见 `dev-generation.json`。索引 2、3、15 的历史回答在其各自监督前缀下有更低的答案 token NLL；索引 0 的正确候选平均 NLL 反而更低，但贪心输出仍为错误答案。这些候选长度和条件前缀可能不同，平均 NLL 只能用于定位，不能当成完整序列偏好或因果解释。索引 15 可能属于别名问题，本轮保持原严格 EM 计分，不追改成绩。

25 条训练问题以实体、属性和两跳定位为主，没有与开发索引 2 的共同类别判断或索引 3 的两个日期比较完全对应的训练形式。索引 0 仍是实体属性定位却答错，因此“训练覆盖不足”只是部分解释，不能概括全部四题。

## E4：结论与下一步

1. **加载／上下文缺陷：** 本轮未发现。CPU 检查与实际前向均通过，故无代码修复或修复后复评需要。
2. **训练是否生效：** 生效。有效权重、logits、训练题生成及训练答案 NLL 均给出证据。v118 原有 20/25 正确使当前训练集可测余量很小。
3. **是否迁移：** 这 16 条对齐开发题没有回答收益。开发答案 NLL 有微小波动，未改变任何生成答案；不能据此承诺更多轮次会有效。
4. **最小后续研究：** 先从现有合格候选中按错误类型盘点是否有明确的共同类别、日期比较及关系属性案例，形成一份不含开发题标签的、少量且独立的训练／评价建议。若现有合格案例不足，再单独提出来源筛查与盲审预算。只有目标覆盖和独立评价集都明确，才提出新 SFT；不继续追加当前 25 题的训练轮次。

开发题没有新变化，匿名双审触发条件未出现。原始 Goal C、Goal D 结果均保留；本轮不是端到端检索成绩或独立泛化结论。

## 复现与追溯

远端工作目录：`/root/autodl-tmp/mini-agent-rl`。CPU 预检与两个 GPU 阶段分别执行：

```bash
.venv/bin/python scripts/diagnose_v133_goal_e.py --mode preflight --model-path /root/autodl-tmp/qwen_08b --output-dir runs/v133-goal-e-diagnostic/remote-preflight
timeout 7200s .venv/bin/python -u scripts/diagnose_v133_goal_e.py --mode infer --model-path /root/autodl-tmp/qwen_08b --output-dir runs/v133-goal-e-diagnostic/remote-run
timeout 7100s .venv/bin/python -u scripts/diagnose_v133_goal_e.py --mode dev-infer --model-path /root/autodl-tmp/qwen_08b --output-dir runs/v133-goal-e-diagnostic/remote-dev-generation
```

首轮 runner SHA-256 `b0120d177e792d41e5d2410a4da0e3b02915746d8a530e25eb534dd2f44d451b`；增加开发题复评模式后的 runner SHA-256 `2e327b5134f47d5fb1e9af3ebd62c5e8ec9c674b84014acd2a2ef78bb2ba3dc`。汇总 runner SHA-256 `0bf5dc5799a5fb1e9af1b94eeba84696f0ecffa235b2b86c430c1ea5d4826a40`。

本地与远端一致：`diagnostic.json` SHA-256 `9952b838042d803d6ceef4e9a0e509a692d1c29e06a7da1c8bf3093db2ae6f89`；`dev-generation.json` SHA-256 `700360e912be4c7e2c0a1d53a5e39a92d2d74b1c3abf98d3aeb2e1bc19592418`。本地 `summary.json` SHA-256 `a73923a84f33d4e24021bf274697949769ae4a2d113509181809743d3ee95f3d`。逐题原始指标、候选概率和活动 adapter 状态均保存在上述 JSON 文件。
