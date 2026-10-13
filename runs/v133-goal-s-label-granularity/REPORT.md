# Goal S：训练答案粒度对真实搜索的影响

状态：`COMPLETE_NO_PRODUCT_UPGRADE`。仅修订两条训练标签，原始数据和产品搜索设置保持冻结。修订标签在两个 seed 中都优于各自原标签对照，但只有 seed 43 超过当前最佳 19/32；seed 42 为 18/32、回归 6/16，因此按预设双 seed 门槛保留 Goal P RL-only。

## 冻结干预与工程核查

16 道训练题的真实可见证据逐题审查后，仅把 SWMRS/We the Kings 的 `band` 改为 `rock band`，把 Jacques Tourneur/Djibril Diop Mambéty 的 `director` 改为 `film director`。两侧文档均直接支持新增限定。`documentary` 和 Josh Trank/Jonathan Frakes 等存在表述或证据边界的题保留原标签。逐题修订的证据来自训练题，详见 `labels.json`；检验答案粒度这一假设本身受到此前已暴露开发题失分的启发，结果不是独立验证。

起点、tokenizer、96 篇语料、16/32 题、Goal R 实际输入 token、Goal Q 搜索入口和基础模型**实际权重文件**哈希均写入 `config.json`。每臂独立从 Goal P RL-only 加载，两遍、每题每遍两次，64 次更新；除两条答案目标外，输入 token、顺序、seed、优化设置一致。四个 checkpoint 均通过 base 不变、LoRA 变化和独立重载 logprob 核查。GPU 进程合计 710.8 秒，低于 7200 秒。

## 同条件真实搜索

| 模型 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 双侧证据 | 无效动作 | 搜索数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 12/16 | 7/16 | 19/32 | 0.7346 | 32/32 | 0 | 32 |
| 42-a | 9/16 | 6/16 | 15/32 | 0.6937 | 32/32 | 0 | 32 |
| 42-b | 12/16 | 6/16 | 18/32 | 0.7208 | 32/32 | 0 | 32 |
| 43-a | 10/16 | 6/16 | 16/32 | 0.7042 | 32/32 | 0 | 32 |
| 43-b | 12/16 | 8/16 | 20/32 | 0.7708 | 32/32 | 0 | 32 |

seed 42：修订标签相对原标签新增 3 题、丢失 0 题；新增为 `5ae30de055429928c42395cf, 5ac259bc5542992f1f2b386a, 5ac169c25542994ab5c67d1a`。seed 43 新增 4 题、丢失 0 题；新增为 `5ae30de055429928c42395cf, 5ac169c25542994ab5c67d1a, g010, g014`。两次均恢复了 `director → film director` 相关题；机场题也从 `public use` 恢复为 `public use airport`。后者不是训练题的直接标签替换，说明干预存在跨题迁移，但数据量不足以证明稳定因果机制。

相对当前最佳，seed 42 修订臂新增 `5ae30de055429928c42395cf, g008`，丢失 `5a7789215542997042120a99, g009, g014`。seed 43 新增 `5ae30de055429928c42395cf, g008, g010`，丢失 `5a7789215542997042120a99, g009`。两臂均把鸡尾酒共同成分 `lemon juice` 错答为仅一侧具有的 `honey`；seed 43 仍有这一事实错误。`Democratic` 与 `Democratic Party` 等严格 EM 差异须单独看待。

## 机制诊断与训练拟合

基线保存的 32 道回答前上下文在各模型上重放，160 条固定上下文输出与各自产品输出逐题相同。训练拟合按原标签／修订标签分别为：42-A 15/16、13/16；42-B 13/16、15/16；43-A 14/16、14/16；43-B 12/16、14/16。训练拟合口径受两条不同目标影响，不能代替开发集产品成绩。

按训练前冻结的语义规则，模型名隐藏后审查独特的“问题—实际证据—答案”组合，再映射回模型。分类是代理辅助解释，**不是独立人工金标准**；每条理由和证据见 `semantic-review.json`。严格 EM 保持原口径。各模型“缺少必要限定／无双侧支持或事实错误”计数：当前最佳 2/6；42-A 6/6；42-B 3/6；43-A 5/5；43-B 3/5。这一审查不会追溯触发复现：seed 43 已由产品总 EM 上升且回归不低于 A 的预设门槛触发。

## 结论与使用

这组配对结果支持：**原训练标签的粒度至少解释了部分 SFT 退步**。两处小修订在两个 seed 都减少了部分过粗回答，使修订臂比同 seed 原标签臂多答对 3 和 4 题。它没有稳定改善产品整体：seed 42 低于现有 19/32，回归也下降；seed 43 达 20/32、回归 8/16，但按预先冻结的双 seed 规则不能单独发布。当前唯一推荐和演示模型继续是 Goal P RL-only。

这些都是反复暴露的开发题，且 CUDA 日志提示部分算子不能严格确定性复现。两 seed 的差异及有限样本使结果不能代表新问题泛化。没有根据 seed 43 的较好成绩追加训练或改动标签。

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_s_label_granularity.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_s_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

完整新运行须指定空的 `GOAL_S_OUT`，先复制冻结的 `labels.json`，再运行 `prepare` 与 `run`；历史运行目录拒绝覆盖。`run` 在产品分数无法直接判断复现时会等待语义审查文件，再以相同命令继续。`training-curves.png`、全部输出和预算台账保留在本目录。
