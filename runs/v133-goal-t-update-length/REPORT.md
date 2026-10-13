# Goal T：修订标签下的短预算训练

状态：`COMPLETE_PRODUCT_UPGRADED`。预设的两个短预算模型均通过产品门槛：seed 42 第 32 次更新为 20/32、回归 7/16；seed 43 为 21/32、回归 8/16。当前产品入口选择 **seed 42 的第 32 次 checkpoint**。Goal P RL-only 基线在本轮复现为 19/32、回归 7/16。

## 设计与工程核查

两个 seed 均独立从 Goal P RL-only 起步，使用 Goal S 冻结的两条修订标签、16 道真实搜索回答前输入和固定题目顺序。每个 seed 连续训练两遍、64 次更新；第 32 次和第 64 次保存 checkpoint，中间不重置优化器。四个 checkpoint 均通过 LoRA 变化、base 不变、独立重载 logprob、随机数与优化器状态核查。基础模型实际权重、起点、数据、代码、软件版本和数值环境哈希见 `config.json`。主运行 GPU 进程共 575.1 秒；另有选中模型 CLI 演示核查约 12.5 秒（包含 SSH 启动，见 `manual-gpu-audit.json`），估计总占用约 587.6 秒。**没有 GPU 时间上限或阶段超时**。

## 同条件真实搜索结果

| 模型 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 双侧证据 | 无效动作 | 搜索数 | 平均延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| incumbent | 12/16 | 7/16 | 19/32 | 0.7346 | 32/32 | 0 | 32 | 1.50s |
| 42-32 | 13/16 | 7/16 | 20/32 | 0.7786 | 32/32 | 0 | 32 | 1.57s |
| 42-64 | 12/16 | 6/16 | 18/32 | 0.7208 | 32/32 | 0 | 32 | 1.54s |
| 43-32 | 13/16 | 8/16 | 21/32 | 0.7891 | 32/32 | 0 | 32 | 1.55s |
| 43-64 | 13/16 | 8/16 | 21/32 | 0.8021 | 32/32 | 0 | 32 | 1.57s |

两 seed 的短预算模型均在严格 EM 上超过当前模型，且回归不下降；长预算 seed 42 降到 18/32、回归 6/16，长预算 seed 43 保持 21/32、回归 8/16。因此结果支持缩短训练能减少**部分**回归，但效果并非两个 seed 都随更新次数单调变化。

incumbent→42-32：新增 2 题（5ae30de055429928c42395cf, g008），丢失 1 题（g009）。

incumbent→43-32：新增 2 题（5ae30de055429928c42395cf, g008），丢失 0 题（无）。

42-32→42-64：新增 0 题（无），丢失 2 题（5a7789215542997042120a99, g014）。

43-32→43-64：新增 1 题（g010），丢失 1 题（g009）。

seed 42 的第 32 次仍正确回答鸡尾酒共同成分 `lemon juice`，第 64 次改成只有一侧包含的 `honey`。seed 43 两个 checkpoint 均保留 `lemon juice`。`director` 与 `film director` 的训练标签修订已在 Goal S 冻结，本轮不重新选择标签。

## 语义边界与历史复核

基线与四个 checkpoint 共 160 条固定上下文输出，逐题与各自产品输出一致；评分按冻结 EM/F1 逐条复算。语义分类沿用 Goal S 预先冻结的规则，对同一问题、实际证据和答案只判一次。分类属于代理辅助审查，不是独立人工金标准。各模型“缺少必要限定／双侧证据不支持或事实错误”计数：当前模型 2/6；42-32 3/6；42-64 3/5；43-32 3/6；43-64 3/4。完整依据见 `semantic-review.json`。

短预算相对基线新增的严格正确主要来自回答格式与粒度：`both are public use airports → public use airport`、`both are dogs → dog`。原回答已经表达了正确事实；本轮**尚未显示新事实问题上的稳定增益**。两短预算模型也把多职业问题的回答收窄为 `film director`，遗漏共同的 `screenwriter`。因此产品升级遵循预先冻结的严格 EM 门槛，不能解释成语义正确率已提高。

Goal T 的第 64 次模型与 Goal S 同 seed 的 64 次模型并非字节相同：seed 42 各为 18/32 和 18/32，seed 43 各为 21/32 和 20/32。两次训练的损失从第 2 次／第 2 次更新起已出现差异，均早于第 32 次保存；不能把历史差异归因于中间保存。CUDA 日志同时报告部分算子非严格确定性。没有因此重跑或事后改选 checkpoint。

当前两套开发题已经多次暴露，这个小样本改进不代表新问题泛化。最终推荐模型只在预设产品门槛下胜出；更强的事实性改进需要另行实验。

## 复现与产品入口

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src:scripts .venv/bin/python scripts/run_v133_goal_t_update_length_v2.py summarize
PYTHONPATH=src:scripts .venv/bin/python scripts/demo_v133_goal_t_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
```

完整新运行需要指定空的 `GOAL_T_OUT`，先执行 `prepare`，再执行 `run`。运行器拒绝覆盖已开始的结果。`training-curves.png`、逐题输出、检查点哈希和资源台账均保留在本目录。
