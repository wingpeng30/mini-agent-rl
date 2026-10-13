# Goal Q：从 Goal J 训练真实搜索模型

状态：`COMPLETED_NO_NEW_MODEL_PROMOTED`。Q0–Q4 已按冻结判据执行。两条从 Goal J 出发的路线均完成适用阶段；轨迹 SFT 后 RL 因预定信号门槛未达到而跳过。唯一推荐的现有产品模型是 **Goal P 的 RL-only adapter**，它在本次真实搜索统一入口达到 19/32，回归集 7/16；本轮新训练模型最高 16/32，未触发 seed 43 复现。

## 真实搜索结果

所有模型使用相同的 96 篇共享文档、BM25-v2、`direct-v090` 提示、`search.text → query` 兼容、greedy、每题最多 4 次模型调用和 3 次搜索。模型只看到问题和工具返回；金标及支持 ID 仅用于训练奖励或事后评分。两个开发集各 16 题，均为已暴露开发集。

| 模型 | 起点与训练 | 主要开发 EM | 回归开发 EM | 合计 EM | F1 | 找齐两侧证据 | 无效动作 | 搜索数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Goal J | 历史 SFT | 6/16 | 7/16 | 13/32 | .5802 | 17/32 | 3 | 29 |
| Goal P RL-only | 历史 v118→RL | **12/16** | 7/16 | **19/32** | **.7346** | **32/32** | 0 | 32 |
| Goal Q direct | Goal J→真实搜索 RL | 5/16 | 7/16 | 12/32 | .5698 | 14/32 | 0 | 32 |
| Goal Q SFT | Goal J→轨迹 SFT | 9/16 | 7/16 | 16/32 | .6948 | **32/32** | 0 | 60 |

历史 RL-only 与 Goal J 的回归正确数持平，总正确数多 6 题，因此按预设基线规则成为当前可用模型。它一题一次搜索，并在这个由已知题目的证据构成的文档库中，32 题均找到两侧支持。轨迹 SFT 也做到 32/32 双侧召回，但平均每题 1.875 次搜索，最终正确数少 3 题。这说明本轮 SFT 改善了搜索覆盖，却没有取得更好的最终答案。

固定证据任务沿用 Goal P 的同权重统一评价：Goal J 主要开发 10/16、回归 9/16；历史 RL-only 为 12/16、7/16，两者合计都是 19/32。固定证据与真实搜索的输入不同，不将其用于本轮产品模型选择。见 `../v133-goal-p-system/p3-training/evaluation-summary.json`。

## 训练执行与分支决定

- Goal J 的 16 题×4 条完整真实搜索采样产生 64 条轨迹：15 题有成功/失败混合组，1 题全失败；达到直接 RL 的 4 题门槛。
- 监督示范共 16 条：7 条从正确且找齐两侧证据的采样轨迹选择，9 条由训练题题面中的实体标题提出查询、实际调用检索器，再接冻结训练答案。示范没有加入开发题；工具返回没有手工替换。
- 直接 RL 从 Goal J 开始，2 遍、32 组、128 条新轨迹、17 次非零方差更新。终局严格 EM 为二元奖励，优势应用于整条轨迹的 assistant 动作；全对/全错组跳过。冻结 reference 始终为 Goal J。
- 轨迹 SFT 从 Goal J 独立开始，16 条示范×2 遍、32 次更新，监督 assistant 搜索和回答动作。SFT 后 64 条新采样中，13 题全对、3 题混合、0 题全错。混合组少于预设的 4 题门槛，因此没有启动 SFT→RL 更新。
- 直接 RL 的 192 个 LoRA tensor 变化，base/reference 不变，重载 logprob 最大差 0。SFT 同样有 192 个 LoRA tensor 变化、base 不变，独立重载通过；重载示范的 loss 有限。完整组记录、曲线和 checkpoint 保存在本目录。
- 三个适用候选统一评价后，均未超过既有可用模型 19/32。没有胜出配方，因此按预设规则不启动 seed 43 复现。累计 GPU 阶段占用 **597.912 秒**，低于 10800 秒上限。

## 逐题变化及解释

历史 RL-only 相比 Goal J 新答对 9 题、丢失 3 题：主要开发新增 6 题且无丢失；回归开发新增 `g009`、`g013`、`g014`，丢失 `g000`、`g008`、`g011`。净回归正确数保持 7/16。轨迹 SFT 相比 Goal J 新答对 4 题、丢失 1 题；直接 RL 新答对 2 题、丢失 3 题。完整 ID 见 `audit.json`，原始答案与实际工具结果见各 `eval-*.jsonl`。

历史 RL-only 相比本轮轨迹 SFT 的部分优势来自答案粒度：例如 `film director` 对 `director`、`documentary film` 对 `documentary`、`novelist` 对 `writer`；另有 `g013` 的 `film director` 对 `writer`，属于实质内容差异，不能用放宽字符串匹配消除。本轮不修改冻结金标。结果支持“检索覆盖与最终回答质量要分别优化”，但只有单次已暴露开发评价，不能推断开放域泛化或训练方法的稳定优劣。

## 验收与复现

本地新增测试核对 Goal P 与 Goal Q 的搜索循环、工具结果和停止语义；CPU 审计重新计算 4 个模型×32 题的 EM/F1、双侧支持和调用预算，全部一致。已用最终选中 adapter 在远端运行一次真实交互命令。统一评价和演示使用同一 Goal Q 搜索循环。运行中没有 GPU/数值失败，没有覆盖历史结果。

AutoDL 工作目录：`/root/autodl-tmp/mini-agent-rl`。复现完整训练需换用新的隔离输出目录；已有 Goal Q 目录拒绝覆盖。下列命令可直接使用当前结果：

```bash
cd /root/autodl-tmp/mini-agent-rl
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_q_selected.py
PYTHONPATH=src .venv/bin/python scripts/demo_v133_goal_q_selected.py --question 'Paul Young and David Draiman, share which mutual occupation?'
PYTHONPATH=src .venv/bin/python scripts/run_v133_goal_q_search.py summarize
```

本地 CPU 逐条复算及绘图：

```bash
PYTHONPATH=src:scripts python scripts/analyze_v133_goal_q_search.py
```

冻结配置、资源台账与选择决定分别见 `config.json`、`state.json`、`decision.json`。`training-curves.png` 展示 SFT loss 与直接 RL 每组平均终局奖励；原始轨迹包含每步输入/生成 token、工具返回、结束原因和调用信息。该语料由同一批题目的冻结证据构成，开发集已经被多轮使用；当前选择仅适用于这个明确范围，仍需独立数据才能检验对新问题的可靠性。
