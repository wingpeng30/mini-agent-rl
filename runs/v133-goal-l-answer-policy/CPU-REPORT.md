# Goal L：无卡准备报告

## 当前状态

`cpu_ready=true`、`gpu_required=true`、`inference_complete=false`、`review_complete=false`。Goal L已创建并保持进行中，等待用户开启GPU实例后执行推理。本阶段没有加载模型权重、生成、训练或提取新题。

## 冻结输入与检查

- 主要诊断集：Goal J新评价16题；探索回归集：Goal G 16题。两者此前均参与过研究决策。
- 模型：v118 adapter与Goal J最终adapter；adapter SHA256分别为 `ea894d235b6779bc8d15d1327a953773f946c347dfa7fa55ee3d2c2d5bd4a0a5`、`21b24dc960c3915e9db89626358b2271d8e37ff8b6e0205d3ab40ddb5eca0fc2`。
- 输入评价集哈希与历史冻结值相同；64条P0历史轨迹逐题绑定到相同集合、问题和输入token哈希。
- 三种提示共96个输入通过现有监督编码器检查，包括前缀一致、目标assistant、结束标记、证据可见和长度限制。最长提示564 tokens；没有截断。
- 无卡环境版本：torch 2.5.1+cu124、transformers 5.17.0、PEFT 0.20.0、tokenizers 0.23.2。历史GPU为NVIDIA GeForce RTX 4090 D。
- CPU回归检查确认合法search与坏JSON分别计数、哈希错配会拒绝、重复配对会拒绝、严格EM评分保持原规则。

## 冻结条件与运行方式

新增生成预算为128条：v118和Goal J adapter各运行P1、P2，每条件32题。P0直接复用64条历史轨迹，不重跑。

```bash
cd /root/autodl-tmp/mini-agent-rl
.venv/bin/python scripts/run_v133_goal_l_answer_policy.py prepare
.venv/bin/python scripts/run_v133_goal_l_answer_policy.py run
.venv/bin/python scripts/run_v133_goal_l_answer_policy.py summarize
```

GPU入口仅在CUDA可用、输入与代码哈希未变化且运行环境与冻结版本一致时启动；要求GPU型号与历史RTX 4090 D相同。推理硬上限7200秒，首两条正式输出用于估时。没有训练入口调用。

复现细节、逐题CPU检查、提示、版本与哈希见同目录 `freeze-manifest.json`、`runtime-config.json`、`cpu-checks.jsonl` 和 `prompts-frozen.jsonl`。
