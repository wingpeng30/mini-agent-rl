# 本地模型基础框架（v0.3.1）

本版本针对 Qwen3.5-0.8B 完成本地推理基线适配。官方模型加载路径为 `AutoProcessor` + `AutoModelForMultimodalLM`；当前 Transformers 固定到 commit `5474a55e920f358d8382f3ecd3377edca979baa1`（构建版本 `5.18.0.dev0`）。

## 本次成果

本阶段完成了下载模型之前的代码准备，没有下载 Qwen 权重，也没有执行 GPU 训练。核心新增项包括：本地模型配置、Transformers 懒加载适配器、硬件预检、严格动作协议、token 记录以及 SFT 数据边界。

## 运行边界

`LocalModelConfig` 的默认模型为 `D:\qwen_08b`，并启用 `local_files_only`。创建配置、导入 Python 包、运行 FakeModel 测试和执行 `local-check` 都不会访问网络。模型权重必须预先存在于该目录。

Agent 只能生成以下动作之一：

```json
{"action": "search", "query": "检索词"}
```

```json
{"action": "answer", "answer": "最终答案"}
```

该限制建立了可训练的策略动作边界。SearchTool 返回的 observation 属于环境状态，不是模型生成 token，不应参与 policy loss。

## RTX 4060 8GB 建议

首次冒烟测试使用 FP16、单条 rollout、最多 256 个新 token。0.5B 模型通常可以直接放入 8GB 显存，暂不依赖 4-bit 量化。等 FP16 推理稳定后，再为 SFT/GRPO 安装 PEFT、TRL 和可选量化依赖。

## 后续路线

1. 安装与 CUDA 匹配的 PyTorch 和本地模型可选依赖。
2. 执行 `local-check`，确认 CUDA、GPU 名称和显存。
3. 首次加载 Qwen 权重并完成一条 rollout 冒烟测试。
4. 用 held-out benchmark 记录 Base 模型基线。
5. 生成并审查 SFT 数据，训练 LoRA adapter。
6. 接入 GRPO，以同任务 rollout group 计算 advantage。

当前版本仍是 RL-ready 推理与数据基础设施，不包含 SFT/GRPO 权重更新。Qwen3.5 的原生上下文很长，但本项目默认限制为 2048 token，并将单 GPU rollout 并发限制为 1。

## 本地验收记录

在 RTX 4060 Laptop 8GB、PyTorch `2.14.0+cu126` 环境中，`local-smoke --model-path D:\qwen_08b` 已成功加载 473 个权重分片并完成 `SUCCEEDED` rollout，SQLite 轨迹正常写入。该次基线答案未命中标准答案，奖励为 0.000，这是后续 SFT 前需要记录的 Base 模型结果。

Transformers 报告的 `causal_conv1d` 和 `flash-linear-attention` fallback 是性能提示，不是功能错误；后续可在确认兼容性后再安装优化 kernel。
