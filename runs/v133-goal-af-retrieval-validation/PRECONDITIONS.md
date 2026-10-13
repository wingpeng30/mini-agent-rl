# Goal AF 实验前提

配置 SHA-256：`2f0d0ff50a354ef0f47ea15a8ccdd55c04219872dfb10824c2723e88d0317b7f`

GPU 输出尚未生成。运行前需要用户确认以上哈希对应的完整清单。

## 模型与运行环境

- 基础模型：`/root/autodl-tmp/qwen_08b`；BF16，单卡，不量化，不训练。
- 已发布 adapter：`/root/autodl-tmp/mini-agent-rl/runs/v133-goal-z-multihop-sft/seed42-trajectory/adapter-final`
- adapter SHA-256：`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`
- 基础模型实际权重 SHA-256：`{"model.safetensors-00001-of-00001.safetensors": "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"}`
- Tokenizer 文件 SHA-256：`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`；tokenizer 关联文件均在 `config.json` inputs 中逐项哈希。

## 评价与搜索设置

- 三臂 A/B/C：BM25-v2 原语料、冻结 BM25-v3 段落方案、冻结 BM25-v4 页面定位加正文选段。
- Goal Q 入口：direct-v090、thinking=false、legacy；top-3；每题最多4次模型调用/3次搜索。
- Greedy：temperature=0、do_sample=false；每次最多2048输入/128新 token。
- 四组固定问题共208题，每臂一次，共624条轨迹；历史确认集只用于AA回归评价。

## 数据、门槛与隐私边界

- 各组数据/原文语料/段落语料/历史输出哈希和题目ID均冻结在 `config.json`。
- 先完成 A 并逐题复核 Goal Z 历史基线；不一致时停止，不执行 B/C。
- 使用冻结证据规则评分实际回答时可见内容；新确认集保持关闭，仅核验哈希。
- 候选开发提升至少4题，开发错误与无效动作不增，四组回归各最多下降1题，AA保护题最多丢失1题。

## 前提核查值

- 软件：Python 3.12.3、torch 2.5.1+cu124、transformers 5.17.0、PEFT 0.20.0、tokenizers 0.23.2、accelerate 1.15.0、CUDA 12.4。
- GPU 与驱动：NVIDIA GeForce RTX 4090 D；NVIDIA GeForce RTX 4090 D, 580.105.08, 24564 MiB。

执行接口：准备完成后使用 `run --approved-manifest-sha256 <上述哈希>`；任何输入变更会使批准失效。
