# v0.9.5：GRPO 正确性补充验收与效果来源分析

日期：2026-09-18。未访问封存最终集；AutoDL 实例保持运行。

## P0：实现与权重审计

训练器现在要求所有 transition 都具备完整 action token、mask、prompt hash 与采样 logprob。首个 policy epoch 的全部 action token ratio 会在首次 optimizer step 前检查；任一 ratio 超出 [0.90, 1.10] 或 P95 超出 [0.99, 1.01] 时拒绝更新。

远端 CPU 回归为 72 项通过。隔离 GPU 探测中，44 个精确重放 token 的 initial ratio 与首次实际反传 ratio 都为 1.0；基础参数和 reference clone 未变化，192 个 LoRA 参数发生有限更新，保存前后的 adapter 张量一致，独立 Python 进程重载评分差为 0。

2048 输入＋128 输出、含 optimizer step 的压力探测为 allocated 15.260 GiB、reserved 16.289 GiB，低于 21 GiB。该驱动没有通过 nvidia-smi 返回当前 Python 进程显存，因此 device_process_gb 为缺失值。该项按协议标记为**资源遥测不可验证**，不得报告为完整 GPU 验收通过，也不得按零处理。

seed42、seed43、seed44 的 v094 adapter 都完成两次独立进程加载：文件 SHA-256、adapter tensor digest 和固定 token 评分逐次一致。权重可重载；但 v094 历史训练缺少“完整首个反传 ratio 在更新前”的逐 token 记录，故历史训练门槛不能追溯为完全通过。

## P1：提升来源

100 题自主评测和 120-case 历史探针均使用相同 DIRECT_SYSTEM，自主评测的 BM25-v2/compact-v3/语料指纹一致；探针的非 system 历史消息哈希也完全一致，因此不需要重跑输入不一致的评测。

总体 EM 提升主要来自 SFT 的 12 条非法输出被 GRPO 转为合法动作。三个 seed 的“非法→合法且正确”分别为 6、6、7 条，而“非法→合法但错误”分别为 6、6、5 条。

在 SFT 与三个 GRPO seed 都合法的 88 个共同样本上，跨 seed 平均 EM 差为 +1.14 个百分点，95% bootstrap 区间 [-2.27, +5.30]；F1 差为 +0.0044，区间 [-0.0265, +0.0391]。这不足以证明合法轨迹上的问答或检索策略已有稳定改善。

冻结探针中，seed43 在空结果条件的直接回答率增加 13.33 个百分点，其中错误直接回答增加 10.00 个百分点；该风险不应被总体 EM 掩盖。三个 seed 的完整证据后正确回答均有小幅增加，但部分证据后继续搜索略有减少，样本量不足以据此宣称策略提升。

## 建议

保留 checkpoint-200 和三个可重载 GRPO adapter 作为协议可靠性研究产物；不进入 50 题扩量。下一轮优先设计并固定一个能区分“合法但错误回答”和“证据充分后的正确停止”的 reward 或训练信号，再从同一起点做新的多 seed 对照。若目标只是部署稳定 JSON Agent，可考虑 GRPO；若目标是证明检索问答能力提升，当前证据仍不足。
# v095 corrected P0/P1 final result

The prior v094 adapters remain reloadable, but their first-backward full-group ratio was not preserved. They are therefore not the effect-evidence basis.  The corrected `checkpoint-200` 30-task × 4-rollout reruns (seeds 42/43/44) were evaluated under the frozen direct prompt, fixed 100-task development set, BM25-v2 corpus, and frozen 120-case probe.

- CPU regression: 72 passed.
- Corrected adapters: all three passed independent-process reload and fixed-sequence audit.
- P0 numerical and LoRA checks passed; the strict GPU acceptance status remains resource-telemetry-unverified because this container cannot expose the running process via `nvidia-smi --query-compute-apps`. It must not be reported as a zero-memory measurement.
- Across the 88 tasks legal for SFT and all three GRPO adapters, mean EM delta is +1.14pp (95% paired bootstrap CI -3.41pp to +5.68pp) and F1 delta is +0.44pp (CI -3.60pp to +4.42pp). Neither is statistically supported.
- The dominant measured improvement is protocol recovery: each seed converts 12 SFT-invalid cases to legal outputs, but 5–6 of those are still wrong. Seed 42 has a positive legal-trajectory signal; seed 44 is negative. This instability rules out selecting a best seed or entering 50-task expansion.

Decision: preserve checkpoint-200 and the three corrected adapters as reproducible pilot artifacts, but do not recommend GRPO as an SFT replacement or expand to 50 tasks. The next priority is reward/trajectory diagnosis focused on legal-but-wrong actions and retrieval decisions, then a pre-registered new pilot; it is not more scale on the present reward.
