# v097：零方差组采样多样性诊断

冻结决策：先只读分析 v096 三 seed 的全部 30 个组。若零方差组的动作序列中位唯一数不超过 2，且总体非法动作率不超过 10%，只把 group size 从 4 改为 8；temperature 保持 0.7。若动作已多样但 reward 仍相同，则不调整采样参数，停止并回到 reward 分辨率设计。

若选择 group size 8，pilot 仍从 checkpoint-200 独立运行 seed42/43/44、前10题、固定前100题语料、v096 reward、BF16、2048/128、BM25-v2、compact-v3、LR 5e-6、clip .2、KL .02、policy epochs 2、max steps 3、21GiB。唯一实验变量为每题 rollout 数 4→8。

每个 seed 仍要求至少 5 个非零方差更新组、无基础设施/数值/显存失败并可重载。任一 seed 不通过即停止，不挑 seed；三组通过后才进行同协议 20 题快速回归。final-test 保持封存。

## 已执行诊断结论

三组共 17 个零方差组，零方差组的动作序列唯一数中位数为 3/4；总体非法动作率 5%。因此没有满足 `median_unique_actions <= 2` 的 group-size 扩展门槛，温度变化预先被禁止。本轮不启动新的三 seed 训练：现有证据显示瓶颈是不同合法动作被压缩成相同的终局 reward，而不是采样数量不足。
