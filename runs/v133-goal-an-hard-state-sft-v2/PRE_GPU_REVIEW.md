# Goal AN GPU前实际前提

准备和CPU核验已完成；本轮GPU尚未授权。

冻结manifest SHA-256：`dc28dc3d72e55f4a38fd2cc849462c7e007a8b0572b4e8f457263d1aadc2c771`。
模型：AA发布Goal Z seed42，权重SHA `d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`；基础模型`/root/autodl-tmp/qwen_08b`。
复用A42：AM普通SFT权重SHA `3bcd1f85382c1d7775fa674c1a6e1e68c80911a69b86de209d20f2ab6269280c`，不重跑64题与24完整探针。

| 训练难例 | Goal Z实际错误 | 正确当前动作答案 | B状态系数 |
|---|---|---|---:|
| Who sings Never Say Never with the performer who gave the Believe Tour? | Justin Bieber | Jaden Smith | 2.956521739 |
| Where does Ossian Donner's father work at? | professor | University of Helsinki | 2.956521739 |
| When was the director of film Return Of The Seven born? | September 3 | September 3, 1922 | 2.956521739 |
| Where was the director of film Terror Mountain born? | June 28, 1898 | Christiansburg, Virginia | 2.956521739 |

其余30状态系数0.739130435；4难例系数2.956521739；总和34。各状态仍只出现一次。
34状态与128轨迹均来自AM冻结材料，未改变目标或历史输入。仅四难例逐边复核；详见hard-state-admission.json的原句及token位置。
每臂128更新，第2、4…68次加入状态；L=轨迹均值NLL+0.25×状态系数×当前动作均值NLL。当前JSON与终止符监督，其余历史mask。
LR5e-6；AdamW(0.9,0.999),eps1e-8,weight_decay0；clip1；只训练LoRA，BF16，单卡，不量化。
共同评价：direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top3、greedy、16384/2048、4调用/3搜索。

部分链探针按AL题序冻结前8题，第一边原文可见、第二边未见；已有23候选，不采集新资料。

- `musique:2hop__864101_135373`，调用1，桥接`Bering Sea`，剩余3调用/2搜索。
- `musique:2hop__478978_124498`，调用1，桥接`Matthieu Chedid`，剩余3调用/2搜索。
- `musique:2hop__862994_69048`，调用1，桥接`Delta State`，剩余3调用/2搜索。
- `musique:2hop__829358_754719`，调用1，桥接`Pedro Leopoldo`，剩余3调用/2搜索。
- `musique:2hop__789565_416401`，调用1，桥接`Wang Yanjun`，剩余3调用/2搜索。
- `musique:2hop__506035_325512`，调用1，桥接`Poplarville`，剩余3调用/2搜索。
- `musique:2hop__337223_56873`，调用1，桥接`Nottingham Forest`，剩余3调用/2搜索。
- `musique:2hop__796221_18803`，调用1，桥接`Shell`，剩余3调用/2搜索。

seed42只训练B；B完整评价64题及24完整/8部分链探针；Goal Z和复用A各8部分链；A/B各4训练拟合检查。第一阶段最多128更新、64轨迹、312生成、192搜索。
B需支持≥23/29、完整探针≥22/24、至少一项实体/角色/关系对象修复，错误≤7、动作问题≤1且各项不高于A，保留AM两道新增成功，7保护题最多丢1，部分链动作不退步。
通过才独立训练seed43 A/B；B相同门槛且较同seed A至少增加2题支持成功和2个完整探针正确。
全轮最多3训练/384更新、192轨迹、896生成、576搜索；87个新增准入答案组合；不运行新确认，不发布。
GPU/数值/身份/哈希/基础设施失败保存现场并停止，不自动重跑或改参；语义失败正常计分。未达门槛不运行seed43。

收到针对这份实际前提的确认后，写入绑定manifest哈希的新gpu-approval.json，再执行。旧AM授权不能用于AN。
