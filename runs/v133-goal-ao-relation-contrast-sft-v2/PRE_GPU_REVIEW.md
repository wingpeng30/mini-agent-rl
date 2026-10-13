# Goal AO 冻结 GPU 前提

Manifest SHA-256：`e28ce625fdd04497413f7d1035c0f7301e28a23ab8790f4b456b1f524de0aaf7`  
起点 Goal Z SHA-256：`d15d3864c1ccb810a6be7e5719d7b5d0ac87c353408ed85d973bb6393ff16b52`；AA发布记录、模型/语料/tokenizer与代码身份在manifest中逐项冻结。CPU预检通过，GPU尚未开始。

训练材料：128条Goal Z原轨迹、34个AM状态、8组/16个派生充分证据状态。材料按case_id+state_id冻结；每个原问题和反事实问题均使用同一历史正文，逐条保存引用、token位置及原始消息。评价ID/问题已排除。见 `evidence-admission.json` 与 `ao-groups.json`。

A每关系组重复原问答两次；B使用原问答与关系对照问答。两臂相同数据数量、顺序、损失系数和128更新。`L=L轨迹+0.25 L_AM+0.25 L组`，组内两个动作均值NLL取平均；当前动作JSON与终止符监督，其余prompt/历史/正文mask。优化器LR5e-6、AdamW(0.9,0.999)/eps1e-8/wd0、clip1、LoRA-only、单卡BF16不量化。

条件统一：Goal Q、direct-v090、first-json-v1、thinking=false、legacy、BM25-v2/top-3、greedy、16384/2048、4调用/3搜索。

seed42两臂各评估64题真实搜索、24完整证据、8缺链动作和16训练拟合状态；AM普通SFT先检查同16状态。探针不继续搜索。seed43只在seed42 B满足PLAN.md全部冻结门槛时执行。总额度最多4训练/512更新、256真实轨迹、1232生成、768搜索；第一阶段最多2训练/256更新、128轨迹、624生成、384搜索。新确认0，不发布。

GPU、数值、身份、哈希或基础设施失败保存现场并停止；不自动重试或改变条件。GPU执行已获用户授权，不再重复确认。
