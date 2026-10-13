# 准备记录

首次远程CPU测试命令未设置scripts导入路径，32项通过、2项因旧脚本模块导入失败。该命令未加载GPU、未采样、未创建实验manifest。配置PYTHONPATH=src:scripts后完整34项通过，再执行CPU prepare并冻结；没有修改模型、alpha或生成参数。

工程及探针均只执行一次；未发生GPU/数值/身份/基础设施失败，没有重跑。准备时查阅Transformers 5.17接口确认旧版cache_position辅助方法已移除，因此实现采用本地版本prepare_inputs_for_generation及独立DynamicCache/显式文本position_ids。

生成代码与评分代码分别冻结于manifest和adjudication-freeze；旧实验账本/分数不改。REPORT.md是聚合原文报告，RESULTS.zh.md是同一结论的人读摘要，无评分修改。
