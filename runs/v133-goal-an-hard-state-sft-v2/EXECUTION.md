# Goal AN执行步骤

当前准备完成，GPU等待用户确认PRE_GPU_REVIEW.md。下列命令均在远程仓库，用.venv/bin/python执行。收到确认后，另存gpu-approval.json，human_confirmed=true且manifest_sha256须精确匹配manifest.json；不复用AM审批。

第一阶段固定顺序：

```bash
export PYTHONPATH=src:scripts
.venv/bin/python scripts/run_v133_goal_an.py probe --model GoalZ --kind partial
.venv/bin/python scripts/run_v133_goal_an.py probe --model 42-A --kind partial
.venv/bin/python scripts/run_v133_goal_an.py probe --model 42-A --kind training-fit
.venv/bin/python scripts/run_v133_goal_an.py train --seed 42 --arm B
.venv/bin/python scripts/run_v133_goal_an.py evaluate --model 42-B
.venv/bin/python scripts/run_v133_goal_an.py probe --model 42-B --kind complete
.venv/bin/python scripts/run_v133_goal_an.py probe --model 42-B --kind partial
.venv/bin/python scripts/run_v133_goal_an.py probe --model 42-B --kind training-fit
.venv/bin/python scripts/analyze_v133_goal_an.py packets --seed 42
```

逐题裁定一次，保存primary-adjudications-seed42.json，再执行：

```bash
.venv/bin/python scripts/analyze_v133_goal_an.py finalize --seed 42
```

裁定文件：answers[model][case_id]保存category、reason、entity_relation_mismatch；complete[model][probe_id]保存correct、reason、binding_repair；partial[model][probe_id]保存legal_missing_relation_search、reason与查询实体来源及缺失关系依据。完全相同问题/实际输入/答案组合复用原裁定。绑定修复必须原同输入A错误、本轮B正确，不能仅动作由search变answer。

训练拟合输出另存诊断，不计入支持成功或探针泛化门槛。新增答案审核第一阶段29组合，全轮最多87组合，不扩来源。若必要关系的已有原文指针未在实际输入出现，不能自动判充分，也不临时搜索补证。

只有seed42-gate.json的route_passed为true才执行seed43独立A/B；每条路线train、evaluate、complete/partial/training-fit probe各一次，然后packets/finalize --seed43。始终从发布Goal Z加载；A42仅复用AM。任何失败锁都拒绝后续GPU执行，不覆盖失败结果，不自动换参/重跑。

GPU额度按调用尝试累计写gpu-usage.jsonl；种子42预定最多312生成、192搜索、128更新；全轮896生成、576搜索、384更新。不会执行新确认或发布。
