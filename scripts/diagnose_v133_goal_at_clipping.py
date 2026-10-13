"""Read-only CPU explanation of fixed-pool policy updates; no new experiment."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/v133-goal-at-supported-action-learning-v1'
updates=[json.loads(x) for x in (OUT/'seed42-B/updates.jsonl').read_text().splitlines()]
groups=[r for r in updates if r['group_diagnostics']]
values=[d for r in groups for d in r['group_diagnostics']]
result=dict(effective_mixed_updates=len(groups),sampled_rollouts=len(values),
 mean_token_clip_fraction=statistics.mean(v['clip_fraction'] for v in values),
 max_token_clip_fraction=max(v['clip_fraction'] for v in values),
 mean_parent_reference_kl_estimator=statistics.mean(v['kl'] for v in values),
 maximum_parent_reference_kl_estimator=max(v['kl'] for v in values),
 updates=[dict(update=r['update'],state_id=r['group_id'],mean_clip_fraction=statistics.mean(d['clip_fraction'] for d in r['group_diagnostics']),
   mean_kl=statistics.mean(d['kl'] for d in r['group_diagnostics'])) for r in groups],
 interpretation='Limited offline pool after shared SFT moved policy away from parent behavior. Clipping/conditional KL diagnose this recipe, not RL generally; no resampling or tuning follows.')
with (OUT/'policy-update-diagnostic.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
with (OUT/'REPORT.md').open('a',encoding='utf-8') as f:
 f.write('\n## 固定池策略更新限制\n\n')
 f.write(f"B有效混合组更新{len(groups)}次，覆盖{len(values)}个采样输出；平均token裁剪比例{result['mean_token_clip_fraction']:.4f}，最大{result['max_token_clip_fraction']:.4f}，平均冻结父策略KL估计{result['mean_parent_reference_kl_estimator']:.4f}。共同轨迹SFT发生在组监督之前，采样池没有刷新，old/ref始终来自AM父模型。该结果只淘汰本轮固定离线池配方，不能概括为所有RL策略无效；裁剪统计也不单独证明语义退步的原因。未因此追加采样或改变clip/KL。\n")
print(json.dumps(result))
