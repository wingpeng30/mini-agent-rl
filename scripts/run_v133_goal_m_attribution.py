"""Freeze the 32 paired P0 development comparisons without new inference."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs/v133-goal-k-attribution/autodl-results/paired-attribution.jsonl'
OUT = ROOT / 'runs/v133-goal-m-readiness/paired-p0.md'

rows = [json.loads(x) for x in SOURCE.read_text(encoding='utf-8').splitlines() if x.strip()]
rows = [r for r in rows if r['split'] != 'train']
assert len(rows) == 32
lines = ['# P0 逐题配对（开发集，事后诊断）', '', '| 集合 | ID | v118 动作/EM/F1 | Goal J 动作/EM/F1 | 变化类 |', '|---|---|---|---|---|']
for r in rows:
    a, b = r['before'], r['after']
    lines.append(f"| {r['split']} | {r['case_id']} | {a['action_type']}/{a['em']}/{a['f1']:.3f} | {b['action_type']}/{b['em']}/{b['f1']:.3f} | {r['attribution_class']} |")
for split in ('new_eval', 'goal_G_regression'):
    both = [r for r in rows if r['split'] == split and r['before']['action_type'] == r['after']['action_type'] == 'answer']
    lines += ['', f"{split} 双方有效 answer：{len(both)} 题；严格 EM {sum(r['before']['em'] for r in both)}→{sum(r['after']['em'] for r in both)}；ID：{', '.join(r['case_id'] for r in both)}。"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(OUT)
