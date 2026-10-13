"""Preserve and unwrap an encoded prose prefix, retaining later notes."""
import json
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'runs/v133-goal-at-supported-action-learning-v1'
p=out/'REPORT.md';data=p.read_text(encoding='utf-8');decoded,end=json.JSONDecoder().raw_decode(data)
if not isinstance(decoded,str) or not decoded.startswith('# Goal AT'):raise RuntimeError('Unexpected report')
with (out/'REPORT-json-encoded-original.txt').open('x',encoding='utf-8') as f:f.write(data)
p.write_text(decoded+data[end:],encoding='utf-8')
with (out/'report-encoding-repair.json').open('x',encoding='utf-8') as f:json.dump(dict(reason='Unwrap JSON prose prefix, preserve appended diagnostic notes; no semantic or score edits',new_model_calls=0),f)
