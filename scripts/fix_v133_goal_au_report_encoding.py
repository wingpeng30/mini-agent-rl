"""Preserve JSON-encoded report, expose the same prose as Markdown."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/v133-goal-au-frozen-reranking-v3'
p=out/'REPORT.md';data=p.read_text(encoding='utf-8');decoded=json.loads(data)
if not isinstance(decoded,str) or not decoded.startswith('# Goal AU'):raise RuntimeError('Unexpected report encoding')
backup=out/'REPORT-json-encoded-original.json'
with backup.open('x',encoding='utf-8') as f:f.write(data)
p.write_text(decoded,encoding='utf-8')
with (out/'report-encoding-repair.json').open('x',encoding='utf-8') as f:
    json.dump(dict(reason='Generic write helper serialized prose as a JSON string; decoded without semantic or score edits',
        prior_sha256=hashlib.sha256(data.encode()).hexdigest(),markdown_sha256=hashlib.sha256(decoded.encode()).hexdigest(),new_model_calls=0),f,indent=2)
