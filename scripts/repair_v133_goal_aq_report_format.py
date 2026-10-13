"""Recover a Markdown report accidentally JSON-encoded; no scoring or inference."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/v133-goal-aq-grounded-action-v1"
path=OUT/"REPORT.md"
original=path.read_bytes()
text=json.loads(original.decode("utf-8"))
if not isinstance(text,str) or not text.startswith("# Goal AQ"):
    raise RuntimeError("only the archived JSON-string serialization error can be repaired")
backup=OUT/"REPORT.initial-json-string.txt"
with backup.open("xb") as handle:handle.write(original)
path.write_text(text,encoding="utf-8",newline="\n")
amendment=dict(scope="Markdown serialization only; decoded content identical",
    old_report_sha256=hashlib.sha256(original).hexdigest(),
    new_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    original_finalizer_sha256=json.loads((OUT/"final-status.json").read_text())["finalizer_sha256"],
    repaired_finalizer_sha256=hashlib.sha256((ROOT/"scripts/finalize_v133_goal_aq.py").read_bytes()).hexdigest(),
    generations_added=0,searches_added=0,adjudications_changed=False)
with (OUT/"report-format-amendment.json").open("x",encoding="utf-8") as handle:
    json.dump(amendment,handle,ensure_ascii=False,indent=2)
print(json.dumps(amendment))
