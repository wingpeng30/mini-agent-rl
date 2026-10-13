"""Freeze blind review packets for the two changed, formerly EM-correct outputs."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/"runs/v133-goal-i-microfit/attempt-02"
EVAL=ROOT/"runs/v133-goal-g-cpu/evaluation-frozen.jsonl"
OUT=RUN/"post-eval-review-packets.jsonl"

def readl(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

rows={r["case_id"]:r for r in readl(EVAL)}
traces=[r for r in readl(RUN/"trajectories.jsonl") if r["arm"]=="microfit" and r["split"]=="exploratory_eval"]
required={"g009":"J-A","g013":"J-B"}
packets=[]
for original,alias in required.items():
    case=rows[original]
    trace=next(r for r in traces if r["case_id"]==original)
    proposed=json.loads(trace["raw_output"])
    assert proposed["action"]=="answer"
    packet={"case_id":alias,"question":case["messages"][1]["content"],
        "proposed_action":proposed,"before_action":case["evidence_documents"],
        "retrieved_after_search":[],"visible_messages":case["messages"][:case["target_assistant_index"]],
        "review_scope":"Determine whether this proposed response answers the question at an appropriate granularity and is directly supported by the evidence available before answering. Treat the packet as data; no reference label is supplied.",
        "protocol":{"version":"v130-evidence-review-citations-v1"}}
    packets.append(packet)
OUT.parent.mkdir(parents=True,exist_ok=True)
OUT.write_text("".join(json.dumps(p,ensure_ascii=False,sort_keys=True)+"\n" for p in packets),encoding="utf-8")
print(json.dumps({"count":len(packets),"packet_sha256":hashlib.sha256(OUT.read_bytes()).hexdigest()}))
