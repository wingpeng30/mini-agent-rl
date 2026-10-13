"""Single authorized DeepSeek blind batch for the frozen Goal I packet."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.v130_deepseek_review import MODEL, _credentials, _no_redirect_opener, run_batch

OUT=ROOT/"runs/v133-goal-i-microfit"
PACKET=OUT/"review-packets.jsonl"
EXPECTED_PACKET_SHA="f39439b6bae666a7a2bc8e3ed5570dd73ae326ea77d3367259bc613bd9314972"

actual=hashlib.sha256(PACKET.read_bytes()).hexdigest()
if actual != EXPECTED_PACKET_SHA:
    raise ValueError("frozen_goal_i_review_packet_hash_mismatch")
packets=[json.loads(line) for line in PACKET.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(packets)!=4 or {p["case_id"] for p in packets}!={"I-A","I-B","I-C","I-D"}:
    raise ValueError("frozen_goal_i_packet_set_mismatch")
manifest=json.loads((OUT/"freeze-manifest.json").read_text(encoding="utf-8"))
if manifest.get("review_packets_sha256") != actual or manifest.get("training_started") is not False:
    raise ValueError("goal_i_review_not_bound_to_frozen_cpu_manifest")
key,base=_credentials()
if not key or base!="https://api.deepseek.com":
    raise RuntimeError("authorized_existing_deepseek_credential_or_endpoint_missing")
batch_dir=OUT/"deepseek"/"i-00-03"
result=run_batch(batch="i-00-03",packets=packets,packet_file_sha256=actual,
    freeze_sha256=hashlib.sha256((OUT/"freeze-manifest.json").read_bytes()).hexdigest(),
    batch_dir=batch_dir,key=key,base=base,opener=_no_redirect_opener(),direct_connection=False)
raw=sorted((batch_dir/"responses-private").glob("attempt-*-raw.json"))[-1]
response=json.loads(raw.read_text(encoding="utf-8"))
identity={"requested_model":MODEL,"response_model":response.get("model"),
    "model_identity_verified":response.get("model")==MODEL,"packet_sha256":actual,
    "review_count":len(result["reviews"]),"approved":sum(v["approved"] for v in result["reviews"])}
(batch_dir/"identity-check.json").write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
if not identity["model_identity_verified"]:
    raise RuntimeError("deepseek_response_model_identity_mismatch")
print(json.dumps({"status":"valid_review_batch_saved",**identity},ensure_ascii=False))
