"""One-shot blind evidence review for Goal I's two strict-EM losses."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.v130_deepseek_review import MODEL,_credentials,_no_redirect_opener,run_batch

RUN=ROOT/"runs/v133-goal-i-microfit/attempt-02"
PACKET=RUN/"post-eval-review-packets.jsonl"
EXPECTED="0ecd7eba9e16e2dea1592eafcfc58c8a8693b24f7f5136c3cfd26fa50db9b5a4"
actual=hashlib.sha256(PACKET.read_bytes()).hexdigest()
if actual!=EXPECTED: raise ValueError("regression_review_packet_hash_mismatch")
packets=[json.loads(x) for x in PACKET.read_text(encoding="utf-8").splitlines() if x.strip()]
if len(packets)!=2 or {x["case_id"] for x in packets}!={"J-A","J-B"}:
    raise ValueError("regression_review_packet_scope_mismatch")
key,base=_credentials()
if not key or base!="https://api.deepseek.com": raise RuntimeError("authorized_deepseek_configuration_missing")
batch_dir=RUN/"deepseek/i-regression-01"
freeze=hashlib.sha256((RUN/"freeze-manifest.json").read_bytes()).hexdigest()
result=run_batch(batch="i-regression-01",packets=packets,packet_file_sha256=actual,
    freeze_sha256=freeze,batch_dir=batch_dir,key=key,base=base,opener=_no_redirect_opener())
raw=sorted((batch_dir/"responses-private").glob("attempt-*-raw.json"))[-1]
identity_raw=json.loads(raw.read_text(encoding="utf-8"))
identity={"requested_model":MODEL,"response_model":identity_raw.get("model"),
    "model_identity_verified":identity_raw.get("model")==MODEL,"packet_sha256":actual,
    "review_count":len(result["reviews"]),"approved":sum(v["approved"] for v in result["reviews"])}
(batch_dir/"identity-check.json").write_text(json.dumps(identity,indent=2)+"\n",encoding="utf-8")
if not identity["model_identity_verified"]: raise RuntimeError("deepseek_model_identity_mismatch")
print(json.dumps({"status":"valid_review_batch_saved",**identity},ensure_ascii=False))
