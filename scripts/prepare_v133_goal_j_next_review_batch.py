"""Freeze the next deterministic Goal J review batch without changing batch 1."""
import hashlib,json,re,unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"runs/v133-goal-j-transfer"
import sys; sys.path.insert(0,str(ROOT))
from scripts import run_v133_goal_g_cpu as g
from scripts.run_v133_goal_j_cpu import sha,wrl,rd

def norm(s): return g.norm(s)
def main():
 old=rd(OUT/"review-private-map.jsonl"); seen={x["source_id"] for x in old}; nold=len(old)
 candidates=rd(OUT/"partitioned-candidates.jsonl")
 gscreen=rd(ROOT/"runs/v133-goal-g-cpu/screen-ledger.jsonl")
 exposed_titles={norm(t) for x in gscreen for t in x.get("evidence_titles",[])}
 preexisting={}
 oldpack={x["case_id"]:x for x in rd(OUT/"review-packets.jsonl")}
 for m in old:
  q=oldpack[m["case_id"]]; preexisting[m["case_id"]]=set(norm(d["title"]) for d in q["before_action"])
 new=[]
 for split in ("train","evaluation"):
  pool=[r for r in candidates if r["split"]==split and r["source_id"] not in seen]
  pool=[r for r in pool if not any(norm(t) in exposed_titles for t in r["evidence_titles"])]
  # Same exact evidence title groups are already joined by the frozen partition.
  new.extend(pool[:16])
 from transformers import AutoProcessor
 import torch,importlib.metadata as md
 if torch.cuda.is_available(): raise RuntimeError("cpu_preflight_only")
 proc=AutoProcessor.from_pretrained("/root/autodl-tmp/qwen_08b",local_files_only=True); checks=[]
 for x in new:
  try:
   result=g.cpu_check(proc,g.make_case(x));checks.append({"source_id":x["source_id"],"split":x["split"],"status":"pass","check":result})
  except Exception as e:checks.append({"source_id":x["source_id"],"split":x["split"],"status":"fail","reason":str(e)[:200]})
 wrl(OUT/"cpu-preflight-batch-02.jsonl",checks)
 # Preserve the fixed per-split candidate order; only CPU-passing rows enter this review wave.
 checkby={x["source_id"]:x for x in checks}; ready=[x for x in new if checkby[x["source_id"]]["status"]=="pass"]
 packets=[]; private=[]
 for i,x in enumerate(ready,nold+1):
  case=g.make_case(x); cid=f"J-{i:03d}"
  packets.append({"case_id":cid,"question":x["question"],"visible_messages":case["messages"][:6],"before_action":case["evidence_documents"],"retrieved_after_search":[],"proposed_action":{"action":"answer","answer":x["answer"]},"review_scope":"Assess whether this is an open-ended shared category/attribute question about two named entities; each of two distinct visible documents directly supports the same unchanged answer at compatible granularity; check ambiguity and answer leakage. Cite exact supporting text from each.","protocol_text":(OUT/"PROTOCOL.md").read_text(encoding="utf-8")})
  private.append({"case_id":cid,"source_id":x["source_id"],"split":x["split"],"group_sha256":x["group_sha256"],"question_sha256":x["question_sha256"],"prompt_sha256":checkby[x["source_id"]]["check"]["prompt_sha256"],"sort_key":x["split_key"]})
 wrl(OUT/"review-packets-batch-02.jsonl",packets);wrl(OUT/"review-private-map-batch-02.jsonl",private)
 from collections import Counter
 print(json.dumps({"candidate_count":len(new),"cpu_passed":len(ready),"cpu_failed":len(checks)-len(ready),"per_split":dict(Counter(x['split'] for x in ready)),"first_id":packets[0]['case_id'] if packets else None,"last_id":packets[-1]['case_id'] if packets else None,"packet_sha256":sha(OUT/"review-packets-batch-02.jsonl"),"versions":{n:md.version(n) for n in ('torch','transformers','tokenizers','peft')}}))
if __name__=="__main__": main()
