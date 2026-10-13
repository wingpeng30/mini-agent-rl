"""Bounded Goal J candidate locator and CPU preparation; no model weights/inference."""
from __future__ import annotations
import hashlib, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/v133-goal-j-transfer"
PROTOCOL=OUT/"PROTOCOL.md"
G=ROOT/"scripts/run_v133_goal_g_cpu.py"
sys.path.insert(0,str(ROOT))
from scripts import run_v133_goal_g_cpu as g

SEED="v133-goal-j-v1"
LOCATOR=re.compile(r"\b(both|share|shared|common|same|each)\b",re.I)
YESNO=re.compile(r"^\s*(is|are|was|were|do|does|did|has|have|had|can|could|will|would|should|may|might|must)\b",re.I)

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def wr(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
def wrl(p,rows): Path(p).write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows),encoding='utf-8')
def rd(p): return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def j_exposure():
 ids,qs,titles,inputs=g.exposure()
 # Every row opened or screened in Goal G is now exposed, not only its final 16.
 for r in rd(ROOT/"runs/v133-goal-g-cpu/screen-ledger.jsonl"):
  ids.add(str(r["source_id"])); qs.add(r["question_sha256"])
 # Explicitly bind Goal G frozen inputs and Goal H/I probes.
 for r in rd(ROOT/"runs/v133-goal-g-cpu/evaluation-frozen.jsonl"):
  ids.add(str(r["source_id"])); qs.add(r["question_sha256"])
 hprobe=ROOT/"runs/v133-goal-h-comparison/fit-probes-frozen.jsonl"
 for r in rd(hprobe):
  ids.add(str(r["source_id"])); qs.add(r["question_sha256"])
  for t in r.get("evidence_titles",[]): titles.add(g.norm(t))
 inputs.update({"runs/v133-goal-g-cpu/screen-ledger.jsonl":sha(ROOT/"runs/v133-goal-g-cpu/screen-ledger.jsonl"),
  "runs/v133-goal-g-cpu/evaluation-frozen.jsonl":sha(ROOT/"runs/v133-goal-g-cpu/evaluation-frozen.jsonl"),
  str(hprobe.relative_to(ROOT)):sha(hprobe),
  "runs/v133-goal-i-microfit/attempt-02/summary.json":sha(ROOT/"runs/v133-goal-i-microfit/attempt-02/summary.json")})
 return ids,qs,titles,inputs

def locate():
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/"locator.jsonl").exists(): raise FileExistsError("goal_j_locator_frozen")
 files=g.source_files(); ids,qs,titles,inputs=j_exposure(); candidates=[]; counts=Counter(); idx=0
 for path in files:
  for batch in __import__('pyarrow.parquet',fromlist=['ParquetFile']).ParquetFile(path).iter_batches(columns=["id","question","type"],batch_size=4096):
   for r in batch.to_pylist():
    q=str(r.get("question") or ""); counts["metadata_rows"]+=1
    if not LOCATOR.search(q) or YESNO.search(q): idx+=1; continue
    counts["locator_matches"]+=1; sid=str(r["id"]); qh=g.qhash(q)
    if sid in ids or qh in qs: counts["exposed_or_duplicate"]+=1; idx+=1; continue
    candidates.append({"source_id":sid,"source_row_index":idx,"question":q,"question_sha256":qh,
      "type":r.get("type"),"sort_key":hashlib.sha256((SEED+sid).encode()).hexdigest()})
    idx+=1
 if idx!=90447: raise ValueError("source_row_count_mismatch")
 candidates.sort(key=lambda x:x["sort_key"]); wrl(OUT/"locator.jsonl",candidates)
 summ={"status":"frozen","protocol_sha256":sha(PROTOCOL),"source_hashes":{str(p.relative_to(ROOT)):sha(p) for p in files},
  "source_provenance_sha256":sha(g.SOURCE/"download-provenance.json"),"input_sha256":inputs,
  "exposure_ids":len(ids),"exposure_questions":len(qs),"exposure_titles":len(titles),"counts":dict(counts),
  "eligible_metadata_count":len(candidates),"locator_sha256":sha(OUT/"locator.jsonl"),"full_text_budget":200}
 wr(OUT/"locator-summary.json",summ); print(json.dumps({"counts":dict(counts),"eligible":len(candidates)}))

def screen():
 summ=json.loads((OUT/"locator-summary.json").read_text(encoding='utf-8'))
 if summ["protocol_sha256"]!=sha(PROTOCOL) or summ["locator_sha256"]!=sha(OUT/"locator.jsonl"): raise ValueError("frozen_locator_changed")
 if (OUT/"screen-ledger.jsonl").exists(): raise FileExistsError("full_text_budget_already_used")
 files=g.source_files(); loc=rd(OUT/"locator.jsonl")[:200]; positions={r["source_row_index"]:r for r in loc}; selected={}; start=0
 import pyarrow.parquet as pq
 for path in files:
  pf=pq.ParquetFile(path)
  for gi in range(pf.metadata.num_row_groups):
   size=pf.metadata.row_group(gi).num_rows; wanted=[i for i in positions if start<=i<start+size]
   if wanted:
    table=pf.read_row_group(gi,columns=["id","answer","supporting_facts","context"])
    for i in wanted: selected[i]={k:table[k][i-start].as_py() for k in table.column_names}
   start+=size
 if set(selected)!=set(positions): raise ValueError("opened_row_mismatch")
 _,_,exposed_titles,_=j_exposure(); ledger=[]; provisional=[]; counts=Counter()
 for l in loc:
  row=selected[l["source_row_index"]]
  if str(row["id"])!=l["source_id"]: raise ValueError("source_id_changed")
  ans=str(row.get("answer") or "").strip(); facts=g.official_facts(row); titles=list(dict.fromkeys(x["title"] for x in facts)) if facts else []
  reason=None
  if not ans or g.norm(ans) in ("yes","no"): reason="boolean_or_empty_answer"
  elif not facts or len(titles)!=2: reason="not_two_resolved_evidence_titles"
  elif any(g.norm(t) in exposed_titles for t in titles): reason="title_overlaps_known_training_dev"
  item={**l,"answer":ans,"supporting_facts":facts,"evidence_titles":titles,"screen_reason":reason or "requires_semantic_and_entity_review"}
  ledger.append(item); counts[item["screen_reason"]]+=1
  if not reason: provisional.append(item)
 wrl(OUT/"screen-ledger.jsonl",ledger); wrl(OUT/"provisional.jsonl",provisional)
 wr(OUT/"screen-summary.json",{"opened_full_rows":len(ledger),"structural_candidates":len(provisional),
  "reason_counts":dict(counts),"ledger_sha256":sha(OUT/"screen-ledger.jsonl"),"provisional_sha256":sha(OUT/"provisional.jsonl")})
 print(json.dumps({"opened":len(ledger),"structural":len(provisional),"reasons":dict(counts)}))

def partition():
 if (OUT/"partitioned-candidates.jsonl").exists(): raise FileExistsError("partition_already_frozen")
 rows=rd(OUT/"provisional.jsonl"); parent=list(range(len(rows)))
 def root(i):
  while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
  return i
 def merge(a,b):
  a=root(a); b=root(b)
  if a!=b: parent[b]=a
 # Exact evidence document identity/title creates a conservative connection.
 title_owner={}
 for i,r in enumerate(rows):
  for t in r["evidence_titles"]:
   k=g.norm(t)
   if k in title_owner: merge(i,title_owner[k])
   else: title_owner[k]=i
 groups={}
 for i,r in enumerate(rows): groups.setdefault(root(i),[]).append(i)
 chosen=[]; conflict=[]
 for indices in groups.values():
  indices.sort(key=lambda i:rows[i]["sort_key"])
  ids=sorted(rows[i]["source_id"] for i in indices)
  digest=hashlib.sha256((SEED+"\n".join(ids)).encode()).hexdigest()
  split="train" if int(digest[:2],16)%2==0 else "evaluation"
  # One case per source/entity component, fixed by source-row hash before review.
  chosen.append({**rows[indices[0]],"split":split,"group_sha256":digest,
   "group_source_ids":ids,"group_case_count":len(indices),"split_key":digest})
  for i in indices[1:]: conflict.append({"source_id":rows[i]["source_id"],"reason":"same_source_entity_group_selected_representative","group_sha256":digest})
 chosen.sort(key=lambda x:(x["split"],x["split_key"]))
 wrl(OUT/"partitioned-candidates.jsonl",chosen); wrl(OUT/"group-exclusions.jsonl",conflict)
 wr(OUT/"partition-summary.json",{"status":"frozen_before_review","grouping":"source ID plus exact normalized evidence-title overlap; clear aliases manually checked for selected review candidates","train_groups":sum(x["split"]=="train" for x in chosen),"evaluation_groups":sum(x["split"]=="evaluation" for x in chosen),"group_exclusions":len(conflict),"partition_sha256":sha(OUT/"partitioned-candidates.jsonl"),"provisional_sha256":sha(OUT/"provisional.jsonl"),"partition_seed":SEED})
 print(json.dumps(json.loads((OUT/"partition-summary.json").read_text())))

def preflight():
 import torch, importlib.metadata as md
 from transformers import AutoProcessor
 if torch.cuda.is_available(): raise RuntimeError("CPU_only_preflight_required")
 if (OUT/"cpu-preflight.jsonl").exists(): raise FileExistsError("cpu_preflight_frozen")
 candidates=[]
 for split in ("train","evaluation"):
  candidates += [r for r in rd(OUT/"partitioned-candidates.jsonl") if r["split"]==split][:24]
 proc=AutoProcessor.from_pretrained("/root/autodl-tmp/qwen_08b",local_files_only=True); result=[]
 for item in candidates:
  try:
   row=g.make_case(item); check=g.cpu_check(proc,row)
   result.append({"source_id":item["source_id"],"split":item["split"],"status":"pass","check":check})
  except Exception as e: result.append({"source_id":item["source_id"],"split":item["split"],"status":"fail","reason":str(e)[:200],"error_type":type(e).__name__})
 wrl(OUT/"cpu-preflight.jsonl",result)
 wr(OUT/"cpu-summary.json",{"candidate_count":len(result),"passed":sum(x["status"]=="pass" for x in result),"failed":sum(x["status"]!="pass" for x in result),"versions":{n:md.version(n) for n in ("torch","transformers","tokenizers","peft")},"cuda_available":torch.cuda.is_available(),"partition_sha256":sha(OUT/"partitioned-candidates.jsonl"),"results_sha256":sha(OUT/"cpu-preflight.jsonl")})
 print(json.dumps(json.loads((OUT/"cpu-summary.json").read_text())))

def review_packets():
 if (OUT/"review-packets.jsonl").exists(): raise FileExistsError("review_packets_frozen")
 rows=rd(OUT/"partitioned-candidates.jsonl"); checks={r["source_id"]:r for r in rd(OUT/"cpu-preflight.jsonl")}
 picked=[]
 for split in ("train","evaluation"):
  pool=[r for r in rows if r["split"]==split]
  seen=0
  for r in pool:
   c=checks.get(r["source_id"])
   if c and c["status"]=="pass": picked.append(r); seen+=1
   if seen>=24: break
 packets=[]; private=[]
 for i,item in enumerate(picked):
  cid=f"J-{i+1:03d}"; case=g.make_case(item)
  packets.append({"case_id":cid,"question":item["question"],"visible_messages":case["messages"][:6],"before_action":case["evidence_documents"],"retrieved_after_search":[],"proposed_action":{"action":"answer","answer":item["answer"]},"review_scope":"Assess whether this is an open-ended shared category/attribute question about two entities; both distinct visible documents directly support the same answer at compatible granularity; check ambiguity and answer leakage. Cite exact supporting text from each document.","protocol_text":PROTOCOL.read_text(encoding="utf-8")})
  private.append({"case_id":cid,"source_id":item["source_id"],"split":item["split"],"group_sha256":item["group_sha256"],"question_sha256":item["question_sha256"],"prompt_sha256":checks[item["source_id"]]["check"]["prompt_sha256"],"sort_key":item["split_key"]})
 wrl(OUT/"review-packets.jsonl",packets); wrl(OUT/"review-private-map.jsonl",private)
 wr(OUT/"review-packets-summary.json",{"count":len(packets),"per_split":{"train":sum(x["split"]=="train" for x in private),"evaluation":sum(x["split"]=="evaluation" for x in private)},"packet_sha256":sha(OUT/"review-packets.jsonl"),"private_map_sha256":sha(OUT/"review-private-map.jsonl"),"reviewer_visibility":"anonymous cases; no source ID, split, model scores, history, or other vote"})
 print(json.dumps(json.loads((OUT/"review-packets-summary.json").read_text())))

if __name__=="__main__":
 import argparse
 p=argparse.ArgumentParser(); p.add_argument("phase",choices=("locate","screen","partition","preflight","review-packets")); a=p.parse_args(); {"locate":locate,"screen":screen,"partition":partition,"preflight":preflight,"review-packets":review_packets}[a.phase]()
