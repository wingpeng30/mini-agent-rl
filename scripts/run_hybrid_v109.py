"""Run resumable v109 direct-answer and multi-candidate mechanisms."""
from __future__ import annotations

import argparse, datetime, hashlib, json, time
from pathlib import Path
from typing import Any

import httpx

from mini_agent_rl.answer_scoring import normalize_answer
from mini_agent_rl.hybrid_answer import (DirectResponse, anonymize_candidates, answer_metrics,
    apply_selection, conservative_boundary_guard, merge_candidates, parse_candidates, validate_citations)
from mini_agent_rl.training.thinking_sft import _env

DIRECT_SYSTEM="""Answer the question from the supplied evidence only. Return JSON with answer and citations. Each citation has document_id and an exact verbatim quote. Prefer the shortest complete answer. If evidence is incomplete, still return the best evidence-supported answer and cite the limitation."""
PASS1_SYSTEM="""Generate up to four distinct evidence-supported candidate answers to the question. Do not see or guess the existing agent answer. Return JSON {\"candidates\":[{\"answer\":...,\"citations\":[{\"document_id\":...,\"quote\":...}]}]}. Quotes must be exact. Include aliases or composed answers when supported."""
PASS2_SYSTEM="""Act as an adversarial answer analyst. Generate up to four plausible evidence-supported answers, checking entity relation, date precision, scope, yes/no polarity and multi-document composition. Return the same candidates JSON schema with exact quotes. Do not use outside knowledge."""
SELECT_SYSTEM="""Select the shortest complete candidate answer that correctly answers the question and is supported by the evidence. Candidate order is arbitrary. Do not prefer explanations, restatements of the question, dates or aliases beyond what the question asks. When candidates are semantically equivalent, choose the shortest canonical entity, date, number, yes/no, or phrase. Return JSON {\"decision\":\"select\"|\"abstain\",\"candidate_id\":\"C1\"|null,\"citations\":[{\"document_id\":...,\"quote\":...}]}. Select only when exact quotes support the answer; otherwise abstain."""

class Ledger:
    def __init__(self,budget:float,input_price:float,output_price:float):
        self.budget=budget; self.input_price=input_price; self.output_price=output_price; self.calls=[]
    @property
    def cost(self): return sum(x["estimated_cny"] for x in self.calls)
    def reserve(self,prompt_tokens:int,max_output:int):
        estimate=prompt_tokens*self.input_price/1_000_000+max_output*self.output_price/1_000_000
        if self.cost+estimate>self.budget: raise RuntimeError(f"budget_reservation_failed cost={self.cost:.4f} reserve={estimate:.4f}")
    def add(self,kind:str,body:dict[str,Any],usage:dict[str,Any],latency:float):
        inp=int(usage.get("prompt_tokens",0)); out=int(usage.get("completion_tokens",0)); cost=inp*self.input_price/1_000_000+out*self.output_price/1_000_000
        self.calls.append({"kind":kind,"model":body.get("model"),"prompt_tokens":inp,"completion_tokens":out,"cached_tokens":int((usage.get("prompt_tokens_details") or {}).get("cached_tokens",0)),"estimated_cny":cost,"latency_seconds":latency})

def call(client,key,ledger,kind,system,payload,max_tokens=8192):
    prompt=json.dumps(payload,ensure_ascii=False,separators=(",",":")); ledger.reserve(max(1,len(prompt)//2),max_tokens)
    request={"model":"deepseek-flash","thinking":{"type":"enabled"},"temperature":0,"max_tokens":max_tokens,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":system},{"role":"user","content":prompt}]}
    started=time.perf_counter(); response=client.post("https://api.deepseek.com/chat/completions",headers={"Authorization":f"Bearer {key}"},json=request); response.raise_for_status(); body=response.json(); ledger.add(kind,body,body.get("usage",{}),time.perf_counter()-started)
    choice=body["choices"][0]; content=choice["message"].get("content") or ""
    if choice.get("finish_reason") not in {"stop",None}: raise RuntimeError(f"{kind}_finish_reason={choice.get('finish_reason')}")
    if not content: raise RuntimeError(f"{kind}_empty_output")
    return content,body.get("model"),body.get("usage",{})

def direct(raw,docs):
    try: parsed=DirectResponse.model_validate_json(raw)
    except Exception as exc: return None,f"invalid_schema: {exc}"
    return (parsed.answer if validate_citations(parsed.citations,docs) else None),(None if validate_citations(parsed.citations,docs) else "invalid_citations")

def run_mechanism(client,key,ledger,row,evidence_name,include_direct):
    docs=row[evidence_name]; payload={"question":row["question"],"evidence":docs}; result={"evidence":evidence_name}
    if include_direct:
        try:
            raw,model,_=call(client,key,ledger,f"direct_{evidence_name}",DIRECT_SYSTEM,payload); answer,error=direct(raw,docs)
            result.update({"direct_raw":raw,"direct_answer":answer,"direct_error":error,"actual_model":model})
        except RuntimeError as exc:
            if "finish_reason=length" not in str(exc) and "empty_output" not in str(exc): raise
            result.update({"direct_raw":"","direct_answer":None,"direct_error":str(exc),"actual_model":None})
    passes=[]; raws=[]
    for index,system in enumerate((PASS1_SYSTEM,PASS2_SYSTEM),1):
        try:
            raw,model,_=call(client,key,ledger,f"candidates_{evidence_name}_p{index}",system,payload)
            values,error=parse_candidates(raw,docs); passes.append(values); raws.append({"raw":raw,"error":error,"accepted":len(values),"model":model})
        except RuntimeError as exc:
            if "finish_reason=length" not in str(exc) and "empty_output" not in str(exc): raise
            passes.append([]); raws.append({"raw":"","error":str(exc),"accepted":0,"model":None})
    merged=merge_candidates(row["original_answer"],passes); anonymous=anonymize_candidates(row["task_id"],merged)
    select_payload={"question":row["question"],"evidence":docs,"candidates":[{"candidate_id":x["candidate_id"],"answer":x["answer"]} for x in anonymous]}
    try:
        raw,model,_=call(client,key,ledger,f"select_{evidence_name}",SELECT_SYSTEM,select_payload)
        selected,status,error=apply_selection(raw,anonymous,row["original_answer"],docs)
    except RuntimeError as exc:
        if "finish_reason=length" not in str(exc) and "empty_output" not in str(exc): raise
        raw="";model=None;selected=row["original_answer"];status="fallback_truncated_selection";error=str(exc)
    guarded,guard_status=conservative_boundary_guard(selected,original=row["original_answer"],candidates=anonymous)
    result.update({"candidate_passes":raws,"candidates":anonymous,"selector_raw":raw,"selected_answer":selected,
        "guarded_answer":guarded,"guard_status":guard_status,"selection_status":status,"selection_error":error,"actual_model":model})
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--inputs",type=Path,required=True); p.add_argument("--env-file",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--report",type=Path,required=True)
    p.add_argument("--limit",type=int,default=20); p.add_argument("--start",type=int,default=0); p.add_argument("--budget-cny",type=float,default=10); p.add_argument("--input-price",type=float,default=4); p.add_argument("--output-price",type=float,default=12); args=p.parse_args()
    all_records=[json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x]
    records=all_records[args.start:args.start+args.limit]; key=_env(args.env_file).get("DEEPSEEK_API_KEY")
    if not key: raise RuntimeError("DEEPSEEK_API_KEY missing")
    ledger=Ledger(args.budget_cny,args.input_price,args.output_price); completed=[]; started=time.perf_counter()
    with httpx.Client(timeout=300) as client:
        models=client.get("https://api.deepseek.com/models",headers={"Authorization":f"Bearer {key}"}); models.raise_for_status(); model_snapshot=models.json()
        for index,row in enumerate(records,1):
            item={"task_id":row["task_id"],"question":row["question"],"gold":row["gold"],"original_answer":row["original_answer"],"baseline":row["metrics"],"prompt_replay_exact":row["prompt_replay_exact"]}
            try:
                item["e0"]=run_mechanism(client,key,ledger,row,"e0",True)
                if row.get("e0_sha256") == row.get("e1_sha256"):
                    item["e1"]={**item["e0"],"evidence":"e1","equivalent_to_e0":True}
                else:
                    item["e1"]=run_mechanism(client,key,ledger,row,"e1",False)
                for name,answer in (("B",item["e0"].get("direct_answer") or row["original_answer"]),("C",item["e0"]["guarded_answer"]),("D",item["e1"]["guarded_answer"])): item[name]=answer_metrics(answer,row["gold"])
                item["status"]="ok"
            except Exception as exc: item.update(status="error",error=f"{type(exc).__name__}: {exc}")
            completed.append(item); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in completed),encoding="utf-8")
            args.report.write_text(json.dumps({"partial":True,"completed":len(completed),"cost_cny":ledger.cost,"calls":ledger.calls},ensure_ascii=False,indent=2),encoding="utf-8")
            print(f"{index}/{len(records)} {item['status']} cost={ledger.cost:.4f}",flush=True)
            if item["status"]=="error": break
    ok=[x for x in completed if x["status"]=="ok"]
    means=lambda k:{m:sum(x[k][m] for x in ok)/len(ok) if ok else None for m in ("strict_em","normalized_em","f1")}
    report={"version":"v109","phase":"training_prefix_mechanism","requested_model":"deepseek-flash","models_snapshot":model_snapshot,"limit":args.limit,"completed":len(completed),"successful":len(ok),"infrastructure_errors":len(completed)-len(ok),
        "start":args.start,"metrics":{"A":means("baseline"),"B":means("B"),"C":means("C"),"D":means("D")},"correct_original_preserve":{"B":sum(x["baseline"]["strict_em"] and x["B"]["strict_em"] for x in ok),"C":sum(x["baseline"]["strict_em"] and x["C"]["strict_em"] for x in ok),"D":sum(x["baseline"]["strict_em"] and x["D"]["strict_em"] for x in ok)},
        "protocol":{"candidate_passes":4*len(ok),"candidate_passes_schema_valid":sum(p["error"] is None for x in ok for e in (x["e0"],x["e1"]) for p in e["candidate_passes"]),"fallbacks":{name:sum(x[name.lower()]["selection_status"].startswith("fallback") for x in ok) for name in ("E0","E1")}},
        "usage":{"price_assumption":{"input_cny_per_million":args.input_price,"output_cny_per_million":args.output_price},"budget_cny":args.budget_cny,"estimated_cny":ledger.cost,"calls":len(ledger.calls),"details":ledger.calls},"elapsed_seconds":time.perf_counter()-started,"final_test_accessed":False}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({k:v for k,v in report.items() if k!="usage"},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
