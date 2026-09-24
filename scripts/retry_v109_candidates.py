"""Apply one generic retry to v109 pass-1 schema failures, then reselect."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import httpx
from mini_agent_rl.hybrid_answer import anonymize_candidates,answer_metrics,apply_selection,conservative_boundary_guard,merge_candidates,parse_candidates
from mini_agent_rl.training.thinking_sft import _env
from run_hybrid_v109 import Ledger,PASS1_SYSTEM,SELECT_SYSTEM,call
def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--results",type=Path,nargs="+",required=True);p.add_argument("--env-file",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--report",type=Path,required=True);p.add_argument("--budget-cny",type=float,required=True);args=p.parse_args()
    inputs={r["task_id"]:r for r in [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x]};by={}
    for path in args.results:
        for line in path.read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            if r.get("status")=="ok":by[r["task_id"]]=r
    failures=[r for r in by.values() if r["e0"]["candidate_passes"][0]["error"] is not None];key=_env(args.env_file).get("DEEPSEEK_API_KEY");ledger=Ledger(args.budget_cny,8,24);rows=[]
    correction=" Your prior response failed schema validation. Use exactly the requested keys candidates, answer, citations, document_id, and quote; include no other keys."
    with httpx.Client(timeout=300) as client:
        for index,r in enumerate(failures,1):
            item=inputs[r["task_id"]];payload={"question":item["question"],"evidence":item["e0"]}
            try:
                raw,model,_=call(client,key,ledger,"candidate_pass1_retry",PASS1_SYSTEM+correction,payload);values,error=parse_candidates(raw,item["e0"])
                if error is not None:raise RuntimeError(error)
                candidates=anonymize_candidates(item["task_id"],merge_candidates(item["original_answer"],[values]))
                select_payload={"question":item["question"],"evidence":item["e0"],"candidates":[{"candidate_id":x["candidate_id"],"answer":x["answer"]} for x in candidates]}
                select_raw,_,_=call(client,key,ledger,"selector_after_retry",SELECT_SYSTEM,select_payload);selected,status,select_error=apply_selection(select_raw,candidates,item["original_answer"],item["e0"]);guarded,guard_status=conservative_boundary_guard(selected,item["original_answer"],candidates)
                row={"task_id":item["task_id"],"status":"ok","retry_raw":raw,"candidates":candidates,"selected":selected,"guarded_answer":guarded,"selection_status":status,"selection_error":select_error,"guard_status":guard_status,"metrics":answer_metrics(guarded,item["gold"]),"model":model}
            except Exception as exc:row={"task_id":item["task_id"],"status":"error","error":f"{type(exc).__name__}: {exc}"}
            rows.append(row);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8");print(f"{index}/{len(failures)} {row['status']} cost={ledger.cost:.4f}",flush=True)
    report={"version":"v109","eligible_failures":len(failures),"retry_success":sum(r["status"]=="ok" for r in rows),"estimated_cny":ledger.cost,"calls":ledger.calls}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!="calls"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
