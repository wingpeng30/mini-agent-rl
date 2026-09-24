"""Rerun only the v109 selector on frozen paid candidate sets."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import httpx
from mini_agent_rl.hybrid_answer import answer_metrics,apply_selection
from mini_agent_rl.training.thinking_sft import _env
from run_hybrid_v109 import Ledger,SELECT_SYSTEM,call

def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--frozen-results",type=Path,required=True);p.add_argument("--env-file",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--report",type=Path,required=True);p.add_argument("--budget-cny",type=float,required=True);args=p.parse_args()
    source={r["task_id"]:r for r in [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x]}; frozen=[json.loads(x) for x in args.frozen_results.read_text(encoding="utf-8").splitlines() if x]
    key=_env(args.env_file).get("DEEPSEEK_API_KEY");ledger=Ledger(args.budget_cny,8,24);rows=[];started=time.perf_counter()
    with httpx.Client(timeout=300) as client:
        for index,old in enumerate(frozen,1):
            item=source[old["task_id"]]; candidates=old["e0"]["candidates"]; docs=item["e0"]
            payload={"question":item["question"],"evidence":docs,"candidates":[{"candidate_id":x["candidate_id"],"answer":x["answer"]} for x in candidates]}
            try:
                raw,model,_=call(client,key,ledger,"selector_v2",SELECT_SYSTEM,payload);answer,status,error=apply_selection(raw,candidates,item["original_answer"],docs)
                row={"task_id":item["task_id"],"question":item["question"],"gold":item["gold"],"original_answer":item["original_answer"],"answer":answer,"status":status,"error":error,"raw":raw,"model":model,"baseline":item["metrics"],"metrics":answer_metrics(answer,item["gold"])}
            except Exception as exc: row={"task_id":item["task_id"],"status":"infrastructure_error","error":f"{type(exc).__name__}: {exc}"}
            rows.append(row);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8");print(f"{index}/{len(frozen)} {row['status']} cost={ledger.cost:.4f}",flush=True)
            if row["status"]=="infrastructure_error":break
    ok=[r for r in rows if r["status"]!="infrastructure_error"]
    mean=lambda where,key:sum(r[where][key] for r in ok)/len(ok) if ok else None
    report={"version":"v109","phase":"selector_v2_training20","completed":len(rows),"successful":len(ok),"metrics":{"A":{k:mean("baseline",k) for k in ("strict_em","normalized_em","f1")},"C2":{k:mean("metrics",k) for k in ("strict_em","normalized_em","f1")}},"correct_original_preserved":sum(r["baseline"]["strict_em"] and r["metrics"]["strict_em"] for r in ok),"correct_original_total":sum(r["baseline"]["strict_em"] for r in ok),"cost_cny":ledger.cost,"calls":ledger.calls,"elapsed_seconds":time.perf_counter()-started}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!="calls"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
