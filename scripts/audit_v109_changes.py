"""Blind, order-randomized evidence audit for changed v109 answers."""
from __future__ import annotations
import argparse,hashlib,json,random,time
from pathlib import Path
import httpx
from pydantic import BaseModel,ConfigDict
from mini_agent_rl.training.thinking_sft import _env
from run_hybrid_v109 import Ledger,call

SYSTEM="""Judge two anonymous answers to a question using only the supplied evidence. Determine whether each answer is semantically correct and evidence-supported, allowing standard aliases while requiring the requested entity, relation, date precision and scope. Return JSON {\"answer_1_correct\":true|false,\"answer_2_correct\":true|false,\"preferred\":\"answer_1\"|\"answer_2\"|\"tie\"|\"neither\",\"reason\":\"...\"}. Do not use outside knowledge."""
class Audit(BaseModel):
    model_config=ConfigDict(extra="forbid")
    answer_1_correct:bool;answer_2_correct:bool;preferred:str;reason:str
def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",type=Path,required=True);p.add_argument("--results",type=Path,nargs="+",required=True);p.add_argument("--env-file",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--report",type=Path,required=True);p.add_argument("--budget-cny",type=float,required=True);args=p.parse_args()
    inputs={r["task_id"]:r for r in [json.loads(x) for x in args.inputs.read_text(encoding="utf-8").splitlines() if x]};by={}
    for path in args.results:
        for line in path.read_text(encoding="utf-8").splitlines():
            r=json.loads(line)
            if r.get("status")=="ok":by[r["task_id"]]=r
    changed=[r for r in by.values() if r["e0"]["guarded_answer"]!=r["original_answer"]];key=_env(args.env_file).get("DEEPSEEK_API_KEY");ledger=Ledger(args.budget_cny,8,24);rows=[]
    with httpx.Client(timeout=300) as client:
        for index,r in enumerate(changed,1):
            order=[("baseline",r["original_answer"]),("hybrid",r["e0"]["guarded_answer"])]
            random.Random(int(hashlib.sha256(r["task_id"].encode()).hexdigest(),16)).shuffle(order)
            payload={"question":r["question"],"evidence":inputs[r["task_id"]]["e0"],"answer_1":order[0][1],"answer_2":order[1][1]}
            try:
                raw,model,_=call(client,key,ledger,"changed_pair_audit",SYSTEM,payload,4096);audit=Audit.model_validate_json(raw)
                correctness={order[0][0]:audit.answer_1_correct,order[1][0]:audit.answer_2_correct};preferred=(order[0][0] if audit.preferred=="answer_1" else order[1][0] if audit.preferred=="answer_2" else audit.preferred)
                row={"task_id":r["task_id"],"order":[x[0] for x in order],"answers":[x[1] for x in order],"correctness":correctness,"preferred":preferred,"reason":audit.reason,"model":model,"status":"ok"}
            except Exception as exc:row={"task_id":r["task_id"],"status":"error","error":f"{type(exc).__name__}: {exc}"}
            rows.append(row);args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),encoding="utf-8");print(f"{index}/{len(changed)} {row['status']} cost={ledger.cost:.4f}",flush=True)
            if row["status"]=="error":break
    ok=[r for r in rows if r["status"]=="ok"]
    report={"version":"v109","changed_pairs":len(changed),"completed":len(ok),"baseline_correct_hybrid_wrong":sum(r["correctness"]["baseline"] and not r["correctness"]["hybrid"] for r in ok),"baseline_wrong_hybrid_correct":sum(not r["correctness"]["baseline"] and r["correctness"]["hybrid"] for r in ok),"both_correct":sum(r["correctness"]["baseline"] and r["correctness"]["hybrid"] for r in ok),"both_wrong":sum(not r["correctness"]["baseline"] and not r["correctness"]["hybrid"] for r in ok),"estimated_cny":ledger.cost,"calls":ledger.calls}
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!="calls"},ensure_ascii=False,indent=2))
if __name__=="__main__":main()
