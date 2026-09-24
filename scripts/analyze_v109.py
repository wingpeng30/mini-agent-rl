"""Merge resumable v109 shards and compute paired mechanism diagnostics."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from mini_agent_rl.answer_scoring import normalize_answer,token_f1

def bootstrap(values,repeats=10000):
    rng=random.Random(42);n=len(values);samples=[sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(repeats)] if n else []
    samples.sort();return {"mean":sum(values)/n if n else None,"low":samples[int(.025*repeats)] if samples else None,"high":samples[int(.975*repeats)] if samples else None}
def main():
    p=argparse.ArgumentParser();p.add_argument("--results",type=Path,nargs="+",required=True);p.add_argument("--cost-reports",type=Path,nargs="*",default=[]);p.add_argument("--output",type=Path,required=True);args=p.parse_args()
    by={}
    for path in args.results:
        for line in path.read_text(encoding="utf-8").splitlines():
            row=json.loads(line)
            if row.get("status")=="ok":by[row["task_id"]]=row
    rows=list(by.values()); metrics={}
    for name in ("baseline","B","C"):
        metrics[name]={key:sum(r[name][key] for r in rows)/len(rows) for key in ("strict_em","normalized_em","f1")}
    wrong=[r for r in rows if not r["baseline"]["strict_em"]]
    exact_recall=sum(any(c["answer"].strip()==r["gold"].strip() for c in r["e0"]["candidates"]) for r in wrong)
    norm_recall=sum(any(normalize_answer(c["answer"])==normalize_answer(r["gold"]) for c in r["e0"]["candidates"]) for r in wrong)
    total_cost=0;calls=0
    for path in args.cost_reports:
        report=json.loads(path.read_text(encoding="utf-8"));usage=report.get("usage") or {}
        total_cost+=float(usage.get("estimated_cny",report.get("cost_cny",0)));calls+=int(usage.get("calls",len(report.get("calls",[]))))
    report={"version":"v109","tasks":len(rows),"metrics":metrics,
        "paired_C_minus_A":{key:bootstrap([r["C"][key]-r["baseline"][key] for r in rows]) for key in ("strict_em","normalized_em","f1")},
        "transitions":{"A_wrong_C_right":sum(not r["baseline"]["strict_em"] and r["C"]["strict_em"] for r in rows),"A_right_C_wrong":sum(r["baseline"]["strict_em"] and not r["C"]["strict_em"] for r in rows),"A_right_C_right":sum(r["baseline"]["strict_em"] and r["C"]["strict_em"] for r in rows)},
        "wrong_original":len(wrong),"candidate_exact_recall_on_strict_wrong":exact_recall,"candidate_normalized_recall_on_strict_wrong":norm_recall,
        "candidate_best_f1":sum(max(token_f1(c["answer"],r["gold"]) for c in r["e0"]["candidates"]) for r in rows)/len(rows),
        "candidate_pass_protocol_valid":sum(p["error"] is None for r in rows for p in r["e0"]["candidate_passes"]),"candidate_pass_total":2*len(rows),
        "selection_fallbacks":sum(r["e0"]["selection_status"].startswith("fallback") for r in rows),"estimated_cny_including_failed_attempts":total_cost,"api_calls_including_failed_attempts":calls,"bootstrap_seed":42,"bootstrap_repeats":10000}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8");print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
