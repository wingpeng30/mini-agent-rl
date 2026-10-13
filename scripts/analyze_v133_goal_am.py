"""CPU-only one-pass Goal AM evidence review and frozen route gate."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

import run_v133_goal_am as am
import finalize_v133_goal_al_report as prior
from prepare_v133_goal_ah import documents

OUT=am.OUT
read,rows,sha=am.read,am.rows,am.sha


def write(path,obj,jsonl=False):
    with Path(path).open("x",encoding="utf-8") as f:
        if jsonl:
            for x in obj:f.write(json.dumps(x,ensure_ascii=False)+"\n")
        else:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")


def combination(question,ids,answer):
    return hashlib.sha256(json.dumps([question,ids,answer],ensure_ascii=False).encode()).hexdigest()


def metric(items):
    calls=[t for r in items for t in r["trace"]]
    action_problem=lambda r:r["reason"] in ("invalid_action","repeated_action") or any(
        t.get("action")=="search" and "observation" not in t for t in r["trace"])
    return dict(cases=len(items),EM=sum(r["em"] for r in items),F1=statistics.mean(r["f1"] for r in items),
        invalid_or_repeated=sum(action_problem(r) for r in items),parse_invalid=sum(r["reason"]=="invalid_action" for r in items),
        repeated=sum(r["reason"]=="repeated_action" for r in items),
        overbudget_search_attempts=sum(t.get("action")=="search" and t["step"]>=3 for t in calls),
        calls=len(calls),searches=sum(r["searches"] for r in items),
        max_input=max(len(t["prompt_token_ids"]) for t in calls),max_output=max(len(t["action_token_ids"]) for t in calls),
        length_stops=sum(t.get("usage",{}).get("generation_stop_reason")=="length" for t in calls),
        body_trimmed=sum(t.get("usage",{}).get("lossy_body_trimmed",0) for t in calls),
        mean_question_seconds=statistics.mean(r["elapsed_seconds"] for r in items),
        mean_generation_seconds=statistics.mean(t["model_seconds"] for t in calls))


def packets(seed):
    tok=am.tokenizer_from_base()
    base_rows={r["case_id"]:r for r in rows(am.AL/"trajectory-B.jsonl")}
    basereview={r["case_id"]:r["B"] for r in rows(am.AL/"semantic-review.jsonl")}
    sources={r["case_id"]:r for r in rows(am.AL/"review-packets.jsonl")}
    manifests=read(am.AL/"manifest.json")
    result=[];probes=[];paired=[];summary={"GoalZ":metric(list(base_rows.values()))}
    for arm in ("A","B"):
        label=f"{seed}-{arm}"
        items=rows(OUT/f"eval-{label}.jsonl")
        if len(items)!=64 or [r["case_id"] for r in items]!=list(base_rows):raise RuntimeError("AM case order/count mismatch")
        summary[arm]=metric(items)
        for row in items:
            c=row["case_id"]
            base=base_rows[c]
            divergence=next((i for i,(a,b) in enumerate(zip(base["trace"],row["trace"]))
                             if a["raw"]!=b["raw"] or a["prompt_token_ids"]!=b["prompt_token_ids"]),None)
            divergence_state=None if divergence is None else dict(call=divergence,
                identical_input=base["trace"][divergence]["prompt_token_ids"]==row["trace"][divergence]["prompt_token_ids"],
                baseline_action=base["trace"][divergence].get("action"),candidate_action=row["trace"][divergence].get("action"),
                baseline_raw=base["trace"][divergence]["raw"],candidate_raw=row["trace"][divergence]["raw"])
            paired.append(dict(case_id=c,model=label,question=row["question"],baseline_answer=base["answer"],answer=row["answer"],
                baseline_em=base["em"],em=row["em"],baseline_reason=base["reason"],reason=row["reason"],first_divergence=divergence_state,
                admitted=c in manifests["admitted"],protected=c in manifests["protected"]))
            if c not in manifests["admitted"]:continue
            final=row["trace"][-1];ids=final["prompt_token_ids"]
            citations=[]
            for edge in sources[c]["edges"]:
                # Use explicitly linked minimal quotations from the AL review;
                # never infer sufficiency from case position or answer hit.
                candidates=[(edge["document_id"],edge["quote"])]
                candidates += [(e["citation"]["document_id"],e["citation"]["quote"])
                               for e in basereview[c]["edges"] if e["edge_id"]==edge["edge_id"] and e.get("citation")]
                found=next((x for d,q in candidates if (x:=prior.cite(ids,tok,d,q)) is not None),None)
                first_return=None
                for t in row["trace"]:
                    if any(d.get("id")==docid and quote in d.get("text","") for docid,quote in candidates
                           for d in t.get("observation",{}).get("result",[])):
                        first_return=t["step"];break
                citations.append(dict(edge_id=edge["edge_id"],subject=edge["subject"],relation=edge["relation"],object=edge["object"],
                    actual_visible=found is not None,citation=found,first_return_search=first_return))
            key=combination(row["question"],ids,row["answer"])
            same=ids==base["trace"][-1]["prompt_token_ids"] and row["answer"]==base["answer"]
            result.append(dict(case_id=c,model=label,question=row["question"],answer=row["answer"],official_answer=row["official_answer"],
                reason=row["reason"],queries=[t["query"] for t in row["trace"] if t.get("query")],
                complete_chain_visible=all(e["actual_visible"] for e in citations),edges=citations,
                combination_sha256=key,same_as_goal_z=same,baseline_answer=base["answer"],baseline_category=basereview[c]["category"],
                reused_exact_verdict=basereview[c] if same else None,first_divergence=divergence_state,
                documents=documents(tok.decode(ids,skip_special_tokens=False)),prompt_token_ids=ids,
                trace=row["trace"],training_use_forbidden=True))
        frozen={x["probe_id"]:x for x in rows(OUT/"frozen-probes.jsonl")}
        for r in rows(OUT/f"probe-{label}.jsonl"):
            src=frozen[r["probe_id"]]
            if r["prompt_token_ids"]!=src["prompt_token_ids"]:raise RuntimeError("probe exact-input mismatch")
            # The collector used content-is-not-None, while ModelResponse's
            # content defaults to an empty string. Correct only this metadata
            # from the already saved parse error/query; never rerun inference.
            original_label=r["action"]
            r["action"]="invalid" if r.get("parse_error") else "search" if r.get("query") is not None else "answer"
            probes.append(dict(model=label,**r,baseline_category=src.get("baseline_category"),
                baseline_answer=src.get("baseline_answer"),baseline_action=src.get("baseline_action"),question=src["question"],
                archived_action_label=original_label,action_label_mechanically_corrected=original_label!=r["action"],
                documents=documents(tok.decode(r["prompt_token_ids"],skip_special_tokens=False)),training_use_forbidden=True))
    write(OUT/f"review-packets-seed{seed}.jsonl",result,True)
    write(OUT/f"paired-ledger-seed{seed}.jsonl",paired,True)
    write(OUT/f"probe-review-packets-seed{seed}.jsonl",probes,True)
    write(OUT/f"mechanical-metrics-seed{seed}.json",summary)
    print(json.dumps(dict(seed=seed,metrics=summary,answer_combinations=len(result),probe_outputs=len(probes)),ensure_ascii=False))


def inspect(seed,case=None):
    for r in rows(OUT/f"review-packets-seed{seed}.jsonl"):
        if case and r["case_id"]!=case:continue
        print(json.dumps({k:r[k] for k in ("model","case_id","question","answer","baseline_answer","baseline_category",
            "reason","complete_chain_visible","same_as_goal_z","queries","edges")},ensure_ascii=False))
        if not r["same_as_goal_z"]:
            print(json.dumps(dict(documents=r["documents"]),ensure_ascii=False))


def finalize(seed):
    corrected=OUT/f"review-packets-seed{seed}-corrected.jsonl"
    packets=rows(corrected if corrected.exists() else OUT/f"review-packets-seed{seed}.jsonl")
    probe_packets=rows(OUT/f"probe-review-packets-seed{seed}.jsonl")
    rulings=read(OUT/f"primary-adjudications-seed{seed}.json")
    basereview={r["case_id"]:r["B"] for r in rows(am.AL/"semantic-review.jsonl")}
    reviews=[];probe_reviews=[];seen={}
    for r in packets:
        label=r["model"];c=r["case_id"]
        verdict=r["reused_exact_verdict"] or rulings["answers"][label][c]
        category=verdict["category"]
        if category=="success" and not r["complete_chain_visible"]:
            raise RuntimeError("success requires explicitly cited complete actual-input chain: "+label+":"+c)
        key=r["combination_sha256"]
        if key in seen and seen[key]!=category:raise RuntimeError("same combination received inconsistent adjudication")
        seen[key]=category
        reviews.append(dict(case_id=c,model=label,answer=r["answer"],category=category,reason=verdict["reason"],
            entity_relation_mismatch=verdict.get("entity_relation_mismatch",False),complete_chain_visible=r["complete_chain_visible"],
            edges=r["edges"],combination_sha256=key,reused_exact_verdict=r["same_as_goal_z"],training_use_forbidden=True))
    for r in probe_packets:
        if r["category"]=="missing_link":
            verdict=dict(correct=r["action"]=="search",premature_answer=r["action"]=="answer",reason="frozen necessary edge missing; next action only")
        else:
            verdict=rulings["probes"][r["model"]][r["probe_id"]]
            if verdict["correct"] and r["action"]!="answer":raise RuntimeError("complete probe success must answer")
        probe_reviews.append(dict(model=r["model"],probe_id=r["probe_id"],category=r["category"],answer=r["answer"],action=r["action"],**verdict))
    metrics=read(OUT/f"mechanical-metrics-seed{seed}.json")
    manifest=read(am.AL/"manifest.json")
    base_success={c for c,r in basereview.items() if r["category"]=="success"}
    base_errors=sum(r["category"] in ("factual_error","unsupported_by_visible_evidence") for r in basereview.values())
    baseline_probe=sum(r["baseline_category"]=="success" for r in rows(OUT/"frozen-probes.jsonl") if r["category"]=="complete")
    gates={};results={};changes=[]
    for arm in ("A","B"):
        label=f"{seed}-{arm}";rr=[r for r in reviews if r["model"]==label];pp=[r for r in probe_reviews if r["model"]==label]
        if len(rr)!=29 or len(pp)!=32:raise RuntimeError("review denominator incomplete")
        success={r["case_id"] for r in rr if r["category"]=="success"};counts=Counter(r["category"] for r in rr)
        errors=counts["factual_error"]+counts["unsupported_by_visible_evidence"]
        candidate={r["case_id"]:r for r in rows(OUT/f"eval-{label}.jsonl")}
        protected_lost=[c for c in manifest["protected"] if c not in success]
        complete_correct=sum(r["correct"] for r in pp if r["category"]=="complete")
        premature=sum(r.get("premature_answer",False) for r in pp if r["category"]=="missing_link")
        gates[arm]=dict(supported_net_plus_two=len(success)-len(base_success)>=2,errors_not_higher=errors<=base_errors,
            invalid_or_repeated_not_higher=metrics[arm]["invalid_or_repeated"]<=metrics["GoalZ"]["invalid_or_repeated"],
            parse_invalid_not_higher=metrics[arm]["parse_invalid"]<=metrics["GoalZ"]["parse_invalid"],
            repeated_not_higher=metrics[arm]["repeated"]<=metrics["GoalZ"]["repeated"],
            overbudget_not_higher=metrics[arm]["overbudget_search_attempts"]<=metrics["GoalZ"]["overbudget_search_attempts"],
            protected_losses_at_most_one=len(protected_lost)<=1,complete_probe_gain_two=complete_correct-baseline_probe>=2,
            missing_probe_premature_not_higher=premature==0)
        results[arm]=dict(supported_success=len(success),categories=dict(counts),errors=errors,
            complete_chain_visible=sum(r["complete_chain_visible"] for r in rr),entity_relation_mismatch=sum(r["entity_relation_mismatch"] for r in rr),
            complete_probe_correct=complete_correct,missing_probe_premature=premature,protected_lost=protected_lost,
            gained=sorted(success-base_success),lost=sorted(base_success-success),passed=all(gates[arm].values()))
        changes += [dict(model=label,case_id=c,direction="gained" if c in success else "lost",
                         packet=next(p for p in packets if p["model"]==label and p["case_id"]==c))
                    for c in sorted(success^base_success)]
    route_passed=any(v["passed"] for v in results.values())
    write(OUT/f"semantic-review-seed{seed}.jsonl",reviews,True)
    write(OUT/f"probe-review-seed{seed}.jsonl",probe_reviews,True)
    write(OUT/f"mechanism-changes-seed{seed}.json",changes)
    gate=dict(seed=seed,route_passed=route_passed,baseline_supported=19,baseline_errors=base_errors,
              baseline_complete_probe_correct=baseline_probe,results=results,gates=gates,metrics=metrics,
              reviewed_unique_combinations=len(seen),released=False)
    write(OUT/f"seed{seed}-gate.json",gate)
    report=[f"# Goal AM seed{seed}结果","",f"固定64题，准入29题；Goal Z复用AL大窗口。探索门槛通过：{route_passed}。未确认35题仅报告EM/F1。","",
            "| 指标 | Goal Z | A普通状态SFT | B字段加权SFT |","|---|---:|---:|---:|"]
    for name,base,a,b in (("EM /64",metrics["GoalZ"]["EM"],metrics["A"]["EM"],metrics["B"]["EM"]),
        ("支持成功 /29",19,results["A"]["supported_success"],results["B"]["supported_success"]),
        ("事实错误+证据不支持",base_errors,results["A"]["errors"],results["B"]["errors"]),
        ("无效/重复/预算外动作题",metrics["GoalZ"]["invalid_or_repeated"],metrics["A"]["invalid_or_repeated"],metrics["B"]["invalid_or_repeated"]),
        ("完整链探针正确 /24",baseline_probe,results["A"]["complete_probe_correct"],results["B"]["complete_probe_correct"]),
        ("缺链探针提前回答 /8",0,results["A"]["missing_probe_premature"],results["B"]["missing_probe_premature"])):
        report.append(f"| {name} | {base} | {a} | {b} |")
    report += ["", "A/B都看过相同128轨迹+34状态并各更新128次；B相对A才衡量字段加权收益。A相对Goal Z测量继续训练加状态监督的整个配方，不能单独归因于新增34状态。",""]
    for arm in ("A","B"):
        report += [f"{arm}新增成功：{results[arm]['gained']}。丢失成功：{results[arm]['lost']}。",
                   f"{arm}门槛：{gates[arm]}。",""]
    report += ["通过仅继续seed43冻结复现；未通过则停止，不扩样本或改参。产品发布仍需完整回归与新确认。",""]
    (OUT/f"REPORT-seed{seed}.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(gate,ensure_ascii=False))


def correct_citations(seed):
    """Apply explicitly reviewed quote pointers; preserve original packets."""
    tok=am.tokenizer_from_base()
    corrections=read(OUT/"citation-corrections.json")
    overrides={(r["case_id"],r["edge_id"]):r for r in corrections}
    packets=rows(OUT/f"review-packets-seed{seed}.jsonl")
    changes=[]
    for r in packets:
        for e in r["edges"]:
            correction=overrides.get((r["case_id"],e["edge_id"]))
            if correction is None:continue
            citation=prior.cite(r["prompt_token_ids"],tok,correction["document_id"],correction["quote"])
            if citation is None:raise RuntimeError("reviewed quote missing in actual input")
            changes.append(dict(case_id=r["case_id"],model=r["model"],edge_id=e["edge_id"],
                previous_citation=e["citation"],corrected_citation=citation,reason=correction["reason"]))
            e["citation"]=citation;e["actual_visible"]=True
            e["first_return_search"]=next((t["step"] for t in r["trace"] if any(
                d.get("id")==correction["document_id"] and correction["quote"] in d.get("text","")
                for d in t.get("observation",{}).get("result",[]))),None)
        r["complete_chain_visible"]=all(e["actual_visible"] for e in r["edges"])
    probe_evidence=[]
    templates={r["case_id"]:r for r in packets if r["model"]==f"{seed}-A"}
    for p in rows(OUT/f"probe-review-packets-seed{seed}.jsonl"):
        if p["category"]!="complete":continue
        case=p["probe_id"].removeprefix("complete:")
        edges=[]
        for e in templates[case]["edges"]:
            c=e["citation"]
            found=prior.cite(p["prompt_token_ids"],tok,c["document_id"],c["quote"])
            if found is None:raise RuntimeError("complete probe necessary quotation not visible")
            edges.append(dict(edge_id=e["edge_id"],subject=e["subject"],relation=e["relation"],object=e["object"],citation=found))
        probe_evidence.append(dict(model=p["model"],probe_id=p["probe_id"],edges=edges,training_use_forbidden=True))
    write(OUT/f"review-packets-seed{seed}-corrected.jsonl",packets,True)
    write(OUT/f"citation-revisions-seed{seed}.json",changes)
    write(OUT/f"probe-evidence-seed{seed}.jsonl",probe_evidence,True)
    print(json.dumps(dict(corrected_pointers=len(changes),complete_probe_evidence=len(probe_evidence))))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["packets","inspect","correct-citations","finalize"])
    p.add_argument("--seed",type=int,choices=[42,43],default=42);p.add_argument("--case")
    a=p.parse_args()
    if a.stage=="packets":packets(a.seed)
    elif a.stage=="inspect":inspect(a.seed,a.case)
    elif a.stage=="correct-citations":correct_citations(a.seed)
    else:finalize(a.seed)


if __name__=="__main__":main()
