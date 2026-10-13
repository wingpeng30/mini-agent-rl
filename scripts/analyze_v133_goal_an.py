"""CPU-only AN evidence packets and reviewed gates; no calls or searches."""
from collections import Counter
import argparse
import json
import run_v133_goal_an as an
import analyze_v133_goal_am as am_review
import finalize_v133_goal_al_report as prior
from prepare_v133_goal_ah import documents

OUT=an.OUT
read,rows,write=an.read,an.rows,an.write


def evaluation(label):
    return rows(an.AM/"eval-42-A.jsonl") if label=="42-A" else rows(OUT/f"eval-{label}.jsonl")


def complete(label):
    if label=="42-A":
        return [r for r in rows(an.AM/"probe-review-seed42.jsonl") if r["model"]=="42-A" and r["category"]=="complete"]
    return rows(OUT/f"probe-{label}-complete.jsonl")


def packets(seed):
    cfg=an.frozen()
    tok=an.am.tokenizer_from_base()
    sources={r["case_id"]:r for r in rows(an.AM/"review-packets-seed42-corrected.jsonl") if r["model"]=="42-A"}
    historical={r["case_id"]:r for r in rows(an.AM/"semantic-review-seed42.jsonl") if r["model"]=="42-A"}
    old={r["case_id"]:r for r in evaluation("42-A")}
    result=[];pair=[];metrics={"42-A":am_review.metric(list(old.values()))}
    labels=["42-B"] if seed==42 else ["43-A","43-B"]
    for label in labels:
        cases=evaluation(label)
        if len(cases)!=64 or [r["case_id"] for r in cases]!=list(old):raise RuntimeError("AN case order/count")
        metrics[label]=am_review.metric(cases)
        for r in cases:
            baseline=old[r["case_id"]]
            first=next((i for i,(x,y) in enumerate(zip(baseline["trace"],r["trace"]))
                if x["raw"]!=y["raw"] or x["prompt_token_ids"]!=y["prompt_token_ids"]),None)
            pair.append(dict(model=label,case_id=r["case_id"],question=r["question"],AM_answer=baseline["answer"],
                answer=r["answer"],em=r["em"],AM_em=baseline["em"],first_divergence=first,
                divergence=None if first is None else dict(same_input=baseline["trace"][first]["prompt_token_ids"]==r["trace"][first]["prompt_token_ids"],
                    previous_raw=baseline["trace"][first]["raw"],candidate_raw=r["trace"][first]["raw"])))
            if r["case_id"] not in sources:continue
            ids=r["trace"][-1]["prompt_token_ids"];edges=[]
            for e in sources[r["case_id"]]["edges"]:
                c=e["citation"]
                found=prior.cite(ids,tok,c["document_id"],c["quote"]) if c else None
                first_return=next((t["step"] for t in r["trace"] if c and any(d.get("id")==c["document_id"] and c["quote"] in d.get("text","")
                    for d in t.get("observation",{}).get("result",[]))),None)
                edges.append(dict(edge_id=e["edge_id"],subject=e["subject"],relation=e["relation"],object=e["object"],citation=found,
                    actual_visible=found is not None,first_return_search=first_return))
            same=ids==baseline["trace"][-1]["prompt_token_ids"] and r["answer"]==baseline["answer"]
            result.append(dict(model=label,case_id=r["case_id"],question=r["question"],answer=r["answer"],
                complete_chain_visible=all(e["actual_visible"] for e in edges),edges=edges,
                combination_sha256=am_review.combination(r["question"],ids,r["answer"]),
                reused_verdict=historical[r["case_id"]] if same else None,
                documents=documents(tok.decode(ids,skip_special_tokens=False)),prompt_token_ids=ids,
                training_use_forbidden=True))
    probes=[]
    partial={r["probe_id"]:r for r in rows(OUT/"partial-probes.jsonl")}
    labels_for_partial=["GoalZ","42-A","42-B"] if seed==42 else ["43-A","43-B"]
    for label in labels_for_partial:
        r=rows(OUT/f"probe-{label}-partial.jsonl")
        if len(r)!=8:raise RuntimeError("8 partial outputs required")
        for p in r:
            source=partial[p["probe_id"]]
            if p["prompt_token_ids"]!=source["prompt_token_ids"]:raise RuntimeError("partial input mismatch")
            repeated=p["action"]=="search" and any(an.am.norm(p["query"])==an.am.norm(q) for q in source["previous_queries"])
            probes.append(dict(**p,question=source["question"],bridge_entity=source["bridge_entity"],missing_relation=source["missing_relation"],
                documents=documents(tok.decode(p["prompt_token_ids"],skip_special_tokens=False)),repeated_query=repeated,
                previous_queries=source["previous_queries"],first_edge_citation=source["first_edge_citation"],training_use_forbidden=True))
    for label in (["42-B"] if seed==42 else ["43-A","43-B"]):
        rr=complete(label)
        if len(rr)!=24:raise RuntimeError("24 complete outputs required")
        probes += [dict(**p,documents=documents(tok.decode(p["prompt_token_ids"],skip_special_tokens=False)),training_use_forbidden=True) for p in rr]
    write(OUT/f"review-packets-seed{seed}.jsonl",result,True)
    write(OUT/f"probe-review-packets-seed{seed}.jsonl",probes,True)
    write(OUT/f"paired-ledger-seed{seed}.jsonl",pair,True)
    write(OUT/f"mechanical-metrics-seed{seed}.json",metrics)
    print(json.dumps(dict(seed=seed,answer_combinations=len(result),probe_outputs=len(probes),metrics=metrics)))


def finalize(seed):
    cfg=an.frozen();rulings=read(OUT/f"primary-adjudications-seed{seed}.json")
    verdicts=[];seen={}
    for p in rows(OUT/f"review-packets-seed{seed}.jsonl"):
        v=p["reused_verdict"] or rulings["answers"][p["model"]][p["case_id"]]
        if v["category"]=="success" and not p["complete_chain_visible"]:raise RuntimeError("success requires quoted complete chain")
        key=p["combination_sha256"]
        if key in seen and seen[key]!=v["category"]:raise RuntimeError("same combination differs")
        seen[key]=v["category"]
        verdicts.append(dict(model=p["model"],case_id=p["case_id"],answer=p["answer"],category=v["category"],reason=v["reason"],
            entity_relation_mismatch=v.get("entity_relation_mismatch",False),edges=p["edges"],complete_chain_visible=p["complete_chain_visible"]))
    metrics=read(OUT/f"mechanical-metrics-seed{seed}.json")
    baseline=[r for r in rows(an.AM/"semantic-review-seed42.jsonl") if r["model"]=="42-A"]
    baseline_complete=complete("42-A")
    reviews={};scores={}
    labels=["42-A","42-B"] if seed==42 else ["43-A","43-B"]
    # GoalZ partial baseline is diagnostic, not substituted for matched A.
    partial_labels=["GoalZ"]+labels if seed==42 else labels
    for label in partial_labels:
        items=[r for r in rows(OUT/f"probe-review-packets-seed{seed}.jsonl") if r["model"]==label and r["category"]=="partial"]
        scored=[]
        for p in items:
            v=rulings["partial"][label][p["probe_id"]]
            legal=v["legal_missing_relation_search"]
            if legal and (p["action"]!="search" or p["repeated_query"]):raise RuntimeError("invalid/repeated cannot be legal repair search")
            scored.append(dict(probe_id=p["probe_id"],action=p["action"],repeated_query=p["repeated_query"],**v))
        if len(scored)!=8:raise RuntimeError("partial adjudication incomplete")
        reviews[label]=dict(partial=scored)
    for label in labels:
        rr=baseline if label=="42-A" else [r for r in verdicts if r["model"]==label]
        pp=baseline_complete if label=="42-A" else complete(label)
        if len(rr)!=29 or len(pp)!=24:raise RuntimeError("review denominator")
        cc=baseline_complete if label=="42-A" else [dict(probe_id=p["probe_id"],**rulings["complete"][label][p["probe_id"]]) for p in pp]
        if label!="42-A":
            for p,c in zip(pp,cc):
                if c["correct"] and p["action"]!="answer":raise RuntimeError("correct complete state must answer")
        reviews[label]["complete"]=cc
        successes={r["case_id"] for r in rr if r["category"]=="success"}
        cats=Counter(r["category"] for r in rr)
        pr=reviews[label]["partial"]
        scores[label]=dict(supported=len(successes),success_ids=sorted(successes),errors=cats["factual_error"]+cats["unsupported_by_visible_evidence"],
            complete=sum(c["correct"] for c in cc),binding_repairs=sum(c.get("binding_repair",False) and c["correct"] for c in cc),
            protected_lost=sorted(set(cfg["protection"])-successes),must_retain_lost=sorted(set(cfg["must_retain"])-successes),
            premature=sum(p["action"]=="answer" for p in pr),invalid=sum(p["action"]=="invalid" for p in pr),
            repeated=sum(p["repeated_query"] for p in pr),legal_search=sum(p["legal_missing_relation_search"] for p in pr),
            categories=dict(cats),complete_chain_visible=sum(r.get("complete_chain_visible",False) for r in rr))
    A=scores[f"{seed}-A"];B=scores[f"{seed}-B"];ma=metrics[f"{seed}-A"];mb=metrics[f"{seed}-B"]
    # A binding repair must be a new correct probe relative to matched A.
    ac={p["probe_id"]:p["correct"] for p in reviews[f"{seed}-A"]["complete"]}
    binding_new=sum(p["correct"] and p.get("binding_repair",False) and not ac[p["probe_id"]] for p in reviews[f"{seed}-B"]["complete"])
    gate=dict(supported_at_least_23=B["supported"]>=23,complete_at_least_22=B["complete"]>=22,
        relative_supported_gain_two=B["supported"]-A["supported"]>=2,relative_complete_gain_two=B["complete"]-A["complete"]>=2,
        new_binding_repair=binding_new>=1,errors_not_higher=B["errors"]<=7 and B["errors"]<=A["errors"],
        action_cases_not_higher=mb["invalid_or_repeated"]<=1 and mb["invalid_or_repeated"]<=ma["invalid_or_repeated"],
        parse_not_higher=mb["parse_invalid"]<=ma["parse_invalid"],repeat_not_higher=mb["repeated"]<=ma["repeated"],
        overbudget_not_higher=mb["overbudget_search_attempts"]<=ma["overbudget_search_attempts"],
        protected_loss_at_most_one=len(B["protected_lost"])<=1,AM_gains_preserved=not B["must_retain_lost"],
        partial_premature_not_higher=B["premature"]<=A["premature"],partial_invalid_not_higher=B["invalid"]<=A["invalid"],
        partial_repeat_not_higher=B["repeated"]<=A["repeated"],partial_legal_search_not_lower=B["legal_search"]>=A["legal_search"])
    passed=all(gate.values())
    result=dict(seed=seed,route_passed=passed,gates=gate,scores=scores,new_binding_repairs=binding_new,metrics=metrics,released=False)
    write(OUT/f"semantic-review-seed{seed}.jsonl",verdicts,True)
    write(OUT/f"probe-review-seed{seed}.json",reviews)
    write(OUT/f"seed{seed}-gate.json",result)
    (OUT/f"REPORT-seed{seed}.md").write_text(f"# Goal AN seed{seed}\n\n探索通过：{passed}。未发布。\n\n```json\n"+json.dumps(result,ensure_ascii=False,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["packets","finalize"]);p.add_argument("--seed",type=int,choices=[42,43],default=42)
    a=p.parse_args()
    if a.stage=="packets":packets(a.seed)
    else:finalize(a.seed)

if __name__=="__main__":main()
