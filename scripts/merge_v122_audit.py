"""Validate independent v122 reviews and gate full end-to-end development replay."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.audit_v121_arms_api import digest
from scripts.audit_v122_api import packets
from scripts.replay_v119_bridge import read, save


def load_unique(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {row["case_id"]:row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate_review_case_{path.name}")
    return result


def review_for(packet: dict, record: dict | None) -> tuple[dict | None, list[str]]:
    problems = []
    if not record or record.get("packet_sha256") != digest(json.dumps(packet,ensure_ascii=False,sort_keys=True)):
        return None, ["missing_or_mismatched_packet"]
    result = record.get("result") or {}
    cases = result.get("cases") or ([result] if result.get("case_id") == packet["case_id"] else [])
    if len(cases) != 1 or cases[0].get("case_id") != packet["case_id"] or len(cases[0].get("traces",[])) != 1:
        return None, ["invalid_case_or_trace_count"]
    source = packet["traces"][0]
    trace = cases[0]["traces"][0]
    if trace.get("trace_id") != source["trace_id"]:
        problems.append("trace_id_mismatch")
    decision_sources = {d["decision_index"]: "\n".join(e["visible_text"] for e in d["visible_evidence"])
                        for d in source["decisions"]}
    searches = {s["search_index"]: "\n".join([d["retrieved_text"] for d in s["top_results"]] +
                                                   [e["visible_text"] for e in s["evidence_after"]])
                for s in source["searches"]}
    decisions = trace.get("decision_reviews",[])
    search_reviews = trace.get("search_reviews",[])
    if {d.get("decision_index") for d in decisions} != set(decision_sources) or len(decisions) != len(decision_sources):
        problems.append("decision_coverage")
    if {s.get("search_index") for s in search_reviews} != set(searches) or len(search_reviews) != len(searches):
        problems.append("search_coverage")
    for d in decisions:
        visible = decision_sources.get(d.get("decision_index"), "")
        if any(not isinstance(q,str) or not q or q not in visible for q in d.get("support_quotes",[])):
            problems.append("invalid_decision_quote")
    for s in search_reviews:
        available = searches.get(s.get("search_index"), "")
        if any(not isinstance(q,str) or not q or q not in available for q in s.get("quotes",[])):
            problems.append("invalid_search_quote")
    last_answer = next((d["decision_index"] for d in reversed(source["decisions"])
                        if d["turn_purpose"] == "action_proposal" and d["action"] == "answer"),None)
    by_index = {d.get("decision_index"):d for d in decisions}
    final_support = by_index.get(last_answer,{}).get("answer_supported") if last_answer else "no_answer"
    if final_support not in {"yes","no","uncertain","no_answer"}:
        problems.append("final_support_missing")
    if record.get("citation_errors"):
        problems.append("reviewer_reported_quote_error")
    reported_final = trace.get("final_answer_supported")
    if last_answer is None and reported_final != "no_answer":
        problems.append("no_answer_status_mismatch")
    return ({"final_support":final_support,"decisions":decisions,"searches":search_reviews}
            if not problems else None), sorted(set(problems))


def merge(report_dir: Path, tasks_path: Path, reviewer_a: Path, reviewer_b: Path, selected_arm: str | None) -> dict:
    tasks = {row["id"]:row for row in (json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    expected = packets(report_dir,tasks)
    a,b = load_unique(reviewer_a),load_unique(reviewer_b)
    all_ids = {packet["case_id"] for packet in expected}
    if set(a)-all_ids or set(b)-all_ids:
        raise ValueError("review_contains_unknown_case")
    by_arm = {arm:Counter() for arm in "ABCD"}
    unresolved = []
    errors = []
    for arm in "ABCD":
        report = read(report_dir / f"fixed-{arm}.json")
        rows = {r["task_id"]:r for r in report["rows"]}
        for task_id,row in rows.items():
            question = tasks[task_id]["question"]
            case_id = digest("v122:" + arm + ":" + question)[:24]
            packet = next(p for p in expected if p["case_id"] == case_id)
            left,left_errors = review_for(packet,a.get(case_id))
            right,right_errors = review_for(packet,b.get(case_id))
            if left_errors or right_errors:
                errors.append({"case_id":case_id,"arm":arm,"a":left_errors,"b":right_errors})
                continue
            by_arm[arm]["valid"] += 1
            ls,rs = left["final_support"],right["final_support"]
            if ls != rs or ls == "uncertain":
                unresolved.append({"case_id":case_id,"arm":arm,"a":ls,"b":rs})
            else:
                by_arm[arm]["support_"+ls] += 1
                if row.get("baseline_em") == 1 and ls == "no":
                    by_arm[arm]["confirmed_baseline_semantic_degradation"] += 1
                if row.get("candidate_em") == 1 and ls == "no":
                    by_arm[arm]["correct_but_unsupported"] += 1
    complete = len(errors) == 0 and sum(v["valid"] for v in by_arm.values()) == 200
    selected = by_arm.get(selected_arm,Counter())
    selected_disputes = sum(item["arm"] == selected_arm for item in unresolved)
    degradation = selected["confirmed_baseline_semantic_degradation"] > 0 or selected["correct_but_unsupported"] > 0
    return {"version":"v122-dual-audit-v1","settings_sha256":read(report_dir / "fixed-A.json")["settings_sha256"],
            "arm":selected_arm,
            "complete_coverage":complete,"unresolved_support_disputes":selected_disputes,
            "confirmed_semantic_degradation":degradation,
            "approved_for_full_e2e":bool(selected_arm and complete and selected_disputes == 0 and not degradation),
            "by_arm":{arm:dict(value) for arm,value in by_arm.items()},
            "invalid_reviews":errors,"unresolved_reviews":unresolved,
            "gold_scores_task_ids_sent_to_reviewers":False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir",type=Path,required=True)
    parser.add_argument("--tasks",type=Path,required=True)
    parser.add_argument("--reviewer-a",type=Path,required=True)
    parser.add_argument("--reviewer-b",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    selected = read(args.report_dir / "fixed-summary.json").get("selected_arm")
    result = merge(args.report_dir,args.tasks,args.reviewer_a,args.reviewer_b,selected)
    save(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in {"invalid_reviews","unresolved_reviews"}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
