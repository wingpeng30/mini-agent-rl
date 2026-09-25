"""Validate coverage and summarize the time-aligned v121 blind audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_v121_arms_api import build_packets, digest


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--reviews", required=True, nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report_dir, task_path = Path(args.report_dir), Path(args.tasks)
    report_a, report_b = [json.loads((report_dir / f"arm-{arm}.json").read_text(encoding="utf-8")) for arm in "AB"]
    task_rows = [json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = {row["id"]: row for row in task_rows}
    rows_by_arm = {arm: {row["task_id"]: row for row in report["rows"]}
                   for arm, report in zip("AB", [report_a, report_b])}
    packets = build_packets(report_a, report_b, tasks)
    records = {}
    corrected_case_ids = set()
    for review_path in args.reviews:
        review_path = Path(review_path)
        current = read_jsonl(review_path)
        records.update({record["case_id"]: record for record in current})
        if "corrections" in review_path.name:
            corrected_case_ids.update(record["case_id"] for record in current)
    counts = defaultdict(lambda: {"final_answer_support": Counter(),
                                 "decision_reviews": defaultdict(Counter),
                                 "search_reviews": defaultdict(Counter),
                                 "decision_count": 0, "search_count": 0,
                                 "invalid_traces": 0, "candidate_answer_count": 0,
                                 "trace_final_status_mismatches": 0})
    valid_traces, valid_cases, invalid_case_ids = 0, 0, []
    quote_error_count = 0
    for packet in packets:
        case_id = packet["case_id"]
        record = records.get(case_id)
        result = (record or {}).get("result", {})
        result_cases = result.get("cases") or ([result] if result.get("case_id") == case_id and isinstance(result.get("traces"), list) else [])
        api_case = result_cases[0] if result_cases else {}
        returned_traces = {trace.get("trace_id"): trace for trace in api_case.get("traces", [])}
        errors = list((record or {}).get("citation_errors", []))
        packet_by_trace = {trace["trace_id"]: trace for trace in packet["traces"]}
        for trace_id, trace in returned_traces.items():
            source_trace = packet_by_trace.get(trace_id, {})
            decision_sources = {item["decision_index"]: "\n".join(e["visible_text"] for e in item["visible_evidence"])
                                for item in source_trace.get("decisions", [])}
            for review in trace.get("decision_reviews", []):
                evidence_text = decision_sources.get(review.get("decision_index"), "")
                if any(not isinstance(quote, str) or not quote or quote not in evidence_text
                       for quote in review.get("support_quotes", [])):
                    errors.append({"trace_id": trace_id, "decision_index": review.get("decision_index")})
            search_sources = {item["search_index"]: "\n".join(
                [d["retrieved_text"] for d in item["top_results"]] + [d["visible_text"] for d in item["evidence_after"]])
                for item in source_trace.get("searches", [])}
            for review in trace.get("search_reviews", []):
                evidence_text = search_sources.get(review.get("search_index"), "")
                if any(not isinstance(quote, str) or not quote or quote not in evidence_text
                       for quote in review.get("quotes", [])):
                    errors.append({"trace_id": trace_id, "search_index": review.get("search_index")})
        quote_error_count += len(errors)
        case_ok = record is not None and len(result_cases) == 1 and len(returned_traces) == 2
        # Recover the local task key without putting it into API-facing packets.
        task_id = next(tid for tid, row in tasks.items() if digest("v121:" + str(row["question"]))[:20] == case_id)
        swapped = int(digest(case_id)[0], 16) % 2 == 1
        for input_trace in packet["traces"]:
            trace_id = input_trace["trace_id"]
            arm = (("B" if trace_id == "trace-1" else "A") if swapped else
                   ("A" if trace_id == "trace-1" else "B"))
            local_row = rows_by_arm[arm][task_id]
            trace = returned_traces.get(trace_id)
            trace_errors = [error for error in errors if error.get("trace_id") == trace_id]
            expected_decisions = {item["decision_index"] for item in input_trace["decisions"]}
            expected_searches = {item["search_index"] for item in input_trace["searches"]}
            actual_decisions = {item.get("decision_index") for item in (trace or {}).get("decision_reviews", [])}
            actual_searches = {item.get("search_index") for item in (trace or {}).get("search_reviews", [])}
            review_by_index = {item.get("decision_index"): item for item in (trace or {}).get("decision_reviews", [])}
            answer_indices = [item["decision_index"] for item in input_trace["decisions"]
                              if item.get("turn_purpose") in {"action_proposal", "answer_after_review"}
                              and item.get("action") == "answer"]
            final_index = answer_indices[-1] if answer_indices else None
            final_review_value = (review_by_index.get(final_index) or {}).get("answer_supported") if final_index else None
            has_candidate = bool(local_row.get("candidate_answer"))
            support_review_valid = (final_review_value in {"yes", "no", "uncertain"}) if has_candidate else True
            trace_ok = bool(trace and expected_decisions == actual_decisions and expected_searches == actual_searches
                            and not trace_errors and support_review_valid)
            if not trace_ok:
                case_ok = False
                counts[arm]["invalid_traces"] += 1
                continue
            valid_traces += 1
            agg = counts[arm]
            if has_candidate:
                agg["candidate_answer_count"] += 1
                agg["final_answer_support"][str(final_review_value)] += 1
                if trace.get("final_answer_supported") == "no_answer":
                    agg["trace_final_status_mismatches"] += 1
            else:
                agg["final_answer_support"]["no_answer_emitted"] += 1
                if trace.get("final_answer_supported") != "no_answer":
                    agg["trace_final_status_mismatches"] += 1
            for review in trace.get("decision_reviews", []):
                decision_input = input_trace["decisions"][review["decision_index"] - 1]
                purpose = decision_input.get("turn_purpose", "model_decision")
                target = agg["decision_reviews"][purpose]
                target["evidence_complete_" + str(review.get("evidence_complete", "missing"))] += 1
                target["action_appropriate_" + str(review.get("action_appropriate", "missing"))] += 1
                target["answer_supported_" + str(review.get("answer_supported", "missing"))] += 1
                agg["decision_count"] += 1
            for review in trace.get("search_reviews", []):
                target = agg["search_reviews"]["initial_shared" if input_trace["searches"][review["search_index"] - 1]["initial_shared"] else "model_search"]
                for field in ("query_grounded", "query_targets_gap", "retrieved_relation_present", "relation_visible_after"):
                    target[field + "_" + str(review.get(field, "missing"))] += 1
                agg["search_count"] += 1
        if case_ok:
            valid_cases += 1
        else:
            invalid_case_ids.append(case_id)
    output = {"version": "v121-time-aligned-blind-audit-summary-v1",
              "reviewer": "deepseek_api_blind",
              "source_sha256": {"arm_a": hashlib.sha256((report_dir / "arm-A.json").read_bytes()).hexdigest(),
                                "arm_b": hashlib.sha256((report_dir / "arm-B.json").read_bytes()).hexdigest(),
                                "tasks": hashlib.sha256(task_path.read_bytes()).hexdigest()},
              "coverage": {"expected_cases": 50, "valid_cases": valid_cases,
                           "expected_traces": 100, "valid_traces": valid_traces,
                           "all_cases_valid": valid_cases == 50 and valid_traces == 100,
                           "corrective_review_cases": len(corrected_case_ids),
                           "unresolved_case_ids": invalid_case_ids},
              "citation_errors": quote_error_count,
              "by_arm": {arm: {"final_answer_support": dict(value["final_answer_support"]),
                               "candidate_answer_count": value["candidate_answer_count"],
                               "decision_reviews": {purpose: dict(c) for purpose, c in value["decision_reviews"].items()},
                               "search_reviews": {kind: dict(c) for kind, c in value["search_reviews"].items()},
                               "decision_count": value["decision_count"],
                               "search_count": value["search_count"],
                               "trace_final_status_mismatches": value["trace_final_status_mismatches"],
                               "invalid_traces": value["invalid_traces"]}
                         for arm, value in counts.items()},
              "gold_scores_and_task_ids_sent_to_api": False}
    output_path = Path(args.output); output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
