"""Validate and aggregate v121 anonymous DeepSeek reviews without exposing gold labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_v121_arms_api import build_packets, digest


def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--initial", required=True)
    ap.add_argument("--corrections", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    report_dir, task_path = Path(args.report_dir), Path(args.tasks)
    ra, rb = (json.loads((report_dir / f"arm-{arm}.json").read_text(encoding="utf-8")) for arm in "AB")
    tasks = {row["id"]: row for row in (json.loads(line) for line in task_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    packets = build_packets(ra, rb, tasks)
    all_records = read_jsonl(Path(args.initial)) + read_jsonl(Path(args.corrections))
    latest = {record["case_id"]: record for record in all_records}
    expected = {packet["case_id"]: packet for packet in packets}
    rows_by_case = {digest("v121:" + str(tasks[task_id]["question"]))[:20]: task_id
                    for task_id in {row["task_id"]: row for row in ra["rows"]}.keys()}
    aggregates = defaultdict(lambda: {"overall": Counter(), "decisions": defaultdict(Counter),
                                        "decision_count": 0, "invalid_traces": 0})
    invalid_cases, valid_cases, valid_traces = [], 0, 0
    correction_ids = {record["case_id"] for record in read_jsonl(Path(args.corrections))}
    for case_id, packet in expected.items():
        record = latest.get(case_id)
        case_result = ((record or {}).get("result", {}).get("cases") or [])
        returned = case_result[0] if case_result else {}
        trace_rows = {row.get("trace_id"): row for row in returned.get("traces", [])}
        errors = (record or {}).get("citation_errors", [])
        case_ok = record is not None and len(case_result) == 1 and len(trace_rows) == 2
        task_id = rows_by_case[case_id]
        is_swapped = int(digest(case_id)[0], 16) % 2 == 1
        for expected_trace in packet["traces"]:
            trace_id = expected_trace["trace_id"]
            trace = trace_rows.get(trace_id)
            arm = ("B" if trace_id == "trace-1" else "A") if is_swapped else ("A" if trace_id == "trace-1" else "B")
            expected_indices = {d["decision_index"] for d in expected_trace["decisions"]}
            actual_indices = {d.get("decision_index") for d in (trace or {}).get("decisions", [])}
            trace_errors = [e for e in errors if e.get("trace_id") == trace_id]
            trace_ok = bool(trace and trace.get("overall") in {"supported", "unsupported", "no_answer", "uncertain"}
                            and actual_indices == expected_indices and not trace_errors)
            if trace_ok:
                valid_traces += 1
                agg = aggregates[arm]
                agg["overall"][trace["overall"]] += 1
                for decision in trace.get("decisions", []):
                    agg["decision_count"] += 1
                    for field in ("evidence_complete", "query_targets_gap", "search_added_relation", "answer_supported"):
                        agg["decisions"][field][str(decision.get(field, "missing"))] += 1
            else:
                case_ok = False
                aggregates[arm]["invalid_traces"] += 1
        if case_ok:
            valid_cases += 1
        else:
            invalid_cases.append(case_id)

    output = {"version": "v121-blind-audit-summary-v1",
              "reviewer": "deepseek_api_blind",
              "source_sha256": {"arm_a": __import__("hashlib").sha256((report_dir / "arm-A.json").read_bytes()).hexdigest(),
                                "arm_b": __import__("hashlib").sha256((report_dir / "arm-B.json").read_bytes()).hexdigest(),
                                "tasks": __import__("hashlib").sha256(task_path.read_bytes()).hexdigest()},
              "coverage": {"expected_cases": 50, "valid_cases": valid_cases,
                           "expected_traces": 100, "valid_traces": valid_traces,
                           "corrective_review_cases": len(correction_ids),
                           "unresolved_case_ids": invalid_cases,
                           "all_cases_valid": valid_cases == 50 and valid_traces == 100},
              "citation_errors_latest": sum(len(latest[c].get("citation_errors", [])) for c in expected if c in latest),
              "by_arm": {arm: {"overall": dict(value["overall"]),
                               "decisions": {k: dict(v) for k, v in value["decisions"].items()},
                               "decision_count": value["decision_count"],
                               "invalid_traces": value["invalid_traces"]}
                         for arm, value in aggregates.items()},
              "opaque_case_ids_only": True,
              "gold_or_scores_in_api_packets": False}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
