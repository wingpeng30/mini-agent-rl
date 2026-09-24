"""Offline v109 rescore with visible failures and explicit retry precedence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mini_agent_rl.hybrid_v110 import answer_metrics


def load(paths):
    records = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                row = json.loads(line)
                records.append({"source": str(path), "line": number, "sha256": digest, "record": row})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--retries", type=Path, nargs="*", default=[])
    p.add_argument("--duplicate-policy", choices=("error", "last-success"), default="error")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    inputs = {}
    for item in load([args.inputs]):
        row = item["record"]
        if row["task_id"] in inputs:
            raise ValueError(f"duplicate input: {row['task_id']}")
        if not row.get("provenance_valid"):
            raise ValueError(f"unverified evidence: {row['task_id']}")
        inputs[row["task_id"]] = row
    attempts, retries = load(args.results), load(args.retries)
    grouped, retry_grouped = {}, {}
    for items, group in ((attempts, grouped), (retries, retry_grouped)):
        for item in items:
            task_id = item["record"]["task_id"]
            if task_id not in inputs:
                raise ValueError(f"unknown task in results: {task_id}")
            group.setdefault(task_id, []).append(item)
    rows, conflicts = [], []
    for task_id, source in inputs.items():
        base_attempts, retry_attempts = grouped.get(task_id, []), retry_grouped.get(task_id, [])
        successful = [x for x in base_attempts if x["record"].get("status") == "ok"]
        successful_retries = [x for x in retry_attempts if x["record"].get("status") == "ok"]
        conflict = len(successful) > 1 or len(successful_retries) > 1
        if conflict:
            conflicts.append(task_id)
        record = {"task_id": task_id, "attempts": base_attempts, "retry_attempts": retry_attempts,
                  "duplicate_success": conflict, "baseline_answer": source["original_answer"],
                  "gold": source["gold"], "baseline": answer_metrics(source["original_answer"], source["gold"])}
        if conflict and args.duplicate_policy == "error":
            record["status"] = "duplicate_conflict"
        elif not successful:
            record["status"] = "missing_or_failed_base"
        else:
            chosen = successful[-1]
            legacy = chosen["record"]
            if legacy.get("gold") != source["gold"] or legacy.get("original_answer") != source["original_answer"]:
                raise ValueError(f"legacy/input identity mismatch: {task_id}")
            answer = legacy["e0"]["guarded_answer"]
            precedence = "last_successful_shard" if len(successful) > 1 else "successful_shard"
            if successful_retries:
                # Failed retries never erase a completed answer. Only explicit --retries
                # inputs can override a base success; all attempts remain in the output.
                chosen = successful_retries[-1]
                answer = chosen["record"]["guarded_answer"]
                precedence = "explicit_successful_retry_over_base"
            direct = legacy["e0"].get("direct_answer") or source["original_answer"]
            record.update(status="ok", hybrid_answer=answer, direct_answer=direct,
                          hybrid=answer_metrics(answer, source["gold"]),
                          direct=answer_metrics(direct, source["gold"]), precedence=precedence,
                          selected_source={k: v for k, v in chosen.items() if k != "record"},
                          legacy_reported_metrics=chosen["record"].get("metrics", legacy.get("C")))
        rows.append(record)
    completed = [r for r in rows if r["status"] == "ok"]
    metrics = {method: {key: sum(r[method][key] for r in completed) / len(completed)
                       for key in completed[0][method]} for method in ("baseline", "direct", "hybrid")} if completed else {}
    report = {"version": "v110", "kind": "legacy_rescore_not_new_experiment",
              "scorer": "mini_agent_rl.hybrid_v110.answer_metrics", "expected_tasks": len(inputs),
              "completed_tasks": len(completed), "coverage_complete": len(completed) == len(inputs),
              "metrics_completed_only": metrics, "denominator": len(completed),
              "attempt_count": len(attempts), "retry_attempt_count": len(retries),
              "failed_attempts": sum(x["record"].get("status") != "ok" for x in attempts),
              "failed_retries": sum(x["record"].get("status") != "ok" for x in retries),
              "duplicate_success_task_ids": conflicts, "duplicate_policy": args.duplicate_policy,
              "retry_precedence": "last explicit successful retry overrides successful shard; failures retained",
              "unresolved": [{"task_id": r["task_id"], "status": r["status"]} for r in rows if r["status"] != "ok"],
              "api_calls": 0, "final_test_accessed": False}
    for path in (args.output, args.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["coverage_complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
