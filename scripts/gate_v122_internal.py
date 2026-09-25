"""Apply preregistered v122 internal gates; never open the sealed 100 here."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_report(report: dict, label: str, mode: str, count: int) -> None:
    if (report.get("version") != "v122-new-internal" or report.get("model_label") != label or
            report.get("mode") != mode or report.get("complete_coverage") is not True or
            len(report.get("rows", [])) != count):
        raise ValueError(f"incomplete_{label}_{mode}_report")


def run(paths: dict[str, Path], semantic: Path, output: Path) -> dict:
    reports = {key: read(path) for key, path in paths.items()}
    for label in ("v101", "v118", "v122"):
        check_report(reports[f"{label}_states"], label, "states", 100)
        check_report(reports[f"{label}_tasks"], label, "tasks", 50)
    first = reports["v101_tasks"]
    for report in reports.values():
        if (report.get("corpus_sha256") != first.get("corpus_sha256") or
                report.get("runner_config") != first.get("runner_config") or
                report.get("code_sha256") != first.get("code_sha256") or
                report.get("model_files") != first.get("model_files")):
            raise ValueError("paired_setting_mismatch")
    task_maps = {label: {row["task_id"]: row for row in reports[f"{label}_tasks"]["rows"]}
                 for label in ("v101", "v118", "v122")}
    if any(len(rows) != 50 or set(rows) != set(task_maps["v101"]) for rows in task_maps.values()):
        raise ValueError("paired_task_ids_mismatch")
    states = reports["v122_states"]["rows"]
    state_sets = [{row["state_id"] for row in reports[f"{label}_states"]["rows"]}
                  for label in ("v101", "v118", "v122")]
    if any(len(ids) != 100 for ids in state_sets) or state_sets[0] != state_sets[1] or state_sets[0] != state_sets[2]:
        raise ValueError("duplicate_or_unpaired_internal_state")
    answers = [row for row in states if row["target"]["action"] == "answer"]
    searches = [row for row in states if row["target"]["action"] == "search"]
    if len(answers) != 50 or len(searches) != 50:
        raise ValueError("not_50_paired_state_actions")
    old_correct = {qid for qid, row in task_maps["v101"].items() if row["normalized_exact_match"]}
    new_correct = {qid for qid, row in task_maps["v122"].items() if row["normalized_exact_match"]}
    metrics = {"v101_correct": len(old_correct), "v118_correct": sum(r["normalized_exact_match"] for r in task_maps["v118"].values()),
               "v122_correct": len(new_correct), "net_correct": len(new_correct) - len(old_correct),
               "retained_v101_correct": len(old_correct & new_correct),
               "required_retained": math.ceil(0.95 * len(old_correct)),
               "v101_f1": sum(r["token_f1"] for r in task_maps["v101"].values()) / 50,
               "v122_f1": sum(r["token_f1"] for r in task_maps["v122"].values()) / 50,
               "protocol_valid_states": sum(r["protocol_valid"] for r in states),
               "protocol_valid_tasks": sum(r["protocol_valid"] for r in task_maps["v122"].values()),
               "supported_answers_correct": sum(r["normalized_target_match"] for r in answers),
               "insufficient_evidence_searches": sum(r["action_match"] for r in searches)}
    audit = read(semantic)
    if audit.get("version") != "v122-semantic-gate-v1" or audit.get("state_cases") != 100:
        raise ValueError("complete_semantic_audit_required")
    checks = {"protocol_at_least_98_percent": metrics["protocol_valid_states"] >= 98 and
              metrics["protocol_valid_tasks"] >= 49,
              "supported_answers_at_least_45": metrics["supported_answers_correct"] >= 45,
              "insufficient_searches_at_least_45": metrics["insufficient_evidence_searches"] >= 45,
              "v101_retention_at_least_95_percent": metrics["retained_v101_correct"] >= metrics["required_retained"],
              "net_correct_at_least_three": metrics["net_correct"] >= 3,
              "f1_not_lower": metrics["v122_f1"] >= metrics["v101_f1"],
              "no_unresolved_or_confirmed_semantic_regression": audit.get("semantic_gate_passed") is True and
                  audit.get("unresolved_disputes") == 0 and audit.get("confirmed_semantic_regressions") == 0}
    result = {"version": "v122-internal-acceptance-v1", "metrics": metrics, "checks": checks,
              "internal_passed": all(checks.values()), "independent100_opened": False, "final_test_accessed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for label in ("v101", "v118", "v122"):
        for mode in ("states", "tasks"):
            parser.add_argument(f"--{label}-{mode}", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {f"{label}_{mode}": getattr(args, f"{label}_{mode}")
             for label in ("v101", "v118", "v122") for mode in ("states", "tasks")}
    print(json.dumps(run(paths, args.semantic, args.output), ensure_ascii=False, indent=2))
