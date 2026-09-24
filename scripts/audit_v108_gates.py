"""Recompute v107 uniqueness and v108 candidate gates without model access."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from mini_agent_rl.answer_selector import parse_candidates


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def verifier_key(row: dict) -> tuple[str, str, str, str, str]:
    payload = json.loads(row["messages"][-2]["content"])
    target = json.loads(row["messages"][-1]["content"])
    metadata = row["metadata"]
    return (metadata["task_id"], metadata["evidence_sha256"],
            payload["candidate_answer"].strip().casefold(), target["decision"],
            target["answer"].strip().casefold())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v107-data", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dedupe: dict[str, dict] = {}
    for name in ("train", "validation"):
        rows = load(args.v107_data / f"{name}.jsonl")
        unique = {}
        for row in rows:
            unique.setdefault(verifier_key(row), row)
        decisions = Counter(row["metadata"]["decision"] for row in unique.values())
        dedupe[name] = {"raw_rows": len(rows), "independent_states": len(unique),
            "duplicates": len(rows) - len(unique), "unique_tasks": len({key[0] for key in unique}),
            "decisions": dict(decisions)}

    candidates = load(args.candidates)
    reparsed = []
    for row in candidates:
        alternatives, error = parse_candidates(row.get("raw", ""), str(row.get("original", "")), row.get("evidence", []))
        reparsed.append({"task_id": row["task_id"], "valid_protocol": error is None,
                         "accepted_alternatives": len(alternatives), "error": error})
    wrong = [row for row in candidates if not row.get("original_correct")]
    validation_wrong = [row for row in wrong if row.get("split") == "validation"]
    report = {"version": "v108", "decision": "stop_before_selector_training",
        "reason": "correct_candidate_missing_and_v107_independent_validation_gate_failed",
        "v107_deduplicated": dedupe,
        "v107_training_gate": {"required": {"train_states": 120, "train_keep": 40,
            "train_replace": 40, "validation_states": 30, "validation_keep": 10,
            "validation_replace": 10}, "passed": False},
        "candidate_generation": {"tasks": len(candidates),
            "source_inputs_auditable": sum(bool(row.get("source_prompt_audit_ok")) for row in candidates),
            "protocol_valid": sum(row["valid_protocol"] for row in reparsed),
            "accepted_alternatives": sum(row["accepted_alternatives"] for row in reparsed),
            "wrong_original": len(wrong),
            "wrong_with_strict_correct_alternative": sum(bool(row.get("attainable_strict")) for row in wrong),
            "validation_wrong": len(validation_wrong),
            "validation_wrong_with_strict_correct_alternative": sum(bool(row.get("attainable_strict")) for row in validation_wrong)},
        "selector_training_run": False, "development_evaluation_run": False,
        "deepseek_additional_calls": 0, "final_test_accessed": False,
        "inputs": {"v107_train_sha256": hashlib.sha256((args.v107_data / "train.jsonl").read_bytes()).hexdigest(),
                   "v107_validation_sha256": hashlib.sha256((args.v107_data / "validation.jsonl").read_bytes()).hexdigest(),
                   "candidate_rows_sha256": hashlib.sha256(args.candidates.read_bytes()).hexdigest()}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
