"""Record-only reconciliation of the Goal AK engineering equality checker."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/v133-goal-ak-single-action-v1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    failure_path = OUT / "failure.json"
    status_path = OUT / "status.json"
    result_path = OUT / "engineer-results.jsonl"
    reconciled_path = OUT / "engineering-reconciliation.json"
    if reconciled_path.exists():
        raise FileExistsError(reconciled_path)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (failure.get("stage"), failure.get("error_type"), failure.get("error"), status.get("status")) != (
        "engineer", "RuntimeError", "16-call first-action engineering gate failed", "stopped_on_failure"
    ):
        raise RuntimeError("saved state differs from the known tuple/list checker failure")
    rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 16:
        raise RuntimeError(f"expected 16 saved engineering generations, found {len(rows)}")
    for row in rows:
        if row.get("actual_action") != row.get("expected_first_action"):
            raise RuntimeError(f"saved action differs for {row.get('case_id')} {row.get('arm')}")
        if row.get("prompt_token_ids_match") is not True or row.get("parse_error") is not None:
            raise RuntimeError(f"saved prompt replay or parse check failed for {row.get('case_id')} {row.get('arm')}")
        if row.get("usage", {}).get("first_json_stop_offset") is None:
            raise RuntimeError(f"first JSON boundary was not recorded for {row.get('case_id')} {row.get('arm')}")
    record = {"status": "pass_from_saved_outputs", "cause": "checker compared Python tuples to JSON-decoded lists",
        "gpu_generations_repeated": 0, "saved_gpu_generations": len(rows),
        "prompt_replays_matching": sum(r["prompt_token_ids_match"] for r in rows),
        "first_actions_matching": sum(r["actual_action"] == r["expected_first_action"] for r in rows),
        "parse_errors": sum(r["parse_error"] is not None for r in rows),
        "first_json_boundaries": sum(r["usage"].get("first_json_stop_offset") is not None for r in rows),
        "engineering_results_sha256": digest(result_path), "original_failure_preserved": digest(failure_path),
        "reconciler_sha256": digest(Path(__file__)), "reconciled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reconciliation_scope": "read-only validation of the existing 16 outputs; no model call, retry, parameter or gate change"}
    reconciled_path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    next_status = {"status": "ready", "stage": "engineer-reconciled-from-saved-outputs", "gpu_generations": 16,
        "original_false_failure_preserved": True, "additional_gpu_generations": 0}
    status_path.write_text(json.dumps(next_status, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
