"""Rebuild the gated minimum cohorts and verify their metadata-free SFT exports."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from finalize_v114_reviews import norm, select_cohort  # noqa: E402

WORK = ROOT / "data/evidence-sft-v114-review"
REPORT = ROOT / "reports/v114-review-gate.json"
MANIFEST = ROOT / "reports/v114-sft-export-manifest.json"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def to_sft_record(row: dict) -> dict:
    messages = list(row["packet"]["visible_messages"])
    messages.append({"role": "assistant", "content": json.dumps(
        row["packet"]["proposed_action"], ensure_ascii=False, separators=(",", ":"))})
    return {"messages": messages, "target_assistant_index": len(messages) - 1,
            "thinking_enabled": False}


def main() -> None:
    report = read_json(REPORT)
    if not report.get("passed") or not report.get("training_permitted") or report.get("gate_errors"):
        raise RuntimeError("v114_dataset_gate_not_passed")
    if report.get("final_test_accessed") or report.get("independent100_evaluated"):
        raise RuntimeError("forbidden_evaluation_access_recorded")
    accepted = read_jsonl(WORK / "accepted.jsonl")
    if len({row["case_id"] for row in accepted}) != len(accepted):
        raise RuntimeError("duplicate_accepted_case_id")
    frozen_internal = set(read_json(WORK / "internal-freeze-v3.json")["ids"])
    overrides = set(report.get("risk_category_recheck", {}).get("applied_cases", []))
    train_pool = [row for row in accepted if row["split"] == "train"]
    internal_pool = [row for row in accepted if row["split"] == "internal"]
    train = select_cohort(train_pool, "train", frozen_internal, overrides)
    internal = select_cohort(internal_pool, "internal", frozen_internal, overrides)
    if train is None or internal is None:
        raise RuntimeError("cohort_reconstruction_failed")
    for split, cohort in (("train", train), ("internal", internal)):
        path = WORK / f"sft-{split}.jsonl"
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        expected = [to_sft_record(row) for row in cohort]
        if records != expected:
            raise RuntimeError(f"export_content_does_not_match_approved_cohort:{split}")
        if any(set(record) != {"messages", "target_assistant_index", "thinking_enabled"}
               for record in records):
            raise RuntimeError(f"export_contains_review_or_private_metadata:{split}")
        if any(record["target_assistant_index"] != len(record["messages"]) - 1
               or record["messages"][-1].get("role") != "assistant"
               for record in records):
            raise RuntimeError(f"invalid_assistant_target_position:{split}")
        if any(row["packet"].get("preflight_error") or
               row["packet"].get("prompt_tokens") is None or row["packet"].get("prompt_tokens") > 2048 or
               row["packet"].get("target_tokens") is None or row["packet"].get("target_tokens") > 128
               for row in cohort):
            raise RuntimeError(f"token_budget_or_truncation_preflight_failed:{split}")
        if len({row["case_id"] for row in cohort}) != len(cohort):
            raise RuntimeError(f"duplicate_selected_state:{split}")
    train_questions = {norm(row["packet"]["question"]) for row in train}
    internal_questions = {norm(row["packet"]["question"]) for row in internal}
    if train_questions & internal_questions:
        raise RuntimeError("normalized_question_isolation_failed")
    if {row["source_id"] for row in internal} != frozen_internal:
        raise RuntimeError("internal_freeze_question_coverage_failed")
    manifest = {
        "version": "v114-sft-export-manifest-v1",
        "gate_report_sha256": sha(REPORT.read_bytes()),
        "internal_freeze_sha256": sha((WORK / "internal-freeze-v3.json").read_bytes()),
        "training_question_freeze_sha256": sha((WORK / "training-question-freeze-v2.json").read_bytes()),
        "exports": {
            "sft-train.jsonl": {"sha256": sha((WORK / "sft-train.jsonl").read_bytes()), "states": len(train),
                                "questions": len(train_questions)},
            "sft-internal.jsonl": {"sha256": sha((WORK / "sft-internal.jsonl").read_bytes()), "states": len(internal),
                                   "questions": len(internal_questions)},
        },
        "question_isolation_overlap": 0,
        "private_mapping_in_exports": False,
        "review_metadata_in_exports": False,
        "independent100_evaluated": False,
        "final_test_accessed": False,
        "training_started": False,
        "gpu_used": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["exports"], ensure_ascii=False))


if __name__ == "__main__":
    main()
