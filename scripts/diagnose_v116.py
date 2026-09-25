"""Paired numeric/action attribution for v115, with semantic status kept separate."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/v115-private-run/unpacked/reports"
OUT = ROOT / "reports/v116-v115-diagnosis.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run():
    s = {m: {r["state_id"]: r for r in read(SOURCE / f"v115-det-{m}-states.json")["rows"]}
         for m in ("v101", "v112", "v115")}
    t = {m: {r["task_id"]: r for r in read(SOURCE / f"v115-det-{m}-tasks.json")["rows"]}
         for m in ("v101", "v112", "v115")}
    if len(s["v101"]) != 100 or any(set(v) != set(s["v101"]) for v in s.values()):
        raise ValueError("state_coverage_mismatch")
    if len(t["v101"]) != 50 or any(set(v) != set(t["v101"]) for v in t.values()):
        raise ValueError("task_coverage_mismatch")
    categories = Counter()
    state_cases = []
    for sid, row in s["v115"].items():
        wanted = row["target"]["action"]
        action = (row["actual_action"] or {}).get("action", "invalid")
        if wanted == "answer":
            category = "supported_correct_answer" if row["normalized_target_match"] else \
                "supported_oversearch" if action == "search" else "supported_unmatched_answer"
        else:
            category = "insufficient_search" if action == "search" else "insufficient_answer_or_invalid"
        categories[category] += 1
        state_cases.append({"state_id": sid, "task_id": row["task_id"], "category": category,
                            "v101_action": s["v101"][sid]["actual_action"],
                            "v115_action": row["actual_action"],
                            "v101_official_target_match": s["v101"][sid]["normalized_target_match"],
                            "v115_official_target_match": row["normalized_target_match"],
                            "semantic_support": "pending_corrected_blind_review"})
    changes = []
    for tid, baseline in t["v101"].items():
        candidate = t["v115"][tid]
        before, after = baseline["normalized_exact_match"], candidate["normalized_exact_match"]
        if before != after:
            changes.append({"task_id": tid, "official_change": "gain" if after else "loss",
                            "v101_answer": baseline.get("final_answer"),
                            "v115_answer": candidate.get("final_answer"),
                            "evidence_support_change": "pending_corrected_blind_review"})
    result = {"version": "v116-v115-action-diagnosis-v1", "same_input_states": 100,
              "categories": dict(categories), "task_official_changes": changes,
              "task_gains": sum(c["official_change"] == "gain" for c in changes),
              "task_losses": sum(c["official_change"] == "loss" for c in changes),
              "state_cases": state_cases, "fixed_review_source": "data/evidence-sft-v116/diagnosis-blind",
              "semantic_support_status": "pending_independent_double_review",
              "independent100_accessed": False, "final_test_accessed": False}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"state_cases", "task_official_changes"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    run()
