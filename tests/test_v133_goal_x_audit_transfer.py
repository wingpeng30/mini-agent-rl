"""Regression checks for Goal X's concrete leak and data gate."""
import importlib.util
import json
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_v133_goal_x_audit_transfer.py"
spec = importlib.util.spec_from_file_location("goal_x", SCRIPT)
goal_x = importlib.util.module_from_spec(spec)
spec.loader.exec_module(goal_x)


def test_full_question_review_excludes_known_evaluation_leak():
    out = ROOT / "runs/v133-goal-x-audit-and-transfer"
    rows = list(goal_x.rows(out / "training-audit.jsonl"))
    assert len(rows) == 512
    assert len({row["case_id"] for row in rows}) == 512
    leak = next(row for row in rows if row["case_id"] == "5a73583e55429901807dafd0")
    assert leak["primary_disposition"] == "historical_overlap"
    assert any(item["tier"] == "evaluation_input_or_output" for item in leak["historical_matches"])
    affiliation = [row for row in rows if row["reviewed_category"] == "affiliation"
                   and row["shared_attribute_question"]]
    assert len(affiliation) == 5
    assert sum(row["primary_disposition"] == "eligible_for_real_search" for row in affiliation) == 4


def test_affiliation_upper_bound_cannot_start_training(monkeypatch):
  with tempfile.TemporaryDirectory() as directory:
    tmp_path = Path(directory)
    monkeypatch.setattr(goal_x, "OUT", tmp_path)
    audit = tmp_path / "training-audit.jsonl"
    trace = tmp_path / "train-real-search.jsonl"
    with audit.open("w", encoding="utf-8") as stream:
        for i in range(4):
            stream.write(json.dumps({"case_id": str(i), "primary_disposition": "eligible_for_real_search"}) + "\n")
    with trace.open("w", encoding="utf-8") as stream:
        for i in range(4):
            stream.write(json.dumps({"case_id": str(i), "reviewed_category": "affiliation",
                "reason": "answer", "both_support_retrieved": True,
                "both_support_ids_visible": True, "both_support_full_text_visible": True,
                "answer_prompt_token_ids": [1, 2], "elapsed_seconds": 1.0,
                "answer_prompt_visibility_flags": {"prompt_truncated": False,
                   "documents_deleted": 0, "lossy_body_trimmed": False}}) + "\n")
    (tmp_path / "state.json").write_text(json.dumps({"status": "x2_search_collected",
        "search_trace_sha256": goal_x.sha(trace)}), encoding="utf-8")
    decision = goal_x.finalize()
    assert decision["status"] == "data_shortfall"
    assert decision["affiliation_maximum_possible_pairs"] == 4
    assert decision["training_started"] is False
    assert decision["selected_product"] == "goal_t_seed42_update32"
