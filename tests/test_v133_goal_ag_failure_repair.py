import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v133_goal_ag_failure_repair.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("goal_ag", SCRIPT)
ag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ag)


def test_report_preserves_detailed_lead_authored_report(monkeypatch):
    report = "# Goal AG\n\nDetailed adjudication and final no-publish decision.\n"
    monkeypatch.setattr(Path, "is_file", lambda path: path.name == "REPORT.md")
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs: report)
    assert ag.render_report() == report


def test_report_can_render_terminal_decision_without_existing_report(monkeypatch):
    import json
    monkeypatch.setattr(Path, "is_file", lambda path: path.name == "confirmation-decision.json")
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs:
        json.dumps({"status": "complete_confirmation_failed", "published": False}))
    report = ag.render_report()
    assert "complete_confirmation_failed" in report
    assert '"published": false' in report


def test_visible_evidence_comes_only_from_parsed_tool_json_and_keeps_escaped_quotes():
    prompt = (
        '<tool_response>{"ok":true,"result":[{"id":"d1","title":"X",'
        '"text":"He said \\"hello\\" and died in 1951."}]}</tool_response>'
    )
    docs = ag.visible_tool_documents(prompt)
    assert len(docs) == 1
    assert docs[0]["document_id"] == "d1"
    assert '"hello"' in docs[0]["text"]


def test_truncated_or_absent_final_input_evidence_is_not_recovered_from_history():
    actual_prompt = '<tool_response>{"ok":true,"result":[]}</tool_response>'
    hidden_full_history = [{"title": "Needed", "text": "the missing relation"}]
    assert ag.visible_tool_documents(actual_prompt) == []
    assert hidden_full_history[0]["text"] not in actual_prompt


def test_only_currently_successful_continuous_search_cases_are_protected():
    rows = [
        {"case_id": "current", "required_evidence_first_found_after_first_search": True},
        {"case_id": "old_success_only", "required_evidence_first_found_after_first_search": True},
    ]
    semantic = {"current": "correct_sufficient", "old_success_only": "factual_error"}
    assert ag.protected_successes(rows, semantic) == {"current"}


def test_material_gate_counts_selected_repairs_and_selected_primary_error_only():
    rows = [{"primary_error": "bridge"} for _ in range(8)] + [
        {"primary_error": "other"} for _ in range(8)]
    assert ag.material_gate(rows, "bridge")["passed"]
    assert not ag.material_gate(rows[:15], "bridge")["passed"]
    assert not ag.material_gate(rows, "other")["passed"] is False


def test_development_and_confirmation_gates_enforce_frozen_boundaries():
    base_protected = {"a", "b", "c"}
    result = ag.development_gate(
        {"dev_success": 40, "dev_bad": 10, "dev_invalid": 0},
        {"dev_success": 36, "dev_bad": 10, "dev_invalid": 0},
        {"old_main": 15, "old_regression": 12},
        {"old_main": 14, "old_regression": 11},
        base_protected, {"b", "c"},
    )
    assert result["passed"]
    assert not ag.development_gate(
        {"dev_success": 39, "dev_bad": 10, "dev_invalid": 0},
        {"dev_success": 36, "dev_bad": 10, "dev_invalid": 0},
        {}, {}, base_protected, {"c"},
    )["passed"]
    assert ag.confirmation_gate(30, 34, 5, 5, 0, 0, 1)
    assert not ag.confirmation_gate(30, 34, 5, 6, 0, 0, 0)


def test_pre_gate_baseline_scoring_does_not_open_confirmation_answers():
    source = SCRIPT.read_text(encoding="utf-8")
    start = source.index("def baseline_semantic_scores()")
    end = source.index("\ndef protected_aa_cases", start)
    baseline_function = source[start:end]
    assert "selected-confirm.jsonl" not in baseline_function
    assert "product-decision.json" in baseline_function


def test_confirmation_multihop_protection_uses_same_confirmation_case_ids():
    cases = {"c1": {"support_titles": ["first", "second"]},
             "c2": {"support_titles": ["first", "second"]},
             "c3": {"support_titles": ["first", "second"]}}
    baseline = [
        {"case_id": "c1", "trace": [
            {"observation": {"result": [{"title": "first"}]}},
            {"observation": {"result": [{"title": "second"}]}}]},
        {"case_id": "c2", "trace": [
            {"observation": {"result": [{"title": "first"}]}},
            {"observation": {"result": [{"title": "second"}]}}]},
        {"case_id": "c3", "trace": [
            {"observation": {"result": [{"title": "first"}, {"title": "second"}]}}]},
    ]
    assert ag.later_support_title_cases(baseline, cases, {"c1", "c3"}) == {"c1"}


def test_identical_final_evidence_deduplicates_even_when_search_histories_differ():
    packet = {}
    base = {"question": "q", "answer": "a", "actual_final_prompt": "p",
            "visible_document_ids": ["d1"], "steps": [{"query": "first"}]}
    other_trace = {**base, "steps": [{"query": "different earlier path"}]}
    first_steps, second_steps = base["steps"].copy(), other_trace["steps"].copy()
    ag.merge_review_packet_item(packet, "same-review", base)
    ag.merge_review_packet_item(packet, "same-review", other_trace)
    assert packet["same-review"]["visible_document_ids"] == ["d1"]
    assert packet["same-review"]["trace_variants"] == [first_steps, second_steps]


def test_review_packet_dedup_rejects_different_visible_final_evidence():
    packet = {"same-review": {"question": "q", "answer": "a", "visible_document_ids": ["d1"]}}
    changed = {"question": "q", "answer": "a", "visible_document_ids": ["d2"]}
    try:
        ag.merge_review_packet_item(packet, "same-review", changed)
    except RuntimeError as exc:
        assert "different final evidence" in str(exc)
    else:
        raise AssertionError("different final evidence must not be merged")
