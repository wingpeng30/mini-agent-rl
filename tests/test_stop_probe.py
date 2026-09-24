"""探针语义测试：人工历史不能冒充模型重复搜索。"""
from mini_agent_rl.domain import ModelResponse, ToolCall
from mini_agent_rl.stop_probe import build_cases, check_split_overlap, classify, compare_probe_reports
from mini_agent_rl.training import harness
from mini_agent_rl.training.gates import evaluate_probe_gate
from unittest.mock import patch
import json


def case():
    return {'case_id':'a:repeated_history','task_id':'a','condition':'repeated_history',
            'history_queries':['a','a'],'gold_answer':'answer'}


def test_injected_repeat_is_not_a_model_repeat():
    result=classify(case(),ModelResponse(content='answer'))
    assert result['answered'] and result['strict_exact_match']==1
    assert not result['repeated_query']


def test_query_normalization_detects_new_repeat():
    response=ModelResponse(tool_calls=[ToolCall(name='search',arguments={'query':'  A  '})])
    assert classify(case(),response)['repeated_query']


def test_invalid_response_is_not_answer():
    result=classify(case(),ModelResponse(content='answer',parse_error='invalid JSON'))
    assert result['invalid_action'] and not result['answered']


def test_probe_comparison_rejects_different_case_sets(tmp_path):
    for name, case_id in (("a", "case-a"), ("b", "case-b")):
        root = tmp_path / name; root.mkdir()
        (root / "decisions.jsonl").write_text(json.dumps({"case_id": case_id, "action_signature": "answer"}) + "\n", encoding="utf-8")
    import pytest
    with pytest.raises(ValueError, match="case_id"):
        compare_probe_reports(tmp_path / "a", tmp_path / "b", tmp_path / "out.json")


def test_harness_git_value_uses_utf8():
    with patch("mini_agent_rl.training.harness.subprocess.check_output", return_value="中文\n") as call:
        assert harness._git_value(["diff", "--binary"]) == "中文"
    assert call.call_args.kwargs["encoding"] == "utf-8"


def test_fixed_probe_cases_keep_gold_out_of_messages_and_predefined_history():
    cases = build_cases("data/hotpot-agent-v041/validation.jsonl", limit=1, corpus_limit=100)
    assert [item["condition"] for item in cases] == ["one_hop", "two_hop", "empty_result", "repeated_history"]
    # 证据文本可以自然包含答案；要验证的是 gold 字段没有被结构化注入模型消息。
    assert all("gold_answer" not in json.dumps(item["messages"], ensure_ascii=False) for item in cases)
    assert cases[-1]["history_queries"][0] == cases[-1]["history_queries"][1]


def test_train_validation_prefixes_have_no_overlap():
    audit = check_split_overlap("data/hotpot-agent-v041/train.jsonl", "data/hotpot-agent-v041/validation.jsonl", 100)
    assert audit["passed"]


def test_gate_marks_missing_memory_unverifiable():
    context = {"lossy_body_trim_rate": 0.0, "history_drop_rate": 0.0,
               "context_budget_errors": 0, "visible_support_retention": 1.0}
    result = evaluate_probe_gate(expected=120, attempted=120, completed=120, failed=0,
                                 context=context, memory={"peak_reserved_gb": None})
    assert result["status"] == "UNVERIFIABLE"
    assert "peak_reserved_gb" in result["missing_fields"]


def test_gate_uses_declared_memory_budget_instead_of_legacy_constant():
    context = {"lossy_body_trim_rate": 0.0, "history_drop_rate": 0.0,
               "context_budget_errors": 0, "visible_support_retention": 1.0}
    result = evaluate_probe_gate(expected=1, attempted=1, completed=1, failed=0,
                                 context=context, memory={"peak_reserved_gb": 8.0}, memory_limit_gb=21.0)
    assert result["status"] == "PASS"
    assert result["memory_limit_gb"] == 21.0
