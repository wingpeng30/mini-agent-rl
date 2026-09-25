"""Targeted guard tests for the v118 frozen experiment path."""
import pytest

from scripts.evaluate_v118 import action_time_messages, legal_action
from scripts.build_v118_blind_audit import action_time_visible_messages
from scripts.review_v118_api import validate
from scripts.verify_v118_freeze import verify


def test_v117_local_freeze_hashes_and_backup_are_intact():
    report = verify()
    assert report["passed"] is True
    assert report["train_states"] == 345
    assert report["internal_states"] == 100
    assert report["corpus_documents"] == 1986


def test_protocol_requires_exactly_one_well_formed_action():
    assert legal_action({"content": "answer", "tool_calls": [], "usage": {}})
    assert legal_action({"content": "", "tool_calls": [{"name": "search", "arguments": {"query": "Q"}}], "usage": {}})
    assert not legal_action({"content": "answer", "tool_calls": [{"name": "search", "arguments": {"query": "Q"}}], "usage": {}})
    assert not legal_action({"content": "", "tool_calls": [{"name": "search", "arguments": {"query": " "}}], "usage": {}})
    assert not legal_action({"content": "answer", "tool_calls": [], "usage": {"output_truncated": True}})
    assert not legal_action({"content": "", "tool_calls": [], "parse_error": "bad_json", "usage": {}})


def test_state_evaluation_uses_only_pre_action_messages():
    row = {"messages": [{"role": "user", "content": "Question"},
                        {"role": "tool", "content": "visible evidence"}],
           "target": {"action": "answer", "answer": "private target"}}
    assert action_time_messages(row) == row["messages"]
    row["messages"].append({"role": "assistant", "content": "target action"})
    with pytest.raises(ValueError, match="before_target"):
        action_time_messages(row)


def test_blind_packet_preserves_the_last_action_time_evidence_message():
    state = {"messages": [{"role": "user", "content": "Question"},
                          {"role": "tool", "content": "last visible evidence"}]}
    assert action_time_visible_messages(state) == state["messages"]
    state["messages"].append({"role": "assistant", "content": "target"})
    with pytest.raises(ValueError, match="invalid_action_time_messages"):
        action_time_visible_messages(state)


def test_blind_audit_quote_validation_uses_raw_message_text():
    messages = [{"role": "tool", "content": 'The "Register" company launched the project.'}]
    packet = {"audit_id": "s-test", "anonymous_outputs": {alias: {"action": {"action": "answer"},
             "pre_action_visible_messages": messages} for alias in "ABC"}}
    decision = {"audit_id": "s-test", "candidates": {alias: {
        "evidence_verdict": "supported", "reason": "quoted support", "citations": ['The "Register" company'],
        "risk_labels": []} for alias in "ABC"}}
    validate(packet, decision, "states")
