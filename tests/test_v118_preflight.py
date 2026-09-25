"""Targeted guard tests for the v118 frozen experiment path."""
import pytest

from scripts.evaluate_v118 import action_time_messages, legal_action
from scripts.build_v118_blind_audit import action_time_visible_messages
from scripts.review_v118_api import validate
from scripts.merge_v118_semantic_reviews import candidate_visible, validate_candidates
from scripts.diagnose_v119_review_dimensions import vote
from scripts.screen_v119_evidence_spans import candidates
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


def test_parsed_document_quote_is_valid_but_serialized_cross_document_quote_is_not():
    import json
    tool = json.dumps({"ok": True, "result": [
        {"id": "one", "title": "First", "text": 'The "Register" is the website.'},
        {"id": "two", "title": "Second", "text": "Situation Publishing is the publisher."},
    ]})
    packet = {"audit_id": "s-parsed", "anonymous_outputs": {
        alias: {"action": {"action": "answer"}, "pre_action_visible_messages": [
            {"role": "user", "content": "Who is the publisher?"},
            {"role": "tool", "content": tool},
        ]} for alias in "ABC"}}
    visible = candidate_visible(packet, "states", "s-parsed", "A")
    assert 'The "Register" is the website.' in visible
    assert 'The "Register" is the website.' not in tool
    good = {"audit_id": "s-parsed", "candidates": {alias: {
        "evidence_verdict": "supported", "reason": "publisher quote",
        "citations": ["Situation Publishing is the publisher."], "risk_labels": [],
    } for alias in "ABC"}}
    validate(packet, good, "states")
    _, invalid = validate_candidates({"s-parsed": good}, {"s-parsed": packet}, "states", "reviewer")
    assert invalid == []
    cross_document_fragment = tool[tool.index('website.') : tool.index('Situation Publishing') + len('Situation Publishing')]
    bad = {**good, "candidates": {alias: {**good["candidates"][alias],
           "citations": [cross_document_fragment]} for alias in "ABC"}}
    _, invalid = validate_candidates({"s-parsed": bad}, {"s-parsed": packet}, "states", "reviewer")
    assert len(invalid) == 3


def test_review_dimensions_fail_closed_on_invalid_or_disputed_votes():
    assert vote("yes", "yes") == "yes"
    assert vote("yes", "no") == "disputed"
    assert vote("yes", "yes", bad_b=True) == "invalid_citation"
    assert vote("supported", "supported", bad_a=True) == "invalid_citation"


def test_span_candidates_are_document_bound_without_a_reference_answer():
    documents = [{"id": "source-1", "title": "Person", "text": "Kylie Minogue was born 28 May 1968."}]
    spans = candidates(documents)
    assert any(item == {"candidate": "28 May 1968", "document_id": "source-1"} for item in spans)
    assert all(item["candidate"] == "Person" or item["candidate"] in documents[0]["text"] for item in spans)
