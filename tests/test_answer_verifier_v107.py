import json

from mini_agent_rl.answer_verifier import parse_and_apply, visible_documents


DOCS = [{"id": "d1", "title": "Person", "text": "Alex was born on March 4, 1980."}]


def test_replace_requires_verbatim_visible_reference():
    raw = json.dumps({"decision": "replace", "answer": "March 4, 1980", "evidence_refs": [{"id": "d1", "quote": "born on March 4, 1980"}]})
    result = parse_and_apply(raw, "1980", DOCS)
    assert result.answer == "March 4, 1980" and result.status == "replace"


def test_keep_and_insufficient_must_preserve_candidate():
    changed = json.dumps({"decision": "keep", "answer": "other", "evidence_refs": []})
    assert parse_and_apply(changed, "original", DOCS).answer == "original"
    insufficient = json.dumps({"decision": "insufficient", "answer": "original", "evidence_refs": []})
    assert parse_and_apply(insufficient, "original", DOCS).status == "insufficient"


def test_invalid_output_and_reference_fall_back():
    assert parse_and_apply("not json", "original", DOCS).answer == "original"
    bad = json.dumps({"decision": "replace", "answer": "x", "evidence_refs": [{"id": "d1", "quote": "not present"}]})
    assert parse_and_apply(bad, "original", DOCS).status == "fallback_invalid_reference"


def test_visible_documents_uses_only_tool_text():
    messages = [{"role": "user", "content": "secret"}, {"role": "tool", "content": json.dumps({"result": DOCS})}]
    assert visible_documents(messages) == DOCS
