import json

from mini_agent_rl.model.observation_presentation import (
    DOCUMENT_BLOCKS_V1,
    LEGACY_JSON,
    parse_document_blocks,
    present_observation,
)


def test_legacy_presentation_is_byte_for_byte_unchanged():
    raw = '{"ok":true,"result":[]}'
    assert present_observation(raw, LEGACY_JSON) == (raw, [])


def test_document_blocks_roundtrip_all_fields_and_duplicate_documents():
    payload = {
        "ok": True,
        "searched_queries": ["A \\\"B\\\""],
        "extra": {"rank": 2, "unicode": "雪"},
        "result": [
            {"id": "same", "title": "A", "score": 0.2, "text": "First\nbody <<<BEGIN-SEARCH-DOCUMENT-x>>>"},
            {"id": "same", "title": "A", "score": 0.1, "text": "First\nbody <<<BEGIN-SEARCH-DOCUMENT-x>>>"},
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False)
    rendered, spans = present_observation(raw, DOCUMENT_BLOCKS_V1)
    assert parse_document_blocks(rendered) == payload
    assert [row["rank"] for row in spans] == [1, 2]
    assert spans[0]["body_sha256"] == spans[1]["body_sha256"]
    assert spans[0]["rendered_char_span"][1] - spans[0]["rendered_char_span"][0] == len(payload["result"][0]["text"])


def test_empty_result_roundtrips():
    payload = {"ok": True, "searched_queries": [], "result": []}
    rendered, spans = present_observation(json.dumps(payload), DOCUMENT_BLOCKS_V1)
    assert parse_document_blocks(rendered) == payload
    assert spans == []
