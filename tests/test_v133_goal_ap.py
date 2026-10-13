import json
import pytest

from run_v133_goal_ap import restore_tool_messages


def test_restore_qwen_tool_wrapper_to_runtime_tool_message():
    payload = {"ok": True, "result": [{"id": "doc-1", "text": "A body with } braces and \"quotes\"."}]}
    messages = [
        {"role": "system", "content": "fixed system"},
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": "{\"action\": \"search\", \"query\": \"x\"}"},
        {"role": "user", "content": "<tool_response>\n" + json.dumps(payload) + "\n</tool_response>"},
    ]

    restored = restore_tool_messages(messages)

    assert restored[:3] == messages[:3]
    assert restored[3]["role"] == "tool"
    assert json.loads(restored[3]["content"]) == payload


def test_restore_rejects_malformed_search_observation():
    messages = [{"role": "user", "content": "<tool_response>\n{\"ok\": true, \"result\": {}}\n</tool_response>"}]
    with pytest.raises(RuntimeError, match="search observation"):
        restore_tool_messages(messages)
