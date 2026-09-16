import pytest

import json

from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import ContextBudgetError, budget_messages
from mini_agent_rl.model.local import ActionParseError, LocalModelConfig, parse_agent_action


def test_parse_search_and_answer_actions():
    content, call = parse_agent_action('{"action":"search","query":"故宫 北京"}')
    assert content == "" and call.name == "search" and call.arguments["query"] == "故宫 北京"
    content, call = parse_agent_action('```json\n{"action":"answer","answer":"北京"}\n```')
    assert content == "北京" and call is None


@pytest.mark.parametrize("raw", ["北京", "[]", '{"action":"search"}', '{"action":"unknown"}'])
def test_reject_invalid_actions(raw):
    with pytest.raises(ActionParseError):
        parse_agent_action(raw)


def test_local_config_does_not_load_model():
    config = LocalModelConfig()
    assert config.model_id == r"D:\qwen_08b"
    assert config.dtype == "bfloat16"
    assert config.enable_thinking is False
    assert config.deterministic is True
    assert config.local_files_only is True
    assert LocalModelConfig(adapter_path="checkpoints/test").adapter_path == "checkpoints/test"
    with pytest.raises(ValueError):
        LocalModelConfig(device="cpu", load_in_4bit=True)


def test_context_budget_preserves_protocol_question_and_valid_tool_json():
    messages = [
        Message(role="system", content="protocol"), Message(role="user", content="question"),
        Message(role="assistant", content='{"action":"search","query":"q"}'),
        Message(role="tool", content=json.dumps({"ok": True, "result": [
            {"id": "1", "title": "first", "text": "x" * 300},
            {"id": "2", "title": "second", "text": "y" * 300},
        ]})),
    ]
    # 测试 tokenizer 用字符数替代 token 数，只验证预算算法和结构边界。
    result = budget_messages(messages, 180, lambda rows: 20 + sum(len(x.content) for x in rows), len)
    assert result.messages[0].content == "protocol" and result.messages[1].content == "question"
    tool = json.loads(next(item.content for item in result.messages if item.role == "tool"))
    assert isinstance(tool["result"], list) and result.usage["prompt_truncated"]
    assert result.usage["dropped_observation_tokens"] > 0


def test_context_budget_rejects_oversized_mandatory_prompt():
    messages = [Message(role="system", content="s" * 100), Message(role="user", content="q" * 100)]
    with pytest.raises(ContextBudgetError, match="context_budget_error"):
        budget_messages(messages, 50, lambda rows: sum(len(x.content) for x in rows), len)
