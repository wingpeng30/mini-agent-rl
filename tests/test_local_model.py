import pytest

import json

from mini_agent_rl.domain import Message
from mini_agent_rl.model.context import ContextBudgetError, budget_messages
from mini_agent_rl.model.local import ActionParseError, LocalModelConfig, first_json_object_end, generation_terminator_ids, parse_agent_action, split_reasoning_action, trim_generation_terminators
from mini_agent_rl.runtime.runner import AgentRunner


def test_parse_search_and_answer_actions():
    content, call = parse_agent_action('{"action":"search","query":"故宫 北京"}')
    assert content == "" and call.name == "search" and call.arguments["query"] == "故宫 北京"
    content, call = parse_agent_action('```json\n{"action":"answer","answer":"北京"}\n```')
    assert content == "北京" and call is None


@pytest.mark.parametrize("raw", ["北京", "[]", '{"action":"search"}', '{"action":"unknown"}'])
def test_reject_invalid_actions(raw):
    with pytest.raises(ActionParseError):
        parse_agent_action(raw)


def test_thinking_parser_uses_only_json_after_closed_think_block():
    reasoning, action = split_reasoning_action('analysis {"action":"answer","answer":"wrong"}</think> {"action":"answer","answer":"right"}', True)
    assert "wrong" in reasoning
    assert parse_agent_action(action)[0] == "right"
    with pytest.raises(ActionParseError):
        split_reasoning_action('unfinished reasoning', True)


def test_thinking_parser_accepts_explicit_block_and_direct_json():
    reasoning, action = split_reasoning_action('<think>need evidence</think>{"action":"search","query":"q"}', True)
    assert reasoning == "need evidence" and parse_agent_action(action)[1].name == "search"
    assert split_reasoning_action('{"action":"answer","answer":"a"}', True)[0] is None


def test_trim_qwen_message_end_and_eos_but_preserve_action_whitespace():
    class Tokenizer:
        eos_token_id = 30

        def convert_tokens_to_ids(self, token):
            return {"<|im_end|>": 10, "<|endoftext|>": 30}[token]

        def decode(self, values, skip_special_tokens=False):
            return {20: "\n"}.get(values[0], "x")

    values, usage = trim_generation_terminators([1, 20, 10, 20, 30], Tokenizer())
    assert values == [1, 20]
    assert usage == {"generated_message_end": True, "generated_eos": True}


def test_protocol_fingerprint_binds_rendered_prompt_and_history_policy():
    runner = object.__new__(AgentRunner)
    runner.enable_thinking = False
    runner.anti_repeat_prompt = False
    runner.prompt_version = "unit"
    runner.max_steps = 3
    direct = runner.protocol_fingerprint()
    assert direct["system_prompt_sha256"] and direct["anti_repeat_prompt"] is False
    runner.anti_repeat_prompt = True
    assert runner.protocol_fingerprint()["system_prompt_sha256"] != direct["system_prompt_sha256"]


def test_generation_stops_at_qwen_message_end_and_model_eos():
    class Tokenizer:
        eos_token_id = 30
        all_special_ids = [10, 30]

        def convert_tokens_to_ids(self, token):
            return 10

    assert generation_terminator_ids(Tokenizer()) == [30, 10]


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


def test_first_json_completion_is_opt_in_and_requires_direct_actions():
    assert LocalModelConfig().action_completion == "eos"
    assert LocalModelConfig(action_completion="first-json-v1").action_completion == "first-json-v1"
    with pytest.raises(ValueError, match="non-thinking"):
        LocalModelConfig(action_completion="first-json-v1", enable_thinking=True)


@pytest.mark.parametrize(("text", "expected"), [
    ('{"action":"search","query":"x } \\\" {"}\\n{"action":"answer","answer":"later"}',
     len('{"action":"search","query":"x } \\\" {"}')),
    (' {"action":{"kind":"search"},"query":"x"} tail', len(' {"action":{"kind":"search"},"query":"x"}')),
    ('{"action":"unknown"} {"action":"answer","answer":"later"}', len('{"action":"unknown"}')),
    ('["not an action"] {"action":"answer","answer":"later"}', None),
    ('{"action":"answer","answer":"brace } in string"}', len('{"action":"answer","answer":"brace } in string"}')),
])
def test_first_json_completion_stops_at_first_structural_object(text, expected):
    assert first_json_object_end(text) == expected


def test_first_json_completion_does_not_skip_invalid_first_action():
    text = '{"action":"unknown"} {"action":"answer","answer":"later"}'
    end = first_json_object_end(text)
    with pytest.raises(ActionParseError):
        parse_agent_action(text[:end])


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


def test_compact_v1_reuses_document_and_removes_nonsemantic_score():
    document = {"id": "d1", "title": "Title", "text": "First sentence. Second sentence.", "score": 99}
    messages = [Message(role="system", content="p"), Message(role="user", content="q"),
                Message(role="assistant", content='{"action":"search","query":"a"}'),
                Message(role="tool", content=json.dumps({"ok": True, "result": [document]})),
                Message(role="assistant", content='{"action":"search","query":"b"}'),
                Message(role="tool", content=json.dumps({"ok": True, "result": [document]}))]
    result = budget_messages(messages, 10_000, lambda rows: sum(len(item.content) for item in rows), len, "compact-v1")
    tools = [json.loads(item.content) for item in result.messages if item.role == "tool"]
    assert "score" not in tools[0]["result"][0]
    assert tools[1]["result"][0] == {"id": "d1", "ref": True}


def test_compact_v2_does_not_merge_same_id_with_different_text():
    first = {"id": "d", "title": "T", "text": "First sentence. Second."}
    changed = {"id": "d", "title": "T", "text": "Different sentence. More."}
    messages = [Message(role="system", content="p"), Message(role="user", content="q"),
                Message(role="tool", content=json.dumps({"ok": True, "result": [first]})),
                Message(role="tool", content=json.dumps({"ok": True, "result": [changed]}))]
    result = budget_messages(messages, 10_000, lambda rows: sum(len(item.content) for item in rows), len, "compact-v2")
    tools = [json.loads(item.content) for item in result.messages if item.role == "tool"]
    assert "text" in tools[0]["result"][0] and "text" in tools[1]["result"][0]
    assert result.document_changes[1]["content_conflict"]
    assert not result.document_changes[1]["reused_reference"]


def test_compact_v1_trims_only_at_sentence_boundary():
    messages = [Message(role="system", content="p"), Message(role="user", content="q"),
                Message(role="assistant", content='{"action":"search","query":"a"}'),
                Message(role="tool", content=json.dumps({"ok": True, "result": [
                    {"id": "d1", "title": "T", "text": "A complete sentence. Another long sentence without end"}]}))]
    result = budget_messages(messages, 140, lambda rows: 20 + sum(len(item.content) for item in rows), len, "compact-v1")
    tool = json.loads(next(item.content for item in result.messages if item.role == "tool"))
    assert tool["result"]
    assert tool["result"][0]["text"].endswith(".")
    assert result.usage["lossy_body_trimmed"]


def test_compact_v3_trims_lower_rank_before_higher_rank():
    docs = [{"id": f"d{i}", "title": f"T{i}",
             "text": "First sentence. " + ("x" * 280) + " End."} for i in range(1, 4)]
    messages = [Message(role="system", content="p"), Message(role="user", content="q"),
                Message(role="assistant", content='{"action":"search","query":"a"}'),
                Message(role="tool", content=json.dumps({"ok": True, "result": docs}))]
    result = budget_messages(messages, 780, lambda rows: 20 + sum(len(item.content) for item in rows), len, "compact-v3")
    changes = result.document_changes
    assert not changes[0]["body_trimmed"]
    assert changes[2]["body_trimmed"]
    assert changes[2]["visible_segments"][0][0] == 0


def test_compact_v3_reference_resolves_to_visible_source():
    document = {"id": "d", "title": "T", "text": "Evidence sentence."}
    messages = [Message(role="system", content="p"), Message(role="user", content="q"),
                Message(role="tool", content=json.dumps({"ok": True, "result": [document]})),
                Message(role="tool", content=json.dumps({"ok": True, "result": [document]}))]
    result = budget_messages(messages, 10_000, lambda rows: sum(len(item.content) for item in rows), len, "compact-v3")
    tools = [json.loads(item.content) for item in result.messages if item.role == "tool"]
    assert tools[1]["result"][0]["ref"] == tools[0]["result"][0]["source"]
