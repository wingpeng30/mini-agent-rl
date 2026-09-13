import pytest

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
