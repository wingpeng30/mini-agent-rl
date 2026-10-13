"""CPU contracts for AN state weighting and probe action collection."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import run_v133_goal_an as an
from mini_agent_rl.domain.models import ModelResponse,ToolCall


def test_state_mass_and_distribution():
    values=[an.normalized_state_weight(c,"B") for c in an.HARD]+[an.normalized_state_weight("easy","B")]*30
    assert sum(values)==pytest.approx(34)
    assert values[0]/values[-1]==pytest.approx(4)
    assert all(an.normalized_state_weight(c,"A")==1 for c in [*an.HARD,"easy"])
    with pytest.raises(ValueError):an.normalized_state_weight("easy","C")


def test_empty_content_is_not_answer():
    assert an.parsed_action(ModelResponse())=="invalid"
    assert an.parsed_action(ModelResponse(tool_calls=[ToolCall(name="search",arguments={"query":"entity relation"})]))=="search"
    assert an.parsed_action(ModelResponse(content="answer"))=="answer"
    assert an.parsed_action(ModelResponse(content="answer",parse_error="bad first JSON"))=="invalid"


def test_shared_settings_are_read_only_and_not_new_limits():
    assert an.PC==an.am.PC and an.PC is not an.am.PC
    assert an.PC["max_input_tokens"]==16384 and an.PC["max_new_tokens"]==2048
    assert an.PC["action_completion"]=="first-json-v1"
    assert an.CAPS==dict(update=384,generation=896,search=576)


def test_client_thinking_field_is_mapped_explicitly():
    from mini_agent_rl.model.local import LocalModelConfig
    config=LocalModelConfig(model_id="not-loaded",enable_thinking=an.PC["thinking"])
    assert config.enable_thinking is False
    assert "thinking" not in type(config).model_fields
