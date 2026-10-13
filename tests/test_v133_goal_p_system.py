"""Goal P live tool-loop and evidence boundary checks."""
import asyncio
import importlib.util
import sys
from pathlib import Path

from mini_agent_rl.domain import ModelResponse, ToolCall
from mini_agent_rl.tools import SearchTool, ToolRegistry


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("goal_p", ROOT / "scripts/run_v133_goal_p_system.py")
GOAL_P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GOAL_P)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    async def generate(self, messages, tools, *, temperature, seed):
        self.messages.append([item.model_copy(deep=True) for item in messages])
        assert temperature == 0 and tools[0].name == "search"
        return next(self.responses)


def registry():
    value = ToolRegistry()
    value.register(SearchTool([{"id": "doc-a", "title": "Paris", "text": "Paris is in France."}], "bm25-v2"))
    return value


CFG = {"system_prompt": "direct-v090", "max_model_calls_per_case": 4,
       "max_searches_per_case": 3, "seed": 42}


def test_real_tool_result_is_replayed_to_model_without_gold():
    client = FakeClient([
        ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": "Paris"})], raw_text='{"action":"search","query":"Paris"}'),
        ModelResponse(content="France", raw_text='{"action":"answer","answer":"France"}'),
    ])
    result = asyncio.run(GOAL_P.ask("Where is Paris?", client, registry(), CFG))
    assert result["reason"] == "answer" and result["searches"] == 1 and result["answer"] == "France"
    observation = result["trace"][0]["observation"]["result"]
    assert observation[0]["id"] == "doc-a"
    assert client.messages[1][-1].role == "tool" and "Paris is in France." in client.messages[1][-1].content
    assert "official_answer" not in str(client.messages)


def test_repeated_query_stops_without_second_tool_execution():
    client = FakeClient([
        ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": "Paris"})], raw_text="search"),
        ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": " PARIS "})], raw_text="search"),
    ])
    result = asyncio.run(GOAL_P.ask("Where is Paris?", client, registry(), CFG))
    assert result["reason"] == "repeated_action" and result["searches"] == 1
    assert result["model_calls"] == 2


def test_search_text_alias_is_narrow_and_auditable():
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_v133_goal_p_variant import SearchTextAliasClient
    accepted = ModelResponse(raw_text='{"action":"search","text":"Paris"}<|im_end|>',
                             parse_error="missing query")
    rejected = ModelResponse(raw_text='{"action","search"}<|im_end|>', parse_error="invalid JSON")
    client = SearchTextAliasClient(FakeClient([accepted, rejected]))
    tools = registry().definitions()
    first = asyncio.run(client.generate([], tools, temperature=0, seed=42))
    second = asyncio.run(client.generate([], tools, temperature=0, seed=42))
    assert first.tool_call.arguments == {"query": "Paris"}
    assert first.usage["protocol_recovery"] == "search_text_alias_v1"
    assert second.parse_error == "invalid JSON"
