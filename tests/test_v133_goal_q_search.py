"""The product and training search loops must expose identical decisions."""
import asyncio
import json
from types import SimpleNamespace


def test_shared_search_loop_matches_goal_p(monkeypatch):
    import run_v133_goal_p_system as p
    import run_v133_goal_q_search as q
    from mini_agent_rl.domain import ModelResponse, ToolCall

    monkeypatch.setattr(q, "load_config", lambda: {"product": {
        "system_prompt": "direct-v090", "max_model_calls_per_case": 4,
        "max_searches_per_case": 3}})
    class Client:
        def __init__(self): self.calls = 0
        async def generate(self, messages, tools, *, temperature, seed):
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(tool_calls=[ToolCall(name="search", arguments={"query": "both entities"})],
                    raw_text='{"action":"search","query":"both entities"}',
                    prompt_token_ids=[1, 2], prompt_attention_mask=[1, 1], action_token_ids=[3])
            return ModelResponse(content="writer", raw_text='{"action":"answer","answer":"writer"}',
                prompt_token_ids=[4, 5], prompt_attention_mask=[1, 1], action_token_ids=[6])
    class Registry:
        def definitions(self): return []
        async def execute(self, name, args):
            return {"ok": True, "result": [{"id": "doc1", "title": "Both", "text": "writer"}]}
    config = {"system_prompt": "direct-v090", "max_model_calls_per_case": 4,
              "max_searches_per_case": 3, "seed": 42}
    before = asyncio.run(p.ask("What?", Client(), Registry(), config))
    after = asyncio.run(q.rollout("What?", Client(), Registry(), sample=False, seed=42))
    assert (before["answer"], before["reason"], before["searches"]) == (
        after["answer"], after["reason"], after["searches"])
    assert before["trace"][0]["observation"] == after["trace"][0]["observation"]
    assert after["trace"][0]["prompt_token_ids"] == [1, 2]


def test_selection_requires_product_gain_and_regression(monkeypatch):
    import run_v133_goal_q_search as q
    captured = {}
    monkeypatch.setattr(q, "write", lambda path, item: captured.update(item))
    values = {
        "goal_j": (13, 6, 3), "old_rl": (12, 5, 2),
        "direct": (14, 5, 2), "sft": (14, 6, 3), "warm_rl": (15, 6, 4),
    }
    monkeypatch.setattr(q, "metrics", lambda name: {
        "model": name, "total": values[name][0], "goal_G_regression": values[name][1],
        "invalid": values[name][2], "searches": 20} if name in values else None)
    assert q.choose() == "sft"
    assert captured["winner"] == "sft"
